Great — I’ll follow the paper’s procedure exactly and provide a ready-to-run Python implementation of the adaptive Hopf CPG learning algorithm as presented in the paper (equations (1)–(8) for learning; (9)–(13) for implementation). The paper explicitly uses five coupled oscillators in each module and updates the algorithm at 200 Hz; I’ll use those defaults unless you tell me otherwise.

Below is a complete Python script (hopf_fit.py) that:

Loads a CSV (either time+knee columns, or a single column of knee angles + user-supplied sampling rate).

Preprocesses (optional filtering, remove mean).

Implements the paper’s learning equations (1)–(8) exactly (including the Ri, phi handling and alpha_dot = eta * x_i * F(t)) and integrates them with a stable RK4 integrator.

Uses the paper’s implementation form for output (weighted sum Q(t) and Gainα option).

Saves plots and learned parameters to files.

Notes before you run

The paper’s equations (1–8) are implemented as-is (no extra weight-decay term unless you request it). If you want the weight decay the earlier message suggested, tell me and I’ll include it.

The paper uses N = 5 oscillators in experiments; that is the default here. You can change N.

The paper reports updating the CPG at 200 Hz; default fs = 200 Hz / dt = 1/200. You can set another fs when running.

hopf_fit.py

Save the following as hopf_fit.py and run python hopf_fit.py --csv your_knee.csv (or add --fs 200 if your CSV has only values).

#!/usr/bin/env python3
"""
hopf_fit.py

Implements the adaptive Hopf oscillator learning rules (paper: Plaza et al., "Design of a Modular Exoskeleton...").
Equations (1)-(8) in the paper are implemented for the learning phase; implementation eqs (9)-(13) form the output.
Citation: design-of-a-mod.pdf (provided). See code comments for mapping to paper equations.

Usage:
    python hopf_fit.py --csv knee.csv          # CSV with columns 'time','knee' OR single column of knee angles
    python hopf_fit.py --csv knee.csv --fs 200 # if CSV has single column, give sampling rate (Hz)

Outputs:
 - results.npz : learned parameters (alpha, omega, phi, x,y trajectories) and reconstruction
 - plots: target_vs_recon.png, osc_contributions.png, error_over_time.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# -------------------------
# Utility functions
# -------------------------
def load_csv(path, fs=None):
    df = pd.read_csv(path)
    # Accept two-column format (time,knee) or single-column of angles
    if {'time','knee'}.issubset(df.columns):
        t = df['time'].values
        y = df['knee'].values
        if fs is None:
            # estimate sampling rate from time vector
            dt = np.median(np.diff(t))
            fs = 1.0 / dt
    else:
        # try to take first numeric column
        col = df.select_dtypes(include=[np.number]).columns[0]
        y = df[col].values
        if fs is None:
            raise ValueError("Single-column CSV detected: please supply sampling rate --fs")
        dt = 1.0 / fs
        t = np.arange(len(y)) * dt
    return t, y, fs

def lowpass(signal, fs, cutoff=15.0, order=2):
    b, a = butter(order, cutoff / (fs/2), btype='low')
    return filtfilt(b, a, signal)

def sgn(x): return np.sign(x) if np.ndim(x) == 0 else np.sign(x)

# -------------------------
# Core Hopf learning update (paper eqs 1-8)
# -------------------------
class AdaptiveHopfLearner:
    def __init__(self, Pteach, t, N=5,
                 gamma=50.0, mu=1.0, eta=0.01, tau=0.1, eps=0.05,
                 omega0=None, gain_alpha=1.0, delta=1e-6):
        """
        Pteach: target signal array (length T)
        t: time vector (length T)
        N: number oscillators (paper uses 5 by default)
        gamma, mu, eta, tau, eps: hyperparameters (paper mentions these concepts; values chosen sensibly)
        omega0: fundamental frequency (rad/s). If None, estimated from Pteach spectrum (dominant rad/s).
        gain_alpha: Gainα from implementation eq (13). Set 1.0 to follow learning directly.
        delta: small number to avoid division by zero (ri ~ 0)
        """
        self.P = Pteach
        self.t = t
        self.fs = 1.0 / (t[1]-t[0])
        self.dt = t[1] - t[0]
        self.N = N
        self.gamma = gamma
        self.mu = mu
        self.eta = eta
        self.tau = tau
        self.eps = eps
        self.delta = delta
        self.gain_alpha = gain_alpha

        # estimate fundamental freq (rad/s) if not supplied
        if omega0 is None:
            self.omega0 = self._estimate_fundamental_rad()
        else:
            self.omega0 = omega0

        # initialize state variables (xi, yi, alphai, omegai, phii)
        self.x = np.zeros(N)
        self.y = np.zeros(N)
        # initialize omegas to multiples (harmonics) of omega0
        # Paper uses learning so initial omegas can be zero or small; choose harmonic spread
        self.omega = np.array([(i+1) * self.omega0 for i in range(N)], dtype=float)
        # alpha initial
        self.alpha = np.zeros(N)
        # phases (phi0 is anchor)
        self.phi = np.zeros(N)
        # For numerical storage
        T = len(t)
        self.x_traj = np.zeros((T, N))
        self.y_traj = np.zeros((T, N))
        self.alpha_traj = np.zeros((T, N))
        self.omega_traj = np.zeros((T, N))
        self.phi_traj = np.zeros((T, N))
        self.Q = np.zeros(T)
        self.F = np.zeros(T)

    def _estimate_fundamental_rad(self):
        # simple FFT peak to estimate dominant frequency (in rad/s)
        # assumes P is periodic and reasonably long
        y = self.P - np.mean(self.P)
        n = len(y)
        yf = np.fft.rfft(y * np.hanning(n))
        f = np.fft.rfftfreq(n, d=self.dt)
        idx = np.argmax(np.abs(yf))
        f0 = f[idx]
        omega0 = 2.0 * np.pi * f0 if f0 > 0 else 2.5  # fallback 2.5 rad/s
        return omega0

    def _Ri(self, i):
        # Ri = (omega_i / omega0) * sgn(x0) * arccos(-y0/r0)    (paper eq 5)
        # uses oscillator 0 (index 0) as reference fundamental
        x0, y0 = self.x[0], self.y[0]
        r0 = np.sqrt(x0*x0 + y0*y0) + 1e-12
        arg = np.clip(-y0 / r0, -1.0, 1.0)
        base_angle = np.arccos(arg)
        Ri = (self.omega[i] / self.omega0) * sgn(x0) * base_angle
        return Ri

    def step(self, k):
        """
        perform one integration step corresponding to t[k] -> t[k+1]
        We'll use simple RK4 integration applied to the vector of variables:
        (x_i, y_i, omega_i, alpha_i, phi_i) per oscillator.
        Note: phi_0 dot = 0 (anchor).
        """
        dt = self.dt

        # current reconstruction Q and error F (paper eq 8)
        Q = np.sum(self.alpha * self.x) * self.gain_alpha
        F = self.P[k] - Q

        # store
        self.Q[k] = Q
        self.F[k] = F

        # compute derivatives according to paper eqs (1)-(7)
        def derivatives(state):
            # state is flattened vector: [x0..xN-1, y0..yN-1, omega0..omegaN-1, alpha0..alphaN-1, phi0..phiN-1]
            N = self.N
            xs = state[0:N]
            ys = state[N:2*N]
            omegas = state[2*N:3*N]
            alphas = state[3*N:4*N]
            phis = state[4*N:5*N]
            dx = np.zeros(N)
            dy = np.zeros(N)
            domega = np.zeros(N)
            dalpha = np.zeros(N)
            dphi = np.zeros(N)

            for i in range(N):
                ri = np.sqrt(xs[i]*xs[i] + ys[i]*ys[i]) + 1e-12
                # Ri uses current global x0,y0 from xs,ys
                # compute Ri exactly as paper eq 5
                x0 = xs[0]; y0 = ys[0]
                r0 = np.sqrt(x0*x0 + y0*y0) + 1e-12
                arg = np.clip(-y0/r0, -1.0, 1.0)
                base_angle = np.arccos(arg)
                Ri = (omegas[i] / self.omega0) * sgn(x0) * base_angle

                # xi_dot (eq 1): gamma(mu - ri^2) xi - omega_i yi + tau sin(Ri - phi_i) + eps * F
                dx[i] = self.gamma * (self.mu - ri*ri) * xs[i] - omegas[i] * ys[i] + self.tau * np.sin(Ri - phis[i]) + self.eps * F
                # yi_dot (eq 2)
                dy[i] = self.gamma * (self.mu - ri*ri) * ys[i] + omegas[i] * xs[i]
                # omega_dot (eq 3): - eps * F * yi / ri
                domega[i] = - self.eps * F * ys[i] / max(ri, self.delta)
                # alpha_dot (eq 4): eta * xi * F
                dalpha[i] = self.eta * xs[i] * F
                # phi_dot: phi_0 dot = 0 (eq 6), others eq 7
                if i == 0:
                    dphi[i] = 0.0
                else:
                    arg2 = np.clip(-ys[i] / ri, -1.0, 1.0)
                    inner = sgn(xs[i]) * np.arccos(arg2)
                    dphi[i] = np.sin(Ri - inner - phis[i])
            return np.concatenate([dx, dy, domega, dalpha, dphi])

        # pack current state
        s0 = np.concatenate([self.x, self.y, self.omega, self.alpha, self.phi])

        # RK4
        k1 = derivatives(s0)
        k2 = derivatives(s0 + 0.5 * dt * k1)
        k3 = derivatives(s0 + 0.5 * dt * k2)
        k4 = derivatives(s0 + dt * k3)
        s_next = s0 + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # unpack
        N = self.N
        self.x = s_next[0:N]
        self.y = s_next[N:2*N]
        self.omega = s_next[2*N:3*N]
        self.alpha = s_next[3*N:4*N]
        self.phi = s_next[4*N:5*N]

    def run(self):
        T = len(self.t)
        for k in range(T):
            # store before step for plotting / analysis
            self.x_traj[k,:] = self.x
            self.y_traj[k,:] = self.y
            self.alpha_traj[k,:] = self.alpha
            self.omega_traj[k,:] = self.omega
            self.phi_traj[k,:] = self.phi
            self.step(k)
        # final Q/F at last step
        # recompute final Q,F
        Q = np.sum(self.alpha * self.x) * self.gain_alpha
        F = self.P[-1] - Q
        self.Q[-1] = Q
        self.F[-1] = F
        return {
            'x': self.x_traj, 'y': self.y_traj, 'alpha': self.alpha_traj,
            'omega': self.omega_traj, 'phi': self.phi_traj,
            'Q': self.Q, 'F': self.F, 't': self.t
        }

# -------------------------
# Main: argument parsing and execution
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV file with knee angles (radians). Columns: time,knee OR single column of knee values.')
    parser.add_argument('--fs', type=float, default=None, help='Sampling rate in Hz if CSV is single column.')
    parser.add_argument('--N', type=int, default=5, help='Number of oscillators (default 5 as in paper).')
    parser.add_argument('--omega0', type=float, default=None, help='Fundamental omega0 (rad/s). If not provided, estimated from data.')
    parser.add_argument('--dt', type=float, default=None, help='Integration timestep (overrides sampling dt).')
    parser.add_argument('--save_prefix', default='hopf_out', help='prefix for saved outputs')
    args = parser.parse_args()

    t, y_raw, fs = load_csv(args.csv, fs=args.fs)
    # Resample or smooth if needed (paper used 200 Hz updates). We'll resample to 200 Hz if original fs differs.
    target_fs = 200.0
    if abs(fs - target_fs) > 1e-6:
        # simple resample via interpolation
        t_new = np.arange(0, t[-1], 1.0/target_fs)
        y = np.interp(t_new, t, y_raw)
        t = t_new
        fs = target_fs
    else:
        y = y_raw

    # Preprocessing: remove mean (paper says oscillatory zero-mean; if not, subtract bias)
    y = y - np.mean(y)

    # optional lowpass (uncomment if noisy)
    y = lowpass(y, fs, cutoff=15.0, order=2)

    # instantiate learner
    learner = AdaptiveHopfLearner(Pteach=y, t=t, N=args.N, omega0=args.omega0)
    print("Initialized learner: N={}, dt={:.5f}, estimated omega0={:.3f} rad/s".format(args.N, learner.dt, learner.omega0))

    results = learner.run()
    # compute final reconstruction
    Q = results['Q']

    # Save results and plotting
    np.savez(f"{args.save_prefix}_results.npz",
             t=t, P=y, Q=Q,
             x=results['x'], y_traj=results['y'], alpha=results['alpha'],
             omega=results['omega'], phi=results['phi'], F=results['F'])

    # Plots
    plt.figure(figsize=(10,4))
    plt.plot(t, y, label='target (Pteach)')
    plt.plot(t, Q, label='reconstruction Q(t)', alpha=0.8)
    plt.legend(); plt.title('Target vs Reconstruction'); plt.xlabel('t (s)')
    plt.savefig(f"{args.save_prefix}_target_vs_recon.png", dpi=150)

    # Error plot
    plt.figure(figsize=(8,3))
    plt.plot(t, results['F'])
    plt.title('Error F(t)'); plt.xlabel('t (s)')
    plt.savefig(f"{args.save_prefix}_error.png", dpi=150)

    # Oscillator contributions (alpha_i * x_i)
    contrib = results['alpha'] * results['x']
    plt.figure(figsize=(10,4))
    for i in range(args.N):
        plt.plot(t, contrib[:,i], label=f'osc {i}')
    plt.title('Oscillator contributions α_i x_i'); plt.xlabel('t (s)')
    plt.legend(ncol=2, fontsize='small')
    plt.savefig(f"{args.save_prefix}_osc_contribs.png", dpi=150)

    print("Saved results to {}_results.npz and PNG figures.".format(args.save_prefix))

if __name__ == "__main__":
    main()

How this maps to the paper

The learner implements eqs (1)–(8) (xi_dot, yi_dot, ω̇i, α̇i, Ri, φ̇i, F(t)). See paper for exact notation and derivation.

The implementation form (Gainα, communication term kij) used for runtime/exoskeleton is separately described in the paper (eqs (9)–(13)); the script focuses on the learning phase (train to reproduce knee trajectory) and produces alpha, omega, phi that you can deploy later per eqs (9)–(13).

The paper used N = 5 oscillators in their module experiments and updated at 200 Hz; those defaults are used. You can change them via the script arguments.

Next steps / what I need from you

Upload your knee CSV (if you haven’t yet). The script assumes radians; if your CSV is degrees I’ll include conversion in the script or you can convert beforehand.

Run the script locally (it’s self-contained, needs numpy, scipy, pandas, matplotlib). If you want, I can run it here — upload the CSV and tell me whether it’s time,knee or single-column + sampling rate.

After a first run, inspect target_vs_recon.png: if details (local peaks / asymmetry) are missing, we can:

increase N (more oscillators / harmonics),

initialize ω_i to a broader harmonic spread,

tune eta, eps, tau, or

add small noise/jitter or multiple cycles to training data — all changes I can make for you.

If you want I’ll now:

(A) run this exact script on your CSV if you upload the CSV, and return the generated plots and learned parameters; or

(B) immediately produce a variant that includes kij synchronization and the implementation-phase equations (9)–(13) so the learned parameters can be injected straight into the exoskeleton runtime.

Which do you want me to do next?

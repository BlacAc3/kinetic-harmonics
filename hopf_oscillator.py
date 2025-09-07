import numpy as np
from scipy.signal import hilbert

class HopfOscillator:
    def __init__(self, dt, mu=1.0, omega=2*np.pi, k=10.0, eta_a=1e-3, eta_b=1e-3, gamma=0.1, coupling_strength=None, state_clip=10.0, data_scale=1.0):
        self.dt = dt
        self.mu = mu
        self.omega = omega
        self.k = k
        self.eta_a = eta_a
        self.eta_b = eta_b
        self.gamma = gamma
        self.coupling_strength = coupling_strength
        self.state_clip = state_clip
        self.data_scale = data_scale

        self.x = 1.0
        self.y = 0.0
        self.a = 1.0
        self.b = 0.0

    def compute_coupling(self, x, index, all_x):
        # self.coupling_strength is expected to be a numpy array
        # representing a coupling matrix.  For example:
        # coupling_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
        # This matrix would define the coupling between two oscillators.
        # In this example, oscillator 0 is coupled to oscillator 1 with a
        # strength of 0.5, and oscillator 1 is coupled to oscillator 0 with
        # a strength of 0.5.  The diagonal elements are typically 0, as an
        # oscillator doesn't couple to itself.

        if self.coupling_strength is None:
            return 0.0
        coupling = 0.0
        for j, other_x in enumerate(all_x):
            if j != index:
                coupling += self.coupling_strength[index, j] * (other_x - x)
        return coupling

    def rk4_step(self, x, y, u_norm, coupling):
        k1_x = (self.mu - (x**2 + y**2)) * x - self.omega * y + self.k * (u_norm - x) + coupling
        k1_y = (self.mu - (x**2 + y**2)) * y + self.omega * x
        print(f"RK4 Step: Initial x={x:.4f}, y={y:.4f}, u_norm={u_norm:.4f}, coupling={coupling:.4f}")
        print(f"RK4 Step: k1_x={k1_x:.4f}, k1_y={k1_y:.4f}")

        k2_x = (self.mu - ((x + self.dt/2*k1_x)**2 + (y + self.dt/2*k1_y)**2)) * (x + self.dt/2*k1_x) - self.omega * (y + self.dt/2*k1_y) + self.k * (u_norm - (x + self.dt/2*k1_x)) + coupling
        k2_y = (self.mu - ((x + self.dt/2*k1_x)**2 + (y + self.dt/2*k1_y)**2)) * (y + self.dt/2*k1_y) + self.omega * (x + self.dt/2*k1_x)
        print(f"RK4 Step: k2_x={k2_x:.4f}, k2_y={k2_y:.4f}")

        k3_x = (self.mu - ((x + self.dt/2*k2_x)**2 + (y + self.dt/2*k2_y)**2)) * (x + self.dt/2*k2_x) - self.omega * (y + self.dt/2*k2_y) + self.k * (u_norm - (x + self.dt/2*k2_x)) + coupling
        k3_y = (self.mu - ((x + self.dt/2*k2_x)**2 + (y + self.dt/2*k2_y)**2)) * (y + self.dt/2*k2_y) + self.omega * (x + self.dt/2*k2_x)
        print(f"RK4 Step: k3_x={k3_x:.4f}, k3_y={k3_y:.4f}")

        k4_x = (self.mu - ((x + self.dt*k3_x)**2 + (y + self.dt*k3_y)**2)) * (x + self.dt*k3_x) - self.omega * (y + self.dt*k3_y) + self.k * (u_norm - (x + self.dt*k3_x)) + coupling
        k4_y = (self.mu - ((x + self.dt*k3_x)**2 + (y + self.dt*k3_y)**2)) * (y + self.dt*k3_y) + self.omega * (x + self.dt*k3_x)
        print(f"RK4 Step: k4_x={k4_x:.4f}, k4_y={k4_y:.4f}")

        x += self.dt / 6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        y += self.dt / 6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

        print(f"RK4 Step: Updated x={x:.4f}, y={y:.4f}")
        return x, y

    def cost_function(self, actual_data, mu, omega):
        if actual_data.ndim == 1:
            actual_data = actual_data.reshape(-1, 1)

        n_timesteps, n_joints = actual_data.shape
        predicted_data = np.zeros_like(actual_data)

        all_x = np.ones(n_joints)
        all_y = np.zeros(n_joints)

        data_mean = actual_data.mean(axis=0)
        data_amp = np.ptp(actual_data, axis=0) / 2
        scaled_data = (actual_data - data_mean) / (data_amp + 1e-9)
        print(f"Cost Function: data_mean={data_mean}, data_amp={data_amp}")
        print(f"Cost Function: scaled_data={scaled_data}")

        for t in range(n_timesteps):
            for i in range(n_joints):
                x = all_x[i]
                y = all_y[i]
                print(f"Cost Function: Initial all_x[{i}]={x:.4f}, all_y[{i}]={y:.4f}")

                u_norm = scaled_data[t,i] * self.data_scale
                print(f"Cost Function: u_norm[{t},{i}]={u_norm:.4f}")

                coupling = 0
                k1_x = (mu - (x**2 + y**2)) * x - omega * y + self.k * (u_norm - x) + coupling
                k1_y = (mu - (x**2 + y**2)) * y + omega * x
                print(f"Cost Function: k1_x={k1_x:.4f}, k1_y={k1_y:.4f}")

                k2_x = (mu - ((x + self.dt/2*k1_x)**2 + (y + self.dt/2*k1_y)**2)) * (x + self.dt/2*k1_x) - omega * (y + self.dt/2*k1_y) + self.k * (u_norm - (x + self.dt/2*k1_x)) + coupling
                k2_y = (mu - ((x + self.dt/2*k1_x)**2 + (y + self.dt/2*k1_y)**2)) * (y + self.dt/2*k1_y) + self.omega * (x + self.dt/2*k1_x)
                print(f"Cost Function: k2_x={k2_x:.4f}, k2_y={k2_y:.4f}")

                k3_x = (mu - ((x + self.dt/2*k2_x)**2 + (y + self.dt/2*k2_y)**2)) * (x + self.dt/2*k2_x) - omega * (y + self.dt/2*k2_y) + self.k * (u_norm - (x + self.dt/2*k2_x)) + coupling
                k3_y = (mu - ((x + self.dt/2*k2_x)**2 + (y + self.dt/2*k2_y)**2)) * (y + self.dt/2*k2_y) + self.omega * (x + self.dt/2*k2_x)
                print(f"Cost Function: k3_x={k3_x:.4f}, k3_y={k3_y:.4f}")

                k4_x = (mu - ((x + self.dt*k3_x)**2 + (y + self.dt*k3_y)**2)) * (x + self.dt*k3_x) - omega * (y + self.dt*k3_y) + self.k * (u_norm - (x + self.dt*k3_x)) + coupling
                k4_y = (mu - ((x + self.dt*k3_x)**2 + (y + self.dt*k3_y)**2)) * (y + self.dt*k3_y) + self.omega * (x + self.dt*k3_x)
                print(f"Cost Function: k4_x={k4_x:.4f}, k4_y={k4_y:.4f}")

                x += self.dt / 6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
                y += self.dt / 6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

                x = np.clip(x, -self.state_clip, self.state_clip)
                y = np.clip(y, -self.state_clip, self.state_clip)

                all_x[i] = x
                all_y[i] = y
                print(f"Cost Function: Updated all_x[{i}]={x:.4f}, all_y[{i}]={y:.4f}")

                predicted_data[t, i] = x
                print(f"Cost Function: predicted_data[{t},{i}]={x:.4f}")

        mse = np.mean((predicted_data - actual_data)**2)
        print(f"Cost Function: MSE={mse:.4f}, mu={mu:.4f}, omega={omega:.4f}")
        return mse

    def run(self, data):
        mu_values = np.linspace(0.001, 0.1, 5)
        omega_values = np.linspace(0.1, 1.0, 5)

        best_mu = None
        best_omega = None
        min_cost = float('inf')

        for mu in mu_values:
            for omega in omega_values:
                cost = self.cost_function(data, mu, omega)
                if cost < min_cost:
                    min_cost = cost
                    best_mu = mu
                    best_omega = omega
                    print(f"Run: New best cost found: min_cost={min_cost:.4f}, best_mu={best_mu:.4f}, best_omega={best_omega:.4f}")

        if best_mu is None or best_omega is None:
            raise ValueError("Grid search failed to find optimal parameters")

        self.mu = best_mu
        self.omega = best_omega
        print(f"Run: Optimal parameters found: mu={self.mu:.4f}, omega={self.omega:.4f}")

        mean = data.mean(axis=0)
        amp = np.ptp(data, axis=0) / 2
        u_norm = (data - mean) / (amp + 1e-9)
        print(f"Run: Data preprocessing: mean={mean}, amp={amp}")
        print(f"Run: Normalized data: u_norm={u_norm}")
        phi_u = np.angle(hilbert(u_norm, axis=0)) # This calculates the phase angle

        n_timesteps = data.shape[0]
        n_joints = 1 if len(data.shape) == 1 else data.shape[1]
        predictions = np.zeros_like(data)

        all_x = np.ones(n_joints)
        all_y = np.zeros(n_joints)
        self.a = 1.0
        self.b = 0.0
        print(f"Run: Initial states: all_x={all_x}, all_y={all_y}, a={self.a:.4f}, b={self.b:.4f}")

        for t in range(n_timesteps):
            if n_joints == 1:
                coupling = 0.0
                print(f"Run: Time step {t}: Single joint, coupling={coupling:.4f}")
                print(f"Run: Before RK4, self.x={self.x:.4f}, self.y={self.y:.4f}")
                self.x, self.y = self.rk4_step(self.x, self.y, u_norm[t], coupling)
                print(f"Run: After RK4, self.x={self.x:.4f}, self.y={self.y:.4f}")
                pred = self.a * self.x + self.b
                print(f"Run: Prediction: pred={pred:.4f}")
                e = u_norm[t] - pred
                print(f"Run: Error: e={e:.4f}")
                self.a += self.eta_a * e * self.x
                self.b += self.eta_b * e
                print(f"Run: Updated a={self.a:.4f}, b={self.b:.4f}")

                phi_x = np.arctan2(self.y, self.x)
                dphi = np.angle(np.exp(1j*(phi_u[t] - phi_x)))
                self.omega += self.gamma * np.sin(dphi) * self.dt
                print(f"Run: Phase adjustment: phi_x={phi_x:.4f}, dphi={dphi:.4f}, omega={self.omega:.4f}")

                predictions[t] = pred
                all_x[0] = self.x
                all_y[0] = self.y
                print(f"Run: Updated all_x[0]={all_x[0]:.4f}, all_y[0]={all_y[0]:.4f}")

            else:
                current_x = all_x.copy()
                current_y = all_y.copy()
                print(f"Run: Time step {t}: Multiple joints, current_x={current_x}, current_y={current_y}")

                for i in range(n_joints):
                    coupling = self.compute_coupling(current_x[i], i, current_x)
                    print(f"Run: Joint {i}: coupling={coupling:.4f}")
                    x, y = self.rk4_step(current_x[i], current_y[i], u_norm[t, i], coupling)
                    all_x[i] = x
                    all_y[i] = y
                    print(f"Run: Joint {i}: After RK4, all_x[{i}]={all_x[i]:.4f}, all_y[{i}]={all_y[i]:.4f}")

                    pred = self.a * x + self.b
                    print(f"Run: Joint {i}: Prediction: pred={pred:.4f}")
                    e = u_norm[t, i] - pred
                    print(f"Run: Joint {i}: Error: e={e:.4f}")
                    self.a += self.eta_a * e * x
                    self.b += self.eta_b * e
                    print(f"Run: Joint {i}: Updated a={self.a:.4f}, b={self.b:.4f}")

                    phi_x = np.arctan2(y, x)
                    dphi = np.angle(np.exp(1j*(phi_u[t, i] - phi_x)))
                    self.omega += self.gamma * np.sin(dphi) * self.dt
                    print(f"Run: Joint {i}: Phase adjustment: phi_x={phi_x:.4f}, dphi={dphi:.4f}, omega={self.omega:.4f}")

                    predictions[t, i] = pred
                    print(f"Run: Joint {i}: predictions[{t},{i}] = {predictions[t,i]:.4f}")

        predictions = predictions * (amp + 1e-9) + mean
        print(f"Run: Final predictions: predictions={predictions}")

        return predictions

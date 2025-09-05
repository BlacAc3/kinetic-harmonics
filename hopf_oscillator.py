import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize
from scipy.stats import norm
import random
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

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
        self.state_clip = state_clip # Add state clip
        self.data_scale = data_scale

        self.x = 1.0  # Initial state
        self.y = 0.0
        self.a = 1.0  # Initial readout parameters
        self.b = 0.0

    def compute_coupling(self, x, index, all_x):
        if self.coupling_strength is None:
            return 0.0
        coupling = 0.0
        for j, other_x in enumerate(all_x):
            if j != index:
                coupling += self.coupling_strength[index, j] * (other_x - x)
        return coupling

    def rk4_step(self, x, y, u_norm, coupling):
        """Performs a single RK4 integration step."""
        k1_x = (self.mu - (x**2 + y**2)) * x - self.omega * y + self.k * (u_norm - x) + coupling
        k1_y = (self.mu - (x**2 + y**2)) * y + self.omega * x

        k2_x = (self.mu - ((x + self.dt/2*k1_x)**2 + (y + self.dt/2*k1_y)**2)) * (x + self.dt/2*k1_x) - self.omega * (y + self.dt/2*k1_y) + self.k * (u_norm - (x + self.dt/2*k1_x)) + coupling
        k2_y = (self.mu - ((x + self.dt/2*k1_x)**2 + (y + self.dt/2*k1_y)**2)) * (y + self.dt/2*k1_y) + self.omega * (x + self.dt/2*k1_x)

        k3_x = (self.mu - ((x + self.dt/2*k2_x)**2 + (y + self.dt/2*k2_y)**2)) * (x + self.dt/2*k2_x) - self.omega * (y + self.dt/2*k2_y) + self.k * (u_norm - (x + self.dt/2*k2_x)) + coupling
        k3_y = (self.mu - ((x + self.dt/2*k2_x)**2 + (y + self.dt/2*k2_y)**2)) * (y + self.dt/2*k2_y) + self.omega * (x + self.dt/2*k2_x)

        k4_x = (self.mu - ((x + self.dt*k3_x)**2 + (y + self.dt*k3_y)**2)) * (x + self.dt*k3_x) - self.omega * (y + self.dt*k3_y) + self.k * (u_norm - (x + self.dt*k3_x)) + coupling
        k4_y = (self.mu - ((x + self.dt*k3_x)**2 + (y + self.dt*k3_y)**2)) * (y + self.dt*k3_y) + self.omega * (x + self.dt*k3_x)

        x += self.dt / 6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        y += self.dt / 6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

        return x, y

    def cost_function(self, actual_data, mu, omega):
        """
        Calculates the Mean Squared Error (MSE) between the predicted data from the Hopf oscillator
        and the actual data.

        Parameters:
        - actual_data: A 2D numpy array representing the actual time series data.
        - mu: The amplitude parameter of the Hopf oscillator.
        - omega: The angular frequency of the Hopf oscillator.

        Returns:
        - mse: The Mean Squared Error.
        """
        # Ensure actual_data is 2D
        if actual_data.ndim == 1:
            actual_data = actual_data.reshape(-1, 1)

        # Direct calculation of predicted data within the cost function
        n_timesteps, n_joints = actual_data.shape
        predicted_data = np.zeros_like(actual_data)

        # Initialize oscillator states
        all_x = np.ones(n_joints)
        all_y = np.zeros(n_joints)

        # Scale the data
        data_mean = actual_data.mean(axis=0)
        data_amp = np.ptp(actual_data, axis=0) / 2
        scaled_data = (actual_data - data_mean) / (data_amp + 1e-9)  # Normalize

        for t in range(n_timesteps):
            for i in range(n_joints):
                x = all_x[i]
                y = all_y[i]

                # Normalize input
                u_norm = scaled_data[t,i] * self.data_scale

                # Calculate the derivatives and update states using RK4
                coupling = 0  # No coupling in cost function
                k1_x = (mu - (x**2 + y**2)) * x - omega * y + self.k * (u_norm - x) + coupling
                k1_y = (mu - (x**2 + y**2)) * y + omega * x

                k2_x = (mu - ((x + self.dt/2*k1_x)**2 + (y + self.dt/2*k1_y)**2)) * (x + self.dt/2*k1_x) - omega * (y + self.dt/2*k1_y) + self.k * (u_norm - (x + self.dt/2*k1_x)) + coupling
                k2_y = (mu - ((x + self.dt/2*k1_x)**2 + (y + self.dt/2*k1_y)**2)) * (y + self.dt/2*k1_y) + omega * (x + self.dt/2*k1_x)

                k3_x = (mu - ((x + self.dt/2*k2_x)**2 + (y + self.dt/2*k2_y)**2)) * (x + self.dt/2*k2_x) - omega * (y + self.dt/2*k2_y) + self.k * (u_norm - (x + self.dt/2*k2_x)) + coupling
                k3_y = (mu - ((x + self.dt/2*k2_x)**2 + (y + self.dt/2*k2_y)**2)) * (y + self.dt/2*k2_y) + omega * (x + self.dt/2*k2_x)

                k4_x = (mu - ((x + self.dt*k3_x)**2 + (y + self.dt*k3_y)**2)) * (x + self.dt*k3_x) - omega * (y + self.dt*k3_y) + self.k * (u_norm - (x + self.dt*k3_x)) + coupling
                k4_y = (mu - ((x + self.dt*k3_x)**2 + (y + self.dt*k3_y)**2)) * (y + self.dt*k3_y) + omega * (x + self.dt*k3_x)

                x += self.dt / 6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
                y += self.dt / 6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

                # Clip the states
                x = np.clip(x, -self.state_clip, self.state_clip)
                y = np.clip(y, -self.state_clip, self.state_clip)

                all_x[i] = x
                all_y[i] = y

                # Store the predicted data
                predicted_data[t, i] = x

        mse = np.mean((predicted_data - actual_data)**2)
        return mse

    def cost_function_with_amplitude_preservation(self, actual_data, mu, omega, alpha=0.1):
        """
        Calculates the cost as a combination of MSE and amplitude preservation.

        Parameters:
        - actual_data: A 2D numpy array representing the actual time series data.
        - mu: The amplitude parameter of the Hopf oscillator.
        - omega: The angular frequency of the Hopf oscillator.
        - alpha: Weighting factor for the amplitude preservation term.

        Returns:
        - cost: The combined cost.
        """
        # Ensure actual_data is 2D
        if actual_data.ndim == 1:
            actual_data = actual_data.reshape(-1, 1)

        # Direct calculation of predicted data within the cost function
        n_timesteps, n_joints = actual_data.shape
        predicted_data = np.zeros_like(actual_data)

        # Initialize oscillator states
        all_x = np.ones(n_joints)
        all_y = np.zeros(n_joints)

        # Scale the data
        data_mean = actual_data.mean(axis=0)
        data_amp = np.ptp(actual_data, axis=0) / 2
        scaled_data = (actual_data - data_mean) / (data_amp + 1e-9)  # Normalize

        for t in range(n_timesteps):
            for i in range(n_joints):
                x = all_x[i]
                y = all_y[i]

                # Normalize input
                u_norm = scaled_data[t,i] * self.data_scale

                # Calculate the derivatives and update states using RK4
                coupling = 0  # No coupling in cost function
                k1_x = (mu - (x**2 + y**2)) * x - omega * y + self.k * (u_norm - x) + coupling
                k1_y = (mu - (x**2 + y**2)) * y + omega * x

                k2_x = (mu - ((x + self.dt/2*k1_x)**2 + (y + self.dt/2*k1_y)**2)) * (x + self.dt/2*k1_x) - omega * (y + self.dt/2*k1_y) + self.k * (u_norm - (x + self.dt/2*k1_x)) + coupling
                k2_y = (mu - ((x + self.dt/2*k1_x)**2 + (y + self.dt/2*k1_y)**2)) * (y + self.dt/2*k1_y) + omega * (x + self.dt/2*k1_x)

                k3_x = (mu - ((x + self.dt/2*k2_x)**2 + (y + self.dt/2*k2_y)**2)) * (x + self.dt/2*k2_x) - omega * (y + self.dt/2*k2_y) + self.k * (u_norm - (x + self.dt/2*k2_x)) + coupling
                k3_y = (mu - ((x + self.dt/2*k2_x)**2 + (y + self.dt/2*k2_y)**2)) * (y + self.dt/2*k2_y) + omega * (x + self.dt/2*k2_x)

                k4_x = (mu - ((x + self.dt*k3_x)**2 + (y + self.dt*k3_y)**2)) * (x + self.dt*k3_x) - omega * (y + self.dt*k3_y) + self.k * (u_norm - (x + self.dt*k3_x)) + coupling
                k4_y = (mu - ((x + self.dt*k3_x)**2 + (y + self.dt*k3_y)**2)) * (y + self.dt*k3_y) + omega * (x + self.dt*k3_x)

                x += self.dt / 6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
                y += self.dt / 6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)

                # Clip the states
                x = np.clip(x, -self.state_clip, self.state_clip)
                y = np.clip(y, -self.state_clip, self.state_clip)

                all_x[i] = x
                all_y[i] = y

                # Store the predicted data
                predicted_data[t,i] = x

        # Calculate MSE
        mse = np.mean((predicted_data - actual_data)**2)

        # Calculate amplitude error
        actual_amplitude = np.max(scaled_data, axis=0) - np.min(scaled_data, axis=0)
        predicted_amplitude = np.max(predicted_data, axis=0) - np.min(predicted_data, axis=0)
        amplitude_error = np.mean((predicted_amplitude - actual_amplitude)**2)

        # Combine MSE and amplitude error
        cost = mse + alpha * amplitude_error
        return cost

    def run(self, data):
        # Optimize parameters using Bayesian optimization
        # Preprocessing - calculate mean and amplitude for scaling back later
        mean = data.mean(axis=0)
        amp = np.ptp(data, axis=0) / 2
        u_norm = (data - mean) / (amp + 1e-9)
        optimizer = BayesianOptimizer(self, bounds={'mu': (0.1, 2.0), 'omega': (0.1, 2.0)}, data_scale = self.data_scale)
        best_params = optimizer.optimize(data)

        if not best_params:
            raise ValueError("Optimization failed, could not find optimal parameters")

        self.mu = best_params['mu']
        self.omega = best_params['omega']


        phi_u = np.angle(hilbert(u_norm, axis=0))


        n_timesteps = data.shape[0]
        n_joints = 1 if len(data.shape) == 1 else data.shape[1]
        predictions = np.zeros_like(data)

        # Initialize oscillator states for each joint
        all_x = np.ones(n_joints)
        all_y = np.zeros(n_joints)
        self.a = 1.0
        self.b = 0.0

        for t in range(n_timesteps):
            # Core loop
            if n_joints == 1:
                coupling = 0.0  # No coupling if only one joint
                self.x, self.y = self.rk4_step(self.x, self.y, u_norm[t], coupling)
                pred = self.a * self.x + self.b
                e = u_norm[t] - pred
                self.a += self.eta_a * e * self.x
                self.b += self.eta_b * e

                phi_x = np.arctan2(self.y, self.x)
                dphi = np.angle(np.exp(1j*(phi_u[t] - phi_x)))
                self.omega += self.gamma * np.sin(dphi) * self.dt

                predictions[t] = pred
                all_x[0] = self.x
                all_y[0] = self.y

            else:
                # Multiple joints
                current_x = all_x.copy()
                current_y = all_y.copy()

                for i in range(n_joints):
                    coupling = self.compute_coupling(current_x[i], i, current_x)
                    x, y = self.rk4_step(current_x[i], current_y[i], u_norm[t, i], coupling)
                    all_x[i] = x
                    all_y[i] = y

                    pred = self.a * x + self.b
                    e = u_norm[t, i] - pred
                    self.a += self.eta_a * e * x
                    self.b += self.eta_b * e

                    phi_x = np.arctan2(y, x)
                    dphi = np.angle(np.exp(1j*(phi_u[t, i] - phi_x)))
                    self.omega += self.gamma * np.sin(dphi) * self.dt

                    predictions[t, i] = pred

        # Scale predictions back to the original range
        predictions = predictions * (amp + 1e-9) + mean


        return predictions

class BayesianOptimizer:
    """
    A class to perform Bayesian Optimization for tuning the parameters of the Hopf Oscillator.
    """

    def __init__(self, hopf_oscillator, bounds, n_iter=25, random_starts=5, data_scale=1.0):
        """
        Initialize the Bayesian Optimizer.

        Parameters:
        - hopf_oscillator: An instance of the HopfOscillator class.
        - bounds: A dictionary specifying the bounds for 'mu' and 'omega'.
                  Example: {'mu': (0.1, 2.0), 'omega': (0.1, 2.0)}
        - n_iter: The number of optimization iterations (default: 25).
        - random_starts: The number of random initial points (default: 5).
        """
        self.hopf_oscillator = hopf_oscillator
        self.bounds = bounds
        self.n_iter = n_iter
        self.random_starts = random_starts
        self.data_scale = data_scale
        self.samples = []
        self.values = []

    def acquisition_function(self, x, gp):
        """
        Compute the acquisition function (Expected Improvement).

        Parameters:
        - x: The input point to evaluate.
        - gp: The Gaussian Process model.

        Returns:
        - ei: The Expected Improvement at point x.
        """
        mean, std = gp.predict(np.array([x]), return_std=True)
        best_value = min(self.values)
        z = (best_value - mean) / (std + 1e-9)
        ei = (best_value - mean) * norm.cdf(z) + std * norm.pdf(z)
        return ei

    def optimize(self, actual_data):
        """
        Perform Bayesian Optimization to find the best parameters for the Hopf Oscillator.

        Parameters:
        - actual_data: A 2D numpy array representing the actual time series data.

        Returns:
        - best_params: A dictionary containing the optimized 'mu' and 'omega'.
        """

        # Initialize the Gaussian Process model
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

        # Random initialization
        for _ in range(self.random_starts):
            mu = random.uniform(*self.bounds['mu'])
            omega = random.uniform(*self.bounds['omega'])
            cost = self.hopf_oscillator.cost_function_with_amplitude_preservation(actual_data, mu, omega, alpha=0.1) # ,data_scale = self.data_scale
            self.samples.append([mu, omega])
            self.values.append(cost)

        # Bayesian Optimization loop
        for _ in range(self.n_iter):
            # Fit the GP model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(self.samples, self.values)

            # Minimize the acquisition function
            def objective(x):
                return -self.acquisition_function(x, gp)

            res = minimize(objective, x0=np.mean(list(self.bounds.values()), axis=1),
                           bounds=list(self.bounds.values()), method='L-BFGS-B')

            # Evaluate the cost function at the new point
            mu, omega = res.x
            cost = self.hopf_oscillator.cost_function(actual_data, mu, omega)
            self.samples.append([mu, omega])
            self.values.append(cost)
        # Return the best parameters
        # The best parameters are determined by finding the index of the minimum value in the `self.values` list,
        # which stores the cost function evaluations for each set of sampled parameters.
        # The corresponding parameters (`mu` and `omega`) at this index are then retrieved from `self.samples`.
        best_index = np.argmin(self.values)
        best_params = {'mu': self.samples[best_index][0], 'omega': self.samples[best_index][1]}
        return best_params

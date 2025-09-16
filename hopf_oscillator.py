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

                if t < 10 and i < 10:
                    print(f"Cost Function: Initial all_x[{i}]={x:.4f}, all_y[{i}]={y:.4f}")

                u_norm = scaled_data[t,i] * self.data_scale
                if t < 10 and i < 10:
                    print(f"Cost Function: u_norm[{t},{i}]={u_norm:.4f}")

                coupling = 0
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

                x = np.clip(x, -self.state_clip, self.state_clip)
                y = np.clip(y, -self.state_clip, self.state_clip)

                all_x[i] = x
                all_y[i] = y

                predicted_data[t, i] = x

        mse = np.mean((predicted_data - actual_data)**2)
        # print(f"Cost Function: MSE={mse:.4f}, mu={mu:.4f}, omega={omega:.4f}") # Removed verbose prints
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
        if actual_data.ndim == 1:
            actual_data = actual_data.reshape(-1, 1)

        n_timesteps, n_joints = actual_data.shape
        predicted_data = np.zeros_like(actual_data)

        all_x = np.ones(n_joints)
        all_y = np.zeros(n_joints)

        data_mean = actual_data.mean(axis=0)
        data_amp = np.ptp(actual_data, axis=0) / 2
        scaled_data = (actual_data - data_mean) / (data_amp + 1e-9)

        for t in range(n_timesteps):
            for i in range(n_joints):
                x = all_x[i]
                y = all_y[i]

                u_norm = scaled_data[t,i] * self.data_scale
                coupling = 0 # No coupling in cost function for parameter optimization

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

                x = np.clip(x, -self.state_clip, self.state_clip)
                y = np.clip(y, -self.state_clip, self.state_clip)

                all_x[i] = x
                all_y[i] = y
                predicted_data[t, i] = x

        # Calculate MSE
        mse = np.mean((predicted_data - actual_data)**2)

        # Calculate amplitude error
        actual_amplitude = actual_data.max(axis=0) - actual_data.min(axis=0)
        predicted_amplitude = predicted_data.max(axis=0) - predicted_data.min(axis=0)
        amplitude_error = np.mean((predicted_amplitude - actual_amplitude)**2)

        # Combine MSE and amplitude error
        cost = mse + alpha * amplitude_error
        return cost

    def run(self, data):
        # Ensure data is 2D for Bayesian Optimization. If the input data is 1D, reshape it to 2D.
        if data.ndim == 1:
            data_2d = data.reshape(-1, 1)
        else:
            data_2d = data

        # Define bounds for Bayesian Optimization.  These bounds specify the search space for the 'mu' and 'omega' parameters.
        # Adjusted mu lower bound to match the original grid search starting point
        bounds = {'mu': (0.001, 2.0), 'omega': (0.1, 2.0)}

        # Initialize Bayesian Optimizer.  Create an instance of the BayesianOptimizer class, passing in the current HopfOscillator instance and the defined bounds.
        optimizer = BayesianOptimizer(self, bounds)

        # Optimize parameters.  Run the Bayesian optimization process to find the best values for 'mu' and 'omega' that minimize the cost function.
        best_params = optimizer.optimize(data_2d)

        # Check if optimization was successful.  If the Bayesian optimization fails to find optimal parameters, raise a ValueError exception.
        if not best_params:
            raise ValueError("Optimization failed, could not find optimal parameters")

        # Update Hopf oscillator parameters with optimized values.  Set the 'mu' and 'omega' attributes of the HopfOscillator instance to the best values found by the optimizer.
        self.mu = best_params['mu']
        self.omega = best_params['omega']
        print(f"Run: Optimal parameters found using Bayesian Optimization: mu={self.mu:.4f}, omega={self.omega:.4f}")

        # Calculate the mean and amplitude of the input data for normalization.
        mean = data.mean(axis=0)
        amp = np.ptp(data, axis=0) / 2
        # Normalize the data to a range centered around zero. This helps with the stability and performance of the oscillator.
        u_norm = (data - mean) / (amp + 1e-9)
        print(f"Run: Data preprocessing: mean={mean}, amp={amp}")
        print(f"Run: Normalized data: u_norm={u_norm}")
        phi_u = np.angle(hilbert(u_norm, axis=0)) # This calculates the phase angle

        # Determine the number of timesteps and joints in the input data.
        n_timesteps = data.shape[0]
        n_joints = 1 if len(data.shape) == 1 else data.shape[1]
        # Initialize an array to store the predictions of the Hopf oscillator.
        predictions = np.zeros_like(data)

        # Initialize the state variables for each joint.
        all_x = np.ones(n_joints)
        all_y = np.zeros(n_joints)
        self.a = 1.0
        self.b = 0.0
        print(f"Run: Initial states: all_x={all_x}, all_y={all_y}, a={self.a:.4f}, b={self.b:.4f}")

        # Iterate over each timestep in the input data.
        for t in range(n_timesteps):
            # Handle the single-joint case.
            if n_joints == 1:
                coupling = 0.0
                if t < 10:
                    print(f"Run: Time step {t}: Single joint, coupling={coupling:.4f}")
                    print(f"Run: Before RK4, self.x={self.x:.4f}, self.y={self.y:.4f}")
                # Update the oscillator state using the RK4 method.
                self.x, self.y = self.rk4_step(self.x, self.y, u_norm[t], coupling)
                if t < 10:
                    print(f"Run: After RK4, self.x={self.x:.4f}, self.y={self.y:.4f}")
                # Calculate the prediction based on the current state and adaptive parameters.
                pred = self.a * self.x + self.b
                if t < 10:
                    print(f"Run: Prediction: pred={pred:.4f}")
                # Calculate the error between the normalized data and the prediction.
                e = u_norm[t] - pred
                if t < 10:
                    print(f"Run: Error: e={e:.4f}")
                # Update the adaptive parameters 'a' and 'b' based on the error.
                self.a += self.eta_a * e * self.x
                self.b += self.eta_b * e
                if t < 10:
                    print(f"Run: Updated a={self.a:.4f}, b={self.b:.4f}")

                # Adjust the oscillator frequency based on the phase difference between the input data and the oscillator state.
                # Calculate the phase angle of the oscillator's current state (x, y).  arctan2 handles the signs of x and y correctly to determine the quadrant.
                phi_x = np.arctan2(self.y, self.x)
                # Calculate the phase difference (dphi) between the input signal's phase (phi_u[t]) and the oscillator's phase (phi_x).
                # The complex exponential and np.angle ensure a correct phase difference calculation within the range of -pi to pi.
                dphi = np.angle(np.exp(1j*(phi_u[t] - phi_x)))
                # Adjust the oscillator's frequency (omega) based on the phase difference.  The adjustment is scaled by the learning rate (self.gamma) and the sine of the phase difference. The dt factor scales the change based on the timestep.
                self.omega += self.gamma * np.sin(dphi) * self.dt
                # Print the phase adjustment details for the first 10 timesteps for debugging.
                if t < 10:
                    print(f"Run: Phase adjustment: phi_x={phi_x:.4f}, dphi={dphi:.4f}, omega={self.omega:.4f}")

                # Store the prediction.
                predictions[t] = pred
                all_x[0] = self.x
                all_y[0] = self.y
                if t < 10:
                    print(f"Run: Updated all_x[0]={all_x[0]:.4f}, all_y[0]={all_y[0]:.4f}")

            # Handle the multi-joint case.
            else:
                # Create copies of the current state variables to avoid modifying them directly during the loop.
                current_x = all_x.copy()
                current_y = all_y.copy()
                if t < 10:
                    print(f"Run: Time step {t}: Multiple joints, current_x={current_x}, current_y={current_y}")

                # Iterate over each joint.
                for i in range(n_joints):
                    # Compute the coupling force from other oscillators.
                    coupling = self.compute_coupling(current_x[i], i, current_x)
                    if t < 10 and i < 10:
                        print(f"Run: Joint {i}: coupling={coupling:.4f}")
                    # Update the oscillator state using the RK4 method.
                    x, y = self.rk4_step(current_x[i], current_y[i], u_norm[t, i], coupling)
                    all_x[i] = x
                    all_y[i] = y
                    if t < 10 and i < 10:
                        print(f"Run: Joint {i}: After RK4, all_x[{i}]={all_x[i]:.4f}, all_y[{i}]={all_y[i]:.4f}")

                    # Calculate the prediction based on the current state and adaptive parameters.
                    pred = self.a * x + self.b
                    if t < 10 and i < 10:
                        print(f"Run: Joint {i}: Prediction: pred={pred:.4f}")
                    # Calculate the error between the normalized data and the prediction.
                    e = u_norm[t, i] - pred
                    if t < 10 and i < 10:
                        print(f"Run: Joint {i}: Error: e={e:.4f}")
                    # Update the adaptive parameters 'a' and 'b' based on the error.
                    self.a += self.eta_a * e * x
                    self.b += self.eta_b * e
                    if t < 10 and i < 10:
                        print(f"Run: Joint {i}: Updated a={self.a:.4f}, b={self.b:.4f}")

                    # Adjust the oscillator frequency based on the phase difference between the input data and the oscillator state.
                    phi_x = np.arctan2(y, x)
                    dphi = np.angle(np.exp(1j*(phi_u[t, i] - phi_x)))
                    self.omega += self.gamma * np.sin(dphi) * self.dt
                    if t < 10 and i < 10:
                        print(f"Run: Joint {i}: Phase adjustment: phi_x={phi_x:.4f}, dphi={dphi:.4f}, omega={self.omega:.4f}")

                    # Store the prediction.
                    predictions[t, i] = pred
                    if t < 10 and i < 10:
                        print(f"Run: Joint {i}: predictions[{t},{i}] = {predictions[t,i]:.4f}")

        # Scale the predictions back to the original data range.
        predictions = predictions * (amp + 1e-9) + mean
        print(f"Run: Final predictions: predictions={predictions}")

        # Return the predictions.
        return predictions


class BayesianOptimizer:
    """
    A class to perform Bayesian Optimization for tuning the parameters of the Hopf Oscillator.
    """

    def __init__(self, hopf_oscillator, bounds, n_iter=25, random_starts=5):
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
        # Ensure x is 2D array for gp.predict
        x_2d = np.array(x).reshape(1, -1)
        mean, std = gp.predict(x_2d, return_std=True)

        # Add a small epsilon to std to avoid division by zero
        std = np.maximum(std, 1e-9)

        best_value = np.min(self.values) # Use numpy min for robustness

        # Calculate Expected Improvement
        z = (best_value - mean) / std
        ei = (best_value - mean) * norm.cdf(z) + std * norm.pdf(z)
        return ei[0] # Return scalar value

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
            cost = self.hopf_oscillator.cost_function_with_amplitude_preservation(actual_data, mu, omega)
            self.samples.append([mu, omega])
            self.values.append(cost)

        # Bayesian Optimization loop
        for _ in range(self.n_iter):
            # Fit the GP model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(np.array(self.samples), np.array(self.values))

            # Minimize the acquisition function
            def objective(x):
                # Acquisition function expects a single point (mu, omega)
                return -self.acquisition_function(x, gp)

            # Use the mean of the bounds for initial guess x0
            x0_mu = np.mean(self.bounds['mu'])
            x0_omega = np.mean(self.bounds['omega'])

            res = minimize(objective, x0=[x0_mu, x0_omega],
                           bounds=list(self.bounds.values()), method='L-BFGS-B')

            # Evaluate the cost function at the new point
            mu, omega = res.x
            # Ensure mu and omega are within bounds after optimization step
            mu = np.clip(mu, self.bounds['mu'][0], self.bounds['mu'][1])
            omega = np.clip(omega, self.bounds['omega'][0], self.bounds['omega'][1])

            cost = self.hopf_oscillator.cost_function_with_amplitude_preservation(actual_data, mu, omega)
            self.samples.append([mu, omega])
            self.values.append(cost)

        # Return the best parameters after all iterations
        best_index = np.argmin(self.values)
        best_params = {'mu': self.samples[best_index][0], 'omega': self.samples[best_index][1]}
        return best_params

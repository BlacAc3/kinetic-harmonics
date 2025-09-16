import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import random
import warnings

class HopfOscillator:
    """
    A class to implement the Hopf oscillator for smoothing human movement data.
    """

    def __init__(self, mu=1.0, omega=1.0, dt=0.03333, data_scale=1.0):
        """
        Initialize the Hopf oscillator.

        Parameters:
        - mu: The amplitude parameter (default: 1.0).
        - omega: The angular frequency (default: 1.0).
        - dt: The time step for integration (default: 0.03333 seconds).
        """
        self.mu = mu
        self.omega = omega
        self.dt = dt
        self.data_scale = data_scale

    def _hopf_calculate(self, x, y, data_input, mu, omega):
        """
        Calculate the derivatives and update the oscillator states.

        Parameters:
        - x: Current state of x.
        - y: Current state of y.
        - data_input: Input data at the current time step.
        - mu: The amplitude parameter of the Hopf oscillator.
        - omega: The angular frequency of the Hopf oscillator.

        Returns:
        - dx: The derivative of x.
        - dy: The derivative of y.
        - new_x: The updated state of x.
        - new_y: The updated state of y.
        """
        dx = mu * (mu - (x**2 + y**2)) * x - omega * y + self.data_scale * data_input
        dy = mu * (mu - (x**2 + y**2)) * y + omega * x

        # Update the states using Euler integration
        new_x = x + dx * self.dt
        new_y = y + dy * self.dt

        return dx, dy, new_x, new_y


    def smooth(self, data):
        """
        Apply the Hopf oscillator to smooth the input data by first optimizing the parameters
        using Bayesian Optimization and then applying the oscillator with the optimized parameters.

        Parameters:
        - data: A 2D numpy array where each column represents a time series to be smoothed. It's expected to be pre-scaled.

        Returns:
        - smoothed_data: A 2D numpy array of the same shape as `data`, containing the smoothed time series.
        """
        # Store original data scale and offset
        mean = np.mean(data)
        amp = np.max(data) - np.min(data)

        # Define bounds for Bayesian Optimization
        bounds = {'mu': (0.1, 2.0), 'omega': (0.1, 2.0)}

        # Initialize Bayesian Optimizer
        optimizer = BayesianOptimizer(self, bounds)

        # Optimize parameters
        best_params = optimizer.optimize(data)

        if not best_params:
            raise ValueError("Optimization failed, could not find optimal parameters")

        # Update Hopf oscillator parameters with optimized values
        self.mu = best_params['mu']
        self.omega = best_params['omega']

        # Apply the Hopf oscillator with optimized parameters
        num_samples, num_series = data.shape
        smoothed_data = np.zeros_like(data)

        # Initialize oscillator states
        x = np.zeros(num_series)
        y = np.zeros(num_series)

        for t in range(num_samples):
            # Calculate the derivatives and update states
            dx, dy, x, y = self._hopf_calculate(x, y, data[t], self.mu, self.omega)

            # Store the smoothed data
            smoothed_data[t] = x

        smoothed_data = smoothed_data * (amp + 1e-9) + mean
        return smoothed_data

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
        # Direct calculation of predicted data within the cost function
        num_samples, num_series = actual_data.shape
        predicted_data = np.zeros_like(actual_data)

        # Initialize oscillator states
        x = np.zeros(num_series)
        y = np.zeros(num_series)

        for t in range(num_samples):
            # Calculate the derivatives and update states
            dx, dy, x, y = self._hopf_calculate(x, y, actual_data[t], mu, omega)

            # Store the predicted data
            predicted_data[t] = x

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
        # Direct calculation of predicted data within the cost function
        num_samples, num_series = actual_data.shape
        predicted_data = np.zeros_like(actual_data)

        # Initialize oscillator states
        x = np.zeros(num_series)
        y = np.zeros(num_series)

        for t in range(num_samples):
            # Calculate the derivatives and update states
            dx, dy, x, y = self._hopf_calculate(x, y, actual_data[t], mu, omega)

            # Store the predicted data
            predicted_data[t] = x

        # Calculate MSE
        mse = np.mean((predicted_data - actual_data)**2)

        # Calculate amplitude error
        actual_amplitude = actual_data.max(axis=0) - actual_data.min(axis=0)
        predicted_amplitude = predicted_data.max(axis=0) - predicted_data.min(axis=0)
        amplitude_error = np.mean((predicted_amplitude - actual_amplitude)**2)

        # Combine MSE and amplitude error
        cost = mse + alpha * amplitude_error
        return cost


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
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

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

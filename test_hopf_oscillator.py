import unittest
import numpy as np
from hopf_oscillator import HopfOscillator

class TestHopfOscillator(unittest.TestCase):

    def test_single_joint(self):
        # Generate some dummy data for a single joint
        dt = 0.01
        t = np.arange(0, 10, dt)
        data = np.sin(2 * np.pi * t)  # Example sinusoidal data

        # Initialize the Hopf oscillator
        oscillator = HopfOscillator(dt=dt)

        # Run the oscillator
        predictions = oscillator.run(data)

        # Assert that the predictions have the same shape as the input data
        self.assertEqual(predictions.shape, data.shape)

    def test_multi_joint(self):
        # Generate some dummy data for multiple joints
        dt = 0.01
        t = np.arange(0, 10, dt)
        num_joints = 3
        data = np.zeros((len(t), num_joints))
        for i in range(num_joints):
            data[:, i] = np.sin(2 * np.pi * t + i * 0.5)  # Example sinusoidal data with phase shifts

        # Initialize the Hopf oscillator with coupling
        coupling_strength = np.zeros((num_joints, num_joints))
        coupling_strength[0, 1] = 0.1  # Couple joint 0 to joint 1
        coupling_strength[1, 2] = 0.2  # Couple joint 1 to joint 2
        coupling_strength[2, 0] = 0.3  # Couple joint 2 to joint 0
        oscillator = HopfOscillator(dt=dt, coupling_strength=coupling_strength)

        # Run the oscillator
        predictions = oscillator.run(data)

        # Assert that the predictions have the same shape as the input data
        self.assertEqual(predictions.shape, data.shape)

if __name__ == '__main__':
    unittest.main()

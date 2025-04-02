# Import required libraries
import unittest
import numpy as np
from scipy.optimize import curve_fit
import sys
import os

# Add parent directory to sys.path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

# Define test case for regression parameter recovery
class TestImpactRegression(unittest.TestCase):

    def test_recover_eta_beta_from_synthetic_data(self):
        """
        Generate synthetic data from a known impact model,
        then recover eta and beta using nonlinear regression
        and check that estimates are close to true values.
        """

        # --- 1. Define ground-truth parameters (from real estimates) ---
        eta_true = 2.071382
        beta_true = 0.657832

        # Optional: fix random seed for reproducibility
        # np.random.seed(42)

        # --- 2. Generate synthetic market impact data ---
        n_points = 1000

        # Simulate volatility (sigma) as positive normal values
        sigma = np.abs(np.random.normal(loc=1.0, scale=0.2, size=n_points))

        # Simulate trade size fractions (q) between 0.01 and 1.0
        q = np.abs(np.random.uniform(low=0.01, high=1.0, size=n_points))

        # Add Gaussian noise to simulate observation errors
        noise = np.random.normal(loc=0.0, scale=0.01, size=n_points)

        # Compute synthetic h values using the true model
        h = eta_true * sigma * (q ** beta_true) + noise

        # --- 3. Define the regression model to fit ---
        def impact_model(q, eta, beta):
            return eta * sigma * (q ** beta)

        # --- 4. Perform nonlinear curve fitting ---
        initial_guess = [1.0, 0.5]  # Start near realistic values
        params, _ = curve_fit(impact_model, q, h, p0=initial_guess, maxfev=100000)

        eta_hat, beta_hat = params

        # --- 5. Validate recovery: estimated params ≈ true params ---
        # Allow small tolerance for numerical error and noise
        self.assertAlmostEqual(eta_hat, eta_true, delta=0.1, msg="Estimated eta is not close to true eta")
        self.assertAlmostEqual(beta_hat, beta_true, delta=0.05, msg="Estimated beta is not close to true beta")

        # Log recovered values
        print(f"Recovered eta = {eta_hat:.6f} (True: {eta_true})")
        print(f"Recovered beta = {beta_hat:.6f} (True: {beta_true})")

# Run test suite when script is executed
if __name__ == '__main__':
    unittest.main()
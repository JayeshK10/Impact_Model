import unittest
import numpy as np
import os
import sys
from scipy.optimize import curve_fit

# Add project root to system path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

# Import residual bootstrap function and directory utility
from Project.src.bootstrap import residual_bootstrap_analysis
from taq.MyDirectories import MyDirectories

import unittest
import numpy as np
import os
from scipy.optimize import curve_fit


# Unit test for residual bootstrap analysis
class TestResidualBootstrap(unittest.TestCase):

    def test_residual_bootstrap_analysis_outputs(self):
        """
        Generate synthetic data from a known η, β impact model,
        fit the curve to estimate them, then apply residual bootstrapping.
        Checks correctness of the returned statistical summary.
        """

        # --- Step 1: Generate synthetic dataset ---
        np.random.seed(42)
        n = 300
        eta_true = 2.071382
        beta_true = 0.657832

        # Simulate independent features and noise
        sigma = np.abs(np.random.normal(loc=1.0, scale=0.2, size=n))
        q = np.random.uniform(low=0.01, high=1.0, size=n)
        noise = np.random.normal(loc=0.0, scale=0.01, size=n)

        # Generate dependent variable (market impact)
        h = eta_true * sigma * (q ** beta_true) + noise

        # --- Step 2: Estimate model parameters using nonlinear least squares ---
        def model(q, eta, beta): 
            return eta * sigma * (q ** beta)

        (eta_hat, beta_hat), _ = curve_fit(model, q, h, p0=[1.0, 0.5], maxfev=100000)

        # --- Step 3: Run residual bootstrap analysis ---
        # Use temporary plot directory for test output
        plot_dir = os.path.join(MyDirectories.getDataDir(), "test")

        summary = residual_bootstrap_analysis(
            h_flat=h,
            sigma_flat=sigma,
            q_flat=q,
            eta_hat=eta_hat,
            beta_hat=beta_hat,
            m=100,  # Fewer resamples for fast test
            plot_dir=plot_dir
        )

        # --- Step 4: Validate output dictionary structure ---
        expected_keys = {
            "eta_hat", "beta_hat",      # Point estimates
            "eta_se", "beta_se",        # Standard errors
            "t_eta", "t_beta",          # t-stats
            "white_stat", "white_pvalue"  # Heteroskedasticity test stats
        }
        self.assertTrue(set(summary.keys()).issuperset(expected_keys))

        # --- Step 5: Check returned values are finite and positive ---
        self.assertTrue(np.isfinite(summary["eta_se"]))
        self.assertTrue(np.isfinite(summary["beta_se"]))
        self.assertGreater(summary["eta_se"], 0)
        self.assertGreater(summary["beta_se"], 0)

        # Optional print for debug
        print("Bootstrap summary:", summary)

# Run this test case if script is executed directly
if __name__ == '__main__':
    unittest.main()
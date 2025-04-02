import unittest
import numpy as np
import os
import sys
from scipy.optimize import curve_fit

# Add project root to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

# Import the function under test and supporting modules
from Project.src.bootstrap import paired_bootstrap_analysis
from taq.MyDirectories import MyDirectories

# Define unit test for bootstrap-based statistical summary
class TestPairedBootstrapAnalysis(unittest.TestCase):

    # Main test case: ensure output keys, shape, and value types from bootstrap
    def test_paired_bootstrap_outputs(self):
        # --- Step 1: Generate synthetic data using known impact model ---
        np.random.seed(123)  # For reproducibility
        n = 300

        eta_true = 2.0
        beta_true = 0.65

        # Create volatility (σ), trade size fraction (q), and synthetic h values
        sigma = np.abs(np.random.normal(loc=1.0, scale=0.2, size=n))
        q = np.random.uniform(low=0.01, high=1.0, size=n)
        noise = np.random.normal(loc=0.0, scale=0.01, size=n)
        h = eta_true * sigma * (q ** beta_true) + noise

        # --- Step 2: Fit the nonlinear model once to get baseline estimates ---
        def model(q, eta, beta):
            return eta * sigma * (q ** beta)

        (eta_hat, beta_hat), _ = curve_fit(model, q, h, p0=[1.0, 0.5], maxfev=100000)

        # --- Step 3: Run paired bootstrap analysis ---
        # Create a temporary directory for plot output
        plot_dir = os.path.join(MyDirectories.getDataDir(), "test")

        # Run the bootstrap with smaller m for faster test
        summary = paired_bootstrap_analysis(
            h_flat=h,
            sigma_flat=sigma,
            q_flat=q,
            eta_hat=eta_hat,
            beta_hat=beta_hat,
            m=100,  # Small sample count for speed
            plot_dir=plot_dir
        )

        # --- Step 4: Validate returned dictionary structure ---
        expected_keys = {
            "eta_hat", "beta_hat",     # Point estimates
            "eta_se", "beta_se",       # Standard errors
            "t_eta", "t_beta",         # t-stats
            "white_stat", "white_pvalue"  # White test for heteroskedasticity
        }
        self.assertTrue(set(summary.keys()).issuperset(expected_keys))

        # --- Step 5: Validate numeric value sanity ---
        self.assertTrue(np.isfinite(summary["eta_se"]) and summary["eta_se"] > 0)
        self.assertTrue(np.isfinite(summary["beta_se"]) and summary["beta_se"] > 0)
        self.assertTrue(np.isfinite(summary["t_eta"]))
        self.assertTrue(np.isfinite(summary["t_beta"]))

        # Optional debug output
        print("Summary:", summary)

# Entry point for running the test
if __name__ == "__main__":
    unittest.main()
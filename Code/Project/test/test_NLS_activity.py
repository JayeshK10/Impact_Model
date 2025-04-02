import unittest
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

from Project.src.NLS_activity import estimate_impact_params_from_df

class TestGroupwiseImpactEstimation(unittest.TestCase):

    def generate_impact_data_with_groupwise_params(
        self,
        n_points=1000,
        eta_high=3.0,
        beta_high=0.9,
        eta_low=1.0,
        beta_low=0.5
    ):
        """
        Generate synthetic impact data with different η and β for high/low activity groups.

        Returns:
            impact_df (combined), high_df (used to generate h_high), low_df (used to generate h_low)
        """
        np.random.seed(42)

        # Common features
        sigma = np.abs(np.random.normal(loc=1.0, scale=0.2, size=n_points))
        avg_daily_value = np.random.uniform(low=1e5, high=1e7, size=n_points)
        q = np.random.uniform(low=1e3, high=1e6, size=n_points) / avg_daily_value
        noise = np.random.normal(loc=0.0, scale=0.01, size=n_points)

        # Base DataFrame (without h yet)
        impact_df = pd.DataFrame({
            "sigma": sigma,
            "q": q,
            "avg_daily_value": avg_daily_value,
            "stock": ["SYNTH"] * n_points,
            "date": pd.date_range("2023-01-01", periods=n_points, freq="D")
        })

        # Median split by ADV
        median_adv = impact_df['avg_daily_value'].median()
        high_df = impact_df[impact_df['avg_daily_value'] > median_adv].copy()
        low_df = impact_df[impact_df['avg_daily_value'] <= median_adv].copy()

        # Add group-specific h
        h_high = eta_high * high_df["sigma"] * (high_df["q"] ** beta_high) + noise[:len(high_df)]
        h_low = eta_low * low_df["sigma"] * (low_df["q"] ** beta_low) + noise[len(high_df):]

        high_df["h"] = h_high
        low_df["h"] = h_low

        return impact_df, high_df, low_df



    def test_groupwise_eta_beta_estimates(self):
        # Ground truth parameters
        eta_high_true = 14.992181
        beta_high_true = 1.220973
        eta_low_true = 0.486902
        beta_low_true = 0.436532


        # Generate data with different params per group
        impact_df, high_df, low_df = (
            self.generate_impact_data_with_groupwise_params(
                n_points=1000,
                eta_high=eta_high_true,
                beta_high=beta_high_true,
                eta_low=eta_low_true,
                beta_low=beta_low_true
            )
        )

        # Estimate from each group
        eta_high_est, beta_high_est = estimate_impact_params_from_df(high_df)
        eta_low_est, beta_low_est = estimate_impact_params_from_df(low_df)

        # Validate split
        median_adv = impact_df['avg_daily_value'].median()
        self.assertTrue((high_df['avg_daily_value'] > median_adv).all(), "High group contains low ADV entries")
        self.assertTrue((low_df['avg_daily_value'] <= median_adv).all(), "Low group contains high ADV entries")


        # Check each group's estimates are close to their respective true values
        self.assertAlmostEqual(eta_high_est, eta_high_true, delta=0.3, msg="High η off from truth")
        self.assertAlmostEqual(beta_high_est, beta_high_true, delta=0.05, msg="High β off from truth")

        self.assertAlmostEqual(eta_low_est, eta_low_true, delta=0.3, msg="Low η off from truth")
        self.assertAlmostEqual(beta_low_est, beta_low_true, delta=0.05, msg="Low β off from truth")

        # Check directional inequality
        self.assertGreater(beta_high_est, beta_low_est, "Expected β_high > β_low")

        print(f"[HIGH] η = {eta_high_est:.4f} (True: {eta_high_true}), β = {beta_high_est:.4f} (True: {beta_high_true})")
        print(f"[LOW ] η = {eta_low_est:.4f} (True: {eta_low_true}), β = {beta_low_est:.4f} (True: {beta_low_true})")

if __name__ == "__main__":
    unittest.main()
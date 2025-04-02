import unittest
from unittest.mock import MagicMock
import os
import sys
import numpy as np

# Add the root project directory to sys.path to import modules
# Enables importing ImpactMetrics from the source directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the class under test
# ImpactMetrics contains the compute_metrics method being validated
from Project.src.ImpactMetrics import ImpactMetrics

# Define a test case for compute_metrics in ImpactMetrics
# Inherits from unittest.TestCase for setup, teardown, and assertions
class TestImpactMetricsCompute(unittest.TestCase):

    # Test full metric computation with all required values mocked
    def test_compute_metrics_complete(self):
        # Create a dummy ImpactMetrics instance
        impact = ImpactMetrics("20070620", "AAPL")

        # Define mock 2-min returns (to calculate volatility)
        returns = [0.01, -0.005, 0.002, 0.0]

        # Mock the method to return bucketed returns
        impact.compute_2min_midquote_returns = MagicMock(return_value={
            i: val for i, val in enumerate(returns)
        })

        # Mock total daily volume
        impact.compute_total_daily_volume = MagicMock(return_value=1_000_000)

        # Mock arrival and terminal prices
        impact.compute_arrival_price = MagicMock(return_value=150.0)
        impact.compute_terminal_price = MagicMock(return_value=151.0)

        # Mock trade imbalance in shares
        impact.compute_imbalance = MagicMock(return_value=5000)

        # Mock VWAP values for 9:30–3:30 and 9:30–4:00
        impact.compute_VWAP = MagicMock(side_effect=[148.5, 149.0])  # VWAP_9_3, VWAP_9_4

        # Execute the method under test
        result, flag = impact.compute_metrics()

        # Ensure no value is missing
        self.assertFalse(flag)

        # Validate average daily traded notional value
        expected_avg_value = 1_000_000 * 149.0
        self.assertAlmostEqual(result['avg_daily_value'], expected_avg_value)

        # Validate volatility computation
        expected_vol = np.std(returns, ddof=1) * np.sqrt(195)
        self.assertAlmostEqual(result['sigma'], expected_vol)

        # Validate temporary impact h = (VWAP_9_3 - arrival_price) - g
        g = (151.0 - 150.0) / 2  # Permanent impact
        h = (148.5 - 150.0) - g  # = -2.0
        self.assertAlmostEqual(result['h'], h)

        # Validate relative trade size q = imbalance_value / scaled ADV
        imbalance_value = 5000 * 149.0
        scaled_adv = expected_avg_value * (6 / 6.5)
        expected_q = imbalance_value / scaled_adv
        self.assertAlmostEqual(result['q'], expected_q)

    # Test behavior when one or more values are missing
    def test_compute_metrics_with_missing_value(self):
        # Create dummy ImpactMetrics instance
        impact = ImpactMetrics("20070620", "AAPL")

        # Mock empty returns (volatility will be None)
        impact.compute_2min_midquote_returns = MagicMock(return_value={})

        # Mock total volume
        impact.compute_total_daily_volume = MagicMock(return_value=1_000_000)

        # Mock arrival price as None to simulate missing data
        impact.compute_arrival_price = MagicMock(return_value=None)
        impact.compute_terminal_price = MagicMock(return_value=151.0)

        # Mock imbalance and VWAPs
        impact.compute_imbalance = MagicMock(return_value=5000)
        impact.compute_VWAP = MagicMock(side_effect=[148.5, 149.0])

        # Run metric computation
        result, flag = impact.compute_metrics()

        # Ensure the missing flag is True due to arrival_price = None
        self.assertTrue(flag)

        # Ensure at least one computed value is missing
        self.assertTrue(any(v is None for v in result.values()))

# Run test cases if script is executed directly
# This allows local testing without needing a test runner
if __name__ == "__main__":
    unittest.main()
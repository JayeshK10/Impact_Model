# Import unittest framework and mocking tools
import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys

# Add parent directory to sys.path for import resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

# Import function under test and checkpoint directory
from Project.src.data_preprocess import stock_metrics_func, CHECKPOINT_DIR

# Test class to validate stock_metrics_func logic and side effects
class TestComputeStockMetrics(unittest.TestCase):

    # Setup shared test inputs
    def setUp(self):
        # Test stock and dates
        self.stock_code = "AAPL"
        self.dates = ["20070620", "20070720"]

        # Format as expected input to stock_metrics_func
        self.stock_date_tuple = (self.stock_code, self.dates)

        # Expected checkpoint path
        self.checkpoint_path = os.path.join(CHECKPOINT_DIR, f"stock_{self.stock_code}.pkl")

    # Test early return when checkpoint file already exists
    @patch("data_preprocess.os.path.exists")
    def test_skip_if_checkpoint_exists(self, mock_exists):
        """Test that function returns None if checkpoint already exists."""

        # Simulate checkpoint file already being present
        mock_exists.return_value = True

        # Call the function
        result = stock_metrics_func(self.stock_date_tuple)

        # Expect early exit without recomputation
        self.assertIsNone(result)

    # Test full metric computation path with mocked dependencies
    @patch("Project.src.data_preprocess.pickle.dump")
    @patch("Project.src.data_preprocess.open", new_callable=mock_open)
    @patch("Project.src.data_preprocess.ImpactMetrics")
    @patch("Project.src.data_preprocess.os.path.exists")
    def test_compute_metrics_and_save_checkpoint(self, mock_exists, mock_IM, mock_file, mock_pickle_dump):
        """Test normal execution: computes metrics, flags, and writes checkpoint."""

        # Simulate checkpoint does not exist
        mock_exists.return_value = False

        # Mock ImpactMetrics instance and its compute_metrics method
        mock_instance = MagicMock()
        mock_instance.compute_metrics.side_effect = [
            # Return two sets of mock results (for each date)
            (
                {
                    "arrival_price": 101.0,
                    "terminal_price": 120.0,
                    "VWAP": 118.5,
                    "volatility": 0.08711,
                    "imbalance_value": 5000 * 149.0,
                    "avg_daily_value": 1000000 * 149.0
                }, 
                False
            ),
            (
                {
                    "arrival_price": 201.0,
                    "terminal_price": 210.0,
                    "VWAP": 204.5,
                    "volatility": None,
                    "imbalance_value": 200 * 205.0,
                    "avg_daily_value": 10000 * 205.0
                }, 
                True
            ),
        ]
        # Use mocked instance as return value of ImpactMetrics
        mock_IM.return_value = mock_instance

        # Run the function (should trigger metric computation and checkpoint writing)
        result = stock_metrics_func(self.stock_date_tuple)

        # Verify file was opened for writing
        mock_file.assert_called_once_with(self.checkpoint_path, "wb")

        # Verify data was dumped via pickle
        mock_pickle_dump.assert_called_once()

        # Check saved object structure: (metrics_dict, flags_dict)
        saved_data = mock_pickle_dump.call_args[0][0]
        self.assertIsInstance(saved_data, tuple)

        # Check that metrics and flags contain keys for both dates
        self.assertEqual(set(saved_data[0].keys()), set(self.dates))  # metrics_dict keys
        self.assertEqual(set(saved_data[1].keys()), set(self.dates))  # flags_dict keys

# Run the tests if this file is executed directly
if __name__ == "__main__":
    unittest.main()
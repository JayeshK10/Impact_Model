# Import test framework and mocking utilities
import unittest
from unittest.mock import patch, mock_open, MagicMock
import sys
import os

# Add parent directory to path for importing project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the function under test
from Project.src.data_preprocess import aggregate_checkpoints

# Define test case for the checkpoint aggregation function
class TestAggregateCheckpoints(unittest.TestCase):

    # Test loading multiple checkpoints and building the DataFrames
    @patch("data_preprocess.os.listdir")             # Mock file listing in checkpoint directory
    @patch("data_preprocess.open", new_callable=mock_open)  # Mock file open
    @patch("data_preprocess.pickle.load")            # Mock pickle.load for file reading
    def test_aggregate_checkpoints(self, mock_pickle_load, mock_file_open, mock_listdir):
        # Simulate two checkpoint files being present
        mock_listdir.return_value = ["stock_AAPL.pkl", "stock_GOOG.pkl"]

        # Simulate what pickle.load would return for each file
        mock_pickle_load.side_effect = [
            (
                {"20070620": {
                    "arrival_price": 101.0,
                    "terminal_price": 120.0,
                    "VWAP": 118.5,
                    "volatility": 0.08711,
                    "imbalance_value": 5000 * 149.0,
                    "avg_daily_value": 1000000 * 149.0
                }},
                {"20070620": False}
            ),
            (
                {"20070620": {
                    "arrival_price": 201.0,
                    "terminal_price": 210.0,
                    "VWAP": 204.5,
                    "volatility": None,
                    "imbalance_value": 200 * 205.0,
                    "avg_daily_value": 10000 * 205.0
                }},
                {"20070620": True}
            )
        ]

        # Call the function under test
        metrics_df, flags_df = aggregate_checkpoints()

        # --- Assertions ---

        # Check that both stocks are included
        self.assertEqual(set(metrics_df.index), {"AAPL", "GOOG"})

        # Check that the date column is correct
        self.assertEqual(set(metrics_df.columns), {"20070620"})

        # Validate metric values for AAPL
        self.assertEqual(metrics_df.loc["AAPL", "20070620"]["arrival_price"], 101.0)

        # Validate flag for GOOG
        self.assertTrue(flags_df.loc["GOOG", "20070620"])

# Run the test suite if script is executed directly
if __name__ == "__main__":
    unittest.main()
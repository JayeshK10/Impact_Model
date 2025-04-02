import unittest
import tempfile
import os
import sys
import pickle
from unittest.mock import patch, MagicMock
import pandas as pd

# Add parent directory to sys.path for import resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

# Import the module under test
from Project.src import data_preprocess

# Define test case for the `main` function

class TestMainFunction(unittest.TestCase):

    # Patch key dependencies used inside main: metric computation and checkpoint aggregation
    @patch("Project.src.data_preprocess.stock_metrics_func")
    @patch("Project.src.data_preprocess.aggregate_checkpoints")
    def test_main_creates_output_files(self, mock_aggregate, mock_compute):
        """
        Test that main() generates the expected pickle output files 
        and includes correct keys/structures in the saved objects.
        """

        # Prepare dummy input data: metric and flag dictionaries
        dummy_metrics_df = {
            "AAPL": {
                "20200101": {
                    "arrival_price": 101.0,
                    "terminal_price": 105.0,
                    "VWAP": 103.5,
                    "volatility": 0.02,
                    "imbalance_value": 100000.0,
                    "avg_daily_value": 1500000.0
                }
            }
        }
        dummy_flags_df = {
            "AAPL": {
                "20200101": False
            }
        }

        # Convert to DataFrames for return value of aggregate_checkpoints()
        metrics_df = pd.DataFrame.from_dict(dummy_metrics_df, orient="index")
        flags_df = pd.DataFrame.from_dict(dummy_flags_df, orient="index")
        mock_aggregate.return_value = (metrics_df, flags_df)

        # Use a temporary directory to intercept saved files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Redirect output path to temporary folder
            data_preprocess.FINAL_DATA_DIR = tmpdir

            # Define dummy stock/date inputs
            list_of_stocks = ["AAPL"]
            list_of_dates = ["20200101"]

            # Run main (serial mode for test isolation)
            data_preprocess.main(list_of_stocks, list_of_dates, use_multiprocessing=False)

            # Define expected output file paths
            data_path = os.path.join(tmpdir, "input_matrix.pkl")
            info_path = os.path.join(tmpdir, "info.pkl")

            # --- Assertions ---

            # Confirm both output files were created
            self.assertTrue(os.path.exists(data_path))
            self.assertTrue(os.path.exists(info_path))

            # Validate structure and keys of saved info.pkl
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                self.assertIn("kept_dates", info)
                self.assertIn("kept_stocks", info)

            # Validate structure and contents of saved input_matrix.pkl
            with open(data_path, "rb") as f:
                matrices = pickle.load(f)
                self.assertIn("arrival_price", matrices)
                self.assertIsInstance(matrices["arrival_price"], pd.DataFrame)



# Run the test suite when this file is executed directly
if __name__ == "__main__":
    unittest.main()
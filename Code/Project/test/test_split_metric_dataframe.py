import unittest
import pandas as pd
import os
import sys

# Extend sys.path to import source modules from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

# Import the function under test
from Project.src.data_preprocess import split_metric_dataframe

# Import unittest for test case structure
import unittest
import pandas as pd

# Test case for validating split_metric_dataframe behavior
class TestSplitMetricDataFrame(unittest.TestCase):

    # Test whether the function splits dictionary-based DataFrame correctly
    def test_split_metric_dataframe(self):
        # Define nested data: each cell is a dict with multiple metrics
        data = {
            "20070620": {
                "AAPL": {"arrival_price": 101.0, "VWAP": 100.5},
                "MSFT": {"arrival_price": 98.0, "VWAP": 97.8}
            },
            "20070720": {
                "AAPL": {"arrival_price": 103.2, "VWAP": 102.1},
                "MSFT": {"arrival_price": 99.1, "VWAP": 98.6}
            }
        }

        # Convert nested dict into a stock x date DataFrame
        # Columns: dates, Rows: stocks, Values: metric dicts
        metrics_df = pd.DataFrame(data).T.transpose()

        # Define metric keys to extract from each dict
        keys = ["arrival_price", "VWAP"]

        # Apply the function under test
        split_dfs = split_metric_dataframe(metrics_df, keys)

        # --- Assertions ---

        # Check that expected metric keys are present in the result
        self.assertIn("arrival_price", split_dfs)
        self.assertIn("VWAP", split_dfs)

        # Verify extracted value for AAPL on 20070620 (arrival price)
        self.assertEqual(split_dfs["arrival_price"].loc["AAPL", "20070620"], 101.0)

        # Verify extracted VWAP for MSFT on 20070720
        self.assertEqual(split_dfs["VWAP"].loc["MSFT", "20070720"], 98.6)

# Run the test suite if this script is executed directly
if __name__ == "__main__":
    unittest.main()
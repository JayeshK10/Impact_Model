import unittest
import pandas as pd
import sys
import os

# Extend path for importing project code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

# Import the function to be tested
from Project.src.data_preprocess import filter_top_market_cap_stocks

# Define test class for filtering top market cap stocks
class TestFilterTopMarketCapStocks(unittest.TestCase):

    # Setup dummy data for tests
    def setUp(self):
        # Create 10 dummy stock tickers
        stocks = [f"STOCK{i}" for i in range(10)]

        # Use two dummy dates
        dates = ["20200101", "20200102"]

        # Build a nested dict: each cell contains metric values including avg_daily_value
        metric_data = {}
        for stock_index, stock in enumerate(stocks):
            row = {}
            for date in dates:
                row[date] = {
                    "arrival_price": 100.0,
                    "terminal_price": 110.0,
                    "VWAP": 105.0,
                    "volatility": 0.02,
                    "imbalance_value": 1000.0,
                    "avg_daily_value": 1000 * (stock_index + 1)  # increasing value
                }
            metric_data[stock] = row

        # Convert to stock x date DataFrame
        self.metrics_df = pd.DataFrame(metric_data).T

        # Create corresponding flags DataFrame: all values set to False (no missingness)
        self.flags_df = pd.DataFrame(False, index=stocks, columns=dates)

    # Test whether the top 5 stocks by avg_daily_value are correctly selected
    def test_top_5_market_cap_stocks(self):
        # Call the function with top_k=5
        filtered_metrics_df, filtered_flags_df, top_stocks = filter_top_market_cap_stocks(
            self.metrics_df,
            self.flags_df,
            top_k=5,
            save_dir=None  # Skip saving to file
        )

        # We expect the top 5 to be STOCK9 to STOCK5 (highest avg_daily_value)
        expected_top_stocks = [f"STOCK{i}" for i in range(9, 4, -1)]

        # Validate that the returned top stock list is correct
        self.assertEqual(top_stocks, expected_top_stocks)

        # Ensure both returned DataFrames contain only the top 5 stocks
        self.assertEqual(set(filtered_metrics_df.index), set(expected_top_stocks))
        self.assertEqual(set(filtered_flags_df.index), set(expected_top_stocks))

        # Columns (dates) should remain unchanged
        self.assertEqual(set(filtered_metrics_df.columns), set(self.metrics_df.columns))
        self.assertEqual(set(filtered_flags_df.columns), set(self.flags_df.columns))

        # Validate DataFrame dimensions: 5 stocks × 2 dates
        self.assertEqual(filtered_metrics_df.shape[0], 5)
        self.assertEqual(filtered_flags_df.shape[0], 5)

# Run test suite if file is executed directly
if __name__ == "__main__":
    unittest.main()
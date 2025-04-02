import unittest
import pandas as pd
import sys
import os


# Extend system path for import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

# Import the function to be tested
from Project.src.data_preprocess import filter_metrics 

# Test case for the filter_metrics function
class TestFilterMetrics(unittest.TestCase):
    def setUp(self):
        # Create a DataFrame with 5 stocks and 3 dates
        # Mock metrics_df with dicts
        self.metrics_df = pd.DataFrame({
            "20200101": {
                "AAPL": {"arrival_price": 101, "terminal_price": 110, "VWAP": 105, "volatility": 0.01, "imbalance_value": 100, "avg_daily_value": 1000},
                "MSFT": {"arrival_price": 102, "terminal_price": 111, "VWAP": 106, "volatility": 0.02, "imbalance_value": 200, "avg_daily_value": 2000},
                "GOOG": {"arrival_price": 103, "terminal_price": 112, "VWAP": 107, "volatility": 0.03, "imbalance_value": 300, "avg_daily_value": 3000},
                "META": {"arrival_price": 104, "terminal_price": 113, "VWAP": 108, "volatility": 0.04, "imbalance_value": 400, "avg_daily_value": 4000},
                "TSLA": {"arrival_price": None, "terminal_price": None, "VWAP": None, "volatility": None, "imbalance_value": None, "avg_daily_value": None},  # All missing
            },
            "20200102": {
                "AAPL": {"arrival_price": 101, "terminal_price": 110, "VWAP": 105, "volatility": 0.01, "imbalance_value": 100, "avg_daily_value": 1000},
                "MSFT": {"arrival_price": 102, "terminal_price": 111, "VWAP": 106, "volatility": 0.02, "imbalance_value": 200, "avg_daily_value": 2000},
                "GOOG": {"arrival_price": 103, "terminal_price": 112, "VWAP": 107, "volatility": 0.03, "imbalance_value": 300, "avg_daily_value": 3000},
                "META": {"arrival_price": 104, "terminal_price": 113, "VWAP": 108, "volatility": 0.04, "imbalance_value": 400, "avg_daily_value": 4000},
                "TSLA": {"arrival_price": None, "terminal_price": None, "VWAP": None, "volatility": None, "imbalance_value": None, "avg_daily_value": None},
            },
            "20200103": {
                "AAPL": {"arrival_price": 108, "terminal_price": 110, "VWAP": 105, "volatility": 0.01, "imbalance_value": 100, "avg_daily_value": 1000},
                "MSFT": {"arrival_price": 102, "terminal_price": 111, "VWAP": 106, "volatility": 0.02, "imbalance_value": 200, "avg_daily_value": 2000},
                "GOOG": {"arrival_price": 103, "terminal_price": 112, "VWAP": 107, "volatility": 0.03, "imbalance_value": 300, "avg_daily_value": 3000},
                "META": {"arrival_price": 104, "terminal_price": 113, "VWAP": 108, "volatility": 0.04, "imbalance_value": 400, "avg_daily_value": 4000},
                "TSLA": {"arrival_price": None, "terminal_price": None, "VWAP": None, "volatility": None, "imbalance_value": None, "avg_daily_value": None},
            },
            "20200104": {
                "AAPL": {"arrival_price": None, "terminal_price": None, "VWAP": None, "volatility": None, "imbalance_value": None, "avg_daily_value": None},
                "MSFT": {"arrival_price": 102, "terminal_price": 111, "VWAP": 106, "volatility": 0.02, "imbalance_value": 200, "avg_daily_value": 2000},
                "GOOG": {"arrival_price": None, "terminal_price": None, "VWAP": None, "volatility": None, "imbalance_value": None, "avg_daily_value": None},
                "META": {"arrival_price": 104, "terminal_price": 113, "VWAP": 108, "volatility": 0.04, "imbalance_value": 400, "avg_daily_value": 4000},
                "TSLA": {"arrival_price": None, "terminal_price": None, "VWAP": None, "volatility": None, "imbalance_value": None, "avg_daily_value": None},
            },
            "20200105": {
                "AAPL": {"arrival_price": None, "terminal_price": None, "VWAP": None, "volatility": None, "imbalance_value": None, "avg_daily_value": None},
                "MSFT": {"arrival_price": None, "terminal_price": None, "VWAP": None, "volatility": None, "imbalance_value": None, "avg_daily_value": None},
                "GOOG": {"arrival_price": None, "terminal_price": None, "VWAP": None, "volatility": None, "imbalance_value": None, "avg_daily_value": None},
                "META": {"arrival_price": None, "terminal_price": None, "VWAP": None, "volatility": None, "imbalance_value": None, "avg_daily_value": None},
                "TSLA": {"arrival_price": None, "terminal_price": None, "VWAP": None, "volatility": None, "imbalance_value": None, "avg_daily_value": None},
            },
        })

        # Corresponding flags (True = missing)
        # Mock flags_df (True = missing, False = present)
        self.flags_df = pd.DataFrame({
            "20200101": {
                "AAPL": False, "MSFT": False, "GOOG": False, "META": False, "TSLA": True
            },
            "20200102": {
                "AAPL": False, "MSFT": False, "GOOG": False, "META": False, "TSLA": True
            },
            "20200103": {
                "AAPL": False, "MSFT": False, "GOOG": False, "META": False, "TSLA": True
            },
            "20200104": {
                "AAPL": True, "MSFT": False, "GOOG": True, "META": False, "TSLA": True
            },
            "20200105": {
                "AAPL": True, "MSFT": True, "GOOG": True, "META": True, "TSLA": True  # All missing
            },
        })


    def test_updated_filter_metrics(self):
        cleaned_df, dropped_dates, dropped_stocks, kept_dates, kept_stocks = filter_metrics(
            self.metrics_df,
            self.flags_df,
            stock_thresh=0.5,      # allow a few missing values in stocks initially
            date_thresh=0.4,       # allow higher threshold to test post-filter step
            max_tolerable_missing_dates=1  # only 1 missing date allowed before capping
        )

        # TSLA has 100% missing across all dates → should be dropped
        # TSLA has 100% missing across all dates → should be dropped
        self.assertIn("TSLA", dropped_stocks)

        # Dates 20200104 (3/5 = 60%) and 20200105 (5/5 = 100%) → should be dropped
        self.assertIn("20200104", dropped_dates)
        self.assertIn("20200105", dropped_dates)

        # Remaining stocks (AAPL, MSFT, GOOG, META) should be kept
        self.assertEqual(set(kept_stocks), {"AAPL", "MSFT", "GOOG", "META"})

        # Remaining dates should be 20200101, 20200102, 20200103
        self.assertEqual(set(kept_dates), {"20200101", "20200102", "20200103"})

        # Final cleaned_df shape should be 4 stocks × 3 dates
        self.assertEqual(cleaned_df.shape, (4, 3))

        # Check that cleaned_df contains correct stocks and dates
        for stock in ["AAPL", "MSFT", "GOOG", "META"]:
            self.assertIn(stock, cleaned_df.index)
        for date in ["20200101", "20200102", "20200103"]:
            self.assertIn(date, cleaned_df.columns)

        # Final assertion: cleaned_df should not contain any missing values
        self.assertFalse(cleaned_df.isna().any().any())

if __name__ == "__main__":
    unittest.main()
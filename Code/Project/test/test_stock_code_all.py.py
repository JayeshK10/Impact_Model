# Imports 
import unittest
import tempfile
import shutil
import os
import sys

# Dynamically add the root project directory to the import path
# Ensures the test can import modules from the main project codebase
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the get_code_names function from the source project
# This is the main function being tested in the unit tests
from Project.src.stock_code_all import get_code_names

# Define a test case class for testing get_code_names
# Inherits from unittest.TestCase to get setup, teardown, and assertion methods
class TestGetCodeNames(unittest.TestCase):

    # setUp is automatically run before each test case
    # It prepares the temporary directory structure and files
    def setUp(self):
        # Create a temporary base directory for testing
        # This isolates test data from the actual filesystem
        self.test_dir = tempfile.mkdtemp()

        # Define a regex pattern to match quote filenames
        # Used to extract stock codes from file names like AAPL_quotes.binRQ
        self.quotes_pattern = r"^(.*?)_quotes\.binRQ$"
        # Define a regex pattern to match trade filenames
        # Used for filenames like AAPL_trades.binRT
        self.trades_pattern = r"^(.*?)_trades\.binRT$"

        # Create the top-level quotes directory inside temp dir
        # Will contain subdirectories for each date
        self.quotes_dir = os.path.join(self.test_dir, "quotes")
        os.makedirs(self.quotes_dir)

        # Define mock quote files by date for testing
        # Each key is a date and value is a list of stock quote files
        self.quotes_data = {
            "20230101": ["AAPL_quotes.binRQ", "MSFT_quotes.binRQ", "GOOG_quotes.binRQ"],
            "20230102": ["AAPL_quotes.binRQ", "MSFT_quotes.binRQ"]
        }

        # Loop through each date and its associated file list
        # Create subdirectories and corresponding files
        for date, files in self.quotes_data.items():
            # Construct the full path for the date-specific directory
            # Create the directory inside the quotes folder
            date_dir = os.path.join(self.quotes_dir, date)
            os.makedirs(date_dir)
            for file in files:
                # Create an empty file with the given name in the date directory
                # Simulates presence of a real data file
                open(os.path.join(date_dir, file), 'a').close()

        # Create the top-level trades directory for mock trade files
        # Will mirror the structure used in quotes_dir
        self.trades_dir = os.path.join(self.test_dir, "trades")
        os.makedirs(self.trades_dir)

        # Define mock trade files for different dates
        # These will be used to test code intersection across days
        self.trades_data = {
            "20230101": ["AAPL_trades.binRT", "MSFT_trades.binRT", "GOOG_trades.binRT"],
            "20230102": ["AAPL_trades.binRT", "GOOG_trades.binRT"]
        }

        # Loop through each date and create directories and mock trade files
        # Mimics real trade data directory layout
        for date, files in self.trades_data.items():
            date_dir = os.path.join(self.trades_dir, date)
            os.makedirs(date_dir)
            for file in files:
                # Create empty file to act as trade data
                # Required for the function to detect stock code
                open(os.path.join(date_dir, file), 'a').close()

    # tearDown is called after each test case
    # Cleans up by removing the temporary test directory
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # Test extracting code names from a single quotes folder
    # Validates that all expected codes are correctly parsed
    def test_get_code_names_single_dir(self):
        test_path = os.path.join(self.quotes_dir, "20230101")
        result = get_code_names(test_path, self.quotes_pattern)
        self.assertEqual(result, {"AAPL", "MSFT", "GOOG"})

    # Test finding common code names across multiple quote days
    # Verifies correct intersection of code names across all folders
    def test_common_code_names_across_quotes(self):
        common = None
        for date in os.listdir(self.quotes_dir):
            path = os.path.join(self.quotes_dir, date)
            if not os.path.isdir(path): continue
            codes = get_code_names(path, self.quotes_pattern)
            common = codes if common is None else common.intersection(codes)
        self.assertEqual(common, {"AAPL", "MSFT"})

    # Test common stock codes across trade data folders
    # Ensures that only codes present in all trade dates are returned
    def test_common_code_names_across_trades(self):
        common = None
        for date in os.listdir(self.trades_dir):
            path = os.path.join(self.trades_dir, date)
            if not os.path.isdir(path): continue
            codes = get_code_names(path, self.trades_pattern)
            common = codes if common is None else common.intersection(codes)
        self.assertEqual(common, {"AAPL", "GOOG"})

    # Test final intersection of codes from both quotes and trades
    # Validates that only universally common codes are returned
    def test_final_common_across_quotes_and_trades(self):
        common_quotes = None
        for date in os.listdir(self.quotes_dir):
            path = os.path.join(self.quotes_dir, date)
            if not os.path.isdir(path): continue
            codes = get_code_names(path, self.quotes_pattern)
            common_quotes = codes if common_quotes is None else common_quotes.intersection(codes)

        common_trades = None
        for date in os.listdir(self.trades_dir):
            path = os.path.join(self.trades_dir, date)
            if not os.path.isdir(path): continue
            codes = get_code_names(path, self.trades_pattern)
            common_trades = codes if common_trades is None else common_trades.intersection(codes)

        common_both = common_quotes.intersection(common_trades)
        self.assertEqual(common_both, {"AAPL"})

# Run all test cases when this file is executed directly
# Useful for standalone testing via command-line or editor
if __name__ == "__main__":
    unittest.main()
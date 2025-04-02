import unittest
import sys
import os

# Add project root to system path to enable module import
# This allows importing ImpactMetrics from the main source directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the ImpactMetrics class containing the compute_terminal_price logic
# This is the core class being tested in this script
from Project.src.ImpactMetrics import ImpactMetrics

# Define a mock reader that simulates TAQQuotesReader for testing
# Allows controlled injection of synthetic quote data
class MockTAQQuotesReader:
    """
    A mock class to simulate TAQQuotesReader.
    Provides getN(), getTimestamp(i), and getPrice(i) methods.
    """

    # Initialize the mock with a list of quotes
    # Each quote should be a dictionary with timestamp and price keys
    def __init__(self, quotes):
        self.quotes = quotes

    # Return the total number of quotes
    # Used by ImpactMetrics to iterate over all data points
    def getN(self):
        """Returns the number of quotes in the dataset."""
        return len(self.quotes)

    # Return the timestamp at the given index
    # Helps determine the order and time range of quotes
    def getTimestamp(self, index):
        """Returns the timestamp of a quote at the given index."""
        return self.quotes[index]['timestamp']

    # Return the price at the given index
    # Used in computing the average of terminal midquote prices
    def getPrice(self, index):
        """Returns the price of a quote at the given index."""
        return self.quotes[index]['price']

# Define a test case for validating compute_terminal_price
# Inherits from unittest.TestCase to use setup and assertion methods
class TestComputeTerminalPrice(unittest.TestCase):

    # Setup runs before each test method
    # Initializes ImpactMetrics and overrides its quote reader with mock data
    def setUp(self):
        """
        Set up mock quote data for testing terminal price calculation.
        """

        # Define mock quotes from 3:59:00 PM to 4:00:00 PM
        # Only the last 5 quotes before or at 4:00 PM should be used
        self.mock_quotes = [
            {'timestamp': 57540000, 'price': 101.4},  # 3:59:00 PM (6th - should be ignored)
            {'timestamp': 57552000, 'price': 101.5},  # 3:59:12 PM (5th)
            {'timestamp': 57564000, 'price': 101.6},  # 3:59:24 PM (4th)
            {'timestamp': 57576000, 'price': 101.7},  # 3:59:36 PM (3rd)
            {'timestamp': 57588000, 'price': 101.9},  # 3:59:48 PM (2nd)
            {'timestamp': 57600000, 'price': 102.0},  # 4:00:00 PM (Terminal Price - 1st)
        ]

        # Set test date and stock code
        # Used by ImpactMetrics for tagging and path logic
        self.date = "20070920"
        self.stock_code = "IBM"
        self.impact_metrics = ImpactMetrics(self.date, self.stock_code)

        # Create and assign a mock quote reader instance
        # This ensures we use our predefined test quotes instead of loading real data
        self.mock_quotes_reader = MockTAQQuotesReader(self.mock_quotes)
        self.impact_metrics.quotes_reader = self.mock_quotes_reader

    # Test the compute_terminal_price method for correct averaging
    # Uses the last five prices (ignores the earliest one)
    def test_compute_terminal_price(self):
        """
        Test that compute_terminal_price correctly calculates the average of the last five
        mid-quote prices before 4:00 PM.
        """

        # Compute the expected average of the last 5 quotes
        expected_terminal_price = (102.0 + 101.9 + 101.7 + 101.6 + 101.5) / 5

        # Call the method being tested
        computed_terminal_price = self.impact_metrics.compute_terminal_price()

        # Compare computed value with expected average
        # Allow slight floating-point error (6 decimal places)
        self.assertAlmostEqual(computed_terminal_price, expected_terminal_price, places=6)

# Run the test if this file is executed directly
# Useful for local debugging and development
if __name__ == "__main__":
    unittest.main()
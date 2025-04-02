import unittest
import sys
import os
# Add the project root to the system path for cross-folder imports
# Enables access to the ImpactMetrics class from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the class that contains market impact metric computations
# We'll test one of its key functions: compute_arrival_price
from Project.src.ImpactMetrics import ImpactMetrics

# Define a mock class that imitates TAQQuotesReader behavior
# This class allows controlled testing by simulating real quote data
class MockTAQQuotesReader:
    """
    A mock class to simulate TAQQuotesReader.
    Provides getN(), getTimestamp(i), and getPrice(i) methods.
    """

    # Initialize with a list of quote dictionaries (timestamp and price)
    # This replaces file-based reading with in-memory quote simulation
    def __init__(self, quotes):
        self.quotes = quotes

    # Return the number of quotes in the mock dataset
    # Used to iterate through quotes inside ImpactMetrics
    def getN(self):
        """Returns the number of quotes in the dataset."""
        return len(self.quotes)

    # Return the timestamp for a specific quote index
    # Used to bucket and filter quotes by time
    def getTimestamp(self, index):
        """Returns the timestamp of a quote at the given index."""
        return self.quotes[index]['timestamp']

    # Return the price for a specific quote index
    # Used for computing midquote values and returns
    def getPrice(self, index):
        """Returns the price of a quote at the given index."""
        return self.quotes[index]['price']

# Define a unit test class for testing compute_arrival_price
# Uses unittest framework to structure and validate behavior
class TestComputeArrivalPrice(unittest.TestCase):

    # Prepare the test environment before each test
    # Sets mock quotes and overrides the real quotes reader
    def setUp(self):
        """
        Set up mock quote data for testing arrival price calculation.
        """

        # Create mock quote data with timestamps and prices
        # Only the first 5 quotes after 9:30 AM should be averaged
        self.mock_quotes = [
            {'timestamp': 34080000, 'price': 99.5},   # 9:28:00 AM (Should be ignored)
            {'timestamp': 34200000, 'price': 100.0},  # 9:30:00 AM (1st)
            {'timestamp': 34320000, 'price': 100.5},  # 9:32:00 AM (2nd)
            {'timestamp': 34440000, 'price': 101.0},  # 9:34:00 AM (3rd)
            {'timestamp': 34560000, 'price': 101.3},  # 9:36:00 AM (4th)
            {'timestamp': 34680000, 'price': 101.2},  # 9:38:00 AM (5th)
            {'timestamp': 34800000, 'price': 101.5},  # 9:40:00 AM (6th - should be ignored)
        ]

        # Set test metadata for initialization
        # Used by ImpactMetrics to tag and store context
        self.date = "20070920"
        self.stock_code = "IBM"
        self.impact_metrics = ImpactMetrics(self.date, self.stock_code)

        # Initialize and inject the mock reader
        # This prevents any file I/O during the test
        self.mock_quotes_reader = MockTAQQuotesReader(self.mock_quotes)
        self.impact_metrics.quotes_reader = self.mock_quotes_reader

    # Define a test for verifying arrival price calculation logic
    # Ensures the method returns the average of the correct 5 quotes
    def test_compute_arrival_price(self):
        """
        Test that compute_arrival_price correctly calculates the average of the first five
        mid-quote prices after 9:30 AM.
        """

        # Compute the expected average manually from mock data
        expected_arrival_price = (100.0 + 100.5 + 101.0 + 101.3 + 101.2) / 5

        # Call the method under test and store the result
        computed_arrival_price = self.impact_metrics.compute_arrival_price()

        # Assert that the values match with high precision
        self.assertAlmostEqual(computed_arrival_price, expected_arrival_price, places=6)

# Execute the test suite if the script is run directly
# Useful for quick local validation or integration into CI
if __name__ == "__main__":
    unittest.main()
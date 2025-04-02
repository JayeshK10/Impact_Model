import unittest
import sys
import os

# Add the root project directory to the Python path
# This allows importing modules from the parent folder structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the ImpactMetrics class from the project module
# This class computes microstructure metrics like midquote returns
from Project.src.ImpactMetrics import ImpactMetrics

# Define a mock class to simulate behavior of TAQQuotesReader
# Used to feed synthetic data into ImpactMetrics during testing
class MockTAQQuotesReader:
    """
    A mock class to simulate TAQQuotesReader.
    Provides getN(), getTimestamp(i), and getPrice(i) methods.
    """
    # Store the mock quote data during initialization
    # The quotes should be a list of dicts with 'timestamp' and 'price'
    def __init__(self, quotes):
        self.quotes = quotes

    # Return the total number of quote entries
    # Required by the ImpactMetrics logic to loop over quotes
    def getN(self):
        """Returns the number of quotes in the dataset."""
        return len(self.quotes)

    # Return the timestamp at a specific index
    # Enables temporal bucketing of the quote stream
    def getTimestamp(self, index):
        """Returns the timestamp of a quote at the given index."""
        return self.quotes[index]['timestamp']

    # Return the price at a specific index
    # Used to compute midquote returns within each time bucket
    def getPrice(self, index):
        """Returns the price of a quote at the given index."""
        return self.quotes[index]['price']

# Define a test case class for midquote return computation
# Uses Python's unittest framework to structure and validate results
class TestCompute2MinMidquoteReturns(unittest.TestCase):

    # Create a controlled test environment before each test
    # Sets up mock quotes and overrides the ImpactMetrics reader
    def setUp(self):
        """
        Set up mock quote data for testing compute_2min_midquote_returns.

        The trading session is defined between 9:30 AM (34200000 ms) and 4:00 PM (57600000 ms),
        and quotes are bucketed into 2-minute intervals (120000 ms).
        
        The mock quotes include:
          - A quote before 9:30 (ignored).
          - Bucket 0: two quotes between 9:30 and 9:31.
          - Bucket 1: two quotes between 9:32 and 9:33.
          - Bucket 2: one quote at 9:34.
          - Bucket 3: two quotes between 9:36 and 9:37.
          - A quote at 4:00 PM (ignored).
        """
        self.mock_quotes = [
            {'timestamp': 34080000, 'price': 99.5},    # 9:28:00 AM (ignored)
            {'timestamp': 34200000, 'price': 100.0},   # 9:30:00 AM, bucket 0 (first quote)
            {'timestamp': 34230000, 'price': 100.2},   # 9:30:30 AM, bucket 0 
            {'timestamp': 34260000, 'price': 100.5},   # 9:31:00 AM, bucket 0 (last quote)
            {'timestamp': 34320000, 'price': 101.0},   # 9:32:00 AM, bucket 1 (first quote)
            {'timestamp': 34410000, 'price': 101.3},   # 9:33:30 AM, bucket 1 
            {'timestamp': 34430000, 'price': 101.5},   # 9:33:50 AM, bucket 1 (last quote)
            {'timestamp': 34440000, 'price': 102.0},   # 9:34:00 AM, bucket 2 (only quote)
            {'timestamp': 34560000, 'price': 102.5},   # 9:36:00 AM, bucket 3 (first quote)
            {'timestamp': 34620000, 'price': 103.0},   # 9:37:00 AM, bucket 3 (last quote)
            {'timestamp': 57600000, 'price': 104.0},   # 4:00:00 PM (ignored)
        ]

        # Specify mock date and stock code to initialize ImpactMetrics
        self.date = "20070920"
        self.stock_code = "IBM"
        self.impact_metrics = ImpactMetrics(self.date, self.stock_code)
        
        # Create and assign the mock quote reader
        # Overrides actual file reading with synthetic data
        self.mock_quotes_reader = MockTAQQuotesReader(self.mock_quotes)
        self.impact_metrics.quotes_reader = self.mock_quotes_reader

    # Define test function for 2-minute midquote return computation
    # Compares computed bucketed returns with expected output
    def test_compute_2min_midquote_returns(self):
        """
        Test that compute_2min_midquote_returns correctly calculates the mid-quote returns
        for each 2-minute bucket within the trading session.
        """
        # Expected bucket returns:
        # Bucket index is computed as: (timestamp - 34200000) // 120000.
        # Bucket 0: quotes at 9:30 and 9:31 => return = (100.5 / 100.0) - 1
        # Bucket 1: quotes at 9:32 and 9:33:50 => return = (101.5 / 101.0) - 1
        # Bucket 2: single quote at 9:34 => return = 0.0 (no return change)
        # Bucket 3: quotes at 9:36 and 9:37 => return = (103.0 / 102.5) - 1
        expected_returns = {
            0: (100.5 / 100.0) - 1,
            1: (101.5 / 101.0) - 1,
            2: 0.0,
            3: (103.0 / 102.5) - 1,
        }

        # Compute the actual returns using ImpactMetrics method
        computed_returns = self.impact_metrics.compute_2min_midquote_returns()

        # Validate that computed buckets match expected keys
        self.assertEqual(set(computed_returns.keys()), set(expected_returns.keys()))

        # Compare expected and actual return values with tolerance
        for bucket in expected_returns:
            self.assertAlmostEqual(computed_returns[bucket], expected_returns[bucket], places=6)

# Trigger all test cases if the script is executed directly
# Useful for standalone test runs via CLI or IDE
if __name__ == "__main__":
    unittest.main()
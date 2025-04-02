import unittest
import sys
import os

# Add the root directory to the Python path
# Enables importing source code from the parent folder structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the ImpactMetrics class that contains trade imbalance logic
# This is the class being tested in this script
from Project.src.ImpactMetrics import ImpactMetrics

# Define a mock class to simulate TAQTradesReader behavior
# Allows injecting synthetic trade data without accessing actual files
class MockTAQTradesReader:
    """
    A mock class to simulate TAQTradesReader.
    Provides getN(), getTimestamp(i), and getSize(i) methods.
    """

    # Initialize with a list of trade dictionaries
    # Each trade must have a timestamp, price, and size
    def __init__(self, trades):
        self.trades = trades

    # Return the number of trades in the dataset
    # Required for iterating over trades during processing
    def getN(self):
        """Returns the number of trades in the dataset."""
        return len(self.trades)

    # Return the timestamp for a given trade index
    # Used to filter trades within the session window
    def getTimestamp(self, index):
        """Returns the timestamp of a trade at the given index."""
        return self.trades[index]['timestamp']
    
    # Return the price for a given trade index
    # Helps determine trade direction using tick rule
    def getPrice(self, index):
        """Returns the price of a trade at the given index."""
        return self.trades[index]['price']

    # Return the size for a given trade index
    # Used to accumulate volume for buy/sell imbalance
    def getSize(self, index):
        """Returns the size of a trade at the given index."""
        return self.trades[index]['size']

# Define a test case class to test the compute_imbalance function
# Inherits from unittest.TestCase to use setup and assertion methods
class TestComputeImbalance(unittest.TestCase):

    # Setup function to run before each test
    # Initializes ImpactMetrics with mock trade data
    def setUp(self):
        """
        Set up mock trade data for testing trade imbalance calculation.
        """

        # Define a sequence of mock trades with price movement and size
        # Only trades within 9:30 AM to 4:00 PM should be included
        self.mock_trades = [
            {'timestamp': 34080000, 'price': 100.0, 'size': 500},   # 9:28:00 AM (Ignored)
            {'timestamp': 34200000, 'price': 100.0, 'size': 1000},  # 9:30:00 AM (Buy, Neutral)
            {'timestamp': 34320000, 'price': 100.5, 'size': 1500},  # 9:32:00 AM (Buy, +1)
            {'timestamp': 34440000, 'price': 101.0, 'size': 2000},  # 9:34:00 AM (Buy, +1)
            {'timestamp': 34560000, 'price': 100.8, 'size': 2500},  # 9:36:00 AM (Sell, -1)
            {'timestamp': 34680000, 'price': 100.6, 'size': 3000},  # 9:38:00 AM (Sell, -1)
            {'timestamp': 34800000, 'price': 100.7, 'size': 3500},  # 9:40:00 AM (Buy, +1)
            {'timestamp': 55800000, 'price': 101.2, 'size': 4000},  # 3:30:00 PM (Ignored)
        ]

        # Define test metadata for ImpactMetrics
        # The object under test is tied to a date and stock code
        self.date = "20070920"
        self.stock_code = "IBM"
        self.impact_metrics = ImpactMetrics(self.date, self.stock_code)

        # Assign the mock trade reader to the ImpactMetrics instance
        # Replaces file-based reading with our controlled test data
        self.mock_trades_reader = MockTAQTradesReader(self.mock_trades)
        self.impact_metrics.trades_reader = self.mock_trades_reader

    # Test the compute_imbalance method with the mock trades
    # Verifies correct accumulation of trade sizes by direction
    def test_compute_imbalance(self):
        """
        Test that compute_imbalance correctly calculates the net trade imbalance
        based on classified trade directions.
        """

        # Expected calculation breakdown:
        # +1500 (buy) +2000 (buy) -2500 (sell) -3000 (sell) +3500 (buy) = +1500
        expected_imbalance = 1500

        # Compute imbalance using the ImpactMetrics logic
        computed_imbalance = self.impact_metrics.compute_imbalance()

        # Compare expected and computed values
        self.assertEqual(computed_imbalance, expected_imbalance)

# Run the test suite if the script is executed directly
# This is useful for local runs and debugging
if __name__ == "__main__":
    unittest.main()
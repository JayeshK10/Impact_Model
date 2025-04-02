import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

from Project.src.ImpactMetrics import ImpactMetrics



class MockTAQTradesReader:
    """
    A mock class to simulate TAQTradesReader.
    Provides getN(), getTimestamp(i), and getSize(i) methods.
    """
    def __init__(self, trades):
        self.trades = trades

    def getN(self):
        """Returns the number of trades in the dataset."""
        return len(self.trades)

    def getTimestamp(self, index):
        """Returns the timestamp of a trade at the given index."""
        return self.trades[index]['timestamp']

    def getSize(self, index):
        """Returns the size of a trade at the given index."""
        return self.trades[index]['size']


class TestComputeTotalDailyVolume(unittest.TestCase):

    def setUp(self):
        """
        Set up the mock trade data for testing.
        """
        self.mock_trades = [
            {'timestamp': 34080000, 'size': 500},   # 9:28:00 AM (Should be ignored)
            {'timestamp': 34200000, 'size': 1000},  # 9:30:00 AM
            {'timestamp': 34320000, 'size': 1500},  # 9:32:00 AM
            {'timestamp': 34440000, 'size': 2000},  # 9:34:00 AM
            {'timestamp': 34560000, 'size': 2500},  # 9:36:00 AM
            {'timestamp': 34680000, 'size': 3000},  # 9:38:00 AM
            {'timestamp': 34800000, 'size': 3500},  # 9:40:00 AM
            {'timestamp': 57601000, 'size': 4000},  # 4:00:01 PM (Should be ignored)
        ]

        self.date = "20070920"
        self.stock_code = "IBM"
        self.impact_metrics = ImpactMetrics(self.date, self.stock_code)

        # Define a mock trades reader instance
        self.mock_trades_reader = MockTAQTradesReader(self.mock_trades)
        self.impact_metrics.trades_reader = self.mock_trades_reader

    def test_compute_total_daily_volume(self):
        """
        Test that compute_total_daily_volume correctly sums trade sizes
        between 9:30 AM and 4:00 PM.
        """
        expected_total_volume = 1000 + 1500 + 2000 + 2500 + 3000 + 3500  # 13500
        computed_volume = self.impact_metrics.compute_total_daily_volume()
        self.assertEqual(computed_volume, expected_total_volume)


if __name__ == "__main__":
    unittest.main()
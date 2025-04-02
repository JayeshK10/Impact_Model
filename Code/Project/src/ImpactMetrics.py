import sys
import os

# Add Directories to use folder as python modules to import code files.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

# Import TAQ Reader code provided in class
from taq.MyDirectories import MyDirectories  # Handles file directory structure for TAQ data files
from taq.TAQTradesReader import TAQTradesReader  # Reads trade data from TAQ binary files
from taq.TAQQuotesReader import TAQQuotesReader  # Reads quote data from TAQ binary files

# Import Impact Utilities code provided in class
from impactUtils.VWAP import VWAP  # Computes Volume Weighted Average Price (VWAP)
from impactUtils.TickTest import TickTest  # Implements the Tick Test for classifying trade direction
from impactUtils.ReturnBuckets import ReturnBuckets  # Implements the Tick Test for classifying trade direction

import numpy as np
# Define a class for calculating market impact metrics
# Includes VWAP, trade imbalance, volatility, and temporary impact
class ImpactMetrics:
    """
    This class calculates impact metrics for a given stock on a given date.
    It reads trade and quote data and computes various metrics, including mid-quote returns,
    total daily volume, arrival price, trade imbalance, VWAP, and terminal price.
    Then returns 
            - Temporary impact (h)
            - Volatility (sigma)
            - Trade size fraction (q)
            - Average daily traded notional value (for filtering stock)
    """

    # Define the timestamp for 9:30 AM in milliseconds
    # This is the market open time used to filter morning data
    START_TS = 9.5 * 3600 * 1000  # 9:30 AM (34200000 ms)

    # Define timestamp for 3:30 PM in milliseconds
    # Used as cutoff for VWAP used in temporary impact calculation
    THREE_THIRTY_TS = (9.5 + 6) * 3600 * 1000  # 3:30 PM (55800000 ms)

    # Define timestamp for 4:00 PM in milliseconds
    # This marks the official market close
    FOUR_TS = 16 * 3600 * 1000  # 4:00 PM (57600000 ms)

    # Define the bucket length in milliseconds (2 minutes)
    # Used to bucket quotes for volatility calculation
    TWO_MIN_MS = 2 * 60 * 1000  # 2 minutes in milliseconds (120000 ms)

    # Initialize the class with the given date and stock code
    # Also load corresponding trade and quote files for the stock
    def __init__(self, date, stock_code):
        """
        Initialize the ImpactMetrics class for a specific date and stock code.

        Parameters:
            date (str): The trading date in YYYYMMDD format.
            stock_code (str): The stock symbol.
        """
        self.date = date
        self.stock_code = stock_code

        # Construct full path to the trade data file for the stock
        # Path follows: <trades_dir>/<date>/<stock>_trades.binRT
        trades_path = MyDirectories.getTradesDir() + f'/{date}/{stock_code}_trades.binRT'

        # Construct full path to the quote data file for the stock
        # Path follows: <quotes_dir>/<date>/<stock>_quotes.binRQ
        quotes_path = MyDirectories.getQuotesDir() + f'/{date}/{stock_code}_quotes.binRQ'

        # Initialize TAQTradesReader with the file path
        # Used to read and analyze raw trade data
        self.trades_reader = TAQTradesReader(trades_path)

        # Initialize TAQQuotesReader with the file path
        # Used to read and analyze mid-quote price data
        self.quotes_reader = TAQQuotesReader(quotes_path)

    # Compute 2-minute mid-quote returns from quote data
    # Buckets data between 9:30 AM and 4:00 PM and computes return per bucket
    def compute_2min_midquote_returns(self):
        """
        Compute 2-minute mid-quote returns by dividing the quotes into 2-minute buckets
        between 9:30 AM and 4:00 PM and calculating:
            return = (last_price / first_price) - 1
        for each bucket.

        Returns:
            dict: A dictionary where keys are bucket indices and values are computed returns.

        Test - test_compute_2min_midquote_returns.py
        """

        # Define 2-minute interval in milliseconds
        bucket_duration = self.TWO_MIN_MS

        # Dictionary to store first and last quote in each 2-minute bucket
        buckets = {}

        # Get the number of available quotes
        N = self.quotes_reader.getN()

        # Iterate through all quotes
        for i in range(N):
            # Get the timestamp of the i-th quote
            timestamp = self.quotes_reader.getTimestamp(i)

            # Skip data after 4:00 PM
            if timestamp >= self.FOUR_TS:
                break

            # Skip data before 9:30 AM
            if timestamp < self.START_TS:
                continue

            # Determine which 2-minute bucket this timestamp falls into
            bucket_idx = (timestamp - self.START_TS) // bucket_duration

            # Get the quote price (mid-quote approximation)
            price = self.quotes_reader.getPrice(i)

            # Initialize the bucket with first and last prices
            if bucket_idx not in buckets:
                buckets[bucket_idx] = {'first': price, 'last': price}
            else:
                # Update last price as we iterate forward
                buckets[bucket_idx]['last'] = price

        # Dictionary to store computed returns for each bucket
        bucket_returns = {}

        # Compute return = (last / first) - 1 for each bucket
        for bucket_idx, prices in buckets.items():
            first_price = prices['first']
            last_price = prices['last']

            # Avoid division by zero
            if first_price != 0:
                bucket_returns[bucket_idx] = (last_price / first_price) - 1
            else:
                bucket_returns[bucket_idx] = None

        # Return the dictionary of bucket-wise returns
        return bucket_returns

    # Compute total trade volume from all trades between 9:30 AM and 4:00 PM
    # Adds up trade sizes for valid timestamps
    def compute_total_daily_volume(self):
        """
        Compute total daily volume by summing trade sizes between 9:30 AM and 4:00 PM.

        Returns:
            int: Total trade volume for the day.

        Test - test_compute_total_daily_volume.py
        """

        # Initialize total volume counter
        total_trade_size = 0

        # Iterate through all trade records
        for i in range(0, self.trades_reader.getN()):
            # Fetch the timestamp for the trade
            timestamp = self.trades_reader.getTimestamp(i)

            # Skip trades after 4:00 PM
            if timestamp >= self.FOUR_TS:
                break

            # Skip trades before 9:30 AM
            if timestamp < self.START_TS:
                continue

            # Add trade size to the daily total
            total_trade_size += self.trades_reader.getSize(i)

        # Return the total volume for valid trades
        return total_trade_size    

    # Compute average mid-quote of first 5 prices after 9:30 AM
    # Used to represent the price at the start of trading
    def compute_arrival_price(self):
        """
        Compute the arrival price as the average of the first five mid-quote prices after 9:30 AM.

        Returns:
            float: Arrival price (average of first five mid-quotes).

        Test - test_compute_arrival_price.py
        """       

        # Initialize sum and counter to accumulate first 5 valid prices
        arrival_mid_quote_price_sum = 0
        counter = 0

        # Loop through quote data in chronological order
        for i in range(self.quotes_reader.getN()):
            # Get timestamp of current quote
            timestamp = self.quotes_reader.getTimestamp(i)

            # Stop once we collect 5 quotes
            if counter >= 5:
                break

            # Ignore quotes after 4:00 PM
            if timestamp >= self.FOUR_TS:
                break

            # Skip quotes before 9:30 AM
            if timestamp < self.START_TS:
                continue

            # Add valid quote price to the total
            arrival_mid_quote_price_sum += self.quotes_reader.getPrice(i)
            counter += 1

        # Return average if we collected any quotes
        return arrival_mid_quote_price_sum / counter if counter > 0 else 0

    # Compute average mid-quote of last 5 prices before 4:00 PM
    # Used to represent the price at the end of trading
    def compute_terminal_price(self):
        """
        Compute the terminal price as the average of the last five mid-quote prices before 4:00 PM.

        Returns:
            float: Terminal price.

        Test - test_compute_terminal_price.py
        """

        # Initialize sum and counter to accumulate last 5 valid prices
        terminal_mid_quote_price_sum = 0
        counter = 0

        # Loop through quotes in reverse (latest to earliest)
        for i in range(self.quotes_reader.getN() - 1, -1, -1):
            # Get timestamp of current quote
            timestamp = self.quotes_reader.getTimestamp(i)

            # Stop once we collect 5 quotes
            if counter >= 5:
                break

            # Stop early if we’re before market open
            if timestamp < self.START_TS:
                break

            # Skip quotes after 4:00 PM
            if timestamp > self.FOUR_TS:
                continue

            # Add valid quote price to the total
            terminal_mid_quote_price_sum += self.quotes_reader.getPrice(i)
            counter += 1

        # Return average if at least one quote was found
        return terminal_mid_quote_price_sum / counter if counter > 0 else 0

    # Compute trade imbalance using Tick Rule between 9:30 AM and 3:30 PM
    # Considers direction of price changes and trade sizes
    def compute_imbalance(self):
        """
        Compute the net trade imbalance (in shares) between 9:30 AM and 3:30 PM using the Tick Test.

        Returns:
            int: Net imbalance in shares.

        Test - test_compute_imbalance.py
        """

        # Initialize net imbalance counter
        imbalance_shares = 0

        # Store previous trade price to compute price direction
        prev_price = None

        # Remember last non-zero tick to handle flat prices
        last_nonzero_tick = 0

        # Iterate through all trades
        for i in range(0, self.trades_reader.getN()):
            # Get the timestamp for the current trade
            timestamp = self.trades_reader.getTimestamp(i)

            # Stop if we've reached or passed 3:30 PM
            if timestamp >= self.THREE_THIRTY_TS:
                break

            # Skip trades before 9:30 AM
            if timestamp < self.START_TS:
                continue

            # Fetch trade price and size
            price = self.trades_reader.getPrice(i)
            size = self.trades_reader.getSize(i)

            # Tick Test to determine trade direction
            if prev_price is None:
                tick_class = 0  # No previous price to compare
            elif price > prev_price:
                tick_class = 1  # Price increased → buy
            elif price < prev_price:
                tick_class = -1  # Price decreased → sell
            else:
                tick_class = last_nonzero_tick  # Flat price → repeat last direction

            # Update last non-zero tick direction
            if tick_class != 0:
                last_nonzero_tick = tick_class

            # Save the current price for next iteration
            prev_price = price

            # Accumulate imbalance based on trade direction and size
            imbalance_shares += tick_class * size

        # Return the total imbalance in shares
        return imbalance_shares

    # Compute VWAP (Volume-Weighted Average Price) for a given time range
    # Delegates to VWAP utility class to handle internal logic
    def compute_VWAP(self, start, end):
        """
        Compute the volume-weighted average price (VWAP) for trades in the given time range.

        Parameters:
            start (int): Start timestamp in milliseconds.
            end (int): End timestamp in milliseconds.

        Returns:
            float: VWAP value.

        Test - Code/impactUtils/Test_VWAP.py
        """

        # Instantiate VWAP helper with trade data and time range
        VWAP_class = VWAP(self.trades_reader, start, end)

        # Return the computed VWAP value
        return VWAP_class.getVWAP()    

    def compute_metrics(self):
        """
        Compute and return a dictionary with all required metrics for Almgren-Chriss impact modeling:
            - Temporary impact (h)
            - Volatility (sigma)
            - Trade size fraction (q)
            - Average daily traded notional value (for filtering stock)

        Also returns a flag indicating if any value is missing or invalid.

        Returns:
            tuple: (result_dict, missing_flag)

        Test - test_compute_metrics.py
        """

        # --- 1. Compute 2-minute mid-quote returns ---
        # Generate bucketed mid-quote return dictionary
        _2min_mid_quote = self.compute_2min_midquote_returns()
        # Extract the return values for volatility computation
        returns = list(_2min_mid_quote.values())

        # --- 2. Compute total daily volume in shares traded ---
        # Aggregates trade sizes across the full day
        total_daily_volume = self.compute_total_daily_volume()

        # --- 3. Compute arrival and terminal prices ---
        # Arrival = average price at start (9:30 AM)
        arrival_price = self.compute_arrival_price()
        # Terminal = average price near end (4:00 PM)
        terminal_price = self.compute_terminal_price()

        # --- 4. Compute trade imbalance in number of shares ---
        # Positive imbalance = more buying; used to scale impact
        imbalance_shares = self.compute_imbalance()

        # --- 5. Compute VWAP from 9:30 to 3:30 and 9:30 to 4:00 ---
        # VWAP_9_3 is used to calculate temporary impact (h)
        VWAP_9_3 = self.compute_VWAP(self.START_TS, self.THREE_THIRTY_TS)
        # VWAP_9_4 is used for dollar notional scaling
        VWAP_9_4 = self.compute_VWAP(self.START_TS, self.FOUR_TS)

        # Ensure both VWAPs are valid and strictly positive
        assert VWAP_9_3 > 0 and VWAP_9_4 > 0, "VWAPs must be positive"

        # --- 6. Compute annualized volatility from intraday returns ---
        # Std deviation scaled by √195 (2-min intervals per trading day)
        volatility = np.std(returns, ddof=1) * np.sqrt(195) if returns else None
        # Sanity check for volatility
        assert volatility is None or volatility >= 0, "Volatility must be non-negative"

        # --- 7. Compute dollar imbalance and average daily notional value ---
        # Dollar value of net imbalance using VWAP
        imbalance_value = imbalance_shares * VWAP_9_4 if VWAP_9_4 is not None else None
        # Total daily notional volume (used for stock filtering and q)
        avg_daily_value = total_daily_volume * VWAP_9_4 if VWAP_9_4 is not None else None

        # Ensure imbalance does not exceed total notional volume
        if imbalance_value is not None and avg_daily_value is not None:
            assert imbalance_value <= avg_daily_value, "Imbalance cannot exceed daily traded value"

        # --- 8. Estimate Permanent (g) and Temporary (h) Market Impact ---
        # Permanent impact = half the mid-price movement across session
        g = (terminal_price - arrival_price) / 2 if arrival_price is not None and terminal_price is not None else None
        # Temporary impact = VWAP deviation from arrival, adjusted for g
        h = (VWAP_9_3 - arrival_price) - g if VWAP_9_3 is not None and arrival_price is not None and g is not None else None

        # --- 9. Compute trade size ratio q for regression ---
        # q = dollar imbalance / scaled average daily value
        scaling_factor = 6 / 6.5  # Market is open 6.5 hrs, but we use 6 for VWAP window
        q = imbalance_value / (scaling_factor * avg_daily_value) if imbalance_value and avg_daily_value else None

        # --- 10. Prepare result dictionary ---
        result = {
            'avg_daily_value': avg_daily_value,  # For filtering out illiquid stocks
            'h': h,                               # Temporary market impact
            'sigma': volatility,                  # Daily volatility measure
            'q': q                                # Normalized trade size
        }

        # --- 11. Flag whether any metrics are missing ---
        # Helps filter out invalid rows before regression
        missing_flag = any(value is None for value in result.values())

        return result, missing_flag
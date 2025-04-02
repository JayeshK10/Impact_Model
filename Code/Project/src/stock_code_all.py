import re
import sys
import os

# Add parent directories to Python path to enable cross-folder imports
# Makes it possible to import project modules while testing or running scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the MyDirectories helper for fetching project folder paths
# This abstraction handles standardized directory locations for TAQ data
from taq import MyDirectories


'''
    Test - test_stock_code_all.py
'''


# Define function to extract code names from files in a directory
# Uses regex pattern matching to capture the base stock codes
def get_code_names(directory, pattern):
    code_names = []
    # Compile the given regex pattern for performance
    # Enables pattern reuse during iteration
    regex = re.compile(pattern)

    # Loop through all files in the specified directory
    # Attempt to match each filename against the pattern
    for filename in os.listdir(directory):
        match = regex.match(filename)
        if match:
            # If match found, extract the stock code using group(1)
            # This assumes the code is captured by the first regex group
            code_names.append(match.group(1))

    # Convert the final list to a set to remove duplicates
    # Returns a unique set of all matched stock codes
    return set(code_names)

# Entry point to execute logic when this script runs directly
# Performs code name extraction and intersection for quotes and trades
if __name__ == "__main__":
    # ----------- Quotes -----------
    # Get root directory path for quote data
    # This typically returns a path like .../taq/quotes
    quotes_directory = MyDirectories.getQuotesDir()

    # List all dated subdirectories inside the quotes directory
    # Each subdirectory contains binary quote files for a given date
    list_of_dates_quotes = os.listdir(quotes_directory)

    # Initialize variable to hold intersection of stock codes across dates
    # Starts as None so it can be replaced by the first date's result
    common_code_names_quotes = None

    # Define the filename pattern for quote files
    # The regex captures any prefix before '_quotes.binRQ'
    quote_pattern = r"^(.*?)_quotes\.binRQ$"

    # Loop through all date folders inside the quotes directory
    for date in list_of_dates_quotes:
        directory_path_quotes = os.path.join(quotes_directory, date)

        # Skip if the current path is not a directory
        # Ensures only folders are processed
        if not os.path.isdir(directory_path_quotes):
            continue

        # Extract stock code names from current date's quote folder
        code_names = get_code_names(directory_path_quotes, quote_pattern)

        # If this is the first valid folder, initialize the common set
        # Else intersect with previous set to get shared codes only
        if common_code_names_quotes is None:
            common_code_names_quotes = code_names
        else:
            common_code_names_quotes = common_code_names_quotes.intersection(code_names)

    # Print the number of codes common across all quote days
    # Verifies consistency of stock data over the dates
    print("Quotes: Common code names across all dates:", len(common_code_names_quotes))

    # ----------- Trades -----------
    # Get root directory path for trade data
    # Typically returns a path like .../taq/trades
    trades_directory = MyDirectories.getTradesDir()

    # List all dated subdirectories inside the trades directory
    # Each subfolder contains trade data files for a specific date
    list_of_dates_trades = os.listdir(trades_directory)

    # Initialize variable to hold common trade codes
    # Will store stock codes shared across all dates
    common_code_names_trades = None

    # Define the regex pattern to identify trade files
    # Captures prefix before '_trades.binRT'
    trade_pattern = r"^(.*?)_trades\.binRT$"

    # Loop through all date folders in the trades directory
    for date in list_of_dates_trades:
        directory_path_trades = os.path.join(trades_directory, date)

        # Skip if not a valid directory
        # Ensures no file entries are processed
        if not os.path.isdir(directory_path_trades):
            continue

        # Extract stock codes from trade filenames using pattern
        code_names = get_code_names(directory_path_trades, trade_pattern)

        # Initialize or update the intersection set
        # Keeps only stock codes that appear on all dates
        if common_code_names_trades is None:
            common_code_names_trades = code_names
        else:
            common_code_names_trades = common_code_names_trades.intersection(code_names)

    # Output the number of trade codes common across all days
    # Helps confirm uniform coverage in trade dataset
    print("Trades: Common code names across all dates:", len(common_code_names_trades))

    # ----------- Final Intersection -----------
    # Compute the intersection of quotes and trades code sets
    # Identifies stock codes consistently available in both datasets
    common_code_names_both = common_code_names_quotes.intersection(common_code_names_trades)

    # Display the number of stock codes common to both quotes and trades
    # Useful for selecting stable stocks for downstream processing
    print("Final: Common code names across both quotes and trades:", len(common_code_names_both))

    # ----------- Save to File -----------
    # Construct full file path for saving the selected stock codes
    # File will be stored inside the data directory
    stock_selection_path_both = os.path.join(MyDirectories.getDataDir(), 'stock_selection_common.txt')

    # Open the output file for writing
    # Each selected stock code will be written to a new line
    with open(stock_selection_path_both, 'w') as f:
        for stock in sorted(common_code_names_both):
            # Write each stock symbol to the file
            # Ensure consistent alphabetical order by sorting
            f.write(stock + '\n')
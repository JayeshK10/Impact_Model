import sys
import os

# Add Directories to use folder as python modules to import code files.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

from Project.src.ImpactMetrics import ImpactMetrics
from taq.MyDirectories import MyDirectories

import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd

# Directory to save per-stock metric checkpoints
# Used to avoid recomputation across script reruns
CHECKPOINT_DIR = os.path.join(MyDirectories.getDataDir(), "checkpoint_common")

# Ensure the checkpoint directory exists
# Prevents FileNotFoundError when saving outputs
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Directory to save final cleaned and aggregated data
# Intended for downstream analysis or regression
FINAL_DATA_DIR = os.path.join(MyDirectories.getDataDir(), "final_data_common")

# Ensure the final output directory exists
# Keeps results organized and persistent
os.makedirs(FINAL_DATA_DIR, exist_ok=True)

# Function to compute metrics for a single stock across multiple dates
# Also handles error catching and checkpoint saving
def stock_metrics_func(stock_date_tuple):
    """
    Compute metrics for a single stock across all dates.
    Args:
        stock_date_tuple: tuple containing (stock_code, list_of_dates)
    Returns:
        tuple: stock_code, metrics_dict, flags_dict

    Test - test_stock_metrics_func.py
    """
    # Unpack stock symbol and list of available dates
    stock_code, list_of_dates = stock_date_tuple

    # Check if checkpoint file already exists for this stock
    # If yes, skip computation to save time
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"stock_{stock_code}.pkl")
    if os.path.exists(checkpoint_path):
        return None  # Already done, avoid recomputation

    # Initialize dictionary to store results per date
    metrics_dict = {}

    # Dictionary to flag dates where metrics are incomplete or errored
    flags_dict = {}

    # Iterate over each trading date for the stock
    for date in list_of_dates:
        try:
            # Initialize ImpactMetrics for the given stock and date
            impact_metrics = ImpactMetrics(date, stock_code)

            # Compute all metrics (h, sigma, q, avg_daily_value)
            metrics, is_missing = impact_metrics.compute_metrics()

        except Exception as e:
            # On error (missing file, parsing issue, etc.), skip this date
            metrics = None
            is_missing = True
            print(f"Error: {stock_code}-{date} → {e}")

        # Store computed results and missing flag
        metrics_dict[date] = metrics
        flags_dict[date] = is_missing

    # Save checkpoint for the entire stock after processing all dates
    # Ensures persistence and resume-ability for future runs
    with open(checkpoint_path, "wb") as f:
        pickle.dump((metrics_dict, flags_dict), f)

    # Return None as we're saving results externally
    return None


# Load all saved per-stock checkpoint files and rebuild two DataFrames:
# 1. `metrics_df`: holds computed metrics (dicts)
# 2. `flags_df`: holds missing flags (bools)
def aggregate_checkpoints():
    """
    Load all stock-level checkpoints and rebuild the full DataFrames.
    Returns:
        metrics_df, flags_df

    Test - test_aggregate_checkpoints.py
    """

    # Dictionary to store per-stock metrics across dates
    stock_metrics = {}

    # Dictionary to store per-stock missing value flags
    stock_flags = {}

    # Loop through all checkpoint files in the checkpoint directory
    for file in os.listdir(CHECKPOINT_DIR):

        # Filter only valid stock checkpoint files
        if file.endswith(".pkl") and file.startswith("stock_"):

            # Extract the stock code from the filename
            stock = file.split("_")[1].split(".")[0]

            # Open and load the checkpoint file
            with open(os.path.join(CHECKPOINT_DIR, file), "rb") as f:
                metrics, flags = pickle.load(f)

            # Store loaded results into dictionaries
            stock_metrics[stock] = metrics
            stock_flags[stock] = flags

    # Convert the metrics dictionary to a DataFrame
    # Rows: stocks, Columns: dates, Cells: metric dicts
    metrics_df = pd.DataFrame.from_dict(stock_metrics, orient="index")

    # Convert the flags dictionary to a DataFrame
    # Rows: stocks, Columns: dates, Cells: boolean flags
    flags_df = pd.DataFrame.from_dict(stock_flags, orient="index")

    # Return both DataFrames
    return metrics_df, flags_df

# Split the main metrics DataFrame (stock x date with dicts)
# Into multiple DataFrames: one for each metric key (e.g., 'h', 'q', etc.)
def split_metric_dataframe(metrics_df, keys):
    """
    Splits a DataFrame of dicts into separate DataFrames for each key.
    
    Args:
        metrics_df (pd.DataFrame): stock x date, with dicts as values
        keys (list of str): keys to extract from each dict
        
    Returns:
        dict: {metric_name: DataFrame}
    
    Test - test_split_metric_dataframe.py
    """

    # Dictionary to hold extracted metric-specific DataFrames
    split_dfs = {}

    # Loop through all keys to extract (e.g., 'h', 'q', 'sigma')
    for key in keys:

        # Apply a lambda across each column (date)
        # Each cell contains a dict — extract `key` if present
        df = metrics_df.apply(lambda col: col.map(lambda d: d.get(key) if isinstance(d, dict) else None))

        # Save the result in the output dictionary
        split_dfs[key] = df

    # Return a dictionary of DataFrames, one per metric
    return split_dfs

# Filter metrics and flags to keep only top-K stocks by average daily value
# Useful for reducing dataset to the most liquid or representative tickers
def filter_top_market_cap_stocks(metrics_df, flags_df, top_k=1500, save_dir=None):
    """
    Filters the metrics and flags DataFrames to retain only the top-K stocks by average daily value
    (used here as a proxy for market cap or liquidity).

    Args:
        metrics_df (pd.DataFrame): Stock * Date DataFrame with metric dicts.
        flags_df (pd.DataFrame): Stock * Date DataFrame with missingness flags.
        top_k (int): Number of top stocks to retain.
        save_dir (str or Path): Directory to save the top stock list (optional).

    Returns:
        filtered_metrics_df (pd.DataFrame): Metrics DataFrame for top-K stocks.
        filtered_flags_df (pd.DataFrame): Flags DataFrame for top-K stocks.
        top_stocks (list): List of top-K stock tickers.

    Test - test_filter_top_market_cap_stocks.py
    """

    # Extract a DataFrame of just avg_daily_value per stock-date
    split_once = split_metric_dataframe(metrics_df, ['avg_daily_value'])
    avg_daily_value_df = split_once['avg_daily_value']

    # Compute mean average daily value across all available dates
    # This serves as a liquidity or size proxy (higher → more liquid)
    avg_daily_value_mean = avg_daily_value_df.mean(axis=1)

    # Select top-K stocks with highest average daily value
    top_stocks = avg_daily_value_mean.sort_values(ascending=False).head(top_k).index.tolist()

    # Filter metrics DataFrame to keep only top-K stocks
    filtered_metrics_df = metrics_df.loc[top_stocks]

    # Filter flags DataFrame to keep only top-K stocks
    filtered_flags_df = flags_df.loc[top_stocks]

    # Optionally save top-K stock tickers to a text file
    if save_dir is not None:
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Define output path
        output_path = os.path.join(save_dir, f"top_{top_k}_market_cap_stocks.txt")

        # Write each ticker to a new line
        with open(output_path, 'w') as f:
            for stock in top_stocks:
                f.write(stock + '\n')

        # Print confirmation for user
        print(f"Saved top {top_k} stock list to: {output_path}")

    # Return the filtered DataFrames and list of selected tickers
    return filtered_metrics_df, filtered_flags_df, top_stocks

# Filter metrics and flags DataFrames by missingness thresholds
# Ensures clean and robust data for regression or downstream modeling
def filter_metrics(metrics_df, flags_df, stock_thresh=0.1, date_thresh=0.01, max_tolerable_missing_dates=10):
    """
    Filters the metrics and flags DataFrames based on missingness in multiple steps:
    
    1. Remove stocks with missing data above stock_thresh.
    2. Remove dates with missing data above date_thresh.
    3. Check if any dates still have missing values:
        a. If number of such dates <= 10 → drop all of them.
        b. If > 10 → drop top 10 dates with most missing values.
    4. Finally, drop all stocks with any remaining NaNs.

    Args:
        metrics_df (pd.DataFrame): DataFrame with stock as rows, date as columns, and metric dicts as values.
        flags_df (pd.DataFrame): DataFrame with boolean values indicating missingness per (stock, date).
        stock_thresh (float): Max allowed fraction of missing dates for a stock.
        date_thresh (float): Max allowed fraction of missing stocks for a date.
        max_tolerable_missing_dates (int): Max number of dates allowed to have any missing values before capping.

    Returns:
        cleaned_metrics_df (pd.DataFrame): Fully cleaned DataFrame (no NaNs).
        dropped_dates (list): List of all dropped dates.
        dropped_stocks (list): List of all dropped stocks.
        kept_dates (list): List of dates retained.
        kept_stocks (list): List of stocks retained.

    Test - test_filter_metrics.py
    """

    # Initialize lists to track what is dropped
    dropped_dates = []
    dropped_stocks = []

    # --- Step 1: Filter out stocks with excessive missing data ---
    # Compute fraction of missing entries for each stock
    missing_per_stock = flags_df.mean(axis=1)

    # Keep only stocks below the missing threshold
    kept_stocks = missing_per_stock[missing_per_stock <= stock_thresh].index.tolist()

    # Identify and record dropped stocks
    dropped_stocks += list(set(flags_df.index) - set(kept_stocks))

    # Apply the filtering to both metrics and flags DataFrames
    metrics_df = metrics_df.loc[kept_stocks]
    flags_df = flags_df.loc[kept_stocks]

    # --- Step 2: Filter out dates with too much missing data ---
    # Compute fraction of missing stocks per date
    missing_per_date = flags_df.mean(axis=0)

    # Retain only dates below the missing threshold
    kept_dates = missing_per_date[missing_per_date <= date_thresh].index.tolist()

    # Track dropped dates
    dropped_dates += list(set(flags_df.columns) - set(kept_dates))

    # Apply the filtering
    metrics_df = metrics_df.loc[:, kept_dates]
    flags_df = flags_df.loc[:, kept_dates]

    # --- Step 3: Handle remaining missing values more carefully ---
    # Recompute missingness after initial pruning
    still_missing_per_date = flags_df.mean(axis=0)

    # Identify dates that still have any missing values
    dates_with_missing = still_missing_per_date[still_missing_per_date > 0]

    # Decide how to deal with remaining "imperfect" dates
    if len(dates_with_missing) <= max_tolerable_missing_dates:
        # If few enough dates → drop all of them
        to_drop = dates_with_missing.index.tolist()
    else:
        # Otherwise, drop top-N worst dates (most missing values)
        to_drop = dates_with_missing.sort_values(ascending=False).head(max_tolerable_missing_dates).index.tolist()

    # Update dropped and kept dates accordingly
    dropped_dates += to_drop
    kept_dates = [d for d in flags_df.columns if d not in to_drop]

    # Drop selected dates from both metrics and flags
    metrics_df = metrics_df.loc[:, kept_dates]
    flags_df = flags_df.loc[:, kept_dates]

    # --- Step 4: Final clean-up: drop any stocks with remaining missing data ---
    # Identify stocks that still have any True flags (missing values)
    final_missing_per_stock = flags_df.any(axis=1)

    # Keep only stocks with no missing values
    stocks_with_no_missing = final_missing_per_stock[~final_missing_per_stock].index.tolist()

    # Track final dropped stocks
    dropped_stocks += final_missing_per_stock[final_missing_per_stock].index.tolist()

    # Final cleaned metrics DataFrame
    cleaned_metrics_df = metrics_df.loc[stocks_with_no_missing, :]

    # Update final kept stock list
    kept_stocks = stocks_with_no_missing

    return cleaned_metrics_df, dropped_dates, dropped_stocks, kept_dates, kept_stocks

# Main function to orchestrate the full preprocessing pipeline
# Includes metric computation, filtering, and serialization
def main(list_of_stocks, list_of_dates, use_multiprocessing=True):
    '''
    Test - test_data_preprocess_main.py
    '''

    # Create a list of (stock, list_of_dates) tuples
    # Each tuple is sent to stock_metrics_func for processing
    stock_date_tuples = [(stock, list_of_dates) for stock in list_of_stocks]

    # Step 1: Run metric computation per stock (parallel or serial)
    if use_multiprocessing:
        # Import multiprocessing tools
        from multiprocessing import Pool, cpu_count

        # Use all CPU cores to parallelize stock processing
        with Pool(cpu_count()) as pool:
            list(tqdm(
                pool.imap_unordered(stock_metrics_func, stock_date_tuples),
                total=len(stock_date_tuples),
                desc="Processing stocks"
            ))
    else:
        # Fallback to serial mode (used during testing/debugging)
        for item in tqdm(stock_date_tuples, desc="Processing stocks"):
            stock_metrics_func(item)

    # Step 2: Aggregate per-stock checkpoints into unified DataFrames
    # Returns: metrics_df (dicts), flags_df (missing flags)
    metrics_df, flags_df = aggregate_checkpoints()

    # Step 3: Select top 1500 stocks by average daily value
    # Used as a proxy for market cap or liquidity
    metrics_df, flags_df, top_stocks = filter_top_market_cap_stocks(
        metrics_df,
        flags_df,
        top_k=1500,
        save_dir=None  # Can optionally specify output dir for top stock list
    )

    # Step 4: Filter out stocks/dates with excessive missing data
    # Applies thresholds and smart heuristics for data quality
    cleaned_metrics_df, dropped_dates, dropped_stocks, kept_dates, kept_stocks = filter_metrics(
        metrics_df,
        flags_df,
        stock_thresh=0.1,                # Max 10% of dates missing per stock
        date_thresh=0.01,                # Max 1% of stocks missing per date
        max_tolerable_missing_dates=10   # Drop top 10 dates with missingness if needed
    )

    # Step 5: Split the final cleaned metric dicts into separate DataFrames
    # Result: {metric_name: pd.DataFrame[stock x date]}
    metric_keys = [
        'h',               # Temporary impact
        'sigma',           # Volatility
        'q',               # Trade size fraction
        'avg_daily_value'  # Liquidity proxy
    ]
    split_dfs = split_metric_dataframe(cleaned_metrics_df, metric_keys)

    # Step 6: Save the cleaned metric matrices
    # Includes per-metric DataFrames and metadata about the cleaning process
    data_path = os.path.join(FINAL_DATA_DIR, "input_matrix.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(split_dfs, f)

    # Save supporting metadata for reproducibility
    info_path = os.path.join(FINAL_DATA_DIR, "info.pkl")
    with open(info_path, "wb") as f:
        pickle.dump({
            "dropped_dates": dropped_dates,
            "dropped_stocks": dropped_stocks,
            "kept_dates": kept_dates,
            "kept_stocks": kept_stocks,
            "top_stocks": top_stocks
        }, f)

    # Step 7: Logging outputs for tracking
    print("\nDrop Dates: ", dropped_dates)
    print("Drop Stocks: ", dropped_stocks)
    print(f"\n Saved data matrix to {data_path}")
    print(f" Saved info data to {info_path}")

# Entrypoint for running the script directly
# Reads dates and stock list, then calls main()
if __name__ == "__main__":

    # Get the directory with quote data organized by date
    quotes_direcotry = MyDirectories.getQuotesDir()

    # List all available trading dates from the quote folders
    list_of_dates = os.listdir(quotes_direcotry)

    # Load the final list of stocks to process (previously filtered)
    stock_selection_path = os.path.join(MyDirectories.getDataDir(), 'stock_selection_common.txt')
    with open(stock_selection_path, 'r') as f:
        list_of_stocks = [line.strip() for line in f]

    # Launch the full preprocessing pipeline
    main(list_of_stocks, list_of_dates, use_multiprocessing=True)
    
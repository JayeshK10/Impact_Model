import sys
import os

# Add parent directory to sys.path to enable local module imports
# This allows importing from `taq` and `Project.src` folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

import pickle
import pandas as pd
import numpy as np
from taq.MyDirectories import MyDirectories
from scipy.optimize import curve_fit

'''
Test - test_NLS.py
'''

# Function to sort columns (dates) in chronological order
def sort_columns_by_date(df):
    # Ensure column labels are datetime objects
    df.columns = pd.to_datetime(df.columns)

    # Sort columns chronologically (left to right)
    return df.sort_index(axis=1)


# Load and preprocess input data for regression
# Returns a long-form DataFrame with columns: stock, date, h, sigma, q
def preprocess():
    # Define path to the saved final metric matrices
    FINAL_DATA_DIR = os.path.join(MyDirectories.getDataDir(), "final_data_common")
    data_path = os.path.join(FINAL_DATA_DIR, "input_matrix.pkl")

    # Load the split metric DataFrames: {"h": df, "sigma": df, "q": df}
    with open(data_path, "rb") as f:
        split_dfs = pickle.load(f)

    # Sort each matrix by date (column order)
    h_df = sort_columns_by_date(split_dfs["h"])
    sigma_df = sort_columns_by_date(split_dfs["sigma"])
    q_df = sort_columns_by_date(split_dfs["q"])

    # Smooth sigma using rolling average (10-day window, row-wise)
    sigma_df = sigma_df.apply(lambda row: row.rolling(window=10, min_periods=1).mean(), axis=1)

    # Convert each matrix from wide to long format (stack = melt)
    h_long = h_df.stack().rename("h")                # MultiIndex: (stock, date)
    sigma_long = sigma_df.stack().rename("sigma")
    q_long = q_df.stack().rename("q")

    # Join all 3 into a single long-form DataFrame
    impact_df = pd.concat([h_long, sigma_long, q_long], axis=1).reset_index()
    impact_df.columns = ['stock', 'date', 'h', 'sigma', 'q']

    # Sort by date and drop any rows with NaNs
    impact_df = impact_df.sort_values(by="date").dropna().reset_index(drop=True)

    return impact_df


# Run nonlinear least squares to estimate eta and beta
# Uses the impact model: h = eta * sigma * q^beta
def estimate_impact_params_from_df(impact_df):
    """
    Estimate eta and beta parameters for the Almgren-Chriss impact model:
        h = eta * sigma * q^beta
    """

    # Flatten all values to 1D arrays
    h_flat = impact_df['h'].values.flatten()
    sigma_flat = impact_df['sigma'].values.flatten()
    q_flat = impact_df['q'].values.flatten()

    # Build a mask to filter valid (non-NaN and positive q) entries
    valid_mask = ~np.isnan(h_flat) & ~np.isnan(sigma_flat) & ~np.isnan(q_flat) & (q_flat > 0)

    # Filter the data accordingly
    h_flat = h_flat[valid_mask]
    sigma_flat = sigma_flat[valid_mask]
    q_flat = q_flat[valid_mask]

    # Define the nonlinear model for curve fitting
    def impact_model(q, eta, beta):
        return eta * sigma_flat * (q ** beta)

    # Initial guess for eta and beta
    initial_guess = [1.0, 0.5]

    # Fit the nonlinear model using curve_fit
    params, _ = curve_fit(impact_model, q_flat, h_flat, p0=initial_guess, maxfev=1000000)

    return params


# Entrypoint to run the full pipeline
if __name__ == "__main__":

    # Step 1: Load and preprocess input matrices
    impact_df = preprocess()

    # Step 2: Estimate eta and beta via nonlinear regression
    eta_hat, beta_hat = estimate_impact_params_from_df(impact_df)

    # Format estimates as strings for output
    eta_text = f"Estimated eta = {eta_hat:.6f}"
    beta_text = f"Estimated beta = {beta_hat:.6f}"

    # Print results to stdout
    print(eta_text)
    print(beta_text)

    # Step 3: Save estimates to a result file
    estimate_path = os.path.join(MyDirectories.getDataDir(), "regression_results.txt")
    with open(estimate_path, "w") as f:
        f.write(eta_text + "\n")
        f.write(beta_text + "\n")
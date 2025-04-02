import sys
import os

# Add Directories to use folder as python modules to import code files.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

import pickle
import pandas as pd
import numpy as np
from taq.MyDirectories import MyDirectories
from scipy.optimize import curve_fit

'''
Test - test_NLS_activity.py
'''
# Utility to ensure date columns in correct order
def sort_columns_by_date(df):
    # Convert column headers to datetime
    df.columns = pd.to_datetime(df.columns)
    # Sort columns chronologically
    return df.sort_index(axis=1)

# Load and reshape input matrices from disk
# Returns a DataFrame with [stock, date, h, sigma, q, avg_daily_value]
def preprocess():
    # Path to input data
    FINAL_DATA_DIR = os.path.join(MyDirectories.getDataDir(), "final_data_common")
    data_path = os.path.join(FINAL_DATA_DIR, "input_matrix.pkl")

    # Load metric DataFrames
    with open(data_path, "rb") as f:
        split_dfs = pickle.load(f)

    # Sort all columns by date for consistency
    h_df = sort_columns_by_date(split_dfs["h"])
    sigma_df = sort_columns_by_date(split_dfs["sigma"])
    q_df = sort_columns_by_date(split_dfs["q"])
    avg_daily_value_df = sort_columns_by_date(split_dfs["avg_daily_value"])

    # Apply a 10-day rolling average smoothing to volatility
    sigma_df = sigma_df.apply(lambda row: row.rolling(window=10, min_periods=1).mean(), axis=1)

    # Stack all metrics and join into a long-form DataFrame
    impact_df = pd.concat([
        h_df.stack().rename("h"),
        sigma_df.stack().rename("sigma"),
        q_df.stack().rename("q"),
        avg_daily_value_df.stack().rename("avg_daily_value")
    ], axis=1).reset_index()

    # Rename columns and clean up
    impact_df.columns = ['stock', 'date', 'h', 'sigma', 'q', 'avg_daily_value']
    impact_df = impact_df.sort_values(by="date").dropna().reset_index(drop=True)
    return impact_df

# Fit the impact model: h = eta * sigma * q^beta
# Returns estimated eta and beta from the given DataFrame
def estimate_impact_params_from_df(impact_df):
    """
    Estimate eta and beta parameters for the Almgren-Chriss impact model:
        h = eta * sigma * q^beta
    """
    # Flatten all values
    h_flat = impact_df['h'].values.flatten()
    sigma_flat = impact_df['sigma'].values.flatten()
    q_flat = impact_df['q'].values.flatten()

    # Filter invalid entries (NaNs, non-positive q)
    valid_mask = ~np.isnan(h_flat) & ~np.isnan(sigma_flat) & ~np.isnan(q_flat) & (q_flat > 0)
    h_flat = h_flat[valid_mask]
    sigma_flat = sigma_flat[valid_mask]
    q_flat = q_flat[valid_mask]

    # Define the nonlinear model
    def impact_model(q, eta, beta):
        return eta * sigma_flat * (q ** beta)

    # Initial guess for [eta, beta]
    initial_guess = [1.0, 0.5]

    # Fit model using SciPy’s curve_fit
    params, _ = curve_fit(impact_model, q_flat, h_flat, p0=initial_guess, maxfev=1000000)
    return params

# Main execution block
if __name__ == "__main__":

    # Step 1: Load and preprocess data
    impact_df = preprocess()

    # Step 2: Split into high vs low activity based on median ADV
    median_adv = impact_df['avg_daily_value'].median()
    high_activity_df = impact_df[impact_df['avg_daily_value'] > median_adv].copy()
    low_activity_df = impact_df[impact_df['avg_daily_value'] <= median_adv].copy()

    # Log the dataset sizes
    print(f"Total entries: {len(impact_df)}")
    print(f"High activity entries: {len(high_activity_df)}")
    print(f"Low activity entries: {len(low_activity_df)}")

    # Step 3: Estimate parameters separately for high and low activity stocks
    eta_hat_h, beta_hat_h = estimate_impact_params_from_df(high_activity_df)
    eta_hat_l, beta_hat_l = estimate_impact_params_from_df(low_activity_df)

    # Format results for printing and saving
    eta_text_h = f"Estimated eta (High activity)= {eta_hat_h:.6f}"
    beta_text_h = f"Estimated beta (High activity)= {beta_hat_h:.6f}"
    eta_text_l = f"Estimated eta (Low activity)= {eta_hat_l:.6f}"
    beta_text_l = f"Estimated beta (Low activity)= {beta_hat_l:.6f}"

    # Output results to console
    print("For High activity")
    print(eta_text_h)
    print(beta_text_h)

    print("For Low activity")
    print(eta_text_l)
    print(beta_text_l)

    # Step 4: Save estimates to file
    estimate_path = os.path.join(MyDirectories.getDataDir(), "regression_results_activity.txt")
    with open(estimate_path, "w") as f:
        f.write(eta_text_h + "\n")
        f.write(beta_text_h + "\n")
        f.write(eta_text_l + "\n")
        f.write(beta_text_l + "\n")
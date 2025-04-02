import sys
import os

# Add Directories to use folder as python modules to import code files.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Add Code directory

import pickle
import pandas as pd
import numpy as np
from taq.MyDirectories import MyDirectories
from scipy.optimize import curve_fit

from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white

# --- Utility to read eta and beta estimates from a text file ---
def read_eta_beta_from_file(file_path):
    """
    Reads eta and beta estimates from a text file with lines like:
        Estimated eta = 2.071382
        Estimated beta = 0.657832

    Args:
        file_path (str): Path to the saved estimate file

    Returns:
        tuple: (eta_hat, beta_hat) as floats
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Parse values from fixed format
    eta_hat = float(lines[0].strip().split("=")[-1])
    beta_hat = float(lines[1].strip().split("=")[-1])
    
    return eta_hat, beta_hat


# --- Sort DataFrame columns chronologically by date ---
def sort_columns_by_date(df):
    """
    Ensures DataFrame columns (dates) are sorted chronologically.
    
    Args:
        df (pd.DataFrame): DataFrame with date-like column headers
    
    Returns:
        pd.DataFrame: Reordered DataFrame
    """
    df.columns = pd.to_datetime(df.columns)
    return df.sort_index(axis=1)


# --- Prepare long-form market impact data for regression ---
def preprocess():
    """
    Preprocess stored input matrix into a long-form DataFrame for regression.
    
    Loads 'h', 'sigma', and 'q' matrices, applies rolling smoothing on volatility,
    stacks them, and returns a flat DataFrame with columns:
        ['stock', 'date', 'h', 'sigma', 'q']
    
    Returns:
        pd.DataFrame: Flattened and cleaned dataset
    
    Tested in part in: test_NLS.py and test_NLS_activity.py
    """
    # Construct path to stored metric DataFrames
    FINAL_DATA_DIR = os.path.join(MyDirectories.getDataDir(), "final_data_common")
    data_path = os.path.join(FINAL_DATA_DIR, "input_matrix.pkl")

    # Load stored dictionary of DataFrames: {"h", "sigma", "q", ...}
    with open(data_path, "rb") as f:
        split_dfs = pickle.load(f)

    # Sort columns in each matrix chronologically by date
    h_df = sort_columns_by_date(split_dfs["h"])
    sigma_df = sort_columns_by_date(split_dfs["sigma"])
    q_df = sort_columns_by_date(split_dfs["q"])

    # Smooth sigma using 10-day rolling average per stock
    sigma_df = sigma_df.apply(lambda row: row.rolling(window=10, min_periods=1).mean(), axis=1)

    # Stack (unpivot) each metric from wide format to long-form
    h_long = h_df.stack().rename("h")         # (stock, date)
    sigma_long = sigma_df.stack().rename("sigma")
    q_long = q_df.stack().rename("q")

    # Join all into a single DataFrame
    impact_df = pd.concat([h_long, sigma_long, q_long], axis=1).reset_index()
    impact_df.columns = ['stock', 'date', 'h', 'sigma', 'q']

    # Sort by date, drop missing rows, and reset index
    impact_df = impact_df.sort_values(by="date").dropna().reset_index(drop=True)
    
    return impact_df


def residual_bootstrap_analysis(h_flat, sigma_flat, q_flat, eta_hat, beta_hat, m=1000, plot_dir="bootstrap_plots"):
    '''
    Perform residual bootstrap for (η, β) parameter estimation.
    
    Args:
        h_flat (np.array): observed impact values
        sigma_flat (np.array): volatility values
        q_flat (np.array): trade size fraction values
        eta_hat (float): estimated η from original regression
        beta_hat (float): estimated β from original regression
        m (int): number of bootstrap samples
        plot_dir (str): directory to save plots

    Returns:
        dict: summary containing bootstrap stats, t-tests, and White test result
    
    Test - test_residual_bootstrap_analysis.py
    '''

    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Define the impact model function
    def impact_model(q, eta, beta):
        return eta * sigma_flat * (q ** beta)

    # --- Step 1: Calculate residuals from fitted model ---
    y_pred = impact_model(q_flat, eta_hat, beta_hat)
    residuals = h_flat - y_pred
    n = len(h_flat)

    # --- Step 2: Run m bootstrap iterations ---
    bootstrap_eta, bootstrap_beta = [], []

    for _ in range(m):
        # Resample residuals with replacement
        resampled_residuals = np.random.choice(residuals, size=n, replace=True)
        # Create synthetic response by adding resampled residuals to fitted values
        y_star = y_pred + resampled_residuals

        try:
            # Fit model to bootstrapped data and store estimates
            popt, _ = curve_fit(impact_model, q_flat, y_star, p0=(eta_hat, beta_hat), maxfev=10000)
            bootstrap_eta.append(popt[0])
            bootstrap_beta.append(popt[1])
        except RuntimeError:
            # Skip if the nonlinear fit fails
            continue

    # --- Step 3: Compute standard errors from bootstrap distributions ---
    eta_se = np.std(bootstrap_eta, ddof=1)
    beta_se = np.std(bootstrap_beta, ddof=1)

    # --- Step 4: t-statistics comparing bootstrap means to point estimates ---
    t_eta = (np.mean(bootstrap_eta) - eta_hat) / eta_se
    t_beta = (np.mean(bootstrap_beta) - beta_hat) / beta_se

    # --- Step 5: Plot bootstrap distributions for η ---
    plt.hist(bootstrap_eta, bins=40, alpha=0.7, color='skyblue')
    plt.axvline(eta_hat, color='red', linestyle='--', label='eta_hat')
    plt.title("Bootstrap Distribution of η")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "bootstrap_eta.png"))
    plt.clf()

    # --- Step 6: Plot bootstrap distributions for β ---
    plt.hist(bootstrap_beta, bins=40, alpha=0.7, color='orange')
    plt.axvline(beta_hat, color='red', linestyle='--', label='beta_hat')
    plt.title("Bootstrap Distribution of β")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "bootstrap_beta.png"))
    plt.clf()

    # --- Step 7: Residuals vs Fitted plot for diagnostics ---
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.savefig(os.path.join(plot_dir, "residuals_vs_fitted.png"))
    plt.clf()

    # --- Step 8: Residual distribution plot (hist + KDE) ---
    sns.histplot(residuals, bins=60, kde=True, color="teal")
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.axvline(0, color='black', linestyle='--')
    plt.savefig(os.path.join(plot_dir, "residual_distribution.png"))
    plt.clf()

    # --- Step 9: White's Test for Heteroskedasticity ---
    # Specification 1: Test against q only
    _, pval1, _, _ = het_white(
                        residuals, 
                        sm.add_constant(np.column_stack([q_flat, q_flat**2]))
                    )

    # Specification 2: Test against σ only 
    _, pval2, _, _ = het_white(
                        residuals, 
                        sm.add_constant(np.column_stack([sigma_flat, sigma_flat**2]))
                    )

    # Specification 3: Test against both q and σ (full model)
    X3 = sm.add_constant(np.column_stack([sigma_flat, q_flat]))
    X3_white = np.column_stack([X3, sigma_flat**2, q_flat**2, sigma_flat*q_flat])
    _, pval3, _, _ = het_white(residuals, X3_white)


    # --- Step 10: Return structured summary of bootstrap results ---
    summary = {
        "eta_hat": eta_hat,
        "beta_hat": beta_hat,
        "bootstrap_eta": np.mean(bootstrap_eta),
        "bootstrap_beta": np.mean(bootstrap_beta),
        "eta_se": eta_se,
        "beta_se": beta_se,
        "t_eta": t_eta,
        "t_beta": t_beta,
        "white_pvalue_q": pval1, 
        "white_pvalue_sigma": pval2, 
        "white_pvalue_both": pval3, 
    }

    # --- Final: Print summary and save diagnostics ---
    print("\n--- Residual Bootstrap Summary ---")
    print(f"eta_hat = {eta_hat:.4f}, SE = {eta_se:.4f}, t = {t_eta:.4f}")
    print(f"beta_hat = {beta_hat:.4f}, SE = {beta_se:.4f}, t = {t_beta:.4f}")
    # print(f"White's Test Statistic: {white_stat:.4f}, p-value: {white_pvalue:.4f} --> {'Homoskedastic' if white_pvalue >= 0.05 else 'Heteroskedastic'}")
    
    print(f"White test p-values:\n q only: {pval1:.4f}\n sigma only: {pval2:.4f}\n both model: {pval3:.4f}")
    print(f"Plots saved in: {plot_dir}")

    return summary


def paired_bootstrap_analysis(h_flat, sigma_flat, q_flat, eta_hat, beta_hat, m=1000, plot_dir="paired_bootstrap_plots"):
    '''
    Perform paired bootstrap on (h, σ, q) data to estimate variability of η and β.

    Args:
        h_flat (np.ndarray): observed impact values
        sigma_flat (np.ndarray): volatility values
        q_flat (np.ndarray): trade size fraction values
        eta_hat (float): point estimate for η from original regression
        beta_hat (float): point estimate for β from original regression
        m (int): number of bootstrap samples
        plot_dir (str): directory to store output plots

    Returns:
        dict: summary with η/β estimates, standard errors, t-stats, and White’s test
    '''

    # Ensure output directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Define original model using σ from original data
    def impact_model(q, eta, beta):
        return eta * sigma_flat * (q ** beta)

    # Generate predicted values from original fit
    y_pred = impact_model(q_flat, eta_hat, beta_hat)

    # --- Step 1: Combine inputs into a tuple list for resampling ---
    data = list(zip(h_flat, sigma_flat, q_flat))
    n = len(data)

    # Store bootstrap estimates
    bootstrap_eta, bootstrap_beta = [], []

    # --- Step 2–3: Run m paired bootstrap resamples ---
    for _ in range(m):
        # Resample full (h, σ, q) tuples with replacement
        resampled_data = [data[i] for i in np.random.randint(0, n, size=n)]
        h_star, sigma_star, q_star = zip(*resampled_data)
        h_star = np.array(h_star)
        sigma_star = np.array(sigma_star)
        q_star = np.array(q_star)

        # Define a new model using the resampled σ
        def model(q, eta, beta):
            return eta * sigma_star * (q ** beta)

        try:
            # Fit impact model on bootstrapped data
            popt, _ = curve_fit(model, q_star, h_star, p0=(eta_hat, beta_hat), maxfev=10000)
            bootstrap_eta.append(popt[0])
            bootstrap_beta.append(popt[1])
        except RuntimeError:
            continue  # Skip if curve_fit fails to converge

    # --- Step 4: Compute bootstrap-based standard errors ---
    eta_se = np.std(bootstrap_eta, ddof=1)
    beta_se = np.std(bootstrap_beta, ddof=1)

    # Compute t-statistics to evaluate stability of η and β
    t_eta = (np.mean(bootstrap_eta) - eta_hat) / eta_se
    t_beta = (np.mean(bootstrap_beta) - beta_hat) / beta_se

    # --- Step 5: Diagnostic Plots ---
    # Histogram of η estimates
    plt.hist(bootstrap_eta, bins=40, alpha=0.7, color='skyblue')
    plt.axvline(eta_hat, color='red', linestyle='--', label='eta_hat')
    plt.title("Paired Bootstrap Distribution of η")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "paired_bootstrap_eta.png"))
    plt.clf()

    # Histogram of β estimates
    plt.hist(bootstrap_beta, bins=40, alpha=0.7, color='orange')
    plt.axvline(beta_hat, color='red', linestyle='--', label='beta_hat')
    plt.title("Paired Bootstrap Distribution of β")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "paired_bootstrap_beta.png"))
    plt.clf()

    # Residual diagnostics from original fit
    residuals = h_flat - y_pred

    # Residuals vs Fitted values plot
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted (Paired Bootstrap)")
    plt.savefig(os.path.join(plot_dir, "paired_residuals_vs_fitted.png"))
    plt.clf()

    # Residual distribution histogram + KDE
    sns.histplot(residuals, bins=60, kde=True, color="teal")
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.axvline(0, color='black', linestyle='--')
    plt.savefig(os.path.join(plot_dir, "residual_distribution.png"))
    plt.clf()

    # --- Step 6: White's Test for Heteroskedasticity ---

    # Specification 1: Test against q only
    _, pval1, _, _ = het_white(
                        residuals, 
                        sm.add_constant(np.column_stack([q_flat, q_flat**2]))
                    )

    # Specification 2: Test against σ only 
    _, pval2, _, _ = het_white(
                        residuals, 
                        sm.add_constant(np.column_stack([sigma_flat, sigma_flat**2]))
                    )

    # Specification 3: Test against both q and σ (full model)
    X3 = sm.add_constant(np.column_stack([sigma_flat, q_flat]))
    X3_white = np.column_stack([X3, sigma_flat**2, q_flat**2, sigma_flat*q_flat])
    _, pval3, _, _ = het_white(residuals, X3_white)


    # --- Step 7: Package results ---
    summary = {
        "eta_hat": eta_hat,
        "beta_hat": beta_hat,
        "bootstrap_eta": np.mean(bootstrap_eta),
        "bootstrap_beta": np.mean(bootstrap_beta),
        "eta_se": eta_se,
        "beta_se": beta_se,
        "t_eta": t_eta,
        "t_beta": t_beta,
        "white_pvalue_q": pval1, 
        "white_pvalue_sigma": pval2, 
        "white_pvalue_both": pval3, 
    }

    # --- Final: Print and return results ---
    print("\n--- Paired Bootstrap Summary ---")
    print(f"eta_hat = {eta_hat:.4f}, SE = {eta_se:.4f}, t = {t_eta:.4f}")
    print(f"beta_hat = {beta_hat:.4f}, SE = {beta_se:.4f}, t = {t_beta:.4f}")
    print(f"White test p-values:\n q only: {pval1:.4f}\n sigma only: {pval2:.4f}\n both model: {pval3:.4f}")
    print(f"Plots saved in: {plot_dir}")

    return summary

def save_bootstrap_summary_txt(summary, file_path):
    """
    Save bootstrap regression summary to a .txt file in the format:
        eta = ...
        t-eta = ...
        beta = ...
        t-beta = ...
    
    Args:
        summary (dict): Dictionary containing keys "eta_hat", "beta_hat", "t_eta", "t_beta"
        file_path (str): Full path to the output .txt file
    """
    with open(file_path, "w") as f:
        f.write(f"eta = {summary['bootstrap_eta']:.6f}\n")
        f.write(f"t-eta = {summary['t_eta']:.6f}\n")
        f.write(f"beta = {summary['bootstrap_beta']:.6f}\n")
        f.write(f"t-beta = {summary['t_beta']:.6f}\n")

def save_full_bootstrap_summary(residual_summary, paired_summary, file_path):
    """
    Save all relevant fields from both residual and paired bootstrap summaries
    into a single .txt file.

    Args:
        residual_summary (dict): Output from residual_bootstrap_analysis
        paired_summary (dict): Output from paired_bootstrap_analysis
        file_path (str): Destination file path
    """
    def write_section(f, title, summary):
        f.write(f"[{title}]\n")
        f.write(f"eta = {summary['eta_hat']:.6f}\n")
        f.write(f"bootstrap_eta = {summary['bootstrap_eta']:.6f}\n")
        f.write(f"eta_se = {summary['eta_se']:.6f}\n")
        f.write(f"t-eta = {summary['t_eta']:.6f}\n")
        f.write(f"beta = {summary['beta_hat']:.6f}\n")
        f.write(f"bootstrap_beta = {summary['bootstrap_beta']:.6f}\n")
        f.write(f"beta_se = {summary['beta_se']:.6f}\n")
        f.write(f"t-beta = {summary['t_beta']:.6f}\n")
        f.write(f"white_pvalue_q = {summary['white_pvalue_q']:.6f}\n")
        f.write(f"white_pvalue_sigma = {summary['white_pvalue_sigma']:.6f}\n\n")
        f.write(f"white_pvalue_both = {summary['white_pvalue_both']:.6f}\n\n")

    with open(file_path, "w") as f:
        write_section(f, "RESIDUAL BOOTSTRAP", residual_summary)
        write_section(f, "PAIRED BOOTSTRAP", paired_summary)

if __name__ == "__main__":

    # Number of bootstrap samples to run
    n_bootstrap = 1000

    # --- Step 1: Load and flatten input data ---
    impact_df = preprocess()  # Loads cleaned h, sigma, q data (sorted by date)

    h_flat = impact_df['h'].values.flatten()
    sigma_flat = impact_df['sigma'].values.flatten()
    q_flat = impact_df['q'].values.flatten()

    # --- Step 2: Filter out invalid/missing entries (NaNs or q <= 0) ---
    valid_mask = ~np.isnan(h_flat) & ~np.isnan(sigma_flat) & ~np.isnan(q_flat) & (q_flat > 0)
    h_flat = h_flat[valid_mask]
    sigma_flat = sigma_flat[valid_mask]
    q_flat = q_flat[valid_mask]

    # --- Step 3: Load previously estimated η and β from file ---
    estimate_path = os.path.join(MyDirectories.getDataDir(), "regression_results.txt")
    eta_hat, beta_hat = read_eta_beta_from_file(estimate_path)

    # --- Step 4: Residual Bootstrap Analysis ---
    residual_summary = residual_bootstrap_analysis(
        h_flat=h_flat,
        sigma_flat=sigma_flat,
        q_flat=q_flat,
        eta_hat=eta_hat,
        beta_hat=beta_hat,
        m=n_bootstrap,
        plot_dir=os.path.join(MyDirectories.getDataDir(), "residual_plots")
    )

    save_path = os.path.join(MyDirectories.getDataDir(), "residual_summary.txt")
    save_bootstrap_summary_txt(residual_summary, save_path)

    # --- Step 5: Paired Bootstrap Analysis ---

    paired_summary = paired_bootstrap_analysis(
        h_flat=h_flat,
        sigma_flat=sigma_flat,
        q_flat=q_flat,
        eta_hat=eta_hat,
        beta_hat=beta_hat,
        m=n_bootstrap,
        plot_dir= os.path.join(MyDirectories.getDataDir(), "paired_bootstrap_output")
    )

    save_path = os.path.join(MyDirectories.getDataDir(), "paired_summary.txt")
    save_bootstrap_summary_txt(paired_summary, save_path)

    # Step 6: Save bootstrap analysis as required 
    summary_file_path = os.path.join(MyDirectories.getDataDir(), "bootstrap_full_summary.txt")
    save_full_bootstrap_summary(residual_summary, paired_summary, summary_file_path)
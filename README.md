# Market Impact Estimation and Bootstrap Analysis

This project implements a full pipeline for estimating market impact parameters using the Almgren-Chriss model with TAQ data. It includes preprocessing, metric computation, nonlinear regression, and bootstrap-based statistical inference.

---

## 📁 Folder Structure

<pre lang="md">
```
📁 Current Directory
├── 📁 Code/                                # Python scripts for regression and bootstrap analysis
│
├── Directory_Structure.txt                # Code folder layout description
├── Trading_Impact_Model.pdf               # Final report
│
├── 📁 Result_Generated/
│   ├── stock_selection_common.txt         # Selected common stocks
│   ├── top_1500_liquid_stocks.txt         # Top 1500 liquid stocks
│   ├── regression_results.txt             # Nonlinear regression output (η, β)
│   ├── regression_results_activity_more.txt  # High/low-activity segments
│   ├── bootstrap_full_summary.txt         # Bootstrap stats and t-tests
│   ├── paired_summary.txt                 # Paired bootstrap results
│   └── residual_summary.txt               # Residuals and White test p-values
```
</pre>


### 📁 Project Hierarchy (Simplified)
Impact Model/
├── Code/
│   ├── impactUtils/              # VWAP, Tick Test, Return Bucket utilities
│   ├── Project/                  # Scripts for data prep, regression, bootstrap
│   └── taq/                      # Readers for TAQ quotes/trades
├── Processed_Data/              # Output: matrices, plots, summaries
├── quotes/                      # Raw TAQ quotes
├── trades/                      # Raw TAQ trades

---

## 🔍 Highlights

- Nonlinear regression using `scipy.optimize.curve_fit`
- Residual and paired bootstrap for robust inference
- White's test for heteroskedasticity detection
- Liquidity-based segmentation for market regime analysis
- Modular and unit-tested codebase

---

## 📄 Report

Please refer to `Trading_Impact_Model.pdf` for a complete explanation of the methodology, results, and interpretation of the findings.

---

## 🛠 Prerequisites

Python 3.10 environment with the following packages:

- numpy  
- pandas  
- scipy  
- scikit-learn  
- statsmodels  
- matplotlib  
- seaborn  
- tqdm

Set up environment with:

```bash
conda create -n impact_model python=3.10
conda activate impact_model
conda install numpy pandas scikit-learn scipy statsmodels matplotlib seaborn tqdm
```
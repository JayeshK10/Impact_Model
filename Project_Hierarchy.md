
## 📁 Folder Structure

<pre lang="md">
```
Impact Model/
├── Code/
│   ├── impactUtils/
│   │   └── (Prof. Lee’s utility code for VWAP, Tick Test, ReturnBucket, FirstBucket, LastBucket, and their tests)
│   ├── Project/
│   │   ├── src/
│   │   │   0. unzip.sh
│   │   │      - Bash script to unzip quotes and trades directories into the desired location.
│   │   │
│   │   │   1. stock_code_all.py
│   │   │      - Loops through TAQ quotes and trades data to identify common stock codes.
│   │   │      - Filters out illiquid stocks.
│   │   │
│   │   │   2. ImpactMetrics.py
│   │   │      - Class that takes a stock and date, and outputs:
│   │   │         - Temporary impact (h)
│   │   │         - Volatility (σ)
│   │   │         - Trade size fraction (q = X / ((6/6.5) * V))
│   │   │         - Average daily value
│   │   │
│   │   │   3. data_preprocess.py
│   │   │      - Runs ImpactMetrics for all stock-date combinations.
│   │   │      - Saves checkpoints.
│   │   │      - Selects top 1500 stocks by average daily value.
│   │   │      - Filters out stocks/dates with missing data.
│   │   │
│   │   │   4. NLS.py
│   │   │      - Non-linear regression for estimating η (eta) and β (beta).
│   │   │
│   │   │   5. NLS_activity.py
│   │   │      - Part b (iii) of HW1.
│   │   │      - Analyzes less active vs more active stocks.
│   │   │
│   │   │   6. Bootstrap.py
│   │   │      - Performs:
│   │   │         a. Residual and paired bootstrap (value + t-values)
│   │   │         b. Statistical analysis:
│   │   │            i. Significance of parameters
│   │   │            ii. Assumptions of non-linear regression
│   │   │            iii. White’s test for heteroskedasticity
│   │
│   │   └── test/
│   │       - Unit tests for each module in src. (Detailed below)
│
│   └── taq/
│       1. TAQQuotesReader.py
│            - Read Quote files and load data. 
│       2. TAQTradesReader.py
│            - Read Trade files and load data. 
│       3. MyDirectories.py
│            - Path for required directories is defined here. 
│
├── Processed_Data/
│   - Stores checkpoints, final matrices, .txt outputs, plots, and all processed information.
│
├── quotes/
│   - Raw TAQ quotes data.
│
├── trades/
│   - Raw TAQ trades data.
```
</pre>
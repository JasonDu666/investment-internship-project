Investment Internship Project

This repository contains the work completed for my Python and data analysis internship.
The project focuses on financial data retrieval, cleaning, and strategy backtesting using QQQ historical price data.

Task 2 — Data Retrieval

The objective of Task 2 was to download historical price data using yfinance (with stooq as a fallback).
The script task-2/fetch_data.py retrieves daily OHLCV data for both QQQ and TQQQ within the following date range:

Start date: 2014-01-01

End date: 2024-01-01

The script outputs two CSV files stored in:

task-2/data/
  ├── QQQ.csv
  └── TQQQ.csv


These files contain the following fields:

open

close

high

low

volume

The virtual environment and other local files are excluded through .gitignore.

Dependencies can be installed with:

pip install yfinance pandas pandas-datareader

QQQ Analysis (Steps 3–7)

The file qqq_analysis.py implements Steps 3–7 of the project, including data cleaning and three investment strategies.

1. Data Loading and Preprocessing

Load QQQ price data from task-2/data/QQQ.csv

Convert the date column to datetime format and set it as the index

Filter the dataset to include only 2015-01-01 to 2025-01-01

Forward-fill missing values

Compute daily returns based on the closing price

2. Dollar-Cost Averaging (DCA)

Investing a fixed amount (1,000 USD) each month from 2015 to 2025.
Results:

Total cost: 108,000

Final value: 253,991.35

Profit: 145,991.35

ROI: 1.35

Total shares accumulated: 626.26

3. Buy-and-Hold Metrics

Assuming an initial capital of 100,000 USD:

Max drawdown: −35.12%

Annualized volatility: 22.21%

Sharpe ratio (risk-free rate 0): 0.84

4. Momentum Strategy (50-day Simple Moving Average)

Simple rule:
Hold QQQ when price is above the 50-day moving average; hold cash otherwise.

Results:

Final equity: 240,712.71

Max drawdown: −18.87%

Annualized volatility: 13.74%

Sharpe ratio (risk-free rate 0): 0.78

How to Run

To execute the QQQ analysis:

python qqq_analysis.py


This will generate all results for DCA, Buy-and-Hold, and Momentum strategies.

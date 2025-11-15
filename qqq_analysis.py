import pandas as pd

# 1. Load QQQ price data generated in Task 2
df = pd.read_csv("task-2/data/QQQ.csv")

# 2. Convert date column to datetime and set as index
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
df = df.sort_index()

# 3. Filter data between 2015-01-01 and 2025-01-01
df = df.loc["2015-01-01":"2025-01-01"]

# 4. Forward-fill missing values
df = df.ffill()

# 5. Compute daily returns
df["return"] = df["close"].pct_change()

print("Number of rows:", len(df))
print(df.head())
print(df.tail())


# ============================
# Strategy 1: Dollar-Cost Averaging (DCA)
# ============================
def strategy_dca(df, monthly_invest=1000):
    """
    Dollar-Cost Averaging strategy:
    Invest a fixed amount at the end of each month.
    """
    df_month = df.resample("ME").last()  # Month-end prices

    total_cost = 0
    total_shares = 0

    for _, row in df_month.iterrows():
        price = row["close"]
        shares = monthly_invest / price
        total_cost += monthly_invest
        total_shares += shares

    # Final portfolio value at the last available close
    final_value = total_shares * df["close"].iloc[-1]
    profit = final_value - total_cost
    roi = profit / total_cost

    return {
        "total_cost": total_cost,
        "final_value": final_value,
        "profit": profit,
        "roi": roi,
        "shares": total_shares,
        "months": len(df_month)
    }


# Run DCA strategy
dca_result = strategy_dca(df)
print("\n=== DCA Strategy Results ===")
for k, v in dca_result.items():
    print(f"{k}: {v}")


# ============================
# Performance Metrics
# ============================
def max_drawdown(equity):
    """
    Compute maximum drawdown.
    """
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    return drawdown.min()


def annualized_volatility(returns, periods_per_year=252):
    """
    Annualized volatility = std(returns) * sqrt(252)
    """
    return returns.std() * (periods_per_year ** 0.5)


def sharpe_ratio(returns, rf=0.0, periods_per_year=252):
    """
    Annualized Sharpe ratio (rf = 0 by default).
    """
    excess_return = returns - rf / periods_per_year
    return excess_return.mean() / excess_return.std() * (periods_per_year ** 0.5)


# Buy & Hold baseline (initial capital = $100,000)
initial_capital = 100_000
df["bh_equity"] = (1 + df["return"].fillna(0)).cumprod() * initial_capital

bh_returns = df["return"].dropna()
bh_mdd = max_drawdown(df["bh_equity"])
bh_vol = annualized_volatility(bh_returns)
bh_sharpe = sharpe_ratio(bh_returns)

print("\n=== Buy & Hold Performance Metrics ===")
print("Max Drawdown:", bh_mdd)
print("Volatility:", bh_vol)
print("Sharpe (rf=0):", bh_sharpe)


# ============================
# Strategy 2: Momentum Strategy
# ============================
def strategy_momentum(df, lookback=50, initial_capital=100_000):
    """
    Simple momentum strategy:
    - Hold QQQ when price > moving average.
    - Stay in cash when price <= moving average.
    """
    s = df.copy()

    # 1. Compute moving average
    s["ma"] = s["close"].rolling(lookback).mean()

    # 2. Position: 1 = long, 0 = cash
    s["position"] = 0
    s.loc[s["close"] > s["ma"], "position"] = 1

    # 3. Use previous day's position to avoid lookahead bias
    s["strategy_return"] = s["position"].shift(1) * s["return"]
    s["strategy_return"] = s["strategy_return"].fillna(0)

    # 4. Equity curve
    s["strategy_equity"] = (1 + s["strategy_return"]).cumprod() * initial_capital

    return s


# Run Momentum strategy
momentum_df = strategy_momentum(df, lookback=50, initial_capital=initial_capital)

mom_returns = momentum_df["strategy_return"].dropna()
mom_mdd = max_drawdown(momentum_df["strategy_equity"])
mom_vol = annualized_volatility(mom_returns)
mom_sharpe = sharpe_ratio(mom_returns)

print("\n=== Momentum Strategy (50-day MA) Metrics ===")
print("Final equity:", momentum_df["strategy_equity"].iloc[-1])
print("Max Drawdown:", mom_mdd)
print("Volatility:", mom_vol)
print("Sharpe (rf=0):", mom_sharpe)

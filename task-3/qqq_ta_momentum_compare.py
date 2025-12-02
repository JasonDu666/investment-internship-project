import pandas as pd
import numpy as np
import os

# -----------------------------
# Paths & constants
# -----------------------------
QQQ_DATA_PATH = "task-2/data/QQQ.csv"
FAANG_DIR = "task-3/data"
FAANG_TICKERS = ["AAPL", "AMZN", "GOOG", "META", "MSFT"]

START_DATE = "2015-01-01"
END_DATE = "2024-01-01"
RISK_FREE = 0.0  # rf = 0 as in previous project


# -----------------------------
# Helpers
# -----------------------------
def load_price_csv(path: str) -> pd.DataFrame:
    """
    Load a price CSV and return a numeric DataFrame indexed by date.
    This works for both QQQ (task-2) and FAANG (task-3) files.
    """
    df = pd.read_csv(path)

    # If we have a "date" column, use it as index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Force numeric for standard OHLCV columns
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort index and clip by date range using boolean masks
    df = df.sort_index()
    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)
    df = df[(df.index >= start) & (df.index <= end)]

    return df


def add_ma(df: pd.DataFrame, windows=(50, 200)) -> pd.DataFrame:
    """
    Add moving average columns maXX based on the 'close' price.
    """
    if "close" not in df.columns:
        raise ValueError("Expected a 'close' column in price DataFrame.")

    df = df.copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    for w in windows:
        df[f"ma{w}"] = df["close"].rolling(w).mean()

    return df


def compute_strategy_returns(price: pd.Series, signal: pd.Series) -> pd.Series:
    """
    Given daily close prices and a 0/1 position signal (1 = invested),
    compute daily strategy returns.
    """
    ret = price.pct_change().fillna(0.0)
    signal = signal.reindex(ret.index).fillna(0)
    strat_ret = ret * signal
    return strat_ret


def summarize_performance(daily_ret: pd.Series, name: str):
    """
    Print performance statistics for a daily return series:
    final equity (starting from 100k), total return, max drawdown, volatility, Sharpe.
    """
    initial_equity = 100_000.0
    equity = (1 + daily_ret).cumprod() * initial_equity

    total_return = equity.iloc[-1] / initial_equity - 1.0

    # Max drawdown
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = drawdown.min()

    # Annualized volatility & Sharpe (rf = 0)
    vol_ann = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else np.nan

    print(f"\n=== {name} ===")
    print(f"Final equity:\t\t{equity.iloc[-1]:,.2f} USD")
    print(f"Total return:\t\t{total_return*100:.2f}%")
    print(f"Max drawdown:\t\t{max_dd*100:.2f}%")
    print(f"Volatility (ann.):\t{vol_ann*100:.2f}%")
    print(f"Sharpe (rf=0):\t\t{sharpe:.3f}")


# -----------------------------
# Strategies
# -----------------------------
def build_baseline_momentum_signal(qqq_with_ma: pd.DataFrame) -> pd.Series:
    """
    Baseline momentum strategy:
    - Invest in QQQ when close > MA(50), otherwise stay in cash.
    """
    signal = (qqq_with_ma["close"] > qqq_with_ma["ma50"]).astype(int)
    signal.name = "baseline_ma50"
    return signal


def build_faang_risk_on_signal() -> pd.Series:
    """
    Build a FAANG-based 'risk-on' signal:
    - For each FAANG stock, compute 200-day MA.
    - For each day, check how many FAANG names are above their MA(200).
    - Risk-on = 1 if at least 3 out of 5 are above MA(200).
    """
    risk_flags = []

    for sym in FAANG_TICKERS:
        path = os.path.join(FAANG_DIR, f"{sym}.csv")
        df = load_price_csv(path)
        df = add_ma(df, windows=(200,))

        above_ma = (df["close"] > df["ma200"]).astype(int)
        above_ma.name = sym
        risk_flags.append(above_ma)

    combo = pd.concat(risk_flags, axis=1).fillna(0)
    count_above = combo.sum(axis=1)

    risk_on = (count_above >= 3).astype(int)
    risk_on.name = "risk_on_faang"

    return risk_on


def main():
    # 1) Load QQQ data and attach MA(50)
    qqq = load_price_csv(QQQ_DATA_PATH)
    qqq = add_ma(qqq, windows=(50,))  # adds column 'ma50'

    # 2) Baseline momentum strategy: MA(50) only
    baseline_signal = build_baseline_momentum_signal(qqq)
    baseline_ret = compute_strategy_returns(qqq["close"], baseline_signal)
    summarize_performance(baseline_ret, name="Baseline QQQ MA(50) Momentum")

    # 3) FAANG risk-on signal based on MA(200)
    risk_on = build_faang_risk_on_signal()
    risk_on = risk_on.reindex(qqq.index).fillna(0)

    # 4) TA-enhanced strategy:
    #    Invest in QQQ only when:
    #    - QQQ > MA(50), AND
    #    - FAANG risk-on == 1
    ta_signal = ((qqq["close"] > qqq["ma50"]) & (risk_on == 1)).astype(int)
    ta_signal.name = "ma50_plus_faang_filter"

    strat_ret_ta = compute_strategy_returns(qqq["close"], ta_signal)
    summarize_performance(
        strat_ret_ta,
        name="TA-Enhanced QQQ Momentum (MA50 + FAANG MA200 Filter)",
    )


if __name__ == "__main__":
    main()

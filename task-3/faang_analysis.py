import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# 基本配置
# ==========================

# 当前脚本所在目录，例如 .../investment-internship-project/task-3
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

FAANG = ["AAPL", "AMZN", "GOOG", "META", "MSFT"]
BENCHMARK = "SPY"

START_DATE = "2014-01-01"
END_DATE = "2024-01-01"

SHORT_MA = 50
LONG_MA = 200


# ==========================
# 工具函数
# ==========================

def load_price(symbol: str) -> pd.DataFrame:
    """
    从 task-3/data/{symbol}.csv 读取价格数据。
    你的 CSV 列结构是：
      symbol, date, open, high, low, close, adj_close, volume
    这里统一处理为：
      - index: date (datetime)
      - 列: open, high, low, close, adj_close, volume （float）
    """
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    # 处理日期列
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.sort_index()

    # 只保留我们需要的数值列，并强制转成数值，防止出现字符串
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[numeric_cols].dropna(subset=["close"])

    # 按日期范围裁剪一下（防止有多余数据）
    df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)]

    return df


def add_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    给单只股票增加技术指标：
      - MA50 / MA200
      - daily_return
      - log_return
    """
    df = df.copy()
    df["MA50"] = df["close"].rolling(SHORT_MA).mean()
    df["MA200"] = df["close"].rolling(LONG_MA).mean()
    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    return df


# ==========================
# 绘图函数
# ==========================

def plot_price_with_ma(symbol: str, df: pd.DataFrame):
    """
    绘制价格 + 均线图
    """
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["close"], label=f"{symbol} Close", linewidth=1)
    if "MA50" in df.columns:
        plt.plot(df.index, df["MA50"], label="MA 50", linewidth=1)
    if "MA200" in df.columns:
        plt.plot(df.index, df["MA200"], label="MA 200", linewidth=1)

    plt.title(f"{symbol} Price with {SHORT_MA}/{LONG_MA}-Day Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_correlation_heatmap(corr: pd.DataFrame):
    """
    绘制周收益率相关性热力图（FAANG + SPY）
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)

    plt.title("Weekly Return Correlation (FAANG + SPY)")
    plt.tight_layout()


def plot_cumulative_returns(stock_dfs: Dict[str, pd.DataFrame], benchmark_df: pd.DataFrame):
    """
    绘制 FAANG vs SPY 的累计收益曲线（基于 daily_return）
    """
    plt.figure(figsize=(10, 5))

    # 画 FAANG
    for sym, df in stock_dfs.items():
        cum = (1 + df["daily_return"].fillna(0)).cumprod()
        plt.plot(cum.index, cum.values, label=sym, linewidth=1)

    # 画基准 SPY
    spy_cum = (1 + benchmark_df["daily_return"].fillna(0)).cumprod()
    plt.plot(spy_cum.index, spy_cum.values, label=BENCHMARK + " (Benchmark)", linewidth=2)

    plt.title("Cumulative Return: FAANG vs SPY (2014–2024)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth (Base = 1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# ==========================
# 主流程
# ==========================

def main():
    # 1. 加载 FAANG 数据并增加技术指标
    stock_dfs: Dict[str, pd.DataFrame] = {}
    for sym in FAANG:
        df = load_price(sym)
        df = add_technical_columns(df)
        stock_dfs[sym] = df
        print(f"{sym} loaded: {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")

    # 2. 加载基准 SPY
    benchmark_df = load_price(BENCHMARK)
    benchmark_df = add_technical_columns(benchmark_df)
    print(f"{BENCHMARK} loaded: {len(benchmark_df)} rows, {benchmark_df.index.min().date()} → {benchmark_df.index.max().date()}")

    # 3. 每只股票画一个 价格+MA 图
    for sym, df in stock_dfs.items():
        plot_price_with_ma(sym, df)

    # 4. 计算周收益率相关性（FAANG + SPY）
    weekly_returns = {}
    for sym, df in stock_dfs.items():
        weekly_returns[sym] = df["close"].resample("W").last().pct_change()

    weekly_returns[BENCHMARK] = benchmark_df["close"].resample("W").last().pct_change()

    weekly_ret_df = pd.DataFrame(weekly_returns).dropna(how="any")
    corr = weekly_ret_df.corr()
    print("\nWeekly return correlation matrix:")
    print(corr)

    plot_correlation_heatmap(corr)

    # 5. 画累计收益曲线（FAANG vs SPY）
    plot_cumulative_returns(stock_dfs, benchmark_df)

    # 展示所有图
    plt.show()


if __name__ == "__main__":
    main()

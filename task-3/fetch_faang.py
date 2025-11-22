import os
import pandas as pd
import yfinance as yf

# ------------ 配置 ------------
START_DATE = "2014-01-01"
END_DATE = "2024-01-01"

# 5只股票 + SP500（这里用 SPY 作为指数代理）
TICKERS = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "SPY"]


def get_data_dir() -> str:
    """
    返回当前脚本所在目录下的 data 文件夹路径
    不受你在哪里运行 python 命令影响
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # task-3 这个目录
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    用 yfinance 下载一只股票的日线数据，并整理成统一格式的 DataFrame
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"Download failed for {ticker}")

    # 统一列名
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    # 去掉时区信息，重置索引
    df.index = df.index.tz_localize(None)
    df.reset_index(inplace=True)

    # 日期统一为字符串 yyyy-mm-dd
    df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["symbol"] = ticker

    # 挑选列并去掉缺失
    df = df[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]]
    df = df.dropna()

    return df


def main():
    data_dir = get_data_dir()
    total_rows = 0

    for t in TICKERS:
        df = fetch_one(t, START_DATE, END_DATE)
        out_path = os.path.join(data_dir, f"{t}.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK] {t}: {len(df)} rows -> {out_path}")
        total_rows += len(df)

    print(f"Done. Total rows: {total_rows}")


if __name__ == "__main__":
    main()

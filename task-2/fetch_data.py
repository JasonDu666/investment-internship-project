# task-2/fetch_data.py
import os
from datetime import datetime
import time
import pandas as pd

START_DATE = "2014-01-01"
END_DATE = "2024-01-01"
DATE_FORMAT = "%Y-%m-%d"
TICKERS = ["QQQ", "TQQQ"]

def fetch_with_yfinance_strict(ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:

    import yfinance as yf

    last_err = None
    for _ in range(max_retries):
        try:

            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if isinstance(df, pd.Series):
                df = df.to_frame().T
            if df is None or df.empty:
                raise RuntimeError("empty dataframe from yfinance")


            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()


            keep = ["Open", "Close", "High", "Low", "Volume"]
            df = df[keep]

            out = pd.DataFrame({
                "open": df["Open"],
                "close": df["Close"],
                "high": df["High"],
                "low": df["Low"],
                "volume": df["Volume"].astype("int64", errors="ignore"),
            })
            return out
        except Exception as e:
            last_err = e
            time.sleep(0.8)
    raise RuntimeError(f"yfinance failed: {last_err}")

def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = fetch_with_yfinance_strict(ticker, start, end)
    df = df.copy()
    df["symbol"] = ticker
    df["date"] = df.index.strftime(DATE_FORMAT)
    df = df[["symbol", "date", "open", "close", "high", "low", "volume"]]
    # 严格截取区间（右开）
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    df = df[(pd.to_datetime(df["date"]) >= start_dt) & (pd.to_datetime(df["date"]) < end_dt)]
    df = df.dropna()
    return df

def main():
    os.makedirs("data", exist_ok=True)
    all_rows = 0
    for t in TICKERS:
        df = fetch_one(t, START_DATE, END_DATE)
        out_path = f"data/{t}.csv"
        df.to_csv(out_path, index=False)
        all_rows += len(df)
        print(f"[OK] {t}: {len(df)} rows -> {out_path}")
    print(f"Done. Total rows: {all_rows}")

if __name__ == "__main__":
    main()

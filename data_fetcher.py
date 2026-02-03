import requests
import pandas as pd
import os
import datetime as dt
import time


# CONFIG

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
]


# HELPER FUNCTION
# =========================
def date_to_ms(date_str):
    """
    Convert date string YYYY-MM-DD to milliseconds
    """
    return int(dt.datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)


# DATA FETCHER
# =========================
def fetch_klines(symbol, interval="1d", start_date="2024-01-01", end_date="2026-01-26"):
    """
    Fetch historical daily candles.
    Handles Binance 1000-limit by looping through time.
    """
    start_time = date_to_ms(start_date)
    end_time = date_to_ms(end_date)

    all_data = []
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,
            "startTime": start_time
        }

        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        all_data += data

        last_open_time = data[-1][0]
        start_time = last_open_time + 1
        time.sleep(0.1)

    df = pd.DataFrame(all_data, columns=COLUMNS)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# SAVE FUNCTION
# =========================
def save_raw_data(df, filename):
    os.makedirs("data/raw", exist_ok=True)
    filepath = os.path.join("data/raw", filename)
    df.to_csv(filepath, index=False)
    print(f"Saved data to {filepath}")


# EXAMPLE USAGE
# =========================
if __name__ == "__main__":

    btc_df = fetch_klines(
        "BTCUSDT",
        interval="1d",
        start_date="2024-01-01",
        end_date="2026-01-26"
    )

    eth_df = fetch_klines(
        "ETHUSDT",
        interval="1d",
        start_date="2024-01-01",
        end_date="2026-01-26"
    )

  
    #  EXCLUDE LAST 27 ROWS for unseen data prediction
    # -------------------------
    btc_inference_df = btc_df.tail(27)
    eth_inference_df = eth_df.tail(27)

    btc_train_df = btc_df.iloc[:-27]
    eth_train_df = eth_df.iloc[:-27]

    # Save training data (unchanged logic)
    save_raw_data(btc_train_df, "btc_2y_daily.csv")
    save_raw_data(eth_train_df, "eth_2y_daily.csv")

    # Save inference data
    save_raw_data(btc_inference_df, "btc_inference_data.csv")
    save_raw_data(eth_inference_df, "eth_inference_data.csv")

    print("\nSummary:")
    print(f"BTC -> Train: {len(btc_train_df)} | Inference: {len(btc_inference_df)}")
    print(f"ETH -> Train: {len(eth_train_df)} | Inference: {len(eth_inference_df)}")


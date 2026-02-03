import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# =========================
# Preprocess for Price Prediction
# =========================
def preprocess_for_price(df, filename):
    """
    Preprocess ETH or BTC DataFrame for next-day price prediction.
    Creates features based on past prices, returns, moving averages, and past volatility.
    """
    selected_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    df = df[selected_cols].copy()

    # Convert timestamp
    df["open_time"] = pd.to_datetime(df["open_time"])
    df.sort_values("open_time", inplace=True)

    # Handle missing values
    df.fillna(method="ffill", inplace=True)

    # Target: next day's closing price
    df["target_price"] = df["close"].shift(-1)

    # Features: returns, moving averages, past volatility
    df["return_1d"] = df["close"].pct_change()  # daily return
    df["ma_3d"] = df["close"].rolling(3).mean().shift(1)  # past 3-day moving average
    df["ma_7d"] = df["close"].rolling(7).mean().shift(1)  # past 7-day moving average
    df["vol_3d"] = (df["high"] - df["low"]).rolling(3).mean().shift(1)  # past 3-day volatility

    # Drop rows with NaN (first few due to rolling, last due to target shift)
    df.dropna(inplace=True)

    # Save cleaned data
    filepath = f"data/cleaned/cleaned_{filename}_price.csv"
    os.makedirs("data/cleaned", exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved data to {filepath}")

    return df

# =========================
# 2️⃣ Prepare Data for Modeling
# =========================
def prepare_data_time_split(df, target_column, test_size=0.2, correlation_threshold=0.9, log_transform=True):
    """
    Prepares crypto time-series data for modeling:
    - Drops multicollinear features (excluding target)
    - Optionally applies log transformation to skewed features
    - Standard scales the features
    - Chronologically splits into train/test sets
    """

    # Exclude target from multicollinearity check
    feature_cols = df.columns.drop([target_column, 'open_time'])
    corr_matrix = df[feature_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > correlation_threshold)]

    # Drop only multicollinear feature columns
    df_reduced = df.drop(columns=to_drop)

    # Sort by time
    df_reduced = df_reduced.sort_values('open_time')

    # Drop 'open_time' before modeling
    df_reduced = df_reduced.drop(columns=["open_time"])

    # Split into features and target
    X = df_reduced.drop(columns=[target_column])
    y = df_reduced[target_column]

    # Apply log transformation safely (only positive features)
    if log_transform:
        log_cols = [col for col in X.columns if (X[col] > 0).all()]
        for col in log_cols:
            X[col] = np.log(X[col])

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Chronological train-test split
    split_index = int(len(X_scaled) * (1 - test_size))
    X_train, X_test = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test, scaler

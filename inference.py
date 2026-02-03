import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIG
# =========================
BTC_MODEL_PATH = "models/rf_btc_model.pkl"
ETH_MODEL_PATH = "models/rf_eth_model.pkl"
BTC_SCALER_PATH = "models/scaler_btc.pkl"
ETH_SCALER_PATH = "models/scaler_eth.pkl"

BTC_INFERENCE_DATA = "data/raw/btc_inference_data.csv"
ETH_INFERENCE_DATA = "data/raw/eth_inference_data.csv"

OUTPUT_DIR = "data/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features used in training
TRAIN_FEATURES = ['open', 'volume', 'return_1d', 'vol_3d']

# =========================
# FEATURE ENGINEERING (SAME AS TRAINING)
# =========================
def preprocess_for_price_inference(df):
    """
    Apply the SAME feature engineering as training (NO target).
    """
    selected_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    df = df[selected_cols].copy()

    df["open_time"] = pd.to_datetime(df["open_time"])
    df.sort_values("open_time", inplace=True)
    df.fillna(method="ffill", inplace=True)

    # Feature engineering
    df["return_1d"] = df["close"].pct_change()
    df["vol_3d"] = (df["high"] - df["low"]).rolling(3).mean().shift(1)

    df.dropna(inplace=True)

    # Subset only features used in training
    X = df[TRAIN_FEATURES].copy()

    return df, X

# =========================
# APPLY LOG + SCALING (SAME AS TRAINING)
# =========================
def transform_features(X, scaler):
    """
    Apply log transform (safe) + scaling using training scaler.
    """
    X = X.copy()
    for col in X.columns:
        if (X[col] > 0).all():  # only positive columns
            X[col] = np.log(X[col])

    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# =========================
# LOAD MODEL & SCALER
# =========================
def load_model_and_scaler(model_path, scaler_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

# =========================
# RUN INFERENCE
# =========================
def run_inference(asset_name, data_path, model_path, scaler_path):
    print(f"\nRunning inference for {asset_name}")

    # Load data
    df = pd.read_csv(data_path)

    # Preprocess and extract model features
    df_processed, X = preprocess_for_price_inference(df)

    # Load model & scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    # Transform features
    X_scaled = transform_features(X, scaler)

    # Predict
    predictions = model.predict(X_scaled)
    df_processed["predicted_next_close"] = predictions

    # Calculate metrics if true next-day close exists
    if "close" in df_processed.columns:
        y_true = df_processed["close"].shift(-1).iloc[:-1]  # align with predicted length
        y_pred = df_processed["predicted_next_close"].iloc[:-1]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print(f"Evaluation Metrics for {asset_name}:")
        print(f"MAE : {mae:.4f}")
        print(f"MSE : {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2  : {r2:.4f}")

    # Save predictions
    output_path = f"{OUTPUT_DIR}/{asset_name.lower()}_price_predictions.csv"
    df_processed.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Total predictions: {len(predictions)}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_inference(
        asset_name="BTC",
        data_path=BTC_INFERENCE_DATA,
        model_path=BTC_MODEL_PATH,
        scaler_path=BTC_SCALER_PATH
    )

    run_inference(
        asset_name="ETH",
        data_path=ETH_INFERENCE_DATA,
        model_path=ETH_MODEL_PATH,
        scaler_path=ETH_SCALER_PATH
    )

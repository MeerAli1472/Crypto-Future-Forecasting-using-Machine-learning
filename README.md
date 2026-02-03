# Crypto Price Prediction Pipeline Documentation

## Overview
This project implements a complete cryptocurrency price prediction pipeline using BTC and ETH historical data.
It covers data fetching, preprocessing, model training, evaluation, and inference.

--------------------------------------------------
# 1. data_fetch.py
Purpose:
Fetch historical OHLCV data from Binance API and save it as CSV.

Key Functions:
- date_to_ms(): Converts date to milliseconds.
- fetch_klines(): Fetches historical data with API limits handling.
- save_raw_data(): Saves raw data to CSV.

Usage:
python data_fetch.py

--------------------------------------------------
# 2. preprocess.py
Purpose:
Prepare raw data for modeling.

Key Functions:
- preprocess_for_price(): Feature engineering (returns, MA, volatility).
- prepare_data_time_split(): Scaling, correlation filtering, time-based split.

--------------------------------------------------
# 3. train.py
Purpose:
Train and evaluate regression models.

Models:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression

Evaluation Metrics:
- MAE
- MSE
- RMSE
- R² Score

--------------------------------------------------
# 4. inference.py
Purpose:
Predict future prices using trained models.

Key Steps:
- Load trained model and scaler
- Preprocess new data
- Generate predictions
- Visualize results

--------------------------------------------------
# 5. Project Workflow
1. Fetch data
2. Preprocess data
3. Train models
4. Save models
5. Run inference

--------------------------------------------------
# 6. Folder Structure
crypto_prediction/
│
├── data/
│   ├── raw/
│   └── cleaned/
├── models/
├── data_fetch.py
├── preprocess.py
├── train.py
├── inference.py

--------------------------------------------------
Notes:
- Always use time-based split for financial data.
- Use same scaler for training and inference.
- Avoid data leakage.


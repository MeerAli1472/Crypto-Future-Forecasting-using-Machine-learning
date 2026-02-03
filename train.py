import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")

# ----------------------------
#Fit Functions
# ----------------------------

def fit_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def fit_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def fit_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

# ----------------------------
#  Evaluation Function
# ----------------------------

def evaluate_model(model, X_test, y_test, plot=True):
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model.__class__.__name__} Evaluation:")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}\n")
    
    if plot:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title(f"{model.__class__.__name__} Predictions vs Actual")
        plt.show()
    
    return mae, mse, rmse, r2

# ----------------------------
# Train & Evaluate All Models
# ----------------------------

def train_all_models(X_train, X_test, y_train, y_test, plot=True):
    """
    Train Linear Regression, Decision Tree, Random Forest
    and evaluate them on test data.
    Returns a dictionary of results.
    """
    results = {}

    # Linear Regression
    lr_model = fit_linear_regression(X_train, y_train)
    results['LinearRegression'] = evaluate_model(lr_model, X_test, y_test, plot=plot)

    # Decision Tree
    dt_model = fit_decision_tree(X_train, y_train)
    results['DecisionTree'] = evaluate_model(dt_model, X_test, y_test, plot=plot)

    # Random Forest
    rf_model = fit_random_forest(X_train, y_train)
    results['RandomForest'] = evaluate_model(rf_model, X_test, y_test, plot=plot)

    return results


# Example Usage
# ----------------------------
if __name__ == "__main__":
    import pandas as pd
    from preprocess import preprocess_for_price, prepare_data_time_split

    # Load cleaned BTC/ETH price data
    btc_df = pd.read_csv("data/cleaned/cleaned_btc_price.csv")
    eth_df = pd.read_csv("data/cleaned/cleaned_eth_price.csv")

    # Prepare BTC data
    X_train_btc, X_test_btc, y_train_btc, y_test_btc, scaler_btc = prepare_data_time_split(
        btc_df, target_column="target_price"
    )

    # Train and evaluate BTC models
    print("===== BTC Models =====")
    btc_results = train_all_models(X_train_btc, X_test_btc, y_train_btc, y_test_btc, plot=True)

    # Prepare ETH data
    X_train_eth, X_test_eth, y_train_eth, y_test_eth, scaler_eth = prepare_data_time_split(
        eth_df, target_column="target_price"
    )

    # Train and evaluate ETH models
    print("===== ETH Models =====")
    eth_results = train_all_models(X_train_eth, X_test_eth, y_train_eth, y_test_eth, plot=True)

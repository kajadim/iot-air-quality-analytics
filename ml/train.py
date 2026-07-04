"""Train and evaluate models that predict PM2.5 / PM10 one hour ahead.

Candidates (scikit-learn), all compared against a persistence baseline
("next hour = current hour", which is surprisingly strong for air quality):

- Ridge regression        — linear reference point
- Random Forest           — bagged trees
- HistGradientBoosting    — gradient boosting, handles NaN natively
- MLPRegressor            — feed-forward neural network. An LSTM was
  considered for the time-series angle, but TensorFlow/Keras has no
  Python 3.14 build; the MLP over lag features covers the neural approach
  and gradient boosting is the usual winner on tabular sensor data anyway.

Split is TIME-BASED (first 80% of the timeline trains, last 20% tests) —
a random split would leak the future into training.

Outputs:
- ml_metrics table       : MAE/RMSE/R2 for every model and target
- ml_predictions table   : test-set predictions of the best model
- ml/models/<target>.joblib : fitted best model
"""

import os
import sqlite3
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_dataset, get_feature_columns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "air_quality.db")
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")

TARGETS = ["pm25_raw", "pm10_raw"]
TRAIN_FRACTION = 0.8


def make_models():
    imputer = SimpleImputer(strategy="median")
    return {
        "Ridge": Pipeline(
            [("imp", imputer), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]
        ),
        "RandomForest": Pipeline(
            [("imp", imputer),
             ("m", RandomForestRegressor(
                 n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42))]
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, random_state=42
        ),
        "MLP": Pipeline(
            [("imp", imputer), ("sc", StandardScaler()),
             ("m", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=60,
                                early_stopping=True, random_state=42))]
        ),
    }


def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def run_target(target_col: str, conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=== Target: {target_col} (t+1h) ===")
    df = build_dataset(target_col=target_col, horizon=1)
    feats = get_feature_columns(df, target_col)

    split_ts = df["timestamp"].quantile(TRAIN_FRACTION)
    train, test = df[df["timestamp"] <= split_ts], df[df["timestamp"] > split_ts]
    print(f"train: {len(train)} rows (do {split_ts:%Y-%m-%d}), test: {len(test)} rows")

    X_train, y_train = train[feats], train["target"]
    X_test, y_test = test[feats], test["target"]

    rows = []
    # persistence baseline: prediction = current value
    rows.append({"target": target_col, "model": "Persistence (baseline)",
                 **evaluate(y_test, test[target_col]), "train_seconds": 0.0})

    fitted = {}
    for name, model in make_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        metrics = evaluate(y_test, model.predict(X_test))
        rows.append({"target": target_col, "model": name, **metrics,
                     "train_seconds": round(elapsed, 1)})
        fitted[name] = model
        print(f"  {name:<22} MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
              f"R2={metrics['r2']:.3f}  ({elapsed:.0f}s)")

    metrics_df = pd.DataFrame(rows)
    trained_only = metrics_df[metrics_df["model"] != "Persistence (baseline)"]
    best_name = trained_only.sort_values("mae").iloc[0]["model"]
    best_model = fitted[best_name]
    print(f"  -> best model: {best_name}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, f"{target_col}.joblib"))

    preds = test[["device_id", "city", "timestamp"]].copy()
    preds["actual"] = y_test.values
    preds["predicted"] = best_model.predict(X_test)
    preds["target"] = target_col
    preds["model"] = best_name
    preds["timestamp"] = preds["timestamp"].dt.tz_convert("UTC").dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return metrics_df, preds


def main():
    conn = sqlite3.connect(DB_PATH)
    all_metrics, all_preds = [], []
    for target in TARGETS:
        metrics_df, preds = run_target(target, conn)
        all_metrics.append(metrics_df)
        all_preds.append(preds)

    pd.concat(all_metrics).to_sql("ml_metrics", conn, if_exists="replace", index=False)
    pd.concat(all_preds).to_sql("ml_predictions", conn, if_exists="replace", index=False)
    conn.close()

    print("\nSaved 'ml_metrics' and 'ml_predictions' tables; models in ml/models/")


if __name__ == "__main__":
    main()

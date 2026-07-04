"""Feature engineering for PM prediction.

Builds a supervised learning dataset from the hourly measurements:
for each device the series is regularized onto an hourly grid, then
lag/rolling features are computed strictly from the past, and the target
is the pollutant value `horizon` hours ahead.

Run directly to inspect the resulting feature matrix:
    python ml/features.py
"""

import os
import sqlite3

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "air_quality.db")

# columns brought in as predictors alongside the target pollutant
EXTRA_PREDICTORS = ["no2_raw", "temp_internal", "humidity_internal"]

LAGS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 24]


def load_hourly(target_col: str) -> pd.DataFrame:
    """Measurements joined with city, one row per device per hour (local time)."""
    cols = ["device_id", "timestamp", target_col] + EXTRA_PREDICTORS
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"""
            SELECT m.{', m.'.join(cols)}, s.city
            FROM measurements m
            JOIN sensors s ON m.device_id = s.device_id
            """,
            conn,
        )
    for c in [target_col] + EXTRA_PREDICTORS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Belgrade")
    )

    # regularize to an hourly grid per device so that shift(1) == "1 hour ago"
    df = (
        df.set_index("timestamp")
        .groupby(["device_id", "city"])[[target_col] + EXTRA_PREDICTORS]
        .resample("h")
        .mean()
        .reset_index()
    )
    return df


def build_dataset(target_col: str = "pm25_raw", horizon: int = 1) -> pd.DataFrame:
    """Feature matrix + `target` column (= target_col shifted `horizon` hours ahead).

    All features use only information available at prediction time.
    """
    df = load_hourly(target_col).sort_values(["device_id", "timestamp"])
    g = df.groupby("device_id")[target_col]

    for lag in LAGS:
        df[f"lag_{lag}h"] = g.shift(lag)
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}h"] = g.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        df[f"roll_std_{w}h"] = g.shift(1).rolling(w, min_periods=max(1, w // 2)).std()
    # change over the last 3 hours — captures rising/falling episodes
    df["delta_3h"] = df[target_col] - df["lag_3h"]

    # calendar features from LOCAL time (heating evenings, rush hours, seasons)
    hour = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    month = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_heating_season"] = month.isin([10, 11, 12, 1, 2, 3]).astype(int)

    df["target"] = g.shift(-horizon)

    # current value is itself a predictor; require it + target + first lag
    df = df.dropna(subset=[target_col, "target", "lag_1h"])
    return df.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame, target_col: str) -> list:
    exclude = {"device_id", "city", "timestamp", "target"}
    return [c for c in df.columns if c not in exclude]


def main():
    for target in ("pm25_raw", "pm10_raw"):
        df = build_dataset(target_col=target, horizon=1)
        feats = get_feature_columns(df, target)
        print(
            f"{target}: {len(df)} rows, {len(feats)} features, "
            f"{df['device_id'].nunique()} devices, "
            f"period {df['timestamp'].min()} -> {df['timestamp'].max()}"
        )
        missing_pct = df[feats].isna().mean().mean() * 100
        print(f"  avg missing across features: {missing_pct:.1f}%")
        print(f"  features: {feats}")


if __name__ == "__main__":
    main()

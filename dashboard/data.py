"""Data access layer for the Streamlit dashboard.

Pure pandas/SQLite — no Streamlit imports here, so everything is testable
from the command line. All timestamps leave this module in local time
(Europe/Belgrade).
"""

import os
import sqlite3

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "air_quality.db")

# label shown in the UI -> column in the measurements table
POLLUTANTS = {
    "PM2.5 [ug/m3]": "pm25_raw",
    "PM10 [ug/m3]": "pm10_raw",
    "PM1 [ug/m3]": "pm1_raw",
    "NO2 [ppb]": "no2_raw",
    "O3 [ppb]": "o3_raw",
    "Temperatura (interna) [degC]": "temp_internal",
    "Vlaznost (interna) [%]": "humidity_internal",
}

AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}


def _connect():
    return sqlite3.connect(DB_PATH)


def load_sensors() -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql("SELECT * FROM sensors", conn)


def load_measurements() -> pd.DataFrame:
    """All readings joined with city, timestamps converted to local time."""
    cols = ["device_id", "timestamp"] + list(POLLUTANTS.values())
    with _connect() as conn:
        df = pd.read_sql(
            f"""
            SELECT m.{', m.'.join(cols)}, s.city
            FROM measurements m
            JOIN sensors s ON m.device_id = s.device_id
            """,
            conn,
        )
    for col in POLLUTANTS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Belgrade")
    )
    return df


def load_aqi() -> pd.DataFrame:
    """Per-reading computed AQI (see analysis/aqi.py), local timestamps."""
    with _connect() as conn:
        df = pd.read_sql("SELECT * FROM aqi", conn)
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Belgrade")
    )
    return df


def load_daily_city_aqi() -> pd.DataFrame:
    with _connect() as conn:
        df = pd.read_sql("SELECT * FROM daily_city_aqi", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_ml_metrics() -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql("SELECT * FROM ml_metrics", conn)


def load_ml_predictions() -> pd.DataFrame:
    """Test-set predictions of the best model per target (see ml/train.py)."""
    with _connect() as conn:
        df = pd.read_sql("SELECT * FROM ml_predictions", conn)
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Belgrade")
    )
    return df


def load_anomalies() -> pd.DataFrame:
    with _connect() as conn:
        df = pd.read_sql("SELECT * FROM anomalies", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def sensor_summary(measurements: pd.DataFrame, aqi: pd.DataFrame,
                   sensors: pd.DataFrame, pollutant_col: str) -> pd.DataFrame:
    """Per-sensor average of the selected pollutant + avg/max AQI, for the map."""
    per_device = (
        measurements.groupby("device_id")
        .agg(value=(pollutant_col, "mean"), n_readings=(pollutant_col, "count"))
        .reset_index()
    )
    aqi_per_device = (
        aqi.groupby("device_id")
        .agg(avg_aqi=("aqi", "mean"), max_aqi=("aqi", "max"))
        .reset_index()
    )
    out = sensors.merge(per_device, on="device_id", how="left")
    out = out.merge(aqi_per_device, on="device_id", how="left")
    return out


def filter_period(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Filter by local dates (inclusive)."""
    ts = df["timestamp"]
    mask = (ts.dt.date >= start) & (ts.dt.date <= end)
    return df[mask]

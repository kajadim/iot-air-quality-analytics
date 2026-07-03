import glob
import os
import sqlite3

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_KG_DIR = os.path.join(BASE_DIR, "data", "raw", "kg")
RAW_NATIONAL_DIR = os.path.join(BASE_DIR, "data", "raw", "national")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

SQLITE_OUT = os.path.join(PROCESSED_DIR, "air_quality.db")

# ---------------------------------------------------------------------------
# Column mapping: original Clarity export column name -> english snake_case
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "Device ID": "device_id",
    "Time [UTC+00:00]": "timestamp",
    "PM1 1-Hour Mean Mass Concentration Calibrated [ug/m3]": "pm1_calibrated",
    "PM1 1-Hour Mean Mass Concentration Raw [ug/m3]": "pm1_raw",
    "PM1 1-Hour Mean Number Concentration [#/cm3]": "pm1_number_conc",
    "PM2.5 1-Hour Mean Mass Concentration Calibrated [ug/m3]": "pm25_calibrated",
    "PM2.5 1-Hour Mean Mass Concentration Raw [ug/m3]": "pm25_raw",
    "PM2.5 1-Hour Mean Number Concentration [#/cm3]": "pm25_number_conc",
    "PM2.5 NowCast Mass Concentration [ug/m3]": "pm25_nowcast",
    "PM2.5 NowCast AQI (US EPA)": "pm25_nowcast_aqi_epa",
    "PM2.5 24-Hour Rolling Mean Mass Concentration Calibrated [ug/m3]": "pm25_24h_calibrated",
    "PM2.5 24-Hour Rolling Mean Mass Concentration Raw [ug/m3]": "pm25_24h_raw",
    "PM2.5 24-Hour Rolling Mean Number Concentration [#/cm3]": "pm25_24h_number_conc",
    "PM2.5 1-Hour Mean AQI (WA DWER)": "pm25_aqi_dwer",
    "PM10 1-Hour Mean Mass Concentration Calibrated [ug/m3]": "pm10_calibrated",
    "PM10 1-Hour Mean Mass Concentration Raw [ug/m3]": "pm10_raw",
    "PM10 1-Hour Mean Number Concentration [#/cm3]": "pm10_number_conc",
    "PM10 24-Hour Rolling Mean Mass Concentration Calibrated [ug/m3]": "pm10_24h_calibrated",
    "PM10 24-Hour Rolling Mean Mass Concentration Raw [ug/m3]": "pm10_24h_raw",
    "PM10 24-Hour Rolling Mean Number Concentration [#/cm3]": "pm10_24h_number_conc",
    "PM10 1-Hour Mean AQI (WA DWER)": "pm10_aqi_dwer",
    "NO2 1-Hour Mean Concentration Calibrated [ppb]": "no2_calibrated",
    "NO2 1-Hour Mean Concentration Raw [ppb]": "no2_raw",
    "NO2 1-Hour Mean AQI (US EPA)": "no2_aqi_epa",
    "NO2 1-Hour Mean AQI (WA DWER)": "no2_aqi_dwer",
    "O3 1-Hour Mean Concentration Raw [ppb]": "o3_raw",
    "Temperature Internal 1-Hour Mean [degC]": "temp_internal",
    "Temperature Ambient 1-Hour Mean [degC]": "temp_ambient",
    "Rel. Humidity Internal 1-Hour Mean [%]": "humidity_internal",
    "Rel. Humidity Ambient 1-Hour Mean [%]": "humidity_ambient",
    "Wind Speed 1-Hour Mean [m/s]": "wind_speed",
    "Wind Direction 1-Hour Mean [degrees]": "wind_direction",
    "Atmospheric Pressure 1-Hour Mean [hPa]": "pressure",
    "latitude": "latitude",
    "longitude": "longitude",
}

FINAL_COLUMNS = list(dict.fromkeys(COLUMN_MAP.values())) + [
    "source_dataset",
    "source_file",
]


def load_folder(folder_path: str, source_dataset: str) -> pd.DataFrame:
    """Load and standardize all CSV files in a folder."""
    frames = []
    csv_paths = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    if not csv_paths:
        print(f"  (no CSV files found in {folder_path})")
        return pd.DataFrame(columns=FINAL_COLUMNS)

    for path in csv_paths:
        df = pd.read_csv(path)
        df = df.rename(columns=COLUMN_MAP)

        for col in COLUMN_MAP.values():
            if col not in df.columns:
                df[col] = pd.NA

        df["source_dataset"] = source_dataset
        df["source_file"] = os.path.basename(path)

        frames.append(df[FINAL_COLUMNS])
        print(f"  loaded {os.path.basename(path)}: {len(df)} rows")

    return pd.concat(frames, ignore_index=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    
    before = len(df)

    df["_completeness"] = df.notna().sum(axis=1)
    df = df.sort_values("_completeness", ascending=False)
    df = df.drop_duplicates(subset=["device_id", "timestamp"], keep="first")
    df = df.drop(columns="_completeness")

    after = len(df)
    print(f"  removed {before - after} duplicate rows ({before} -> {after})")
    return df

def drop_empty_columns(df: pd.DataFrame, essential_cols: list) -> pd.DataFrame:
   
    empty_cols = [
        col for col in df.columns
        if col not in essential_cols and df[col].isna().all()
    ]
 
    if empty_cols:
        print(f"  dropping {len(empty_cols)} fully-empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)
    else:
        print("  no fully-empty columns found")
 
    return df

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading Kragujevac (kg) files...")
    kg_df = load_folder(RAW_KG_DIR, source_dataset="kg")

    print("Loading national (all locations) files...")
    national_df = load_folder(RAW_NATIONAL_DIR, source_dataset="national")

    print("Merging datasets...")
    full_df = pd.concat([kg_df, national_df], ignore_index=True)
    print(f"  total rows before cleaning: {len(full_df)}")

    # Parse timestamp
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True, errors="coerce")

    # Drop rows with no timestamp or no device_id (unusable)
    full_df = full_df.dropna(subset=["timestamp", "device_id"])

    print("Removing duplicate readings...")
    full_df = deduplicate(full_df)

    print(f"Final dataset: {len(full_df)} rows, {full_df['device_id'].nunique()} unique devices")

    print("Checking for fully-empty columns...")
    essential_cols = ["device_id", "timestamp", "latitude", "longitude", "source_dataset", "source_file"]
    full_df = drop_empty_columns(full_df, essential_cols)
    
    # Save to SQLite (shared, queryable database)
    conn = sqlite3.connect(SQLITE_OUT)
    # sqlite doesn't support timezone-aware datetimes directly -> store as ISO string
    full_df_sql = full_df.copy()
    full_df_sql["timestamp"] = full_df_sql["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    full_df_sql.to_sql("measurements", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved SQLite -> {SQLITE_OUT}")


if __name__ == "__main__":
    main()
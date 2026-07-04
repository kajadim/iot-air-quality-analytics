import os
import sqlite3

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "air_quality.db")

NON_MEASUREMENT_COLUMNS = {
    "device_id", "timestamp", "latitude", "longitude",
    "source_dataset", "source_file",
}


def main():
    conn = sqlite3.connect(DB_PATH)

    existing_cols = pd.read_sql("PRAGMA table_info(measurements)", conn)["name"].tolist()
    agg_columns = [c for c in existing_cols if c not in NON_MEASUREMENT_COLUMNS]
    print(f"Aggregating {len(agg_columns)} pollutant/weather columns: {agg_columns}")

    query = f"""
    SELECT m.device_id, m.timestamp, s.city, {', '.join('m.' + c for c in agg_columns)}
    FROM measurements m
    JOIN sensors s ON m.device_id = s.device_id
    """
    df = pd.read_sql(query, conn)

    for col in agg_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # timestamps are stored as UTC ISO strings; daily/monthly buckets must use
    # local time so that day boundaries match what people in Serbia experience
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Belgrade")

    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")

    # --- Daily average per city ---
    daily = (
        df.groupby(["city", "date"])[agg_columns]
        .mean()
        .reset_index()
        .sort_values(["city", "date"])
    )
    daily.to_sql("daily_city_avg", conn, if_exists="replace", index=False)
    print(f"Saved 'daily_city_avg' -> {len(daily)} rows")

    # --- Monthly average per city ---
    monthly = (
        df.groupby(["city", "month"])[agg_columns]
        .mean()
        .reset_index()
        .sort_values(["city", "month"])
    )
    monthly.to_sql("monthly_city_avg", conn, if_exists="replace", index=False)
    print(f"Saved 'monthly_city_avg' -> {len(monthly)} rows")

    conn.close()

    print()
    print("Preview - daily_city_avg:")
    print(daily.head())


if __name__ == "__main__":
    main()
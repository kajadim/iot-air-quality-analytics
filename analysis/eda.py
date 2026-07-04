import os
import sqlite3

import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "air_quality.db")
EDA_DIR = os.path.join(BASE_DIR, "data", "processed", "eda")

# Metadata columns to exclude from numeric analysis
NON_MEASUREMENT_COLUMNS = {
    "device_id", "timestamp", "latitude", "longitude",
    "source_dataset", "source_file",
}


def load_data(conn):
    existing_cols = pd.read_sql("PRAGMA table_info(measurements)", conn)["name"].tolist()
    measure_cols = [c for c in existing_cols if c not in NON_MEASUREMENT_COLUMNS]

    query = f"""
    SELECT m.device_id, m.timestamp, s.city, {', '.join('m.' + c for c in measure_cols)}
    FROM measurements m
    JOIN sensors s ON m.device_id = s.device_id
    """
    df = pd.read_sql(query, conn, parse_dates=["timestamp"])
    for col in measure_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, measure_cols


def descriptive_stats(df, measure_cols):
    stats = df[measure_cols].describe().T
    stats["missing_pct"] = df[measure_cols].isna().mean() * 100
    return stats


def correlation_matrix(df, measure_cols):
    return df[measure_cols].corr()


def diurnal_pattern(df):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    return df.groupby("hour")[["pm25_raw", "pm10_raw"]].mean()


def seasonal_pattern(df):
    df = df.copy()
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    return df.groupby("month")[["pm25_raw", "pm10_raw"]].mean()


def city_ranking(df):
    return (
        df.groupby("city")["pm25_raw"]
        .mean()
        .sort_values(ascending=False)
        .rename("avg_pm25")
    )


def main():
    os.makedirs(EDA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    df, measure_cols = load_data(conn)
    conn.close()
    print(f"Loaded {len(df)} rows, {len(measure_cols)} measurement columns")

    # --- Descriptive statistics ---
    stats = descriptive_stats(df, measure_cols)
    stats.to_csv(os.path.join(EDA_DIR, "descriptive_stats.csv"))
    print("\n=== Descriptive statistics ===")
    print(stats[["mean", "std", "min", "max", "missing_pct"]].round(2))

    # --- Correlation matrix ---
    corr = correlation_matrix(df, measure_cols)
    corr.to_csv(os.path.join(EDA_DIR, "correlation_matrix.csv"))

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation matrix - pollutants & weather")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "correlation_matrix.png"), dpi=150)
    plt.close()

    # Highlight strongest correlations with PM2.5 (excluding itself)
    if "pm25_raw" in corr.columns:
        pm25_corr = corr["pm25_raw"].drop("pm25_raw").sort_values(ascending=False)
        print("\n=== Strongest correlations with pm25_raw ===")
        print(pm25_corr.round(2))

    # --- Diurnal pattern ---
    diurnal = diurnal_pattern(df)
    diurnal.to_csv(os.path.join(EDA_DIR, "diurnal_pattern.csv"))
    diurnal.plot(title="Average PM2.5 / PM10 by hour of day", figsize=(8, 5))
    plt.xlabel("hour")
    plt.ylabel("ug/m3")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "diurnal_pattern.png"), dpi=150)
    plt.close()

    # --- Seasonal pattern ---
    seasonal = seasonal_pattern(df)
    seasonal.to_csv(os.path.join(EDA_DIR, "seasonal_pattern.csv"))
    seasonal.plot(title="Average PM2.5 / PM10 by month", figsize=(10, 5))
    plt.xlabel("month")
    plt.ylabel("ug/m3")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "seasonal_pattern.png"), dpi=150)
    plt.close()

    # --- City ranking ---
    ranking = city_ranking(df)
    ranking.to_csv(os.path.join(EDA_DIR, "city_ranking.csv"))
    print("\n=== Cities ranked by average PM2.5 ===")
    print(ranking.round(2))

    print(f"\nAll EDA outputs saved to: {EDA_DIR}")


if __name__ == "__main__":
    main()
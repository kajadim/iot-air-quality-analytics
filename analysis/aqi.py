"""AQI computation from raw concentrations, per US EPA methodology.

AQI is COMPUTED here from pollutant concentrations using EPA breakpoint
tables + linear interpolation — the AQI columns that came with the Clarity
export are used only to cross-check this computation, never copied.

Notes:
- PM2.5 breakpoints are the pre-2024 EPA values, matching the era of the
  export (2022/23) so the cross-check against the export's own EPA AQI
  columns is apples-to-apples. (EPA lowered the "Good" PM2.5 range in 2024.)
- O3 is excluded: the EPA 1-hour O3 index is only defined for very high
  concentrations (>= 125 ppb) and o3_raw is ~86% missing in this dataset.
- Inputs use the 24-hour rolling raw PM columns (EPA AQI for PM is defined
  on 24h averages), falling back to the 1-hour raw value when 24h is null.

Outputs (tables in air_quality.db):
- aqi            : per reading — per-pollutant AQI, overall AQI, dominant
                   pollutant, category
- daily_city_aqi : daily max AQI per city (local Europe/Belgrade days)
"""

import os
import sqlite3

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "air_quality.db")

# (C_lo, C_hi, I_lo, I_hi) per EPA technical assistance document
BREAKPOINTS = {
    # ug/m3, 24-hour mean, truncated to 0.1 (pre-2024 table, see module docstring)
    "pm25": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ],
    # ug/m3, 24-hour mean, truncated to integer
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    # ppb, 1-hour mean, truncated to integer
    "no2": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
}

TRUNCATE_DECIMALS = {"pm25": 1, "pm10": 0, "no2": 0}

CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]


def concentration_to_aqi(values: pd.Series, pollutant: str) -> pd.Series:
    """Vectorized EPA linear interpolation for one pollutant."""
    decimals = TRUNCATE_DECIMALS[pollutant]
    # EPA prescribes truncation (not rounding) before lookup
    factor = 10 ** decimals
    conc = np.floor(values.astype(float) * factor) / factor

    aqi = pd.Series(np.nan, index=values.index)
    for c_lo, c_hi, i_lo, i_hi in BREAKPOINTS[pollutant]:
        mask = (conc >= c_lo) & (conc <= c_hi)
        aqi[mask] = (i_hi - i_lo) / (c_hi - c_lo) * (conc[mask] - c_lo) + i_lo
    # above the highest breakpoint -> cap at 500 (EPA: "Beyond the AQI")
    aqi[conc > BREAKPOINTS[pollutant][-1][1]] = 500
    return aqi.round()


def aqi_category(aqi: pd.Series) -> pd.Series:
    cat = pd.Series(pd.NA, index=aqi.index, dtype="object")
    for lo, hi, label in CATEGORIES:
        cat[(aqi >= lo) & (aqi <= hi)] = label
    return cat


def cross_check(computed: pd.Series, reference: pd.Series, name: str):
    """Compare our computed AQI against the AQI column shipped in the export."""
    ref = pd.to_numeric(reference, errors="coerce")
    both = computed.notna() & ref.notna()
    if both.sum() == 0:
        print(f"  {name}: no overlapping values to cross-check")
        return
    diff = (computed[both] - ref[both]).abs()
    print(
        f"  {name}: {both.sum()} readings compared | "
        f"MAE = {diff.mean():.2f} | within +/-1 point: {(diff <= 1).mean() * 100:.1f}%"
    )


def main():
    conn = sqlite3.connect(DB_PATH)

    existing_cols = pd.read_sql("PRAGMA table_info(measurements)", conn)["name"].tolist()

    wanted = [
        "device_id", "timestamp",
        "pm25_raw", "pm25_24h_raw", "pm10_raw", "pm10_24h_raw", "no2_raw",
        # export AQI columns, kept only for cross-checking (may not all exist)
        "pm25_nowcast", "pm25_nowcast_aqi_epa", "no2_aqi_epa",
    ]
    cols = [c for c in wanted if c in existing_cols]

    query = f"""
    SELECT m.{', m.'.join(cols)}, s.city
    FROM measurements m
    JOIN sensors s ON m.device_id = s.device_id
    """
    df = pd.read_sql(query, conn)
    for c in cols:
        if c not in ("device_id", "timestamp"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"Loaded {len(df)} readings")

    # --- compute per-pollutant AQI from concentrations ---
    # EPA PM AQI is defined on 24h means; fall back to the 1h raw value
    # where the 24h rolling column is missing
    pm25_conc = df["pm25_24h_raw"].fillna(df["pm25_raw"])
    pm10_conc = df["pm10_24h_raw"].fillna(df["pm10_raw"])

    df["pm25_aqi"] = concentration_to_aqi(pm25_conc, "pm25")
    df["pm10_aqi"] = concentration_to_aqi(pm10_conc, "pm10")
    df["no2_aqi"] = concentration_to_aqi(df["no2_raw"], "no2")

    pollutant_aqi = df[["pm25_aqi", "pm10_aqi", "no2_aqi"]]
    df["aqi"] = pollutant_aqi.max(axis=1)
    # idxmax raises on all-NaN rows -> compute only where at least one AQI exists
    has_any = pollutant_aqi.notna().any(axis=1)
    df["dominant_pollutant"] = pd.NA
    df.loc[has_any, "dominant_pollutant"] = (
        pollutant_aqi[has_any].idxmax(axis=1).str.replace("_aqi", "", regex=False)
    )
    df["aqi_category"] = aqi_category(df["aqi"])

    # --- cross-check against the export's own AQI columns ---
    print("\nCross-check vs AQI columns shipped in the Clarity export:")
    if "pm25_nowcast_aqi_epa" in df.columns and "pm25_nowcast" in df.columns:
        # NowCast AQI must be recomputed from the NowCast concentration
        nowcast_aqi = concentration_to_aqi(df["pm25_nowcast"], "pm25")
        cross_check(nowcast_aqi, df["pm25_nowcast_aqi_epa"], "PM2.5 NowCast AQI (EPA)")
    if "no2_aqi_epa" in df.columns:
        cross_check(df["no2_aqi"], df["no2_aqi_epa"], "NO2 1h AQI (EPA)")

    # --- save per-reading AQI table ---
    out_cols = [
        "device_id", "timestamp", "city",
        "pm25_aqi", "pm10_aqi", "no2_aqi",
        "aqi", "dominant_pollutant", "aqi_category",
    ]
    df[out_cols].to_sql("aqi", conn, if_exists="replace", index=False)
    print(f"\nSaved 'aqi' table -> {len(df)} rows")

    # --- daily max AQI per city (local days) ---
    local_ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Belgrade")
    df["date"] = local_ts.dt.date.astype(str)
    daily = (
        df.groupby(["city", "date"])
        .agg(
            aqi=("aqi", "max"),
            pm25_aqi=("pm25_aqi", "max"),
            pm10_aqi=("pm10_aqi", "max"),
            no2_aqi=("no2_aqi", "max"),
        )
        .reset_index()
    )
    daily["aqi_category"] = aqi_category(daily["aqi"])
    daily.to_sql("daily_city_aqi", conn, if_exists="replace", index=False)
    print(f"Saved 'daily_city_aqi' table -> {len(daily)} rows")

    conn.close()

    print("\nCategory distribution (per reading):")
    print(df["aqi_category"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

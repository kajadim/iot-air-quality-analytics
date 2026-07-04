"""Detection of extreme pollution episodes (anomalies) per city per day.

Two complementary criteria on daily city averages:

1. statistical — robust z-score (median/MAD, resistant to the outliers
   we are looking for) computed per city; |z| > 3 flags a day that is
   extreme relative to THAT city's own typical air.
2. threshold  — the day exceeds the EPA 24h "Unhealthy" concentration
   (PM2.5 > 55.4 ug/m3 or PM10 > 154 ug/m3), i.e. objectively bad air
   regardless of what is normal locally.

Output: 'anomalies' table in air_quality.db (used by the dashboard).
"""

import os
import sqlite3

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "air_quality.db")

Z_LIMIT = 3.0
PM25_UNHEALTHY_24H = 55.4  # ug/m3, EPA breakpoint (24h)
PM10_UNHEALTHY_24H = 154.0


def robust_z(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series(np.nan, index=series.index)
    # 0.6745 = scale factor so that MAD-z is comparable to a normal z-score
    return 0.6745 * (series - med) / mad


def main():
    conn = sqlite3.connect(DB_PATH)
    daily = pd.read_sql(
        "SELECT city, date, pm25_raw, pm10_raw FROM daily_city_avg", conn
    )

    daily["z_pm25"] = daily.groupby("city")["pm25_raw"].transform(robust_z)
    daily["z_pm10"] = daily.groupby("city")["pm10_raw"].transform(robust_z)

    daily["is_statistical"] = (
        (daily["z_pm25"].abs() > Z_LIMIT) | (daily["z_pm10"].abs() > Z_LIMIT)
    )
    daily["is_threshold"] = (
        (daily["pm25_raw"] > PM25_UNHEALTHY_24H)
        | (daily["pm10_raw"] > PM10_UNHEALTHY_24H)
    )

    anomalies = daily[daily["is_statistical"] | daily["is_threshold"]].copy()
    anomalies["severity"] = np.where(
        anomalies["is_statistical"] & anomalies["is_threshold"], "extreme",
        np.where(anomalies["is_threshold"], "unhealthy", "unusual"),
    )

    out_cols = ["city", "date", "pm25_raw", "pm10_raw",
                "z_pm25", "z_pm10", "is_statistical", "is_threshold", "severity"]
    anomalies[out_cols].to_sql("anomalies", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Analyzed {len(daily)} city-days -> {len(anomalies)} anomalous days")
    print("\nBy severity:")
    print(anomalies["severity"].value_counts().to_string())
    print("\nBy city:")
    print(anomalies["city"].value_counts().to_string())
    print("\nTop 10 worst days (by PM2.5):")
    top = anomalies.sort_values("pm25_raw", ascending=False).head(10)
    print(top[["city", "date", "pm25_raw", "pm10_raw", "severity"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()

import os
import sqlite3

import pandas as pd
import reverse_geocoder as rg

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "air_quality.db")


def main():
    conn = sqlite3.connect(DB_PATH)

    devices = pd.read_sql(
        "SELECT DISTINCT device_id, latitude, longitude FROM measurements",
        conn,
    )

    coords = list(zip(devices["latitude"], devices["longitude"]))
    print(f"Reverse geocoding {len(coords)} unique sensor coordinates...")
    results = rg.search(coords)

    devices["city"] = [r["name"] for r in results]
    devices["region"] = [r["admin1"] for r in results]
    devices["country_code"] = [r["cc"] for r in results]

    devices.to_sql("sensors", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Saved {len(devices)} sensors -> 'sensors' table in {DB_PATH}")
    print(devices.sort_values("city").to_string(index=False))


if __name__ == "__main__":
    main()
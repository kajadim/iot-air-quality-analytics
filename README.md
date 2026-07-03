# IoT Air Quality Analytics — Data Pipeline (Member 1)

This part of the project covers collecting, cleaning, and preparing the air quality data, which is then used by the dashboard and the ML models.

## 1. Data

The data was provided by the professor, collected from Clarity air quality sensors. Two sources:

- **`data/raw/kg/`** — monthly CSV files for Kragujevac only (2 sensors)
- **`data/raw/national/`** — CSV files for sensors across Serbia, exported at irregular (sometimes overlapping) time intervals

Files in both folders are the original, unmodified exports — the ETL scripts read them but never modify them.

## 2. Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Running the pipeline

The scripts must be run in this order (each one depends on the output of the previous one):

```bash
python etl/load_data.py        # loads, cleans, and merges all CSV files -> data/processed/air_quality.db
python etl/map_locations.py    # adds a 'sensors' table (device_id -> city), using reverse_geocoder
python etl/aggregate.py        # adds 'daily_city_avg' and 'monthly_city_avg' tables
```

The result is a single SQLite database: **`data/processed/air_quality.db`**

## 4. What each script does

### `etl/load_data.py`

- Loads all `.csv` files from `data/raw/kg` and `data/raw/national`
- Renames the original (long) column names to short, English, snake_case names (e.g. `PM2.5 1-Hour Mean Mass Concentration Raw [ug/m3]` → `pm25_raw`)
- Adds `source_dataset` (`"kg"` or `"national"`) and `source_file` (original file name) columns
- Parses the `timestamp` column and drops rows with no valid timestamp or `device_id`
- Removes duplicate readings (same `device_id` + `timestamp`), keeping the row with the most complete data when a duplicate exists
- **Automatically drops columns that are 100% empty** across the whole dataset (e.g. `pm25_calibrated`, `wind_speed`, `pressure` — this particular Clarity export never included them)
- Saves the result into the `measurements` table

### `etl/map_locations.py`

- Takes the unique `(device_id, latitude, longitude)` combinations from `measurements`
- Uses the `reverse_geocoder` library (offline, based on the GeoNames dataset) to assign a city to each location
- Saves the result into the `sensors` table

**Accuracy note:** `reverse_geocoder` assigns the **nearest known settlement in its database**, which for smaller towns isn't always fully precise (e.g. for a sensor actually located in Babusnica, the library returned the nearby town of Bela Palanka, since Babusnica isn't in its database). All locations were manually cross-checked against a more precise geographic search (Google Places).

### `etl/aggregate.py`

- Joins `measurements` with `sensors` on `device_id`
- Automatically aggregates all numeric measurement columns (excludes only `device_id`, `timestamp`, `latitude`, `longitude`, `source_dataset`, `source_file`)
- Produces two tables: `daily_city_avg` (average per city per day) and `monthly_city_avg` (average per city per month)

## 5. Database schema

| Table              | Description                                                               |
| ------------------ | ------------------------------------------------------------------------- |
| `measurements`     | Raw, cleaned readings — one row = one sensor reading at one point in time |
| `sensors`          | `device_id → latitude, longitude, city, region, country_code`             |
| `daily_city_avg`   | Average values per city, per day                                          |
| `monthly_city_avg` | Average values per city, per month                                        |

Key columns in `measurements` (after dropping empty ones): `pm1_raw`, `pm25_raw`, `pm25_24h_raw`, `pm25_aqi_dwer`, `pm10_raw`, `pm10_24h_raw`, `pm10_aqi_dwer`, `no2_raw`, `no2_aqi_epa`, `no2_aqi_dwer`, `o3_raw`, `temp_internal`, `humidity_internal`, `latitude`, `longitude`.

## 6. Known data limitations

- **Calibrated values (`*_calibrated`), ambient temperature/humidity, wind, and pressure are not available** in this export — those columns were completely empty and got dropped automatically. Use the `_raw` values for analysis/ML.
- **`o3_raw` is rarely populated** (~86% of readings are missing) — use with caution.
- **The `kg` and `national` datasets overlap by sensor** (the two Kragujevac devices exist in both), but not by time period — there were no true duplicate readings, just complementary time ranges.
- **Belgrade has multiple sensors spread across different municipalities** (Novi Beograd, Palilula, Stari Grad, Vracar) — the `sensors` table keeps them split by municipality rather than merging them into a single "Belgrade" row.

## 7. Folder structure

```
iot-air-quality-analytics/
├── data/
│   ├── raw/
│   │   ├── kg/
│   │   └── national/
│   └── processed/
│       └── air_quality.db
├── etl/
│   ├── load_data.py
│   ├── map_locations.py
│   └── aggregate.py
├── requirements.txt
├── .gitignore
└── README.md
```

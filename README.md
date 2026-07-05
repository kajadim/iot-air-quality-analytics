# IoT Air Quality Analytics

Analysis and prediction of air pollution in Serbia based on historical data from Clarity IoT air-quality sensors. The project covers the full pipeline: **ETL** (loading and cleaning raw CSV exports) → **statistical analysis + EPA AQI computation** → **interactive Streamlit dashboard** → **ML prediction** of PM2.5/PM10 one hour ahead, with anomaly detection.

## Features

- **ETL pipeline** — loads 31 raw CSV exports, standardizes column names, parses timestamps (UTC), deduplicates overlapping exports (~33k duplicates removed), drops fully-empty columns, and writes everything into a single SQLite database (~136k measurements, 17 sensors, Feb 2022 – Mar 2023)
- **Geocoding** — assigns a city to each sensor from its GPS coordinates (offline `reverse_geocoder`, GeoNames)
- **EDA** — descriptive statistics, correlation matrix, diurnal and seasonal patterns, city ranking
- **AQI** — Air Quality Index computed from concentrations using the official US EPA methodology (breakpoint tables + linear interpolation), cross-validated against the AQI columns shipped in the export (100% agreement within ±1 point on ~110k readings)
- **Dashboard** — interactive Streamlit app: sensor map, time series, AQI analysis, ML predictions
- **ML** — PM2.5/PM10 prediction one hour ahead (4 models vs. a persistence baseline, time-based split) and detection of extreme pollution episodes

## Tech stack

| | |
|---|---|
| Language | Python |
| Data processing | pandas, numpy |
| Storage | SQLite (single database file, all pipeline stages communicate through its tables) |
| Dashboard | Streamlit, Folium (interactive map), Plotly (charts) |
| ML | scikit-learn |
| Geocoding | reverse_geocoder (offline, GeoNames) |

## Project structure

```
etl/          load_data.py, map_locations.py, aggregate.py   — CSV → SQLite, geocoding, aggregates
analysis/     eda.py, aqi.py                                 — statistics, EPA AQI + cross-check
dashboard/    app.py (Streamlit UI), data.py (data access)   — interactive dashboard
ml/           features.py, train.py, anomalies.py            — feature engineering, models, anomalies
data/         raw/ and processed/ (not in git)
```

## Data

The `data/` directory is **not** in git. Before running the pipeline, place the raw CSV exports locally:

- `data/raw/kg/` — monthly files for Kragujevac (2 sensors)
- `data/raw/national/` — exports for sensors across Serbia (irregular, partly overlapping intervals)

Raw files are never modified — everything derived is written to `data/processed/air_quality.db`.

## Getting started

```bash
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# 1) ETL — must run in this order
python etl/load_data.py        # CSVs -> 'measurements' table
python etl/map_locations.py    # + 'sensors' table (reverse geocoding)
python etl/aggregate.py        # + daily/monthly per-city aggregates

# 2) Analysis
python analysis/eda.py         # statistics & plots -> data/processed/eda/
python analysis/aqi.py         # + 'aqi' and 'daily_city_aqi' tables (US EPA)

# 3) ML
python ml/train.py             # trains & compares models -> ml_metrics, ml_predictions
python ml/anomalies.py         # extreme episodes -> 'anomalies' table

# 4) Dashboard
streamlit run dashboard/app.py
```

## Dashboard

Four tabs with shared sidebar filters (period, locations, pollutant):

1. **Sensor map** (Folium) — markers colored by average AQI category, sized by the average value of the selected pollutant
2. **Time series** (Plotly) — hourly/daily/monthly series per city + diurnal profile by local hour
3. **AQI analysis** — daily max AQI heatmap (city × date), category distribution per city, dominant pollutant
4. **ML prediction** — model comparison, predicted vs. actual per sensor on the test period, detected anomalies

## ML results

Time-based 80/20 split (random splitting would leak future data into training). All models are compared against a persistence baseline ("next hour = current hour"). Test-period results (t+1h):

| Model | PM2.5 MAE | PM2.5 R² | PM10 MAE | PM10 R² |
| --- | --- | --- | --- | --- |
| Persistence (baseline) | 8.30 | 0.810 | 11.59 | 0.802 |
| Ridge | 8.26 | 0.830 | 11.63 | 0.823 |
| Random Forest | 7.85 | 0.842 | **11.02** | 0.833 |
| HistGradientBoosting | 7.93 | 0.841 | 11.05 | 0.830 |
| **MLP (neural network)** | **7.85** | **0.843** | 11.24 | 0.836 |

Best models (by MAE): **MLP for PM2.5, Random Forest for PM10** — saved to `ml/models/*.joblib`.

Anomaly detection combines a robust statistical criterion (median/MAD z-score per city, |z| > 3) with an absolute EPA threshold (daily mean PM2.5 > 55.4 or PM10 > 154 µg/m³): **443 anomalous city-days**, overwhelmingly in the heating season (worst: Bela Palanka, 2022-12-16, PM2.5 = 216 µg/m³).

## Design decisions

- **SQLite as the single artifact** — one file, zero administration, SQL queries; every script reads from and writes to the same database, and all tables are written with `if_exists="replace"`, so reruns are idempotent.
- **AQI is computed, not copied** — the export's own AQI columns are used only to cross-validate our EPA implementation, never copied into results.
- **UTC storage, local-time analysis** — timestamps are stored as UTC ISO-8601 strings; every consumer converts to `Europe/Belgrade` before deriving hours/days/months, because diurnal and daily patterns follow local human behavior (heating, traffic).
- **Time-based train/test split** — the first 80% of the timeline trains, the last 20% tests; a random split would leak the future into training.
- **No MQTT / live ingestion (by design)** — the input is a static, already-exported historical dataset. In a production IoT setup a `sensor → MQTT broker → ingestion service` chain would feed the same `measurements` table; with no live data it would be an artificial layer with no function.
- **MLP instead of LSTM** — TensorFlow/Keras has no build for Python 3.14; an MLP over lag features covers the neural approach, and tree ensembles are the usual winners on tabular sensor data anyway.

## Known data limitations

- Calibrated values (`*_calibrated`), ambient temperature/humidity, wind, and pressure are entirely empty in this export (auto-dropped) — analysis uses the `_raw` columns.
- `o3_raw` is ~86% missing — excluded from AQI, used with caution elsewhere.
- City names come from the nearest-settlement match in GeoNames and can be imprecise for small towns (e.g., a sensor in Babušnica resolves to Bela Palanka).
- Belgrade sensors are kept split by municipality (Novi Beograd, Palilula, Stari Grad, Vračar), not merged.
- The dataset covers ~13 months, so seasonal conclusions rest on a single winter.

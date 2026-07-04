"""IoT Air Quality dashboard (Streamlit + Folium + Plotly).

Run with:  streamlit run dashboard/app.py
"""

import os
import sys

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as d

st.set_page_config(page_title="Kvalitet vazduha — Srbija", layout="wide")


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data
def get_sensors():
    return d.load_sensors()


@st.cache_data
def get_measurements():
    return d.load_measurements()


@st.cache_data
def get_aqi():
    return d.load_aqi()


@st.cache_data
def get_daily_city_aqi():
    return d.load_daily_city_aqi()


sensors = get_sensors()
measurements = get_measurements()
aqi = get_aqi()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filteri")

min_date = measurements["timestamp"].dt.date.min()
max_date = measurements["timestamp"].dt.date.max()

period = st.sidebar.date_input(
    "Period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
start, end = (period[0], period[1]) if len(period) == 2 else (period[0], max_date)

all_cities = sorted(sensors["city"].unique())
cities = st.sidebar.multiselect("Lokacije", all_cities, default=all_cities)

pollutant_label = st.sidebar.selectbox("Zagadjivac / velicina", list(d.POLLUTANTS))
pollutant_col = d.POLLUTANTS[pollutant_label]

m_filt = d.filter_period(measurements, start, end)
m_filt = m_filt[m_filt["city"].isin(cities)]
aqi_filt = d.filter_period(aqi, start, end)
aqi_filt = aqi_filt[aqi_filt["city"].isin(cities)]

st.title("Kvalitet vazduha u Srbiji — Clarity senzori")
st.caption(
    f"{len(m_filt):,} merenja | {m_filt['device_id'].nunique()} senzora | "
    f"period {start} — {end} (lokalno vreme, Europe/Belgrade)"
)

tab_map, tab_series, tab_aqi, tab_ml = st.tabs(
    ["Mapa senzora", "Vremenske serije", "AQI analiza", "ML predikcija"]
)

# ---------------------------------------------------------------------------
# Tab 1: sensor map
# ---------------------------------------------------------------------------
with tab_map:
    st.subheader(f"Prosecno {pollutant_label} po senzoru u izabranom periodu")

    summary = d.sensor_summary(m_filt, aqi_filt, sensors, pollutant_col)
    summary = summary[summary["city"].isin(cities)]

    fmap = folium.Map(location=[44.2, 20.9], zoom_start=7, tiles="cartodbpositron")

    def aqi_color(avg_aqi):
        if pd.isna(avg_aqi):
            return "#999999"
        for bound, label in [(50, "Good"), (100, "Moderate"),
                             (150, "Unhealthy for Sensitive Groups"),
                             (200, "Unhealthy"), (300, "Very Unhealthy")]:
            if avg_aqi <= bound:
                return d.AQI_COLORS[label]
        return d.AQI_COLORS["Hazardous"]

    vmax = summary["value"].max()
    for _, row in summary.iterrows():
        # color = sensor's average AQI category, size = avg of selected pollutant
        color = aqi_color(row["avg_aqi"])
        if pd.isna(row["value"]):
            radius, value_txt = 6, "(nema merenja)"
        else:
            radius = 6 + 14 * (row["value"] / vmax if vmax else 0)
            value_txt = f"{pollutant_label}: {row['value']:.1f}"

        popup = folium.Popup(
            f"<b>{row['city']}</b><br>Uredjaj: {row['device_id']}<br>{value_txt}",
            max_width=250,
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=popup,
            tooltip=f"{row['city']} ({row['device_id']})",
        ).add_to(fmap)

    st_folium(fmap, height=520, use_container_width=True, returned_objects=[])

    st.markdown(
        "Boja markera = prosecna AQI kategorija senzora u periodu "
        "(zelena=Good ... bordo=Hazardous), velicina = prosecna vrednost "
        "izabranog zagadjivaca."
    )
    st.dataframe(
        summary[["device_id", "city", "value", "n_readings", "avg_aqi", "max_aqi"]]
        .rename(columns={"value": pollutant_label})
        .sort_values(pollutant_label, ascending=False),
        width="stretch",
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Tab 2: time series
# ---------------------------------------------------------------------------
with tab_series:
    st.subheader(f"{pollutant_label} kroz vreme")

    granularity = st.radio(
        "Granularnost", ["Sat", "Dan", "Mesec"], horizontal=True
    )
    rule = {"Sat": "h", "Dan": "D", "Mesec": "MS"}[granularity]

    series = (
        m_filt.set_index("timestamp")
        .groupby("city")[pollutant_col]
        .resample(rule)
        .mean()
        .reset_index()
    )

    if series[pollutant_col].notna().sum() == 0:
        st.info("Nema podataka za izabrani filter.")
    else:
        fig = px.line(
            series,
            x="timestamp",
            y=pollutant_col,
            color="city",
            labels={"timestamp": "vreme", pollutant_col: pollutant_label, "city": "grad"},
        )
        fig.update_layout(height=480, legend_title_text="")
        st.plotly_chart(fig, width="stretch")

    # diurnal profile (uses local hour — that's why UTC->local conversion matters)
    st.subheader("Dnevni profil (prosek po satu u danu)")
    diurnal = (
        m_filt.assign(sat=m_filt["timestamp"].dt.hour)
        .groupby(["city", "sat"])[pollutant_col]
        .mean()
        .reset_index()
    )
    fig2 = px.line(
        diurnal, x="sat", y=pollutant_col, color="city",
        labels={"sat": "sat u danu (lokalno)", pollutant_col: pollutant_label, "city": "grad"},
    )
    fig2.update_layout(height=380, legend_title_text="")
    st.plotly_chart(fig2, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3: AQI analysis
# ---------------------------------------------------------------------------
with tab_aqi:
    st.subheader("AQI (izracunat iz koncentracija, US EPA)")

    daily_aqi = get_daily_city_aqi()
    daily_aqi = daily_aqi[
        (daily_aqi["date"].dt.date >= start)
        & (daily_aqi["date"].dt.date <= end)
        & (daily_aqi["city"].isin(cities))
    ]

    if daily_aqi.empty:
        st.info("Nema AQI podataka za izabrani filter.")
    else:
        fig = px.density_heatmap(
            daily_aqi, x="date", y="city", z="aqi",
            histfunc="max",
            color_continuous_scale=[
                (0.0, "#00e400"), (0.2, "#ffff00"), (0.4, "#ff7e00"),
                (0.6, "#ff0000"), (0.8, "#8f3f97"), (1.0, "#7e0023"),
            ],
            range_color=[0, 250],
            labels={"date": "datum", "city": "grad", "aqi": "dnevni max AQI"},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Raspodela AQI kategorija po gradu (broj dana)")
        order = list(d.AQI_COLORS)
        counts = (
            daily_aqi.groupby(["city", "aqi_category"], observed=True)
            .size()
            .reset_index(name="dana")
        )
        fig3 = px.bar(
            counts, x="city", y="dana", color="aqi_category",
            category_orders={"aqi_category": order},
            color_discrete_map=d.AQI_COLORS,
            labels={"city": "grad", "dana": "broj dana", "aqi_category": "kategorija"},
        )
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, width="stretch")

        st.subheader("Dominantni zagadjivac (po ocitavanju)")
        dom = (
            aqi_filt.dropna(subset=["dominant_pollutant"])
            .groupby(["city", "dominant_pollutant"], observed=True)
            .size()
            .reset_index(name="ocitavanja")
        )
        fig4 = px.bar(
            dom, x="city", y="ocitavanja", color="dominant_pollutant",
            labels={"city": "grad", "ocitavanja": "broj ocitavanja",
                    "dominant_pollutant": "zagadjivac"},
        )
        fig4.update_layout(height=380)
        st.plotly_chart(fig4, width="stretch")

# ---------------------------------------------------------------------------
# Tab 4: ML predictions & anomalies
# ---------------------------------------------------------------------------
with tab_ml:
    st.subheader("Predikcija PM2.5 / PM10 jedan sat unapred")

    try:
        ml_metrics = d.load_ml_metrics()
        ml_preds = d.load_ml_predictions()
        anomalies = d.load_anomalies()
    except Exception:
        st.warning(
            "ML rezultati jos nisu izracunati. Pokreni: "
            "`python ml/train.py` i `python ml/anomalies.py`."
        )
        st.stop()

    target_label = st.radio(
        "Ciljna velicina", ["PM2.5", "PM10"], horizontal=True, key="ml_target"
    )
    target_col = {"PM2.5": "pm25_raw", "PM10": "pm10_raw"}[target_label]

    st.markdown("**Poredjenje modela** (vremenski split: poslednjih ~20% za test)")
    mt = ml_metrics[ml_metrics["target"] == target_col]
    st.dataframe(
        mt[["model", "mae", "rmse", "r2", "train_seconds"]].round(3),
        width="stretch", hide_index=True,
    )
    best_model_name = ml_preds.loc[ml_preds["target"] == target_col, "model"].iloc[0]
    st.caption(f"Najbolji model (najmanji MAE): **{best_model_name}** — "
               "grafik ispod prikazuje njegove predikcije na test periodu.")

    preds = ml_preds[
        (ml_preds["target"] == target_col) & (ml_preds["city"].isin(cities))
    ]
    pred_city = st.selectbox("Grad", sorted(preds["city"].unique()))
    device_opts = sorted(preds.loc[preds["city"] == pred_city, "device_id"].unique())
    pred_device = st.selectbox("Senzor", device_opts)

    p = preds[preds["device_id"] == pred_device].sort_values("timestamp")
    long = p.melt(
        id_vars="timestamp", value_vars=["actual", "predicted"],
        var_name="serija", value_name="vrednost",
    )
    figp = px.line(
        long, x="timestamp", y="vrednost", color="serija",
        color_discrete_map={"actual": "#636efa", "predicted": "#ef553b"},
        labels={"timestamp": "vreme", "vrednost": f"{target_label} [ug/m3]"},
    )
    figp.update_layout(height=450, legend_title_text="")
    st.plotly_chart(figp, width="stretch")

    mae_dev = (p["actual"] - p["predicted"]).abs().mean()
    st.caption(f"MAE za ovaj senzor na test periodu: {mae_dev:.2f} ug/m3 "
               f"({len(p)} sati)")

    # --- anomalies ---
    st.subheader("Detekcija anomalija — ekstremne epizode zagadjenja")
    st.markdown(
        "- **unusual** — statisticki neuobicajen dan za taj grad (robusni z-score > 3)\n"
        "- **unhealthy** — dnevni prosek preko EPA 'Unhealthy' praga "
        "(PM2.5 > 55.4 ili PM10 > 154 ug/m3)\n"
        "- **extreme** — oba kriterijuma istovremeno"
    )
    an = anomalies[
        (anomalies["city"].isin(cities))
        & (anomalies["date"].dt.date >= start)
        & (anomalies["date"].dt.date <= end)
    ]
    if an.empty:
        st.info("Nema anomalija za izabrani filter.")
    else:
        figa = px.scatter(
            an, x="date", y="pm25_raw", color="severity", symbol="city",
            color_discrete_map={
                "unusual": "#ffa15a", "unhealthy": "#ef553b", "extreme": "#7e0023",
            },
            labels={"date": "datum", "pm25_raw": "dnevni prosek PM2.5 [ug/m3]",
                    "severity": "ozbiljnost", "city": "grad"},
        )
        figa.update_traces(marker_size=9)
        figa.update_layout(height=420)
        st.plotly_chart(figa, width="stretch")

        st.dataframe(
            an.sort_values("pm25_raw", ascending=False)
            [["city", "date", "pm25_raw", "pm10_raw", "severity"]]
            .round(1),
            width="stretch", hide_index=True,
        )

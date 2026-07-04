# IoT Air Quality Analytics — analiza i predikcija zagađenja vazduha

Analiza i predikcija zagađenja vazduha u Srbiji na osnovu istorijskih podataka sa Clarity senzora. Projekat pokriva ceo tok: ETL (učitavanje i čišćenje podataka) → statistička analiza + AQI → interaktivni Streamlit dashboard → ML predikcija PM2.5/PM10.

**Tech stack:** Python, pandas, numpy, SQLite (jedinstvena baza), **Streamlit** (dashboard), Folium (interaktivna mapa), Plotly (grafici), scikit-learn (ML).

## Status projekta

| Celina | Status |
| --- | --- |
| ETL pipeline (učitavanje, čišćenje, deduplikacija, geokodiranje, agregacija) | ✅ Gotovo |
| EDA (statistika, korelacije, dnevni/sezonski obrasci) | ✅ Gotovo |
| AQI — izračunavanje iz koncentracija (US EPA) + unakrsna provera | ✅ Gotovo |
| Streamlit dashboard (mapa senzora, vremenske serije, AQI analiza, filteri) | ✅ Gotovo |
| ML — feature engineering, predikcija PM2.5/PM10, evaluacija | ✅ Gotovo |
| Detekcija anomalija + integracija ML u dashboard | ✅ Gotovo |

## 1. Podaci

Podatke je obezbedio profesor (Clarity air quality senzori). Dva izvora:

- **`data/raw/kg/`** — mesečni CSV fajlovi samo za Kragujevac (2 senzora)
- **`data/raw/national/`** — CSV fajlovi za senzore širom Srbije, izvezeni u nepravilnim (delom preklapajućim) intervalima — npr. 4 različita exporta za februar 2023.

Sirovi fajlovi se nikad ne menjaju — ETL ih samo čita. Ceo `data/` folder je van git-a; fajlove treba lokalno postaviti pre pokretanja.

### Zašto nema MQTT-a?

U realnoj IoT arhitekturi senzor bi kontinuirano slao merenja preko **MQTT brokera**: `senzor → MQTT broker (publish/subscribe) → ingestion servis → baza`, a dashboard bi se osvežavao u realnom vremenu. Mi umesto toga radimo **offline obradu istorijskog dataseta**: podaci su već izvezeni u CSV fajlove i novi ne pristižu, pa bi MQTT sloj bio veštački dodatak bez funkcije. Fokus projekta je na čišćenju, analizi, vizualizaciji i predikciji nad postojećom istorijom; opisani MQTT lanac je tačka u kojoj bi se ovaj sistem proširio u produkcijsku verziju.

## 2. Pokretanje

```bash
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# 1) ETL — obavezno ovim redosledom
python etl/load_data.py        # CSV -> data/processed/air_quality.db (tabela measurements)
python etl/map_locations.py    # + tabela sensors (device_id -> grad, reverse geocoding)
python etl/aggregate.py        # + tabele daily_city_avg i monthly_city_avg (lokalno vreme)

# 2) Analiza
python analysis/eda.py         # statistika, korelacije, obrasci -> data/processed/eda/
python analysis/aqi.py         # + tabele aqi i daily_city_aqi (US EPA proračun)

# 3) ML
python ml/train.py             # trenira i poredi modele -> ml_metrics, ml_predictions
python ml/anomalies.py         # detekcija ekstremnih epizoda -> anomalies

# 4) Dashboard
streamlit run dashboard/app.py
```

## 3. Struktura projekta (po ulogama iz SCRUM plana)

```
etl/          Član 1 — Data & Backend: load_data.py, map_locations.py, aggregate.py
analysis/     Analiza: eda.py (EDA), aqi.py (US EPA AQI + unakrsna provera)
dashboard/    Član 2 — Vizualizacija: app.py (Streamlit UI), data.py (pristup podacima)
ml/           Član 3 — ML/Predikcija: features.py, train.py, anomalies.py
```

## 4. ETL pipeline (Član 1)

### `etl/load_data.py`
- Učitava sve `.csv` fajlove iz `data/raw/kg` i `data/raw/national`
- Preimenuje originalne (duge) nazive kolona u kratke snake_case nazive (npr. `PM2.5 1-Hour Mean Mass Concentration Raw [ug/m3]` → `pm25_raw`)
- Dodaje `source_dataset` (`"kg"`/`"national"`) i `source_file` kolone
- Parsira `timestamp` (UTC) i izbacuje redove bez validnog vremena ili `device_id`
- **Deduplikacija preklapajućih exporta** po ključu `(device_id, timestamp)` — zadržava red sa najviše popunjenih vrednosti (na realnim podacima: ~33k uklonjenih duplikata od ~169k redova)
- Automatski izbacuje kolone koje su 100% prazne (u ovom exportu: sve `*_calibrated`, ambijentalna temperatura/vlažnost, vetar, pritisak)
- Rezultat: tabela `measurements` (~136k redova, 17 senzora)

### `etl/map_locations.py`
- Uzima jedinstvene `(device_id, latitude, longitude)` kombinacije
- `reverse_geocoder` (offline, GeoNames) dodeljuje grad svakoj lokaciji → tabela `sensors`
- **Napomena o preciznosti:** biblioteka dodeljuje najbliže poznato naselje iz svoje baze — za manja mesta ume da promaši (npr. senzor u Babušnici dobija Belu Palanku). Lokacije su ručno proverene kroz precizniju pretragu (Google Places).

### `etl/aggregate.py`
- Spaja `measurements` + `sensors`, konvertuje UTC → **Europe/Belgrade** (dnevne/mesečne granice prate lokalno vreme)
- Automatski agregira sve numeričke kolone → tabele `daily_city_avg` i `monthly_city_avg`

## 5. Analiza (analysis/)

### `analysis/eda.py`
Deskriptivna statistika sa procentom nedostajućih vrednosti, korelaciona matrica, dnevni profil (po lokalnom satu), sezonski obrazac (po mesecu), rangiranje gradova po PM2.5. Izlazi (CSV + PNG) idu u `data/processed/eda/`.

### `analysis/aqi.py`
- **AQI se računa iz koncentracija** po US EPA metodologiji (breakpoint tabele + linearna interpolacija, sa propisanim odsecanjem decimala) — AQI kolone iz exporta se **ne prepisuju**
- PM2.5 i PM10 na 24h kliznim prosecima (fallback na 1h vrednost), NO2 na 1h; O3 izostavljen (EPA 1h indeks definisan tek ≥125 ppb, a o3_raw je ~86% prazan)
- Ukupan AQI = max po zagađivačima + dominantni zagađivač + kategorija
- **Unakrsna provera** na postojećim AQI kolonama iz exporta: naš NO2 AQI se poklapa sa exportovanim `no2_aqi_epa` u okviru ±1 poena na 100% od ~110k merenja (MAE 0.45)
- Rezultat: tabele `aqi` (po merenju) i `daily_city_aqi` (dnevni max po gradu)

## 6. Dashboard (dashboard/)

`streamlit run dashboard/app.py` — tri taba, sa filterima (period, lokacije, zagađivač) u sidebar-u:

1. **Mapa senzora** (Folium) — markeri obojeni po prosečnoj AQI kategoriji, veličina po prosečnoj vrednosti izabranog zagađivača, popup sa detaljima
2. **Vremenske serije** (Plotly) — po satu/danu/mesecu za izabrane gradove + dnevni profil po lokalnom satu
3. **AQI analiza** — heatmap dnevnog max AQI (grad × datum), raspodela kategorija po gradu, dominantni zagađivač
4. **ML predikcija** — tabela poređenja modela, grafik predikcija vs. stvarnih vrednosti po senzoru na test periodu, prikaz detektovanih anomalija

`dashboard/data.py` je čist sloj za pristup podacima (bez Streamlit-a), pa je testabilan iz komandne linije.

## 6a. ML — predikcija PM2.5/PM10 (ml/)

### `ml/features.py`
Serija svakog senzora se svodi na satnu rešetku, pa se grade feature-i **isključivo iz prošlosti**: lagovi (1, 2, 3, 6, 12, 24h), klizni proseci/devijacije (3h, 24h), promena u poslednja 3h, plus kalendarski feature-i iz lokalnog vremena (sin/cos sata i meseca, vikend, grejna sezona) i prateća merenja (NO2, interna temperatura/vlažnost). Cilj: vrednost zagađivača **1h unapred**. (~105k redova × 22 feature-a)

### `ml/train.py`
Vremenski split (prvih 80% vremena trening, poslednjih 20% test — slučajan split bi "procureo" budućnost u trening). Svi modeli se porede sa **persistence baseline-om** ("sledeći sat = trenutni sat"). Rezultati na test periodu (t+1h):

| Model | PM2.5 MAE | PM2.5 R² | PM10 MAE | PM10 R² |
| --- | --- | --- | --- | --- |
| Persistence (baseline) | 8.30 | 0.810 | 11.59 | 0.802 |
| Ridge | 8.26 | 0.830 | 11.63 | 0.823 |
| Random Forest | 7.85 | 0.842 | **11.02** | 0.833 |
| HistGradientBoosting | 7.93 | 0.841 | 11.05 | 0.830 |
| **MLP (neuronska mreža)** | **7.85** | **0.843** | 11.24 | 0.836 |

Najbolji modeli: **MLP za PM2.5, Random Forest za PM10** (po MAE). LSTM je razmatran za vremenske serije, ali TensorFlow/Keras nema build za Python 3.14 — MLP nad lag feature-ima pokriva neuronski pristup, a na tabularnim senzorskim podacima gradient boosting / šume su ionako standardno najjače. Modeli se čuvaju u `ml/models/*.joblib`, metrike u tabeli `ml_metrics`, predikcije najboljeg modela u `ml_predictions`.

### `ml/anomalies.py`
Ekstremne epizode po gradu i danu, dva komplementarna kriterijuma: **statistički** (robusni z-score preko medijane/MAD po gradu, |z| > 3 — "neuobičajeno za taj grad") i **apsolutni** (dnevni prosek preko EPA 24h "Unhealthy" praga: PM2.5 > 55.4, PM10 > 154 µg/m³). Na realnim podacima: 443 anomalna grad-dana, dominiraju zimske epizode u grejnoj sezoni (najgori dan: Bela Palanka, 16.12.2022, PM2.5 = 216 µg/m³). Rezultat u tabeli `anomalies`.

## 7. Šema baze (`data/processed/air_quality.db`)

| Tabela | Opis |
| --- | --- |
| `measurements` | Očišćena merenja — jedan red = jedno očitavanje senzora (UTC ISO timestamp) |
| `sensors` | `device_id → latitude, longitude, city, region, country_code` |
| `daily_city_avg` / `monthly_city_avg` | Proseci po gradu po danu/mesecu (lokalni dani) |
| `aqi` | Izračunati AQI po merenju (po zagađivaču + ukupan + kategorija) |
| `daily_city_aqi` | Dnevni max AQI po gradu |
| `ml_metrics` | MAE/RMSE/R² svih modela po ciljnoj veličini |
| `ml_predictions` | Predikcije najboljeg modela na test periodu |
| `anomalies` | Detektovane ekstremne epizode (grad, dan, kriterijumi, ozbiljnost) |

## 8. Poznata ograničenja podataka

- **Kalibrisane vrednosti, ambijentalna temperatura/vlažnost, vetar i pritisak ne postoje** u ovom exportu (kolone 100% prazne, automatski izbačene) → za analizu/ML se koriste `_raw` vrednosti
- **`o3_raw` je ~86% prazan** — koristiti oprezno
- `kg` i `national` dataseti dele dva kragujevačka senzora, ali pokrivaju komplementarne periode (nema pravih duplikata između njih)
- Beograd ima senzore u više opština (Novi Beograd, Palilula, Stari Grad, Vračar) — drže se odvojeno, ne spajaju se u jedan "Beograd"

## 9. SCRUM

Projekat je rađen po SCRUM planu (tim od 3 člana: Data & Backend, Vizualizacija/Dashboard, ML/Predikcija) kroz tri sprinta — plan i sprint dokumentacija se vode interno. Tech stack za dashboard je u toku rada zaključen na **Streamlit** (umesto Dash opcije iz prvobitnog plana).

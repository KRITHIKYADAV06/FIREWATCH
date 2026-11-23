# FIREWATCH 🔥

**Harmonizing NASA MODIS and VIIRS fire hotspots into one consistent burning-activity calendar for India, 2003–2026.**

NASA Space Apps Challenge 2026 · Challenge: *Harmonization of MODIS and VIIRS Hot Spots*

🌐 **Live app:** https://firewatch-india.streamlit.app · 💻 **Code:** https://github.com/KRITHIKYADAV06/FIREWATCH

> **Post-submission updates (23 Nov 2026):**
> - **State / district picker** for Punjab (23 districts) and Haryana (22 districts), including the newer districts Malerkotla and Charkhi Dadri
> - **Dashboard Area filter** lists your selected area, both states and every district
> - **India-wide yearly globe** shows the hotspots for any year, with your selected area outlined
> - **Light theme** with improved readability
>
> The version submitted for judging on 15 Nov 2026 is tagged [`submission-2026-11-15`](https://github.com/KRITHIKYADAV06/FIREWATCH/tree/submission-2026-11-15).

---

## The problem

NASA records active fires with two families of satellites:

| Sensor | Pixel size | Record used |
|---|---|---|
| MODIS (Aqua) | 1 km | 2003–2025 |
| VIIRS (S-NPP, NOAA-20) | 375 m | 2012 onward |

VIIRS sees much smaller fires, so it reports about **7.8× more hotspots** than MODIS over India (2012–2018). Joining the two records naively creates a **fake jump in 2012**: "more fires this year" can simply mean "a better sensor this year", and long-term trends become wrong.

## Our solution

1. **Clean:** 11.5 million NASA FIRMS hotspots; vegetation fires only (type 0), nominal/high confidence, on a 0.1° grid.
2. **Fit:** 12 monthly ratios (VIIRS ÷ MODIS), one per calendar month, fitted on the 2012–2018 overlap. Fire behaviour changes with the season, so one single ratio is not enough.
3. **Harmonize:** before Feb 2012, MODIS × monthly ratio; then VIIRS S-NPP as observed; after S-NPP stopped (2 Nov 2026), NOAA-20 is spliced in on the same scale. Every year from 2003 to 2026 is on the VIIRS S-NPP scale.
4. **Validate:** on 2019–2025, years the ratios never saw.

## Results

**Validation, held-out 2019–2025, India monthly totals**

| Method | MAE (hotspots/month) | MAPE | Pearson r |
|---|---|---|---|
| **FIREWATCH monthly ratio** | **4,288** | **15.9%** | **0.98** |
| One pooled ratio | 8,758 | 28.2% | 0.94 |
| Naive splice (no correction) | 31,107 | 88.1% | 0.94 |

The harmonization is about **7× more accurate** than joining the records without correction.

**Experimental next-month forecast (machine learning)**

Gradient boosting on each 0.1° cell's own history (recent months, same-month median, season, location), trained to 2023, tuned on 2024, tested on 2025:

| Method (2025 test) | MAE per cell-month | Top-10% hit rate |
|---|---|---|
| **FIREWATCH model** | **1.28** | **63%** |
| Climatology (same-month median) | 1.36 | 62% |
| Seasonal naive (same month last year) | 1.52 | 53% |

The forecast is labelled **experimental** in the app.

Charts: `outputs/charts/` · Details: [`docs/methodology.md`](docs/methodology.md), [`docs/limitations.md`](docs/limitations.md)

## Features

- **Choose any area in India:** draw on the map, type coordinates, or pick a point and radius
- **Pick a state or district:** Punjab or Haryana, as a whole state or any of their 45 districts (real outlines, shown by name)
- **Burning-activity calendar:** which months and years burned most, 2003–2026
- **Compare areas on the Dashboard:** the Area filter lists your selected area, Punjab, Haryana and every district
- **Trends** over the years
- **This month vs normal:** compared with the historical median; above the 90th percentile is flagged as unusual
- **Spatial analysis** (map layers, below)
- **Hotspot map** and **CSV downloads** (calendar and hotspots)
- **India-wide yearly globe:** pick any year from 2003 to 2026 to see that year's hotspots across India, with your selected area outlined and its total for the year
- **AI summary:** Google Gemini explains *only* FIREWATCH's computed numbers, labelled as AI-generated
- **Weather context:** NASA POWER rainfall and temperature for the selected area

### Map & spatial analysis

Users can select an area of interest by drawing a rectangle, circle or polygon on the map, by entering latitude/longitude boundaries or a point with a radius, or by choosing a state (Punjab or Haryana) and, optionally, one of its districts. The selected area is analysed using a 0.1° grid, where a cell is included when its centre falls inside the selected area. The spatial analysis provides three layers: yearly fire hotspot trends (increasing, decreasing or no significant trend, using a Theil-Sen slope and the Mann-Kendall test), hot and cold spot clusters using the Getis-Ord Gi* method, and chronic burning areas where hotspots are detected in the selected months in at least 80% of the years. The map also displays detected hotspots within the selected area.

## How to run (Windows)

Use **Python 3.11**.

```bat
git clone https://github.com/KRITHIKYADAV06/FIREWATCH.git
cd FIREWATCH
python -m pip install -r requirements.txt
python -m pytest -v tests/
streamlit run app/dashboard.py
```

The app opens at http://localhost:8501.

**Data:** the full data files are too large for GitHub. Put them in `data/processed/` (`fires.parquet`, `cell_days.parquet`, `cells_monthly.parquet`, `ratios.parquet`, `validation.json`), or use the live app.

**Optional AI summary:** create a `.env` file with `GEMINI_API_KEY=your_key` (never commit it).

**Troubleshooting**
- `ModuleNotFoundError: No module named 'scipy'` → `python -m pip install scipy`
- Any other `ModuleNotFoundError` → `python -m pip install -r requirements.txt` again
- "cannot load the monthly dataset" → the data files are not in `data/processed/`
- In CMD use `cd /d D:\path`; `$HOME\...` paths only work in PowerShell

## Project structure

```
app/        Streamlit dashboard (pages, HTML dashboard, helpers)
src/        harmonize, validation, forecast, data_access, grid, geo, spatial_analysis, weather, ai_summary
tests/      automated tests (pytest)
docs/       methodology, limitations, data contract, how to run
outputs/    charts
data/boundaries/  Punjab and Haryana state and district outlines (built by src/build_boundaries.py)
```

## Data

- **NASA FIRMS** active fire archive: MODIS C6.1 (Aqua), VIIRS S-NPP, VIIRS NOAA-20, India, through **31 July 2026**
- Vegetation fires only (type 0), nominal/high confidence, 0.1° grid
- **NASA POWER** for weather context
- **OpenStreetMap** administrative boundaries for the Punjab and Haryana state and district outlines (© OpenStreetMap contributors, ODbL 1.0), simplified to about 200 m. They are for analysis only, not an authoritative boundary map.

## Limitations

- Accurate for **regional totals** (an area, district or state), **not single 0.1° cells**: at cell level the error is not better than the naive approach.
- Zero detections are "zero FIRMS detections", **not proof that no fire occurred** (clouds, smoke and small fires can be missed).
- Data runs through 31 July 2026; near-real-time data is future work.
- The forecast is experimental and is not an official fire prediction.
- A few very small districts or areas are smaller than one 0.1° grid cell, so analysis selects no cells there; use the whole state or a point and radius instead.

## Future work

Near-real-time FIRMS data · state and district boundaries beyond Punjab and Haryana · land-cover context

## Team

| Member | Role |
|---|---|
| S Krithik Yadav | Team lead, data science & ML |
| Parth Pillai | Data engineering & deployment |
| Avinke1 | Backend |
| Alisha Maharana | Geospatial analysis & mapping |
| Anthony Fernandes | Frontend |
| Adrita Sarkar | Testing & documentation |

## AI-use disclosure

We used Claude (Anthropic), ChatGPT (OpenAI) and Gemini (Google) as coding and writing assistants. All data, results and numbers come from our own code run on NASA data, and every AI-assisted change was reviewed and tested by the team (automated tests plus manual checks). The in-app AI summary is generated by Google Gemini from FIREWATCH's computed numbers only and is labelled as AI-generated.

## Acknowledgment

We acknowledge the use of data and imagery from NASA's Fire Information for Resource Management System (FIRMS), part of NASA's Earth Science Data and Information System (ESDIS). Weather data: NASA POWER Project. State and district outlines: © OpenStreetMap contributors (ODbL 1.0).

## License

MIT

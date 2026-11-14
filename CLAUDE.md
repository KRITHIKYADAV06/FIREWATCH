# FIREWATCH — project context (read this first) · v3

NASA Space Apps Challenge 2026, challenge "Harmonization of MODIS and VIIRS Hot Spots".
Build window: Sat 14 Nov 09:00 to Sun 15 Nov 23:59 IST (we submit by 21:30).
Team of 6. Lead: Krithik (data scientist + ML, research, API integration). Python only.

## Challenge (official)
Build a web app that harmonizes MODIS and VIIRS active-fire hotspots into a burning-activity
calendar. Users select an area of interest (draw on map or lat/lon) and a date range, see
which months/years had most fire, compare the current month with the historical median,
and download analytics plus a CSV of the hotspots found. Use accessible colour ramps.

## Data (NASA FIRMS, India-wide)
- MODIS Aqua C6.1 archive 2003-2025 (1 km)
- VIIRS S-NPP 375 m archive 2012 to 2 Nov 2026 (S-NPP stopped on 2 Nov 2026)
- VIIRS NOAA-20 375 m archive 2018+ and API VIIRS_NOAA20_NRT for the present
- Optional after MVP: Meteomatics / NASA POWER weather, ESA WorldCover (Planetary Computer),
  Gemini summary of OUR computed numbers only (labelled "AI-generated summary")
- Demo AOI: Punjab + Haryana (lon 73.8-77.6, lat 27.6-32.6)

## Method
1. Keep type == 0; confidence low/nominal/high (MODIS <30 low, 30-79 nominal, >=80 high); drop low.
2. Grid to 0.1 degree cells (cell_id "30.4_75.8"); count hotspots and active cell-days.
3. Monthly ratio r_m = sum VIIRS / sum MODIS on 2012-2018, for counts and cell-days.
   Reference scale = VIIRS. Pre-2012 = MODIS x r_m. Check NOAA-20 vs S-NPP ratio on 2018-2026.
4. Validate on 2019-2025: MAE, MAPE, Pearson r vs naive splice (r = 1).
5. Per AOI, per calendar month: median, 25th/75th/90th percentile; current month vs median.
   Unusual = above 90th percentile. Isolation Forest = stretch only.

## Data contract (data/processed/, Parquet)
- fires.parquet: sensor, satellite, date, time, lat, lon, confidence, frp, daynight, type, source
- cell_days.parquet: cell_id, cell_lat, cell_lon, date, sensor, satellite, n_hotspots, frp_sum
- ratios.parquet: month, ratio_count, ratio_celldays, fit_years, kind
- cells_monthly.parquet: cell_id, cell_lat, cell_lon, year, month, raw_modis, raw_viirs,
  harmonized_count, harmonized_celldays
- validation.json: mae, mape, r, naive_mae, naive_mape, test_years

## Files and owners
- src/harmonize.py, validation.py, anomaly.py, weather.py, ai_summary.py — Krithik
- src/ingest.py, FIRMS download + NRT fetch, data checks — Friend (data engineering; acting lead Sat 16:00-22:00)
- src/grid.py, map tab, landcover.py — Geospatial junior
- src/data_access.py — Backend junior: cells_in_aoi, monthly_series, calendar_matrix,
  vs_median, hotspots, get_validation, to_csv; all cached with @st.cache_data
- app/dashboard.py — Frontend junior. Tabs: Calendar, Map & AOI, Trends, vs Median, Methods
- docs/, README.md, demo deck, submission text — Tester/Docs junior

## Rules for the AI assistant
- Complete runnable files, not fragments. Windows 11, Anaconda, VS Code.
- Follow the data contract exactly; never rename columns.
- Simple, commented code for beginners. No new libraries without asking.
- Never fabricate data, results or accuracy numbers. No fire prediction or burned-area claims.
- Keys only in .env (FIRMS_MAP_KEY, METEOMATICS_USER/PASS, GEMINI_API_KEY). Never commit data or .env.
- Show the FIRMS acknowledgment in the app footer and README.

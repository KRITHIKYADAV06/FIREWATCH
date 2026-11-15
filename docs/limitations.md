# FIREWATCH limitations

## Data coverage

- The archive runs **through 2026-07-31** (MODIS Aqua through 2025-12-31;
  VIIRS S-NPP, NOAA-20 and NOAA-21 through 2026-07-31). August 2026 onward is
  not in the app's data.
- **Near-real-time (NRT) data is future work.** The app does not yet pull
  `VIIRS_NOAA20_NRT` from the FIRMS API, so "current month" in the app means
  the latest month in the archive, not today.

## Validation result: regional totals yes, single cells no

From `data/processed/validation.json` (test years 2019-2025, ratios fit on
2012-2018):

- India-wide monthly totals: MAE 4,288 vs naive splice 31,107; MAPE 15.9 %
  vs 88.1 %; r 0.980 vs 0.940. The harmonization works well at this level.
- **Cell level (0.1 degree): MAE 6.28 vs naive 5.91 (r 0.76 vs 0.77).** At
  single-cell level the monthly ratio is slightly *worse* than not scaling at
  all, and cell MAPE is 132 %. A single India-wide multiplier cannot fix a
  cell where MODIS saw 0 or 1 fires but VIIRS saw many; small-number noise
  dominates.
- **So the method is for regional totals (an AOI, a district, a state), not
  for single cells.** Per-cell harmonized values on the map should be read as
  a pattern, not as an exact count for that cell.

## Known methodological limitations

- **Monthly ratios are pooled across all of India.** A single `r_m` per
  calendar month does not capture regional differences in canopy cover,
  typical fire size, or cloud cover between e.g. Punjab stubble burning and
  forest fires elsewhere. The ratio is a reasonable India-wide average, not a
  per-cell correction.
- **The 2012-2018 fit window is itself imperfect data.** S-NPP only starts 20
  January 2012, so January 2012 is a partial month; months with too little
  MODIS activity in the fit window fall back to the pooled annual ratio
  instead of a month-specific one (see `MIN_MODIS_TOTAL` in
  `src/harmonize.py`).
- **Pre-2012 harmonized counts are an estimate, not a measurement.** Before
  VIIRS existed, `harmonized_count = raw_modis x ratio_count[month]`. This
  assumes the same MODIS-to-VIIRS relationship held in e.g. 2003 as it did in
  2012-2018, which cannot be directly checked (no VIIRS data exists before
  2012 to compare against).
- **Hotspot counts are not burned area or fire risk.** A hotspot is a
  satellite detection of thermal anomaly at one point in time; FIREWATCH makes
  no claim about area burned, fire severity, or the probability of future
  fire.
- **The forecast (`src/forecast.py`) is explicitly experimental.** It
  predicts next month's hotspot count for a cell from that cell's own recent
  history. It has no weather, land cover, or neighboring-cell information, so
  it cannot anticipate a fire season driven by, say, an unusually dry winter
  that has no precedent in that cell's own lag history. Its output is
  labelled `"experimental"` and the code reports plainly (via
  `forecast_metrics.json`'s `model_beats_seasonal_naive` /
  `model_beats_climatology` keys) whether it beats the seasonal-naive and
  climatology baselines. On the 2025 test year it does (MAE 1.284 vs 1.520
  seasonal naive and 1.356 climatology; RMSE 4.697 vs 6.102 and 6.552), but
  the margin over climatology is small, and it is one test year only.
- **Cells with very little historical activity are excluded from the
  forecast** (fewer than 5 fires total), so the forecast has no opinion about
  rarely-burning cells; it is silent there, not predicting zero risk.

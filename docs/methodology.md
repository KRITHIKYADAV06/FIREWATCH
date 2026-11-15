# FIREWATCH methodology

## Why harmonization is needed

NASA FIRMS gives us fire detections from three different sensors over different
periods:

- MODIS Aqua (1 km), 2003-2025
- VIIRS S-NPP (375 m), 2012 - 2 Nov 2026
- VIIRS NOAA-20 (375 m), 2018 onwards

VIIRS detects far more, smaller fires than MODIS because its pixels are about
8x finer. A raw hotspot count that simply switches sensor at 2012 and 2018
would show a sudden jump that has nothing to do with real fire activity - it
would just be a change of instrument. Harmonization puts every year onto ONE
scale (VIIRS S-NPP) so that "more hotspots this year" always means more fire,
not a better sensor.

## Steps (implemented in `src/harmonize.py` and `src/validation.py`)

1. **Filter detections** (`src/ingest.py`): keep `type == 0` (vegetation
   fires only); keep confidence nominal/high; for MODIS, drop confidence
   below 30 (its "low" bucket).
2. **Grid** (`src/grid.py`): assign every hotspot to a 0.1 degree cell,
   `cell_id = "<lat floor>_<lon floor>"`. Aggregate to cell-days
   (`cell_days.parquet`): one row per cell, date, sensor, satellite, with the
   hotspot count and FRP sum for that cell-day.
3. **Fit monthly ratios** (`src/harmonize.py`): for each calendar month `m`,
   `r_m = sum(S-NPP) / sum(Aqua)` pooled across India, fit on 2012-2018 (the
   years both sensors overlap). The same is done for NOAA-20 vs S-NPP on their
   2018+ overlap, to splice NOAA-20 onto the S-NPP scale once S-NPP data ends.
4. **Build the harmonized series** (`cells_monthly.parquet`):
   - before Feb 2012: `Aqua x r_m` (MODIS scaled up to the VIIRS scale)
   - Feb 2012 until S-NPP ends: S-NPP as observed
   - after S-NPP ends: `NOAA-20 / (NOAA-20/S-NPP ratio)`
   Raw MODIS and raw VIIRS columns are also kept, unscaled, so the "before"
   series can be reconstructed for comparison.
5. **Validate on 2019-2025** (`src/validation.py`): this window is held out
   from the 2012-2018 ratio fit, so it tests whether the harmonization
   generalizes to years the ratio never saw. For each cell-month in
   2019-2025:
   - `truth` = `harmonized_count` (the VIIRS-scale value FIRMS actually
     measured there)
   - `predicted` = `raw_modis x ratio_count[month]` (what our harmonization
     would have said, using ONLY the MODIS detections and the monthly ratio)
   - `naive` = `raw_modis` (the baseline of not scaling MODIS at all)
   - `pooled` = `raw_modis x one single pooled ratio` (a baseline with no
     month-to-month seasonality)
   MAE, MAPE and Pearson r are computed both on India-wide monthly totals and
   on individual cell-months (restricted to cell-months where the truth or
   the prediction is greater than zero).
6. **Per-AOI calendar statistics** (`src/data_access.py`): for a chosen area
   and calendar month, the historical median and 25th/75th/90th percentiles
   across years are computed from `cells_monthly.parquet`, and the current
   month is compared against them. A month above the 90th percentile is
   flagged as unusual.

## Validation results

`src/validation.py` writes `data/processed/validation.json` with the metrics
described in step 5 above. **At the time this document was generated, that
file does not exist** - `data/processed/cells_monthly.parquet` and
`ratios.parquet` are not present in this environment (they are produced by
`src/ingest.py` + `src/harmonize.py`, which need a real FIRMS download and a
`FIRMS_MAP_KEY` in `.env`). This document intentionally reports no numbers
rather than inventing them. Once those two files exist, running
`python src/validation.py` will populate `validation.json`, and this section
should be updated with the real `mae`, `mape`, `r`, and baseline figures it
contains.

## Forecast (experimental, out of scope for the harmonization itself)

`src/forecast.py` predicts next month's `harmonized_count` per cell from that
cell's own recent history (lags, a 3-month mean, the same-month median of
earlier years, calendar-month seasonality, and location), using
`HistGradientBoostingRegressor(loss="poisson")`, trained on data through 2023
and tested on 2025. It is explicitly labelled `"experimental"` in its output
and is not a fire-risk or burned-area prediction. Its metrics, like the
validation metrics above, can only be computed once real harmonized data is
available; see `docs/limitations.md`.

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

All numbers below are copied from `data/processed/validation.json`, written
by `python src/validation.py`. Ratios were fit on 2012-2018; the test years
2019-2025 were never seen by the fit. Truth = S-NPP VIIRS count as observed;
prediction = `raw_modis x ratio_count[month]`.

**India-wide monthly totals** (n = 84 test months, 2019-2025):

| Method | MAE (hotspots/month) | MAPE | Pearson r |
|---|---:|---:|---:|
| FIREWATCH monthly ratio | **4,288** | **15.9 %** | **0.980** |
| Single pooled ratio (7.837) | 8,758 | 28.2 % | 0.940 |
| Naive splice (raw MODIS, r = 1) | 31,107 | 88.1 % | 0.940 |

The month-specific ratio cuts the error of a naive splice by about 86 % and
roughly halves the error of a single pooled ratio, so the seasonal shape of
`r_m` matters.

**Individual 0.1 degree cell-months** (n = 449,315 cell-months where truth or
raw MODIS > 0; MAPE on the 436,631 with truth > 0):

| Method | MAE | MAPE | Pearson r |
|---|---:|---:|---:|
| FIREWATCH monthly ratio | 6.28 | 132.3 % | 0.759 |
| Single pooled ratio | 6.25 | 127.1 % | 0.769 |
| Naive splice | 5.91 | 92.8 % | 0.769 |

At single-cell level the harmonization does **not** beat the naive splice
(see `docs/limitations.md`): the method is designed for, and validated on,
regional / AOI totals.

**Fitted ratios** (`ratios.parquet`): MODIS-to-VIIRS `ratio_count` ranges
from 5.0 (October) to 17.7 (June); the NOAA-20 / S-NPP ratio on 2018-2026 is
close to 1 (0.97-1.08) in every month except July (1.20).

Charts in `outputs/charts/`: `harmonization_before_after.png`,
`monthly_ratios.png`, `validation_pred_vs_actual.png`.

## Forecast (experimental, out of scope for the harmonization itself)

`src/forecast.py` predicts next month's `harmonized_count` per cell from that
cell's own recent history (lags 1/2/3/12, a 3-month mean, the same-month
median of earlier years, calendar-month seasonality, and location), using
`HistGradientBoostingRegressor(loss="poisson", max_iter=300,
learning_rate=0.05)`. Trained on 2014-2023, learning rate chosen on 2024,
tested on 2025 (24,068 cells with at least 5 hotspots in total; 288,816 test
cell-months). It is explicitly labelled `"experimental"` in its output and is
not a fire-risk or burned-area prediction.

Results from `data/processed/forecast_metrics.json` (test year 2025):

| Method | MAE | RMSE | Top-10 % hit rate |
|---|---:|---:|---:|
| Forecast model | **1.284** | **4.697** | **0.632** |
| Climatology (same-month median) | 1.356 | 6.552 | 0.616 |
| Seasonal naive (same month last year) | 1.520 | 6.102 | 0.530 |

The model beats both baselines on MAE and RMSE. Top-10 % hit rate = share of
the top 10 % of 2025 cell-months by actual count that are also in the top
10 % by prediction. The published forecast is for 2026-08, from a model
refitted on all labelled months. Backtest chart:
`outputs/charts/forecast_backtest.png`.

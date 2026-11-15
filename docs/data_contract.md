# FIREWATCH data contract

All processed data lives in `data/processed/` as Parquet files. Column names
below must never be renamed - every module (`src/harmonize.py`,
`src/validation.py`, `src/forecast.py`, `src/data_access.py`,
`app/dashboard.py`) reads and writes these exact names.

## `fires.parquet`

One row per FIRMS hotspot detection, after filtering (see
`docs/methodology.md`).

| column     | meaning                                              |
|------------|-------------------------------------------------------|
| sensor     | "MODIS" or "VIIRS"                                   |
| satellite  | "Aqua", "S-NPP", "NOAA-20", or "NOAA-21"             |
| date       | detection date                                       |
| time       | detection time                                       |
| lat, lon   | detection coordinates                                |
| confidence | FIRMS confidence (low/nominal/high, or 0-100)        |
| frp        | fire radiative power                                 |
| daynight   | "D" or "N"                                           |
| type       | FIRMS detection type (0 = vegetation fire)           |
| source     | which FIRMS product this row came from               |

## `cell_days.parquet`

One row per 0.1 degree cell, date, sensor and satellite.

| column               | meaning                                   |
|----------------------|--------------------------------------------|
| cell_id              | `"<lat floor>_<lon floor>"`, e.g. `"30.4_75.8"` |
| cell_lat, cell_lon   | lower-left corner of the cell             |
| date                 | the day                                   |
| sensor, satellite    | as in `fires.parquet`                     |
| n_hotspots           | detections in that cell on that day       |
| frp_sum              | sum of FRP in that cell on that day       |

## `ratios.parquet`

One row per calendar month and ratio kind.

| column          | meaning                                                   |
|-----------------|-------------------------------------------------------------|
| month           | 1-12                                                        |
| ratio_count     | VIIRS / MODIS (or NOAA-20 / S-NPP) ratio of hotspot counts   |
| ratio_celldays  | same ratio, for active cell-days instead of raw counts      |
| fit_years       | the year range the ratio was fit on, e.g. `"2012-2018"`     |
| kind            | `"modis_viirs"` or `"n20_snpp"`                              |

## `cells_monthly.parquet`

One row per cell and calendar month. A cell-month missing from this file
means zero detections from every sensor that month (the "zero rule").

| column              | meaning                                               |
|---------------------|---------------------------------------------------------|
| cell_id             | as above                                               |
| cell_lat, cell_lon  | as above                                               |
| year, month         | the calendar month                                     |
| raw_modis           | unscaled MODIS Aqua count that month (NaN = no MODIS data that month) |
| raw_viirs           | unscaled VIIRS count that month, whichever satellite was active (NaN = no VIIRS data that month) |
| harmonized_count    | hotspot count on the VIIRS S-NPP scale (see methodology) |
| harmonized_celldays | active cell-days on the VIIRS S-NPP scale               |

## `validation.json`

Written by `src/validation.py`. Keys: `mae`, `mape`, `r`, `naive_mae`,
`naive_mape`, `test_years`, plus the extras `fit_years`, `pooled_ratio`,
`pooled_mae`, `pooled_mape`, `pooled_r`, and the cell-month level
`cell_mae`, `cell_mape`, `cell_r`, `cell_naive_mae`, `cell_naive_mape`,
`cell_pooled_mae`, `cell_pooled_mape`, `cell_pooled_r`, `cell_n`.

## `forecast.parquet` / `forecast_metrics.json`

Written by `src/forecast.py`. `forecast.parquet` has one row per cell for the
month after the latest month in `cells_monthly.parquet`:
`cell_id, cell_lat, cell_lon, year, month, predicted, label` (`label` is
always the string `"experimental"`). `forecast_metrics.json` has `train`,
`valid`, `test` blocks (each with `model`, `seasonal_naive`, `climatology`
sub-blocks of `mae`, `rmse`, `top10pct_hit_rate`, plus `n`), and
`beats_seasonal_naive`, `beats_climatology`, `honest_summary`,
`train_end_year`, `valid_year`, `test_year`.

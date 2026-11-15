# FIREWATCH limitations

## Data availability (as of this document)

`data/processed/cells_monthly.parquet` and `data/processed/ratios.parquet` do
not exist in this environment, so:

- `src/validation.py` cannot produce real `mae`, `mape`, `r`, or baseline
  numbers, and `data/processed/validation.json` was not written.
- `src/forecast.py` cannot be trained or evaluated, and
  `data/processed/forecast.parquet` / `forecast_metrics.json` were not
  written.
- The three required PNGs in `outputs/charts/` (before/after harmonization,
  monthly ratios, predicted vs actual) and `forecast_backtest.png` were not
  generated.

These two files are produced by `src/ingest.py` (FIRMS download, needs
`FIRMS_MAP_KEY` in `.env`) followed by `src/harmonize.py`. Both `src/
validation.py` and `src/forecast.py` are complete and unit-tested
(`tests/test_validation.py`, `tests/test_forecast.py`, run against small
synthetic tables with a known answer) and will run end-to-end as soon as
those two parquet files exist - see `docs/how_to_run.md`.

## Known methodological limitations, independent of data availability

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
  `forecast_metrics.json`'s `beats_seasonal_naive` / `beats_climatology` /
  `honest_summary` keys) if it fails to beat the seasonal-naive or
  climatology baselines - this must not be overridden or hidden once real
  numbers are available.
- **Cells with very little historical activity are excluded from the
  forecast** (fewer than 5 fires total), so the forecast has no opinion about
  rarely-burning cells; it is silent there, not predicting zero risk.

# How to run FIREWATCH's data science pipeline

Windows 11 / Anaconda, from the repo root, with `requirements.txt` installed
(`pip install -r requirements.txt`) and a `.env` file holding
`FIRMS_MAP_KEY=...` (see `README.md`).

## 1. Get real data and build the harmonized tables

```
python src/ingest.py
python src/harmonize.py
```

This downloads FIRMS detections, filters and grids them, and writes
`data/processed/cell_days.parquet`, `ratios.parquet` and
`cells_monthly.parquet` (see `docs/data_contract.md`).

## 2. Validate the harmonization (2019-2025 holdout)

```
python src/validation.py
```

Writes `data/processed/validation.json` and three charts in
`outputs/charts/`: `before_after_harmonization.png`, `monthly_ratios.png`,
`predicted_vs_actual.png`.

## 3. Train the experimental next-month forecast

```
python src/forecast.py
```

Writes `data/processed/forecast.parquet`, `forecast_metrics.json`, and
`outputs/charts/forecast_backtest.png`.

## 4. Run the tests

```
python -m pytest -v tests/
```

`tests/test_validation.py` and `tests/test_forecast.py` use small synthetic
tables with a known answer, so they pass without any real FIRMS data.
`tests/test_data_access.py`, `test_frontend_utils.py`, `test_grid.py` and
`test_spatial_analysis.py` cover the rest of the pipeline the same way.

## Notes

- `src/validation.py` and `src/forecast.py` both raise a clear
  `FileNotFoundError` if `cells_monthly.parquet` / `ratios.parquet` are
  missing, instead of inventing numbers.
- Chart generation needs `matplotlib`; model training in `src/forecast.py`
  needs `scikit-learn`. If either is missing, the scripts print a message and
  skip just that step rather than crashing.

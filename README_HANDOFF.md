# FIREWATCH — Data Engineering handoff

## Verified output

- `data/processed/fires_clean.csv` contains the real, filtered FIRMS archive observations. No generated/synthetic observations.
- `reports/ingestion_quality.json` details source files, drops and satellite-year coverage.
- `reports/handoff_validation.json` confirms exact 11-column schema and QA checks.
- `data/sample/punjab_haryana_2023_11.csv` is a real, filtered demo-area sample.
- `reports/monthly_satellite_features.csv` is a **supplemental** time-series table for EDA. Do not substitute it for the official `cells_monthly.parquet`.

## Mac steps to create required Parquet files

Within the FIREWATCH Git repository (on `feature/data-engineering`):

1. Activate your existing environment: `source .venv/bin/activate`
2. Install existing project requirements: `python -m pip install -r requirements.txt`
3. Copy the provided CSV into `data/processed/fires_clean.csv` and the included scripts to `src/`.
4. Run `python src/create_parquet_handoff.py` to build **`data/processed/fires.parquet`**.
5. With the Geospatial junior's approval, run `python src/create_parquet_handoff.py --cell-days` if their `grid.py` is not ready. This creates **`data/processed/cell_days.parquet`** using 0.1-degree grid cells, preserving the exact contract.
6. Run `python src/validate_handoff.py` to verify the CSV and regenerate monthly EDA features.

Do not commit CSV, Parquet or .env. Push only scripts and QA documentation as approved, on `feature/data-engineering`; open a PR to main.

## Important scientific limitations

- A **broad India bounding box** (lat 6–38.5, lon 68–98) was applied, NOT an India administrative boundary polygon. Some cross-border points can remain. Krithik should approve this spatial approximation.
- Only MODIS **Aqua** observations are used, as specified by the team methodology; MODIS Terra was excluded.
- VIIRS S-NPP, NOAA-20, and **Krithik-approved NOAA-21** are separate satellite streams. The harmonization model must select/normalize satellites carefully; summing unadjusted VIIRS streams inflates counts.
- VIIRS S-NPP records after 2026-11-02 are excluded by the agreed cutoff. MODIS Aqua 2026 was excluded from the core design.
- Uploaded **2026 NRT files do not contain `type`**, so none of those NRT records passed the strict `type == 0` filter. Krithik must decide how to handle them; they were **not** mislabelled as vegetation fires.
- The latest validated archive date is **2026-07-31**, so the current-month panel is NOT current. This is a blocker for a November 2026 demo unless a scientifically defensible NRT treatment is approved.
- Data represents satellite active-fire **detections**, not fire events, burned area, or predictions.
- The required **harmonization and validation** steps are Krithik's responsibility; clean detection data alone does not establish accurate sensor calibration.

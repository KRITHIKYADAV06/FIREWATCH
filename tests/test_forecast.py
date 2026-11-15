# tests/test_forecast.py
"""
Tests for src/forecast.py on a small synthetic cell x month table.
Run from the repo root:   pytest -v tests/test_forecast.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import forecast as fc  # noqa: E402


def _cells_monthly(cell_id="30.0_75.0", start_ym=(2013, 1), n_months=36, count_fn=None):
    """One cell, `n_months` consecutive months, count = count_fn(year, month)."""
    if count_fn is None:
        count_fn = lambda year, month: 5.0 + month      # deterministic, varies by month
    rows = []
    year, month = start_ym
    for _ in range(n_months):
        rows.append({"cell_id": cell_id, "cell_lat": 30.05, "cell_lon": 75.05,
                     "year": year, "month": month, "harmonized_count": count_fn(year, month)})
        month += 1
        if month > 12:
            month = 1
            year += 1
    return pd.DataFrame(rows)


def test_build_full_grid_fills_missing_months_with_zero():
    cm = _cells_monthly(n_months=3)             # only Jan-Mar 2013 present
    full = fc.build_full_grid(cm, grid_start_ym=201301, min_total_fires=0)
    # the grid runs from 2013-01 to the LAST month in the data (Mar 2013): 3 rows
    assert len(full) == 3
    assert (full["harmonized_count"] >= 0).all()


def test_build_full_grid_drops_low_activity_cells():
    busy = _cells_monthly(cell_id="30.0_75.0", n_months=12,
                           count_fn=lambda y, m: 2.0)        # total 24 >= 5
    quiet = _cells_monthly(cell_id="40.0_80.0", n_months=12,
                            count_fn=lambda y, m: 0.2)        # total 2.4 < 5
    cm = pd.concat([busy, quiet], ignore_index=True)
    full = fc.build_full_grid(cm, grid_start_ym=201301, min_total_fires=5)
    assert set(full["cell_id"]) == {"30.0_75.0"}


def test_features_use_only_strictly_earlier_months_no_leakage():
    cm = _cells_monthly(n_months=36)
    full = fc.build_full_grid(cm, grid_start_ym=201301, min_total_fires=0)
    sup = fc.make_supervised(full)

    # For every row, the lag1 feature must equal the harmonized_count of the
    # PREVIOUS month, and the target must equal the NEXT month's count -
    # i.e. features strictly precede, and never equal, the label's own month.
    lookup = full.set_index(["cell_id", "year", "month"])["harmonized_count"]
    for _, row in sup.sample(min(15, len(sup)), random_state=0).iterrows():
        cur_period = pd.Period(f"{row['year']}-{row['month']:02d}", freq="M")
        prev_period = cur_period - 1
        key = (row["cell_id"], prev_period.year, prev_period.month)
        assert row["lag1"] == pytest.approx(lookup.loc[key])
        next_period = cur_period + 1
        target_key = (row["cell_id"], next_period.year, next_period.month)
        assert row["target"] == pytest.approx(lookup.loc[target_key])

    # month_median_prevyears for a January row must never include that same
    # January's own count (it is the current row, not an earlier year).
    jan_rows = sup[sup["month"] == 1]
    for _, row in jan_rows.iterrows():
        earlier = full[(full["cell_id"] == row["cell_id"]) & (full["month"] == 1) &
                        (full["year"] < row["year"])]
        if len(earlier):
            assert row["month_median_prevyears"] == pytest.approx(earlier["harmonized_count"].median())


def test_make_supervised_output_shape_and_no_missing_features():
    cm = _cells_monthly(n_months=36)
    full = fc.build_full_grid(cm, grid_start_ym=201301, min_total_fires=0)
    sup = fc.make_supervised(full)
    assert len(sup) > 0
    for col in fc.FEATURE_COLS + ["target", "target_year", "target_month"]:
        assert col in sup.columns
        assert sup[col].notna().all()


def test_split_train_valid_test_is_time_ordered():
    cm = _cells_monthly(start_ym=(2013, 1), n_months=12 * 13)   # 2013..2025
    full = fc.build_full_grid(cm, grid_start_ym=201301, min_total_fires=0)
    sup = fc.make_supervised(full)
    train, valid, test = fc.split_train_valid_test(sup)
    if len(train):
        assert train["target_year"].max() <= fc.TRAIN_END_YEAR
    if len(valid):
        assert (valid["target_year"] == fc.VALID_YEAR).all()
    if len(test):
        assert (test["target_year"] == fc.TEST_YEAR).all()


def test_metrics_known_values():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 6.0])
    assert fc.mae(actual, pred) == pytest.approx(0.5)
    assert fc.rmse(actual, pred) == pytest.approx(1.0)


def test_top_decile_hit_rate_perfect_prediction():
    actual = np.arange(20.0)
    assert fc.top_decile_hit_rate(actual, actual) == pytest.approx(1.0)


def test_load_cells_monthly_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        fc.load_cells_monthly(tmp_path / "cells_monthly.parquet")


def test_fit_model_and_forecast_shape_needs_sklearn():
    pytest.importorskip("sklearn")
    cm = _cells_monthly(start_ym=(2013, 1), n_months=12 * 13)   # 2013..2025
    full = fc.build_full_grid(cm, grid_start_ym=201301, min_total_fires=0)
    sup = fc.make_supervised(full)
    train, valid, test = fc.split_train_valid_test(sup)
    model = fc.fit_model(train)
    pred = model.predict(test[fc.FEATURE_COLS])
    assert len(pred) == len(test)
    assert (pred >= 0).all()          # Poisson loss -> non-negative predictions

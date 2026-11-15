# tests/test_validation.py
"""
Tests for src/validation.py on small synthetic tables with a known answer.
Run from the repo root:   pytest -v tests/test_validation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import validation as val  # noqa: E402


def _ratios(ratio_count=2.0, fit_years="2012-2018"):
    """A flat monthly MODIS->VIIRS ratio of `ratio_count` for every month."""
    rows = [{"month": m, "ratio_count": ratio_count, "ratio_celldays": ratio_count,
             "fit_years": fit_years, "kind": "modis_viirs"} for m in range(1, 13)]
    return pd.DataFrame(rows)


def _cells(years, ratio=2.0, noise=None):
    """One cell, one fire per month-index; raw_viirs/harmonized_count = raw_modis * ratio
    (+ optional per-row noise), so the relationship is exactly known."""
    rows = []
    for i, year in enumerate(years):
        for month in range(1, 13):
            modis = 10.0 + month          # varies a bit so MAPE isn't trivially 0/0
            truth = modis * ratio
            if noise is not None:
                truth += noise(year, month)
            rows.append({
                "cell_id": "30.0_75.0", "cell_lat": 30.05, "cell_lon": 75.05,
                "year": year, "month": month,
                "raw_modis": modis, "raw_viirs": truth,
                "harmonized_count": truth, "harmonized_celldays": 1.0,
            })
    return pd.DataFrame(rows)


def test_check_fit_years_accepts_2012_2018():
    assert val.check_fit_years(_ratios(fit_years="2012-2018")) == "2012-2018"


def test_check_fit_years_rejects_other_window():
    with pytest.raises(ValueError):
        val.check_fit_years(_ratios(fit_years="2015-2020"))


def test_pooled_ratio_recovers_known_constant():
    cells = _cells(years=[2012, 2013, 2014, 2015, 2016, 2017, 2018], ratio=3.0)
    assert val.pooled_ratio(cells) == pytest.approx(3.0)


def test_perfect_prediction_gives_zero_error_and_r_one():
    ratios = _ratios(ratio_count=2.0)
    cells = _cells(years=list(range(2012, 2019)) + list(range(2019, 2026)), ratio=2.0)
    result, test = val.run_validation_inmemory(cells, ratios)
    assert result["mae"] == pytest.approx(0.0, abs=1e-9)
    assert result["mape"] == pytest.approx(0.0, abs=1e-9)
    assert result["r"] == pytest.approx(1.0, abs=1e-9)
    # naive (raw_modis, unscaled) should be much worse than our harmonization
    assert result["naive_mae"] > result["mae"]


def test_mape_ignores_zero_actual_rows():
    actual = np.array([0.0, 10.0])
    pred = np.array([5.0, 11.0])
    # only the actual=10 row should count: |10-11|/10 = 10%
    assert val.mape(actual, pred) == pytest.approx(10.0)


def test_mae_and_pearson_known_values():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 6.0])
    assert val.mae(actual, pred) == pytest.approx(0.5)
    assert val.pearson_r(actual, actual) == pytest.approx(1.0)


def test_cell_month_metrics_drops_all_zero_rows():
    ratios = _ratios(ratio_count=1.0)
    cells = _cells(years=[2019], ratio=1.0)
    # add an all-zero cell-month that must be excluded from the cell-level score
    extra = pd.DataFrame([{
        "cell_id": "40.0_80.0", "cell_lat": 40.05, "cell_lon": 80.05,
        "year": 2019, "month": 1, "raw_modis": 0.0, "raw_viirs": 0.0,
        "harmonized_count": 0.0, "harmonized_celldays": 0.0,
    }])
    cells = pd.concat([cells, extra], ignore_index=True)
    pooled_r = val.pooled_ratio(_cells(years=[2012, 2013], ratio=1.0))
    test = val.build_holdout(cells, ratios, pooled_r)
    cell_metrics = val.cell_month_metrics(test)
    assert cell_metrics["n"] == 12   # the all-zero cell-month is excluded


def test_load_inputs_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        val.load_inputs(tmp_path / "cells_monthly.parquet", tmp_path / "ratios.parquet")


def test_build_holdout_raises_on_month_with_no_ratio():
    ratios = _ratios()
    ratios = ratios[ratios["month"] != 1]   # drop January's ratio
    cells = _cells(years=[2019], ratio=2.0)
    with pytest.raises(ValueError, match="no modis_viirs ratio"):
        val.build_holdout(cells, ratios, pooled_r=2.0)

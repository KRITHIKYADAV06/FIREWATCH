# tests/test_data_access.py
"""
Tests for src/data_access.py on a small synthetic dataset written to a temp folder.
Run from the repo root:   pytest -v tests/test_data_access.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_access as da  # noqa: E402

CELL_A, CELL_B, CELL_C = "30.0_75.0", "30.0_75.1", "31.0_76.0"


def _write(df, path):
    """Parquet when pyarrow exists (normal), pickle otherwise (minimal runners)."""
    if da._HAS_PYARROW:
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)


def _row(cell, year, month, v):
    lat, lon = (float(x) + 0.05 for x in cell.split("_"))
    ym = year * 100 + month
    return {
        "cell_id": cell, "cell_lat": lat, "cell_lon": lon, "year": year, "month": month,
        "raw_modis": float(v) if year <= 2025 else np.nan,
        "raw_viirs": float(v * 7) if ym >= 201201 else np.nan,
        "harmonized_count": float(v * 7),
        "harmonized_celldays": float(min(v, 31)),
    }


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    rows = []
    for y in range(2003, 2026):          # cell A burns every October except 2010 (a real zero year)
        if y != 2010:
            rows.append(_row(CELL_A, y, 10, y - 2000))
    rows.append(_row(CELL_B, 2015, 10, 4))
    rows.append(_row(CELL_C, 2003, 1, 1))  # defines the start of the coverage window
    rows.append(_row(CELL_C, 2026, 7, 2))  # defines the end of the coverage window
    rows.append(_row(CELL_A, 2026, 7, 9))
    _write(pd.DataFrame(rows), proc / "cells_monthly.parquet")

    fires = pd.DataFrame({
        "sensor": ["VIIRS", "VIIRS", "VIIRS", "VIIRS"],
        "satellite": ["S-NPP", "NOAA-20", "NOAA-21", "S-NPP"],
        "date": ["2026-07-05", "2026-07-06", "2026-07-06", "2026-07-07"],
        "time": ["0800", "0850", "0900", "0810"],
        "lat": [30.05, 30.06, 30.07, 30.3],   # last point sits exactly on a grid line
        "lon": [75.05, 75.06, 75.07, 75.05],
        "confidence": ["high"] * 4, "frp": [5.0, 6.0, 7.0, 8.0],
        "daynight": ["D"] * 4, "type": [0] * 4, "source": ["archive"] * 4,
    })
    _write(fires, proc / "fires.parquet")

    monkeypatch.setattr(da, "PROCESSED_DIR", proc)
    monkeypatch.setattr(da, "SAMPLE_DIR", tmp_path / "data" / "sample")
    da.clear_all_caches()
    yield proc
    da.clear_all_caches()


def test_cells_in_aoi_all_formats():
    assert set(da.cells_in_aoi((29.9, 30.2, 74.9, 75.2))) == {CELL_A, CELL_B}
    assert set(da.cells_in_aoi({"type": "bbox", "coords": (29.9, 30.2, 74.9, 75.2)})) == {CELL_A, CELL_B}
    assert da.cells_in_aoi(point=(30.05, 75.05), radius_km=5) == [CELL_A]
    assert set(da.cells_in_aoi(polygon=[(29.9, 74.9), (30.2, 74.9), (30.2, 75.2), (29.9, 75.2)])) == {CELL_A, CELL_B}
    with pytest.raises(ValueError):
        da.cells_in_aoi((32.6, 27.6, 73.8, 77.6))   # min > max


def test_zero_rule():
    h = da.monthly_series([CELL_A], "2010-01", "2010-12")
    assert h.loc[h.month == 10, "value"].item() == 0.0                    # zero year inside window = 0
    v = da.monthly_series([CELL_A], "2011-01", "2011-12", mode="raw_viirs")
    assert v["value"].isna().all()                                        # VIIRS before 2012 = no data
    m = da.monthly_series([CELL_A], "2026-01", "2026-12", mode="raw_modis")
    assert m["value"].isna().all()                                        # no MODIS in 2026
    late = da.monthly_series([CELL_A], "2026-08", "2026-12")
    assert late["value"].isna().all()                                     # after the last data month
    assert list(h.columns) == ["year", "month", "value"]


def test_calendar_matrix_shape_and_totals():
    cm = da.calendar_matrix([CELL_A])
    assert cm.shape == (24, 12) and list(cm.columns) == list(range(1, 13))
    assert cm.loc[2015, 10] == 15 * 7 and cm.loc[2015, 3] == 0.0
    assert pd.isna(cm.loc[2026, 8])


def test_vs_median_baseline():
    r = da.vs_median([CELL_A], 2025, 10)
    expected = [0.0 if y == 2010 else (y - 2000) * 7.0 for y in range(2003, 2025)]  # 2003-2024, zero kept
    assert r["n_baseline_years"] == len(expected) == 22
    assert r["median"] == pytest.approx(np.median(expected))
    assert r["p90"] == pytest.approx(np.percentile(expected, 90))
    assert r["current"] == 25 * 7 and r["unusual"] is True and r["insufficient"] is False
    for k in ("current", "median", "p25", "p75", "p90", "pct_rank", "unusual", "z"):
        assert k in r


def test_vs_median_insufficient(monkeypatch):
    monkeypatch.setattr(da, "MIN_BASELINE_YEARS", 100)
    da.clear_all_caches()
    r = da.vs_median([CELL_A], 2025, 10)
    assert r["insufficient"] is True and r["median"] is None and r["unusual"] is False


def test_fire_now_and_compare():
    f = da.fire_now([CELL_A])
    assert (f["year"], f["month"]) == (2026, 7)
    assert f["data_provenance"] == "standard_archive"
    c = da.compare([CELL_A], 2015, year_b=2016)
    assert c["total_a"] == 105.0 and c["total_b"] == 112.0


def test_hotspots_noaa21_grid_edge_and_cap():
    hs = da.hotspots([CELL_A], "2026-07-01", "2026-07-31")
    assert "NOAA-21" not in set(hs["satellite"])
    assert len(hs) == 2 and hs.attrs["total"] == 2
    edge = da.hotspots(["30.3_75.0"], "2026-07-01", "2026-07-31")
    assert len(edge) == 1                      # 30.3 lands in cell 30.3, not 30.2
    capped = da.hotspots([CELL_A], "2026-07-01", "2026-07-31", max_rows=1)
    assert len(capped) == 1 and capped.attrs["sampled"] is True


def test_validation_and_csv():
    assert da.get_validation() == {}           # file missing -> {} (never invented)
    assert da.to_csv(pd.DataFrame({"a": [1]})).startswith(b"a")

"""
Tests for src/spatial_analysis.py on small synthetic tables.
All values are detected hotspots, not burned area (made-up test numbers only).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import spatial_analysis as sa  # noqa: E402


def _row(cell_id, year, month, count):
    lat, lon = map(float, cell_id.split("_"))
    return {
        "cell_id": cell_id,
        "cell_lat": round(lat + 0.05, 2),
        "cell_lon": round(lon + 0.05, 2),
        "year": year,
        "month": month,
        "raw_modis": 0.0,
        "raw_viirs": float(count),
        "harmonized_count": float(count),
        "harmonized_celldays": 1.0,
    }


YEARS = list(range(2003, 2026))  # 23 years


@pytest.fixture
def trend_table():
    rows = []
    for i, year in enumerate(YEARS):
        rows.append(_row("30.0_75.0", year, 11, 1 + i))          # rising
        rows.append(_row("30.0_75.1", year, 11, 30 - i))         # falling
        rows.append(_row("30.0_75.2", year, 11, 5 if i % 2 else 6))  # flat
    for year in (2005, 2010, 2015):
        rows.append(_row("30.0_75.3", year, 11, 50))             # only 3 years
    return pd.DataFrame(rows)


def test_trend_labels(trend_table):
    result = sa.trend_per_cell(trend_table).set_index("cell_id")
    assert result.loc["30.0_75.0", "trend"] == sa.TREND_UP
    assert result.loc["30.0_75.0", "slope_per_year"] == pytest.approx(1.0)
    assert result.loc["30.0_75.1", "trend"] == sa.TREND_DOWN
    assert result.loc["30.0_75.2", "trend"] == sa.TREND_NONE
    # Fewer than 5 active years -> left out.
    assert "30.0_75.3" not in result.index


def test_trend_missing_years_count_as_zero():
    # Fires only in the first 6 years, then none: should be Decreasing.
    rows = [_row("31.0_76.0", year, 10, 20) for year in YEARS[:6]]
    result = sa.trend_per_cell(pd.DataFrame(rows))
    assert result["active_years"].iloc[0] == 6
    assert result["trend"].iloc[0] == sa.TREND_DOWN


@pytest.fixture
def cluster_table():
    # 15 x 15 grid of cells (lat 30.0-31.4, lon 75.0-76.4), value 1 everywhere,
    # with a 3 x 3 block of 100s around lat 30.7, lon 75.7.
    rows = []
    for r in range(15):
        for c in range(15):
            cell = f"{30 + r / 10:.1f}_{75 + c / 10:.1f}"
            value = 100 if (6 <= r <= 8 and 6 <= c <= 8) else 1
            rows.append(_row(cell, 2020, 11, value))
    return pd.DataFrame(rows)


def test_cluster_hot_spot(cluster_table):
    result = sa.hotspot_clusters(cluster_table, 2020, (10, 11)).set_index("cell_id")
    assert len(result) == 225
    assert result.loc["30.7_75.7", "cluster"] == sa.HOT
    assert result.loc["30.7_75.7", "n_neighbours"] == 9
    assert result.loc["30.0_75.0", "cluster"] == sa.NOT_SIG
    assert result.loc["30.0_75.0", "n_neighbours"] == 4  # corner cell


def test_cluster_month_filter_gives_no_signal(cluster_table):
    # No fires in March -> all zero -> nothing significant.
    result = sa.hotspot_clusters(cluster_table, 2020, (3,))
    assert (result["cluster"] == sa.NOT_SIG).all()


@pytest.fixture
def chronic_table():
    rows = []
    for i, year in enumerate(YEARS):
        rows.append(_row("29.5_75.5", year, 10, 10))              # every year
        if i % 2 == 0:
            rows.append(_row("29.5_75.6", year, 11, 10))          # 12 of 23
        rows.append(_row("29.5_75.7", year, 4, 10))               # April only
    return pd.DataFrame(rows)


def test_chronic_burning(chronic_table):
    result = sa.chronic_burning(chronic_table, months=(10, 11)).set_index("cell_id")
    assert bool(result.loc["29.5_75.5", "chronic"]) is True
    assert result.loc["29.5_75.5", "burn_years"] == 23
    assert bool(result.loc["29.5_75.6", "chronic"]) is False
    assert result.loc["29.5_75.6", "burn_years"] == 12
    # April fires are outside the season -> cell not listed.
    assert "29.5_75.7" not in result.index


def test_aoi_summary(chronic_table):
    summary = sa.aoi_summary(
        ["29.5_75.5", "29.5_75.6", "99.9_99.9"],  # last id has no data
        cells_df=chronic_table,
        area_km2=1000.0,
    )
    assert summary["label"] == sa.HOTSPOT_LABEL
    assert summary["active_cells"] == 2
    assert summary["total_harmonized_count"] == pytest.approx(230 + 120)
    assert summary["density_per_1000_km2"] == pytest.approx(350)
    # Whole table: 230 (every year) + 120 (12 years) + 230 (April cell).
    assert summary["india_total"] == pytest.approx(580)
    assert summary["share_of_india_pct"] == pytest.approx(100 * 350 / 580)
    top = summary["top_cells"]
    assert list(top["cell_id"]) == ["29.5_75.5", "29.5_75.6"]
    assert top["share_of_aoi_pct"].sum() == pytest.approx(100)


def test_aoi_summary_empty_area(chronic_table):
    summary = sa.aoi_summary([], cells_df=chronic_table)
    assert summary["active_cells"] == 0
    assert summary["total_harmonized_count"] == 0
    assert summary["top_cells"].empty

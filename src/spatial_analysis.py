"""
FIREWATCH spatial analysis on the 0.1 degree grid.

Input: cells_monthly.parquet (via data_access.load_cells_monthly()), columns
    cell_id, cell_lat, cell_lon, year, month, raw_modis, raw_viirs,
    harmonized_count, harmonized_celldays
The file only stores cell-months WITH fires, so a missing row means 0.

Every number here is a count of DETECTED HOTSPOTS (VIIRS-equivalent,
harmonized), not burned area. Pre-2012 values are MODIS x monthly ratio.

Functions
---------
trend_per_cell(cells_df, years)         Theil-Sen slope + Mann-Kendall p per cell
hotspot_clusters(cells_df, year, months) Getis-Ord Gi* z-score on the grid (3x3)
chronic_burning(cells_df, months)        cells burning in >= 80% of years
aoi_summary(cell_ids)                    totals for an area of interest

Libraries: pandas, numpy, scipy only.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

HOTSPOT_LABEL = "Detected hotspots, not burned area"

# Study period: full years only (2026 is still in progress).
DEFAULT_YEARS = (2003, 2025)

# Significance settings.
P_VALUE_LIMIT = 0.05      # trend test
Z_LIMIT = 1.96            # Gi* hot / cold spot (about 95 %)
MIN_ACTIVE_YEARS = 5      # trend: need fires in at least 5 years
CHRONIC_SHARE = 0.8       # chronic: fires in at least 80 % of years

# Labels (also used by the map legends in geo.py).
TREND_UP = "Increasing"
TREND_DOWN = "Decreasing"
TREND_NONE = "No significant trend"
HOT = "Hot spot"
COLD = "Cold spot"
NOT_SIG = "Not significant"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _check_columns(cells_df, needed):
    missing = [c for c in needed if c not in cells_df.columns]
    if missing:
        raise ValueError(f"cells_df is missing columns: {missing}")


def _year_list(years):
    """(2003, 2025) -> [2003, 2004, ..., 2025] (both ends included)."""
    start, end = int(years[0]), int(years[1])
    if start > end:
        raise ValueError("years must be (start, end) with start <= end")
    return list(range(start, end + 1))


def _cell_positions(cells_df):
    """One row per cell_id with its centre lat/lon."""
    return (
        cells_df[["cell_id", "cell_lat", "cell_lon"]]
        .drop_duplicates("cell_id")
        .set_index("cell_id")
    )


def _annual_table(cells_df, years):
    """
    Wide table: one row per cell, one column per year, value = sum of
    harmonized_count in that year. Missing years are filled with 0.
    """
    year_list = _year_list(years)
    data = cells_df[cells_df["year"].between(year_list[0], year_list[-1])]
    table = data.pivot_table(
        index="cell_id",
        columns="year",
        values="harmonized_count",
        aggfunc="sum",
        fill_value=0.0,
    )
    return table.reindex(columns=year_list, fill_value=0.0).astype(float)


# ---------------------------------------------------------------------------
# 1. Trend per cell
# ---------------------------------------------------------------------------
def trend_per_cell(cells_df, years=DEFAULT_YEARS, min_active_years=MIN_ACTIVE_YEARS,
                   p_limit=P_VALUE_LIMIT):
    """
    Long-term trend of annual detected hotspots for each 0.1 degree cell.

    - annual harmonized_count per cell (years with no fires count as 0)
    - only cells with fires in >= `min_active_years` years
    - slope: scipy.stats.theilslopes (hotspots per year, robust to outliers)
    - p-value: scipy.stats.kendalltau (Mann-Kendall test against year)
    - label: Increasing / Decreasing if p < p_limit, else No significant trend

    Returns one row per cell:
    cell_id, cell_lat, cell_lon, active_years, total_count, mean_per_year,
    slope_per_year, kendall_tau, p_value, trend
    """
    _check_columns(cells_df, ["cell_id", "cell_lat", "cell_lon", "year", "harmonized_count"])
    columns = ["cell_id", "cell_lat", "cell_lon", "active_years", "total_count",
               "mean_per_year", "slope_per_year", "kendall_tau", "p_value", "trend"]

    table = _annual_table(cells_df, years)
    active_years = (table > 0).sum(axis=1)
    table = table[active_years >= min_active_years]
    if table.empty:
        return pd.DataFrame(columns=columns)

    x = np.array(table.columns, dtype=float)
    rows = []
    for cell_id, values in zip(table.index, table.to_numpy()):
        slope = stats.theilslopes(values, x)[0]
        tau, p_value = stats.kendalltau(x, values)

        # A flat series gives tau = NaN: treat it as "no trend".
        if p_value is None or not np.isfinite(p_value):
            label = TREND_NONE
        elif p_value < p_limit and tau > 0:
            label = TREND_UP
        elif p_value < p_limit and tau < 0:
            label = TREND_DOWN
        else:
            label = TREND_NONE

        rows.append({
            "cell_id": cell_id,
            "active_years": int((values > 0).sum()),
            "total_count": float(values.sum()),
            "mean_per_year": float(values.mean()),
            "slope_per_year": float(slope),
            "kendall_tau": float(tau) if np.isfinite(tau) else np.nan,
            "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            "trend": label,
        })

    result = pd.DataFrame(rows).join(_cell_positions(cells_df), on="cell_id")
    result = result[columns].sort_values("cell_id").reset_index(drop=True)
    result.attrs["note"] = f"{HOTSPOT_LABEL}. Trend of annual harmonized counts, {years[0]}-{years[1]}."
    return result


# ---------------------------------------------------------------------------
# 2. Hot / cold spot clusters (Getis-Ord Gi*)
# ---------------------------------------------------------------------------
def _cell_index(cell_ids):
    """'30.4_75.8' -> integer grid row (304) and column (758)."""
    parts = pd.Series(cell_ids).str.split("_", expand=True).astype(float)
    rows = np.rint(parts[0].to_numpy() * 10).astype(int)
    cols = np.rint(parts[1].to_numpy() * 10).astype(int)
    return rows, cols


def _neighbour_sum(grid):
    """Sum of each cell and its 8 neighbours (3x3 window), zero outside."""
    padded = np.pad(grid, 1, mode="constant", constant_values=0)
    total = np.zeros_like(grid, dtype=float)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            total += padded[1 + dr: 1 + dr + grid.shape[0], 1 + dc: 1 + dc + grid.shape[1]]
    return total


def hotspot_clusters(cells_df, year, months=None, z_limit=Z_LIMIT):
    """
    Getis-Ord Gi* statistic on the 0.1 degree grid for one year.

    Study area = every cell that appears anywhere in `cells_df` (any year),
    so cells with no fires in the chosen period count as 0.
    Value per cell = sum of harmonized_count in `year` (and `months`, if given).
    Neighbours = the 3x3 block around a cell (itself included), weight 1,
    only cells that are in the study area.

        Gi* = (sum_j w_ij x_j - mean * W_i) /
              (S * sqrt((n * W_i - W_i**2) / (n - 1)))

    with W_i = number of neighbours (binary weights, so sum w = sum w^2),
    mean and S the mean and standard deviation of all n cells.

    Label: Hot spot (z > z_limit), Cold spot (z < -z_limit), Not significant.
    A Hot spot means a cluster of many detected hotspots, not burned area.

    Returns: cell_id, cell_lat, cell_lon, value, n_neighbours, gi_z, cluster
    """
    _check_columns(cells_df, ["cell_id", "cell_lat", "cell_lon", "year", "month", "harmonized_count"])
    columns = ["cell_id", "cell_lat", "cell_lon", "value", "n_neighbours", "gi_z", "cluster"]

    positions = _cell_positions(cells_df)
    if positions.empty:
        return pd.DataFrame(columns=columns)

    period = cells_df[cells_df["year"] == int(year)]
    if months is not None:
        period = period[period["month"].isin([int(m) for m in months])]
    values = (
        period.groupby("cell_id")["harmonized_count"].sum()
        .reindex(positions.index, fill_value=0.0)
        .to_numpy(dtype=float)
    )

    # Put every cell on a dense 2D array (rows = lat, cols = lon).
    rows, cols = _cell_index(positions.index)
    rows -= rows.min()
    cols -= cols.min()
    shape = (rows.max() + 1, cols.max() + 1)
    value_grid = np.zeros(shape)
    member_grid = np.zeros(shape)
    value_grid[rows, cols] = values
    member_grid[rows, cols] = 1.0

    n = len(values)
    mean = values.mean()
    std = math.sqrt(max((values ** 2).mean() - mean ** 2, 0.0))

    local_sum = _neighbour_sum(value_grid)[rows, cols]
    weights = _neighbour_sum(member_grid)[rows, cols]   # W_i (3x3 members)

    if n < 2 or std == 0:
        z = np.zeros(n)
    else:
        denom = std * np.sqrt((n * weights - weights ** 2) / (n - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(denom > 0, (local_sum - mean * weights) / denom, 0.0)

    labels = np.where(z > z_limit, HOT, np.where(z < -z_limit, COLD, NOT_SIG))

    result = positions.reset_index()
    result["value"] = values
    result["n_neighbours"] = weights.astype(int)
    result["gi_z"] = z
    result["cluster"] = labels
    result = result[columns].sort_values("cell_id").reset_index(drop=True)
    month_text = "all months" if months is None else f"months {list(months)}"
    result.attrs["note"] = f"{HOTSPOT_LABEL}. Getis-Ord Gi* for {year}, {month_text}."
    return result


# ---------------------------------------------------------------------------
# 3. Chronic burning
# ---------------------------------------------------------------------------
def chronic_burning(cells_df, months=(10, 11), years=DEFAULT_YEARS, share=CHRONIC_SHARE):
    """
    Cells that burn in the same season almost every year.

    For each cell: the number of years (within `years`) with
    harmonized_count > 0 in any of `months`. A cell is chronic when that
    is >= `share` (80 %) of all years in the period.
    Default months (10, 11) = the Oct-Nov paddy-residue season in Punjab/Haryana.

    Returns cells with at least one burning year:
    cell_id, cell_lat, cell_lon, burn_years, n_years, burn_share, chronic
    """
    _check_columns(cells_df, ["cell_id", "cell_lat", "cell_lon", "year", "month", "harmonized_count"])
    columns = ["cell_id", "cell_lat", "cell_lon", "burn_years", "n_years", "burn_share", "chronic"]

    year_list = _year_list(years)
    data = cells_df[
        cells_df["year"].between(year_list[0], year_list[-1])
        & cells_df["month"].isin([int(m) for m in months])
        & (cells_df["harmonized_count"] > 0)
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)

    burn_years = data.groupby("cell_id")["year"].nunique().rename("burn_years")
    result = burn_years.to_frame().join(_cell_positions(cells_df)).reset_index()
    result["n_years"] = len(year_list)
    result["burn_share"] = result["burn_years"] / result["n_years"]
    # Small tolerance so e.g. 0.8 computed as 0.7999999 still counts.
    result["chronic"] = result["burn_share"] >= share - 1e-9
    result = result[columns].sort_values("cell_id").reset_index(drop=True)
    result.attrs["note"] = (
        f"{HOTSPOT_LABEL}. Years with detected hotspots in months {list(months)}, "
        f"{years[0]}-{years[1]}; chronic = at least {share:.0%} of years."
    )
    return result


# ---------------------------------------------------------------------------
# 4. AOI summary
# ---------------------------------------------------------------------------
def _cell_area_km2(lat):
    """Area (km²) of one 0.1 x 0.1 degree cell at latitude `lat` (same as grid.py)."""
    return 11.132 * 11.132 * np.cos(np.radians(lat))


def aoi_summary(cell_ids, cells_df=None, area_km2=None, years=DEFAULT_YEARS, top_n=10):
    """
    Summary numbers for an area of interest (list of cell_ids).

    cells_df  : cells_monthly table; loaded with data_access.load_cells_monthly()
                when not given.
    area_km2  : area of the drawn AOI (e.g. grid.aoi_area_km2(aoi)). When not
                given, the area of the listed 0.1 degree cells is used.
    years     : (start, end) period, both included.

    Returns a dict:
      label, years, area_km2, area_basis, active_cells,
      total_harmonized_count, density_per_1000_km2,
      india_total, share_of_india_pct, top_cells (DataFrame)
    """
    if cells_df is None:
        import data_access as da   # local import: keeps tests free of Streamlit
        cells_df = da.load_cells_monthly()
    _check_columns(cells_df, ["cell_id", "cell_lat", "cell_lon", "year", "harmonized_count"])

    year_list = _year_list(years)
    period = cells_df[cells_df["year"].between(year_list[0], year_list[-1])]
    per_cell = period.groupby("cell_id")["harmonized_count"].sum()

    wanted = pd.Index(pd.unique(pd.Series(list(cell_ids), dtype=object)))
    # Cells of the AOI that have records in the period (others are simply 0).
    aoi_counts = per_cell[per_cell.index.isin(wanted)]

    total = float(aoi_counts.sum())
    india_total = float(per_cell.sum())
    active_cells = int((aoi_counts > 0).sum())

    positions = _cell_positions(cells_df)
    if area_km2 is None:
        lats = positions.reindex(wanted)["cell_lat"].dropna()
        area_km2 = float(_cell_area_km2(lats.to_numpy()).sum())
        area_basis = "sum of the selected 0.1° cells"
    else:
        area_km2 = float(area_km2)
        area_basis = "drawn area of interest"

    top = (
        aoi_counts[aoi_counts > 0]
        .sort_values(ascending=False)
        .head(top_n)
        .rename("harmonized_count")
        .to_frame()
        .join(positions)
        .reset_index()
        .rename(columns={"index": "cell_id"})
    )
    top["share_of_aoi_pct"] = (100 * top["harmonized_count"] / total) if total > 0 else 0.0
    top = top[["cell_id", "cell_lat", "cell_lon", "harmonized_count", "share_of_aoi_pct"]]

    return {
        "label": HOTSPOT_LABEL,
        "years": (year_list[0], year_list[-1]),
        "area_km2": area_km2,
        "area_basis": area_basis,
        "active_cells": active_cells,
        "total_harmonized_count": total,
        "density_per_1000_km2": (1000 * total / area_km2) if area_km2 > 0 else float("nan"),
        "india_total": india_total,
        "share_of_india_pct": (100 * total / india_total) if india_total > 0 else 0.0,
        "top_cells": top,
    }

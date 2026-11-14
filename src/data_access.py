# src/data_access.py
"""
FIREWATCH Backend Data Access Layer (src/data_access.py)
Owner: Backend Junior (branch: feature/backend)

Plain-function Streamlit backend (no FastAPI/uvicorn/pydantic/httpx).
- All loaders and query functions use @st.cache_data.
- Strictly adheres to CLAUDE.md data contracts and column names.
- ZERO RULE: `cells_monthly.parquet` stores only cell-months WITH fires.
  Inside each column's valid coverage window (detected from data), a missing
  cell-month = 0.0; outside the coverage window = None (NaN in DataFrames).
- Historical baselines (`vs_median`) include zero-fire years inside the valid
  coverage window, use ALL other valid years (never just the user's date range),
  and exclude the target year from its own baseline.
- Core satellites: MODIS Aqua, VIIRS S-NPP, VIIRS NOAA-20 (NOAA-21 excluded).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

try:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    ds = None
    pq = None
    _HAS_PYARROW = False

F = TypeVar("F", bound=Callable[..., Any])

try:
    import streamlit as st
    cache_data = st.cache_data(show_spinner=False)
except ImportError:
    # Lightweight fallback decorator with .clear() when streamlit isn't installed in test runner
    class _FallbackCache:
        def __call__(self, func: F) -> F:
            func.clear = lambda: None  # type: ignore[attr-defined]
            return func
    cache_data = _FallbackCache()
    st = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Paths, Constants & Schemas
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SAMPLE_DIR = ROOT_DIR / "data" / "sample"

PUNJAB_HARYANA_BBOX: Tuple[float, float, float, float] = (27.6, 32.6, 73.8, 77.6)

DEFAULT_AOI: Dict[str, Any] = {
    "type": "bbox",
    "coords": PUNJAB_HARYANA_BBOX,
    "name": "Punjab + Haryana",
}
DEFAULT_START_DATE = "2003-01-01"
DEFAULT_END_DATE = "2025-12-31"

CELLS_MONTHLY_COLUMNS = [
    "cell_id",
    "cell_lat",
    "cell_lon",
    "year",
    "month",
    "raw_modis",
    "raw_viirs",
    "harmonized_count",
    "harmonized_celldays",
]

FIRES_COLUMNS = [
    "sensor",
    "satellite",
    "date",
    "time",
    "lat",
    "lon",
    "confidence",
    "frp",
    "daynight",
    "type",
    "source",
]

CELL_DAYS_COLUMNS = [
    "cell_id",
    "cell_lat",
    "cell_lon",
    "date",
    "sensor",
    "satellite",
    "n_hotspots",
    "frp_sum",
]

RATIOS_COLUMNS = [
    "month",
    "ratio_count",
    "ratio_celldays",
    "fit_years",
    "kind",
]

MIN_BASELINE_YEARS = 5      # fewer valid baseline years -> "insufficient historical data"
DEFAULT_MAX_HOTSPOT_ROWS = 20000  # cap rows sent to the map / CSV preview

VALID_MODES = ("harmonized", "raw_modis", "raw_viirs")
VALID_METRICS = ("count", "celldays")

# Strict approved core satellites: MODIS Aqua, VIIRS S-NPP, VIIRS NOAA-20 (excludes NOAA-21 / J2)
APPROVED_CORE_SATELLITES = {
    "AQUA", "A",
    "SNPP", "S-NPP", "N", "SUOMI-NPP", "SUOMI NPP",
    "NOAA-20", "NOAA20", "1", "J1", "JPSS-1",
}


# ---------------------------------------------------------------------------
# Multi-Page Shared Session State Helper
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize shared AOI and date filters in `st.session_state` for all pages."""
    if st is None:
        return
    defaults: Dict[str, Any] = {
        "aoi": DEFAULT_AOI.copy(),
        "start_date": DEFAULT_START_DATE,
        "end_date": DEFAULT_END_DATE,
        "mode": "harmonized",
        "metric": "count",
        "metric_used": "count",
        "selected_year": 2025,
        "selected_month": 11,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def clear_all_caches() -> None:
    """Clear all @st.cache_data function caches."""
    for fn in (
        load_cells_monthly,
        _unique_grid_cells,
        _column_valid_year_months,
        cells_in_aoi,
        monthly_series,
        calendar_matrix,
        vs_median,
        fire_now,
        compare,
        hotspots,
        get_validation,
        load_ratios,
        to_csv,
    ):
        if hasattr(fn, "clear"):
            fn.clear()


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------
def _resolve_file(filename: str, fallbacks: Optional[Sequence[str]] = None) -> Optional[Path]:
    """Locate `filename` in data/processed/ first, then data/sample/."""
    candidates = [filename] + list(fallbacks or [])
    for name in candidates:
        p_proc = PROCESSED_DIR / name
        if p_proc.exists():
            return p_proc
        p_samp = SAMPLE_DIR / name
        if p_samp.exists():
            return p_samp
    return None


def _read_parquet_or_pickle(
    path: Path,
    columns: Optional[List[str]] = None,
    filters: Optional[List[Tuple[str, str, Any]]] = None,
) -> pd.DataFrame:
    """Read Parquet via PyArrow (or pickle fallback when pyarrow is unavailable in minimal test runners)."""
    if _HAS_PYARROW:
        return pd.read_parquet(path, columns=columns, filters=filters or None)
    df = pd.read_pickle(path)
    if filters:
        for col, op, val in filters:
            if col not in df.columns:
                continue
            if op == ">=":
                df = df[df[col] >= val]
            elif op == "<=":
                df = df[df[col] <= val]
            elif op == "==":
                df = df[df[col] == val]
    if columns is not None:
        df = df[columns].copy()
    return df.copy()


def _resolve_column_and_metric(mode: str, metric: str) -> Tuple[str, str]:
    """
    Map (mode, metric) to the exact contract column in `cells_monthly.parquet`
    and return `(column_name, metric_used)`.
    Note: raw_modis and raw_viirs ignore `metric` in `cells_monthly.parquet`
    and return `metric_used='count'` so the UI can grey out the Cell-days toggle.
    """
    mode_clean = (mode or "harmonized").strip().lower()
    metric_clean = (metric or "count").strip().lower()

    if mode_clean not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}.")
    if metric_clean not in VALID_METRICS:
        raise ValueError(f"Invalid metric '{metric}'. Must be one of {VALID_METRICS}.")

    if mode_clean == "raw_modis":
        return "raw_modis", "count"
    if mode_clean == "raw_viirs":
        return "raw_viirs", "count"

    if metric_clean == "celldays":
        return "harmonized_celldays", "celldays"
    return "harmonized_count", "count"


def _parse_year_month(date_or_year: Union[str, int, pd.Timestamp], is_end: bool = False) -> Tuple[int, int]:
    """Convert 'YYYY-MM-DD', 'YYYY-MM', int year, or Timestamp into (year, month)."""
    if isinstance(date_or_year, (int, np.integer)) or (str(date_or_year).strip().isdigit() and len(str(date_or_year).strip()) == 4):
        year = int(str(date_or_year).strip())
        if not 2000 <= year <= 2100:
            raise ValueError(f"Year {year} is outside the supported range 2000-2100.")
        return year, (12 if is_end else 1)
    s = str(date_or_year).strip()
    try:
        dt = pd.to_datetime(s)
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime) as exc:
        raise ValueError(f"Invalid date '{date_or_year}'. Use YYYY-MM or YYYY-MM-DD.") from exc
    if not 2000 <= int(dt.year) <= 2100:
        raise ValueError(f"Date '{date_or_year}' is outside the supported range 2000-2100.")
    return int(dt.year), int(dt.month)


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized great-circle Haversine distance in kilometers."""
    r_earth = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r_earth * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _points_in_polygon_latlon(
    lats: np.ndarray,
    lons: np.ndarray,
    poly_latlon: Sequence[Sequence[float]],
) -> np.ndarray:
    """
    Vectorized ray-casting point-in-polygon check where polygon vertices
    are a sequence of `(lat, lon)` pairs.
    """
    ring = [(float(pt[0]), float(pt[1])) for pt in poly_latlon]
    if len(ring) < 3:
        raise ValueError("polygon must contain at least 3 (lat, lon) vertices.")

    n = len(ring)
    inside = np.zeros(len(lats), dtype=bool)
    lat1, lon1 = ring[0]
    for i in range(1, n + 1):
        lat2, lon2 = ring[i % n]
        cond_lat = (lats > min(lat1, lat2)) & (lats <= max(lat1, lat2))
        cond_lon = lons <= max(lon1, lon2)
        mask = cond_lat & cond_lon
        if np.any(mask):
            if lat1 != lat2:
                lon_intersect = (lats[mask] - lat1) * (lon2 - lon1) / (lat2 - lat1) + lon1
            else:
                lon_intersect = lon1
            flip = (lon1 == lon2) | (lons[mask] <= lon_intersect)
            idx = np.where(mask)[0]
            inside[idx[flip]] = ~inside[idx[flip]]
        lat1, lon1 = lat2, lon2
    return inside


def _assign_cell_id(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Floor lat/lon to 0.1° grid southwest corner cell_id formatted as '30.4_75.8'."""
    c_lat = np.floor(lat.astype(float) * 10.0 + 1e-9) / 10.0
    c_lon = np.floor(lon.astype(float) * 10.0 + 1e-9) / 10.0
    return c_lat.map(lambda v: f"{v:.1f}") + "_" + c_lon.map(lambda v: f"{v:.1f}")


def _filter_core_satellites(df: pd.DataFrame) -> pd.DataFrame:
    """Strict allowlist: keep MODIS Aqua, VIIRS S-NPP, and VIIRS NOAA-20; exclude NOAA-21."""
    if "satellite" not in df.columns or df.empty:
        return df
    sat_upper = df["satellite"].astype(str).str.strip().str.upper()
    return df[sat_upper.isin(APPROVED_CORE_SATELLITES)].copy()


# ---------------------------------------------------------------------------
# Cached Dataset Loaders & Coverage Window Detection
# ---------------------------------------------------------------------------
@cache_data
def load_cells_monthly() -> pd.DataFrame:
    """
    Load `data/processed/cells_monthly.parquet` (or sample fallback) without
    renaming columns or overwriting sensor NaNs.
    """
    path = _resolve_file("cells_monthly.parquet", fallbacks=["cells_monthly_sample.parquet"])
    if path is None:
        return pd.DataFrame(columns=CELLS_MONTHLY_COLUMNS)

    df = _read_parquet_or_pickle(path)
    missing_cols = [c for c in CELLS_MONTHLY_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"cells_monthly.parquet is missing contract columns: {missing_cols}")

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["cell_lat"] = df["cell_lat"].astype(float)
    df["cell_lon"] = df["cell_lon"].astype(float)
    return df


@cache_data
def _unique_grid_cells() -> pd.DataFrame:
    """Cached table of unique `(cell_id, cell_lat, cell_lon)` in the dataset."""
    df = load_cells_monthly()
    if df.empty:
        return pd.DataFrame(columns=["cell_id", "cell_lat", "cell_lon"])
    return (
        df[["cell_id", "cell_lat", "cell_lon"]]
        .drop_duplicates(subset=["cell_id"])
        .sort_values("cell_id")
        .reset_index(drop=True)
    )


@cache_data
def _column_valid_year_months() -> Dict[str, set]:
    """
    ZERO RULE COVERAGE DETECTION:
    `cells_monthly.parquet` stores ONLY cell-months WITH fires.
    Detect each column's valid `(year, month)` coverage window from the data:
      - For each column (`raw_modis`, `raw_viirs`, `harmonized_count`, `harmonized_celldays`),
        find the earliest `(year, month)` and latest `(year, month)` where that column has
        non-NaN observations anywhere in the dataset.
      - Every `(year, month)` inside `[min_ym .. max_ym]` for that column is inside its active
        coverage window (unless all rows present in the dataset for that `(year, month)` have
        `col` explicitly set to `NaN`, e.g., when a sensor was offline while another sensor
        recorded fires).
      - Inside a column's valid coverage window: a missing cell-month means 0.0 fires.
      - Outside a column's valid coverage window: None / NaN (never 0).
    """
    df = load_cells_monthly()
    value_cols = ["raw_modis", "raw_viirs", "harmonized_count", "harmonized_celldays"]
    if df.empty:
        return {col: set() for col in value_cols}

    ym_counts = df.groupby(["year", "month"], as_index=False)[value_cols].count()
    # Set of (year, month) pairs that appear in the table at all
    present_yms = set(zip(ym_counts["year"].astype(int), ym_counts["month"].astype(int)))

    valid_map: Dict[str, set] = {}
    for col in value_cols:
        col_mask = ym_counts[col] > 0
        if not col_mask.any():
            valid_map[col] = set()
            continue

        non_nan_yms = set(
            zip(ym_counts.loc[col_mask, "year"].astype(int), ym_counts.loc[col_mask, "month"].astype(int))
        )
        start_ym = min(non_nan_yms)
        end_ym = max(non_nan_yms)

        periods = pd.date_range(
            start=f"{start_ym[0]:04d}-{start_ym[1]:02d}-01",
            end=f"{end_ym[0]:04d}-{end_ym[1]:02d}-01",
            freq="MS",
        )
        window_yms = set(zip(periods.year.astype(int), periods.month.astype(int)))

        # If a (year, month) is present in the table (because another sensor had fires)
        # but THIS column has 0 non-NaN entries in that month, exclude it from valid coverage.
        explicitly_nan_yms = present_yms - non_nan_yms
        valid_map[col] = window_yms - explicitly_nan_yms

    return valid_map


# ---------------------------------------------------------------------------
# 1. cells_in_aoi(bbox=None, point=None, radius_km=None, polygon=None)
# ---------------------------------------------------------------------------
@cache_data
def cells_in_aoi(
    bbox: Optional[Union[Tuple[float, float, float, float], List[float], Dict[str, Any]]] = None,
    point: Optional[Tuple[float, float]] = None,
    radius_km: Optional[float] = None,
    polygon: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[str]:
    """
    Single function mapping an Area of Interest (AOI) to a sorted list of `cell_id`s.

    Supported inputs:
    1) `bbox=(min_lat, max_lat, min_lon, max_lon)` e.g. `(27.6, 32.6, 73.8, 77.6)`,
       or dict `{"min_lat": 27.6, "max_lat": 32.6, "min_lon": 73.8, "max_lon": 77.6}`.
    2) `point=(lat, lon)` and `radius_km=float` (Haversine distance in km).
    3) `polygon=[(lat, lon), (lat, lon), ...]` (list of `(lat, lon)` vertices).
    4) Map dict passed as first argument (`bbox`):
       `{"type": "bbox" | "polygon" | "point", "coords": ..., "radius_km": ...}`
       (directly from `st.session_state["aoi"]`).
    """
    cells_df = _unique_grid_cells()
    if cells_df.empty:
        return []

    if isinstance(bbox, dict):
        aoi_dict = bbox
        aoi_type = str(aoi_dict.get("type", "bbox")).strip().lower()
        if aoi_type in ("bbox", "box"):
            if "coords" in aoi_dict and aoi_dict["coords"] is not None:
                bbox = tuple(aoi_dict["coords"])
            else:
                bbox = (
                    float(aoi_dict["min_lat"]),
                    float(aoi_dict["max_lat"]),
                    float(aoi_dict["min_lon"]),
                    float(aoi_dict["max_lon"]),
                )
        elif aoi_type == "point":
            point = (
                tuple(aoi_dict["coords"])
                if "coords" in aoi_dict and aoi_dict["coords"] is not None
                else (float(aoi_dict["lat"]), float(aoi_dict["lon"]))
            )
            radius_km = float(aoi_dict.get("radius_km", radius_km if radius_km is not None else 50.0))
            bbox = None
        elif aoi_type == "polygon":
            polygon = aoi_dict.get("coords") or aoi_dict.get("polygon") or aoi_dict.get("coordinates")
            bbox = None
        else:
            raise ValueError(f"Unsupported AOI dict type: '{aoi_type}'. Expected 'bbox', 'point', or 'polygon'.")

    lats = cells_df["cell_lat"].to_numpy()
    lons = cells_df["cell_lon"].to_numpy()

    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox must have 4 elements: (min_lat, max_lat, min_lon, max_lon).")
        min_lat, max_lat, min_lon, max_lon = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90 and -180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("bbox latitudes must be within -90..90 and longitudes within -180..180.")
        if min_lat > max_lat or min_lon > max_lon:
            raise ValueError(
                f"Invalid bbox ({min_lat}, {max_lat}, {min_lon}, {max_lon}): "
                "expected (min_lat, max_lat, min_lon, max_lon) with min <= max."
            )
        mask = (lats >= min_lat) & (lats <= max_lat) & (lons >= min_lon) & (lons <= max_lon)
        return cells_df.loc[mask, "cell_id"].tolist()

    if point is not None:
        if len(point) != 2:
            raise ValueError("point must be a 2-tuple: (lat, lon).")
        if radius_km is None or float(radius_km) < 0:
            raise ValueError("radius_km must be a non-negative number when point is provided.")
        lat0, lon0 = float(point[0]), float(point[1])
        if not (-90 <= lat0 <= 90 and -180 <= lon0 <= 180):
            raise ValueError("point must be (lat, lon) with lat in -90..90 and lon in -180..180.")
        dists = _haversine_km(lat0, lon0, lats, lons)
        mask = dists <= float(radius_km)
        return cells_df.loc[mask, "cell_id"].tolist()

    if polygon is not None:
        inside = _points_in_polygon_latlon(lats, lons, polygon)
        return cells_df.loc[inside, "cell_id"].tolist()

    return cells_df["cell_id"].tolist()


# ---------------------------------------------------------------------------
# 2. monthly_series(cells, start, end, mode="harmonized", metric="count")
# ---------------------------------------------------------------------------
@cache_data
def monthly_series(
    cells: Sequence[str],
    start: Union[str, int] = "2003-01-01",
    end: Union[str, int] = "2025-12-31",
    mode: str = "harmonized",
    metric: str = "count",
) -> pd.DataFrame:
    """
    Compute the monthly time series for `cells` between `start` and `end`.

    ZERO RULE:
    - Returns DataFrame with exact columns: `['year', 'month', 'value']`.
    - Inside the column's valid coverage window: missing cell-month = 0.0.
    - Outside the column's valid coverage window: NaN / None (never 0).
    - Sets `df.attrs['metric_used']` ('count' or 'celldays') and updates `st.session_state['metric_used']`.
    """
    col_name, metric_used = _resolve_column_and_metric(mode, metric)
    if st is not None:
        try:
            st.session_state["metric_used"] = metric_used
        except Exception:
            pass

    start_y, start_m = _parse_year_month(start, is_end=False)
    end_y, end_m = _parse_year_month(end, is_end=True)
    if (start_y, start_m) > (end_y, end_m):
        raise ValueError(f"start ({start}) cannot be after end ({end}).")

    periods = pd.date_range(
        start=f"{start_y:04d}-{start_m:02d}-01",
        end=f"{end_y:04d}-{end_m:02d}-01",
        freq="MS",
    )
    spine = pd.DataFrame({
        "year": periods.year.astype(int),
        "month": periods.month.astype(int),
    })

    df = load_cells_monthly()
    valid_yms = _column_valid_year_months().get(col_name, set())

    if df.empty or not valid_yms or (cells is not None and len(cells) == 0):
        spine["value"] = np.nan
        spine.attrs["metric_used"] = metric_used
        return spine[["year", "month", "value"]]

    in_valid_window = pd.Series(
        [(int(y), int(m)) in valid_yms for y, m in zip(spine["year"], spine["month"])],
        index=spine.index,
    )

    sub = df
    if cells is not None:
        sub = sub[sub["cell_id"].isin(set(cells))]
    sub = sub[(sub["year"] >= start_y) & (sub["year"] <= end_y)]

    grouped = (
        sub.groupby(["year", "month"], as_index=False)[col_name]
        .sum(min_count=1)
        .rename(columns={col_name: "raw_sum"})
    )

    merged = pd.merge(spine, grouped, on=["year", "month"], how="left")
    merged["value"] = np.where(
        in_valid_window,
        merged["raw_sum"].fillna(0.0),
        np.nan,
    )

    out = merged[["year", "month", "value"]].copy()
    out.attrs["metric_used"] = metric_used
    return out


# ---------------------------------------------------------------------------
# 3. calendar_matrix(cells, mode="harmonized", metric="count")
# ---------------------------------------------------------------------------
@cache_data
def calendar_matrix(
    cells: Sequence[str],
    mode: str = "harmonized",
    metric: str = "count",
) -> pd.DataFrame:
    """
    Build a `year × month (1..12)` matrix for the burning-activity calendar across ALL dataset years.
    - Index: `year`
    - Columns: `1, 2, ..., 12`
    - Inside valid sensor coverage window: missing cell-month = 0.0
    - Outside valid sensor coverage window: NaN
    - Sets `matrix.attrs['metric_used']` ('count' or 'celldays').
    """
    _, metric_used = _resolve_column_and_metric(mode, metric)
    df = load_cells_monthly()
    if df.empty:
        empty_mat = pd.DataFrame(columns=list(range(1, 13)), dtype=float)
        empty_mat.index.name = "year"
        empty_mat.columns.name = "month"
        empty_mat.attrs["metric_used"] = metric_used
        return empty_mat

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    ts = monthly_series(
        cells=cells,
        start=f"{min_year:04d}-01-01",
        end=f"{max_year:04d}-12-31",
        mode=mode,
        metric=metric,
    )

    mat = ts.pivot(index="year", columns="month", values="value")
    mat = mat.reindex(index=list(range(min_year, max_year + 1)), columns=list(range(1, 13)))
    mat.index.name = "year"
    mat.columns.name = "month"
    mat.attrs["metric_used"] = metric_used
    return mat


# ---------------------------------------------------------------------------
# 4. vs_median(cells, year, month, metric="count")
# ---------------------------------------------------------------------------
@cache_data
def vs_median(
    cells: Sequence[str],
    year: int,
    month: int,
    metric: str = "count",
) -> Dict[str, Any]:
    """
    Compare `(year, month)` harmonized activity against the historical baseline
    of that same calendar `month` across ALL OTHER valid years in the dataset
    (never restricted to the user's selected date range).

    ZERO RULE:
    - Because `calendar_matrix` fills missing cell-months inside the valid coverage
      window with `0.0`, zero-fire years are included in the historical baseline!
    - Excludes the target `year` from its own baseline.
    - Returns exact keys: `current`, `median`, `p25`, `p75`, `p90`, `pct_rank`, `unusual`, `z`, `metric_used`.
      (Values are `None` if `(year, month)` is outside coverage or `cells` is empty.)
    """
    year_int = int(year)
    month_int = int(month)
    if not 1 <= month_int <= 12:
        raise ValueError(f"month must be in 1..12, got {month}.")

    mat = calendar_matrix(cells=cells, mode="harmonized", metric=metric)
    metric_used = mat.attrs.get("metric_used", metric)

    if mat.empty or month_int not in mat.columns:
        return {
            "current": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "pct_rank": None,
            "unusual": False,
            "z": None,
            "insufficient": True,
            "n_baseline_years": 0,
            "metric_used": metric_used,
        }

    month_col = mat[month_int]
    raw_current = month_col.get(year_int, np.nan)
    current_val: Optional[float] = None if pd.isna(raw_current) else float(raw_current)

    # Exclude target year from its own baseline; keep all other valid (non-NaN) years (including 0.0 years!)
    baseline = month_col.drop(index=year_int, errors="ignore").dropna().astype(float)

    if baseline.empty or current_val is None or len(baseline) < MIN_BASELINE_YEARS:
        return {
            "current": current_val,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "pct_rank": None,
            "unusual": False,
            "z": None,
            "insufficient": True,
            "n_baseline_years": int(len(baseline)),
            "metric_used": metric_used,
        }

    base_arr = baseline.to_numpy(dtype=float)
    median_val = float(np.median(base_arr))
    p25_val = float(np.percentile(base_arr, 25))
    p75_val = float(np.percentile(base_arr, 75))
    p90_val = float(np.percentile(base_arr, 90))
    pct_rank_val = float(np.mean(base_arr <= current_val) * 100.0)

    mean_val = float(np.mean(base_arr))
    std_val = float(np.std(base_arr, ddof=0))
    z_val = float((current_val - mean_val) / std_val) if std_val > 0.0 else 0.0

    return {
        "current": current_val,
        "median": median_val,
        "p25": p25_val,
        "p75": p75_val,
        "p90": p90_val,
        "pct_rank": pct_rank_val,
        "unusual": bool(current_val > p90_val),
        "z": z_val,
        "insufficient": False,
        "n_baseline_years": int(len(base_arr)),
        "metric_used": metric_used,
    }


# ---------------------------------------------------------------------------
# 5. fire_now(cells, metric="count")
# ---------------------------------------------------------------------------
@cache_data
def fire_now(
    cells: Sequence[str],
    metric: str = "count",
) -> Dict[str, Any]:
    """
    Summarize the latest valid observation month for `cells`, including its
    historical median comparison (`vs_median`) and evidence-based NRT provenance
    from `fires.parquet` (excluding NOAA-21).
    """
    mat = calendar_matrix(cells=cells, mode="harmonized", metric=metric)
    metric_used = mat.attrs.get("metric_used", metric)

    if mat.empty or (cells is not None and len(cells) == 0):
        return {
            "year": None,
            "month": None,
            "latest_observation_date": None,
            "current": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "pct_rank": None,
            "unusual": False,
            "z": None,
            "insufficient": True,
            "n_baseline_years": 0,
            "is_provisional_nrt": None,
            "data_provenance": "unknown",
            "metric_used": metric_used,
        }

    # Find latest (year, month) with non-NaN coverage in the calendar matrix
    # Latest (year, month) with a non-NaN value. Avoids DataFrame.stack(dropna=...),
    # which raises on newer pandas versions.
    valid_mask = mat.notna()
    years_with_data = [int(y) for y in mat.index if bool(valid_mask.loc[y].any())]
    if not years_with_data:
        return {
            "year": None,
            "month": None,
            "latest_observation_date": None,
            "current": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "pct_rank": None,
            "unusual": False,
            "z": None,
            "insufficient": True,
            "n_baseline_years": 0,
            "is_provisional_nrt": None,
            "data_provenance": "unknown",
            "metric_used": metric_used,
        }

    latest_year = max(years_with_data)
    latest_month = int(max(int(m) for m in mat.columns if bool(valid_mask.loc[latest_year, m])))

    stats = vs_median(cells=cells, year=latest_year, month=latest_month, metric=metric)

    # Check provenance in fires.parquet for that month (using hotspots with PyArrow pushdown)
    ym_start = f"{latest_year:04d}-{latest_month:02d}-01"
    ym_end = (pd.Timestamp(latest_year, latest_month, 1) + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
    pts = hotspots(cells=cells, start=ym_start, end=ym_end, max_rows=None)

    provenance = "unknown"
    is_nrt: Optional[bool] = None
    latest_date: Optional[str] = None  # only set from real hotspot records, never invented

    if not pts.empty:
        if "date" in pts.columns:
            latest_date = str(pts["date"].max())[:10]
        if "source" in pts.columns:
            sources = [str(s).strip().upper() for s in pts["source"].dropna().unique() if str(s).strip()]
            has_nrt = any("NRT" in s or "PROVISIONAL" in s or s == "RT" for s in sources)
            has_arch = any(tok in s for s in sources for tok in ("SP", "ARCHIVE", "C61", "C6.1", "STANDARD"))
            if has_nrt and has_arch:
                provenance = "mixed"
                is_nrt = True
            elif has_nrt:
                provenance = "provisional_nrt"
                is_nrt = True
            elif has_arch:
                provenance = "standard_archive"
                is_nrt = False

    return {
        "year": latest_year,
        "month": latest_month,
        "latest_observation_date": latest_date,
        **stats,
        "is_provisional_nrt": is_nrt,
        "data_provenance": provenance,
    }


# ---------------------------------------------------------------------------
# 6. compare(cells_a, year_a, cells_b=None, year_b=None, mode="harmonized", metric="count", month=None)
# ---------------------------------------------------------------------------
@cache_data
def compare(
    cells_a: Sequence[str],
    year_a: int,
    cells_b: Optional[Sequence[str]] = None,
    year_b: Optional[int] = None,
    mode: str = "harmonized",
    metric: str = "count",
    month: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare two years (`year_a` vs `year_b`) and/or two areas (`cells_a` vs `cells_b`).
    Evaluates totals and percentage change strictly over overlapping valid months (`overlapping_months`).
    """
    eff_cells_b = cells_b if cells_b is not None else cells_a
    eff_year_b = int(year_b) if year_b is not None else int(year_a)
    year_a_int = int(year_a)

    start_m = int(month) if month is not None else 1
    end_m = int(month) if month is not None else 12

    ts_a = monthly_series(
        cells=cells_a,
        start=f"{year_a_int:04d}-{start_m:02d}-01",
        end=f"{year_a_int:04d}-{end_m:02d}-28",
        mode=mode,
        metric=metric,
    )
    ts_b = monthly_series(
        cells=eff_cells_b,
        start=f"{eff_year_b:04d}-{start_m:02d}-01",
        end=f"{eff_year_b:04d}-{end_m:02d}-28",
        mode=mode,
        metric=metric,
    )

    valid_a = ts_a.dropna(subset=["value"]).set_index("month")["value"].to_dict()
    valid_b = ts_b.dropna(subset=["value"]).set_index("month")["value"].to_dict()

    overlapping_months = sorted(set(valid_a.keys()) & set(valid_b.keys()))
    if not overlapping_months:
        return {
            "year_a": year_a_int,
            "year_b": eff_year_b,
            "overlapping_months": [],
            "total_a": None,
            "total_b": None,
            "diff": None,
            "pct_change": None,
            "series_a": ts_a,
            "series_b": ts_b,
            "metric_used": ts_a.attrs.get("metric_used", metric),
        }

    tot_a = float(sum(valid_a[m] for m in overlapping_months))
    tot_b = float(sum(valid_b[m] for m in overlapping_months))
    diff = tot_b - tot_a
    pct_change: Optional[float] = ((diff / tot_a) * 100.0) if tot_a > 0.0 else None

    return {
        "year_a": year_a_int,
        "year_b": eff_year_b,
        "overlapping_months": overlapping_months,
        "total_a": tot_a,
        "total_b": tot_b,
        "diff": diff,
        "pct_change": pct_change,
        "series_a": ts_a,
        "series_b": ts_b,
        "metric_used": ts_a.attrs.get("metric_used", metric),
    }


# ---------------------------------------------------------------------------
# 7. hotspots(cells, start, end)
# ---------------------------------------------------------------------------
@cache_data
def hotspots(
    cells: Sequence[str],
    start: str,
    end: str,
    max_rows: Optional[int] = DEFAULT_MAX_HOTSPOT_ROWS,
) -> pd.DataFrame:
    """
    Load raw hotspot rows from `data/processed/fires.parquet` for `cells` and `[start, end]`.
    Uses PyArrow predicate pushdown (`filters=`) on `date`, `lat`, and `lon` bounding box
    FIRST so reading large India-wide Parquet files stays fast (<5s), excludes NOAA-21,
    and then filters to exact `cells`. Never renames contract columns.
    """
    path = _resolve_file("fires.parquet", fallbacks=["fires_sample.parquet"])
    if path is None or (cells is not None and len(cells) == 0):
        return pd.DataFrame(columns=FIRES_COLUMNS)

    cell_set = set(cells) if cells is not None else set()
    if cell_set:
        parts = pd.Series(list(cell_set), dtype=str).str.split("_", expand=True)
        sw_lats = parts[0].astype(float)
        sw_lons = parts[1].astype(float)
        min_lat = float(sw_lats.min()) - 1e-6
        max_lat = float(sw_lats.max()) + 0.1 + 1e-6
        min_lon = float(sw_lons.min()) - 1e-6
        max_lon = float(sw_lons.max()) + 0.1 + 1e-6
    else:
        min_lat, max_lat, min_lon, max_lon = -90.0, 90.0, -180.0, 180.0

    start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
    end_str = pd.to_datetime(end).strftime("%Y-%m-%d")

    bbox_filters: List[Tuple[str, str, Any]] = [
        ("lat", ">=", min_lat),
        ("lat", "<=", max_lat),
        ("lon", ">=", min_lon),
        ("lon", "<=", max_lon),
    ]

    if _HAS_PYARROW:
        schema = pq.read_schema(path)
        date_field = schema.field("date") if "date" in schema.names else None
        if date_field is not None and (
            "string" in str(date_field.type).lower() or "utf8" in str(date_field.type).lower()
        ):
            pa_filters = bbox_filters + [("date", ">=", start_str), ("date", "<=", end_str)]
            df = pd.read_parquet(path, filters=pa_filters)
        else:
            df = pd.read_parquet(path, filters=bbox_filters)
            if not df.empty and "date" in df.columns:
                dt_series = pd.to_datetime(df["date"])
                df = df[(dt_series >= pd.to_datetime(start_str)) & (dt_series <= pd.to_datetime(end_str))]
    else:
        df = _read_parquet_or_pickle(path, filters=bbox_filters)
        if not df.empty and "date" in df.columns:
            dt_series = pd.to_datetime(df["date"])
            df = df[(dt_series >= pd.to_datetime(start_str)) & (dt_series <= pd.to_datetime(end_str))]

    if df.empty:
        return pd.DataFrame(columns=FIRES_COLUMNS)

    df = _filter_core_satellites(df)
    if df.empty:
        return pd.DataFrame(columns=FIRES_COLUMNS)

    if cell_set:
        if "cell_id" in df.columns:
            df = df[df["cell_id"].isin(cell_set)]
        else:
            row_cells = _assign_cell_id(df["lat"], df["lon"])
            df = df[row_cells.isin(cell_set)].copy()

    keep_cols = [c for c in FIRES_COLUMNS if c in df.columns]
    out = df[keep_cols].reset_index(drop=True)
    total = int(len(out))
    # Cap rows for the map / preview; random sample with a fixed seed (reproducible).
    # Pass max_rows=None to get every row (e.g. for a full CSV export).
    if max_rows is not None and total > int(max_rows):
        out = out.sample(n=int(max_rows), random_state=42).sort_values(["date", "time"]).reset_index(drop=True)
    out.attrs["total"] = total
    out.attrs["sampled"] = bool(max_rows is not None and total > int(max_rows))
    return out


# ---------------------------------------------------------------------------
# 8. get_validation(), load_ratios() & 9. to_csv(df)
# ---------------------------------------------------------------------------
@cache_data
def get_validation() -> Dict[str, Any]:
    """
    Load held-out validation metrics from `data/processed/validation.json`.
    Returns `{}` if the file is missing (never invents metrics).
    """
    path = _resolve_file("validation.json", fallbacks=["validation_sample.json"])
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@cache_data
def load_ratios() -> pd.DataFrame:
    """Load `data/processed/ratios.parquet` (or empty DataFrame if missing)."""
    path = _resolve_file("ratios.parquet", fallbacks=["ratios_sample.parquet"])
    if path is None:
        return pd.DataFrame(columns=RATIOS_COLUMNS)
    return _read_parquet_or_pickle(path)


@cache_data
def to_csv(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to UTF-8 CSV bytes for `st.download_button`."""
    if df is None or df.empty:
        return b""
    include_idx = df.index.name is not None
    return df.to_csv(index=include_idx).encode("utf-8")

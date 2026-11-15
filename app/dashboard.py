"""FIREWATCH: Streamlit-only NASA FIRMS dashboard.

Place this file at app/dashboard.py in the team repository. It consumes the existing
src/data_access.py and src/geo.py; it does not introduce a REST API or change their schemas.
Run: streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_access as da  # noqa: E402
import geo  # noqa: E402
from frontend_utils import (  # noqa: E402
    DEFAULT_AOI, aoi_label, clamp_month_range, file_signature,
    month_end, month_options, month_start, normalize_aoi, safe_script_json,
)

st.set_page_config(page_title="FIREWATCH | NASA Satellite Fire Monitor", page_icon="🔥", layout="wide")

FIRMS_ACK = (
    "We acknowledge the use of data and/or imagery from NASA's Fire Information "
    "for Resource Management System (FIRMS), part of NASA LANCE and ESDIS. "
    "https://www.earthdata.nasa.gov/data/tools/firms"
)
MAP_LIMIT = 2_000
PREVIEW_EXPORT_LIMIT = 20_000
MAX_FULL_EXPORT_MONTHS = 12

st.markdown("""
<style>
  .stApp {background:linear-gradient(180deg,#071023 0%,#050912 60%); color:#E6ECF8;}
  [data-testid="stSidebar"] {background:linear-gradient(180deg,#101d35,#080f20); border-right:1px solid #253755;}
  h1,h2,h3 {letter-spacing:-.025em;}
  [data-testid="stMetric"] {background:#101d32; border:1px solid #263956; border-radius:14px; padding:12px;}
  .fw-intro {border-left:4px solid #FF7A1A; padding:12px 18px; margin-bottom:16px;
    background:linear-gradient(90deg,rgba(255,122,26,.10),rgba(12,20,42,.20)); border-radius:8px;}
  .fw-overline {color:#FFAE6D; font-size:.78rem; letter-spacing:.20em; font-weight:700;}
  .fw-intro h2 {margin:3px 0 0; font-size:1.55rem; color:#f5f7ff;}
  .fw-intro p {margin:6px 0 0; color:#B4C0DA;}
  [data-testid="stSidebar"] .stButton button {border:1px solid #39516e;}
</style>
""", unsafe_allow_html=True)


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_scalar(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _scalar(v) for k, v in value.items()}
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _period(cm: pd.DataFrame) -> tuple[str, str]:
    if cm is None or cm.empty or not {"year", "month"}.issubset(cm.columns):
        raise ValueError("Missing or empty cells_monthly.parquet (year/month columns required)")
    months = cm[["year", "month"]].dropna().astype(int)
    months = months[(months["month"] >= 1) & (months["month"] <= 12)]
    if months.empty:
        raise ValueError("No valid monthly observations")
    keys = months["year"] * 100 + months["month"]
    first, last = int(keys.min()), int(keys.max())
    return f"{first // 100:04d}-{first % 100:02d}", f"{last // 100:04d}-{last % 100:02d}"


def _series(cells: list[str], end: str, mode: str, metric: str, name: str) -> pd.DataFrame:
    frame = da.monthly_series(cells, "2003-01", end, mode, metric)
    if isinstance(frame, pd.Series):
        frame = frame.reset_index()
    if not {"year", "month"}.issubset(frame.columns):
        raise RuntimeError(f"monthly_series returned unexpected columns: {list(frame.columns)}")
    val = "value" if "value" in frame.columns else next((c for c in frame.columns if c not in {"year", "month"}), None)
    if val is None:
        raise RuntimeError("monthly_series returned no measure")
    return frame[["year", "month", val]].rename(columns={val: name})


@st.cache_data(show_spinner="Loading NASA FIRMS monthly data…")
def build_rows(aoi: dict, end: str, version: tuple) -> list[dict]:
    """Use data_access contracts. Return explicit 0 versus unavailable (None)."""
    cells = da.cells_in_aoi(aoi)
    frames = [
        _series(cells, end, "harmonized", "count", "harmonized_count"),
        _series(cells, end, "harmonized", "celldays", "harmonized_celldays"),
        _series(cells, end, "raw_modis", "count", "raw_modis"),
        _series(cells, end, "raw_viirs", "count", "raw_viirs"),
    ]
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=["year", "month"], how="outer", validate="one_to_one")
    result = result.sort_values(["year", "month"])
    return [
        {key: int(val) if key in ("year", "month") else _scalar(val)
         for key, val in row.items()}
        for row in result.to_dict("records")
    ]


@st.cache_data(show_spinner=False)
def provisional_month(version: tuple) -> str | None:
    """Find the first verified *core-satellite* NRT month, never guess a date."""
    try:
        import pyarrow.dataset as ds
        import pyarrow.compute as pc

        path = next((ROOT / path for path in ("data/processed/fires.parquet", "data/sample/fires.parquet")
                     if (ROOT / path).is_file()), None)
        if path is None:
            return None
        dataset = ds.dataset(str(path), format="parquet")
        names = dataset.schema.names
        if not {"source", "date", "satellite"}.issubset(names):
            return None
        nrt = ds.field("source").isin(["nrt", "NRT"])
        core = ds.field("satellite").isin([
            "Aqua", "AQUA", "A", "S-NPP", "SNPP", "N", "NOAA-20", "NOAA20", "1", "J1", "JPSS-1"
        ])
        column = dataset.to_table(columns=["date"], filter=nrt & core)["date"]
        if len(column) == 0:
            return None
        oldest = pc.min(column).as_py()
        return str(oldest)[:7] if oldest is not None else None
    except (ImportError, OSError, ValueError, TypeError, KeyError):
        return None


def _validation() -> dict | None:
    values = da.get_validation()
    return _scalar(values) if isinstance(values, dict) and values else None


def _ensure_state():
    # Earlier dashboard shipped a 4-tuple AOI; geo_v4 requires an AOI dict.
    try:
        st.session_state["aoi"] = normalize_aoi(st.session_state.get("aoi", geo.DEFAULT_AOI))
    except (ValueError, TypeError) as exc:
        st.warning(f"Invalid saved area was reset: {exc}")
        st.session_state["aoi"] = normalize_aoi(DEFAULT_AOI)

    signature = file_signature(ROOT)
    if st.session_state.get("_data_version") != signature:
        if st.session_state.get("_data_version") is not None:
            da.clear_all_caches()
            st.cache_data.clear()
        st.session_state["_data_version"] = signature
    return signature


def _get_period() -> tuple[str, str, list[str]]:
    first, last = _period(da.load_cells_monthly())
    choices = month_options(first, last)
    default_start = choices[max(0, len(choices) - 60)]
    start, end = clamp_month_range(
        st.session_state.get("start_month", default_start),
        st.session_state.get("end_month", last), first, last,
    )
    st.session_state["start_month"] = start
    st.session_state["end_month"] = end
    return first, last, choices


def _sidebar(first: str, last: str, choices: list[str]):
    with st.sidebar:
        st.markdown("### 🔥 FIREWATCH")
        st.caption("NASA Space Apps Challenge · MODIS/VIIRS harmonization")
        st.caption("**Selected area**")
        st.caption(aoi_label(st.session_state["aoi"]))
        start, end = st.select_slider(
            "Shared analysis period", options=choices,
            value=(st.session_state["start_month"], st.session_state["end_month"]),
            help="Controls the map preview, download period and initial dashboard view.",
        )
        st.session_state["start_month"], st.session_state["end_month"] = start, end
        if st.button("↻ Refresh data files", width="stretch"):
            da.clear_all_caches()
            st.cache_data.clear()
            st.session_state["_data_version"] = file_signature(ROOT)
            st.rerun()
        st.caption(f"Data reporting window: {first} to {last}")
        st.warning("Detected hotspots are not burned area or a complete record of all fires.")
        st.caption(FIRMS_ACK)


def page_map():
    st.markdown('<div class="fw-intro"><div class="fw-overline">01 · GEOSPATIAL EXPLORATION</div>'
                '<h2>Map & Area of Interest</h2>'
                '<p>Draw, select or refine an area. Grid-cell analysis and exact hotspot points are distinguished.</p></div>',
                unsafe_allow_html=True)
    st.caption("The rectangle is a geographic bounding box, not a verified state boundary.")
    aoi = normalize_aoi(st.session_state["aoi"])
    try:
        cells = da.cells_in_aoi(aoi)
        with st.spinner("Loading filtered hotspot preview…"):
            records = da.hotspots(cells, month_start(st.session_state["start_month"]),
                                  month_end(st.session_state["end_month"]), max_rows=MAP_LIMIT)
        total = int(getattr(records, "attrs", {}).get("total", len(records)))
        st.markdown(f"**{total:,} detected hotspots, {st.session_state['start_month']} to "
                    f"{st.session_state['end_month']}** · raw detections, all satellites "
                    "(Aqua + S-NPP + NOAA-20)")
        st.caption("The Dashboard shows the harmonized record, which is scaled to S-NPP only, "
                   "so its counts are lower than this raw all-satellite total.")
        if getattr(records, "attrs", {}).get("sampled"):
            st.info(f"Hotspot markers are a reproducible sample (up to {MAP_LIMIT:,}). Analytics use full aggregated data.")
        geo.area_map(hotspots_df=records)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        st.error("Map data could not be loaded for this selection.")
        st.caption(f"Technical detail: {exc}")
        # Still let geo.py render the AOI selector without points.
        geo.area_map(hotspots_df=pd.DataFrame(columns=["lat", "lon", "satellite"]))
    st.caption("Map displays **exactly clipped detected hotspot positions**; analysis selects whole 0.1° cells by centre.")
    if st.checkbox("Show optional India-wide yearly globe", value=False):
        try:
            monthly = da.load_cells_monthly()
            years = sorted(int(x) for x in monthly["year"].dropna().unique())
            selected = st.select_slider("Globe year", options=years, value=years[-1])
            st.caption("This is a national overview, not restricted to the selected AOI.")
            st.plotly_chart(geo.globe_figure(monthly, year=selected), width="stretch")
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Globe view unavailable: {exc}")


def page_dashboard():
    st.markdown('<div class="fw-intro"><div class="fw-overline">02 · HARMONIZED FIRE RECORD</div>'
                '<h2>Burning activity calendar</h2>'
                '<p>Compare months, inspect sensor differences, and investigate unusually active periods.</p></div>',
                unsafe_allow_html=True)
    aoi = normalize_aoi(st.session_state["aoi"])
    _first, last = _period(da.load_cells_monthly())
    try:
        rows = build_rows(aoi, last, st.session_state["_data_version"])
        if not rows:
            st.info("No monthly observations are available for this area.")
            return
        data = {
            "regions": [{"id": "aoi", "name": "Selected area"}],
            "monthly": {"aoi": rows},
            "validation": _validation(),
            "provisional_from": provisional_month(st.session_state["_data_version"]),
            "initial_start": st.session_state["start_month"],
            "initial_end": st.session_state["end_month"],
        }
        html = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
        # Bundle Plotly into the iframe by default for a reliable offline NASA demo.
        # External fonts remain optional; browser system fonts work offline.
        from plotly.offline import get_plotlyjs
        cdn = '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>'
        if cdn in html:
            script = get_plotlyjs().replace("</script", r"<\/script")
            html = html.replace(cdn, "<script>" + script + "</script>", 1)
        payload = "<script>window.FIREWATCH_DATA=" + safe_script_json(data) + ";</script></head>"
        if "</head>" not in html:
            raise RuntimeError("Embedded dashboard HTML has no closing head tag")
        html = html.replace("</head>", payload, 1)
        # A long history needs more room than the earlier fixed 1700-pixel iframe.
        height = max(1500, min(2350, 1000 + len(rows) // 12 * 26))
        st.components.v1.html(html, height=height, scrolling=False)
        st.caption("The filters inside the chart adjust this view only. The shared Streamlit "
                   "period in the sidebar governs the map and downloads.")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        st.error("The calendar could not be generated from the available data.")
        st.caption(f"Technical detail: {exc}")


def page_downloads():
    st.markdown('<div class="fw-intro"><div class="fw-overline">03 · REPRODUCIBLE RESEARCH</div>'
                '<h2>Download filtered observations</h2>'
                '<p>Export the chosen geographic area and a fully inclusive date window.</p></div>',
                unsafe_allow_html=True)
    aoi = normalize_aoi(st.session_state["aoi"])
    start, end = st.session_state["start_month"], st.session_state["end_month"]
    st.caption(f"**Area:** {aoi_label(aoi)} · **Period:** {start} through {end}")
    try:
        cells = da.cells_in_aoi(aoi)
        matrix = da.calendar_matrix(cells)
        st.download_button("⬇ Download calendar (CSV)", data=da.to_csv(matrix),
                           file_name="firewatch_calendar.csv", mime="text/csv", width="stretch")
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        st.error(f"Calendar export unavailable: {exc}")

    st.divider()
    st.markdown("#### Hotspot positions")
    st.caption("A sampled download is limited to 20,000 records. A full export is available "
               "for short ranges; it can use significant RAM on very busy regions.")
    mode = st.radio("Export mode", ["Preview (up to 20,000 rows)", "Full matching records"], horizontal=True)
    options = month_options(start, end)
    full = mode.startswith("Full")
    if full and len(options) > MAX_FULL_EXPORT_MONTHS:
        st.warning(f"For full exports, narrow the shared period to {MAX_FULL_EXPORT_MONTHS} months or fewer.")
        return
    key = (json.dumps(aoi, sort_keys=True), start, end, full, st.session_state["_data_version"])
    if st.button("Prepare hotspot CSV", type="primary", width="stretch"):
        try:
            with st.spinner("Reading filtered NASA FIRMS detections…"):
                points = da.hotspots(cells, month_start(start), month_end(end),
                                    max_rows=None if full else PREVIEW_EXPORT_LIMIT)
                content = da.to_csv(points)
            st.session_state["_hotspot_export"] = (key, content, len(points), bool(points.attrs.get("sampled")))
        except (ValueError, KeyError, OSError, RuntimeError, MemoryError) as exc:
            st.session_state.pop("_hotspot_export", None)
            st.error(f"Export could not be prepared: {exc}")
    cached = st.session_state.get("_hotspot_export")
    if cached and cached[0] == key:
        _, content, count, sampled = cached
        st.success(f"Prepared {count:,} observations" + (" (sampled preview)" if sampled else " (complete matching result)"))
        st.download_button("⬇ Save hotspot CSV", data=content,
                           file_name=f"firewatch_hotspots_{start}_to_{end}.csv",
                           mime="text/csv", width="stretch")


_ensure_state()
try:
    earliest, latest, month_choices = _get_period()
except (ValueError, KeyError, OSError, RuntimeError) as exc:
    st.error("FIREWATCH cannot load the approved monthly dataset. No results are being invented.")
    st.code(str(exc))
    st.stop()
_sidebar(earliest, latest, month_choices)

pg = st.navigation([
    st.Page(page_map, title="Map & Area", icon="🗺️", url_path="map"),
    st.Page(page_dashboard, title="Dashboard", icon="📊", url_path="dashboard"),
    st.Page(page_downloads, title="Downloads", icon="📥", url_path="downloads"),
])
pg.run()

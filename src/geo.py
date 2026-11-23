import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from folium.plugins import Draw, HeatMap
from streamlit_folium import st_folium

import data_access as da
import spatial_analysis as sa
from grid import _convert_drawing_to_aoi, aoi_area_km2, cell_bounds, points_in_aoi

# Every map and table here shows DETECTED HOTSPOTS, not burned area.
HOTSPOT_LABEL = sa.HOTSPOT_LABEL


DEFAULT_AOI = {
    "type": "bbox",
    "coords": (27.6, 32.6, 73.8, 77.6),
}

SATELLITE_COLORS = {
    "Aqua": "#E69F00",
    "S-NPP": "#56B4E9",
    "NOAA-20": "#009E73",
}

HEAT_GRADIENT = {
    0.2: "#ffffb2",
    0.4: "#fecc5c",
    0.6: "#fd8d3c",
    0.8: "#f03b20",
    1.0: "#bd0026",
}

SATELLITE_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

MAX_HOTSPOT_DOTS = 2000


def _satellite_name(value):
    name = str(value).strip().upper()

    if "AQUA" in name or name == "A":
        return "Aqua"

    if "SNPP" in name or "S-NPP" in name or "SUOMI" in name:
        return "S-NPP"

    if "NOAA-20" in name or "NOAA20" in name or name in ("J1", "1"):
        return "NOAA-20"

    return "Other"


def _satellite_color(value):
    return SATELLITE_COLORS.get(_satellite_name(value), "#FFFFFF")


def _sync_box_inputs(aoi):
    """Synchronize box input widgets whenever the AOI changes."""
    if st.session_state.get("_box_sync_aoi") == aoi:
        return

    if aoi["type"] == "bbox":
        min_lat, max_lat, min_lon, max_lon = aoi["coords"]

        st.session_state["box_min_lat"] = float(min_lat)
        st.session_state["box_max_lat"] = float(max_lat)
        st.session_state["box_min_lon"] = float(min_lon)
        st.session_state["box_max_lon"] = float(max_lon)

    st.session_state["_box_sync_aoi"] = aoi.copy()


BOUNDARIES_FILE = pathlib.Path(__file__).resolve().parents[1] / "data" / "boundaries" / "punjab_haryana.json"
WHOLE_STATE = "Whole state"


@st.cache_data(show_spinner=False)
def load_boundaries():
    """State and district outlines for the picker (built by src/build_boundaries.py).

    Returns the parsed JSON, or None if the file is missing.
    """
    if not BOUNDARIES_FILE.is_file():
        return None
    return json.loads(BOUNDARIES_FILE.read_text(encoding="utf-8"))


def region_aoi(boundaries, state, district=WHOLE_STATE):
    """Polygon AOI dict for a state, or for one district of that state."""
    state_data = boundaries["states"][state]
    if district == WHOLE_STATE:
        coords, name = state_data["coords"], f"{state} (whole state)"
    else:
        coords, name = state_data["districts"][district]["coords"], f"{district} district, {state}"
    return {
        "type": "polygon",
        "coords": [(float(lat), float(lon)) for lat, lon in coords],
        "name": name,
    }


def _select_region():
    """State + district dropdowns. Returns the AOI dict, or None if no boundaries file."""
    boundaries = load_boundaries()
    if boundaries is None:
        st.warning("Boundary file data/boundaries/punjab_haryana.json is missing.")
        return None

    c1, c2 = st.columns(2)
    with c1:
        state = st.selectbox("State", list(boundaries["states"]), key="region_state")
    with c2:
        district = st.selectbox(
            "District",
            [WHOLE_STATE] + list(boundaries["states"][state]["districts"]),
            key=f"region_district_{state}",  # one key per state, so a switch resets it
        )
    st.caption(
        "Outlines: © OpenStreetMap contributors (ODbL 1.0), simplified to about 200 m. "
        "For analysis only, not an authoritative boundary map."
    )
    return region_aoi(boundaries, state, district)


def _select_aoi():
    if "aoi" not in st.session_state:
        st.session_state["aoi"] = DEFAULT_AOI.copy()

    _sync_box_inputs(st.session_state["aoi"])

    choice = st.radio(
        "Choose your area",
        ["Draw on map", "Box (lat/lon)", "Point + radius", "State / district"],
        horizontal=True,
        key="aoi_choice",
    )

    old_aoi = st.session_state["aoi"].copy()

    if choice == "State / district":
        region = _select_region()
        if region is not None:
            st.session_state["aoi"] = region

    if choice == "Box (lat/lon)":
        current = st.session_state["aoi"]

        if current["type"] == "bbox":
            start = current["coords"]
        else:
            start = DEFAULT_AOI["coords"]

        # Seed the widget values through session_state only (passing a
        # `value=` as well makes Streamlit warn about two sources of truth).
        # Streamlit forgets widget values while the widgets are hidden, so
        # re-seed every time the user switches into Box mode.
        if not st.session_state.get("_box_widgets_shown"):
            for key, value in zip(
                ["box_min_lat", "box_max_lat", "box_min_lon", "box_max_lon"],
                start,
            ):
                st.session_state[key] = float(value)

        c1, c2 = st.columns(2)

        with c1:
            min_lat = st.number_input(
                "Minimum latitude", -90.0, 90.0, key="box_min_lat",
            )
            min_lon = st.number_input(
                "Minimum longitude", -180.0, 180.0, key="box_min_lon",
            )

        with c2:
            max_lat = st.number_input(
                "Maximum latitude", -90.0, 90.0, key="box_max_lat",
            )
            max_lon = st.number_input(
                "Maximum longitude", -180.0, 180.0, key="box_max_lon",
            )

        if min_lat < max_lat and min_lon < max_lon:
            st.session_state["aoi"] = {
                "type": "bbox",
                "coords": (min_lat, max_lat, min_lon, max_lon),
            }
        else:
            st.warning("Minimum values must be smaller than maximum values.")

    elif choice == "Point + radius":
        c1, c2 = st.columns(2)

        with c1:
            lat = st.number_input(
                "Latitude",
                -90.0,
                90.0,
                30.0,
                key="point_lat",
            )

        with c2:
            lon = st.number_input(
                "Longitude",
                -180.0,
                180.0,
                75.0,
                key="point_lon",
            )

        radius = st.slider(
            "Radius (km)",
            5,
            300,
            50,
            key="point_radius",
        )

        st.session_state["aoi"] = {
            "type": "point",
            "coords": (lat, lon),
            "radius_km": radius,
        }

    # Remember whether the box inputs were on screen in this run.
    st.session_state["_box_widgets_shown"] = choice == "Box (lat/lon)"

    return old_aoi


def _aoi_bounds(aoi):
    kind = aoi["type"]

    if kind == "bbox":
        min_lat, max_lat, min_lon, max_lon = aoi["coords"]
        return [[min_lat, min_lon], [max_lat, max_lon]]

    if kind == "polygon":
        points = aoi["coords"]

        return [
            [
                min(point[0] for point in points),
                min(point[1] for point in points),
            ],
            [
                max(point[0] for point in points),
                max(point[1] for point in points),
            ],
        ]

    if kind == "point":
        lat, lon = aoi["coords"]
        delta_lat = aoi["radius_km"] / 111.32
        delta_lon = aoi["radius_km"] / max(
            111.32 * math.cos(math.radians(lat)),
            1e-6,
        )

        return [
            [
                max(-90, lat - delta_lat),
                max(-180, lon - delta_lon),
            ],
            [
                min(90, lat + delta_lat),
                min(180, lon + delta_lon),
            ],
        ]

    raise ValueError(f"Unsupported AOI type: {kind}")


def _aoi_area_km2(aoi):
    """Approximate AOI area in km² (shared helper lives in grid.py)."""
    return aoi_area_km2(aoi)


def _signed_area(ring):
    """Signed shoelace area of a [(lat, lon), ...] ring; the sign tells the winding."""
    total = 0.0
    for (y1, x1), (y2, x2) in zip(ring, list(ring[1:]) + [ring[0]]):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def split_bridged_ring(coords):
    """Split a ring built by src/build_boundaries.py back into its pieces.

    The builder joins pieces as: piece0, then for each further piece
    [anchor] + piece + [piece[0]], where anchor = piece0[0]. A plain polygon
    (e.g. one drawn on the map) comes back unchanged as a single piece.
    """
    coords = [tuple(p) for p in coords]
    anchor = coords[0]
    cuts = [i for i in range(1, len(coords)) if coords[i] == anchor]
    if not cuts:
        return [coords]
    pieces = [coords[:cuts[0]]]
    for start, end in zip(cuts, cuts[1:] + [len(coords)]):
        piece = coords[start + 1:end]
        if len(piece) > 1 and piece[-1] == piece[0]:
            piece = piece[:-1]  # drop the "bridge back" vertex
        if len(piece) >= 3:
            pieces.append(piece)
    return pieces


def _draw_selected_aoi(map_obj, aoi):
    """Draw the selected AOI and zoom the map to its bounds."""
    bounds = _aoi_bounds(aoi)

    style = {
        "color": "#ff8c00",
        "weight": 4,
        "fillColor": "#ff8c00",
        "fillOpacity": 0.12,
    }

    if aoi["type"] == "bbox":
        folium.Rectangle(
            bounds=bounds,
            name="Selected AOI",
            **style,
        ).add_to(map_obj)

    elif aoi["type"] == "polygon":
        # A district outline may be several pieces joined by zero-width bridges;
        # draw each piece on its own so no bridge lines show. Holes: outline only.
        pieces = split_bridged_ring(aoi["coords"])
        outer_sign = _signed_area(pieces[0]) >= 0
        for piece in pieces:
            is_hole = (_signed_area(piece) >= 0) != outer_sign
            folium.Polygon(
                locations=piece,
                name="Selected AOI",
                **(dict(style, fillOpacity=0, weight=2) if is_hole else style),
            ).add_to(map_obj)

    elif aoi["type"] == "point":
        lat, lon = aoi["coords"]

        folium.Circle(
            location=[lat, lon],
            radius=float(aoi["radius_km"]) * 1000,
            name="Selected AOI",
            **style,
        ).add_to(map_obj)

    map_obj.fit_bounds(bounds)


def _add_hotspot_layers(map_obj, hotspots_df, aoi):
    if hotspots_df is None or hotspots_df.empty:
        return

    if not {"lat", "lon"}.issubset(hotspots_df.columns):
        st.warning("Hotspot data must contain lat and lon columns.")
        return

    valid = hotspots_df.dropna(subset=["lat", "lon"]).copy()

    mask = points_in_aoi(
        valid["lat"],
        valid["lon"],
        aoi,
    )

    valid = valid.loc[mask]

    total = len(valid)
    sampled = total > MAX_HOTSPOT_DOTS

    sample = (
        valid.sample(n=MAX_HOTSPOT_DOTS, random_state=42)
        if sampled
        else valid
    )

    if sampled:
        st.caption(
            f"Showing {MAX_HOTSPOT_DOTS:,} of {total:,} "
            "hotspots (sampled)"
        )

    heat_points = []
    groups = {}

    for row in sample.itertuples(index=False):
        record = row._asdict()

        lat = float(record["lat"])
        lon = float(record["lon"])

        heat_points.append([lat, lon])

        satellite = record.get("satellite", "Other")
        satellite = _satellite_name(satellite)

        groups.setdefault(satellite, []).append((lat, lon))

    for satellite, points in groups.items():
        feature_group = folium.FeatureGroup(
            name=f"Detected hotspots — {satellite}",
            show=True,
        )

        color = SATELLITE_COLORS.get(satellite, "#FFFFFF")

        for lat, lon in points:
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=f"{satellite} hotspot",
            ).add_to(feature_group)

        feature_group.add_to(map_obj)

    if heat_points:
        heat_layer = folium.FeatureGroup(
            name="Density (from displayed sample)",
            show=True,
        )

        HeatMap(
            heat_points,
            name="Density (from displayed sample)",
            radius=18,
            blur=15,
            min_opacity=0.3,
            gradient=HEAT_GRADIENT,
        ).add_to(heat_layer)

        heat_layer.add_to(map_obj)


def _draw_legend(map_obj):
    legend = """
    <div style="position:fixed;bottom:25px;left:25px;
    z-index:9999;background:white;color:black;padding:10px;
    border-radius:6px;font-size:12px;">
    <b>Detected hotspots</b><br>
    <span style="color:#E69F00;">●</span> Aqua<br>
    <span style="color:#56B4E9;">●</span> S-NPP<br>
    <span style="color:#009E73;">●</span> NOAA-20<br>
    <div style="margin-top:4px;"><b>Density (from displayed sample)</b></div>
    <div style="width:150px;height:10px;background:linear-gradient(to right,
    #ffffb2,#fecc5c,#fd8d3c,#f03b20,#bd0026);"></div>
    <div style="display:flex;justify-content:space-between;width:150px;">
    <span>Low</span><span>High</span></div>
    </div>
    """

    map_obj.get_root().html.add_child(
        folium.Element(legend)
    )


# ---------------------------------------------------------------------------
# Analysis layers: 0.1° squares coloured by a class, with a legend.
# Colours come from ColorBrewer RdBu / PuOr (colour-blind safe).
# ---------------------------------------------------------------------------
TREND_COLORS = {
    sa.TREND_UP: "#B2182B",     # red
    sa.TREND_DOWN: "#2166AC",   # blue
    sa.TREND_NONE: "#BDBDBD",   # grey
}

CLUSTER_COLORS = {
    sa.HOT: "#B2182B",
    sa.COLD: "#2166AC",
}

CHRONIC_COLORS = {
    "Chronic (≥80% of years)": "#B35806",
    "Some years (<80%)": "#FDB863",
}


def _cell_squares_layer(map_obj, cells_df, class_col, colors, name, tooltip_cols):
    """
    Add one 0.1° square per row of `cells_df`, coloured by `class_col`
    (rows whose class is not in `colors` are skipped). Returns the layer.
    """
    features = []
    for row in cells_df.to_dict("records"):
        color = colors.get(row[class_col])
        if color is None:
            continue
        lat0, lat1, lon0, lon1 = cell_bounds(row["cell_id"])
        tip = "<br>".join(
            f"{col}: {row[col]:.3g}" if isinstance(row[col], float) else f"{col}: {row[col]}"
            for col in tooltip_cols
        )
        features.append({
            "type": "Feature",
            "properties": {"color": color, "tip": tip},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0],
                ]],
            },
        })

    layer = folium.FeatureGroup(name=name, show=True)
    if features:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            style_function=lambda f: {
                "fillColor": f["properties"]["color"],
                "color": f["properties"]["color"],
                "weight": 0.3,
                "fillOpacity": 0.7,
            },
            tooltip=folium.GeoJsonTooltip(fields=["tip"], labels=False),
        ).add_to(layer)
    layer.add_to(map_obj)
    return layer


def _add_class_legend(map_obj, title, colors):
    """Small square-swatch legend in the bottom-right corner."""
    rows = "".join(
        f'<div><span style="display:inline-block;width:12px;height:12px;'
        f'background:{color};margin-right:6px;border:1px solid #555;"></span>{label}</div>'
        for label, color in colors.items()
    )
    legend = f"""
    <div style="position:fixed;bottom:25px;right:25px;z-index:9999;
    background:white;color:black;padding:10px;border-radius:6px;font-size:12px;">
    <b>{title}</b><br>{rows}
    <div style="margin-top:4px;font-style:italic;">{HOTSPOT_LABEL}</div>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend))


def trend_layer(map_obj, trend_df, show_no_trend=True):
    """Trend of annual detected hotspots: red = increasing, blue = decreasing, grey = none."""
    colors = dict(TREND_COLORS)
    if not show_no_trend:
        colors.pop(sa.TREND_NONE)
    _cell_squares_layer(
        map_obj, trend_df, "trend", colors, "Trend in detected hotspots",
        ["cell_id", "trend", "slope_per_year", "p_value", "active_years"],
    )
    _add_class_legend(map_obj, "Trend (Theil-Sen, p<0.05)", colors)
    return map_obj


def cluster_layer(map_obj, cluster_df):
    """Getis-Ord Gi* clusters: red = hot spot, blue = cold spot (not significant hidden)."""
    _cell_squares_layer(
        map_obj, cluster_df, "cluster", CLUSTER_COLORS, "Hot/cold spot clusters",
        ["cell_id", "cluster", "gi_z", "value"],
    )
    _add_class_legend(map_obj, "Getis-Ord Gi* (|z| > 1.96)", CLUSTER_COLORS)
    return map_obj


def chronic_layer(map_obj, chronic_df):
    """Cells burning in the chosen season: dark = chronic (≥80% of years), light = some years."""
    data = chronic_df.copy()
    labels = list(CHRONIC_COLORS)
    data["chronic_class"] = np.where(data["chronic"].astype(bool), labels[0], labels[1])
    _cell_squares_layer(
        map_obj, data, "chronic_class", CHRONIC_COLORS, "Chronic burning",
        ["cell_id", "chronic_class", "burn_years", "n_years"],
    )
    _add_class_legend(map_obj, "Chronic burning", CHRONIC_COLORS)
    return map_obj


def _aoi_key(value):
    """Canonical form of an AOI for comparing: lists/tuples -> tuples, numbers -> rounded floats."""
    if isinstance(value, dict):
        return tuple(sorted((k, _aoi_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_aoi_key(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    return value


def area_map(hotspots_df=None):
    """Display the AOI selector and hotspot overlays."""
    old_aoi = _select_aoi()
    aoi = st.session_state["aoi"]

    map_obj = folium.Map(
        location=[30.1, 75.7],
        zoom_start=7,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    folium.TileLayer(
        tiles=SATELLITE_TILES,
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        name="Esri World Imagery",
        overlay=False,
        control=True,
    ).add_to(map_obj)

    _draw_selected_aoi(map_obj, aoi)

    Draw(
        draw_options={
            "polyline": False,
            "polygon": True,
            "rectangle": True,
            "circle": True,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(map_obj)

    _add_hotspot_layers(map_obj, hotspots_df, aoi)
    _draw_legend(map_obj)

    folium.LayerControl(collapsed=True).add_to(map_obj)

    map_data = st_folium(
        map_obj,
        use_container_width=True,
        height=560,
        key="firewatch_area_map",
        returned_objects=["all_drawings"],
    )

    drawings = (map_data or {}).get("all_drawings") or []

    if (
        st.session_state.get("aoi_choice") == "Draw on map"
        and drawings
    ):
        try:
            new_aoi = _convert_drawing_to_aoi(drawings[-1])

            if _aoi_key(new_aoi) != _aoi_key(st.session_state["aoi"]):
                st.session_state["aoi"] = new_aoi
                st.session_state.pop("hotspot_data", None)
                st.rerun()

        except (ValueError, TypeError, KeyError) as exc:
            st.warning(f"Could not use drawing: {exc}")

    # NOTE: do not call _sync_box_inputs() here. The box number_inputs are
    # already drawn in this run, and changing their session_state keys after
    # that crashes Streamlit. The call at the top of _select_aoi() syncs them
    # on the next run instead.
    if old_aoi != st.session_state["aoi"]:
        st.session_state.pop("hotspot_data", None)

    # Rerun once for EVERY mode (Box, Point, Draw) when the area really
    # changed, so the sidebar label and other pages use the new area.
    # Compare a canonical form (tuples vs lists, 50 vs 50.0 count as equal)
    # so this can never loop.
    if _aoi_key(old_aoi) != _aoi_key(st.session_state["aoi"]):
        st.rerun()

    aoi = st.session_state["aoi"]

    st.caption(
        "Analysis uses whole 0.1° grid cells whose centre is inside "
        "your area; dots show detected hotspots inside the exact area."
    )

    try:
        cell_ids = da.cells_in_aoi(aoi)

        st.write(
            f"**Grid cells with fire records: {len(cell_ids)}**"
        )

    except Exception as exc:
        st.error(f"Could not calculate grid cells: {exc}")

    st.write(
        f"**AOI area: {_aoi_area_km2(aoi):,.2f} km²**"
    )

    return aoi


def globe_figure(cells_df, year=None):
    """Create a globe showing detected hotspot density by grid cell."""
    if cells_df is None or cells_df.empty:
        return px.scatter_geo(
            title="No detected hotspots data available"
        )

    data = cells_df.copy()

    if year is not None and "year" in data.columns:
        data = data[data["year"] == year]

    required = {"cell_lat", "cell_lon"}

    if not required.issubset(data.columns):
        raise ValueError(
            "cells_df must contain cell_lat and cell_lon columns"
        )

    value_col = (
        "harmonized_count"
        if "harmonized_count" in data.columns
        else "n_hotspots"
    )

    if value_col not in data.columns:
        raise ValueError(
            "Missing harmonized_count or n_hotspots column"
        )

    grouped = (
        data.groupby(
            ["cell_lat", "cell_lon"],
            as_index=False,
        )
        .agg(
            **{
                value_col: (value_col, "sum"),
                "cell_id": (
                    "cell_id",
                    "first",
                ) if "cell_id" in data.columns else (
                    value_col,
                    lambda values: "",
                ),
            }
        )
    )

    fig = px.scatter_geo(
        grouped,
        lat="cell_lat",
        lon="cell_lon",
        size=value_col,
        color=value_col,
        color_continuous_scale="YlOrRd",
        projection="orthographic",
        hover_data={
            "cell_id": True,
            "cell_lat": True,
            "cell_lon": True,
            value_col: True,
        },
        title="Harmonized fire activity (VIIRS-equivalent hotspot counts)",
    )

    fig.update_geos(
        projection_rotation=dict(lat=22, lon=79),
        showcountries=True,
        showland=True,
        landcolor="#303030",
        showocean=True,
        oceancolor="#101820",
        bgcolor="#101820",
    )

    fig.update_layout(
        paper_bgcolor="#101820",
        font_color="white",
    )

    return fig


# ---------------------------------------------------------------------------
# Analysis section (cached so the maths runs once per setting)
# ---------------------------------------------------------------------------
ANALYSIS_LAYERS = [
    "Trend 2003–2025",
    "Hot/cold spot clusters (Getis-Ord Gi*)",
    "Chronic burning",
]
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


@st.cache_data(show_spinner="Computing trends…")
def _cached_trend(cell_ids):
    cells = da.load_cells_monthly()
    return sa.trend_per_cell(cells[cells["cell_id"].isin(cell_ids)])


@st.cache_data(show_spinner="Computing clusters…")
def _cached_clusters(year, months):
    # Whole India grid, so cells at the AOI edge still see their neighbours.
    return sa.hotspot_clusters(da.load_cells_monthly(), year, months)


@st.cache_data(show_spinner="Finding chronic burning…")
def _cached_chronic(months):
    return sa.chronic_burning(da.load_cells_monthly(), months)


@st.cache_data(show_spinner="Summarising area…")
def _cached_summary(cell_ids, area_km2, years):
    return sa.aoi_summary(list(cell_ids), da.load_cells_monthly(), area_km2=area_km2, years=years)


def analysis_section(aoi):
    """Layer picker + analysis map + AOI summary table for the selected area."""
    st.subheader("Analysis")
    st.caption(
        f"{HOTSPOT_LABEL}. Harmonized VIIRS-equivalent counts per 0.1° cell "
        "(pre-2012 = MODIS × monthly ratio). Shown for cells inside the selected area."
    )

    cell_ids = tuple(da.cells_in_aoi(aoi))
    if not cell_ids:
        st.info("No grid cells with detected hotspots in this area.")
        return

    layer = st.selectbox("Analysis layer", ANALYSIS_LAYERS, key="analysis_layer")
    month_fmt = MONTH_NAMES.get

    if layer == ANALYSIS_LAYERS[0]:
        table = _cached_trend(cell_ids)
        class_col = "trend"
        note = (
            "Annual harmonized count per cell, 2003–2025 (years without fires = 0). "
            "Theil-Sen slope; Mann-Kendall (Kendall tau) p < 0.05. "
            f"Only cells with detected hotspots in ≥{sa.MIN_ACTIVE_YEARS} years."
        )
    elif layer == ANALYSIS_LAYERS[1]:
        c1, c2 = st.columns(2)
        with c1:
            year = st.selectbox(
                "Year", list(range(sa.DEFAULT_YEARS[1], sa.DEFAULT_YEARS[0] - 1, -1)),
                key="cluster_year",
            )
        with c2:
            months = st.multiselect(
                "Months (empty = whole year)", list(MONTH_NAMES),
                default=[10, 11], format_func=month_fmt, key="cluster_months",
            )
        table = _cached_clusters(int(year), tuple(sorted(months)) or None)
        table = table[table["cell_id"].isin(cell_ids)]
        class_col = "cluster"
        note = (
            "Getis-Ord Gi* z-score on the India 0.1° grid, 3×3 neighbourhood. "
            "Hot spot: z > 1.96; Cold spot: z < -1.96 (≈95 %)."
        )
    else:
        months = st.multiselect(
            "Burning season months", list(MONTH_NAMES),
            default=[10, 11], format_func=month_fmt, key="chronic_months",
        )
        if not months:
            st.warning("Pick at least one month.")
            return
        table = _cached_chronic(tuple(sorted(months)))
        table = table[table["cell_id"].isin(cell_ids)].copy()
        labels = list(CHRONIC_COLORS)
        table["chronic_class"] = np.where(table["chronic"].astype(bool), labels[0], labels[1])
        class_col = "chronic_class"
        note = (
            "Years (2003–2025) with detected hotspots in the chosen months. "
            "Chronic = at least 80 % of years."
        )

    st.caption(note)

    map_obj = folium.Map(
        location=[30.1, 75.7], zoom_start=7,
        # OpenStreetMap tiles need no API key (CARTO now watermarks keyless use).
        tiles="OpenStreetMap", control_scale=True,
    )
    _draw_selected_aoi(map_obj, aoi)
    if layer == ANALYSIS_LAYERS[0]:
        trend_layer(map_obj, table)
    elif layer == ANALYSIS_LAYERS[1]:
        cluster_layer(map_obj, table)
    else:
        chronic_layer(map_obj, table)
    folium.LayerControl(collapsed=True).add_to(map_obj)

    st_folium(
        map_obj, use_container_width=True, height=520,
        key="firewatch_analysis_map", returned_objects=[],
    )

    counts = table[class_col].value_counts()
    st.write(
        "**Cells per class:** "
        + ", ".join(f"{label}: {int(n)}" for label, n in counts.items())
    )
    st.download_button(
        "Download this layer (CSV)",
        data=da.to_csv(table),
        file_name=f"firewatch_{class_col}_layer.csv",
        mime="text/csv",
    )

    # ---- AOI summary table ----
    st.markdown("#### Area summary")
    years = st.slider(
        "Years", sa.DEFAULT_YEARS[0], sa.DEFAULT_YEARS[1], sa.DEFAULT_YEARS,
        key="summary_years",
    )
    summary = _cached_summary(cell_ids, float(_aoi_area_km2(aoi)), tuple(years))
    overview = pd.DataFrame({
        "Measure": [
            "Area (km²)",
            "Active grid cells",
            "Total harmonized count",
            "Density per 1,000 km²",
            "Share of India total (%)",
        ],
        "Value": [
            f"{summary['area_km2']:,.0f}",
            f"{summary['active_cells']:,}",
            f"{summary['total_harmonized_count']:,.0f}",
            f"{summary['density_per_1000_km2']:,.1f}",
            f"{summary['share_of_india_pct']:.1f}",
        ],
    })
    st.dataframe(overview, hide_index=True)
    st.write(f"**Top {len(summary['top_cells'])} cells, {years[0]}–{years[1]}**")
    st.dataframe(summary["top_cells"].round(2), hide_index=True)
    st.caption(f"{HOTSPOT_LABEL}. Area basis: {summary['area_basis']}.")


if __name__ == "__main__":
    st.set_page_config(
        page_title="FIREWATCH Geospatial",
        layout="wide",
    )

    st.title("FIREWATCH — Geospatial Analysis")

    if "aoi" not in st.session_state:
        st.session_state["aoi"] = DEFAULT_AOI.copy()

    try:
        cells = da.load_cells_monthly()

        if cells is not None and not cells.empty:
            st.subheader("3D Detected hotspots globe")

            st.plotly_chart(
                globe_figure(cells, year=2023),
                width="stretch",
            )

        else:
            st.warning("No monthly cell data was found.")

    except Exception as exc:
        st.error(
            f"Could not load monthly cell data: {exc}"
        )

    st.subheader("Area Selection and Detected hotspots")

    area_map(
        st.session_state.get("hotspot_data")
    )

    if st.button("Load / Refresh Hotspots for Selected Area"):
        try:
            selected_ids = da.cells_in_aoi(
                st.session_state["aoi"]
            )

            st.session_state["hotspot_data"] = da.hotspots(
                selected_ids,
                "2023-11-01",
                "2023-11-30",
            )

            st.rerun()

        except Exception as exc:
            st.error(
                f"Could not load hotspots: {exc}"
            )

    st.subheader(
        "Selected Area — Punjab + Haryana "
        "(rectangle, not state borders)"
    )

    st.json(st.session_state["aoi"])

    try:
        analysis_section(st.session_state["aoi"])
    except Exception as exc:
        st.error(f"Could not run the analysis: {exc}")

    st.caption(
        "We acknowledge the use of data and/or imagery from NASA's Fire Information "
        "for Resource Management System (FIRMS) (https://www.earthdata.nasa.gov/firms), "
        "part of NASA's Earth Science Data and Information System (ESDIS)."
    )
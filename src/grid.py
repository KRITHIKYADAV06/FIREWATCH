"""
FIREWATCH grid helpers (0.1 degree cells) and area-of-interest (AOI) geometry.

cell_id format: "<lat floor>_<lon floor>", e.g. "30.4_75.8" (lower-left corner).
The floor rule (floor(x * 10 + 1e-9) / 10) matches src/create_parquet_handoff.py,
so cells made here line up with cell_days.parquet / cells_monthly.parquet.

All counts are detected hotspots, not burned area.
"""
import math

import numpy as np
import pandas as pd

# shapely is optional: if it is installed we use it as an extra polygon check.
try:
    from shapely.geometry import Polygon as _ShapelyPolygon
except ImportError:  # pragma: no cover - depends on the local install
    _ShapelyPolygon = None

# Highest valid cell corners. A point exactly on lat 90 / lon 180 would
# otherwise get a cell "90.0_..." / "..._180.0" that lies outside the globe.
MAX_CELL_LAT = 89.9
MAX_CELL_LON = 179.9

# Mean Earth radius (km) and km per degree of latitude.
EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG = 111.32


def _validated_coordinates(lat, lon):
    try:
        lat_values = np.asarray(lat, dtype=float)
        lon_values = np.asarray(lon, dtype=float)
        lat_values, lon_values = np.broadcast_arrays(lat_values, lon_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid coordinates") from exc

    if (
        not np.all(np.isfinite(lat_values))
        or not np.all(np.isfinite(lon_values))
        or np.any((lat_values < -90) | (lat_values > 90))
        or np.any((lon_values < -180) | (lon_values > 180))
    ):
        raise ValueError("Invalid coordinates")

    return lat_values, lon_values


def assign_cell(lat, lon):
    if isinstance(lat, pd.Series) or isinstance(lon, pd.Series):
        lat_s = lat if isinstance(lat, pd.Series) else pd.Series(lat)
        lon_s = lon if isinstance(lon, pd.Series) else pd.Series(
            lon, index=lat_s.index
        )
        if len(lat_s) != len(lon_s):
            raise ValueError("Invalid coordinates")

        la, lo = _validated_coordinates(lat_s.to_numpy(), lon_s.to_numpy())
        a = np.floor(la * 10 + 1e-9) / 10
        b = np.floor(lo * 10 + 1e-9) / 10
        # Clamp the poles / antimeridian into the last real cell.
        a = np.minimum(a, MAX_CELL_LAT)
        b = np.minimum(b, MAX_CELL_LON)
        a[a == 0] = 0
        b[b == 0] = 0

        return pd.DataFrame({
            "cell_id": [f"{x:.1f}_{y:.1f}" for x, y in zip(a, b)],
            "cell_lat": np.round(a + 0.05, 2),
            "cell_lon": np.round(b + 0.05, 2),
        }, index=lat_s.index)

    la, lo = _validated_coordinates(lat, lon)
    a = float(np.floor(float(la) * 10 + 1e-9) / 10)
    b = float(np.floor(float(lo) * 10 + 1e-9) / 10)
    # Clamp the poles / antimeridian into the last real cell.
    a = min(a, MAX_CELL_LAT)
    b = min(b, MAX_CELL_LON)
    a = 0.0 if a == 0 else a
    b = 0.0 if b == 0 else b
    return f"{a:.1f}_{b:.1f}", round(a + 0.05, 2), round(b + 0.05, 2)


def add_cells(df, lat_col="lat", lon_col="lon", errors="raise"):
    if errors not in {"raise", "drop"}:
        raise ValueError("errors must be 'raise' or 'drop'")

    result = df.copy()
    lat = pd.to_numeric(result[lat_col], errors="coerce")
    lon = pd.to_numeric(result[lon_col], errors="coerce")

    valid = (
        lat.notna() & lon.notna()
        & np.isfinite(lat) & np.isfinite(lon)
        & lat.between(-90, 90) & lon.between(-180, 180)
    )

    dropped = int((~valid).sum())
    if dropped and errors == "raise":
        raise ValueError("Invalid coordinates")

    result = result.loc[valid].copy()
    cells = assign_cell(lat.loc[valid], lon.loc[valid])
    result["cell_id"] = cells["cell_id"]
    result["cell_lat"] = cells["cell_lat"]
    result["cell_lon"] = cells["cell_lon"]
    result.attrs["dropped"] = dropped
    return result


def cell_bounds(cell_id):
    lat, lon = map(float, cell_id.split("_"))
    return lat, round(lat + 0.1, 1), lon, round(lon + 0.1, 1)


def cell_area_km2(lat):
    """Approximate area (km²) of one 0.1° x 0.1° cell centred at latitude `lat`."""
    return 11.132 * 11.132 * np.cos(np.radians(lat))


def _shoelace_area(points):
    """Signed-free area of a ring of (x, y) points (same units squared)."""
    n = len(points)
    twice_area = sum(
        points[i][0] * points[(i + 1) % n][1]
        - points[(i + 1) % n][0] * points[i][1]
        for i in range(n)
    )
    return abs(twice_area) / 2


def aoi_area_km2(aoi):
    """
    Approximate AOI area in km².

    point  -> pi * r²
    bbox   -> height x width at the mean latitude
    polygon-> shoelace formula on a local equirectangular projection
    Good enough for state-sized areas; not a survey-grade area.
    """
    kind = aoi["type"]

    if kind == "point":
        radius = float(aoi["radius_km"])
        return math.pi * radius * radius

    if kind == "bbox":
        min_lat, max_lat, min_lon, max_lon = aoi["coords"]
        height = KM_PER_DEG * (max_lat - min_lat)
        mean_lat = (min_lat + max_lat) / 2
        width = KM_PER_DEG * math.cos(math.radians(mean_lat)) * (max_lon - min_lon)
        return abs(height * width)

    if kind == "polygon":
        points = aoi["coords"]
        mean_lat = sum(p[0] for p in points) / len(points)
        xy = [
            (lon * KM_PER_DEG * math.cos(math.radians(mean_lat)), lat * KM_PER_DEG)
            for lat, lon in points
        ]
        return _shoelace_area(xy)

    raise ValueError(f"Unsupported AOI type: {kind}")


def _segments_intersect(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (
            q[1] - p[1]
        ) * (r[0] - p[0])

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def _convert_drawing_to_aoi(drawing):
    try:
        geometry = drawing["geometry"]
        kind = geometry["type"]
        coordinates = geometry["coordinates"]

        if kind == "Point":
            lon, lat = coordinates[:2]
            _validated_coordinates(lat, lon)
            # Leaflet.draw circles arrive as a Point with properties.radius
            # in METRES, so convert to km here.
            radius = (drawing.get("properties") or {}).get("radius")
            if radius is None or not math.isfinite(float(radius)) or float(radius) <= 0:
                raise ValueError("Invalid coordinates")
            return {"type": "point", "coords": (float(lat), float(lon)),
                    "radius_km": float(radius) / 1000}

        if kind != "Polygon":
            raise ValueError("Unsupported drawing geometry")

        if len(coordinates) != 1:
            raise ValueError("Polygons with holes are not supported")

        ring = coordinates[0]
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]

        points = []
        for lon, lat, *rest in ring:
            _validated_coordinates(lat, lon)
            points.append((float(lat), float(lon)))

        if len(points) < 3:
            raise ValueError("Invalid coordinates")

        # The same corner twice (e.g. a double-click) makes a degenerate shape.
        if len(set(points)) != len(points):
            raise ValueError("Polygon has repeated corners")

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if j in (i, i + 1) or (i == 0 and j == len(points) - 1):
                    continue
                if _segments_intersect(
                    (points[i][1], points[i][0]),
                    (points[(i + 1) % len(points)][1],
                     points[(i + 1) % len(points)][0]),
                    (points[j][1], points[j][0]),
                    (points[(j + 1) % len(points)][1],
                     points[(j + 1) % len(points)][0]),
                ):
                    raise ValueError("Self-intersecting polygon")

        # All corners on one line -> zero area (shoelace on lon/lat degrees).
        # Checked after self-intersection, because a bow-tie also sums to 0.
        if _shoelace_area([(lon, lat) for lat, lon in points]) < 1e-12:
            raise ValueError("Polygon has zero area")

        # Extra check when shapely is installed (optional dependency).
        if _ShapelyPolygon is not None and not _ShapelyPolygon(
            [(lon, lat) for lat, lon in points]
        ).is_valid:
            raise ValueError("Invalid polygon shape")

        distinct = set(points)
        lats = sorted({p[0] for p in distinct})
        lons = sorted({p[1] for p in distinct})

        corners = {(a, b) for a in lats for b in lons}
        is_bbox = (
            len(distinct) == 4 and len(lats) == 2 and len(lons) == 2
            and distinct == corners
            and all(
                points[i][0] == points[(i + 1) % len(points)][0]
                or points[i][1] == points[(i + 1) % len(points)][1]
                for i in range(len(points))
            )
        )

        if is_bbox:
            return {
                "type": "bbox",
                "coords": (lats[0], lats[1], lons[0], lons[1]),
            }

        return {"type": "polygon", "coords": points}

    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("Invalid coordinates") from exc


def points_in_aoi(lat_series, lon_series, aoi):
    lat, lon = _validated_coordinates(lat_series, lon_series)
    lat, lon = np.broadcast_arrays(lat, lon)

    kind = aoi["type"]
    if kind == "bbox":
        min_lat, max_lat, min_lon, max_lon = aoi["coords"]
        return (
            (lat >= min_lat) & (lat <= max_lat)
            & (lon >= min_lon) & (lon <= max_lon)
        )

    if kind == "point":
        centre_lat, centre_lon = aoi["coords"]
        radius = float(aoi["radius_km"])
        r = 6371.0088
        p1, p2 = np.radians(lat), math.radians(centre_lat)
        dp = p1 - p2
        dl = np.radians(lon - centre_lon)
        h = np.sin(dp / 2) ** 2 + np.cos(p1) * math.cos(p2) * np.sin(dl / 2) ** 2
        distance = 2 * r * np.arcsin(np.minimum(1, np.sqrt(h)))
        return distance <= radius

    if kind == "polygon":
        points = aoi["coords"]
        inside = np.zeros(lat.shape, dtype=bool)
        on_edge = np.zeros(lat.shape, dtype=bool)

        for i, (y1, x1) in enumerate(points):
            y2, x2 = points[(i + 1) % len(points)]
            cross = (
                (lon - x1) * (y2 - y1)
                - (lat - y1) * (x2 - x1)
            )
            on_edge |= (
                np.isclose(cross, 0)
                & (lon >= min(x1, x2)) & (lon <= max(x1, x2))
                & (lat >= min(y1, y2)) & (lat <= max(y1, y2))
            )
            crosses = ((y1 > lat) != (y2 > lat)) & (
                lon < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-300) + x1
            )
            inside ^= crosses

        return inside | on_edge

    raise ValueError("Unsupported AOI type")
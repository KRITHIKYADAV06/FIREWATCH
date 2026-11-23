"""Pure, independently testable helpers for FIREWATCH's Streamlit frontend.

This module does not depend on Streamlit, PyArrow, or the NASA datasets.
"""
from __future__ import annotations

import calendar
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_AOI = {"type": "bbox", "coords": (27.6, 32.6, 73.8, 77.6)}


def normalize_aoi(value: Any) -> dict:
    """Upgrade legacy tuples and validate AOIs before giving them to geo.py."""
    if value is None:
        value = DEFAULT_AOI
    if isinstance(value, (list, tuple)) and len(value) == 4:
        value = {"type": "bbox", "coords": value}
    if not isinstance(value, dict):
        raise ValueError("AOI must be a bounding box, point or polygon")
    kind = value.get("type", "bbox")
    coords = value.get("coords")
    if kind == "bbox":
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError("Bounding box requires four coordinates")
        a, b, c, d = (float(x) for x in coords)
        if not all(map(math.isfinite, (a, b, c, d))) or not (-90 <= a < b <= 90 and -180 <= c < d <= 180):
            raise ValueError("Invalid latitude/longitude bounding box")
        return {"type": "bbox", "coords": (a, b, c, d)}
    if kind == "point":
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            raise ValueError("Point requires latitude and longitude")
        lat, lon = map(float, coords)
        radius = float(value.get("radius_km", 0))
        if not all(map(math.isfinite, (lat, lon, radius))) or not (-90 <= lat <= 90 and -180 <= lon <= 180 and 0 < radius <= 5000):
            raise ValueError("Invalid point or radius")
        return {"type": "point", "coords": (lat, lon), "radius_km": radius}
    if kind == "polygon":
        if not isinstance(coords, (list, tuple)) or len(coords) < 3:
            raise ValueError("Polygon needs at least three vertices")
        pts = []
        for point in coords:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Invalid polygon vertex")
            lat, lon = map(float, point)
            if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Invalid polygon coordinate")
            pts.append((lat, lon))
        # Geo module does full topology validation; this helper validates the transport contract.
        result = {"type": "polygon", "coords": pts}
        # Keep the name of a picked state/district (e.g. "Ludhiana district, Punjab").
        if isinstance(value.get("name"), str) and value["name"].strip():
            result["name"] = value["name"].strip()
        return result
    raise ValueError(f"Unsupported AOI type: {kind}")


def aoi_label(aoi: dict) -> str:
    aoi = normalize_aoi(aoi)
    if aoi["type"] == "bbox":
        a, b, c, d = aoi["coords"]
        return f"Bounding box · {a:.2f}–{b:.2f}°N, {c:.2f}–{d:.2f}°E"
    if aoi["type"] == "point":
        lat, lon = aoi["coords"]
        return f"Radius {aoi['radius_km']:g} km · {lat:.2f}°N, {lon:.2f}°E"
    if aoi.get("name"):
        return aoi["name"]
    return f"Polygon · {len(aoi['coords'])} vertices"


def month_start(ym: str) -> str:
    year, month = validate_month(ym)
    return f"{year:04d}-{month:02d}-01"


def month_end(ym: str) -> str:
    year, month = validate_month(ym)
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def validate_month(ym: str) -> tuple[int, int]:
    if not isinstance(ym, str) or len(ym) != 7 or ym[4] != "-":
        raise ValueError("Expected YYYY-MM")
    try:
        year, month = int(ym[:4]), int(ym[5:])
        date(year, month, 1)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid year or month") from exc
    if not (2003 <= year <= 2100):
        raise ValueError("Month outside supported range")
    return year, month


def clamp_month_range(start: str, end: str, first: str, last: str) -> tuple[str, str]:
    for item in (start, end, first, last):
        validate_month(item)
    if first > last:
        raise ValueError("Invalid dataset period")
    if start > last or end < first:
        return first, last
    s = max(first, min(start, last))
    e = max(first, min(end, last))
    if s > e:
        s, e = first, last
    return s, e


def month_options(first: str, last: str) -> list[str]:
    y0, m0 = validate_month(first)
    y1, m1 = validate_month(last)
    if (y0, m0) > (y1, m1):
        raise ValueError("Start after end")
    return [f"{y:04d}-{m:02d}" for y in range(y0, y1 + 1)
            for m in range(1, 13) if (y, m) >= (y0, m0) and (y, m) <= (y1, m1)]


def file_signature(root: Path) -> tuple:
    """Cache freshness token. The app also invalidates upstream data_access caches."""
    entries = []
    for name in ("fires.parquet", "cell_days.parquet", "cells_monthly.parquet", "ratios.parquet", "validation.json"):
        paths = (root / "data" / "processed" / name, root / "data" / "sample" / name)
        present = next((p for p in paths if p.is_file()), None)
        if present:
            stat = present.stat()
            entries.append((name, str(present), stat.st_size, stat.st_mtime_ns))
        else:
            entries.append((name, None, 0, 0))
    return tuple(entries)


def safe_script_json(data: Any) -> str:
    """Serialize JSON for a script tag without permitting script-tag termination."""
    return (json.dumps(data, allow_nan=False, ensure_ascii=False)
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))

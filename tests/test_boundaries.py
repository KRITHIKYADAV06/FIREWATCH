"""Tests for the Punjab + Haryana state/district picker data."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from build_boundaries import _single_ring  # noqa: E402
from data_access import _points_in_polygon_latlon  # noqa: E402
from frontend_utils import aoi_label, normalize_aoi  # noqa: E402
from grid import aoi_area_km2  # noqa: E402

BOUNDARIES = ROOT / "data" / "boundaries" / "punjab_haryana.json"


def _square(lon0, lat0, size=1.0):
    """Closed GeoJSON ring [lon, lat] for a square."""
    return [[lon0, lat0], [lon0 + size, lat0], [lon0 + size, lat0 + size],
            [lon0, lat0 + size], [lon0, lat0]]


def test_two_part_district_keeps_both_parts():
    # Two separate 1x1 degree squares; the second one wound the other way round.
    geometry = {"type": "MultiPolygon",
                "coordinates": [[_square(75, 30)], [_square(77, 30)[::-1]]]}
    ring, parts = _single_ring(geometry)
    assert parts == 2
    lats = np.array([30.5, 30.5, 30.5, 32.0])
    lons = np.array([75.5, 77.5, 76.5, 75.5])   # part 1, part 2, gap, outside
    inside = _points_in_polygon_latlon(lats, lons, ring)
    assert inside.tolist() == [True, True, False, False]
    # Area of the joined ring = area of both squares (bridges add nothing).
    one = aoi_area_km2({"type": "polygon", "coords": _single_ring(
        {"type": "Polygon", "coordinates": [_square(75, 30)]})[0]})
    both = aoi_area_km2({"type": "polygon", "coords": ring})
    assert both == pytest.approx(2 * one, rel=0.02)


def test_named_polygon_label():
    aoi = normalize_aoi({"type": "polygon", "coords": [(30, 75), (31, 75), (30, 76)],
                         "name": "Ludhiana district, Punjab"})
    assert aoi["name"] == "Ludhiana district, Punjab"
    assert aoi_label(aoi) == "Ludhiana district, Punjab"
    # Unnamed polygons keep the old label.
    assert "vertices" in aoi_label({"type": "polygon", "coords": [(30, 75), (31, 75), (30, 76)]})


def test_boundaries_file_has_both_states():
    data = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
    assert set(data["states"]) == {"Punjab", "Haryana"}
    for state in data["states"].values():
        assert len(state["coords"]) >= 3
        assert len(state["districts"]) >= 20
        for district in state["districts"].values():
            lats = [p[0] for p in district["coords"]]
            lons = [p[1] for p in district["coords"]]
            # Every vertex sits in the Punjab/Haryana region.
            assert 27.0 < min(lats) and max(lats) < 33.0
            assert 73.5 < min(lons) and max(lons) < 78.0
    assert "Ludhiana" in data["states"]["Punjab"]["districts"]
    assert "Karnal" in data["states"]["Haryana"]["districts"]

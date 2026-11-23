"""Tests for the Punjab + Haryana state/district picker data."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from build_boundaries import _join_ways, single_ring  # noqa: E402
from data_access import _points_in_polygon_latlon  # noqa: E402
from frontend_utils import aoi_label, normalize_aoi  # noqa: E402
from geo import split_bridged_ring  # noqa: E402
from grid import aoi_area_km2  # noqa: E402

BOUNDARIES = ROOT / "data" / "boundaries" / "punjab_haryana.json"


def _square(lon0, lat0, size=1.0):
    """Closed ring [lon, lat] for a square."""
    return [[lon0, lat0], [lon0 + size, lat0], [lon0 + size, lat0 + size],
            [lon0, lat0 + size], [lon0, lat0]]


def _area(ring):
    return aoi_area_km2({"type": "polygon", "coords": ring})


def test_two_part_district_keeps_both_parts():
    # Two separate 1x1 degree squares; the second one wound the other way round.
    ring, parts = single_ring([[_square(75, 30)], [_square(77, 30)[::-1]]])
    assert parts == 2
    lats = np.array([30.5, 30.5, 30.5, 32.0])
    lons = np.array([75.5, 77.5, 76.5, 75.5])   # part 1, part 2, gap, outside
    assert _points_in_polygon_latlon(lats, lons, ring).tolist() == [True, True, False, False]
    one = _area(single_ring([[_square(75, 30)]])[0])
    assert _area(ring) == pytest.approx(2 * one, rel=0.02)
    # The map splits the ring back into the two squares (no bridge line drawn).
    pieces = split_bridged_ring(ring)
    assert len(pieces) == 2 and all(len(p) == 4 for p in pieces)


def test_hole_is_left_out():
    outer, hole = _square(75, 30, 2.0), _square(75.5, 30.5, 1.0)
    ring, parts = single_ring([[outer, hole]])
    assert parts == 1
    inside = _points_in_polygon_latlon(np.array([30.2, 31.0]), np.array([75.2, 76.0]), ring)
    assert inside.tolist() == [True, False]                   # in the square, in the hole
    assert _area(ring) == pytest.approx(_area(single_ring([[outer]])[0]) * 3 / 4, rel=0.02)


def test_osm_ways_join_into_a_ring():
    # Three OSM ways forming a triangle, one of them stored backwards.
    ways = [[(75, 30), (76, 30)], [(76, 31), (76, 30)], [(76, 31), (75, 30)]]
    rings = _join_ways(ways)
    assert len(rings) == 1 and rings[0][0] == rings[0][-1] and len(rings[0]) == 4


def test_named_polygon_label():
    aoi = normalize_aoi({"type": "polygon", "coords": [(30, 75), (31, 75), (30, 76)],
                         "name": "Ludhiana district, Punjab"})
    assert aoi["name"] == "Ludhiana district, Punjab"
    assert aoi_label(aoi) == "Ludhiana district, Punjab"
    assert "vertices" in aoi_label({"type": "polygon", "coords": [(30, 75), (31, 75), (30, 76)]})


def test_boundaries_file_has_all_districts():
    data = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
    states = data["states"]
    assert set(states) == {"Punjab", "Haryana"}
    assert len(states["Punjab"]["districts"]) == 23
    assert len(states["Haryana"]["districts"]) == 22
    # Districts created after 2016 are there as separate areas.
    assert "Malerkotla" in states["Punjab"]["districts"]
    assert "Charkhi Dadri" in states["Haryana"]["districts"]
    assert {"Nuh", "Gurugram"} <= set(states["Haryana"]["districts"])
    for state in states.values():
        assert len(state["coords"]) >= 3
        for district in state["districts"].values():
            lats = [p[0] for p in district["coords"]]
            lons = [p[1] for p in district["coords"]]
            assert 27.0 < min(lats) and max(lats) < 33.0
            assert 73.5 < min(lons) and max(lons) < 78.0

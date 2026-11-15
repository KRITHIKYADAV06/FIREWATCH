import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grid import (
    _convert_drawing_to_aoi,
    add_cells,
    assign_cell,
    cell_area_km2,
    cell_bounds,
    points_in_aoi,
)


def test_assign_cell():
    result = assign_cell(30.3, 75.05)
    assert result == ("30.3_75.0", 30.35, 75.05)


def test_cell_bounds():
    assert cell_bounds("30.4_75.8") == (
        30.4, 30.5, 75.8, 75.9
    )


def test_add_cells():
    df = pd.DataFrame({
        "lat": [30.3, 30.4],
        "lon": [75.05, 75.85],
    })

    result = add_cells(df)

    assert "cell_id" in result.columns
    assert len(result) == 2
    assert result.iloc[0]["cell_id"] == "30.3_75.0"


def test_assign_cell_edge_cases():
    assert assign_cell(30.0, 75.99)[0] == "30.0_75.9"
    assert assign_cell(-0.05, 0.0)[0] == "-0.1_0.0"


def test_assign_cell_series_preserves_index():
    lat = pd.Series([30.45, 31.01], index=[7, 9])
    lon = pd.Series([75.83, 76.2], index=[7, 9])

    result = assign_cell(lat, lon)

    assert list(result.index) == [7, 9]
    assert list(result["cell_id"]) == [
        "30.4_75.8",
        "31.0_76.2",
    ]


def test_right_triangle_stays_polygon():
    drawing = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [2, 0], [0, 1], [0, 0]]
            ],
        }
    }

    result = _convert_drawing_to_aoi(drawing)

    assert result["type"] == "polygon"


def test_real_rectangle_becomes_bbox():
    drawing = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [75, 30],
                    [76, 30],
                    [76, 31],
                    [75, 31],
                    [75, 30],
                ]
            ],
        }
    }

    result = _convert_drawing_to_aoi(drawing)

    assert result == {
        "type": "bbox",
        "coords": (30.0, 31.0, 75.0, 76.0),
    }


def test_polygon_with_hole_raises():
    drawing = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [75, 30],
                    [76, 30],
                    [76, 31],
                    [75, 31],
                    [75, 30],
                ],
                [
                    [75.2, 30.2],
                    [75.3, 30.2],
                    [75.3, 30.3],
                    [75.2, 30.2],
                ],
            ],
        }
    }

    with pytest.raises(
        ValueError,
        match="Polygons with holes are not supported",
    ):
        _convert_drawing_to_aoi(drawing)


def test_bow_tie_polygon_raises():
    drawing = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [0, 0],
                    [2, 2],
                    [0, 2],
                    [2, 0],
                    [0, 0],
                ]
            ],
        }
    }

    with pytest.raises(
        ValueError,
        match="Self-intersecting polygon",
    ):
        _convert_drawing_to_aoi(drawing)


def test_invalid_vertex_latitude_raises():
    drawing = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [75, 30],
                    [76, 30],
                    [76, 95],
                    [75, 31],
                    [75, 30],
                ]
            ],
        }
    }

    with pytest.raises(
        ValueError,
        match="Invalid coordinates",
    ):
        _convert_drawing_to_aoi(drawing)


def test_invalid_coordinates_raise():
    with pytest.raises(ValueError):
        assign_cell(95, 75)

    with pytest.raises(ValueError):
        assign_cell(float("nan"), 75)


def test_add_cells_drop_invalid_rows():
    df = pd.DataFrame({
        "lat": [30.3, float("nan"), 30.5],
        "lon": [75.05, 75.2, 75.5],
    })

    result = add_cells(df, errors="drop")

    assert len(result) == 2
    assert result.attrs["dropped"] == 1
    assert result["cell_id"].notna().all()


def test_bbox_clipping_excludes_outside_point():
    aoi = {
        "type": "bbox",
        "coords": (30.445, 30.455, 75.445, 75.455),
    }

    lat = pd.Series([30.45, 30.499])
    lon = pd.Series([75.45, 75.499])

    result = points_in_aoi(lat, lon, aoi)

    assert list(result) == [True, False]


def test_cell_area_is_positive():
    assert cell_area_km2(30) > 0

# ---------------------------------------------------------------------------
# Mapping fixes (v4 review)
# ---------------------------------------------------------------------------
from grid import aoi_area_km2


def test_circle_radius_metres_to_km():
    # Leaflet.draw sends circle radius in metres.
    drawing = {
        "geometry": {"type": "Point", "coordinates": [75.8, 30.4]},
        "properties": {"radius": 25000},
    }
    aoi = _convert_drawing_to_aoi(drawing)
    assert aoi["type"] == "point"
    assert aoi["coords"] == (30.4, 75.8)
    assert aoi["radius_km"] == pytest.approx(25.0)
    # A 25 km circle is about 1,963 km².
    assert aoi_area_km2(aoi) == pytest.approx(1963.5, abs=1)


def test_collinear_polygon_raises():
    drawing = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[75, 30], [76, 31], [77, 32], [75, 30]]],
        }
    }
    with pytest.raises(ValueError, match="zero area"):
        _convert_drawing_to_aoi(drawing)


def test_repeated_corner_raises():
    drawing = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[75, 30], [76, 30], [76, 30], [76, 31], [75, 30]]],
        }
    }
    with pytest.raises(ValueError, match="repeated corners"):
        _convert_drawing_to_aoi(drawing)


def test_assign_cell_clamps_pole_and_antimeridian():
    assert assign_cell(90, 180) == ("89.9_179.9", 89.95, 179.95)
    result = assign_cell(pd.Series([90.0]), pd.Series([180.0]))
    assert result["cell_id"].iloc[0] == "89.9_179.9"


def test_floor_rule_unchanged():
    # Same +1e-9 floor rule as src/create_parquet_handoff.py
    assert assign_cell(30.4, 75.8)[0] == "30.4_75.8"
    assert assign_cell(30.39999999999, 75.8)[0] == "30.4_75.8"


def test_bbox_area_is_reasonable():
    # 1° x 1° near 30° N is about 111 km x 96 km.
    area = aoi_area_km2({"type": "bbox", "coords": (30.0, 31.0, 75.0, 76.0)})
    assert 10000 < area < 11000

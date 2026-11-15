import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from frontend_utils import (normalize_aoi, aoi_label, month_start, month_end, validate_month,
                            clamp_month_range, month_options, safe_script_json, file_signature)


def test_old_tuple_aoi_upgrades_to_geo_dict():
    a = normalize_aoi((27.6, 32.6, 73.8, 77.6))
    assert a == {"type": "bbox", "coords": (27.6, 32.6, 73.8, 77.6)}
    assert "Bounding box" in aoi_label(a)


def test_polygon_and_radius_preserved():
    p = normalize_aoi({"type": "polygon", "coords": [(30, 75), (31, 75), (30, 76)]})
    assert len(p["coords"]) == 3
    r = normalize_aoi({"type": "point", "coords": (30, 75), "radius_km": 5})
    assert r["radius_km"] == 5


@pytest.mark.parametrize("bad", [(-91, 30, 75, 76), (31, 30, 75, 76), (27, 33, 77, 76), (float('nan'), 30, 75, 76)])
def test_invalid_bbox_is_rejected(bad):
    with pytest.raises(ValueError):
        normalize_aoi(bad)


def test_invalid_point_radius_rejected():
    with pytest.raises(ValueError):
        normalize_aoi({"type": "point", "coords": (30, 75), "radius_km": -5})


def test_end_of_month_includes_final_day():
    assert month_start("2026-11") == "2026-11-01"
    assert month_end("2026-11") == "2026-11-30"
    assert month_end("2024-02") == "2024-02-29"
    assert month_end("2025-02") == "2025-02-28"


def test_date_validation_and_clamping():
    with pytest.raises(ValueError):
        validate_month("2025-13")
    assert clamp_month_range("2024-01", "2026-11", "2010-01", "2014-12") == ("2010-01", "2014-12")
    assert clamp_month_range("2020-01", "2020-12", "2010-01", "2025-12") == ("2020-01", "2020-12")


def test_month_list_includes_full_year_turnover():
    assert month_options("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]


def test_json_injection_is_neutralized():
    text = safe_script_json({"name": '</script><script>alert(1)</script>&'})
    assert "</script>" not in text
    assert "\\u003c" in text


def test_file_signature_tracks_changed_file(tmp_path):
    folder = tmp_path / "data" / "processed"
    folder.mkdir(parents=True)
    before = file_signature(tmp_path)
    (folder / "cells_monthly.parquet").write_bytes(b"sample test only")
    after = file_signature(tmp_path)
    assert before != after

"""Build the small Punjab + Haryana boundaries file used by the area picker.

One-time step (the output file is committed, so the app does not need this at run time).

Input: two GeoJSON files downloaded from geoBoundaries (gbOpen, release 9469f09):
  - geoBoundaries-IND-ADM1_simplified.geojson  (states; DataMeet India / Election
    Commission of India, 2011; CC BY 2.5 IN)
  - geoBoundaries-IND-ADM2_simplified.geojson  (districts; Pathways Data Pvt. Ltd. /
    lgdirectory.gov.in, 2021; ODbL 1.0)
  https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/
  https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/

Run:
  python src/build_boundaries.py ADM1.geojson ADM2.geojson
Output: data/boundaries/punjab_haryana.json

Uses only the Python standard library (no new packages).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "boundaries" / "punjab_haryana.json"

# ISO code in the ADM1 file -> name shown in the app (plain ASCII, no macrons).
STATES = {"IN-PB": "Punjab", "IN-HR": "Haryana"}


def _ring_area(ring):
    """Signed shoelace area of a [lon, lat] ring (square degrees; only used to compare parts)."""
    total = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _ring_centroid(ring):
    """Area-weighted centroid (lon, lat) of a [lon, lat] ring."""
    area = _ring_area(ring)
    cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return cx / (6 * area), cy / (6 * area)


def _inside(lon, lat, ring):
    """Ray-casting point-in-polygon for a [lon, lat] ring."""
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > lat) != (y2 > lat):
            x_cross = (lat - y1) * (x2 - x1) / (y2 - y1) + x1
            if lon < x_cross:
                inside = not inside
    return inside


def _outer_rings(geometry):
    """Outer ring of every part (Polygon has one part, MultiPolygon several)."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [part[0] for part in geometry["coordinates"]]
    raise ValueError(f"Unexpected geometry type: {geometry['type']}")


def _largest_part(geometry):
    """Largest outer ring (used only to place a district inside its state)."""
    return max(_outer_rings(geometry), key=lambda r: abs(_ring_area(r)))


def _to_latlon(ring):
    """[lon, lat] GeoJSON ring -> [[lat, lon], ...] rounded to 4 decimals (about 11 m)."""
    out = []
    for lon, lat in ring:
        point = [round(lat, 4), round(lon, 4)]
        if not out or out[-1] != point:
            out.append(point)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()  # the app closes the ring itself
    return out


def _single_ring(geometry):
    """All parts of a (Multi)Polygon as ONE [lat, lon] ring the app can use.

    The app's polygon AOI is a single ring. Detached parts are joined with a
    zero-width "bridge": the ring goes from the first part's start vertex to
    the next part, around it, and back along the same line. The two bridge
    edges are identical, so they cancel in the even-odd inside test and no
    area is added or lost. Returns (ring, number_of_parts).
    """
    parts = sorted(_outer_rings(geometry), key=lambda r: -abs(_ring_area(r)))
    # Same winding direction for every part, so shoelace areas add up (not cancel).
    parts = [_to_latlon(p if _ring_area(p) > 0 else p[::-1]) for p in parts]
    ring = list(parts[0])
    anchor = parts[0][0]
    for part in parts[1:]:
        ring += [anchor] + part + [part[0]]  # bridge out, go round, bridge back
    return ring, len(parts)


def build(adm1_path: Path, adm2_path: Path) -> dict:
    adm1 = json.loads(Path(adm1_path).read_text(encoding="utf-8"))
    adm2 = json.loads(Path(adm2_path).read_text(encoding="utf-8"))

    states = {}
    for feature in adm1["features"]:
        iso = feature["properties"].get("shapeISO")
        if iso in STATES:
            states[STATES[iso]] = {"geometry": feature["geometry"],
                                   "ring": _largest_part(feature["geometry"]), "districts": {}}
    missing = set(STATES.values()) - set(states)
    if missing:
        raise SystemExit(f"States not found in ADM1 file: {sorted(missing)}")

    # A district belongs to the state whose outline contains its centroid.
    for feature in adm2["features"]:
        name = feature["properties"]["shapeName"].strip()
        lon, lat = _ring_centroid(_largest_part(feature["geometry"]))
        for state in states.values():
            if _inside(lon, lat, state["ring"]):
                if name in state["districts"]:
                    raise SystemExit(f"Duplicate district name: {name}")
                coords, parts = _single_ring(feature["geometry"])
                state["districts"][name] = {"coords": coords, "parts": parts}
                break

    return {
        "description": "Punjab and Haryana state and district outlines for the FIREWATCH area picker.",
        "sources": [
            {"layer": "states", "dataset": "geoBoundaries gbOpen IND ADM1 (simplified)",
             "origin": "DataMeet India community, Election Commission of India", "year": "2011",
             "license": "CC BY 2.5 IN",
             "url": "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/"},
            {"layer": "districts", "dataset": "geoBoundaries gbOpen IND ADM2 (simplified)",
             "origin": "Pathways Data Pvt. Ltd., lgdirectory.gov.in", "year": "2021",
             "license": "ODbL 1.0 (this derived file is shared under ODbL 1.0)",
             "url": "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/"},
        ],
        "notes": "Coordinates are [lat, lon], rounded to 4 decimals. Multi-part areas are "
                 "joined into one ring by zero-width bridges (parts > 1); no area is dropped.",
        "states": {
            name: {"coords": _single_ring(s["geometry"])[0], "parts": _single_ring(s["geometry"])[1],
                   "districts": dict(sorted(s["districts"].items()))}
            for name, s in states.items()
        },
    }


def main(argv):
    if len(argv) != 3:
        raise SystemExit("Usage: python src/build_boundaries.py ADM1.geojson ADM2.geojson")
    data = build(Path(argv[1]), Path(argv[2]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    for name, state in data["states"].items():
        print(f"{name}: {len(state['coords'])} vertices, {len(state['districts'])} districts")
        for dname, d in state["districts"].items():
            flag = f"  ({d['parts']} parts joined)" if d["parts"] > 1 else ""
            print(f"   - {dname}: {len(d['coords'])} vertices{flag}")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv)

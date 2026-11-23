"""Build the Punjab + Haryana state and district boundaries file used by the area picker.

One-time step (the output file is committed, so the app does not need this at run time).

Source: OpenStreetMap administrative boundaries (© OpenStreetMap contributors,
ODbL 1.0), downloaded once from the Overpass API with this query:

  [out:json][timeout:300];
  area["ISO3166-2"="IN-PB"]->.pb; area["ISO3166-2"="IN-HR"]->.hr;
  ( rel(area.pb)["boundary"="administrative"]["admin_level"="5"];
    rel(area.hr)["boundary"="administrative"]["admin_level"="5"];
    rel["ISO3166-2"~"^IN-(PB|HR)$"]["admin_level"="4"]; );
  out geom;

OpenStreetMap is used because it has the districts created after 2016
(e.g. Charkhi Dadri, Malerkotla) as separate areas.

Run:
  python src/build_boundaries.py osm_punjab_haryana.json
Output: data/boundaries/punjab_haryana.json

Uses only the Python standard library (no new packages).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "boundaries" / "punjab_haryana.json"

STATES = {"IN-PB": "Punjab", "IN-HR": "Haryana"}

# Outlines are simplified to about 0.002 degrees (~200 m). Analysis grid cells are
# 0.1 degrees (~11 km), so this does not change which cells an area selects.
TOLERANCE_DEG = 0.002


# ---------------------------------------------------------------------------
# Small geometry helpers ([lon, lat] rings, plain Python)
# ---------------------------------------------------------------------------
def _ring_area(ring):
    """Signed shoelace area of a [lon, lat] ring (square degrees; only for comparing)."""
    total = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _inside(lon, lat, ring):
    """Ray-casting point-in-polygon for a [lon, lat] ring."""
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > lat) != (y2 > lat):
            x_cross = (lat - y1) * (x2 - x1) / (y2 - y1) + x1
            if lon < x_cross:
                inside = not inside
    return inside


def _inside_part(lon, lat, part):
    """Inside the outer ring and not inside any hole."""
    return _inside(lon, lat, part[0]) and not any(_inside(lon, lat, h) for h in part[1:])


def _bbox(ring):
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _seg_dist(p, a, b):
    """Distance from point p to segment a-b (degrees; fine for simplifying)."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def _rdp(points, tol):
    """Ramer-Douglas-Peucker line simplification (iterative, keeps both ends)."""
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        best, idx = 0.0, None
        for k in range(i + 1, j):
            d = _seg_dist(points[k], points[i], points[j])
            if d > best:
                best, idx = d, k
        if idx is not None and best > tol:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return [p for p, k in zip(points, keep) if k]


def _simplify_ring(ring, tol=TOLERANCE_DEG):
    """Simplify a closed ring; always keeps at least 3 vertices."""
    pts = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else list(ring)
    if len(pts) <= 4:
        return pts
    far = max(range(len(pts)), key=lambda k: math.hypot(pts[k][0] - pts[0][0], pts[k][1] - pts[0][1]))
    first = _rdp(pts[: far + 1], tol)
    second = _rdp(pts[far:] + [pts[0]], tol)
    out = first + second[1:-1]
    return out if len(out) >= 3 else pts


def _to_latlon(ring):
    """[lon, lat] ring -> [[lat, lon], ...] rounded to 4 decimals (about 11 m)."""
    out = []
    for lon, lat in ring:
        point = [round(lat, 4), round(lon, 4)]
        if not out or out[-1] != point:
            out.append(point)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()  # the app closes the ring itself
    return out


def single_ring(parts):
    """All parts ([outer, hole, ...] lists) as ONE simplified [lat, lon] ring the app can use.

    The app's polygon AOI is a single ring. Detached parts are joined with a
    zero-width "bridge": the ring goes from the first part's start vertex to
    the next piece, around it, and back along the same line. The two bridge
    edges are identical, so they cancel in the even-odd inside test and no
    area is added or lost. Holes are joined the same way, wound the other way
    round, so they are left out. The map splits the ring back into pieces at
    the anchor vertex, so the bridges are not drawn. Returns (ring, n_parts).
    """
    parts = sorted(parts, key=lambda p: -abs(_ring_area(p[0])))
    pieces, n_parts = [], 0
    for part in parts:
        for k, r in enumerate(part):
            want_positive = k == 0  # outer counter-clockwise, holes clockwise
            r = r if (_ring_area(r) > 0) == want_positive else r[::-1]
            latlon = _to_latlon(_simplify_ring(r))
            if len(latlon) >= 3:
                pieces.append(latlon)
                n_parts += k == 0
            elif k == 0:
                break
    ring = list(pieces[0])
    anchor = pieces[0][0]
    for piece in pieces[1:]:
        ring += [anchor] + piece + [piece[0]]  # bridge out, go round, bridge back
    return ring, n_parts


# ---------------------------------------------------------------------------
# OpenStreetMap relation -> polygon parts
# ---------------------------------------------------------------------------
def _join_ways(ways):
    """Join way coordinate lists into closed rings by matching end points."""
    ways = [w[:] for w in ways if len(w) >= 2]
    rings = []
    while ways:
        ring = ways.pop(0)
        while ring[0] != ring[-1]:
            end = ring[-1]
            for i, w in enumerate(ways):
                if w[0] == end:
                    ring += w[1:]
                elif w[-1] == end:
                    ring += w[::-1][1:]
                else:
                    continue
                ways.pop(i)
                break
            else:
                raise ValueError(f"Ring does not close (open end at {end})")
        if len(ring) >= 4:
            rings.append(ring)
    return rings


def relation_parts(element):
    """[[outer, hole, ...], ...] in [lon, lat] for an Overpass 'out geom' relation."""
    outer_ways, inner_ways = [], []
    for m in element.get("members", []):
        if m.get("type") != "way" or "geometry" not in m:
            continue
        coords = [(round(p["lon"], 7), round(p["lat"], 7)) for p in m["geometry"] if p]
        (inner_ways if m.get("role") == "inner" else outer_ways).append(coords)
    parts = [[list(r)] for r in _join_ways(outer_ways)]
    for hole in _join_ways(inner_ways):
        lon, lat = hole[0]
        for part in parts:
            if _inside(lon, lat, part[0]):
                part.append(list(hole))
                break
    return parts


def build(osm_path: Path):
    osm = json.loads(Path(osm_path).read_text(encoding="utf-8"))
    elements = [e for e in osm["elements"] if e.get("type") == "relation"]

    states = {}
    for e in elements:
        iso = e["tags"].get("ISO3166-2")
        if e["tags"].get("admin_level") == "4" and iso in STATES:
            parts = relation_parts(e)
            states[STATES[iso]] = {"parts": parts, "boxes": [_bbox(p[0]) for p in parts],
                                   "districts": {}}
    missing = set(STATES.values()) - set(states)
    if missing:
        raise SystemExit(f"States not found in OSM file: {sorted(missing)}")

    skipped = []
    for e in elements:
        if e["tags"].get("admin_level") != "5":
            continue
        tags = e["tags"]
        name = (tags.get("name:en") or tags.get("name") or "").strip()
        name = name.removesuffix(" District").removesuffix(" district").strip()
        parts = relation_parts(e)
        # Area vote: a 20 x 20 grid of sample points over the district; the state
        # containing most of them wins. Districts of neighbouring states that only
        # touch Punjab/Haryana get (almost) no votes and are left out.
        votes, total = {}, 0
        for part in parts:
            x0, y0, x1, y1 = _bbox(part[0])
            for i in range(20):
                for j in range(20):
                    lon = x0 + (i + 0.5) * (x1 - x0) / 20
                    lat = y0 + (j + 0.5) * (y1 - y0) / 20
                    if not _inside_part(lon, lat, part):
                        continue
                    total += 1
                    for sname, s in states.items():
                        if any(b[0] <= lon <= b[2] and b[1] <= lat <= b[3] and _inside_part(lon, lat, p)
                               for p, b in zip(s["parts"], s["boxes"])):
                            votes[sname] = votes.get(sname, 0) + 1
        home = max(votes, key=votes.get) if votes else None
        if home is None or votes[home] < 0.5 * total:
            skipped.append(name)
            continue
        if name in states[home]["districts"]:
            raise SystemExit(f"Duplicate district name in {home}: {name}")
        coords, n_parts = single_ring(parts)
        states[home]["districts"][name] = {"coords": coords, "parts": n_parts,
                                           "osm_relation": e["id"]}

    result = {
        "description": "Punjab and Haryana state and district outlines for the FIREWATCH area picker.",
        "sources": [{"dataset": "OpenStreetMap administrative boundaries (admin_level 4 and 5)",
                     "attribution": "© OpenStreetMap contributors",
                     "license": "ODbL 1.0 (this derived file is shared under ODbL 1.0)",
                     "url": "https://www.openstreetmap.org/copyright",
                     "osm_timestamp": osm.get("osm3s", {}).get("timestamp_osm_base", "")}],
        "notes": ("Coordinates are [lat, lon], rounded to 4 decimals and simplified to about "
                  f"{TOLERANCE_DEG} degrees. Multi-part areas are joined into one ring by "
                  "zero-width bridges (parts > 1)."),
        "states": {},
    }
    for name in STATES.values():
        coords, n_parts = single_ring(states[name]["parts"])
        result["states"][name] = {"coords": coords, "parts": n_parts,
                                  "districts": dict(sorted(states[name]["districts"].items()))}
    return result, skipped


def main(argv):
    if len(argv) != 2:
        raise SystemExit("Usage: python src/build_boundaries.py osm_punjab_haryana.json")
    data, skipped = build(Path(argv[1]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    for name, state in data["states"].items():
        print(f"{name}: {len(state['coords'])} vertices, {len(state['districts'])} districts")
        for dname, d in state["districts"].items():
            extra = f" ({d['parts']} parts)" if d["parts"] > 1 else ""
            print(f"   - {dname}: {len(d['coords'])} vertices{extra}")
    print("Left out (mostly outside Punjab/Haryana):", ", ".join(skipped) or "none")
    print(f"OSM data timestamp: {data['sources'][0]['osm_timestamp']}")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv)

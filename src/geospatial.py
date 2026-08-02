"""
Georeferencing + GeoJSON export for UK detections.

The detection boxes from src.counting.predict_boxes_global are in the site clip's
GLOBAL PIXEL space (col_off + x, row_off + y). tile_index.csv carries the clip's
geo-referencing: each tile's ground top-left (ulx, uly), the pixel size, and the
CRS (EPSG:27700, British National Grid). This module recovers the clip origin from
that and maps any global pixel to real ground coordinates, then writes GeoJSON that
QGIS opens directly.

Ground mapping (north-up grid):
    X = origin_x + gx * px
    Y = origin_y - gy * px        (image rows increase southward)
where origin = the ground coordinate of clip pixel (0, 0).
"""
from __future__ import annotations

from pathlib import Path
import csv
import json


def load_geotransform(site_dir: Path):
    """Return (origin_x, origin_y, px, crs) for a site's clip, from tile_index.csv.

    origin is the ground coordinate of global clip pixel (0,0); px is the ground
    size of one pixel (m). Recovered from any tile row via
    origin_x = ulx - col_off*px,  origin_y = uly + row_off*px.
    """
    rows = list(csv.DictReader(open(Path(site_dir) / "tile_index.csv")))
    if not rows:
        raise ValueError(f"empty tile_index in {site_dir}")
    r = rows[0]
    px = float(r["pixel_size"])
    origin_x = float(r["ulx"]) - int(r["col_off"]) * px
    origin_y = float(r["uly"]) + int(r["row_off"]) * px
    crs = r.get("crs", "EPSG:27700")
    return origin_x, origin_y, px, crs


def pixel_to_ground(gx, gy, geo):
    """Map a global clip pixel (gx, gy) to ground (X, Y) in the clip CRS."""
    origin_x, origin_y, px, _ = geo
    return origin_x + gx * px, origin_y - gy * px


def box_centroid(box):
    """Centroid (px) of an (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _epsg_num(crs: str) -> str:
    return crs.strip().split(":")[-1]


def _crs_member(crs: str) -> dict:
    # QGIS honours this named-CRS member even though RFC 7946 favours WGS84.
    return {"type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{_epsg_num(crs)}"}}


def write_points_geojson(path: Path, points, crs: str = "EPSG:27700"):
    """points: list of dicts with 'x','y' (ground coords) plus any extra props."""
    feats = []
    for p in points:
        props = {k: v for k, v in p.items() if k not in ("x", "y")}
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [p["x"], p["y"]]},
                      "properties": props})
    fc = {"type": "FeatureCollection", "crs": _crs_member(crs), "features": feats}
    Path(path).write_text(json.dumps(fc))
    return len(feats)


def write_polygons_geojson(path: Path, polygons, crs: str = "EPSG:27700"):
    """polygons: list of dicts with 'ring' (list of [X,Y] ground coords, closed)
    plus any extra props."""
    feats = []
    for poly in polygons:
        props = {k: v for k, v in poly.items() if k != "ring"}
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": [poly["ring"]]},
                      "properties": props})
    fc = {"type": "FeatureCollection", "crs": _crs_member(crs), "features": feats}
    Path(path).write_text(json.dumps(fc))
    return len(feats)


def box_to_ground_ring(box, geo):
    """Convert an (x1,y1,x2,y2) pixel box to a closed ground-coordinate ring."""
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    return [list(pixel_to_ground(cx, cy, geo)) for cx, cy in corners]

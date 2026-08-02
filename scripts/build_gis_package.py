#!/usr/bin/env python3
"""
build_gis_package.py — assemble a single QGIS-ready GeoPackage for the entrance /
occupancy spatial analysis.

Bundles, all in EPSG:27700 (British National Grid) so they line up with the
georeferenced site clips (`data/uk_retail/processed/<SITE>/<SITE>_clip.tif`):

  car_points        detected cars (centroids) for the 20 uk_test sites  [uk_adapt]
  aoi_boundary      car-park boundary polygon per site (from aois.gpkg)
  store_point       registry store location (retailer, town, capacity)
  retail_entrance   EMPTY — you digitise the shop door(s) in QGIS
  carpark_entrance  EMPTY — you digitise the vehicle entrance(s) in QGIS

The two entrance layers ship empty but with a fixed schema (site_id, kind, notes)
so you can start digitising immediately and the analysis script reads them back.

Also writes WGS84 GeoJSON copies of car_points + aoi for quick web/OSM overlay
(geojson.io, umap, etc.).

Usage:
    python scripts/build_gis_package.py
Outputs -> outputs/gis/
"""
from __future__ import annotations
from pathlib import Path
import json
import shutil
import tempfile
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
SPATIAL = ROOT / "outputs/spatial"
AOI_GPKG = ROOT / "data/uk_retail/aois.gpkg"
REGISTRY = ROOT / "data/uk_retail/geolytix/processed/uk_site_registry_final.csv"
OUT = ROOT / "outputs/gis"
GPKG = OUT / "carpark_occupancy.gpkg"
BNG = "EPSG:27700"


def test_sites(reg: pd.DataFrame) -> list[str]:
    m = (reg.accepted == "Yes") & (reg.proposed_split == "uk_test")
    return list(reg.loc[m, "site_id"])


def load_car_points(sites: list[str]) -> gpd.GeoDataFrame:
    frames = []
    for s in sites:
        p = SPATIAL / f"{s}_car_points.geojson"
        if not p.exists():
            print(f"  ! {s}: no car_points.geojson, skipped")
            continue
        d = json.loads(p.read_text())
        rows = []
        for f in d["features"]:
            x, y = f["geometry"]["coordinates"]
            pr = f.get("properties", {})
            rows.append({"site_id": pr.get("site_id", s),
                         "model": pr.get("model", "uk_adapt"),
                         "geometry": Point(x, y)})
        frames.append(gpd.GeoDataFrame(rows, crs=BNG))
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=BNG)
    gdf.insert(0, "car_id", range(1, len(gdf) + 1))
    return gdf


def empty_points(schema_cols: list[str]) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame({c: pd.Series(dtype="object") for c in schema_cols},
                           geometry=gpd.GeoSeries([], crs=BNG), crs=BNG)
    return gdf


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(REGISTRY)
    sites = test_sites(reg)
    print(f"{len(sites)} test sites")

    # --- car points ---
    cars = load_car_points(sites)
    print(f"car_points: {len(cars)} cars across {cars.site_id.nunique()} sites")

    # --- aoi boundaries ---
    aoi = gpd.read_file(AOI_GPKG, engine="pyogrio")
    aoi = aoi[aoi.site_id.isin(sites)].to_crs(BNG).reset_index(drop=True)
    aoi = aoi[["site_id", "capacity", "geometry"]]

    # --- store points from registry (WGS84 -> BNG) ---
    r = reg[reg.site_id.isin(sites)].copy()
    stores = gpd.GeoDataFrame(
        r[["site_id", "retailer", "store_name", "town", "estimated_capacity"]]
         .rename(columns={"estimated_capacity": "capacity"}),
        geometry=gpd.points_from_xy(r.longitude, r.latitude), crs="EPSG:4326"
    ).to_crs(BNG).reset_index(drop=True)

    # --- empty entrance layers to digitise ---
    retail = empty_points(["site_id", "kind", "notes"])
    carpark = empty_points(["site_id", "kind", "notes"])

    # --- write single GeoPackage (layer per feature class) ---
    # GPKG is SQLite; SQLite file locking fails on some mounted/networked folders,
    # so build in local temp then copy the finished file to outputs/.
    tmp = Path(tempfile.mkdtemp()) / "carpark_occupancy.gpkg"
    layers = [("car_points", cars), ("aoi_boundary", aoi), ("store_point", stores),
              ("retail_entrance", retail), ("carpark_entrance", carpark)]
    empty_layers = {"retail_entrance", "carpark_entrance"}
    for i, (name, gdf) in enumerate(layers):
        kw = {"geometry_type": "Point"} if name in empty_layers else {}
        gdf.to_file(tmp, layer=name, driver="GPKG", engine="pyogrio",
                    append=(i > 0), **kw)
    # Copy (O_TRUNC), not unlink+write: some mounts block unlink but allow
    # overwrite. Zero any stale SQLite -journal so QGIS/GDAL won't see a
    # half-open transaction.
    shutil.copyfile(tmp, GPKG)
    jnl = GPKG.with_name(GPKG.name + "-journal")
    if jnl.exists():
        open(jnl, "wb").close()

    # --- WGS84 GeoJSON copies for web / OSM overlay ---
    # (write to temp then copy — same mount-unlink limitation as the GPKG)
    def _write_geojson(gdf, name):
        t = Path(tempfile.mkdtemp()) / name
        gdf.to_file(t, driver="GeoJSON")
        shutil.copyfile(t, OUT / name)
    _write_geojson(cars.to_crs(4326), "car_points_wgs84.geojson")
    _write_geojson(aoi.to_crs(4326), "aoi_boundary_wgs84.geojson")

    print(f"\nGeoPackage -> {GPKG}")
    print("layers:", ["car_points", "aoi_boundary", "store_point",
                      "retail_entrance", "carpark_entrance"])
    print(f"WGS84 GeoJSON -> {OUT}/car_points_wgs84.geojson, aoi_boundary_wgs84.geojson")


if __name__ == "__main__":
    main()

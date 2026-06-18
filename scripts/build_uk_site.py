#!/usr/bin/env python3
"""
build_uk_site.py  —  Clip + tile + pre-label one UK site, end to end.

Run from the PROJECT ROOT (the carpark-project folder), inside your .venv:

    python scripts/build_uk_site.py UK001

What it does (one site at a time):
  1. Finds the raw Digimap tiles in   data/uk_retail/raw/<SITE>/*.jpg   (+ their .jgw world files)
  2. Mosaics them into one image, assigns EPSG:27700
  3. Clips the mosaic to the <SITE> polygon in the AOI GeoPackage
  4. Tiles the clip into 640x640 with overlap, skipping near-empty tiles
  5. Pre-labels each tile with the combined model at low confidence
  6. Writes YOLO-format labels + tile_index.csv (so boxes can map back to ground coords)

Outputs go to:  data/uk_retail/processed/<SITE>/
  <SITE>_mosaic.tif   <SITE>_clip.tif   tiles/<SITE>_r{r}_c{c}.png + .txt
  tile_index.csv   data.yaml

This is a SMOKE TEST: run it on UK001, eyeball the outputs, then repeat for the rest.
The imagery is licensed (EDINA/Getmapping) so everything stays local; this is not run anywhere else.
"""
import sys, csv, glob
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.mask import mask as rio_mask
from rasterio.io import MemoryFile
import geopandas as gpd
from PIL import Image

# ---------------- config ----------------
PROJECT_ROOT = Path.cwd()                  # run from the repo root
TILE = 640                                 # tile size in pixels
OVERLAP = 128                              # ~20% overlap so edge cars survive
STRIDE = TILE - OVERLAP                     # 512
PRELABEL_CONF = 0.20                        # LOW on purpose: over-suggest, you delete false positives
TARGET_CRS = "EPSG:27700"                   # British National Grid
SKIP_IF_EMPTY_FRAC = 0.98                   # skip tiles that are essentially all background
MODEL_PATH = PROJECT_ROOT / "models/runs/combined_yolov8s_25cm/weights/best.pt"
# ----------------------------------------

def find_aoi_gpkg():
    for name in ("aois.gpkg", "aosis.gpkg"):
        p = PROJECT_ROOT / "data/uk_retail" / name
        if p.exists():
            return p
    raise FileNotFoundError("No aois.gpkg / aosis.gpkg found in data/uk_retail/")

def load_site_polygon(gpkg, site_id):
    gdf = gpd.read_file(gpkg)                       # reads first layer
    if "site_id" not in gdf.columns:
        raise ValueError(f"'site_id' column not in {gpkg}; columns = {list(gdf.columns)}")
    sub = gdf[gdf["site_id"] == site_id]
    if sub.empty:
        raise ValueError(f"No polygon with site_id == '{site_id}' in {gpkg}. "
                         f"Found: {sorted(gdf['site_id'].dropna().unique())}")
    sub = sub.to_crs(TARGET_CRS)
    return [geom for geom in sub.geometry]

def open_with_crs(path):
    """Open a JPEG+.jgw; the world file gives the transform but no CRS, so assign 27700."""
    src = rasterio.open(path)
    if src.crs is None:
        # re-wrap with the known CRS without touching pixels
        prof = src.profile
        prof.update(crs=rasterio.crs.CRS.from_string(TARGET_CRS))
        data = src.read()
        mem = MemoryFile()
        with mem.open(**prof) as tmp:
            tmp.write(data)
        src.close()
        return mem.open()
    return src

def main(site_id):
    raw_dir = PROJECT_ROOT / "data/uk_retail/raw" / site_id
    out_dir = PROJECT_ROOT / "data/uk_retail/processed" / site_id
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(glob.glob(str(raw_dir / "*.jpg")) + glob.glob(str(raw_dir / "*.jpeg")))
    if not jpgs:
        raise FileNotFoundError(f"No .jpg tiles in {raw_dir}")
    print(f"[1] {site_id}: found {len(jpgs)} raw tile(s)")

    # ---- 2. mosaic ----
    srcs = [open_with_crs(p) for p in jpgs]
    mosaic, out_transform = rio_merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
                transform=out_transform, crs=TARGET_CRS, count=mosaic.shape[0])
    mosaic_path = out_dir / f"{site_id}_mosaic.tif"
    with rasterio.open(mosaic_path, "w", **meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    print(f"[2] mosaic written: {mosaic_path.name}  ({mosaic.shape[2]}x{mosaic.shape[1]} px)")

    # ---- 3. clip to AOI ----
    polys = load_site_polygon(find_aoi_gpkg(), site_id)
    with rasterio.open(mosaic_path) as src:
        clipped, clip_transform = rio_mask(src, polys, crop=True, filled=True, nodata=0)
        clip_meta = src.meta.copy()
    clip_meta.update(height=clipped.shape[1], width=clipped.shape[2],
                     transform=clip_transform, nodata=0)
    clip_path = out_dir / f"{site_id}_clip.tif"
    with rasterio.open(clip_path, "w", **clip_meta) as dst:
        dst.write(clipped)
    H, W = clipped.shape[1], clipped.shape[2]
    px = clip_transform.a   # ground size of one pixel (m) — should be ~0.25
    print(f"[3] clip written: {clip_path.name}  ({W}x{H} px, pixel≈{px:.3f} m)")

    # ---- 4. tile ----
    rgb = np.transpose(clipped[:3], (1, 2, 0)).astype(np.uint8)   # HxWx3

    def offsets(extent):
        """Tile start offsets covering `extent` px with TILE/STRIDE.
        Stops once a tile reaches the edge, so an extent <= TILE yields a
        single tile (no redundant overlapping tiles)."""
        if extent <= TILE:
            return [0]
        offs = list(range(0, extent - TILE + 1, STRIDE))
        # ensure the final strip up to the edge is covered
        if offs[-1] != extent - TILE:
            offs.append(extent - TILE)
        return offs

    index_rows, kept, skipped = [], 0, 0
    for r0 in offsets(H):
        for c0 in offsets(W):
            r1, c1 = min(r0 + TILE, H), min(c0 + TILE, W)
            tile = rgb[r0:r1, c0:c1]
            # pad to full TILE so the model always sees 640x640
            pad = np.zeros((TILE, TILE, 3), np.uint8)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            empty_frac = np.mean(np.all(pad == 0, axis=2))
            if empty_frac > SKIP_IF_EMPTY_FRAC:
                skipped += 1
                continue
            name = f"{site_id}_r{r0:05d}_c{c0:05d}"
            Image.fromarray(pad).save(tiles_dir / f"{name}.png")
            ulx, uly = clip_transform * (c0, r0)   # ground coords of this tile's top-left
            index_rows.append(dict(tile=name, site_id=site_id, col_off=c0, row_off=r0,
                                   width=TILE, height=TILE, ulx=ulx, uly=uly,
                                   pixel_size=px, crs=TARGET_CRS))
            kept += 1
    print(f"[4] tiled: {kept} tiles kept, {skipped} near-empty skipped")

    with open(out_dir / "tile_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader(); w.writerows(index_rows)

    # ---- 5. pre-label with the combined model ----
    if not MODEL_PATH.exists():
        print(f"[5] SKIPPED pre-labelling — model not found at {MODEL_PATH}")
    else:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        total = 0
        for row in index_rows:
            tp = tiles_dir / f"{row['tile']}.png"
            res = model.predict(str(tp), conf=PRELABEL_CONF, imgsz=TILE, verbose=False)[0]
            lines = []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2 / TILE, (y1 + y2) / 2 / TILE
                bw, bh = (x2 - x1) / TILE, (y2 - y1) / TILE
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (tiles_dir / f"{row['tile']}.txt").write_text("\n".join(lines))
            total += len(lines)
        print(f"[5] pre-labelled {kept} tiles — {total} candidate boxes "
              f"({total / max(kept,1):.1f} per tile) at conf={PRELABEL_CONF}")

    # ---- 6. data.yaml for CVAT/Roboflow import ----
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\ntrain: tiles\nval: tiles\nnc: 1\nnames: ['car']\n")
    print(f"[6] done. Outputs in {out_dir}")

if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "UK001"
    main(site)
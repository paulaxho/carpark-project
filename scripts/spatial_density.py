#!/usr/bin/env python3
"""
spatial_density.py — turn detections into spatial data: georeferenced car points
and a parking-density heatmap per site.

Pipeline per site:
  1. run the model at its operating threshold, de-duplicate across tile seams;
  2. box -> centroid, georeferenced to EPSG:27700 via tile_index.csv;
  3. export a GeoJSON of car points (and optionally box polygons) that opens in
     QGIS for joining with contextual data (entrances, aisles, road access);
  4. render a kernel-density (KDE) heatmap of where cars cluster, overlaid on the
     reconstructed site image.

A single capture shows WHERE parking concentrates; repeated captures over time
would show how those hotspots grow and shift — the activity signal.

Defaults: the uk_adapt model at its F1-optimal threshold (occupancy_thresholds.csv).

Usage (from project root):
    python scripts/spatial_density.py                 # all 20 test sites
    python scripts/spatial_density.py UK002 UK023     # just these
    python scripts/spatial_density.py UK002 --polygons --bandwidth 5
"""
import sys, csv, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.counting import load_offsets, predict_boxes_global, dedup_keep
from src.geospatial import (load_geotransform, pixel_to_ground, box_centroid,
                            box_to_ground_ring, write_points_geojson,
                            write_polygons_geojson)

import numpy as np

PROCESSED = ROOT / "data/uk_retail/processed"
REGISTRY = ROOT / "data/uk_retail/geolytix/processed/uk_site_registry_final.csv"
THRESHOLDS = ROOT / "outputs/stats/occupancy_thresholds.csv"
GEO_OUT = ROOT / "outputs/spatial"
FIG_OUT = ROOT / "outputs/figures/density"
MERGE_IOU = 0.5
IMGSZ = 640
WEIGHTS = {
    "baseline": ROOT / "yolov8s.pt",
    "transfer": ROOT / "models/runs/combined_yolov8s_25cm/weights/best.pt",
    "uk_adapt": ROOT / "models/runs/uk_adapt_yolov8s_25cm/weights/best.pt",
}


def test_sites():
    return [r["site_id"] for r in csv.DictReader(open(REGISTRY))
            if r["accepted"] == "Yes" and r["proposed_split"] == "uk_test"]


def threshold_for(model):
    if THRESHOLDS.exists():
        for r in csv.DictReader(open(THRESHOLDS)):
            if r["model"] == model:
                return float(r["tau"])
    return {"baseline": 0.004, "transfer": 0.012, "uk_adapt": 0.477}[model]


def build_canvas(site_dir, offs):
    from PIL import Image
    W = max(co + w for co, ro, w, h in offs.values())
    H = max(ro + h for co, ro, w, h in offs.values())
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    tiles = Path(site_dir) / "tiles"
    for stem, (co, ro, w, h) in offs.items():
        p = tiles / f"{stem}.png"
        if p.exists():
            canvas.paste(Image.open(p).convert("RGB"), (co, ro))
    return canvas, W, H


def kde_surface(cx, cy, W, H, step, sigma_px):
    """Gaussian KDE on a coarse grid (pure numpy). Returns (xs, ys, density)."""
    xs = np.arange(0, W, step)
    ys = np.arange(0, H, step)
    gx, gy = np.meshgrid(xs, ys)
    dens = np.zeros_like(gx, dtype=float)
    if len(cx):
        inv = 1.0 / (2.0 * sigma_px ** 2)
        for x, y in zip(cx, cy):
            dens += np.exp(-((gx - x) ** 2 + (gy - y) ** 2) * inv)
    return xs, ys, dens


def render_heatmap(site, site_dir, offs, cx, cy, out_png, sigma_m, px, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canvas, W, H = build_canvas(site_dir, offs)
    step = max(2, round(1.0 / px))                    # ~1 m grid cells
    sigma_px = max(4.0, sigma_m / px)                 # bandwidth in pixels
    _, _, dens = kde_surface(cx, cy, W, H, step, sigma_px)
    dens = np.ma.masked_less(dens, dens.max() * 0.04) if dens.max() > 0 else dens

    fig, ax = plt.subplots(figsize=(W / 120, H / 120))
    ax.imshow(canvas, extent=[0, W, H, 0])
    ax.imshow(np.asarray(canvas).mean(2), cmap="gray", alpha=0.45, extent=[0, W, H, 0])
    hm = ax.imshow(dens, cmap="inferno", alpha=0.6, extent=[0, W, H, 0],
                   interpolation="bilinear")
    ax.scatter(cx, cy, s=6, c="cyan", edgecolor="none", alpha=0.6)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title(f"{site} — parking density ({len(cx)} cars, {model}, "
                 f"{sigma_m:.0f} m bandwidth)", fontsize=11)
    cb = fig.colorbar(hm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("relative parking density", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sites", nargs="*", help="site IDs (default: all uk_test)")
    ap.add_argument("--model", default="uk_adapt", choices=list(WEIGHTS))
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--bandwidth", type=float, default=6.0, help="KDE bandwidth (m)")
    ap.add_argument("--polygons", action="store_true", help="also export box polygons")
    args = ap.parse_args()

    from ultralytics import YOLO
    GEO_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    sites = args.sites or test_sites()
    conf = args.conf if args.conf is not None else threshold_for(args.model)
    model = YOLO(str(WEIGHTS[args.model]))
    print(f"model={args.model}  conf={conf:.3f}  bandwidth={args.bandwidth} m  sites={len(sites)}")

    for site in sites:
        site_dir = PROCESSED / site
        if not (site_dir / "tile_index.csv").exists():
            print(f"  ! {site}: no tile_index.csv, skipped"); continue
        offs = load_offsets(site_dir)
        geo = load_geotransform(site_dir)
        px = geo[2]; crs = geo[3]

        kept = dedup_keep([(b, t) for b, t, _ in
                           predict_boxes_global(model, site_dir, conf, IMGSZ, offs)], MERGE_IOU)
        boxes = [b for b, _ in kept]

        # centroids -> georeferenced points
        cx, cy, points = [], [], []
        for b in boxes:
            px_c, py_c = box_centroid(b)
            cx.append(px_c); cy.append(py_c)
            gx, gy = pixel_to_ground(px_c, py_c, geo)
            points.append({"x": gx, "y": gy, "site_id": site, "model": args.model})
        n = write_points_geojson(GEO_OUT / f"{site}_car_points.geojson", points, crs)

        if args.polygons:
            polys = [{"ring": box_to_ground_ring(b, geo), "site_id": site} for b in boxes]
            write_polygons_geojson(GEO_OUT / f"{site}_car_boxes.geojson", polys, crs)

        render_heatmap(site, site_dir, offs, np.array(cx), np.array(cy),
                       FIG_OUT / f"{site}_density.png", args.bandwidth, px, args.model)
        print(f"  {site}: {n} cars -> {site}_car_points.geojson + {site}_density.png")

    print(f"\nGeoJSON -> {GEO_OUT}\nHeatmaps -> {FIG_OUT}")
    print("Open the .geojson in QGIS (EPSG:27700) to join with contextual layers.")


if __name__ == "__main__":
    main()

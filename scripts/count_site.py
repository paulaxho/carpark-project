#!/usr/bin/env python3
"""
count_site.py — true unique counts for a site, de-duplicating tile overlaps.

Reports BOTH:
  - detected_unique : model pre-labels, de-duped across tiles
  - truth_unique    : hand-annotated ground truth, de-duped across tiles
  - recall          : detected_unique / truth_unique

The de-duplication (map every box to global clip coords via tile_index.csv and
merge boxes from DIFFERENT tiles that overlap at IoU >= threshold) now lives in
src/counting.py, shared with select_threshold.py and estimate_occupancy.py.

GROUND TRUTH is read from the tile .txt files in tiles/.
DETECTED is read from a saved copy of the model pre-labels (tiles/prelabels/) if
present; otherwise the combined model is re-run on the tiles.

Usage (from project root):
    python scripts/count_site.py UK011 [IOU]
Default IoU merge threshold = 0.5
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.counting import (load_offsets, to_xyxy, dedup, truth_boxes_global,
                          predict_boxes_global)

ROOT = Path.cwd()
site = sys.argv[1] if len(sys.argv) > 1 else "UK011"
IOU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

site_dir = ROOT / "data/uk_retail/processed" / site
tiles = site_dir / "tiles"
MODEL = ROOT / "models/runs/combined_yolov8s_25cm/weights/best.pt"
PRELABEL_CONF = 0.20

offs = load_offsets(site_dir)

# ---- TRUTH: from tiles/<tile>.txt ----
truth_boxes = truth_boxes_global(site_dir, offs)
truth_unique, truth_dups = dedup(truth_boxes, IOU)
truth_sum = len(truth_boxes)

# ---- DETECTED: prefer saved prelabels/, else re-run model ----
prelabel_dir = tiles / "prelabels"
det_boxes = None
if prelabel_dir.exists():
    det_boxes = []
    for stem, (co, ro, W, H) in offs.items():
        p = prelabel_dir / f"{stem}.txt"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            q = line.split()
            if len(q) != 5:
                continue
            _, cx, cy, w, h = map(float, q)
            det_boxes.append((to_xyxy(co + cx * W, ro + cy * H, w * W, h * H), stem))
    det_src = "saved prelabels/"
elif MODEL.exists():
    det_boxes = [(bb, tile) for bb, tile, _ in
                 predict_boxes_global(MODEL, site_dir, PRELABEL_CONF, offs=offs)]
    det_src = "re-run model"
else:
    det_src = "UNAVAILABLE (no prelabels/ and no model)"

print(f"Site {site}  (IoU merge {IOU})  tiles={len(offs)}")
print(f"  TRUTH    per-tile sum {truth_sum:4d}  dups {truth_dups:3d}  -> UNIQUE {truth_unique}")
if det_boxes is not None:
    det_unique, det_dups = dedup(det_boxes, IOU)
    det_sum = len(det_boxes)
    print(f"  DETECTED per-tile sum {det_sum:4d}  dups {det_dups:3d}  -> UNIQUE {det_unique}   (src: {det_src})")
    rec = 100 * det_unique / truth_unique if truth_unique else 0
    print(f"  RECALL (detected_unique / truth_unique) = {rec:.1f}%")
    print(f"\n  tracker row: detected={det_unique}, truth={truth_unique}, recall_pct={rec:.1f}")
else:
    print(f"  DETECTED {det_src}")
    print(f"\n  tracker row: detected=?, truth={truth_unique}")

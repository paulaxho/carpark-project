#!/usr/bin/env python3
"""
count_site.py — true unique counts for a site, de-duplicating tile overlaps.

Reports BOTH:
  - detected_unique : model pre-labels, de-duped across tiles
  - truth_unique    : hand-annotated ground truth, de-duped across tiles
  - recall          : detected_unique / truth_unique

Multi-tile sites overlap, so a car in a seam can be boxed twice. Summing per-tile
boxes over-counts. We map every box to global clip coords (via tile_index.csv) and
merge boxes from DIFFERENT tiles that overlap heavily (IoU >= threshold) = same car.

GROUND TRUTH is read from the tile .txt files in tiles/.
DETECTED is read from a saved copy of the model pre-labels. Because annotation
overwrote the pre-labels in tiles/, this script looks for them in tiles/prelabels/
if present; otherwise it re-runs the model on the tiles to get detections.

Usage (from project root):
    python scripts/count_site.py UK011 [IOU]
Default IoU merge threshold = 0.5
"""
import sys, csv
from pathlib import Path

ROOT = Path.cwd()
site = sys.argv[1] if len(sys.argv) > 1 else "UK011"
IOU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

base  = ROOT / "data/uk_retail/processed" / site
tiles = base / "tiles"
index = base / "tile_index.csv"
MODEL = ROOT / "models/runs/combined_yolov8s_25cm/weights/best.pt"
PRELABEL_CONF = 0.20
TILE = 640

offs = {}
with open(index) as f:
    for row in csv.DictReader(f):
        offs[row["tile"]] = (int(row["col_off"]), int(row["row_off"]),
                             int(row["width"]), int(row["height"]))

def xyxy(gx, gy, w, h): return (gx-w/2, gy-h/2, gx+w/2, gy+h/2)
def iou(a, b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,ix1-ix0),max(0,iy1-iy0); inter=iw*ih
    if inter<=0: return 0.0
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def dedup(boxes):
    """boxes: list of (xyxy_global, tile). Returns (unique_count, dup_count)."""
    kept=[]; dups=0
    for bb,tile in boxes:
        if any(kt!=tile and iou(bb,kb)>=IOU for kb,kt in kept):
            dups+=1
        else:
            kept.append((bb,tile))
    return len(kept), dups

def load_boxes_from_txt(get_path):
    boxes=[]; per_tile={}
    for stem,(co,ro,W,H) in offs.items():
        p=get_path(stem)
        n=0
        if p and p.exists():
            for line in p.read_text().splitlines():
                q=line.split()
                if len(q)!=5: continue
                _,cx,cy,w,h=map(float,q)
                boxes.append((xyxy(co+cx*W, ro+cy*H, w*W, h*H), stem)); n+=1
        per_tile[stem]=n
    return boxes, per_tile

# ---- TRUTH: from tiles/<tile>.txt ----
truth_boxes, truth_per = load_boxes_from_txt(lambda s: tiles / f"{s}.txt")
truth_unique, truth_dups = dedup(truth_boxes)

# ---- DETECTED: prefer saved prelabels/, else re-run model ----
prelabel_dir = tiles / "prelabels"
if prelabel_dir.exists():
    det_boxes, det_per = load_boxes_from_txt(lambda s: prelabel_dir / f"{s}.txt")
    det_src = "saved prelabels/"
elif MODEL.exists():
    from ultralytics import YOLO
    m = YOLO(str(MODEL)); det_boxes=[]; det_per={}
    for stem,(co,ro,W,H) in offs.items():
        res = m.predict(str(tiles / f"{stem}.png"), conf=PRELABEL_CONF, imgsz=TILE, verbose=False)[0]
        n=0
        for b in res.boxes:
            x1,y1,x2,y2=b.xyxy[0].tolist()
            cx,cy=(x1+x2)/2,(y1+y2)/2; w,h=(x2-x1),(y2-y1)
            det_boxes.append((xyxy(co+cx, ro+cy, w, h), stem)); n+=1
        det_per[stem]=n
    det_src = "re-run model"
else:
    det_boxes=None; det_src="UNAVAILABLE (no prelabels/ and no model)"

print(f"Site {site}  (IoU merge {IOU})  tiles={len(offs)}")
print(f"  TRUTH    per-tile sum {sum(truth_per.values()):4d}  dups {truth_dups:3d}  -> UNIQUE {truth_unique}")
if det_boxes is not None:
    det_unique, det_dups = dedup(det_boxes)
    print(f"  DETECTED per-tile sum {sum(det_per.values()):4d}  dups {det_dups:3d}  -> UNIQUE {det_unique}   (src: {det_src})")
    rec = 100*det_unique/truth_unique if truth_unique else 0
    print(f"  RECALL (detected_unique / truth_unique) = {rec:.1f}%")
    print(f"\n  tracker row: detected={det_unique}, truth={truth_unique}, recall_pct={rec:.1f}")
else:
    print(f"  DETECTED {det_src}")
    print(f"\n  tracker row: detected=?, truth={truth_unique}")
#!/usr/bin/env python3
"""
src/counting.py — shared counting / de-duplication core.

A car that straddles a tile seam is boxed in two overlapping tiles; summing
per-tile boxes over-counts it. The fix (used for both ground truth and model
predictions) is to map every box to GLOBAL clip coordinates via tile_index.csv,
then merge boxes from DIFFERENT tiles that overlap heavily (IoU >= threshold):
those are the same physical car.

Coordinate conventions
----------------------
- Ground-truth .txt files are YOLO-normalised: "cls cx cy w h" in [0,1] of the
  tile. Global pixel box = tile offset + (cx*W, cy*H, w*W, h*H).
- Model boxes come back in tile PIXEL coordinates (xyxy); global = tile offset +
  those pixels directly (no *W/*H).
"""
from __future__ import annotations
from pathlib import Path
import csv

DEFAULT_IMGSZ = 640
DEFAULT_MERGE_IOU = 0.5


# --------------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------------- #
def load_offsets(site_dir: Path) -> dict:
    """tile stem -> (col_off, row_off, width, height) from tile_index.csv."""
    offs = {}
    with open(Path(site_dir) / "tile_index.csv") as f:
        for row in csv.DictReader(f):
            offs[row["tile"]] = (int(row["col_off"]), int(row["row_off"]),
                                 int(row["width"]), int(row["height"]))
    return offs


def to_xyxy(cx, cy, w, h):
    """Centre/size -> (x0, y0, x1, y1)."""
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def dedup(boxes, iou_thr: float = DEFAULT_MERGE_IOU):
    """Merge same-car boxes seen in different tiles.

    boxes : list of (xyxy_global, tile_stem).
    Returns (unique_count, duplicate_count). A box is a duplicate only if it
    overlaps a kept box from a DIFFERENT tile at IoU >= iou_thr (overlaps within
    the same tile are genuine separate cars and are never merged).
    """
    kept, dups = [], 0
    for bb, tile in boxes:
        if any(kt != tile and iou(bb, kb) >= iou_thr for kb, kt in kept):
            dups += 1
        else:
            kept.append((bb, tile))
    return len(kept), dups


def dedup_keep(boxes, iou_thr: float = DEFAULT_MERGE_IOU):
    """Same de-duplication as dedup(), but returns the kept boxes rather than a
    count. Returns a list of (xyxy_global, tile_stem) for the surviving unique
    boxes (used for visualisation, where the actual boxes are needed)."""
    kept = []
    for bb, tile in boxes:
        if any(kt != tile and iou(bb, kb) >= iou_thr for kb, kt in kept):
            continue
        kept.append((bb, tile))
    return kept


# --------------------------------------------------------------------------- #
#  Ground truth (from tile .txt)
# --------------------------------------------------------------------------- #
def truth_boxes_global(site_dir: Path, offs: dict | None = None):
    """Global-coord GT boxes for a site: list of (xyxy_global, tile_stem)."""
    site_dir = Path(site_dir)
    offs = offs or load_offsets(site_dir)
    tiles = site_dir / "tiles"
    boxes = []
    for stem, (co, ro, W, H) in offs.items():
        p = tiles / f"{stem}.txt"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            q = line.split()
            if len(q) != 5:
                continue
            _, cx, cy, w, h = map(float, q)
            boxes.append((to_xyxy(co + cx * W, ro + cy * H, w * W, h * H), stem))
    return boxes


def truth_boxes_local(site_dir: Path, offs: dict | None = None):
    """Per-tile GT in tile-pixel coords: {tile_stem: [xyxy_local, ...]}."""
    site_dir = Path(site_dir)
    offs = offs or load_offsets(site_dir)
    tiles = site_dir / "tiles"
    out = {}
    for stem, (co, ro, W, H) in offs.items():
        p = tiles / f"{stem}.txt"
        rows = []
        if p.exists():
            for line in p.read_text().splitlines():
                q = line.split()
                if len(q) != 5:
                    continue
                _, cx, cy, w, h = map(float, q)
                rows.append(to_xyxy(cx * W, cy * H, w * W, h * H))
        out[stem] = rows
    return out


def truth_count(site_dir: Path, iou_thr: float = DEFAULT_MERGE_IOU) -> int:
    """De-duplicated unique GT car count for a site."""
    n, _ = dedup(truth_boxes_global(site_dir), iou_thr)
    return n


# --------------------------------------------------------------------------- #
#  Predictions (lazy Ultralytics)
# --------------------------------------------------------------------------- #
def _load_model(model):
    """Accept a YOLO instance or a weights path; return a YOLO instance."""
    if hasattr(model, "predict"):
        return model
    from ultralytics import YOLO  # lazy: keeps this module import-light
    return YOLO(str(model))


def predict_boxes_local(model, site_dir: Path, conf: float,
                        imgsz: int = DEFAULT_IMGSZ, offs: dict | None = None):
    """Per-tile predictions in tile-pixel coords.

    Returns {tile_stem: [(xyxy_local, score), ...]}. Class-agnostic: every
    predicted box is kept regardless of label (single-class models emit one
    class anyway; the stock baseline is scored class-agnostically to match the
    single_cls detection results).
    """
    site_dir = Path(site_dir)
    offs = offs or load_offsets(site_dir)
    m = _load_model(model)
    tiles = site_dir / "tiles"
    out = {}
    for stem in offs:
        png = tiles / f"{stem}.png"
        rows = []
        if png.exists():
            res = m.predict(str(png), conf=conf, imgsz=imgsz, verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                score = float(b.conf[0]) if b.conf is not None else 1.0
                rows.append(((x1, y1, x2, y2), score))
        out[stem] = rows
    return out


def predict_boxes_global(model, site_dir: Path, conf: float,
                         imgsz: int = DEFAULT_IMGSZ, offs: dict | None = None):
    """Global-coord predictions: list of (xyxy_global, tile_stem, score)."""
    site_dir = Path(site_dir)
    offs = offs or load_offsets(site_dir)
    local = predict_boxes_local(model, site_dir, conf, imgsz, offs)
    boxes = []
    for stem, (co, ro, W, H) in offs.items():
        for (x1, y1, x2, y2), score in local.get(stem, []):
            boxes.append(((co + x1, ro + y1, co + x2, ro + y2), stem, score))
    return boxes


def site_count(model, site_dir: Path, conf: float,
               iou_thr: float = DEFAULT_MERGE_IOU,
               imgsz: int = DEFAULT_IMGSZ, offs: dict | None = None) -> int:
    """De-duplicated unique predicted car count for a site at confidence `conf`."""
    boxes = [(bb, tile) for bb, tile, _ in
             predict_boxes_global(model, site_dir, conf, imgsz, offs)]
    n, _ = dedup(boxes, iou_thr)
    return n


# --------------------------------------------------------------------------- #
#  Threshold selection support (tile-level detection matching)
# --------------------------------------------------------------------------- #
def match_scores(pred_local: dict, truth_local: dict, iou_match: float = 0.5):
    """Greedy per-tile TP/FP labelling for an F1 sweep.

    For each tile, predictions are sorted by score (desc) and greedily matched to
    the highest-IoU unmatched GT box with IoU >= iou_match. Returns
    (scored_flags, n_truth) where scored_flags is a list of (score, is_tp) over
    all tiles and n_truth is the total number of GT boxes.
    """
    scored, n_truth = [], 0
    for stem, gts in truth_local.items():
        n_truth += len(gts)
        preds = sorted(pred_local.get(stem, []), key=lambda r: -r[1])
        used = [False] * len(gts)
        for pbox, score in preds:
            best_j, best_iou = -1, iou_match
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                v = iou(pbox, g)
                if v >= best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                used[best_j] = True
                scored.append((score, 1))
            else:
                scored.append((score, 0))
    return scored, n_truth


def f1_optimal_threshold(scored, n_truth):
    """Given (score, is_tp) pairs and the GT total, return the confidence that
    maximises F1, plus (f1, precision, recall) at that threshold.

    Sweeps candidate thresholds at each observed score. Ties broken toward the
    higher threshold (fewer false positives)."""
    if n_truth == 0 or not scored:
        return 0.0, 0.0, 0.0, 0.0
    order = sorted(scored, key=lambda r: -r[0])  # high score first
    best = (0.0, 0.0, 0.0, 0.0)  # (tau, f1, P, R)
    tp = fp = 0
    for i, (score, is_tp) in enumerate(order):
        tp += is_tp
        fp += (1 - is_tp)
        # only evaluate at a genuine threshold boundary (last of equal scores)
        if i + 1 < len(order) and order[i + 1][0] == score:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_truth
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        if f1 > best[1]:
            best = (score, f1, precision, recall)
    return best

#!/usr/bin/env python3
"""
select_threshold.py — freeze each model's deployment confidence threshold.

The occupancy count must be produced at a real operating point, not the
threshold-free mAP setting. Following Section 3.6, the operating threshold is the
confidence that maximises F1 on the CALIBRATION-VAL sites (UK020, UK009) -- the
frozen fine-tune monitoring split. It is chosen with NO contact with the 20 test
sites, then applied unchanged in estimate_occupancy.py.

For each model:
  1. predict on the pooled calibration-val tiles at a low confidence floor,
  2. greedily match detections to GT at IoU 0.5 (tile-level, class-agnostic),
  3. sweep the confidence and take tau = argmax F1.

Output: outputs/stats/occupancy_thresholds.csv
    model, weights, tau, f1_val, precision_val, recall_val, n_truth

Usage (from project root):
    python scripts/select_threshold.py
"""
import sys, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.counting import (load_offsets, predict_boxes_local, truth_boxes_local,
                          match_scores, f1_optimal_threshold)

VAL_SITES = ["UK020", "UK009"]        # frozen calibration_val_sites (configs/yolo_uk.yaml)
CONF_FLOOR = 0.001                     # sweep from here
IMGSZ = 640
MATCH_IOU = 0.5

MODELS = {
    "baseline": ROOT / "yolov8s.pt",
    "transfer": ROOT / "models/runs/combined_yolov8s_25cm/weights/best.pt",
    "uk_adapt": ROOT / "models/runs/uk_adapt_yolov8s_25cm/weights/best.pt",
}
PROCESSED = ROOT / "data/uk_retail/processed"
OUT = ROOT / "outputs/stats/occupancy_thresholds.csv"


def threshold_for(weights: Path):
    """F1-optimal confidence for one model over the pooled val sites."""
    from ultralytics import YOLO
    model = YOLO(str(weights))            # load once, reuse across sites
    scored_all, n_truth_all = [], 0
    for site in VAL_SITES:
        site_dir = PROCESSED / site
        offs = load_offsets(site_dir)
        pred = predict_boxes_local(model, site_dir, CONF_FLOOR, IMGSZ, offs)
        truth = truth_boxes_local(site_dir, offs)
        scored, n_truth = match_scores(pred, truth, MATCH_IOU)
        scored_all += scored
        n_truth_all += n_truth
    tau, f1, p, r = f1_optimal_threshold(scored_all, n_truth_all)
    return tau, f1, p, r, n_truth_all


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, weights in MODELS.items():
        if not weights.exists():
            sys.exit(f"missing weights for {name}: {weights}")
        print(f"== {name}: sweeping F1 on {VAL_SITES} ==")
        tau, f1, p, r, n = threshold_for(weights)
        print(f"   tau={tau:.4f}  F1={f1:.3f}  P={p:.3f}  R={r:.3f}  (n_truth={n})")
        rows.append({"model": name, "weights": str(weights.relative_to(ROOT)),
                     "tau": round(tau, 4), "f1_val": round(f1, 4),
                     "precision_val": round(p, 4), "recall_val": round(r, 4),
                     "n_truth": n})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()

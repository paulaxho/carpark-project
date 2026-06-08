"""
Evaluate a trained YOLOv8 detector and report the full metric suite.

Reports, for the chosen split:
    Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, and AP-small SEPARATELY
(per SOD-YOLO, Li et al. 2024 -- a strong headline mAP must not hide weak
tiny-car performance).

Supports CROSS-DATASET evaluation: --model gives the weights, --data gives the
dataset to test on, so they can differ. This is the core comparison experiment:

    # within-dataset (sanity: should match the training val numbers)
    python scripts/evaluate.py --model models/runs/cowc_yolov8s_25cm/weights/best.pt \
        --data configs/yolo_cowc.yaml  --name cowc_on_cowc

    # cross-dataset (the generalisation tests)
    python scripts/evaluate.py --model models/runs/cowc_yolov8s_25cm/weights/best.pt \
        --data configs/yolo_vedai.yaml --name cowc_on_vedai
    python scripts/evaluate.py --model models/runs/vedai_yolov8s_25cm/weights/best.pt \
        --data configs/yolo_cowc.yaml  --name vedai_on_cowc
    python scripts/evaluate.py --model models/runs/vedai_yolov8s_25cm/weights/best.pt \
        --data configs/yolo_vedai.yaml --name vedai_on_vedai

Each run appends one row to outputs/stats/cross_dataset_eval.csv so the four
directions build a single comparison table.

Note on AP-small: Ultralytics computes COCO-style area-based AP (small/medium/
large) only when a COCO-format split is available; for a plain YOLO dataset the
per-size breakdown may be reported as -1 (not available). Where that happens the
script records it honestly as 'n/a' rather than inventing a number. In this
project nearly every car is "small" by the COCO 32x32 px definition, so the
overall mAP is already effectively an all-small score -- which is itself the
point worth stating in the writeup.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def f1_from(precision: float, recall: float) -> float:
    """Harmonic mean of P and R; 0 if both are 0."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_ap_small(metrics) -> str:
    """Pull COCO-style small-object AP if Ultralytics exposed it, else 'n/a'.

    Ultralytics stores area-based AP on the box object when available
    (metrics.box.maps gives per-class mAP@0.5:0.95; small/medium/large area
    splits live on the underlying eval only for COCO-format data). We probe a
    few known locations and fall back to 'n/a' rather than guessing.
    """
    box = getattr(metrics, "box", None)
    if box is None:
        return "n/a"
    # Some Ultralytics versions expose .ap_small / area results; probe defensively.
    for attr in ("ap_small", "maps_small", "ap50_small"):
        val = getattr(box, attr, None)
        if val is not None:
            try:
                fval = float(val)
                # Ultralytics uses -1 as the "not computed / not available" sentinel.
                if fval < 0:
                    return "n/a"
                return f"{fval:.4f}"
            except (TypeError, ValueError):
                pass
    return "n/a"


def main(args) -> None:
    model_path = resolve(args.model)
    data_path = resolve(args.data)
    if not model_path.exists():
        raise SystemExit(f"weights not found: {model_path}")
    if not data_path.exists():
        raise SystemExit(f"data config not found: {data_path}")

    print(f"[eval] model = {model_path}")
    print(f"[eval] data  = {data_path}  (split={args.split})")
    print(f"[eval] device = {args.device}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        device=args.device,
        iou=args.iou,
        conf=args.conf,
        plots=True,
        project=str(PROJECT_ROOT / "outputs" / "eval_runs"),
        name=args.name,
        exist_ok=True,
    )

    # mean_results() returns [precision, recall, mAP50, mAP50-95]
    p, r, map50, map5095 = metrics.box.mean_results()
    f1 = f1_from(p, r)
    ap_small = extract_ap_small(metrics)

    print("\n" + "=" * 56)
    print(f"  RESULTS: {args.name}")
    print("=" * 56)
    print(f"  Precision      : {p:.4f}")
    print(f"  Recall         : {r:.4f}")
    print(f"  F1             : {f1:.4f}")
    print(f"  mAP@0.5        : {map50:.4f}")
    print(f"  mAP@0.5:0.95   : {map5095:.4f}")
    print(f"  AP-small       : {ap_small}")
    print("=" * 56)

    # Append one row to the shared cross-dataset CSV.
    stats_dir = PROJECT_ROOT / "outputs" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    csv_path = stats_dir / "cross_dataset_eval.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "name", "model", "data", "split",
                        "conf", "iou",
                        "precision", "recall", "f1", "mAP50", "mAP50_95", "ap_small"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), args.name,
                    model_path.name, data_path.name, args.split,
                    args.conf, args.iou,
                    f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}",
                    f"{map50:.4f}", f"{map5095:.4f}", ap_small])

    print(f"[eval] appended row to {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"[eval] plots in outputs/eval_runs/{args.name}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a YOLOv8 detector (supports cross-dataset).")
    ap.add_argument("--model", required=True,
                    help="path to trained weights, e.g. models/runs/cowc_yolov8s_25cm/weights/best.pt")
    ap.add_argument("--data", required=True,
                    help="YOLO data config to test ON, e.g. configs/yolo_vedai.yaml")
    ap.add_argument("--split", default="val",
                    help="which split to evaluate: val (default) or test")
    ap.add_argument("--name", default="eval",
                    help="run name; also the row label in the cross-dataset CSV")
    ap.add_argument("--device", default="0", help='"0" for CUDA, "cpu", or "mps"')
    ap.add_argument("--conf", type=float, default=0.001,
                    help="confidence threshold for metric computation (0.001 = standard mAP eval)")
    ap.add_argument("--iou", type=float, default=0.6,
                    help="NMS IoU threshold during evaluation")
    main(ap.parse_args())
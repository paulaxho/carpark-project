"""
Fine-tune YOLOv8 on the COWC car-detection dataset (transfer learning).

Starts from pretrained weights (default yolov8s.pt) and fine-tunes on the
640 px tiles built by build_cowc.py. This is the inductive, homogeneous
transfer-learning route the methodology commits to -- NOT training from scratch.

Defaults are chosen to match the resolution-study precedent (Gliaubiciute et al.,
2023): yolov8s, imgsz=640, ~100 epochs. Model size is a flag so a larger model
(yolov8m.pt) can be trialled later without code changes.

Usage:
    # baseline run on Apple Silicon (MPS)
    python scripts/train.py

    # try a bigger model / more epochs / different image size
    python scripts/train.py --model yolov8m.pt --epochs 150 --imgsz 768

    # quick sanity run (few epochs) to confirm the data loads and trains
    python scripts/train.py --epochs 3 --name smoke_test

After training, weights land in models/runs/<name>/weights/best.pt and the
val curves / confusion matrix / sample predictions in models/runs/<name>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(args) -> None:
    data_yaml = (Path(args.data) if Path(args.data).is_absolute()
                 else PROJECT_ROOT / args.data)
    if not data_yaml.exists():
        raise SystemExit(f"data config not found: {data_yaml}")

    print(f"[train] model={args.model}  data={data_yaml}")
    print(f"[train] imgsz={args.imgsz}  epochs={args.epochs}  batch={args.batch}"
          f"  device={args.device}")

    model = YOLO(args.model)            # pretrained weights -> fine-tune

    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,              # key lever for tiny cars: raise if recall is low
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,           # "mps" on Apple Silicon, "cpu", or 0 for CUDA
        patience=args.patience,       # early-stop if val mAP plateaus
        project=str(PROJECT_ROOT / "models" / "runs"),
        name=args.name,
        exist_ok=True,
        # single class -> no class imbalance across categories to weight
        seed=args.seed,
        plots=True,                   # write PR/loss curves + confusion matrix
    )

    print("\n[train] done.")
    print(f"[train] best weights: models/runs/{args.name}/weights/best.pt")
    print("[train] next: evaluate on the held-out val scene (scripts/evaluate.py),")
    print("        reporting P/R/F1, mAP@0.5, mAP@0.5:0.95 and AP-small separately.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8 on COWC.")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="pretrained weights: yolov8n/s/m/l/x.pt (default s)")
    ap.add_argument("--data", default="configs/yolo_cowc.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16,
                    help="lower to 8/4 if MPS runs out of memory")
    ap.add_argument("--device", default="mps", help='"mps", "cpu", or 0 for CUDA')
    ap.add_argument("--patience", type=int, default=20,
                    help="early-stop patience on val mAP")
    ap.add_argument("--name", default="cowc_yolov8s_25cm")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
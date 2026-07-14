#!/usr/bin/env python3
"""
Run the three UK retail experiments, scored on the held-out 20 uk_test sites.

  baseline   : stock YOLOv8 (yolov8s.pt)            -> UK test
  transfer   : combined COWC+VEDAI model            -> UK test
  uk_adapt   : transfer model fine-tuned on the 8   -> UK test
               calibration train sites, monitored on
               the 2 named calibration val sites

This is a THIN orchestrator. It does not reimplement evaluation: each scoring
step shells out to scripts/evaluate.py (the existing cross-dataset evaluator),
so every UK result lands in outputs/stats/cross_dataset_eval.csv in the same
format and with the same honest AP-small handling as the COWC/VEDAI runs.
Fine-tuning shells out to scripts/train.py if present, else calls Ultralytics
directly with matched hyperparameters.

Metric convention follows evaluate.py: conf=0.001, iou=0.6 -> standard mAP.
(A separate F1-operating-point sweep is intentionally NOT folded in here; mAP is
threshold-free and that is what the results table reports.)

Outputs:
  outputs/eval_runs/<name>/             per-run val artefacts + plots
  outputs/stats/cross_dataset_eval.csv  one appended row per eval (shared table)

Usage
-----
    python scripts/run_uk_experiments.py check
    python scripts/run_uk_experiments.py baseline
    python scripts/run_uk_experiments.py transfer
    python scripts/run_uk_experiments.py uk_adapt
    python scripts/run_uk_experiments.py all

Hardware target: NVIDIA RTX 4070 Ti SUPER (16 GB), CUDA.
"""

from pathlib import Path
import argparse, subprocess, sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS      = PROJECT_ROOT / "scripts"

UK_CONFIG    = PROJECT_ROOT / "configs/yolo_uk.yaml"
YOLO_UK      = PROJECT_ROOT / "data/uk_retail/yolo_uk"

STOCK_WEIGHTS    = PROJECT_ROOT / "yolov8s.pt"
COMBINED_WEIGHTS = PROJECT_ROOT / "models/runs/combined_yolov8s_25cm/weights/best.pt"
ADAPT_RUN_DIR    = PROJECT_ROOT / "models/runs/uk_adapt_yolov8s_25cm"
ADAPT_WEIGHTS    = ADAPT_RUN_DIR / "weights/best.pt"

# fine-tune hyperparams (640px tiles, 16 GB RTX 4070 Ti SUPER)
IMGSZ, BATCH, EPOCHS, PATIENCE, DEVICE, SEED = 640, 16, 100, 20, "0", 42


def sh(cmd):
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def evaluate(model: Path, name: str, single_cls: bool = False):
    """Score `model` on the held-out UK test split via the existing evaluator.

    single_cls: class-agnostic scoring. Needed only for the stock-COCO baseline,
    whose class 0 is 'person'; without it the baseline is credited only for its
    'person' predictions, not its real car (COCO class 2) detections. It is a
    no-op for the single-class transfer/uk_adapt models (1 class -> 1 class).
    """
    cmd = [sys.executable, SCRIPTS / "evaluate.py",
           "--model", model, "--data", UK_CONFIG, "--split", "test",
           "--name", name, "--device", DEVICE]
    if single_cls:
        cmd.append("--single-cls")
    sh(cmd)


def check():
    print(f"config           : {UK_CONFIG}  exists={UK_CONFIG.exists()}")
    print(f"test tiles        : {len(list((YOLO_UK/'images/test').glob('*.png')))}")
    print(f"train tiles       : {len(list((YOLO_UK/'images/train').glob('*.png')))}")
    print(f"val tiles         : {len(list((YOLO_UK/'images/val').glob('*.png')))}")
    print(f"stock weights     : {STOCK_WEIGHTS}  exists={STOCK_WEIGHTS.exists()}")
    print(f"combined weights  : {COMBINED_WEIGHTS}  exists={COMBINED_WEIGHTS.exists()}")
    if not (YOLO_UK / "images/test").exists():
        sys.exit("Dataset not assembled - run scripts/assemble_uk_dataset.py first.")
    print("OK.")


def baseline():
    if not STOCK_WEIGHTS.exists():
        sys.exit(f"missing {STOCK_WEIGHTS}")
    print("== baseline (stock YOLOv8, class-agnostic) -> UK test ==")
    # single_cls=True: score any detection on a real car, regardless of the COCO
    # label the stock model assigns. Fair, artifact-free floor; consistent with
    # the single-class transfer/uk_adapt evaluations.
    evaluate(STOCK_WEIGHTS, "baseline_on_uk_test_singlecls", single_cls=True)


def transfer():
    if not COMBINED_WEIGHTS.exists():
        sys.exit(f"missing combined weights: {COMBINED_WEIGHTS}")
    print("== transfer (combined COWC+VEDAI) -> UK test ==")
    evaluate(COMBINED_WEIGHTS, "transfer_on_uk_test")


def _finetune():
    """Fine-tune the transfer model on calibration train, monitor on calibration val."""
    uk_data = YOLO_UK / "uk_data.yaml"   # train=images/train, val=images/val
    train_py = SCRIPTS / "train.py"
    if train_py.exists():
        # Prefer your own training entrypoint for consistency.
        sh([sys.executable, train_py,
            "--weights", COMBINED_WEIGHTS, "--data", uk_data,
            "--imgsz", IMGSZ, "--batch", BATCH, "--epochs", EPOCHS,
            "--name", "uk_adapt_yolov8s_25cm", "--device", DEVICE])
    else:
        from ultralytics import YOLO
        YOLO(str(COMBINED_WEIGHTS)).train(
            data=str(uk_data), imgsz=IMGSZ, batch=BATCH, epochs=EPOCHS,
            patience=PATIENCE, device=DEVICE, seed=SEED,
            project=str(PROJECT_ROOT / "models/runs"),
            name="uk_adapt_yolov8s_25cm", exist_ok=True)


def uk_adapt():
    if not COMBINED_WEIGHTS.exists():
        sys.exit(f"missing combined weights: {COMBINED_WEIGHTS}")
    print("== uk_adapt: fine-tune transfer model on calibration ==")
    _finetune()
    if not ADAPT_WEIGHTS.exists():
        sys.exit(f"fine-tune finished but weights missing: {ADAPT_WEIGHTS}\n"
                 f"(if train.py uses a different run dir, point evaluate.py at it)")
    print("== uk_adapt -> UK test ==")
    evaluate(ADAPT_WEIGHTS, "uk_adapt_on_uk_test")


def all_():
    baseline(); transfer(); uk_adapt()
    print("\nAll three done. Results appended to outputs/stats/cross_dataset_eval.csv")


CMDS = {"check": check, "baseline": baseline, "transfer": transfer,
        "uk_adapt": uk_adapt, "all": all_}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=CMDS)
    CMDS[ap.parse_args().cmd]()
#!/usr/bin/env python3
"""
visualise_predictions.py — full-site qualitative detection results on UK test sites.

For each site it reconstructs the whole car park from its 640x640 tiles, runs a
model at a fixed confidence, de-duplicates detections across tile seams, matches
them to the ground truth at IoU 0.5, and draws the result colour-coded:

    GREEN  = correct detection (true positive)
    RED    = false alarm       (false positive)
    AMBER  = missed vehicle    (false negative, ground-truth box the model missed)

A caption reports predicted vs true counts and P/R/F1 for that site. This gives
supervisor-ready pictures rather than just aggregate metrics.

Defaults: the uk_adapt (fine-tuned) model at its F1-optimal operating threshold,
read from outputs/stats/occupancy_thresholds.csv (falls back to 0.477).

Usage (from project root):
    python scripts/visualise_predictions.py                 # all 20 test sites
    python scripts/visualise_predictions.py UK002 UK023     # just these
    python scripts/visualise_predictions.py --model transfer --sites UK013
    python scripts/visualise_predictions.py --clean         # predictions only, no GT match

Output: outputs/figures/predictions/<SITE>_<model>.png
"""
import sys, csv, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.counting import (load_offsets, predict_boxes_global, truth_boxes_global,
                          dedup_keep, iou)

PROCESSED = ROOT / "data/uk_retail/processed"
REGISTRY = ROOT / "data/uk_retail/geolytix/processed/uk_site_registry_final.csv"
THRESHOLDS = ROOT / "outputs/stats/occupancy_thresholds.csv"
OUTDIR = ROOT / "outputs/figures/predictions"

WEIGHTS = {
    "baseline": ROOT / "yolov8s.pt",
    "transfer": ROOT / "models/runs/combined_yolov8s_25cm/weights/best.pt",
    "uk_adapt": ROOT / "models/runs/uk_adapt_yolov8s_25cm/weights/best.pt",
}
MERGE_IOU = 0.5
MATCH_IOU = 0.5
IMGSZ = 640

GREEN = (60, 200, 90)     # true positive
RED   = (220, 60, 50)     # false positive
AMBER = (240, 175, 40)    # false negative (missed)


# --------------------------------------------------------------------------- #
#  Pure logic (testable without imagery / PIL)
# --------------------------------------------------------------------------- #
def match_boxes(preds, truths, iou_thr=MATCH_IOU):
    """Greedy IoU matching of predicted to truth boxes (both are xyxy lists).

    Returns (tp_idx, fp_idx, fn_idx): indices into preds that are true positives,
    indices into preds that are false positives, and indices into truths that
    were never matched (misses). Each truth is matched to at most one prediction.
    """
    used_truth = [False] * len(truths)
    tp_idx, fp_idx = [], []
    for pi, pb in enumerate(preds):
        best_j, best = -1, iou_thr
        for tj, tb in enumerate(truths):
            if used_truth[tj]:
                continue
            v = iou(pb, tb)
            if v >= best:
                best, best_j = v, tj
        if best_j >= 0:
            used_truth[best_j] = True
            tp_idx.append(pi)
        else:
            fp_idx.append(pi)
    fn_idx = [j for j, u in enumerate(used_truth) if not u]
    return tp_idx, fp_idx, fn_idx


def load_test_sites():
    out = []
    for r in csv.DictReader(open(REGISTRY)):
        if r["accepted"] == "Yes" and r["proposed_split"] == "uk_test":
            out.append(r["site_id"])
    return out


def load_threshold(model):
    if THRESHOLDS.exists():
        for r in csv.DictReader(open(THRESHOLDS)):
            if r["model"] == model:
                return float(r["tau"])
    return {"baseline": 0.004, "transfer": 0.012, "uk_adapt": 0.477}[model]


# --------------------------------------------------------------------------- #
#  Rendering (PIL imported lazily)
# --------------------------------------------------------------------------- #
def build_canvas(site_dir, offs):
    from PIL import Image
    W = max(co + w for co, ro, w, h in offs.values())
    H = max(ro + h for co, ro, w, h in offs.values())
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    tiles = Path(site_dir) / "tiles"
    for stem, (co, ro, w, h) in offs.items():
        p = tiles / f"{stem}.png"
        if p.exists():
            canvas.paste(Image.open(p).convert("RGB"), (co, ro))
    return canvas


def _font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_site(site, model_name, site_dir, offs, preds, truths, clean=False):
    from PIL import ImageDraw
    canvas = build_canvas(site_dir, offs)
    draw = ImageDraw.Draw(canvas)
    lw = max(2, round(min(canvas.size) / 400))

    if clean:
        for b in preds:
            draw.rectangle(b, outline=GREEN, width=lw)
        tp = fp = fn = None
    else:
        tp_idx, fp_idx, fn_idx = match_boxes(preds, truths)
        for j in fn_idx:                                   # draw misses first (under)
            draw.rectangle(truths[j], outline=AMBER, width=lw)
        for pi in tp_idx:
            draw.rectangle(preds[pi], outline=GREEN, width=lw)
        for pi in fp_idx:
            draw.rectangle(preds[pi], outline=RED, width=lw)
        tp, fp, fn = len(tp_idx), len(fp_idx), len(fn_idx)

    _caption(canvas, draw, site, model_name, len(preds), len(truths), tp, fp, fn)
    return canvas


def _caption(canvas, draw, site, model_name, n_pred, n_truth, tp, fp, fn):
    W, _ = canvas.size
    fs = max(16, round(W / 55))
    font = _font(fs)
    if tp is None:
        line = f"{site} ({model_name}):  predicted {n_pred}"
    else:
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / n_truth if n_truth else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        line = (f"{site} ({model_name}):  pred {n_pred}  vs  truth {n_truth}   |   "
                f"correct {tp}  false {fp}  missed {fn}   |   "
                f"P {prec:.2f}  R {rec:.2f}  F1 {f1:.2f}")
    pad = round(fs * 0.4)
    tb = draw.textbbox((0, 0), line, font=font)
    bar_h = (tb[3] - tb[1]) + 2 * pad
    draw.rectangle([0, 0, W, bar_h], fill=(20, 20, 20))
    draw.text((pad, pad - tb[1]), line, fill=(240, 240, 240), font=font)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sites", nargs="*", help="site IDs (default: all uk_test)")
    ap.add_argument("--model", default="uk_adapt", choices=list(WEIGHTS))
    ap.add_argument("--conf", type=float, default=None, help="override confidence")
    ap.add_argument("--clean", action="store_true", help="predictions only, no GT match")
    args = ap.parse_args()

    from ultralytics import YOLO
    OUTDIR.mkdir(parents=True, exist_ok=True)

    sites = args.sites or load_test_sites()
    conf = args.conf if args.conf is not None else load_threshold(args.model)
    model = YOLO(str(WEIGHTS[args.model]))
    print(f"model={args.model}  conf={conf:.3f}  sites={len(sites)}")

    for site in sites:
        site_dir = PROCESSED / site
        if not (site_dir / "tile_index.csv").exists():
            print(f"  ! {site}: no tile_index.csv, skipped"); continue
        offs = load_offsets(site_dir)
        preds = [b for b, _ in dedup_keep(
            [(b, t) for b, t, _ in predict_boxes_global(model, site_dir, conf, IMGSZ, offs)],
            MERGE_IOU)]
        truths = [b for b, _ in dedup_keep(truth_boxes_global(site_dir, offs), MERGE_IOU)]
        img = draw_site(site, args.model, site_dir, offs, preds, truths, clean=args.clean)
        out = OUTDIR / f"{site}_{args.model}.png"
        img.save(out)
        print(f"  {site}: pred {len(preds)}  truth {len(truths)}  -> {out.name}")

    print(f"\nWritten to {OUTDIR}")


if __name__ == "__main__":
    main()

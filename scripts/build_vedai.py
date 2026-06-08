"""
Build the VEDAI car-detection dataset in YOLO format, resolution-matched to 25 cm.

Config-driven, mirroring build_cowc.py: all parameters live in configs/vedai.yaml
(raw paths, resolution numbers, the car-like class filter, split fraction). This
script reads that config and produces the processed dataset.

VEDAI differs from COWC in three ways this handles:

  1. Multi-class -> car-like only. VEDAI labels many object types; only genuine
     passenger vehicles (config: car_like_class_ids, default {1,3,10,12,13} =
     car, pickup, van, small car, large car) are kept and collapsed to class
     0 = vehicle. Boats, planes, tractors, buses, motorcycles, trucks and the
     'other' class are dropped.

  2. Resolution. The VEDAI 512 subset is 12.5 cm/pixel. To match the project's
     25 cm pipeline it is downsampled by 0.5 to 25 cm. Because YOLO boxes are
     NORMALISED, the boxes are converted at native resolution and are unchanged
     by the downsample - only the image pixels are resampled. 512 px -> 256 px.

  3. No tiling. Each VEDAI image is already small (256 px after downsample,
     < one 640 tile), so images are written directly as single training samples.
     YOLO letterboxes them to the network input size.

The split is at the IMAGE level: VEDAI frames are independent locations (sparse,
non-consecutive ids), so there is no tile-overlap leakage risk and frames can be
split directly. A fixed seed makes the split reproducible.

Usage:
    python scripts/build_vedai.py --config configs/vedai.yaml --dry-run
    python scripts/build_vedai.py --config configs/vedai.yaml --limit 20
    python scripts/build_vedai.py --config configs/vedai.yaml --clean
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.vedai_annotations import (
    parse_vedai_annotation_file,
    records_to_yolo_boxes,
    save_yolo_labels,
)
from src.resample import resample_image, resolution_scale


def resolve_path(p):
    """Make a config path absolute against the project root if it is relative."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def discover_pairs(img_dir: Path, ann_dir: Path):
    """Return sorted (image_id, image_path, annotation_path) for usable colour pairs.

    Takes the intersection of colour images and numeric-stem annotations, so any
    orphan annotation (no matching image) or non-numeric extra is ignored.
    """
    colour_images = {p.stem.replace("_co", ""): p for p in img_dir.glob("*_co.png")}
    annotations = {p.stem: p for p in ann_dir.glob("*.txt") if p.stem.isdigit()}
    usable_ids = sorted(set(colour_images) & set(annotations))
    return [(i, colour_images[i], annotations[i]) for i in usable_ids]


def assign_splits(image_ids, val_fraction, seed):
    """Reproducibly assign each image id to 'train' or 'val'."""
    ids = list(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    val_set = set(ids[:n_val])
    return {i: ("val" if i in val_set else "train") for i in ids}


def main(args) -> None:
    cfg = yaml.safe_load(open(resolve_path(args.config)))

    img_dir = resolve_path(cfg["raw_images_dir"])
    ann_dir = resolve_path(cfg["raw_annotations_dir"])
    out_root = resolve_path(cfg["processed_dir"])
    meta_dir = resolve_path(cfg["metadata_dir"])

    source_cm = cfg["source_resolution_cm"]
    target_cm = cfg["target_resolution_cm"]
    allowed = set(cfg["car_like_class_ids"])
    val_fraction = cfg["val_fraction"]
    split_seed = cfg["split_seed"]

    if not img_dir.exists() or not ann_dir.exists():
        raise SystemExit(f"VEDAI raw data not found under {img_dir.parent}")

    if args.clean and out_root.exists() and not args.dry_run:
        shutil.rmtree(out_root)
        print(f"[build] cleared {out_root}")

    scale = resolution_scale(source_cm, target_cm)
    pairs = discover_pairs(img_dir, ann_dir)
    if args.limit:
        pairs = pairs[: args.limit]

    splits = assign_splits([i for i, _, _ in pairs], val_fraction, split_seed)
    n_val = sum(1 for v in splits.values() if v == "val")

    print(f"[build] {len(pairs)} usable image/annotation pairs")
    print(f"[build] resolution {source_cm}cm -> {target_cm}cm (scale {scale})")
    print(f"[build] keeping car-like classes {sorted(allowed)} -> class 0")
    print(f"[build] split: {len(pairs) - n_val} train / {n_val} val (seed {split_seed})")
    if args.dry_run:
        print("[build] dry-run: nothing written")
        return

    stats_rows = []
    totals = {"kept": 0, "dropped": 0, "images": 0, "empty": 0}

    for image_id, img_path, ann_path in pairs:
        split = splits[image_id]
        image = Image.open(img_path).convert("RGB")
        native_w, native_h = image.size  # 512 x 512

        # Convert boxes at NATIVE resolution. Normalised boxes are unchanged by
        # the later downsample, so this needs no rescaling.
        records = parse_vedai_annotation_file(ann_path)
        n_all = len(records)
        boxes = records_to_yolo_boxes(
            records, native_w, native_h,
            class_id=0, car_like_only=True, allowed=allowed,
        )
        n_kept = len(boxes)

        # Resample the IMAGE pixels to 25 cm (512 -> 256).
        resampled = resample_image(image, scale)

        out_img = out_root / "images" / split / f"{image_id}.png"
        out_lbl = out_root / "labels" / split / f"{image_id}.txt"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        resampled.save(out_img)
        save_yolo_labels(boxes, out_lbl)  # empty file if no car-like vehicles

        totals["kept"] += n_kept
        totals["dropped"] += (n_all - n_kept)
        totals["images"] += 1
        if n_kept == 0:
            totals["empty"] += 1
        stats_rows.append({
            "image_id": image_id, "split": split,
            "vehicles_all": n_all, "vehicles_kept": n_kept,
            "vehicles_dropped": n_all - n_kept,
            "out_size_px": resampled.size[0],
        })

    meta_dir.mkdir(parents=True, exist_ok=True)
    stats_path = meta_dir / "vedai_build_stats.csv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    print(f"\n[build] done. images written: {totals['images']}")
    print(f"[build] car-like boxes kept: {totals['kept']}  dropped (non-car): {totals['dropped']}")
    print(f"[build] images with no car-like vehicle (kept as negatives): {totals['empty']}")
    print(f"[build] per-image stats: {stats_path.relative_to(PROJECT_ROOT)}")
    print(f"[build] output: {out_root.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the VEDAI car dataset (25 cm, YOLO).")
    ap.add_argument("--config", default="configs/vedai.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N images")
    ap.add_argument("--clean", action="store_true", help="clear previous output first")
    main(ap.parse_args())
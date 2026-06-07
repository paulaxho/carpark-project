"""
Build the COWC YOLO dataset end-to-end, driven entirely by configs/cowc.yaml.

For every scene it runs the pipeline in order:
    load -> resample (15->25 cm) -> extract car + negative points -> scale
         -> assign train/val/test split -> tile into that split's folders
and writes a per-scene stats CSV to the metadata folder.

Usage:
    # validate the config + print the plan, do no work
    python scripts/build_cowc.py --config configs/cowc.yaml --dry-run

    # smoke test on ONE scene first (Stage-4 gate)
    python scripts/build_cowc.py --config configs/cowc.yaml --scene 03559

    # process the first N discovered scenes
    python scripts/build_cowc.py --config configs/cowc.yaml --limit 2

    # the real run over all scenes (optionally wiping previous output first)
    python scripts/build_cowc.py --config configs/cowc.yaml --clean
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.resample import resolution_scale, resample_image
from src.annotations import (
    extract_points_from_mask,
    find_car_mask,
    find_negative_mask,
    scale_points,
)
from src.splits import assign_split, split_dirs, summarise_split, validate_split_config
from src.tiling import tile_scene

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Scene discovery (mirrors notebook 01: anchor on the car-annotation mask)
# ---------------------------------------------------------------------------
def discover_scenes(scenes_root: Path) -> dict[str, dict]:
    """Return {scene_name: {region, scene_image, car_mask}} for every scene
    found under scenes_root, located via its *_Annotated_Cars.png mask."""
    found: dict[str, dict] = {}
    for car_mask in sorted(scenes_root.rglob("*_Annotated_Cars.png")):
        stem = car_mask.name[: -len("_Annotated_Cars.png")]
        candidates = [car_mask.with_name(stem + ext)
                      for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg")]
        scene_image = next((c for c in candidates if c.exists()), None)
        found[stem] = {
            "region": car_mask.parent.name,
            "scene_image": scene_image,
            "car_mask": car_mask,
        }
    return found


def resolve_path(p: str | Path) -> Path:
    """Make a config path absolute against the project root if it is relative."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def clean_output(processed_dir: Path) -> None:
    """Remove existing train/val/test image+label tiles so a rebuild starts clean."""
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = processed_dir / sub / split
            if d.exists():
                shutil.rmtree(d)
    print("[clean] cleared previous train/val/test tiles")


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------
def process_scene(scene_name: str, info: dict, cfg: dict, scale: float) -> dict:
    """Run the full pipeline for one scene and return its stats dict."""
    processed_dir = resolve_path(cfg["processed_dir"])

    if info["scene_image"] is None:
        return {"scene": scene_name, "region": info["region"],
                "split": "SKIPPED", "note": "scene image not found"}

    # load -> resample to target resolution
    image = Image.open(info["scene_image"]).convert("RGB")
    image = resample_image(image, scale)

    # car points -> scale to target resolution
    car_xs, car_ys = extract_points_from_mask(info["car_mask"])
    car_xs, car_ys = scale_points(car_xs, car_ys, scale)

    # negative points (optional, only if enabled and the mask exists)
    neg_xs = neg_ys = None
    if cfg.get("use_negatives", True):
        neg_mask = find_negative_mask(info["scene_image"])
        if neg_mask.exists():
            nx, ny = extract_points_from_mask(neg_mask)
            neg_xs, neg_ys = scale_points(nx, ny, scale)

    # choose split, then tile into that split's folders
    split = assign_split(scene_name, cfg["val_scenes"], cfg.get("test_scenes"))
    img_dir, lbl_dir = split_dirs(processed_dir, split)

    stats = tile_scene(
        image=image,
        car_xs=car_xs, car_ys=car_ys,
        neg_xs=neg_xs, neg_ys=neg_ys,
        scene_name=scene_name,
        out_img_dir=img_dir, out_lbl_dir=lbl_dir,
        tile_size=cfg["tile_size"],
        stride=cfg["stride"],
        box_px=cfg["box_px"],
        keep_empty_fraction=cfg["keep_empty_fraction"],
        skip_blank_std=cfg["skip_blank_std"],
        rng_seed=cfg.get("rng_seed", 0),
    )
    stats["region"] = info["region"]
    stats["split"] = split
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args) -> None:
    cfg = yaml.safe_load(open(resolve_path(args.config)))
    scale = resolution_scale(cfg["source_resolution_cm"], cfg["target_resolution_cm"])
    scenes_root = resolve_path(cfg["scenes_root"])
    processed_dir = resolve_path(cfg["processed_dir"])
    metadata_dir = resolve_path(cfg["metadata_dir"])
    exclude = set(cfg.get("exclude_scenes", []))

    print(f"[build_cowc] scenes_root: {scenes_root}")
    print(f"[build_cowc] scale (source/target): {scale:.3f}  box_px: {cfg['box_px']}")

    # 1) discover scenes on disk
    found = discover_scenes(scenes_root)
    if not found:
        sys.exit(f"No *_Annotated_Cars.png found under {scenes_root}")
    disk_scenes = sorted(found)

    # 2) cross-check against the inventory CSV (warn, don't fail)
    inv_path = resolve_path(cfg["inventory_csv"])
    if inv_path.exists():
        inv_scenes = set(pd.read_csv(inv_path)["scene"].astype(str))
        missing = inv_scenes - set(disk_scenes)
        extra = set(disk_scenes) - inv_scenes
        if missing:
            print(f"[warn] in inventory but not on disk: {sorted(missing)}")
        if extra:
            print(f"[warn] on disk but not in inventory: {sorted(extra)}")

    # 3) validate the split config up front (fail fast on typos / overlap / no val)
    summary = validate_split_config(disk_scenes, cfg["val_scenes"], cfg.get("test_scenes"))
    print("\n[plan] split (whole scenes):")
    for split_name in ("train", "val", "test"):
        names = summary[split_name]
        print(f"   {split_name:<5} {len(names):>3} scenes"
              + (f"  e.g. {names[:3]}" if names else ""))
    if exclude:
        print(f"   excluded: {sorted(exclude)}")

    # 4) decide which scenes to actually process
    if args.scene:
        if args.scene not in found:
            sys.exit(f"--scene {args.scene!r} not found. Available e.g.: {disk_scenes[:5]}")
        todo = [args.scene]
    else:
        todo = [s for s in disk_scenes if s not in exclude]
        if args.limit:
            todo = todo[: args.limit]

    if args.dry_run:
        print(f"\n[dry-run] would process {len(todo)} scene(s); no work done.")
        return

    if args.clean and not args.scene:
        clean_output(processed_dir)

    # 5) process
    print(f"\n[build_cowc] processing {len(todo)} scene(s)...")
    rows = []
    for i, scene_name in enumerate(todo, 1):
        split = assign_split(scene_name, cfg["val_scenes"], cfg.get("test_scenes"))
        print(f"  ({i}/{len(todo)}) {scene_name}  -> {split}")
        rows.append(process_scene(scene_name, found[scene_name], cfg, scale))

    # 6) write stats CSV
    df = pd.DataFrame(rows)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.scene}" if args.scene else ("_limited" if args.limit else "")
    out_csv = metadata_dir / f"cowc_build_stats{suffix}.csv"
    df.to_csv(out_csv, index=False)

    # 7) totals
    print("\n[build_cowc] done.")
    if "tiles_written" in df:
        for split_name in ("train", "val", "test"):
            sub = df[df.get("split") == split_name]
            if len(sub):
                print(f"   {split_name:<5} tiles_written={int(sub['tiles_written'].sum()):>6}"
                      f"  car_instances={int(sub['car_label_instances'].sum()):>7}"
                      f"  cars_in_scene={int(sub['cars_in_scene'].sum()):>7}")
    print(f"   stats -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the COWC YOLO dataset.")
    ap.add_argument("--config", default="configs/cowc.yaml")
    ap.add_argument("--scene", help="process only this scene (smoke test)")
    ap.add_argument("--limit", type=int, help="process only the first N scenes")
    ap.add_argument("--dry-run", action="store_true", help="validate + print plan only")
    ap.add_argument("--clean", action="store_true",
                    help="remove previous train/val/test tiles before a full build")
    main(ap.parse_args())
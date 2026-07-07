#!/usr/bin/env python3
"""
Assemble the UK retail YOLO dataset from per-site tiles.

Walks data/uk_retail/processed/UKxxx/tiles/, pairs each real tile PNG with its
.txt label, and routes it using the FROZEN registry split
(uk_site_registry_final.csv -> proposed_split). The split is read, never derived.

Two things are written:

  1. The calibration / test image+label tree (for all three experiments):
        data/uk_retail/yolo_uk/images/{calibration,test}/
        data/uk_retail/yolo_uk/labels/{calibration,test}/

  2. An internal train / val tree INSIDE calibration, for the uk_adapt fine-tune.
     The val sites are NAMED in configs/yolo_uk.yaml (calibration_val_sites) --
     explicit and deterministic, matching src/splits.py philosophy. Holdout is
     SITE-level so overlapping tiles never leak cars across train/val:
        data/uk_retail/yolo_uk/images/{train,val}/
        data/uk_retail/yolo_uk/labels/{train,val}/

Notes
-----
* *_check.png QC overlays are skipped.
* A tile with no matching .txt is skipped and logged (0-car / near-empty tiles -
  not errors).
* Files are COPIED by default. Set LINK=True to symlink and save disk.

Usage
-----
    python scripts/assemble_uk_dataset.py
"""

from pathlib import Path
import csv, shutil, sys, yaml

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/apple/Documents/carpark-project")
REGISTRY_CSV = PROJECT_ROOT / "data/uk_retail/geolytix/processed/uk_site_registry_final.csv"
UK_CONFIG    = PROJECT_ROOT / "configs/yolo_uk.yaml"

PROCESSED_DIR = PROJECT_ROOT / "data/uk_retail/processed"
OUT_DIR       = PROJECT_ROOT / "data/uk_retail/yolo_uk"

LINK = False
CLASS_NAMES = ["car"]

SPLIT_FOLDER = {
    "uk_train_calibration": "calibration",
    "uk_test": "test",
}
# ---------------------------------------------------------------------------


def load_split(registry_csv: Path) -> dict:
    """site_id -> 'calibration' | 'test' from the frozen registry. No derivation."""
    split = {}
    with open(registry_csv, newline="") as f:
        for row in csv.DictReader(f):
            sid = row["site_id"].strip()
            raw = row["proposed_split"].strip()
            if raw not in SPLIT_FOLDER:
                print(f"  ! {sid}: split '{raw}' not recognised - SKIPPED")
                continue
            split[sid] = SPLIT_FOLDER[raw]
    return split


def load_val_sites(config_path: Path) -> set:
    """Read the explicit named calibration val sites from configs/yolo_uk.yaml."""
    cfg = yaml.safe_load(config_path.read_text())
    sites = set(cfg.get("calibration_val_sites", []))
    if not sites:
        print("  ! no calibration_val_sites in config - uk_adapt will have no val")
    return sites


def place(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve()) if LINK else shutil.copy2(src, dst)


def iter_site_tiles(sid: str):
    """Yield (png, txt, n_boxes) for tiles of a site; txt is None for empty tiles."""
    tiles_dir = PROCESSED_DIR / sid / "tiles"
    if not tiles_dir.is_dir():
        return
    for png in sorted(tiles_dir.glob("*.png")):
        if png.stem.endswith("_check"):
            continue
        txt = png.with_suffix(".txt")
        if not txt.exists():
            yield (png, None, 0)
            continue
        n = sum(1 for ln in txt.read_text().splitlines() if ln.strip())
        yield (png, txt, n)


def main():
    if not REGISTRY_CSV.exists():
        sys.exit(f"Registry not found: {REGISTRY_CSV}")
    if not UK_CONFIG.exists():
        sys.exit(f"Config not found: {UK_CONFIG}")

    split = load_split(REGISTRY_CSV)
    val_sites = load_val_sites(UK_CONFIG)

    bad = [s for s in val_sites if split.get(s) != "calibration"]
    if bad:
        sys.exit(f"calibration_val_sites names non-calibration site(s): {bad}")

    print(f"Loaded split: {sum(v=='calibration' for v in split.values())} calibration / "
          f"{sum(v=='test' for v in split.values())} test")
    print(f"uk_adapt val sites (named): {sorted(val_sites)}\n")

    counts = {k: {"tiles": 0, "boxes": 0, "empty": 0}
              for k in ("calibration", "test", "train", "val")}

    for sid, fold in sorted(split.items()):
        st = sb = se = 0
        adapt_fold = (("val" if sid in val_sites else "train")
                      if fold == "calibration" else None)
        for png, txt, n in iter_site_tiles(sid):
            if txt is None:
                se += 1; counts[fold]["empty"] += 1
                if adapt_fold:
                    counts[adapt_fold]["empty"] += 1
                continue
            place(png, OUT_DIR / "images" / fold / png.name)
            place(txt, OUT_DIR / "labels" / fold / txt.name)
            counts[fold]["tiles"] += 1; counts[fold]["boxes"] += n
            st += 1; sb += n
            if adapt_fold:
                place(png, OUT_DIR / "images" / adapt_fold / png.name)
                place(txt, OUT_DIR / "labels" / adapt_fold / txt.name)
                counts[adapt_fold]["tiles"] += 1; counts[adapt_fold]["boxes"] += n

        tag = f" -> {adapt_fold}" if adapt_fold else ""
        print(f"  {sid:6s} [{fold:11s}{tag:>9s}] tiles={st:2d} boxes={sb:4d} empty={se}")

    yaml_path = OUT_DIR / "uk_data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"# Assembled by assemble_uk_dataset.py from the frozen registry split.\n"
        f"# train/val below = the uk_adapt fine-tune split (named calibration val).\n"
        f"# Evaluate on the held-out test set via configs/yolo_uk.yaml split=test.\n"
        f"path: {OUT_DIR.resolve()}\n"
        f"train: images/train     # 8 calibration train sites (uk_adapt fine-tune)\n"
        f"val: images/val         # 2 named calibration val sites (monitoring)\n"
        f"test: images/test       # 20 held-out sites - evaluation only\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )

    print("\n--- summary ---")
    for fold in ("calibration", "test", "train", "val"):
        c = counts[fold]
        print(f"  {fold:11s}: {c['tiles']:3d} tiles, {c['boxes']:4d} boxes, "
              f"{c['empty']} empty skipped")
    print(f"\nuk_data.yaml -> {yaml_path}")
    print("Done. Sanity-check a few image/label pairs before training.")


if __name__ == "__main__":
    main()
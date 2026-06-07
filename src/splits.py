"""
Scene-level dataset splitting for COWC.

The split MUST be at the scene level, never the tile level. Because tiling uses
overlapping windows (stride < tile_size), neighbouring tiles from the same scene
share pixels -- often the same cars. If those tiles were split randomly, the
"validation" set would contain cars the model already saw in training, and val
mAP would be inflated. Holding out WHOLE scenes guarantees every validation tile
comes from a scene the model never trained on.

The split is fully explicit and deterministic: val_scenes / test_scenes are named
lists in configs/cowc.yaml, so the split is reproducible and documentable in the
methodology (no random seed to remember, no run-to-run drift).

For COWC, a two-way train/val split is enough -- val is only for monitoring the
fine-tuning curve. The genuine TEST set lives in UK retail imagery
(data/uk_retail/), the domain the detector is actually deployed to, so
test_scenes is optional and may be left empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


VALID_SPLITS = ("train", "val", "test")


def assign_split(
    scene_name: str,
    val_scenes: Iterable[str],
    test_scenes: Optional[Iterable[str]] = None,
) -> str:
    """Return 'train', 'val', or 'test' for a single scene.

    Everything not explicitly named as val/test is train.
    """
    val = set(val_scenes)
    test = set(test_scenes or [])
    if scene_name in test:
        return "test"
    if scene_name in val:
        return "val"
    return "train"


def split_dirs(processed_dir: Path, split: str) -> tuple[Path, Path]:
    """Return (images_dir, labels_dir) for a split, e.g.
    (processed/images/train, processed/labels/train). This is what tiling.py
    writes into, so the split decision is made once, here, per scene.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}.")
    processed_dir = Path(processed_dir)
    return (
        processed_dir / "images" / split,
        processed_dir / "labels" / split,
    )


def summarise_split(
    all_scenes: Iterable[str],
    val_scenes: Iterable[str],
    test_scenes: Optional[Iterable[str]] = None,
) -> dict[str, list[str]]:
    """Group every discovered scene into train/val/test for a planning log."""
    groups: dict[str, list[str]] = {s: [] for s in VALID_SPLITS}
    for scene in all_scenes:
        groups[assign_split(scene, val_scenes, test_scenes)].append(scene)
    return groups


def validate_split_config(
    all_scenes: Iterable[str],
    val_scenes: Iterable[str],
    test_scenes: Optional[Iterable[str]] = None,
    require_val: bool = True,
) -> dict[str, list[str]]:
    """Check the split config against the scenes actually found on disk.

    Catches the common, silent failures:
      * a scene named in BOTH val and test (a config mistake);
      * a val/test scene name that matches no real scene (a typo -- it would
        otherwise be ignored, leaving that split short and the scene in train);
      * an empty val set when one is required (no monitoring signal).

    Raises ValueError on any of these. Returns the split summary on success.
    """
    all_set = set(all_scenes)
    val = list(val_scenes)
    test = list(test_scenes or [])

    overlap = set(val) & set(test)
    if overlap:
        raise ValueError(
            f"Scene(s) listed in both val and test: {sorted(overlap)}."
        )

    unknown = [s for s in (val + test) if s not in all_set]
    if unknown:
        raise ValueError(
            "Split config names scenes not found on disk "
            f"(check for typos): {sorted(unknown)}."
        )

    summary = summarise_split(all_set, val, test)
    if require_val and not summary["val"]:
        raise ValueError(
            "No validation scenes resolved -- training would have no held-out "
            "signal to monitor. Add at least one scene to val_scenes."
        )
    return summary
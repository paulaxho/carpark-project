"""
Tiling utilities for COWC.

Slides a fixed window over a (already-downsampled, 25 cm) scene and writes
YOLO training tiles plus per-tile label files.

Design decisions wired in here (each settled in earlier work):

  * OVERLAP. stride < tile_size (e.g. 512 < 640) so a car near a tile edge
    still appears whole in the neighbouring tile. Combined with edge-flush
    tiles (below) this gives full coverage and removes the edge losses seen
    with no-overlap tiling.

  * TILE-LOCAL NORMALISATION. Boxes are normalised against the TILE size
    (640), never the scene size. Points are converted to tile-local
    coordinates first (filter_points_inside_tile), then boxed against the
    tile dimensions. This is the trap that produced 10x-too-small boxes.

  * EDGE SAFETY. YOLO coordinates are clipped to valid [0,1] ranges
    inside boxes.point_to_yolo_box.

  * NEGATIVES NEVER BECOME BOXES. Car points and hard-negative points are
    kept in separate arrays. Only car points are boxed. Negatives are used
    only to decide which car-free tiles are worth keeping as background.

  * EMPTY-TILE RETENTION. A tile with no cars but containing a hard negative
    (boat, A/C unit, trailer ...) is always kept with an EMPTY label file, so
    YOLO learns it as background and false positives drop. A controlled
    fraction of pure-background tiles is also kept (keep_empty_fraction),
    sized relative to the number of positive tiles.

  * NO-DATA SKIP. COWC scenes have uniform no-data padding (the grey border
    of a mosaic). Near-uniform tiles are skipped so the model is not trained
    on blank fill.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.boxes import (
    filter_points_inside_tile,
    point_to_yolo_box,
    save_yolo_labels,
)

Image.MAX_IMAGE_PIXELS = None


def _tile_origins(extent: int, tile_size: int, stride: int) -> list[int]:
    """Top-left positions along one axis, with a final edge-flush tile so the
    far edge is fully covered (not just multiples of the stride)."""
    if extent <= tile_size:
        return [0]
    origins = list(range(0, extent - tile_size + 1, stride))
    last = extent - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def _as_array(a: Optional[np.ndarray]) -> np.ndarray:
    return np.asarray([], dtype=int) if a is None else np.asarray(a)


def tile_scene(
    image: Image.Image,
    car_xs: np.ndarray,
    car_ys: np.ndarray,
    scene_name: str,
    out_img_dir: Path,
    out_lbl_dir: Path,
    neg_xs: Optional[np.ndarray] = None,
    neg_ys: Optional[np.ndarray] = None,
    tile_size: int = 640,
    stride: int = 512,
    box_px: int = 14,
    keep_empty_fraction: float = 0.12,
    skip_blank_std: float = 1.0,
    class_id: int = 0,
    rng_seed: int = 0,
) -> dict:
    """Tile one scene into 640x640 YOLO images + labels.

    Args:
        image: the scene at the TARGET resolution (25 cm), RGB.
        car_xs, car_ys: car centres, scaled to the target resolution.
        scene_name: used to prefix tile filenames.
        out_img_dir, out_lbl_dir: destination folders (e.g. images/train,
            labels/train) -- the split decision is made by the caller.
        neg_xs, neg_ys: hard-negative centres (optional), scaled to target res.
        tile_size, stride, box_px, keep_empty_fraction, skip_blank_std: see config.

    Returns:
        A stats dict for the per-scene QC log.
    """
    # --- input validation -------------------------------------------------
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive.")
    if box_px <= 0:
        raise ValueError("box_px must be positive.")
    if not 0 <= keep_empty_fraction <= 1:
        raise ValueError("keep_empty_fraction must be between 0 and 1.")
    if len(car_xs) != len(car_ys):
        raise ValueError("car_xs and car_ys must have the same length.")
    if (neg_xs is None) != (neg_ys is None) or (
        neg_xs is not None and len(neg_xs) != len(neg_ys)
    ):
        raise ValueError(
            "neg_xs and neg_ys must be provided together and match in length."
        )

    out_img_dir = Path(out_img_dir)
    out_lbl_dir = Path(out_lbl_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if image.mode != "RGB":
        image = image.convert("RGB")

    car_xs, car_ys = _as_array(car_xs), _as_array(car_ys)
    neg_xs, neg_ys = _as_array(neg_xs), _as_array(neg_ys)

    width, height = image.size
    rng = np.random.default_rng(rng_seed)

    stats = dict(
        scene=scene_name, tiles_scanned=0, blank_skipped=0,
        positive_tiles=0, hard_negative_tiles=0,
        pure_empty_available=0, pure_empty_kept=0,
        tiles_written=0, cars_in_scene=int(len(car_xs)),
        car_label_instances=0,
    )

    def _stem(left: int, top: int) -> str:
        return f"{scene_name}__x{left}_y{top}"

    def _write_tile(left: int, top: int, boxes: list) -> None:
        crop = image.crop((left, top, left + tile_size, top + tile_size))
        crop.save(out_img_dir / f"{_stem(left, top)}.png")
        save_yolo_labels(boxes, out_lbl_dir / f"{_stem(left, top)}.txt")
        stats["tiles_written"] += 1

    pure_empty: list[tuple[int, int]] = []

    for top in _tile_origins(height, tile_size, stride):
        for left in _tile_origins(width, tile_size, stride):
            stats["tiles_scanned"] += 1

            crop = image.crop((left, top, left + tile_size, top + tile_size))
            if np.asarray(crop).std() < skip_blank_std:
                stats["blank_skipped"] += 1
                continue

            lxs, lys = filter_points_inside_tile(car_xs, car_ys, left, top, tile_size)

            if len(lxs) > 0:
                boxes = [point_to_yolo_box(int(x), int(y), tile_size, tile_size,
                                           box_px, class_id) for x, y in zip(lxs, lys)]
                _write_tile(left, top, boxes)
                stats["positive_tiles"] += 1
                stats["car_label_instances"] += len(boxes)
            else:
                nlx, _ = filter_points_inside_tile(neg_xs, neg_ys, left, top, tile_size)
                if len(nlx) > 0:                       # hard-negative tile: always keep
                    _write_tile(left, top, [])
                    stats["hard_negative_tiles"] += 1
                else:                                  # pure background: defer, sample later
                    pure_empty.append((left, top))

    # Keep a controlled fraction of pure-background tiles, sized relative to
    # the number of positive tiles, sampled reproducibly.
    stats["pure_empty_available"] = len(pure_empty)
    n_keep = int(round(keep_empty_fraction * stats["positive_tiles"]))
    n_keep = min(n_keep, len(pure_empty))
    if n_keep > 0:
        idx = rng.choice(len(pure_empty), size=n_keep, replace=False)
        for i in sorted(idx.tolist()):
            left, top = pure_empty[i]
            _write_tile(left, top, [])
        stats["pure_empty_kept"] = n_keep

    return stats
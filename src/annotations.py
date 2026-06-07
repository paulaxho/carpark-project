"""
Annotation utilities for COWC.

COWC provides annotation masks rather than bounding boxes.

For each *_Annotated_Cars.png:
    non-zero alpha pixels = car centre points

For each *_Annotated_Negatives.png:
    non-zero alpha pixels = hard negative centre points

These centre points can then be scaled when the imagery is downsampled
from 15 cm/pixel to 25 cm/pixel.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def extract_points_from_mask(mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract centre-point coordinates from a COWC RGBA annotation mask.

    Returns:
        xs: x coordinates
        ys: y coordinates
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask = Image.open(mask_path).convert("RGBA")
    arr = np.array(mask)

    alpha = arr[:, :, 3]
    rgb_nonzero = np.any(arr[:, :, :3] > 0, axis=2)

    if np.all(alpha > 0):
        point_mask = rgb_nonzero
    else:
        point_mask = alpha > 0

    ys, xs = np.where(point_mask)

    return xs, ys


def find_car_mask(image_path: Path) -> Path:
    """
    Given an original COWC image path, return its matching car annotation mask.
    """
    return image_path.with_name(
        image_path.stem + "_Annotated_Cars.png"
    )


def find_negative_mask(image_path: Path) -> Path:
    """
    Given an original COWC image path, return its matching hard-negative mask.
    """
    return image_path.with_name(
        image_path.stem + "_Annotated_Negatives.png"
    )


def scale_points(
    xs: np.ndarray,
    ys: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale annotation coordinates by the same factor used to resample the image.
    """
    if scale <= 0:
        raise ValueError("Scale must be positive.")

    xs_scaled = np.round(xs * scale).astype(int)
    ys_scaled = np.round(ys * scale).astype(int)

    return xs_scaled, ys_scaled


def count_points(mask_path: Path) -> int:
    """
    Count annotated points in a COWC mask.
    """
    xs, _ = extract_points_from_mask(mask_path)
    return int(len(xs))
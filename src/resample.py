"""
Utilities for resolution-matching COWC imagery.

COWC is standardised to 15 cm/pixel.
Target UK Getmapping imagery is approximately 25 cm/pixel.

Scale factor:
    new_size = old_size * (15 / 25) = old_size * 0.6
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def resolution_scale(source_cm: float, target_cm: float) -> float:
    """Return scale factor for converting source resolution to target resolution."""
    if source_cm <= 0 or target_cm <= 0:
        raise ValueError("Resolution values must be positive.")
    return source_cm / target_cm


def resample_image(
    image: Image.Image,
    scale: float,
    resample_method: int = Image.Resampling.BILINEAR,
) -> Image.Image:
    """Resize image by scale factor."""
    if scale <= 0:
        raise ValueError("Scale must be positive.")

    old_width, old_height = image.size
    new_width = int(round(old_width * scale))
    new_height = int(round(old_height * scale))

    return image.resize((new_width, new_height), resample=resample_method)



def save_resampled_image(
    input_path: Path,
    output_path: Path,
    source_cm: float = 15.0,
    target_cm: float = 25.0,
) -> None:
    """Load an image, downsample it to target resolution, and save it."""
    scale = resolution_scale(source_cm, target_cm)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    resampled = resample_image(image, scale)

    resampled.save(output_path)

    print(f"Saved resampled image: {output_path}")
    print(f"Original size: {image.size}")
    print(f"New size: {resampled.size}")
    print(f"Scale factor: {scale:.3f}")
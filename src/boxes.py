"""
Bounding-box utilities.

COWC provides centre-point annotations, not bounding boxes.
This module converts centre points into fixed-size YOLO boxes.

YOLO format:
    class_id x_center y_center width height

All coordinates are normalised between 0 and 1, relative to the TILE (e.g. 640),
never the whole scene.
"""

from typing import List, Tuple

import numpy as np


def point_to_yolo_box(
    x: int,
    y: int,
    image_width: int,
    image_height: int,
    box_px: int,
    class_id: int = 0,
) -> List[float]:
    """Convert one centre point into a fixed-size YOLO box, clipped to the tile.

    image_width / image_height are the TILE dimensions (e.g. 640), not the scene.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    if box_px <= 0:
        raise ValueError("box_px must be positive.")

    half = box_px / 2.0

    # Pixel corners, clipped to the tile so no box spills past the edge.
    x1 = max(x - half, 0.0)
    y1 = max(y - half, 0.0)
    x2 = min(x + half, image_width)
    y2 = min(y + half, image_height)

    # Recompute centre/size from the clipped corners, then normalise.
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    w_norm = (x2 - x1) / image_width
    h_norm = (y2 - y1) / image_height

    return [class_id, x_center, y_center, w_norm, h_norm]


def points_to_yolo_boxes(
    xs: np.ndarray,
    ys: np.ndarray,
    image_width: int,
    image_height: int,
    box_px: int,
    class_id: int = 0,
) -> List[List[float]]:
    """Batch wrapper around point_to_yolo_box for many centre points."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length.")
    return [
        point_to_yolo_box(int(x), int(y), image_width, image_height, box_px, class_id)
        for x, y in zip(xs, ys)
    ]


def format_yolo_box(box: List[float], decimals: int = 6) -> str:
    """Convert one YOLO box list into a text line."""
    class_id, x_center, y_center, width, height = box
    return (
        f"{int(class_id)} "
        f"{x_center:.{decimals}f} "
        f"{y_center:.{decimals}f} "
        f"{width:.{decimals}f} "
        f"{height:.{decimals}f}"
    )


def save_yolo_labels(
    boxes: List[List[float]],
    output_path,
    decimals: int = 6,
) -> None:
    """Save YOLO labels to a .txt file (empty list -> empty file = background tile)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for box in boxes:
            f.write(format_yolo_box(box, decimals=decimals) + "\n")


def filter_points_inside_tile(
    xs: np.ndarray,
    ys: np.ndarray,
    left: int,
    top: int,
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only points inside a tile and convert them to tile-local coordinates."""
    right = left + tile_size
    bottom = top + tile_size

    mask = (xs >= left) & (xs < right) & (ys >= top) & (ys < bottom)

    local_xs = xs[mask] - left
    local_ys = ys[mask] - top

    return local_xs, local_ys
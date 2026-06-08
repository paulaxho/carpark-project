"""
VEDAI annotation utilities.

VEDAI 512 annotations contain one row per vehicle.

Observed format (14 whitespace-separated columns):
    x_center y_center orientation class_id <flag> <flag> x1 x2 x3 x4 y1 y2 y3 y4

Class scheme for THIS VEDAI download (confirmed against the dataset, NOT the
"standard" VEDAI scheme which differs):
    1  car          <- keep
    2  truck
    3  pickup       <- keep
    4  tractor
    5  camping
    6  boat
    7  motorcycle
    9  bus
    10 van           <- keep
    11 other         (junk/miscellaneous - DROP, ~955 instances)
    12 small car     <- keep
    13 large car     <- keep
    23 board/boat
    31 plane

For this car-detection project, only genuine passenger-vehicle classes are
kept and all are collapsed into a single project class:
    0 = vehicle

Everything not in CAR_LIKE_CLASS_IDS is dropped. Using an allowlist (rather
than a blocklist) means any unexpected / unmapped class id is dropped by
default rather than silently mislabelled as a car.
"""

from pathlib import Path
from typing import List, Set

import numpy as np

# Confirmed car-like classes for THIS VEDAI version: car, pickup, van, small car, large car.
CAR_LIKE_CLASS_IDS: Set[int] = {1, 3, 10, 12, 13}


def parse_vedai_annotation_file(annotation_path: Path) -> List[dict]:
    """
    Parse one VEDAI annotation .txt file.

    Returns a list of vehicle records containing:
        centre_x, centre_y, orientation, original_class_id, polygon_xs, polygon_ys
    """
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    records = []

    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 14:
                raise ValueError(
                    f"Unexpected VEDAI annotation format in {annotation_path}: "
                    f"expected 14 columns, got {len(parts)}"
                )

            centre_x = float(parts[0])
            centre_y = float(parts[1])
            orientation = float(parts[2])
            original_class_id = int(parts[3])

            polygon_xs = np.array([float(v) for v in parts[6:10]])
            polygon_ys = np.array([float(v) for v in parts[10:14]])

            records.append(
                {
                    "centre_x": centre_x,
                    "centre_y": centre_y,
                    "orientation": orientation,
                    "original_class_id": original_class_id,
                    "polygon_xs": polygon_xs,
                    "polygon_ys": polygon_ys,
                }
            )

    return records


def is_car_like(record: dict, allowed: Set[int] = CAR_LIKE_CLASS_IDS) -> bool:
    """True if the record's original VEDAI class is a kept car-like class."""
    return record["original_class_id"] in allowed


def filter_car_like(
    records: List[dict],
    allowed: Set[int] = CAR_LIKE_CLASS_IDS,
) -> List[dict]:
    """Keep only car-like records; drop boats, planes, tractors, 'other', etc."""
    return [r for r in records if is_car_like(r, allowed)]


def polygon_to_yolo_box(
    polygon_xs: np.ndarray,
    polygon_ys: np.ndarray,
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> List[float]:
    """
    Convert a VEDAI oriented polygon to an axis-aligned YOLO box.

    YOLO format:
        class_id x_center y_center width height   (all normalised to the image)
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")

    x1 = max(float(np.min(polygon_xs)), 0.0)
    y1 = max(float(np.min(polygon_ys)), 0.0)
    x2 = min(float(np.max(polygon_xs)), float(image_width))
    y2 = min(float(np.max(polygon_ys)), float(image_height))

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        raise ValueError("Invalid polygon produced zero-area box.")

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    w_norm = box_width / image_width
    h_norm = box_height / image_height

    return [class_id, x_center, y_center, w_norm, h_norm]


def records_to_yolo_boxes(
    records: List[dict],
    image_width: int,
    image_height: int,
    class_id: int = 0,
    car_like_only: bool = True,
    allowed: Set[int] = CAR_LIKE_CLASS_IDS,
) -> List[List[float]]:
    """
    Convert parsed VEDAI vehicle records to YOLO boxes.

    By default only car-like classes are kept (car_like_only=True) and all are
    collapsed to one project class (0 = vehicle). The set of kept class ids can
    be overridden via `allowed` (the build pipeline passes it from the config).
    Set car_like_only=False to convert every record regardless of class.
    """
    if car_like_only:
        records = filter_car_like(records, allowed)

    boxes = []
    for record in records:
        box = polygon_to_yolo_box(
            polygon_xs=record["polygon_xs"],
            polygon_ys=record["polygon_ys"],
            image_width=image_width,
            image_height=image_height,
            class_id=class_id,
        )
        boxes.append(box)

    return boxes


def format_yolo_box(box: List[float], decimals: int = 6) -> str:
    """Convert one YOLO box into a text line."""
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
    output_path: Path,
    decimals: int = 6,
) -> None:
    """Save YOLO labels to a .txt file (creates an empty file if no boxes)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for box in boxes:
            f.write(format_yolo_box(box, decimals=decimals) + "\n")
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.annotations import (
    extract_points_from_mask,
    find_car_mask,
    find_negative_mask,
    scale_points,
)

image_path = Path(
    "data/cowc/raw/cowc/datasets/ground_truth_sets/Vaihingen_ISPRS/"
    "TOP_Mosaic_09cm_scaled_15cm_Gray.png"
)

car_mask = find_car_mask(image_path)
neg_mask = find_negative_mask(image_path)

car_xs, car_ys = extract_points_from_mask(car_mask)
neg_xs, neg_ys = extract_points_from_mask(neg_mask)

scale = 0.6

car_xs_25, car_ys_25 = scale_points(car_xs, car_ys, scale)
neg_xs_25, neg_ys_25 = scale_points(neg_xs, neg_ys, scale)

print("Car mask:", car_mask)
print("Negative mask:", neg_mask)
print("Cars at 15cm:", len(car_xs))
print("Negatives at 15cm:", len(neg_xs))
print("First 5 car points at 15cm:")
print(list(zip(car_xs[:5], car_ys[:5])))
print("First 5 car points at 25cm:")
print(list(zip(car_xs_25[:5], car_ys_25[:5])))
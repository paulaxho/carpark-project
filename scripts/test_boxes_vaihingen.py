import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.annotations import extract_points_from_mask, find_car_mask, scale_points
from src.resample import resolution_scale, resample_image
from src.boxes import points_to_yolo_boxes, format_yolo_box

image_path = Path(
    "data/cowc/raw/cowc/datasets/ground_truth_sets/Vaihingen_ISPRS/"
    "TOP_Mosaic_09cm_scaled_15cm_Gray.png"
)

scale = resolution_scale(15.0, 25.0)

image = Image.open(image_path).convert("RGB")
image_25 = resample_image(image, scale)

width, height = image_25.size

car_mask = find_car_mask(image_path)
xs, ys = extract_points_from_mask(car_mask)
xs_25, ys_25 = scale_points(xs, ys, scale)

boxes = points_to_yolo_boxes(
    xs=xs_25[:5],
    ys=ys_25[:5],
    image_width=width,
    image_height=height,
    box_px=14,
    class_id=0,
)

print("25cm image size:", width, height)
print("First 5 YOLO boxes:")
for box in boxes:
    print(format_yolo_box(box))
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.vedai_annotations import (
    parse_vedai_annotation_file,
    records_to_yolo_boxes,
    format_yolo_box,
)

img_path = Path("data/vedai/raw/Vehicules512/00000000_co.png")
ann_path = Path("data/vedai/raw/Annotations512/00000000.txt")

image = Image.open(img_path).convert("RGB")
width, height = image.size

records = parse_vedai_annotation_file(ann_path)
boxes = records_to_yolo_boxes(records, width, height, class_id=0)

print("Image:", img_path)
print("Annotation:", ann_path)
print("Image size:", image.size)
print("Records:", len(records))
print("First parsed record:")
print(records[0])
print("\nYOLO boxes:")
for box in boxes:
    print(format_yolo_box(box))
import sys
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.vedai_annotations import (
    parse_vedai_annotation_file,
    records_to_yolo_boxes,
)

IMG_DIR = PROJECT_ROOT / "data/vedai/raw/Vehicules512"
ANN_DIR = PROJECT_ROOT / "data/vedai/raw/Annotations512"
OUT_DIR = PROJECT_ROOT / "outputs/figures/vedai_yolo_checks"

OUT_DIR.mkdir(parents=True, exist_ok=True)

sample_ids = ["00000000", "00000066", "00000100", "00000200", "00000500"]

for img_id in sample_ids:
    img_path = IMG_DIR / f"{img_id}_co.png"
    ann_path = ANN_DIR / f"{img_id}.txt"

    if not img_path.exists() or not ann_path.exists():
        print(f"Skipping missing pair: {img_id}")
        continue

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    records = parse_vedai_annotation_file(ann_path)
    boxes = records_to_yolo_boxes(records, w, h, class_id=0)

    # Draw original VEDAI polygons in red
    for record in records:
        polygon = list(zip(record["polygon_xs"], record["polygon_ys"]))
        draw.polygon(polygon, outline="red")

    # Draw converted YOLO boxes in yellow
    for box in boxes:
        _, xc, yc, bw, bh = box

        xc *= w
        yc *= h
        bw *= w
        bh *= h

        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2

        draw.rectangle((x1, y1, x2, y2), outline="yellow", width=2)

    out_path = OUT_DIR / f"{img_id}_vedai_yolo_check.png"
    img.save(out_path)
    print("Saved:", out_path)
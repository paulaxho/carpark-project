#!/usr/bin/env python3
"""
make_cvat_zip.py  —  Package a site's pre-label .txt into a CVAT YOLO 1.1 import zip.

Run from the project root:
    python scripts/make_cvat_zip.py UK001

Produces:  data/uk_retail/processed/<SITE>/<SITE>_cvat_prelabels.zip
Upload that in CVAT:  task -> Actions -> Upload annotations -> format "YOLO 1.1".

CVAT YOLO 1.1 expects this layout inside the zip:
    obj.names                      (one class name per line)
    obj.data                       (meta)
    obj_train_data/<image>.txt     (the YOLO labels, one per image)
    train.txt                      (list of image paths)
"""
import sys, zipfile
from pathlib import Path

ROOT = Path.cwd()
site = sys.argv[1] if len(sys.argv) > 1 else "UK001"
tiles = ROOT / "data/uk_retail/processed" / site / "tiles"
out_zip = ROOT / "data/uk_retail/processed" / site / f"{site}_cvat_prelabels.zip"

txts = sorted(tiles.glob("*.txt"))
if not txts:
    raise SystemExit(f"No .txt label files in {tiles}")

obj_names = "car\n"
obj_data = (
    "classes = 1\n"
    "train = data/train.txt\n"
    "names = data/obj.names\n"
    "backup = backup/\n"
)
# image list: CVAT matches labels to images by base filename
train_list = "\n".join(f"data/obj_train_data/{t.stem}.png" for t in txts) + "\n"

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("obj.names", obj_names)
    z.writestr("obj.data", obj_data)
    z.writestr("train.txt", train_list)
    for t in txts:
        z.writestr(f"obj_train_data/{t.name}", t.read_text())

print(f"Wrote {out_zip}  ({len(txts)} label file(s))")
print("In CVAT: task -> Actions -> Upload annotations -> 'YOLO 1.1' -> select this zip.")
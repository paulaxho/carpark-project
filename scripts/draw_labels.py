#!/usr/bin/env python3
"""
draw_labels.py — overlay YOLO .txt boxes on a tile image to eyeball the labels.

Usage (from project root):
    python scripts/draw_labels.py UK001

Looks for the tile + .txt in data/uk_retail/processed/<SITE>/tiles/,
writes <SITE>_check.png next to them, and prints the box count.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path.cwd()
site = sys.argv[1] if len(sys.argv) > 1 else "UK001"
tiles = ROOT / "data/uk_retail/processed" / site / "tiles"

pngs = sorted(tiles.glob("*.png"))
if not pngs:
    raise SystemExit(f"No tile .png in {tiles}")

total = 0
for png in pngs:
    txt = png.with_suffix(".txt")
    img = Image.open(png).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    n = 0
    if txt.exists():
        for line in txt.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            _, cx, cy, w, h = map(float, p)
            x0 = (cx - w/2) * W; y0 = (cy - h/2) * H
            x1 = (cx + w/2) * W; y1 = (cy + h/2) * H
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
            n += 1
    out = png.with_name(f"{png.stem}_check.png")
    img.save(out)
    total += n
    print(f"{png.name}: {n} boxes -> {out.name}")

print(f"TOTAL: {total} boxes drawn")
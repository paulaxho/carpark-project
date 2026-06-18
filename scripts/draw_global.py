#!/usr/bin/env python3
"""
draw_global.py — overlay ALL tiles' boxes onto the full site clip in global coords,
so seam duplicates are visible. Lets you eyeball the true unique car count.

Usage (from project root):
    python scripts/draw_global.py UK005
Writes <SITE>_global_check.png in the processed/<SITE>/ folder.
"""
import sys, csv
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path.cwd(); site=sys.argv[1] if len(sys.argv)>1 else "UK005"
base=ROOT/"data/uk_retail/processed"/site
clip=base/f"{site}_clip.tif"
index=base/"tile_index.csv"
tiles=base/"tiles"

img=Image.open(clip).convert("RGB"); Wc,Hc=img.size
draw=ImageDraw.Draw(img)
offs={}
with open(index) as f:
    for row in csv.DictReader(f):
        offs[row["tile"]]=(int(row["col_off"]),int(row["row_off"]),int(row["width"]),int(row["height"]))

colors={}
palette=[(0,255,0),(255,0,0),(0,200,255),(255,255,0)]
for i,t in enumerate(sorted(offs)): colors[t]=palette[i%len(palette)]

total=0
for txt in sorted(tiles.glob("*.txt")):
    if txt.stem.endswith("_check") or txt.stem not in offs: continue
    co,ro,W,H=offs[txt.stem]; col=colors[txt.stem]
    for line in txt.read_text().splitlines():
        p=line.split()
        if len(p)!=5: continue
        _,cx,cy,w,h=map(float,p)
        gx=co+cx*W; gy=ro+cy*H; gw=w*W; gh=h*H
        draw.rectangle([gx-gw/2,gy-gh/2,gx+gw/2,gy+gh/2],outline=col,width=2)
        total+=1
out=base/f"{site}_global_check.png"
img.save(out)
print(f"clip size {Wc}x{Hc}; drew {total} boxes (per-tile sum) -> {out.name}")
print("Each tile a different colour. Where two colours stack on one car = a seam duplicate.")
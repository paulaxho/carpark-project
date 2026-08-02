#!/usr/bin/env python3
"""
entrance_progress.py — checklist of which sites still need entrances digitised.

Reads outputs/gis/carpark_occupancy.gpkg and, for every uk_test site, reports
whether a retail_entrance and a carpark_entrance point exist, and whether the
occupancy analysis has produced a figure for it. Re-run any time after saving
edits in QGIS.

Writes:
  outputs/gis/entrance_progress.csv
  outputs/gis/entrance_progress.md
and prints the table.

Usage:  python scripts/entrance_progress.py
"""
from __future__ import annotations
from pathlib import Path
import shutil, tempfile
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
GPKG = ROOT / "outputs/gis/carpark_occupancy.gpkg"
REGISTRY = ROOT / "data/uk_retail/geolytix/processed/uk_site_registry_final.csv"
FIGS = ROOT / "outputs/figures/entrance"
OUT = ROOT / "outputs/gis"


def _count_by_site(layer):
    g = gpd.read_file(GPKG, layer=layer)
    if len(g) == 0 or "site_id" not in g.columns:
        return {}
    return g.dropna(subset=["site_id"]).groupby("site_id").size().to_dict()


def _save(df_text, path):
    t = Path(tempfile.mkdtemp()) / path.name
    t.write_text(df_text)
    shutil.copyfile(t, path)


def main():
    reg = pd.read_csv(REGISTRY)
    sites = list(reg[(reg.accepted == "Yes") &
                     (reg.proposed_split == "uk_test")].site_id)
    cars = _count_by_site("car_points")
    retail = _count_by_site("retail_entrance")
    carpark = _count_by_site("carpark_entrance")

    rows = []
    for s in sites:
        r_ok = retail.get(s, 0) > 0
        c_ok = carpark.get(s, 0) > 0
        analysed = (FIGS / f"{s}_retail.png").exists() or (FIGS / f"{s}_carpark.png").exists()
        status = "done" if (r_ok and c_ok) else ("partial" if (r_ok or c_ok) else "todo")
        rows.append({"site_id": s, "cars": cars.get(s, 0),
                     "retail_entrance": retail.get(s, 0),
                     "carpark_entrance": carpark.get(s, 0),
                     "analysed": "yes" if analysed else "no",
                     "status": status})
    df = pd.DataFrame(rows)
    done = (df.status == "done").sum()
    _save(df.to_csv(index=False), OUT / "entrance_progress.csv")

    tick = {"done": "✅ done", "partial": "🟡 partial", "todo": "⬜ todo"}
    md = [f"# Entrance digitising progress\n",
          f"**{done} / {len(df)} sites complete** "
          f"({(df.status=='partial').sum()} partial, "
          f"{(df.status=='todo').sum()} not started)\n",
          "| Site | Cars | Retail | Car-park | Analysed | Status |",
          "|---|---:|:---:|:---:|:---:|---|"]
    for _, r in df.iterrows():
        rc = "✔" if r.retail_entrance else "—"
        cc = "✔" if r.carpark_entrance else "—"
        md.append(f"| {r.site_id} | {r.cars} | {rc} | {cc} | "
                  f"{r.analysed} | {tick[r.status]} |")
    md.append("\n_Re-run `python scripts/entrance_progress.py` after saving QGIS "
              "edits to refresh._")
    _save("\n".join(md), OUT / "entrance_progress.md")

    print(f"{done}/{len(df)} sites complete\n")
    print(df.to_string(index=False))
    print(f"\n-> {OUT}/entrance_progress.md  (+ .csv)")


if __name__ == "__main__":
    main()

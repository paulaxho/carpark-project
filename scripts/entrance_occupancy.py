#!/usr/bin/env python3
"""
entrance_occupancy.py — does parking cluster near the entrances?

Reads the QGIS GeoPackage built by build_gis_package.py and, for every site that
has digitised entrance points, tests whether detected cars sit closer to the
entrance than the car park's available space does — i.e. whether near-entrance
bays fill first.

Method (needs only cars + AOI + entrance points; no empty-bay map required):
  * For each car, distance to the nearest entrance of its site (retail / carpark).
  * Sample the AOI polygon on a regular grid -> the distribution of distances the
    *available parking space* has to that entrance (the null "if parking were
    uniform" reference).
  * Relative fill in a distance ring =
        (share of cars in ring) / (share of available area in ring).
    >1 means that band is fuller than the lot average; a curve that falls with
    distance is the "fills from the entrance outward" signal.
  * Headline stats per site:
      dist_ratio  = mean(car dist) / mean(area dist).  <1 => cars nearer entrance.
      spearman    = corr(ring distance, relative fill). <0 => fills near entrance.

Outputs:
  outputs/stats/entrance_occupancy_cars.csv      per-car distances
  outputs/stats/entrance_occupancy_summary.csv   per-site, per-entrance-type stats
  outputs/figures/entrance/<SITE>_<kind>.png     per-site gradient figure

Usage:
  python scripts/entrance_occupancy.py                # uses digitised entrances
  python scripts/entrance_occupancy.py --demo         # provisional: store=retail entrance
  python scripts/entrance_occupancy.py UK002 UK023    # limit to sites
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
GPKG = ROOT / "outputs/gis/carpark_occupancy.gpkg"
STATS = ROOT / "outputs/stats"
FIGS = ROOT / "outputs/figures/report"
GRID_M = 2.0          # AOI sampling grid (m)
RING_M = 10.0         # distance-ring width for the gradient (m)


def _read(layer):
    return gpd.read_file(GPKG, layer=layer)


def _spearman(x, y):
    """Spearman rank correlation via numpy/pandas (no scipy dependency)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else np.nan


def nearest_dist(pts: gpd.GeoSeries, targets: gpd.GeoSeries) -> np.ndarray:
    """Min distance from each pt to any target (metres, EPSG:27700)."""
    tx = np.array([g.x for g in targets]); ty = np.array([g.y for g in targets])
    out = np.empty(len(pts))
    for i, g in enumerate(pts):
        out[i] = np.hypot(tx - g.x, ty - g.y).min()
    return out


def sample_polygon(poly, step=GRID_M) -> np.ndarray:
    """Regular grid of points inside a polygon -> (N,2) ground coords."""
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx, step); ys = np.arange(miny, maxy, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    from shapely import contains_xy
    m = contains_xy(poly, pts[:, 0], pts[:, 1])
    return pts[m]


def gradient(car_d, area_d, ring=RING_M):
    """Relative-fill per distance ring + Spearman(distance, fill)."""
    dmax = max(car_d.max(), area_d.max())
    edges = np.arange(0, dmax + ring, ring)
    mid = (edges[:-1] + edges[1:]) / 2
    car_h, _ = np.histogram(car_d, bins=edges)
    area_h, _ = np.histogram(area_d, bins=edges)
    car_s = car_h / car_h.sum()
    area_s = np.where(area_h.sum() > 0, area_h / area_h.sum(), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(area_s > 0, car_s / area_s, np.nan)
    ok = area_h > 0
    rho = _spearman(mid[ok], rel[ok]) if ok.sum() > 2 else np.nan
    return mid, rel, ok, rho


def figure(site, kind, car_d, area_d, mid, rel, ok, rho, ratio, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(0, max(car_d.max(), area_d.max()) + RING_M, RING_M)
    ax1.hist(area_d, bins=bins, density=True, alpha=0.45, label="available area",
             color="#888")
    ax1.hist(car_d, bins=bins, density=True, alpha=0.6, label="cars", color="#2a7")
    ax1.set_xlabel(f"distance to {kind} entrance (m)"); ax1.set_ylabel("density")
    ax1.set_title(f"{site}: where cars sit vs available space"); ax1.legend()
    ax2.axhline(1.0, color="#aaa", ls="--", lw=1)
    ax2.plot(mid[ok], rel[ok], "o-", color="#c33")
    ax2.set_xlabel(f"distance to {kind} entrance (m)")
    ax2.set_ylabel("relative fill  (cars ÷ area share)")
    ax2.set_title(f"gradient  (ratio={ratio:.2f}, ρ={rho:+.2f})")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def aggregate_dotplot(rows, out_png, tag=""):
    """Retail distance-ratio per site vs a reference line at 1.0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # width-aware sizes: FIG_W in, INCL = \includegraphics fraction (0.72\textwidth)
    FIG_W, INCL, REF_W = 6.4, 0.72, 14.0
    k = (FIG_W / INCL) / REF_W
    TITLE_FS, LABEL_FS, TICK_FS = 17*k, 13*k, 11*k

    r = [x for x in rows if x["entrance"] == "retail"]
    if not r:
        return
    r = sorted(r, key=lambda x: x["dist_ratio"])
    y = np.arange(len(r))
    ratios = [x["dist_ratio"] for x in r]
    labels = [x["site_id"] for x in r]
    colours = ["#2a8" if v < 0.95 else "#c33" if v > 1.05 else "#999" for v in ratios]
    fig, ax = plt.subplots(figsize=(FIG_W, max(3, 0.34 * len(r) + 1)))
    ax.axvline(1.0, color="#444", ls="--", lw=1, zorder=1)
    ax.hlines(y, 1.0, ratios, color="#ccc", lw=1, zorder=1)
    ax.scatter(ratios, y, c=colours, s=70, zorder=2, edgecolor="k", linewidth=0.4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=TICK_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.set_xlabel("Retail-entrance distance ratio  (cars / available space)", fontsize=LABEL_FS)
    ax.set_title("Do cars cluster near the shop entrance?  (ratio <1 = yes)"
                 + ("   [DEMO]" if tag else ""), fontsize=TITLE_FS, fontweight="bold")
    ax.margins(y=0.02)
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)


def small_multiples(grad, out_png, tag="", n=4):
    """2x2 panel of representative sites' retail relative-fill gradients."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # width-aware sizes: 9 in @ \textwidth
    FIG_W, INCL, REF_W = 9.0, 1.0, 14.0
    k = (FIG_W / INCL) / REF_W
    TITLE_FS, PANEL_FS, LABEL_FS, TICK_FS = 17*k, 15*k, 13*k, 12*k

    items = [(s, d) for s, d in grad.items()]
    if not items:
        return
    items.sort(key=lambda kv: kv[1]["ratio"])
    if len(items) <= n:
        pick = items
    else:
        idx = np.linspace(0, len(items) - 1, n).round().astype(int)
        pick = [items[i] for i in idx]
    rows_n = int(np.ceil(len(pick) / 2))
    fig, axes = plt.subplots(rows_n, 2, figsize=(FIG_W, 3.4 * rows_n), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, (s, d) in zip(axes.ravel(), pick):
        ax.set_visible(True)
        ax.axhline(1.0, color="#aaa", ls="--", lw=1)
        ax.plot(d["mid"][d["ok"]], d["rel"][d["ok"]], "o-", color="#c33", ms=5)
        ax.set_title(f"{s}  (ratio={d['ratio']:.2f}, ρ={d['rho']:+.2f})",
                     fontsize=PANEL_FS, fontweight="bold")
        ax.set_xlabel("distance to retail entrance (m)", fontsize=LABEL_FS)
        ax.set_ylabel("relative fill", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
    fig.suptitle("Parking fill vs distance to the retail entrance, representative sites"
                 + ("   [DEMO]" if tag else ""), fontsize=TITLE_FS, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=300); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sites", nargs="*")
    ap.add_argument("--demo", action="store_true",
                    help="provisional: use store_point as the retail entrance")
    ap.add_argument("--per-site", action="store_true",
                    help="also write the per-site two-panel PNGs (off by default)")
    args = ap.parse_args()

    STATS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    cars = _read("car_points"); aoi = _read("aoi_boundary")
    retail = _read("retail_entrance"); carpark = _read("carpark_entrance")
    if args.demo:
        store = _read("store_point")[["site_id", "geometry"]].copy()
        store["kind"] = "retail_DEMO"
        retail = store
        print("!! DEMO: using registry store location as a stand-in retail "
              "entrance. Replace with digitised entrances for real results.")

    sites = args.sites or sorted(cars.site_id.unique())
    rows, per_car = [], []
    retail_grad = {}
    for kind, ent in [("retail", retail), ("carpark", carpark)]:
        for site in sites:
            e = ent[ent.site_id == site]
            c = cars[cars.site_id == site]
            a = aoi[aoi.site_id == site]
            if len(e) == 0 or len(c) == 0 or len(a) == 0:
                continue
            area_pts = sample_polygon(a.geometry.iloc[0])
            if len(area_pts) == 0:
                continue
            car_d = nearest_dist(c.geometry, e.geometry)
            ex = e.geometry.x.values; ey = e.geometry.y.values
            area_d = np.min(np.hypot(area_pts[:, None, 0] - ex[None, :],
                                     area_pts[:, None, 1] - ey[None, :]), axis=1)
            mid, rel, ok, rho = gradient(car_d, area_d)
            ratio = car_d.mean() / area_d.mean()
            if args.per_site:
                figure(site, kind, car_d, area_d, mid, rel, ok, rho, ratio,
                       FIGS / f"{site}_{kind}.png")
            if kind == "retail":
                retail_grad[site] = {"mid": mid, "rel": rel, "ok": ok,
                                     "ratio": float(ratio), "rho": float(rho)}
            verdict = ("clusters near entrance" if ratio < 0.95 and (rho < 0 or np.isnan(rho))
                       else "away from entrance" if ratio > 1.05 else "no clear gradient")
            rows.append({"site_id": site, "entrance": kind, "n_cars": len(c),
                         "n_entrances": len(e),
                         "mean_car_dist_m": round(float(car_d.mean()), 1),
                         "mean_area_dist_m": round(float(area_d.mean()), 1),
                         "dist_ratio": round(float(ratio), 3),
                         "spearman_dist_fill": round(float(rho), 3) if not np.isnan(rho) else None,
                         "verdict": verdict})
            for cid, d in zip(c.car_id.values, car_d):
                per_car.append({"car_id": int(cid), "site_id": site,
                                "entrance": kind, "dist_m": round(float(d), 1)})
            print(f"  {site} [{kind}]: {len(c)} cars, ratio={ratio:.2f}, "
                  f"rho={rho:+.2f} -> {verdict}")

    if not rows:
        print("\nNo sites had entrance points. Digitise retail_entrance / "
              "carpark_entrance in QGIS (or run with --demo), then re-run.")
        return
    summ = pd.DataFrame(rows).sort_values(["entrance", "site_id"])
    tag = "_demo" if args.demo else ""
    summ.to_csv(STATS / f"entrance_occupancy_summary{tag}.csv", index=False)
    pd.DataFrame(per_car).to_csv(STATS / f"entrance_occupancy_cars{tag}.csv", index=False)

    # headline consolidated figures (retail entrance)
    aggregate_dotplot(rows, FIGS / f"fig_entrance_ratios{tag}.png", tag)
    small_multiples(retail_grad, FIGS / f"fig_entrance_gradients{tag}.png", tag)

    n_retail = len(retail_grad)
    print(f"\nsummary -> {STATS}/entrance_occupancy_summary{tag}.csv")
    print(f"headline figures -> {FIGS}/fig_entrance_ratios{tag}.png, "
          f"fig_entrance_gradients{tag}.png  ({n_retail} retail sites)")
    if n_retail < 4:
        print(f"  (only {n_retail} site(s) digitised — the consolidated figures "
              "fill out as you add more)")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
estimate_occupancy.py — car counts and occupancy rates on the 20 UK test sites.

Runs each of the three models (baseline / transfer / uk_adapt) at its frozen
operating threshold (from select_threshold.py), de-duplicates cars across tile
seams, and divides by the registry's estimated capacity. Reports counting error
and occupancy-rate error side by side, per model.

  count            : de-duplicated unique cars per site (src.counting.site_count)
  true occupancy   : truth_count / estimated_capacity
  pred occupancy   : pred_count  / estimated_capacity

Honesty note: the capacity denominator is a visual estimate and is NOT
independently verifiable, so occupancy-rate error carries denominator error that
the counting error does not. That is exactly why both are reported.

Inputs
------
  outputs/stats/occupancy_thresholds.csv   (run select_threshold.py first)
  data/uk_retail/geolytix/processed/uk_site_registry_final.csv
  data/uk_retail/processed/<SITE>/tiles + tile_index.csv

Outputs
-------
  outputs/stats/occupancy_eval.csv     per (site, model) row
  outputs/stats/occupancy_summary.csv  per-model aggregate metrics
  outputs/figures/fig_occupancy_scatter.png   predicted vs truth COUNT
  outputs/figures/fig_occupancy_rate.png      predicted vs true OCCUPANCY rate

Usage (from project root):
    python scripts/select_threshold.py      # first: freeze thresholds
    python scripts/estimate_occupancy.py    # then: counts + occupancy
"""
import sys, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.counting import load_offsets, truth_count, site_count

import numpy as np

REGISTRY = ROOT / "data/uk_retail/geolytix/processed/uk_site_registry_final.csv"
THRESHOLDS = ROOT / "outputs/stats/occupancy_thresholds.csv"
PROCESSED = ROOT / "data/uk_retail/processed"
STATS = ROOT / "outputs/stats"
FIGS = ROOT / "outputs/figures"
MERGE_IOU = 0.5
IMGSZ = 640

WEIGHTS = {
    "baseline": ROOT / "yolov8s.pt",
    "transfer": ROOT / "models/runs/combined_yolov8s_25cm/weights/best.pt",
    "uk_adapt": ROOT / "models/runs/uk_adapt_yolov8s_25cm/weights/best.pt",
}
MODEL_ORDER = ["baseline", "transfer", "uk_adapt"]
SMALL_LOT = 30  # truth below this: MAPE flagged as unstable


# --------------------------------------------------------------------------- #
def load_test_sites():
    """(site_id, capacity) for accepted uk_test sites, in registry order."""
    out = []
    for r in csv.DictReader(open(REGISTRY)):
        if r["accepted"] == "Yes" and r["proposed_split"] == "uk_test":
            out.append((r["site_id"], float(r["estimated_capacity"])))
    return out


def load_thresholds():
    if not THRESHOLDS.exists():
        sys.exit(f"missing {THRESHOLDS}\nRun: python scripts/select_threshold.py")
    return {r["model"]: float(r["tau"]) for r in csv.DictReader(open(THRESHOLDS))}


def metrics(pred, truth):
    """Counting-error metrics for aligned arrays of per-site counts."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    err = pred - truth
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # MAPE on all sites, and on lots with truth >= SMALL_LOT (stable subset)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(err) / truth
    mape_all = float(np.mean(ape)) * 100
    big = truth >= SMALL_LOT
    mape_big = float(np.mean(ape[big])) * 100 if big.any() else float("nan")
    return dict(MAE=mae, RMSE=rmse, bias=bias, R2=r2,
                MAPE_all_pct=mape_all, MAPE_big_pct=mape_big, n_big=int(big.sum()))


def occ_metrics(pred_occ, true_occ):
    pred_occ, true_occ = np.asarray(pred_occ, float), np.asarray(true_occ, float)
    err = pred_occ - true_occ
    mae_pp = float(np.mean(np.abs(err))) * 100
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true_occ - true_occ.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(MAE_occ_pp=mae_pp, R2_occ=r2)


# --------------------------------------------------------------------------- #
def main():
    from ultralytics import YOLO
    STATS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    sites = load_test_sites()
    taus = load_thresholds()
    print(f"{len(sites)} test sites; thresholds: "
          + ", ".join(f"{m}={taus[m]:.3f}" for m in MODEL_ORDER))

    # truth is model-independent: compute once per site
    caps = {s: c for s, c in sites}
    truth = {s: truth_count(PROCESSED / s, MERGE_IOU) for s, _ in sites}

    per_site_rows = []          # long-format CSV
    preds = {m: {} for m in MODEL_ORDER}
    for m in MODEL_ORDER:
        print(f"\n== counting with {m} (tau={taus[m]:.3f}) ==")
        model = YOLO(str(WEIGHTS[m]))          # load once, reuse across sites
        for s, cap in sites:
            offs = load_offsets(PROCESSED / s)
            pc = site_count(model, PROCESSED / s, taus[m], MERGE_IOU, IMGSZ, offs)
            preds[m][s] = pc
            per_site_rows.append(dict(
                site_id=s, capacity=cap, model=m,
                truth_count=truth[s], pred_count=pc,
                true_occ=round(truth[s] / cap, 4), pred_occ=round(pc / cap, 4),
                abs_err=abs(pc - truth[s])))
            print(f"   {s}: truth={truth[s]:3d} pred={pc:3d} cap={cap:.0f}")

    # per-site CSV
    eval_csv = STATS / "occupancy_eval.csv"
    with open(eval_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_site_rows[0].keys()))
        w.writeheader(); w.writerows(per_site_rows)

    # per-model summary
    summ_rows = []
    order = [s for s, _ in sites]
    t_arr = [truth[s] for s in order]
    t_occ = [truth[s] / caps[s] for s in order]
    for m in MODEL_ORDER:
        p_arr = [preds[m][s] for s in order]
        p_occ = [preds[m][s] / caps[s] for s in order]
        row = {"model": m, "tau": round(taus[m], 4)}
        row.update({k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in metrics(p_arr, t_arr).items()})
        row.update({k: round(v, 4) for k, v in occ_metrics(p_occ, t_occ).items()})
        summ_rows.append(row)
    summ_csv = STATS / "occupancy_summary.csv"
    with open(summ_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ_rows[0].keys()))
        w.writeheader(); w.writerows(summ_rows)

    _figures(order, truth, caps, preds)

    print(f"\nWritten:\n  {eval_csv}\n  {summ_csv}"
          f"\n  {FIGS/'fig_occupancy_scatter.png'}\n  {FIGS/'fig_occupancy_rate.png'}")
    print("\nSummary:")
    for r in summ_rows:
        print(f"  {r['model']:9s} MAE={r['MAE']:.1f} RMSE={r['RMSE']:.1f} "
              f"bias={r['bias']:+.1f} R2={r['R2']:.3f}  "
              f"occ MAE={r['MAE_occ_pp']:.1f}pp R2={r['R2_occ']:.3f}")


def _figures(order, truth, caps, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#e8862a", "transfer": "#3a6ea5", "uk_adapt": "#3ba55d"}
    labels = {"baseline": "Baseline", "transfer": "Transfer", "uk_adapt": "UK-adaptation"}

    # ---- count scatter ----
    t = [truth[s] for s in order]
    hi = max(t + [max(preds[m].values()) for m in MODEL_ORDER]) * 1.05
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, hi], [0, hi], "--", color="#999", lw=1, label="perfect (y = x)")
    for m in MODEL_ORDER:
        ax.scatter(t, [preds[m][s] for s in order], s=42, alpha=0.8,
                   color=colors[m], edgecolor="white", linewidth=0.6, label=labels[m])
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("True car count (de-duplicated ground truth)")
    ax.set_ylabel("Predicted car count")
    ax.set_title("Counting accuracy on the 20 UK test sites")
    ax.legend(frameon=False); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(FIGS / "fig_occupancy_scatter.png", dpi=200)
    plt.close(fig)

    # ---- occupancy-rate scatter ----
    to = [truth[s] / caps[s] * 100 for s in order]
    hi2 = max(to + [max(preds[m][s] / caps[s] * 100 for s in order)
                    for m in MODEL_ORDER]) * 1.05
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, hi2], [0, hi2], "--", color="#999", lw=1, label="perfect (y = x)")
    for m in MODEL_ORDER:
        ax.scatter(to, [preds[m][s] / caps[s] * 100 for s in order], s=42, alpha=0.8,
                   color=colors[m], edgecolor="white", linewidth=0.6, label=labels[m])
    ax.set_xlim(0, hi2); ax.set_ylim(0, hi2)
    ax.set_xlabel("True occupancy rate (% of capacity)")
    ax.set_ylabel("Predicted occupancy rate (%)")
    ax.set_title("Occupancy-rate accuracy on the 20 UK test sites")
    ax.legend(frameon=False); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(FIGS / "fig_occupancy_rate.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

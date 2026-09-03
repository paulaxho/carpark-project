# Estimating UK Retail Car-Park Occupancy from High-Resolution Aerial Imagery

A domain-adapted object-detection pipeline that detects and counts vehicles in dense UK
retail car parks from ~25 cm aerial imagery, turns those counts into occupancy estimates,
and maps where within each lot demand concentrates.

MSc Emerging Digital Technologies dissertation, University College London (2026).
Author: Paula Xho Vilajeti. Academic supervisor: Dr David Vidal Tomas. Industrial
supervisor: Toby Merritt (Knight Frank).

## Summary

A YOLOv8s detector is trained on two public overhead-vehicle datasets (COWC and VEDAI),
resolution-matched to 25 cm, then adapted on a purpose-built set of 30 hand-labelled UK
retail car parks. Detection and counting are evaluated as **separate** quantities, and the
located detections are re-expressed as georeferenced points for a within-lot spatial
analysis.

**Headline results (20 held-out UK test sites):**

- Detection mAP@0.5 rose across the adaptation ladder: **0.267** (stock baseline) →
  **0.528** (zero-shot transfer) → **0.886** (fine-tuned on 8 UK sites).
- Fine-tuning on just eight sites recovered most of a 0.41 domain gap — a small in-domain
  set outweighed a large volume of out-of-domain data.
- The adapted detector counted to a mean absolute error of **5.4 cars/site** and recovered
  occupancy to within **2.6 percentage points** of capacity (R² = 0.97), while the
  unadapted models returned negative R².
- Parking concentrated near the store's pedestrian entrance on **18/20** sites, with no
  comparable pull toward the vehicle entrance.

## Repository structure

```
src/            core library
  annotations.py  centre-point / box handling
  boxes.py        box geometry and conversion
  resample.py     resolution matching (downsampling)
  tiling.py       overlapping sliding-window tiling
  splits.py       scene/site-level split logic
  counting.py     seam-aware de-duplication and site counts
  geospatial.py   pixel -> British National Grid georeferencing
scripts/        runnable pipeline stages (see Usage)
configs/        dataset + YOLO training configs
notebooks/      inspection and report-figure notebooks
data/           datasets (imagery not tracked — see Data availability)
models/runs/    per-run args.yaml + results.csv (weights not tracked)
outputs/        stats, figures, GIS exports
requirements.txt
```

## Data availability

The **code**, the **hand-verified UK vehicle annotations** (YOLO `.txt`), the digitised
**entrance and car-park-boundary layers** (`aois.gpkg`), the **site registry**, and each
run's `args.yaml` / `results.csv` are included.

The **UK aerial imagery is licensed** through UCL via the EDINA Aerial Digimap
(Getmapping) service and **cannot be redistributed**. It is therefore not in this
repository. The site registry records the exact tiles and capture dates needed to
re-download the imagery through Digimap; with the imagery in place, every result can be
regenerated from this code.

Public sources: COWC (Mundhenk et al., 2016) and VEDAI (Razakarivony & Jurie, 2016) are
available from their original providers. The UK site list derives from the open GeoLytix
Supermarket Retail Points dataset.

## Environment

Model training/inference ran on an NVIDIA GPU (Rocky Linux 9.8, Python 3.9, PyTorch
2.3.0+cu121, Ultralytics YOLOv8 8.2.0); data preparation, annotation and spatial analysis
ran locally on macOS (Python 3.11). All Python dependencies are pinned in
`requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GDAL must be installed first via the system package manager, e.g.:
#   brew install gdal && pip install GDAL==3.13.0
```

Reproducibility: a fixed seed (`seed=0`) and deterministic mode were used; the scene- and
site-level splits are frozen assignments read from config and the registry. Exact
bitwise reproduction is not guaranteed across GPUs.

## Usage

Get the data first (see "Data availability"). Download and place:
    - COWC   -> data/cowc/raw/
    - VEDAI  -> data/vedai/raw/
    - UK 25 cm Digimap imagery -> data/uk_retail/<site>/imagery/
      (request the exact tiles + capture dates listed in the site registry)
 The build scripts below assume the raw data is already in place; they do not download it.

Run from the project root. Representative order:

```bash
# 1. Build resolution-matched source datasets (25 cm, tiled, scene-split)
python scripts/build_cowc.py
python scripts/build_vedai.py

# 2. Train the source and combined models
python scripts/train.py --config configs/yolo_cowc.yaml
python scripts/train.py --config configs/yolo_vedai.yaml
python scripts/train.py --config configs/yolo_combined.yaml

# 3. Prepare the UK dataset (per-site clip/tile, then finalise + assemble)
python scripts/build_uk_site.py UK001            # repeat per site
python scripts/finalise_registry.py
python scripts/assemble_uk_dataset.py

# 4. Run the three UK experiments (baseline / transfer / UK-adaptation)
python scripts/run_uk_experiments.py

# 5. Freeze the F1-optimal operating threshold, then estimate counts + occupancy
python scripts/select_threshold.py
python scripts/estimate_occupancy.py

# 6. Spatial analysis: GIS package, density surfaces, entrance-proximity
python scripts/build_gis_package.py
python scripts/spatial_density.py
python scripts/entrance_occupancy.py

# 7. Report figures
jupyter lab notebooks/07_report_figures.ipynb
```

Many scripts accept site IDs and options; run with `-h`/`--help` for details.

## Outputs

- `outputs/stats/` — detection, counting, occupancy and entrance-proximity CSVs
- `outputs/figures/` — report figures (detection ladder, domain gap, occupancy, entrance)
- `outputs/spatial/` and `outputs/gis/` — georeferenced points and the QGIS GeoPackage

## Citation

```
Vilajeti, P. X. (2026). Estimating UK Retail Car-Park Occupancy from High-Resolution
Aerial Imagery: A Domain-Adapted Object-Detection Approach. MSc dissertation,
University College London.
```

## Acknowledgements

Thanks to Dr David Vidal Tomas and Toby Merritt for their supervision, and to Knight Frank
for hosting the project.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from src.resample import save_resampled_image

input_path = Path(
    "data/cowc/raw/cowc/datasets/ground_truth_sets/Vaihingen_ISPRS/"
    "TOP_Mosaic_09cm_scaled_15cm_Gray.png"
)

output_path = Path(
    "outputs/figures/vaihingen_25cm_test.png"
)

save_resampled_image(
    input_path=input_path,
    output_path=output_path,
    source_cm=15.0,
    target_cm=25.0,
)
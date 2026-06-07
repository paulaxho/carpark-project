import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.annotations import (
    extract_points_from_mask,
    find_car_mask,
    find_negative_mask,
    scale_points,
)
from src.resample import resolution_scale, resample_image
from src.tiling import tile_scene


def main():
    image_path = Path(
        "data/cowc/raw/cowc/datasets/ground_truth_sets/Vaihingen_ISPRS/"
        "TOP_Mosaic_09cm_scaled_15cm_Gray.png"
    )

    out_img_dir = Path("data/cowc/processed/images/test_tiling")
    out_lbl_dir = Path("data/cowc/processed/labels/test_tiling")

    scale = resolution_scale(source_cm=15.0, target_cm=25.0)

    image_15 = Image.open(image_path).convert("RGB")
    image_25 = resample_image(image_15, scale)

    car_xs, car_ys = extract_points_from_mask(find_car_mask(image_path))
    neg_xs, neg_ys = extract_points_from_mask(find_negative_mask(image_path))

    car_xs_25, car_ys_25 = scale_points(car_xs, car_ys, scale)
    neg_xs_25, neg_ys_25 = scale_points(neg_xs, neg_ys, scale)

    stats = tile_scene(
        image=image_25,
        car_xs=car_xs_25,
        car_ys=car_ys_25,
        neg_xs=neg_xs_25,
        neg_ys=neg_ys_25,
        scene_name="vaihingen_25cm",
        out_img_dir=out_img_dir,
        out_lbl_dir=out_lbl_dir,
        tile_size=640,
        stride=512,
        box_px=14,
        keep_empty_fraction=0.12,
        skip_blank_std=1.0,
        rng_seed=42,
    )

    print("\nTiling test complete.")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\nImages saved to: {out_img_dir}")
    print(f"Labels saved to: {out_lbl_dir}")


if __name__ == "__main__":
    main()
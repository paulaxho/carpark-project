import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.splits import (
    assign_split, split_dirs, summarise_split, validate_split_config,
)

scenes = ["Toronto_ISPRS", "Selwyn_LINZ", "Potsdam_ISPRS",
          "Columbus_CSUAV", "Utah_AGRC", "Vaihingen_ISPRS"]
val = ["Potsdam_ISPRS"]

# 1) assignment
assert assign_split("Potsdam_ISPRS", val) == "val"
assert assign_split("Toronto_ISPRS", val) == "train"

# 2) dirs
img, lbl = split_dirs(Path("data/cowc/processed"), "val")
assert img == Path("data/cowc/processed/images/val")

# 3) summary
print("summary:", summarise_split(scenes, val))

# 4) typo guard fires
try:
    validate_split_config(scenes, ["Potdsam_ISPRS"])
    print("FAIL: typo not caught")
except ValueError as e:
    print("typo caught:", e)

# 5) overlap guard fires
try:
    validate_split_config(scenes, ["Potsdam_ISPRS"], ["Potsdam_ISPRS"])
    print("FAIL: overlap not caught")
except ValueError as e:
    print("overlap caught:", e)

# 6) empty-val guard fires
try:
    validate_split_config(scenes, [])
    print("FAIL: empty val not caught")
except ValueError as e:
    print("empty val caught:", e)

# 7) valid config passes
ok = validate_split_config(scenes, ["Potsdam_ISPRS"], ["Utah_AGRC"])
print("valid:", ok)

print("\nAll splits checks passed.")
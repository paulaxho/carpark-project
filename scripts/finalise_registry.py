#!/usr/bin/env python3
"""
finalise_registry.py
--------------------
Freezes the UK site registry: re-derives car_park_size_class from estimated_capacity
and validates structure. The train/test split is LOCKED and is preserved as-is
(this script never re-splits).

Usage:
    python finalise_registry.py uk_site_registry_final.csv uk_site_registry_final.csv
    # in-place is fine (reads, then writes back)

Bands (change in size_band() if you revise them):
    Small  : capacity < 75
    Medium : 75 <= capacity <= 200
    Large  : capacity > 200
"""
import sys, csv, datetime
from collections import Counter

EXPECTED_COLS = 22
RESOLUTION_CM = "25"

def size_band(c):
    if c < 75:
        return "Small"
    if c <= 200:
        return "Medium"
    return "Large"

def main(inp, outp):
    rows = list(csv.reader(open(inp)))
    hdr, body = rows[0], rows[1:]
    idx = {n: i for i, n in enumerate(hdr)}
    problems, changes = [], []

    if len(hdr) != EXPECTED_COLS:
        problems.append(f"Header has {len(hdr)} columns, expected {EXPECTED_COLS}")
    bad_len = [(r[0], len(r)) for r in body if len(r) != len(hdr)]
    if bad_len:
        problems.append(f"Rows with wrong field count: {bad_len}")

    bad_res = [r[0] for r in body if r[idx['imagery_resolution_cm']] != RESOLUTION_CM]
    if bad_res:
        problems.append(f"Non-{RESOLUTION_CM}cm resolution: {bad_res}")

    for r in body:
        d = r[idx['imagery_capture_date']].strip()
        if d:
            try:
                datetime.date.fromisoformat(d)
            except ValueError:
                problems.append(f"{r[0]}: unparseable date '{d}'")

    for r in body:
        if r[idx['accepted']] == "Yes":
            if r[idx['estimated_capacity']].strip() == "":
                problems.append(f"{r[0]}: accepted but missing estimated_capacity")
            if r[idx['proposed_split']].strip() in ("", "TBD", "N/A"):
                problems.append(f"{r[0]}: accepted but split not assigned ({r[idx['proposed_split']]})")

    for r in body:
        cap = r[idx['estimated_capacity']].strip()
        if cap == "":
            continue
        try:
            cval = float(cap)
        except ValueError:
            problems.append(f"{r[0]}: non-numeric capacity '{cap}'")
            continue
        new = size_band(cval)
        if new != r[idx['car_park_size_class']]:
            changes.append((r[0], int(cval), r[idx['car_park_size_class']], new))
            r[idx['car_park_size_class']] = new

    if changes:
        print("Size-class corrections (capacity -> old -> new):")
        for sid, c, old, new in changes:
            print(f"  {sid}: {c:>3}  {old:<6} -> {new}")
    else:
        print("Size classes already consistent with capacity.")

    acc = [r for r in body if r[idx['accepted']] == "Yes"]
    print("\nSplit (locked, unchanged):", dict(Counter(r[idx['proposed_split']] for r in acc)))
    print("Size-class distribution:", dict(Counter(r[idx['car_park_size_class']] for r in acc)))

    print("\n" + ("VALIDATION ISSUES:" if problems else "No structural problems found. Registry frozen."))
    for p in problems:
        print("  - " + p)

    with open(outp, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nWritten: {outp}")

if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "data/uk_retail/geolytix/processed/uk_site_registry_final.csv"
    b = sys.argv[2] if len(sys.argv) > 2 else a
    main(a, b)

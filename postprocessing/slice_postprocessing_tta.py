"""
Slice-to-Slice Post-Processing
================================
Takes raw YOLO detections (from extract_raw_detections.py) and filters them
based on consistency across consecutive slices within the same patient.

Implements three rules:
  Rule A — keep if detection appears in >= 2 consecutive slices
  Rule B — keep if detection appears in >= 3 consecutive slices
  Rule C — keep if confidence > 0.5 OR appears in >= 2 consecutive slices

A "match" between two detections in neighbouring slices is determined by
centroid distance (in pixels) being below a threshold.
"""

import csv
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(".")
INPUT_CSV    = BASE_DIR / "runs" / "raw_detections_tta.csv"
OUTPUT_DIR   = BASE_DIR / "runs" / "slice_postprocessing_tta"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CENTROID_DIST_THRESH = 30.0   # pixels — how close centroids must be to count as "same lesion"
CONF_HIGH_THRESH      = 0.5   # used in Rule C

# ── Load detections ─────────────────────────────────────────────────────────
print(f"Loading detections from {INPUT_CSV}")

detections = []
with open(INPUT_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        detections.append({
            "filename":   row["filename"],
            "patient_id": row["patient_id"],
            "phase":      row["phase"],
            "slice_num":  int(row["slice_num"]),
            "confidence": float(row["confidence"]),
            "cx":         float(row["cx"]),
            "cy":         float(row["cy"]),
            "x1":         float(row["x1"]),
            "y1":         float(row["y1"]),
            "x2":         float(row["x2"]),
            "y2":         float(row["y2"]),
        })

print(f"Loaded {len(detections)} detections")

# ── Group by patient ────────────────────────────────────────────────────────
by_patient = defaultdict(list)
for d in detections:
    by_patient[d["patient_id"]].append(d)

for pid in by_patient:
    by_patient[pid].sort(key=lambda d: d["slice_num"])

print(f"Detections span {len(by_patient)} patients")

# ── Matching function ───────────────────────────────────────────────────────
def centroid_distance(d1, d2):
    return ((d1["cx"] - d2["cx"]) ** 2 + (d1["cy"] - d2["cy"]) ** 2) ** 0.5


def count_consecutive_matches(detection, all_patient_dets):
    """
    For a given detection, count how many consecutive slices (including itself)
    have a matching detection nearby (by centroid distance).
    Returns the length of the longest consecutive run containing this detection.
    """
    slice_num = detection["slice_num"]

    # Build a quick lookup: slice_num -> list of detections in that slice
    slice_lookup = defaultdict(list)
    for d in all_patient_dets:
        slice_lookup[d["slice_num"]].append(d)

    def has_match_in_slice(ref_det, target_slice):
        for cand in slice_lookup.get(target_slice, []):
            if centroid_distance(ref_det, cand) <= CENTROID_DIST_THRESH:
                return True
        return False

    # Count forward
    run = 1
    s = slice_num + 1
    while has_match_in_slice(detection, s):
        run += 1
        s += 1

    # Count backward
    s = slice_num - 1
    while has_match_in_slice(detection, s):
        run += 1
        s -= 1

    return run


# ── Apply filtering rules ───────────────────────────────────────────────────
print("Computing consecutive-slice run lengths for each detection...")

results_a, results_b, results_c = [], [], []

for pid, dets in by_patient.items():
    for d in dets:
        run_length = count_consecutive_matches(d, dets)
        d["run_length"] = run_length

        # Rule A: keep if run >= 2
        if run_length >= 2:
            results_a.append(d)

        # Rule B: keep if run >= 3
        if run_length >= 3:
            results_b.append(d)

        # Rule C: keep if high confidence OR run >= 2
        if d["confidence"] > CONF_HIGH_THRESH or run_length >= 2:
            results_c.append(d)

print(f"\nOriginal detections : {len(detections)}")
print(f"Rule A (>=2 consecutive)        : {len(results_a)} kept ({len(detections)-len(results_a)} removed)")
print(f"Rule B (>=3 consecutive)        : {len(results_b)} kept ({len(detections)-len(results_b)} removed)")
print(f"Rule C (conf>0.5 OR >=2 consec) : {len(results_c)} kept ({len(detections)-len(results_c)} removed)")

# ── Save each rule's filtered detections ────────────────────────────────────
def save_csv(rows, path):
    fieldnames = ["filename", "patient_id", "phase", "slice_num", "confidence",
                  "x1", "y1", "x2", "y2", "cx", "cy", "run_length"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

save_csv(results_a, OUTPUT_DIR / "filtered_rule_a.csv")
save_csv(results_b, OUTPUT_DIR / "filtered_rule_b.csv")
save_csv(results_c, OUTPUT_DIR / "filtered_rule_c.csv")

# Also save the unfiltered detections with run_length attached, for reference
all_with_run = [d for dets in by_patient.values() for d in dets]
save_csv(all_with_run, OUTPUT_DIR / "all_detections_with_run_length.csv")

print(f"\nSaved filtered results to {OUTPUT_DIR}")
print("Done.")

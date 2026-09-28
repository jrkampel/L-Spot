"""
Hard Negative Mining: Step 2 — Build Updated Classifier Dataset
===================================================================
Combines:
  - Original tumour patches (unchanged, from extract_classifier_patches.py)
  - NEW hard negative patches (cropped from YOLO's actual false positives
    on the training set, identified in find_train_false_positives.py)
  - Original random healthy patches (kept too, for diversity)

This gives the classifier exposure to both generic healthy tissue AND
the specific kind of mistake YOLO makes — making it much better suited
to its actual filtering task.
"""

import cv2
import csv
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR          = Path(".")
TRAIN_IMAGES_DIR  = BASE_DIR / "dataset" / "train" / "images"
FALSE_POS_CSV     = BASE_DIR / "runs" / "train_false_positives.csv"
EXISTING_DATASET  = BASE_DIR / "classifier_dataset"   # has train/tumour, train/healthy etc.
OUTPUT_DATASET    = BASE_DIR / "classifier_dataset_v2"

PATCH_SIZE = 64
VAL_SPLIT_HOLDOUT_PATIENTS = 0.15  # fraction of patients held out for val hard negatives

# ── Helper ──────────────────────────────────────────────────────────────────
def extract_patch(img, cx, cy, patch_size):
    h, w = img.shape[:2]
    half = patch_size // 2
    x1 = int(cx - half); y1 = int(cy - half)
    x2 = x1 + patch_size; y2 = y1 + patch_size
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > w: x1 -= (x2 - w); x2 = w
    if y2 > h: y1 -= (y2 - h); y2 = h
    x1, y1 = max(0, x1), max(0, y1)
    patch = img[y1:y2, x1:x2]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size))
    return patch


# ── Step A: Copy existing tumour and healthy patches into new dataset ───────
print("Setting up new dataset structure...")

import shutil

for split in ["train", "val"]:
    for cls in ["tumour", "healthy", "hard_negative"]:
        (OUTPUT_DATASET / split / cls).mkdir(parents=True, exist_ok=True)

# Copy existing tumour + healthy patches (train and val) as-is
for split in ["train", "val"]:
    for cls in ["tumour", "healthy"]:
        src_dir = EXISTING_DATASET / split / cls
        dst_dir = OUTPUT_DATASET / split / cls
        if src_dir.exists():
            for f in src_dir.glob("*.png"):
                shutil.copy(f, dst_dir / f.name)

print("Copied existing tumour and healthy patches.")

# ── Step B: Load false positives and split into train/val by patient ───────
print(f"Loading false positives from {FALSE_POS_CSV}")

with open(FALSE_POS_CSV, "r") as f:
    reader = csv.DictReader(f)
    false_positives = list(reader)

print(f"Loaded {len(false_positives)} false positives")

# Group by patient ID (extracted from filename, e.g. P0002_C1_slice_016)
import re
PATTERN = re.compile(r"(P\d+)_")

def get_patient_id(filename):
    m = PATTERN.match(filename)
    return m.group(1) if m else "UNKNOWN"

patients = sorted(set(get_patient_id(fp["filename"]) for fp in false_positives))
n_val_patients = max(1, int(len(patients) * VAL_SPLIT_HOLDOUT_PATIENTS))
val_patients = set(patients[:n_val_patients])

print(f"Found {len(patients)} unique patients with false positives")
print(f"Holding out {len(val_patients)} patients for val: {val_patients}")

# ── Step C: Extract patches for each false positive ─────────────────────────
print("Extracting hard negative patches...")

image_cache = {}
n_train_saved = 0
n_val_saved = 0
n_skipped = 0

for i, fp in enumerate(false_positives):
    filename = fp["filename"]
    patient_id = get_patient_id(filename)
    split = "val" if patient_id in val_patients else "train"

    if filename not in image_cache:
        img_path = TRAIN_IMAGES_DIR / f"{filename}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            n_skipped += 1
            continue
        image_cache[filename] = img
    img = image_cache[filename]

    cx, cy = float(fp["cx"]), float(fp["cy"])
    patch = extract_patch(img, cx, cy, PATCH_SIZE)

    if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
        n_skipped += 1
        continue

    out_name = f"{filename}_fp{i:04d}.png"
    out_path = OUTPUT_DATASET / split / "hard_negative" / out_name
    cv2.imwrite(str(out_path), patch)

    if split == "train":
        n_train_saved += 1
    else:
        n_val_saved += 1

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"Hard negative patches saved (train) : {n_train_saved}")
print(f"Hard negative patches saved (val)   : {n_val_saved}")
print(f"Skipped                             : {n_skipped}")
print(f"New dataset at                      : {OUTPUT_DATASET}")
print(f"{'='*60}")

# Print final dataset composition
print("\nFinal dataset composition:")
for split in ["train", "val"]:
    for cls in ["tumour", "healthy", "hard_negative"]:
        count = len(list((OUTPUT_DATASET / split / cls).glob("*.png")))
        print(f"  {split}/{cls}: {count}")

"""
Run YOLOv8m inference on the full test set and save all raw detections
to a CSV file, including patient ID and slice number parsed from filename.

This is Step 1 of Slice-to-Slice Post-Processing.
"""

from ultralytics import YOLO
import csv
import re
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(".")
MODEL_PATH  = BASE_DIR / "runs" / "liver_tumour_yolov8" / "weights" / "best.pt"
TEST_IMAGES = BASE_DIR / "dataset" / "test" / "images"
OUTPUT_CSV  = BASE_DIR / "runs" / "raw_detections_tta.csv"
CONF_THRESH = 0.10   # low threshold to capture all possible detections; we filter later

# ── Filename parser ────────────────────────────────────────────────────────
# Expected format: P0002_C1_slice_016.png
PATTERN = re.compile(r"(P\d+)_([A-Za-z0-9]+)_slice_(\d+)")

def parse_filename(stem):
    """Returns (patient_id, phase, slice_number) or (None, None, None) if no match."""
    m = PATTERN.match(stem)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None

# ── Run inference ──────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

print(f"Running inference on {TEST_IMAGES}")
results = model.predict(
    source=str(TEST_IMAGES),
    conf=CONF_THRESH,
    imgsz=512,
    save=False,
    augment=True,   # enable test-time augmentation
    stream=True,   # process one at a time to save memory
)

# ── Collect detections ─────────────────────────────────────────────────────
rows = []
n_images = 0
n_detections = 0

for r in results:
    n_images += 1
    stem = Path(r.path).stem
    patient_id, phase, slice_num = parse_filename(stem)

    if patient_id is None:
        print(f"  WARNING: could not parse filename {stem}, skipping detections for this image")
        continue

    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        continue

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        cx   = (x1 + x2) / 2
        cy   = (y1 + y2) / 2

        rows.append({
            "filename": stem,
            "patient_id": patient_id,
            "phase": phase,
            "slice_num": slice_num,
            "class": cls,
            "confidence": round(conf, 4),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "cx": round(cx, 2),
            "cy": round(cy, 2),
        })
        n_detections += 1

# ── Save to CSV ─────────────────────────────────────────────────────────────
print(f"Saving {n_detections} detections from {n_images} images to {OUTPUT_CSV}")

with open(OUTPUT_CSV, "w", newline="") as f:
    fieldnames = ["filename", "patient_id", "phase", "slice_num", "class",
                  "confidence", "x1", "y1", "x2", "y2", "cx", "cy"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Done.")
print(f"Total images processed : {n_images}")
print(f"Total detections saved : {n_detections}")

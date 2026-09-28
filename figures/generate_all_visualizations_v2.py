"""
Generate Visualisations for ALL test images with confidence scores
=================================================================
Side-by-side comparison of baseline vs 3-way ensemble with confidence
scores printed on each bounding box.

Colour legend:
  GREEN = Ground truth
  RED   = Baseline YOLOv8m detection
  BLUE  = 3-way ensemble detection
"""

import cv2
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR        = Path(".")
TEST_IMAGES_DIR = BASE_DIR / "dataset" / "test" / "images"
TEST_LABELS_DIR = BASE_DIR / "dataset" / "test" / "labels"
BASELINE_CSV    = BASE_DIR / "runs" / "raw_detections.csv"
ENSEMBLE_CSV    = BASE_DIR / "runs" / "three_way_ensemble" / "ensemble_2of3_detections.csv"
OUTPUT_DIR      = BASE_DIR / "runs" / "visualizations_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 512
GREEN = (0, 255, 0)
RED   = (0, 0, 255)
BLUE  = (255, 100, 0)

def load_gt_boxes(stem):
    label_path = TEST_LABELS_DIR / f"{stem}.txt"
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, w, h = map(float, parts)
                x1 = (cx - w/2) * IMG_SIZE; y1 = (cy - h/2) * IMG_SIZE
                x2 = (cx + w/2) * IMG_SIZE; y2 = (cy + h/2) * IMG_SIZE
                boxes.append([x1, y1, x2, y2])
    return boxes

def draw_box_with_conf(img, box, color, conf=None):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if conf is not None:
        label = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

print("Loading detections...")
baseline_by_image = defaultdict(list)
with open(BASELINE_CSV, "r") as f:
    for row in csv.DictReader(f):
        baseline_by_image[row["filename"]].append({
            "box": [float(row["x1"]), float(row["y1"]),
                    float(row["x2"]), float(row["y2"])],
            "conf": float(row["confidence"])
        })

ensemble_by_image = defaultdict(list)
with open(ENSEMBLE_CSV, "r") as f:
    for row in csv.DictReader(f):
        conf = float(row["confidence"]) if "confidence" in row else None
        ensemble_by_image[row["filename"]].append({
            "box": [float(row["x1"]), float(row["y1"]),
                    float(row["x2"]), float(row["y2"])],
            "conf": conf
        })

all_stems    = sorted([f.stem for f in TEST_LABELS_DIR.glob("*.txt")])
tumour_stems = [s for s in all_stems if load_gt_boxes(s)]
print(f"Found {len(tumour_stems)} test images with ground truth tumours")

saved = 0
for stem in tumour_stems:
    img_path = TEST_IMAGES_DIR / f"{stem}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    gt_boxes      = load_gt_boxes(stem)
    baseline_dets = baseline_by_image.get(stem, [])
    ensemble_dets = ensemble_by_image.get(stem, [])

    left = img.copy()
    for box in gt_boxes:
        draw_box_with_conf(left, box, GREEN, conf=None)
    for det in baseline_dets:
        draw_box_with_conf(left, det["box"], RED, conf=det["conf"])
    cv2.putText(left, "Baseline YOLOv8m", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    right = img.copy()
    for box in gt_boxes:
        draw_box_with_conf(right, box, GREEN, conf=None)
    for det in ensemble_dets:
        draw_box_with_conf(right, det["box"], BLUE, conf=det["conf"])
    cv2.putText(right, "3-Way Ensemble", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    separator = np.ones((img.shape[0], 4, 3), dtype=np.uint8) * 255
    combined  = cv2.hconcat([left, separator, right])

    cv2.imwrite(str(OUTPUT_DIR / f"{stem}.png"), combined)
    saved += 1

    if saved % 100 == 0:
        print(f"  Saved {saved}/{len(tumour_stems)}...")

print(f"\nDone. {saved} images saved to {OUTPUT_DIR}")
print("Legend: GREEN = ground truth | RED = baseline | BLUE = 3-way ensemble")

"""
Hard Negative Mining: Step 1 — Identify YOLO's False Positives on Train Set
==============================================================================
Runs YOLOv8m baseline inference on the TRAINING set images, then compares
detections against ground truth to identify false positives.

These false positive locations will be used as "hard negative" examples
to retrain the binary classifier — teaching it specifically what YOLO's
mistakes look like, rather than generic healthy tissue.
"""

from ultralytics import YOLO
import csv
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(".")
MODEL_PATH      = BASE_DIR / "runs" / "liver_tumour_yolov8" / "weights" / "best.pt"
TRAIN_IMAGES    = BASE_DIR / "dataset" / "train" / "images"
TRAIN_LABELS    = BASE_DIR / "dataset" / "train" / "labels"
OUTPUT_CSV      = BASE_DIR / "runs" / "train_false_positives.csv"
CONF_THRESH     = 0.10
IMG_SIZE        = 512
IOU_THRESHOLD   = 0.30   # below this IoU with any GT box, a detection counts as a false positive

# ── Helper functions ────────────────────────────────────────────────────────

def load_gt_boxes_for_image(stem):
    label_path = TRAIN_LABELS / f"{stem}.txt"
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, w, h = map(float, parts)
                x1 = (cx - w / 2) * IMG_SIZE
                y1 = (cy - h / 2) * IMG_SIZE
                x2 = (cx + w / 2) * IMG_SIZE
                y2 = (cy + h / 2) * IMG_SIZE
                boxes.append([x1, y1, x2, y2])
    return boxes


def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ── Run inference ──────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

print(f"Running inference on training images: {TRAIN_IMAGES}")
results = model.predict(
    source=str(TRAIN_IMAGES),
    conf=CONF_THRESH,
    imgsz=512,
    save=False,
    stream=True,
)

# ── Identify false positives ───────────────────────────────────────────────
false_positives = []
n_images = 0
n_total_detections = 0
n_false_positives = 0

for r in results:
    n_images += 1
    stem = Path(r.path).stem

    gt_boxes = load_gt_boxes_for_image(stem)
    boxes = r.boxes

    if boxes is None or len(boxes) == 0:
        continue

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        n_total_detections += 1

        det_box = [x1, y1, x2, y2]
        best_iou = max([iou(det_box, gt) for gt in gt_boxes], default=0.0)

        if best_iou < IOU_THRESHOLD:
            # This is a false positive
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            false_positives.append({
                "filename": stem,
                "confidence": round(conf, 4),
                "x1": round(x1, 2), "y1": round(y1, 2),
                "x2": round(x2, 2), "y2": round(y2, 2),
                "cx": round(cx, 2), "cy": round(cy, 2),
            })
            n_false_positives += 1

    if n_images % 1000 == 0:
        print(f"  Processed {n_images} images, found {n_false_positives} false positives so far...")

# ── Save results ─────────────────────────────────────────────────────────────
print(f"\nSaving {n_false_positives} false positives to {OUTPUT_CSV}")

with open(OUTPUT_CSV, "w", newline="") as f:
    fieldnames = ["filename", "confidence", "x1", "y1", "x2", "y2", "cx", "cy"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(false_positives)

print("Done.")
print(f"Total training images processed : {n_images}")
print(f"Total detections                : {n_total_detections}")
print(f"Total false positives found     : {n_false_positives}")

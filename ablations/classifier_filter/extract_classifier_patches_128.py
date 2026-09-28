import cv2
import numpy as np
from pathlib import Path
import random

BASE_DIR      = Path(".")
DATASET_DIR   = BASE_DIR / "dataset"
OUTPUT_DIR    = BASE_DIR / "classifier_dataset_128"
PATCH_SIZE    = 128
SEARCH_RADIUS = 150
MAX_ATTEMPTS  = 30
SPLITS        = ["train", "val", "test"]

random.seed(42)

def read_yolo_labels(label_path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, w, h = map(float, parts)
                boxes.append((cx, cy, w, h))
    return boxes

def yolo_to_pixel_box(box, img_w, img_h):
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2

def boxes_overlap(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    return x2 > x1 and y2 > y1

def extract_patch(img, cx, cy, patch_size):
    h, w = img.shape[:2]
    half = patch_size // 2
    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > w: x1 -= (x2 - w); x2 = w
    if y2 > h: y1 -= (y2 - h); y2 = h
    x1, y1 = max(0, x1), max(0, y1)
    return img[y1:y2, x1:x2]

def find_healthy_patch(img, tumour_boxes_pixel, img_w, img_h):
    anchor = random.choice(tumour_boxes_pixel)
    anchor_cx = (anchor[0] + anchor[2]) / 2
    anchor_cy = (anchor[1] + anchor[3]) / 2
    half = PATCH_SIZE // 2
    for _ in range(MAX_ATTEMPTS):
        angle = random.uniform(0, 2 * np.pi)
        dist  = random.uniform(PATCH_SIZE, SEARCH_RADIUS)
        cx = anchor_cx + dist * np.cos(angle)
        cy = anchor_cy + dist * np.sin(angle)
        if cx - half < 0 or cx + half > img_w or cy - half < 0 or cy + half > img_h:
            continue
        candidate_box = [cx - half, cy - half, cx + half, cy + half]
        if any(boxes_overlap(candidate_box, tb) for tb in tumour_boxes_pixel):
            continue
        return cx, cy
    return None, None

total_tumour  = 0
total_healthy = 0
total_skipped = 0

for split in SPLITS:
    img_dir = DATASET_DIR / split / "images"
    lbl_dir = DATASET_DIR / split / "labels"

    out_tumour_dir  = OUTPUT_DIR / split / "tumour"
    out_healthy_dir = OUTPUT_DIR / split / "healthy"
    out_tumour_dir.mkdir(parents=True, exist_ok=True)
    out_healthy_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(img_dir.glob("*.png"))
    print(f"[{split}] Processing {len(image_files)} images...")

    for img_path in image_files:
        stem     = img_path.stem
        lbl_path = lbl_dir / (stem + ".txt")

        boxes_norm = read_yolo_labels(lbl_path)
        if not boxes_norm:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        boxes_pixel  = [yolo_to_pixel_box(b, img_w, img_h) for b in boxes_norm]

        for i, box in enumerate(boxes_pixel):
            tx1, ty1, tx2, ty2 = box
            tcx = (tx1 + tx2) / 2
            tcy = (ty1 + ty2) / 2

            tumour_patch = extract_patch(img, tcx, tcy, PATCH_SIZE)
            if tumour_patch.shape[0] != PATCH_SIZE or tumour_patch.shape[1] != PATCH_SIZE:
                total_skipped += 1
                continue

            cv2.imwrite(str(out_tumour_dir / f"{stem}_t{i:02d}.png"), tumour_patch)
            total_tumour += 1

            hcx, hcy = find_healthy_patch(img, boxes_pixel, img_w, img_h)
            if hcx is None:
                total_skipped += 1
                continue

            healthy_patch = extract_patch(img, hcx, hcy, PATCH_SIZE)
            if healthy_patch.shape[0] != PATCH_SIZE or healthy_patch.shape[1] != PATCH_SIZE:
                total_skipped += 1
                continue

            cv2.imwrite(str(out_healthy_dir / f"{stem}_h{i:02d}.png"), healthy_patch)
            total_healthy += 1

    print(f"[{split}] Done.")

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"Total tumour patches:  {total_tumour}")
print(f"Total healthy patches: {total_healthy}")
print(f"Total skipped:         {total_skipped}")
print(f"Output: {OUTPUT_DIR}")
print(f"{'='*60}")

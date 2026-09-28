"""
Combined Filtering: Slice Post-Processing + Classifier
=========================================================
Takes the slice-post-processed detections (Rule A, B, C) and runs each
through the trained classifier for a second filtering pass.

This tests whether combining both approaches compounds the benefit.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import csv
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR         = Path(".")
POSTPROC_DIR     = BASE_DIR / "runs" / "slice_postprocessing"
TEST_IMAGES_DIR  = BASE_DIR / "dataset" / "test" / "images"
TEST_LABELS_DIR  = BASE_DIR / "dataset" / "test" / "labels"
CLASSIFIER_PATH  = BASE_DIR / "runs" / "classifier" / "best_classifier.pt"
OUTPUT_DIR       = BASE_DIR / "runs" / "combined_filtering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE     = 64
IMG_SIZE       = 512
IOU_THRESHOLD  = 0.30
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

print(f"Using device: {DEVICE}")

# ── Load classifier ─────────────────────────────────────────────────────────
print("Loading classifier...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


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


def classify_detections(det_list, image_cache):
    kept, removed = [], []
    with torch.no_grad():
        for det in det_list:
            filename = det["filename"]
            if filename not in image_cache:
                img_path = TEST_IMAGES_DIR / f"{filename}.png"
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_cache[filename] = img
            img = image_cache[filename]

            cx, cy = float(det["cx"]), float(det["cy"])
            patch = extract_patch(img, cx, cy, PATCH_SIZE)
            patch_tensor = transform(patch).unsqueeze(0).to(DEVICE)
            output = model(patch_tensor)
            pred = output.argmax(dim=1).item()

            if pred == 1:
                kept.append(det)
            else:
                removed.append(det)
    return kept, removed


# ── Ground truth + evaluation helpers ──────────────────────────────────────
def load_gt_boxes():
    gt = {}
    for label_file in TEST_LABELS_DIR.glob("*.txt"):
        stem = label_file.stem
        boxes = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, w, h = map(float, parts)
                    x1 = (cx - w / 2) * IMG_SIZE
                    y1 = (cy - h / 2) * IMG_SIZE
                    x2 = (cx + w / 2) * IMG_SIZE
                    y2 = (cy + h / 2) * IMG_SIZE
                    boxes.append([x1, y1, x2, y2])
        gt[stem] = boxes
    for img_file in TEST_IMAGES_DIR.glob("*.png"):
        if img_file.stem not in gt:
            gt[img_file.stem] = []
    return gt


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


def evaluate(det_list, gt_boxes):
    by_image = defaultdict(list)
    for d in det_list:
        by_image[d["filename"]].append([float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])])

    tp, fp, fn = 0, 0, 0
    all_stems = set(gt_boxes.keys()) | set(by_image.keys())

    for stem in all_stems:
        gts  = list(gt_boxes.get(stem, []))
        dets = list(by_image.get(stem, []))
        matched_gt = set()
        for det_box in dets:
            best_iou, best_idx = 0.0, -1
            for j, gt_box in enumerate(gts):
                if j in matched_gt:
                    continue
                iou_val = iou(det_box, gt_box)
                if iou_val > best_iou:
                    best_iou, best_idx = iou_val, j
            if best_iou >= IOU_THRESHOLD:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn += len(gts) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


# ── Main ────────────────────────────────────────────────────────────────────
print("Loading ground truth...")
gt_boxes = load_gt_boxes()

configs = {
    "Rule A": POSTPROC_DIR / "filtered_rule_a.csv",
    "Rule B": POSTPROC_DIR / "filtered_rule_b.csv",
    "Rule C": POSTPROC_DIR / "filtered_rule_c.csv",
}

image_cache = {}
results_summary = []

print(f"\n{'Config':<35}{'Precision':>10}{'Recall':>10}{'F1':>10}{'TP':>6}{'FP':>6}{'FN':>6}")
print("-" * 85)

for name, path in configs.items():
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        det_list = list(reader)

    # Stage before classifier
    metrics_before = evaluate(det_list, gt_boxes)
    print(f"{name + ' alone':<35}{metrics_before['precision']:>10}{metrics_before['recall']:>10}"
          f"{metrics_before['f1']:>10}{metrics_before['tp']:>6}{metrics_before['fp']:>6}{metrics_before['fn']:>6}")
    results_summary.append((f"{name} alone", metrics_before))

    # Apply classifier on top
    kept, removed = classify_detections(det_list, image_cache)
    metrics_after = evaluate(kept, gt_boxes)
    print(f"{name + ' + Classifier':<35}{metrics_after['precision']:>10}{metrics_after['recall']:>10}"
          f"{metrics_after['f1']:>10}{metrics_after['tp']:>6}{metrics_after['fp']:>6}{metrics_after['fn']:>6}")
    results_summary.append((f"{name} + Classifier", metrics_after))

    # Save combined filtered set
    out_path = OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_plus_classifier.csv"
    if kept:
        fieldnames = list(kept[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)

# ── Save summary ─────────────────────────────────────────────────────────
summary_path = OUTPUT_DIR / "combined_evaluation_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["config", "precision", "recall", "f1", "tp", "fp", "fn"])
    for name, m in results_summary:
        writer.writerow([name, m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"]])

print(f"\nSummary saved to {summary_path}")
print("Done.")

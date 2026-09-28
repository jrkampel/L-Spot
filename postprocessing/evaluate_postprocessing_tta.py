"""
Evaluate Slice-to-Slice Post-Processing Results
=================================================
Compares each filtered detection set (Rule A, B, C) and the unfiltered
baseline against ground truth YOLO labels on the test set.

Computes Precision, Recall, F1 at the detection level using IoU matching.
"""

import csv
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(".")
RAW_CSV         = BASE_DIR / "runs" / "raw_detections_tta.csv"
POSTPROC_DIR    = BASE_DIR / "runs" / "slice_postprocessing_tta"
TEST_LABELS_DIR = BASE_DIR / "dataset" / "test" / "labels"
TEST_IMAGES_DIR = BASE_DIR / "dataset" / "test" / "images"
IMG_SIZE        = 512
IOU_THRESHOLD   = 0.30   # detection counts as a true positive if IoU with GT >= this

# ── Load ground truth boxes ─────────────────────────────────────────────────
def load_gt_boxes():
    """Returns dict: filename_stem -> list of [x1, y1, x2, y2] in pixel coords"""
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

    # Also register images with no label file as having zero GT boxes
    for img_file in TEST_IMAGES_DIR.glob("*.png"):
        if img_file.stem not in gt:
            gt[img_file.stem] = []

    return gt


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def load_detections(csv_path):
    """Returns dict: filename_stem -> list of [x1, y1, x2, y2]"""
    dets = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dets[row["filename"]].append([
                float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"])
            ])
    return dets


def evaluate(detections, gt_boxes):
    """
    Match detections to GT boxes per image using greedy IoU matching.
    Returns precision, recall, f1, tp, fp, fn.
    """
    tp, fp, fn = 0, 0, 0

    all_image_stems = set(gt_boxes.keys()) | set(detections.keys())

    for stem in all_image_stems:
        gts  = list(gt_boxes.get(stem, []))
        dets = list(detections.get(stem, []))

        matched_gt = set()

        for det_box in dets:
            best_iou = 0.0
            best_idx = -1
            for i, gt_box in enumerate(gts):
                if i in matched_gt:
                    continue
                iou_val = iou(det_box, gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i

            if best_iou >= IOU_THRESHOLD:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn += len(gts) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn
    }


# ── Main ────────────────────────────────────────────────────────────────────
print("Loading ground truth boxes...")
gt_boxes = load_gt_boxes()
print(f"Loaded GT for {len(gt_boxes)} images")

configs = {
    "Unfiltered (TTA baseline)": RAW_CSV,
    "Rule A (>=2 consecutive)": POSTPROC_DIR / "filtered_rule_a.csv",
    "Rule B (>=3 consecutive)": POSTPROC_DIR / "filtered_rule_b.csv",
    "Rule C (conf>0.5 OR >=2)": POSTPROC_DIR / "filtered_rule_c.csv",
}

print(f"\n{'Config':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
print("-" * 80)

results_summary = []
for name, path in configs.items():
    dets = load_detections(path)
    metrics = evaluate(dets, gt_boxes)
    print(f"{name:<30} {metrics['precision']:>10} {metrics['recall']:>10} {metrics['f1']:>10} "
          f"{metrics['tp']:>6} {metrics['fp']:>6} {metrics['fn']:>6}")
    results_summary.append((name, metrics))

# Save summary to CSV
summary_path = POSTPROC_DIR / "evaluation_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["config", "precision", "recall", "f1", "tp", "fp", "fn"])
    for name, m in results_summary:
        writer.writerow([name, m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"]])

print(f"\nSummary saved to {summary_path}")

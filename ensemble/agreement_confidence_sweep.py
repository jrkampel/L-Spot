"""
Confidence-Gated Agreement Sweep
====================================
Tests whether requiring a MINIMUM confidence on both the standard and TTA
detection (not just box overlap) improves the "agreed" detection set.

Re-uses the agreement logic from ensemble_standard_tta.py but sweeps
different minimum confidence floors applied to both sides before matching.
"""

import csv
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR         = Path(".")
STANDARD_CSV     = BASE_DIR / "runs" / "raw_detections.csv"
TTA_CSV          = BASE_DIR / "runs" / "raw_detections_tta.csv"
TEST_LABELS_DIR  = BASE_DIR / "dataset" / "test" / "labels"
TEST_IMAGES_DIR  = BASE_DIR / "dataset" / "test" / "images"
OUTPUT_DIR       = BASE_DIR / "runs" / "agreement_confidence_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE              = 512
IOU_THRESHOLD         = 0.30
AGREEMENT_IOU_THRESH  = 0.30
TTA_CONF_FLOOR        = 0.30   # our previously tuned TTA floor, kept fixed

# Sweep: minimum confidence required on BOTH sides for a detection to count as "agreed"
MIN_CONF_GATES = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]

# ── Load detections ──────────────────────────────────────────────────────────
print("Loading standard detections...")
with open(STANDARD_CSV, "r") as f:
    standard_dets_all = list(csv.DictReader(f))

print("Loading TTA detections (filtering conf >= 0.30)...")
with open(TTA_CSV, "r") as f:
    tta_dets_all_raw = list(csv.DictReader(f))
tta_dets_all = [d for d in tta_dets_all_raw if float(d["confidence"]) >= TTA_CONF_FLOOR]

print(f"  Standard: {len(standard_dets_all)} | TTA: {len(tta_dets_all)}")

# ── Helper functions ─────────────────────────────────────────────────────────
def box_of(d):
    return [float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])]


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


def find_agreement(standard_dets, tta_dets):
    standard_by_image = defaultdict(list)
    for d in standard_dets:
        standard_by_image[d["filename"]].append(d)

    tta_by_image = defaultdict(list)
    for d in tta_dets:
        tta_by_image[d["filename"]].append(d)

    agreed = []
    all_filenames = set(standard_by_image.keys()) | set(tta_by_image.keys())

    for filename in all_filenames:
        std_list = standard_by_image.get(filename, [])
        tta_list = tta_by_image.get(filename, [])
        matched_tta_idx = set()

        for std_d in std_list:
            std_box = box_of(std_d)
            for j, tta_d in enumerate(tta_list):
                if j in matched_tta_idx:
                    continue
                tta_box = box_of(tta_d)
                if iou(std_box, tta_box) >= AGREEMENT_IOU_THRESH:
                    agreed.append(tta_d)
                    matched_tta_idx.add(j)
                    break

    return agreed


# ── Ground truth + evaluation ──────────────────────────────────────────────
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


def evaluate(det_list, gt_boxes):
    by_image = defaultdict(list)
    for d in det_list:
        by_image[d["filename"]].append(box_of(d))

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


# ── Sweep ────────────────────────────────────────────────────────────────────
print("Loading ground truth...")
gt_boxes = load_gt_boxes()

print(f"\n{'MinConf':<10}{'N_agreed':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}{'TP':>6}{'FP':>6}{'FN':>6}")
print("-" * 80)

results_summary = []

for gate in MIN_CONF_GATES:
    std_filtered = [d for d in standard_dets_all if float(d["confidence"]) >= gate]
    tta_filtered = [d for d in tta_dets_all if float(d["confidence"]) >= gate]

    agreed = find_agreement(std_filtered, tta_filtered)
    metrics = evaluate(agreed, gt_boxes)

    print(f"{gate:<10}{len(agreed):<12}{metrics['precision']:>10}{metrics['recall']:>10}"
          f"{metrics['f1']:>10}{metrics['tp']:>6}{metrics['fp']:>6}{metrics['fn']:>6}")

    results_summary.append((gate, len(agreed), metrics))

# ── Save summary ─────────────────────────────────────────────────────────
summary_path = OUTPUT_DIR / "confidence_gate_sweep_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["min_confidence_gate", "n_agreed", "precision", "recall", "f1", "tp", "fp", "fn"])
    for gate, n, m in results_summary:
        writer.writerow([gate, n, m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"]])

print(f"\nSummary saved to {summary_path}")
print("Done.")

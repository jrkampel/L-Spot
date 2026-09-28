"""
TTA Confidence Threshold Sweep
=================================
Tests multiple confidence thresholds on the raw TTA detections BEFORE
applying slice post-processing, to see if stripping TTA's weakest
detections at the source improves the precision/recall/F1 trade-off.

For each threshold, also applies Rule B slice post-processing on top,
to test the full combined pipeline.
"""

import csv
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR         = Path(".")
RAW_TTA_CSV      = BASE_DIR / "runs" / "raw_detections_tta.csv"
TEST_LABELS_DIR  = BASE_DIR / "dataset" / "test" / "labels"
TEST_IMAGES_DIR  = BASE_DIR / "dataset" / "test" / "images"
OUTPUT_DIR       = BASE_DIR / "runs" / "tta_threshold_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE       = 512
IOU_THRESHOLD  = 0.30
CENTROID_DIST_THRESH = 30.0

CONF_THRESHOLDS_TO_TEST = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# ── Load raw TTA detections ─────────────────────────────────────────────────
print(f"Loading TTA detections from {RAW_TTA_CSV}")
with open(RAW_TTA_CSV, "r") as f:
    reader = csv.DictReader(f)
    all_detections = list(reader)

print(f"Loaded {len(all_detections)} TTA detections")

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


# ── Slice post-processing (Rule B) ──────────────────────────────────────────
def centroid_distance(d1, d2):
    return ((float(d1["cx"]) - float(d2["cx"])) ** 2 + (float(d1["cy"]) - float(d2["cy"])) ** 2) ** 0.5


def apply_rule_b(det_list):
    by_patient = defaultdict(list)
    for d in det_list:
        by_patient[d["patient_id"]].append(d)
    for pid in by_patient:
        by_patient[pid].sort(key=lambda d: int(d["slice_num"]))

    kept = []
    for pid, dets in by_patient.items():
        slice_lookup = defaultdict(list)
        for d in dets:
            slice_lookup[int(d["slice_num"])].append(d)

        for d in dets:
            slice_num = int(d["slice_num"])

            def has_match(target_slice):
                for cand in slice_lookup.get(target_slice, []):
                    if centroid_distance(d, cand) <= CENTROID_DIST_THRESH:
                        return True
                return False

            run = 1
            s = slice_num + 1
            while has_match(s):
                run += 1
                s += 1
            s = slice_num - 1
            while has_match(s):
                run += 1
                s -= 1

            if run >= 3:   # Rule B
                kept.append(d)

    return kept


# ── Sweep ────────────────────────────────────────────────────────────────────
print("Loading ground truth...")
gt_boxes = load_gt_boxes()

print(f"\n{'Conf>=':<10}{'N_dets':<10}{'Raw Prec':>10}{'Raw Rec':>10}{'Raw F1':>10}"
      f"{'+RuleB Prec':>13}{'+RuleB Rec':>12}{'+RuleB F1':>11}")
print("-" * 100)

results_summary = []

for thresh in CONF_THRESHOLDS_TO_TEST:
    filtered = [d for d in all_detections if float(d["confidence"]) >= thresh]
    raw_metrics = evaluate(filtered, gt_boxes)

    ruleb_filtered = apply_rule_b(filtered)
    ruleb_metrics = evaluate(ruleb_filtered, gt_boxes)

    print(f"{thresh:<10}{len(filtered):<10}{raw_metrics['precision']:>10}{raw_metrics['recall']:>10}"
          f"{raw_metrics['f1']:>10}{ruleb_metrics['precision']:>13}{ruleb_metrics['recall']:>12}"
          f"{ruleb_metrics['f1']:>11}")

    results_summary.append((thresh, len(filtered), raw_metrics, ruleb_metrics))

# ── Save summary ─────────────────────────────────────────────────────────
summary_path = OUTPUT_DIR / "tta_threshold_sweep_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["conf_threshold", "n_detections", "raw_precision", "raw_recall", "raw_f1",
                      "ruleb_precision", "ruleb_recall", "ruleb_f1"])
    for thresh, n, raw_m, ruleb_m in results_summary:
        writer.writerow([thresh, n, raw_m["precision"], raw_m["recall"], raw_m["f1"],
                          ruleb_m["precision"], ruleb_m["recall"], ruleb_m["f1"]])

print(f"\nSummary saved to {summary_path}")
print("Done.")

"""
Three-Way Ensemble: Standard YOLOv8m + TTA YOLOv8m + YOLOv10m
==================================================================
Extends the two-way agreement ensemble by adding a third, independently
trained model (YOLOv10m) as an additional vote.

A detection is kept if it is confirmed by AT LEAST 2 of the 3 sources
(matched by IoU overlap with each other).

This tests whether genuinely independent model architecture diversity
recovers additional true positives beyond what two views of the same
model (standard + TTA) can achieve.
"""

import csv
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR         = Path(".")
STANDARD_CSV     = BASE_DIR / "runs" / "raw_detections.csv"
TTA_CSV          = BASE_DIR / "runs" / "raw_detections_tta.csv"
YOLO10_CSV       = BASE_DIR / "runs" / "raw_detections_yolo10.csv"
TEST_LABELS_DIR  = BASE_DIR / "dataset" / "test" / "labels"
TEST_IMAGES_DIR  = BASE_DIR / "dataset" / "test" / "images"
OUTPUT_DIR       = BASE_DIR / "runs" / "three_way_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE              = 512
IOU_THRESHOLD         = 0.30
AGREEMENT_IOU_THRESH  = 0.30
TTA_CONF_FLOOR        = 0.30   # tuned TTA threshold from earlier

# ── Load detections ──────────────────────────────────────────────────────────
print("Loading detection sources...")

with open(STANDARD_CSV, "r") as f:
    standard_dets = list(csv.DictReader(f))
for d in standard_dets:
    d["source"] = "standard"

with open(TTA_CSV, "r") as f:
    tta_dets_raw = list(csv.DictReader(f))
tta_dets = [d for d in tta_dets_raw if float(d["confidence"]) >= TTA_CONF_FLOOR]
for d in tta_dets:
    d["source"] = "tta"

with open(YOLO10_CSV, "r") as f:
    yolo10_dets = list(csv.DictReader(f))
for d in yolo10_dets:
    d["source"] = "yolo10"

print(f"  Standard: {len(standard_dets)}")
print(f"  TTA (conf>=0.30): {len(tta_dets)}")
print(f"  YOLOv10m: {len(yolo10_dets)}")

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


# ── Cluster detections per image across all 3 sources ──────────────────────
def by_image(dets):
    d = defaultdict(list)
    for det in dets:
        d[det["filename"]].append(det)
    return d


standard_by_img = by_image(standard_dets)
tta_by_img      = by_image(tta_dets)
yolo10_by_img   = by_image(yolo10_dets)

all_filenames = set(standard_by_img.keys()) | set(tta_by_img.keys()) | set(yolo10_by_img.keys())

print(f"\nClustering detections across {len(all_filenames)} images...")

two_of_three = []
three_of_three = []

for filename in all_filenames:
    std_list    = standard_by_img.get(filename, [])
    tta_list    = tta_by_img.get(filename, [])
    yolo10_list = yolo10_by_img.get(filename, [])

    # Build a pool of all detections in this image, tagged by source
    pool = [(d, "standard") for d in std_list] + \
           [(d, "tta") for d in tta_list] + \
           [(d, "yolo10") for d in yolo10_list]

    used = [False] * len(pool)

    for i, (det_i, src_i) in enumerate(pool):
        if used[i]:
            continue
        box_i = box_of(det_i)
        cluster = [(det_i, src_i)]
        used[i] = True

        for j in range(i + 1, len(pool)):
            if used[j]:
                continue
            det_j, src_j = pool[j]
            if src_j == src_i:
                continue  # don't match within the same source
            box_j = box_of(det_j)
            if iou(box_i, box_j) >= AGREEMENT_IOU_THRESH:
                cluster.append((det_j, src_j))
                used[j] = True

        sources_in_cluster = set(s for _, s in cluster)
        n_sources = len(sources_in_cluster)

        # Use the detection with highest confidence in the cluster as representative
        best_det = max(cluster, key=lambda x: float(x[0]["confidence"]))[0]

        if n_sources >= 2:
            two_of_three.append(best_det)
        if n_sources >= 3:
            three_of_three.append(best_det)

print(f"Detections confirmed by >=2 sources : {len(two_of_three)}")
print(f"Detections confirmed by all 3 sources: {len(three_of_three)}")

# ── Evaluation ───────────────────────────────────────────────────────────────
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
    by_image_dets = defaultdict(list)
    for d in det_list:
        by_image_dets[d["filename"]].append(box_of(d))

    tp, fp, fn = 0, 0, 0
    all_stems = set(gt_boxes.keys()) | set(by_image_dets.keys())

    for stem in all_stems:
        gts  = list(gt_boxes.get(stem, []))
        dets = list(by_image_dets.get(stem, []))
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


print("\nLoading ground truth and evaluating...")
gt_boxes = load_gt_boxes()

metrics_2of3 = evaluate(two_of_three, gt_boxes)
metrics_3of3 = evaluate(three_of_three, gt_boxes)

print(f"\n{'Config':<35}{'Precision':>10}{'Recall':>10}{'F1':>10}{'TP':>6}{'FP':>6}{'FN':>6}")
print("-" * 85)
print(f"{'>=2 of 3 sources agree':<35}{metrics_2of3['precision']:>10}{metrics_2of3['recall']:>10}"
      f"{metrics_2of3['f1']:>10}{metrics_2of3['tp']:>6}{metrics_2of3['fp']:>6}{metrics_2of3['fn']:>6}")
print(f"{'All 3 sources agree':<35}{metrics_3of3['precision']:>10}{metrics_3of3['recall']:>10}"
      f"{metrics_3of3['f1']:>10}{metrics_3of3['tp']:>6}{metrics_3of3['fp']:>6}{metrics_3of3['fn']:>6}")

# ── Save results ─────────────────────────────────────────────────────────────
summary_path = OUTPUT_DIR / "three_way_evaluation_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["config", "precision", "recall", "f1", "tp", "fp", "fn"])
    writer.writerow([">=2 of 3 agree", metrics_2of3["precision"], metrics_2of3["recall"],
                      metrics_2of3["f1"], metrics_2of3["tp"], metrics_2of3["fp"], metrics_2of3["fn"]])
    writer.writerow(["All 3 agree", metrics_3of3["precision"], metrics_3of3["recall"],
                      metrics_3of3["f1"], metrics_3of3["tp"], metrics_3of3["fp"], metrics_3of3["fn"]])

print(f"\nResults saved to {OUTPUT_DIR}")
print("Done.")

# ── Save all-3-agree detections CSV (needed for mAP50 computation) ───────────
all3_path = OUTPUT_DIR / "ensemble_all3_detections.csv"
fieldnames = ["filename", "patient_id", "phase", "slice_num", "class", "confidence", "x1", "y1", "x2", "y2", "cx", "cy"]
with open(all3_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(three_of_three)
print(f"All-3-agree detections saved to {all3_path}")

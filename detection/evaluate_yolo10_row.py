"""
Evaluate the standalone YOLOv10m detection source under the DETECTION PROTOCOL,
to fill the "YOLOv10m unfiltered" row of the detection-protocol table
(confidence >= 0.10, IoU >= 0.30).

Matching / GT logic is copied verbatim from three_way_ensemble.py so this row is
directly comparable to the YOLOv8m row (P=0.566, R=0.674, F1=0.615, TP 781/FP 599/FN 378).

mAP50 is added here (three_way_ensemble.py did not compute it) using the standard
single-IoU-threshold COCO-style method: rank detections by confidence, greedily
match at IoU >= 0.50... NOTE: your detection-protocol tables report mAP50 at the
detection-protocol IoU. The YOLOv8m row shows mAP50 = 0.437 at IoU>=0.30, so mAP50
here is computed at IoU >= 0.30 to match that column. See MAP_IOU below.
"""

import csv
from pathlib import Path
from collections import defaultdict

# ── Config (mirrors three_way_ensemble.py) ──────────────────────────────────
BASE_DIR        = Path(".")
YOLO10_CSV      = BASE_DIR / "runs" / "raw_detections_yolo10.csv"
TEST_LABELS_DIR = BASE_DIR / "dataset" / "test" / "labels"
TEST_IMAGES_DIR = BASE_DIR / "dataset" / "test" / "images"

IMG_SIZE      = 512
IOU_THRESHOLD = 0.30    # for TP/FP/FN, same as three_way_ensemble.py
MAP_IOU       = 0.30    # matches the 0.437 mAP50 reported for YOLOv8m in this table
CONF_FLOOR    = 0.10    # detection-protocol extraction floor


# ── Helpers (copied from three_way_ensemble.py) ─────────────────────────────
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
    """Identical TP/FP/FN logic to three_way_ensemble.py."""
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


def compute_map50(det_list, gt_boxes, map_iou=MAP_IOU):
    """
    Single-threshold average precision (area under the PR curve) at IoU>=map_iou.
    Ranks all detections by confidence, greedily matches each to an unused GT.
    Uses the continuous (all-points) PR integration, i.e. sum of P*dR.
    """
    # Sort detections globally by confidence, descending
    dets = sorted(det_list, key=lambda d: float(d["confidence"]), reverse=True)

    total_gt = sum(len(v) for v in gt_boxes.values())
    if total_gt == 0:
        return 0.0

    matched = defaultdict(set)   # stem -> set of matched GT indices
    tp_flags, fp_flags = [], []

    for d in dets:
        stem = d["filename"]
        det_box = box_of(d)
        gts = gt_boxes.get(stem, [])
        best_iou, best_idx = 0.0, -1
        for j, gt_box in enumerate(gts):
            if j in matched[stem]:
                continue
            iou_val = iou(det_box, gt_box)
            if iou_val > best_iou:
                best_iou, best_idx = iou_val, j
        if best_iou >= map_iou and best_idx >= 0:
            tp_flags.append(1); fp_flags.append(0)
            matched[stem].add(best_idx)
        else:
            tp_flags.append(0); fp_flags.append(1)

    # Cumulative precision / recall
    cum_tp = cum_fp = 0
    precisions, recalls = [], []
    for t, fpf in zip(tp_flags, fp_flags):
        cum_tp += t; cum_fp += fpf
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / total_gt)

    # All-points AP: integrate precision over recall
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r
    return round(ap, 3)


# ── Run ─────────────────────────────────────────────────────────────────────
print("Loading YOLOv10m detections...")
with open(YOLO10_CSV, "r") as f:
    yolo10_dets = list(csv.DictReader(f))

n_total = len(yolo10_dets)
yolo10_dets = [d for d in yolo10_dets if float(d["confidence"]) >= CONF_FLOOR]
print(f"  Total rows in CSV      : {n_total}")
print(f"  After conf >= {CONF_FLOOR}      : {len(yolo10_dets)}")

print("Loading ground truth...")
gt_boxes = load_gt_boxes()
print(f"  Test slices (GT keys)  : {len(gt_boxes)}")
print(f"  Total GT tumour boxes  : {sum(len(v) for v in gt_boxes.values())}")

m = evaluate(yolo10_dets, gt_boxes)
map50 = compute_map50(yolo10_dets, gt_boxes)

print("\n" + "=" * 78)
print("YOLOv10m unfiltered  (conf >= 0.10, IoU >= 0.30)  --- detection-protocol row")
print("=" * 78)
print(f"{'':<12}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
print(f"{'YOLOv10m':<12}{m['precision']:>8}{m['recall']:>8}{m['f1']:>8}"
      f"{map50:>8}{m['tp']:>7}{m['fp']:>7}{m['fn']:>7}")
print("=" * 78)
print("\nLaTeX row (fill into tab:detection_protocol_baseline):")
print(f"YOLOv10m unfiltered & {m['precision']:.3f} & {m['recall']:.3f} & {m['f1']:.3f} "
      f"& {map50:.3f} & {m['tp']} & {m['fp']} & {m['fn']} \\\\")
print("\nNOTE: mAP50 here is computed by this script, not by your compute_map50.py.")
print("If you want it guaranteed identical to your other mAP50 numbers, run")
print("compute_map50.py on runs/raw_detections_yolo10.csv and use that value instead.")

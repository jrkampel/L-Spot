"""
Compute mAP50 for all pipeline stages.
mAP50 = area under PR curve at IoU threshold 0.50, averaged across all test images.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

IMG_SIZE = 512
IOU_THRESHOLD = 0.50
LABEL_DIR = "./dataset/test/labels"

# ── helper functions ──────────────────────────────────────────────────────────

def load_ground_truth():
    """Load all GT boxes from test label files. Returns dict: filename -> list of [x1,y1,x2,y2]"""
    gt = defaultdict(list)
    for fname in os.listdir(LABEL_DIR):
        if not fname.endswith(".txt"):
            continue
        stem = fname.replace(".txt", "")
        with open(os.path.join(LABEL_DIR, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                x1 = (cx - w / 2) * IMG_SIZE
                y1 = (cy - h / 2) * IMG_SIZE
                x2 = (cx + w / 2) * IMG_SIZE
                y2 = (cy + h / 2) * IMG_SIZE
                gt[stem].append([x1, y1, x2, y2])
    return gt


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def compute_map50(detections_df, gt, conf_col="confidence"):
    """
    detections_df: DataFrame with columns filename, x1, y1, x2, y2, confidence
    gt: dict filename -> list of GT boxes
    Returns mAP50 (float)
    """
    # Sort all detections by confidence descending
    df = detections_df.copy()
    df = df.sort_values(conf_col, ascending=False).reset_index(drop=True)

    total_gt = sum(len(v) for v in gt.values())
    if total_gt == 0:
        return 0.0

    # Track which GT boxes have been matched
    gt_matched = defaultdict(lambda: defaultdict(bool))  # filename -> gt_idx -> matched

    tp_list = []
    fp_list = []

    for _, row in df.iterrows():
        fname = row["filename"]
        det_box = [row["x1"], row["y1"], row["x2"], row["y2"]]
        gt_boxes = gt.get(fname, [])

        best_iou = 0.0
        best_idx = -1
        for i, gt_box in enumerate(gt_boxes):
            s = iou(det_box, gt_box)
            if s > best_iou:
                best_iou = s
                best_idx = i

        if best_iou >= IOU_THRESHOLD and not gt_matched[fname][best_idx]:
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[fname][best_idx] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    recalls = tp_cum / total_gt
    precisions = tp_cum / (tp_cum + fp_cum)

    # Compute area under PR curve using 101-point interpolation (COCO style)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec_at_rec = precisions[recalls >= t]
        ap += (np.max(prec_at_rec) if len(prec_at_rec) > 0 else 0.0)
    ap /= 101

    return round(ap, 3)


def load_csv(path, conf_col="confidence"):
    df = pd.read_csv(path)
    # Normalise filename column (strip extension if present)
    df["filename"] = df["filename"].str.replace(r"\.(png|jpg|jpeg)$", "", regex=True)
    return df


# ── load ground truth once ────────────────────────────────────────────────────

print("Loading ground truth...")
gt = load_ground_truth()
print(f"  {len(gt)} test images with GT boxes, {sum(len(v) for v in gt.values())} total GT boxes\n")

# ── define all pipeline stages ────────────────────────────────────────────────

BASE    = "./runs"
SP      = f"{BASE}/slice_postprocessing"
SP_TTA  = f"{BASE}/slice_postprocessing_tta"
ENS     = f"{BASE}/ensemble_standard_tta"
ENS3    = f"{BASE}/three_way_ensemble"

stages = [
    # (label, csv_path, conf_col)
    ("Unfiltered YOLOv8m baseline",         f"{BASE}/raw_detections.csv",                    "confidence"),
    ("Rule A (>=2 consecutive slices)",     f"{SP}/filtered_rule_a.csv",                     "confidence"),
    ("Rule B (>=3 consecutive slices)",     f"{SP}/filtered_rule_b.csv",                     "confidence"),
    ("Rule C (conf>0.5 OR >=2 slices)",     f"{SP}/filtered_rule_c.csv",                     "confidence"),
    ("TTA + Rule B",                        f"{SP_TTA}/filtered_rule_b.csv",                 "confidence"),
    ("Standard AND TTA agreement",          f"{ENS}/final_ensemble_detections.csv",          "confidence"),
    ("L-Spot (>=2 of 3 agree)",             f"{ENS3}/ensemble_2of3_detections.csv",          "confidence"),
]

# ── compute mAP50 for each stage ──────────────────────────────────────────────

print(f"{'Stage':<45} {'mAP50':>7}")
print("-" * 54)

results = {}
for label, path, conf_col in stages:
    if not os.path.exists(path):
        print(f"{label:<45} {'FILE NOT FOUND':>7}")
        continue
    df = load_csv(path, conf_col)
    if conf_col not in df.columns:
        # Try fallback
        alt = [c for c in df.columns if "conf" in c.lower()]
        if alt:
            conf_col = alt[0]
        else:
            print(f"{label:<45} {'NO CONF COL':>7}")
            continue
    ap = compute_map50(df, gt, conf_col)
    results[label] = ap
    print(f"{label:<45} {ap:>7.3f}")

print("\nDone. Use these mAP50 values to update your LaTeX tables.")

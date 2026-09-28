"""
Unified Evaluation Across All Pipeline Stages
===============================================
For every serialised detection CSV, computes both metric families used in the
thesis, so that a single table can report them side by side:

  - Precision / Recall / F1  : IoU >= 0.30, greedy matching, detections taken
                               in file order (reproduces evaluate_postprocessing.py
                               and three_way_ensemble.py exactly)
  - mAP50                    : detections sorted by descending confidence,
                               matched at IoU 0.50, PR curve integrated by
                               101-point interpolation (COCO style)

Known stages are evaluated under fixed labels. Any other detection CSV found
under runs/ is auto-discovered and evaluated too, so nothing is silently missed.

Output: runs/all_stages_metrics.csv
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(".")
RUNS_DIR        = BASE_DIR / "runs"
TEST_LABELS_DIR = BASE_DIR / "dataset" / "test" / "labels"
TEST_IMAGES_DIR = BASE_DIR / "dataset" / "test" / "images"
OUTPUT_CSV      = RUNS_DIR / "all_stages_metrics.csv"

IMG_SIZE      = 512
PRF_IOU       = 0.30   # Section 3.1.4 matching threshold
MAP_IOU       = 0.50   # mAP50 is defined at 0.50

# (label, path relative to BASE_DIR, confidence floor or None)
STAGES = [
    ("YOLOv8m baseline",              "runs/raw_detections.csv",                                 None),
    ("YOLOv8m + TTA (conf>=0.10)",    "runs/raw_detections_tta.csv",                             None),
    ("YOLOv8m + TTA (conf>=0.30)",    "runs/raw_detections_tta.csv",                             0.30),
    ("YOLOv10m",                      "runs/raw_detections_yolo10.csv",                          None),
    ("Rule A (>=2 consecutive)",      "runs/slice_postprocessing/filtered_rule_a.csv",           None),
    ("Rule B (>=3 consecutive)",      "runs/slice_postprocessing/filtered_rule_b.csv",           None),
    ("Rule C (conf>0.5 OR >=2)",      "runs/slice_postprocessing/filtered_rule_c.csv",           None),
    ("TTA + Rule A",                  "runs/slice_postprocessing_tta/filtered_rule_a.csv",       None),
    ("TTA + Rule B",                  "runs/slice_postprocessing_tta/filtered_rule_b.csv",       None),
    ("TTA + Rule C",                  "runs/slice_postprocessing_tta/filtered_rule_c.csv",       None),
    ("Two-source: full ensemble",     "runs/ensemble_standard_tta/final_ensemble_detections.csv", None),
    ("L-Spot (>=2 of 3 agree)",       "runs/three_way_ensemble/ensemble_2of3_detections.csv",    None),
    ("L-Spot (all 3 agree)",          "runs/three_way_ensemble/ensemble_all3_detections.csv",    None),
    ("CNN+ViT hybrid filtered",       "runs/cnn_vit_filtering/hybrid_filtered_detections.csv",   None),
]

BOX_COLS = ["x1", "y1", "x2", "y2"]


# ── Ground truth ────────────────────────────────────────────────────────────
def load_gt_boxes():
    """filename_stem -> list of [x1, y1, x2, y2] in pixel coords."""
    gt = {}
    for label_file in TEST_LABELS_DIR.glob("*.txt"):
        boxes = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, w, h = map(float, parts)
                    boxes.append([
                        (cx - w / 2) * IMG_SIZE, (cy - h / 2) * IMG_SIZE,
                        (cx + w / 2) * IMG_SIZE, (cy + h / 2) * IMG_SIZE,
                    ])
        gt[label_file.stem] = boxes
    # images with no label file contribute zero GT boxes
    for img_file in TEST_IMAGES_DIR.glob("*.png"):
        gt.setdefault(img_file.stem, [])
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


# ── Loading ─────────────────────────────────────────────────────────────────
def load_detections(path, conf_floor=None):
    """Returns list of dicts: {stem, box, conf}, preserving file order."""
    dets = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return dets
        if not all(c in reader.fieldnames for c in BOX_COLS):
            return None  # not a detection CSV
        has_conf = "confidence" in reader.fieldnames
        for row in reader:
            try:
                conf = float(row["confidence"]) if has_conf else 1.0
            except (TypeError, ValueError):
                conf = 1.0
            if conf_floor is not None and conf < conf_floor:
                continue
            stem = row["filename"]
            for ext in (".png", ".jpg", ".jpeg"):
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
            try:
                box = [float(row[c]) for c in BOX_COLS]
            except (TypeError, ValueError):
                continue
            dets.append({"stem": stem, "box": box, "conf": conf})
    return dets


# ── Metric 1: P / R / F1 at IoU 0.30, file order, greedy ────────────────────
def eval_prf(dets, gt):
    by_image = defaultdict(list)
    for d in dets:
        by_image[d["stem"]].append(d["box"])

    tp = fp = fn = 0
    for stem in set(gt.keys()) | set(by_image.keys()):
        gts = gt.get(stem, [])
        matched = set()
        for det_box in by_image.get(stem, []):
            best_iou, best_idx = 0.0, -1
            for j, gt_box in enumerate(gts):
                if j in matched:
                    continue
                v = iou(det_box, gt_box)
                if v > best_iou:
                    best_iou, best_idx = v, j
            if best_iou >= PRF_IOU:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1
        fn += len(gts) - len(matched)

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp, fp, fn


# ── Metric 2: mAP50, confidence-sorted, 101-point interpolation ─────────────
def eval_map50(dets, gt):
    total_gt = sum(len(v) for v in gt.values())
    if total_gt == 0 or not dets:
        return 0.0

    ordered = sorted(dets, key=lambda d: d["conf"], reverse=True)
    matched = defaultdict(set)
    tp_list, fp_list = [], []

    for d in ordered:
        gts = gt.get(d["stem"], [])
        best_iou, best_idx = 0.0, -1
        for i, gt_box in enumerate(gts):
            v = iou(d["box"], gt_box)
            if v > best_iou:
                best_iou, best_idx = v, i
        if best_iou >= MAP_IOU and best_idx not in matched[d["stem"]]:
            tp_list.append(1); fp_list.append(0)
            matched[d["stem"]].add(best_idx)
        else:
            tp_list.append(0); fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    recalls = tp_cum / total_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec = precisions[recalls >= t]
        ap += float(np.max(prec)) if prec.size else 0.0
    return ap / 101


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("Loading ground truth...")
    gt = load_gt_boxes()
    total_gt = sum(len(v) for v in gt.values())
    print(f"  {len(gt)} test images, {total_gt} ground-truth boxes\n")

    curated_paths = {str(BASE_DIR / p) for _, p, _ in STAGES}
    rows = []

    def run(label, path, conf_floor=None):
        if not os.path.exists(path):
            print(f"{label:<34} FILE NOT FOUND")
            return
        dets = load_detections(path, conf_floor)
        if dets is None:
            return  # not a detection CSV
        p, r, f1, tp, fp, fn = eval_prf(dets, gt)
        ap = eval_map50(dets, gt)
        print(f"{label:<34} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {ap:>7.3f} "
              f"{tp:>6} {fp:>6} {fn:>6}  (n={len(dets)})")
        rows.append([label, len(dets), round(p, 3), round(r, 3), round(f1, 3),
                     round(ap, 3), tp, fp, fn, path])

    header = f"{'Stage':<34} {'P':>7} {'R':>7} {'F1':>7} {'mAP50':>7} {'TP':>6} {'FP':>6} {'FN':>6}"
    print(header)
    print("-" * len(header))

    for label, rel, floor in STAGES:
        run(label, str(BASE_DIR / rel), floor)

    # ── Auto-discover any detection CSV not in the curated list ─────────────
    extras = []
    for path in sorted(RUNS_DIR.rglob("*.csv")):
        s = str(path)
        if s in curated_paths or s == str(OUTPUT_CSV):
            continue
        if "summary" in path.name.lower() or "sweep" in path.name.lower():
            continue
        with open(path, "r") as f:
            head = f.readline()
        if not all(c in head for c in BOX_COLS):
            continue
        extras.append(path)

    if extras:
        print("\n-- auto-discovered --")
        for path in extras:
            run(str(path.relative_to(RUNS_DIR)), str(path))

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "n_detections", "precision", "recall", "f1",
                    "map50", "tp", "fp", "fn", "source_csv"])
        w.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")
    print(f"P/R/F1 at IoU >= {PRF_IOU} (file order, greedy); "
          f"mAP50 at IoU {MAP_IOU} (confidence-sorted, 101-point).")


if __name__ == "__main__":
    main()

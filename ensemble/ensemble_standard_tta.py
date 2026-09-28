"""
Standard + TTA Ensemble (Agreement Voting)
==============================================
Combines detections from standard YOLO inference and TTA inference.

Logic:
  - A detection is "high confidence" if it appears in BOTH standard and TTA
    detection sets at a similar location (matched by IoU).
  - A detection that appears in ONLY ONE of the two sets is "uncertain" and
    is kept only if it also passes slice-consistency (Rule B: >=3 consecutive
    slices) — using disagreement as a signal that extra evidence is needed.

This tests whether combining two independently-trained-but-related detection
sources (same model, different inference strategy) gives a better trade-off
than either alone.
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
OUTPUT_DIR       = BASE_DIR / "runs" / "ensemble_standard_tta"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE              = 512
IOU_THRESHOLD         = 0.30     # for matching detections to GT
AGREEMENT_IOU_THRESH  = 0.30     # for matching standard <-> TTA detections to each other
CENTROID_DIST_THRESH  = 30.0
TTA_CONF_FLOOR        = 0.30     # apply our tuned TTA threshold before anything else

# ── Load detections ──────────────────────────────────────────────────────────
print("Loading standard detections...")
with open(STANDARD_CSV, "r") as f:
    standard_dets = list(csv.DictReader(f))
print(f"  {len(standard_dets)} standard detections")

print("Loading TTA detections (filtering conf >= 0.30)...")
with open(TTA_CSV, "r") as f:
    tta_dets_all = list(csv.DictReader(f))
tta_dets = [d for d in tta_dets_all if float(d["confidence"]) >= TTA_CONF_FLOOR]
print(f"  {len(tta_dets)} TTA detections (after threshold)")

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


def centroid_distance(d1, d2):
    return ((float(d1["cx"]) - float(d2["cx"])) ** 2 + (float(d1["cy"]) - float(d2["cy"])) ** 2) ** 0.5


# ── Step 1: Find agreement between standard and TTA per image ──────────────
print("\nMatching standard and TTA detections per image...")

standard_by_image = defaultdict(list)
for d in standard_dets:
    standard_by_image[d["filename"]].append(d)

tta_by_image = defaultdict(list)
for d in tta_dets:
    tta_by_image[d["filename"]].append(d)

agreed = []       # detections confirmed by both
standard_only = []  # detections only in standard
tta_only = []        # detections only in TTA

all_filenames = set(standard_by_image.keys()) | set(tta_by_image.keys())

for filename in all_filenames:
    std_list = standard_by_image.get(filename, [])
    tta_list = tta_by_image.get(filename, [])

    matched_std_idx = set()
    matched_tta_idx = set()

    for i, std_d in enumerate(std_list):
        std_box = box_of(std_d)
        for j, tta_d in enumerate(tta_list):
            if j in matched_tta_idx:
                continue
            tta_box = box_of(tta_d)
            if iou(std_box, tta_box) >= AGREEMENT_IOU_THRESH:
                # Agreement found — use the TTA detection (broader recall coverage)
                # but mark as agreed
                agreed.append(tta_d)
                matched_std_idx.add(i)
                matched_tta_idx.add(j)
                break

    for i, std_d in enumerate(std_list):
        if i not in matched_std_idx:
            standard_only.append(std_d)

    for j, tta_d in enumerate(tta_list):
        if j not in matched_tta_idx:
            tta_only.append(tta_d)

print(f"Agreed detections (both models)  : {len(agreed)}")
print(f"Standard-only detections          : {len(standard_only)}")
print(f"TTA-only detections                : {len(tta_only)}")

# ── Step 2: Apply Rule B (slice consistency) to the uncertain (single-source) detections ──
def apply_rule_b_filter(det_list, all_patient_dets_lookup):
    """Keep only detections with >=3 consecutive slice run length."""
    kept = []
    by_patient = defaultdict(list)
    for d in det_list:
        by_patient[d["patient_id"]].append(d)

    for pid, dets in by_patient.items():
        # Use the full patient detection pool (agreed + this subset) for matching context
        context_dets = all_patient_dets_lookup.get(pid, dets)
        slice_lookup = defaultdict(list)
        for d in context_dets:
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

            if run >= 3:
                kept.append(d)

    return kept


# Build per-patient context (all detections combined) for slice matching
all_combined = agreed + standard_only + tta_only
context_by_patient = defaultdict(list)
for d in all_combined:
    context_by_patient[d["patient_id"]].append(d)

print("\nApplying Rule B slice-consistency filter to uncertain (single-source) detections...")
standard_only_filtered = apply_rule_b_filter(standard_only, context_by_patient)
tta_only_filtered = apply_rule_b_filter(tta_only, context_by_patient)

print(f"Standard-only surviving Rule B filter : {len(standard_only_filtered)} / {len(standard_only)}")
print(f"TTA-only surviving Rule B filter        : {len(tta_only_filtered)} / {len(tta_only)}")

# ── Final ensemble result ───────────────────────────────────────────────────
final_ensemble = agreed + standard_only_filtered + tta_only_filtered
print(f"\nFinal ensemble detection count: {len(final_ensemble)}")

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


print("\nLoading ground truth and evaluating...")
gt_boxes = load_gt_boxes()

metrics_agreed_only = evaluate(agreed, gt_boxes)
metrics_final = evaluate(final_ensemble, gt_boxes)

print(f"\n{'Config':<35}{'Precision':>10}{'Recall':>10}{'F1':>10}{'TP':>6}{'FP':>6}{'FN':>6}")
print("-" * 85)
print(f"{'Agreed only (both models)':<35}{metrics_agreed_only['precision']:>10}{metrics_agreed_only['recall']:>10}"
      f"{metrics_agreed_only['f1']:>10}{metrics_agreed_only['tp']:>6}{metrics_agreed_only['fp']:>6}{metrics_agreed_only['fn']:>6}")
print(f"{'Full ensemble (agreed + Rule B singles)':<35}{metrics_final['precision']:>10}{metrics_final['recall']:>10}"
      f"{metrics_final['f1']:>10}{metrics_final['tp']:>6}{metrics_final['fp']:>6}{metrics_final['fn']:>6}")

# ── Save results ─────────────────────────────────────────────────────────────
fieldnames = ["filename", "patient_id", "phase", "slice_num", "confidence", "x1", "y1", "x2", "y2", "cx", "cy"]

def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

save_csv(final_ensemble, OUTPUT_DIR / "final_ensemble_detections.csv")

summary_path = OUTPUT_DIR / "ensemble_evaluation_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["config", "precision", "recall", "f1", "tp", "fp", "fn"])
    writer.writerow(["Agreed only", metrics_agreed_only["precision"], metrics_agreed_only["recall"],
                      metrics_agreed_only["f1"], metrics_agreed_only["tp"], metrics_agreed_only["fp"], metrics_agreed_only["fn"]])
    writer.writerow(["Full ensemble", metrics_final["precision"], metrics_final["recall"],
                      metrics_final["f1"], metrics_final["tp"], metrics_final["fp"], metrics_final["fn"]])

print(f"\nResults saved to {OUTPUT_DIR}")
print("Done.")

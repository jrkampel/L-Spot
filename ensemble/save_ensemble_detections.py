"""
Re-run the 3-way ensemble logic and save the actual >=2-of-3 detections
to CSV (not just the summary metrics) so they can be used for visualisation.
"""

import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR         = Path(".")
STANDARD_CSV     = BASE_DIR / "runs" / "raw_detections.csv"
TTA_CSV          = BASE_DIR / "runs" / "raw_detections_tta.csv"
YOLO10_CSV       = BASE_DIR / "runs" / "raw_detections_yolo10.csv"
OUTPUT_DIR       = BASE_DIR / "runs" / "three_way_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AGREEMENT_IOU_THRESH = 0.30
TTA_CONF_FLOOR = 0.30

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

def by_image(dets):
    d = defaultdict(list)
    for det in dets:
        d[det["filename"]].append(det)
    return d

standard_by_img = by_image(standard_dets)
tta_by_img      = by_image(tta_dets)
yolo10_by_img   = by_image(yolo10_dets)

all_filenames = set(standard_by_img.keys()) | set(tta_by_img.keys()) | set(yolo10_by_img.keys())

print(f"Clustering detections across {len(all_filenames)} images...")

two_of_three = []

for filename in all_filenames:
    std_list    = standard_by_img.get(filename, [])
    tta_list    = tta_by_img.get(filename, [])
    yolo10_list = yolo10_by_img.get(filename, [])

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
                continue
            box_j = box_of(det_j)
            if iou(box_i, box_j) >= AGREEMENT_IOU_THRESH:
                cluster.append((det_j, src_j))
                used[j] = True

        sources_in_cluster = set(s for _, s in cluster)
        n_sources = len(sources_in_cluster)
        best_det = max(cluster, key=lambda x: float(x[0]["confidence"]))[0]

        if n_sources >= 2:
            two_of_three.append(best_det)

print(f"Detections confirmed by >=2 sources: {len(two_of_three)}")

# Save to CSV
fieldnames = ["filename", "patient_id", "phase", "slice_num", "class", "confidence",
              "x1", "y1", "x2", "y2", "cx", "cy", "source"]

out_path = OUTPUT_DIR / "ensemble_2of3_detections.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(two_of_three)

print(f"Saved to {out_path}")
print("Done.")

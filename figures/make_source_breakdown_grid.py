"""
make_source_breakdown_grid.py

Qualitative results figure for L-Spot: one row per example slice, four columns
showing each detection source plus the final consensus:

    YOLOv8m | YOLOv8m (TTA) | YOLOv10m | L-Spot

Every panel is the SAME CT slice; each column just draws that source's boxes.
Ground truth is drawn in green on every panel for reference.
    green = ground truth, red = source detection, blue = L-Spot consensus.

Run on the cluster (images live there):
    python ./make_source_breakdown_grid.py
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

# ---- paths ----------------------------------------------------------
BASE      = "."
IMG_DIR   = f"{BASE}/dataset/test/images"
LABEL_DIR = f"{BASE}/dataset/test/labels"
OUTPUT    = f"{BASE}/qualitative_sources_grid.png"

# (column header, csv path, box colour)
SOURCES = [
    ("YOLOv8m",       f"{BASE}/runs/raw_detections.csv",                              "#E24B4A"),
    ("YOLOv8m (TTA)", f"{BASE}/runs/raw_detections_tta.csv",                          "#E24B4A"),
    ("YOLOv10m",      f"{BASE}/runs/raw_detections_yolo10.csv",                       "#E24B4A"),
    ("L-Spot",        f"{BASE}/runs/three_way_ensemble/ensemble_2of3_detections.csv", "#1E7BD6"),
]
TTA_CONF_FLOOR = 0.30   # matches the floor the ensemble applies to TTA

# ---- the four example slices ----------------------------------------
EXAMPLES = [
    ("(a)", "P0070_C1_slice_074"),
    ("(b)", "P0085_C1_slice_039"),
    ("(c)", "P0018_C1_slice_027"),
    ("(d)", "P0004_C1_slice_005"),
]

# ---- style ----------------------------------------------------------
IMG_SIZE        = 512
GT_COLOUR       = "#2ECC40"
HEADER_FONTSIZE = 17
ROWLABEL_FONT   = 15
CONF_FONTSIZE   = 10
GT_LW           = 1.4
DET_LW          = 2.0
PANEL_IN        = 2.7
DPI             = 300
# ---------------------------------------------------------------------


def stem(name):
    return str(name).replace(".png", "").replace(".jpg", "").replace(".jpeg", "")


def load_dets(csv_path, conf_floor=0.0):
    by = defaultdict(list)
    if not os.path.exists(csv_path):
        print(f"  WARNING: missing {csv_path}")
        return by
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            c = float(r["confidence"])
            if c < conf_floor:
                continue
            by[stem(r["filename"])].append(
                ([float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])], c)
            )
    return by


def load_gt(s):
    path = os.path.join(LABEL_DIR, s + ".txt")
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 5:
                continue
            _, cx, cy, w, h = map(float, p[:5])
            boxes.append([(cx - w / 2) * IMG_SIZE, (cy - h / 2) * IMG_SIZE,
                          (cx + w / 2) * IMG_SIZE, (cy + h / 2) * IMG_SIZE])
    return boxes


print("Loading detection sources...")
source_dets = []
for header, path, _ in SOURCES:
    floor = TTA_CONF_FLOOR if "TTA" in header else 0.0
    d = load_dets(path, floor)
    source_dets.append(d)
    print(f"  {header}: {len(d)} slices with detections")

n_rows, n_cols = len(EXAMPLES), len(SOURCES)
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(PANEL_IN * n_cols, PANEL_IN * n_rows), dpi=DPI)
if n_rows == 1:
    axes = axes.reshape(1, -1)

for ri, (row_label, s) in enumerate(EXAMPLES):
    img_path = os.path.join(IMG_DIR, s + ".png")
    gt_boxes = load_gt(s)
    if not os.path.exists(img_path):
        print(f"  WARNING: image not found: {img_path}")
    for ci, (header, path, colour) in enumerate(SOURCES):
        ax = axes[ri, ci]
        ax.axis("off")
        if os.path.exists(img_path):
            ax.imshow(mpimg.imread(img_path), cmap="gray")
        for g in gt_boxes:
            ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1],
                         fill=False, edgecolor=GT_COLOUR, linewidth=GT_LW))
        for bx, c in source_dets[ci].get(s, []):
            ax.add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1],
                         fill=False, edgecolor=colour, linewidth=DET_LW))
            ax.text(bx[0], max(bx[1] - 7, 12), f"{c:.2f}",
                    color=colour, fontsize=CONF_FONTSIZE, weight="bold")
        if ri == 0:
            ax.set_title(header, fontsize=HEADER_FONTSIZE, fontweight="bold", pad=9)
        if ci == 0:
            ax.text(-0.06, 0.5, row_label, ha="right", va="center",
                    fontsize=ROWLABEL_FONT, fontweight="bold", transform=ax.transAxes)

plt.subplots_adjust(wspace=0.03, hspace=0.05, left=0.045, right=0.995,
                    top=0.93, bottom=0.005)
plt.savefig(OUTPUT, dpi=DPI, bbox_inches="tight")
print(f"\nSaved: {OUTPUT}")
print("green = ground truth | red = source detection | blue = L-Spot")

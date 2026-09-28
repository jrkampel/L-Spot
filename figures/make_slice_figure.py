"""Build the slice-consistency figure for the presentation.

Top row    : detection persisting across 3 consecutive slices (retained).
Bottom row : detection on a single slice only (removed).

Only the detection closest to the track centroid is drawn on each panel,
so overlapping duplicate boxes are suppressed.

Run with:
    module load miniforge/24.7.1
    python make_slice_figure.py
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image
from pathlib import Path

DET = "runs/slice_postprocessing/all_detections_with_run_length.csv"
IMG_DIR = Path("dataset/test/images")
OUT = "slice_consistency_figure.png"

# (patient, three consecutive slices to display, slices that carry a box)
RETAINED = ("P0018", [24, 25, 26], [24, 25, 26])
REMOVED = ("P0018", [1, 2, 3], [2])

# Alternatives if the retained lesion is too large to read well:
#   RETAINED = ("P0092", [46, 47, 48], [46, 47, 48])
#   RETAINED = ("P0032", [21, 22, 23], [21, 22, 23])

CROP = 340          # crop window in pixels, centred on the detection
KEEP_COLOR = "#1D9E75"
DROP_COLOR = "#D85A30"

det = pd.read_csv(DET)


def centre_of(patient, slice_nums):
    """Mean centroid of the detections forming the track."""
    m = det[(det.patient_id == patient) & (det.slice_num.isin(slice_nums))]
    if len(m) == 0:
        return 256.0, 256.0
    return float(m.cx.mean()), float(m.cy.mean())


def best_box(patient, slice_num, cx_ref, cy_ref):
    """Single detection on this slice nearest the track centroid."""
    m = det[(det.patient_id == patient) & (det.slice_num == slice_num)]
    if len(m) == 0:
        return None
    m = m.copy()
    m["d2"] = (m.cx - cx_ref) ** 2 + (m.cy - cy_ref) ** 2
    r = m.sort_values("d2").iloc[0]
    return float(r.x1), float(r.y1), float(r.x2), float(r.y2), float(r.confidence)


def find_image(patient, slice_num):
    for name in (f"{patient}_C1_slice_{slice_num:03d}.png",
                 f"{patient}_slice_{slice_num:03d}.png"):
        p = IMG_DIR / name
        if p.exists():
            return p
    hits = sorted(IMG_DIR.glob(f"{patient}*slice_{slice_num:03d}.png"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{patient} slice {slice_num}")


fig, axes = plt.subplots(2, 3, figsize=(9, 6.4))

rows = [
    (*RETAINED, KEEP_COLOR, "Retained  (run length 3)"),
    (*REMOVED, DROP_COLOR, "Removed  (run length 1)"),
]

for row, (patient, slices, box_slices, color, label) in enumerate(rows):
    cx, cy = centre_of(patient, box_slices)
    x0 = max(0, min(512 - CROP, int(cx - CROP / 2)))
    y0 = max(0, min(512 - CROP, int(cy - CROP / 2)))

    for col, s in enumerate(slices):
        ax = axes[row, col]
        img = Image.open(find_image(patient, s)).convert("L")
        ax.imshow(img.crop((x0, y0, x0 + CROP, y0 + CROP)),
                  cmap="gray", vmin=0, vmax=255)

        if s in box_slices:
            box = best_box(patient, s, cx, cy)
            if box is not None:
                bx1, by1, bx2, by2, conf = box
                ax.add_patch(patches.Rectangle(
                    (bx1 - x0, by1 - y0), bx2 - bx1, by2 - by1,
                    linewidth=2, edgecolor=color, facecolor="none"))
                print(f"  {patient} slice {s}: "
                      f"{bx2 - bx1:.0f}x{by2 - by1:.0f} px, conf {conf:.3f}")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#B4B2A9")
            spine.set_linewidth(0.8)

        if row == 0:
            offset = col - 1
            ax.set_title("slice k" if offset == 0 else f"slice k{offset:+d}",
                         fontsize=12, pad=8)

    axes[row, 0].set_ylabel(label, fontsize=12, labelpad=10)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight",
            facecolor="white", transparent=False)
print(f"\nWrote {OUT}")
print(f"Retained: {RETAINED[0]} slices {RETAINED[1]}")
print(f"Removed:  {REMOVED[0]} slices {REMOVED[1]}")

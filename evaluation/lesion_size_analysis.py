"""Lesion size distribution and recall by size band.

Reads the ground-truth YOLO labels for the test partition, characterises the
distribution of annotated lesion area, then reports recall within size bands
for the baseline detector and for the L-Spot consensus output.

Run with:
    module load miniforge/24.7.1
    python lesion_size_analysis.py
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

LABELS = Path("dataset/test/labels")
IMG_W = IMG_H = 512
IOU_T = 0.30

# Detection sets to evaluate. Add or remove rows as needed.
SOURCES = [
    ("YOLOv8m baseline", "runs/raw_detections.csv"),
    ("Rule B", "runs/slice_postprocessing/filtered_rule_b.csv"),
    ("L-Spot (>=2 of 3)", "runs/three_way_ensemble/ensemble_2of3_detections.csv"),
]

OUT_CSV = "runs/lesion_size_recall.csv"
OUT_FIG = "lesion_size_distribution.png"


# ---------------------------------------------------------------- ground truth

def load_ground_truth():
    rows = []
    for f in sorted(LABELS.glob("*.txt")):
        stem = f.name[:-4]
        for line in f.read_text().split("\n"):
            p = line.split()
            if len(p) < 5:
                continue
            _, xc, yc, w, h = (float(v) for v in p[:5])
            rows.append({
                "filename": stem,
                "x1": (xc - w / 2) * IMG_W,
                "y1": (yc - h / 2) * IMG_H,
                "x2": (xc + w / 2) * IMG_W,
                "y2": (yc + h / 2) * IMG_H,
                "w": w * IMG_W,
                "h": h * IMG_H,
            })
    gt = pd.DataFrame(rows)
    gt["area"] = gt.w * gt.h
    gt["side"] = np.sqrt(gt.area)
    return gt


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua


def recall_by_band(gt, det_csv, bands):
    """Greedy highest-IoU matching, one detection per ground-truth box."""
    det = pd.read_csv(det_csv)
    by_slice = {k: v[["x1", "y1", "x2", "y2"]].values.tolist()
                for k, v in det.groupby("filename")}

    matched = []
    for stem, g in gt.groupby("filename"):
        boxes = list(by_slice.get(stem, []))
        used = set()
        for _, r in g.iterrows():
            gtb = (r.x1, r.y1, r.x2, r.y2)
            best, bi = None, 0.0
            for i, d in enumerate(boxes):
                if i in used:
                    continue
                v = iou(gtb, d)
                if v > bi:
                    best, bi = i, v
            hit = bi >= IOU_T
            if hit:
                used.add(best)
            matched.append(hit)

    out = gt.copy()
    out["hit"] = matched
    out["band"] = pd.cut(out.side, bins=bands["edges"],
                         labels=bands["labels"], right=False)
    return out


def main():
    gt = load_ground_truth()
    if len(gt) == 0:
        raise SystemExit(f"No labels found under {LABELS}")

    print(f"{len(gt)} annotated lesion instances across "
          f"{gt.filename.nunique()} slices\n")

    q = gt.side.describe(percentiles=[.05, .25, .5, .75, .95])
    print("Lesion size (square-root of box area, pixels):")
    print(q.round(1).to_string())
    print(f"\nMedian box: {gt.w.median():.0f} x {gt.h.median():.0f} px")
    print(f"Median area: {gt.area.median():.0f} px^2 "
          f"({gt.area.median() / (IMG_W * IMG_H) * 100:.2f}% of slice)")
    print(f"Smallest / largest side: {gt.side.min():.0f} / {gt.side.max():.0f} px\n")

    bands = {
        "edges": [0, 40, 60, 80, 120, np.inf],
        "labels": ["<40", "40-60", "60-80", "80-120", ">=120"],
    }

    print(f"Lesions per size band (sqrt area, px):")
    counts = pd.cut(gt.side, bins=bands["edges"],
                    labels=bands["labels"], right=False).value_counts().sort_index()
    print(counts.to_string(), "\n")

    results = {}
    for name, csv in SOURCES:
        if not Path(csv).exists():
            print(f"[skip] {name}: {csv} not found")
            continue
        m = recall_by_band(gt, csv, bands)
        r = m.groupby("band", observed=False).hit.agg(["sum", "count"])
        r["recall"] = (r["sum"] / r["count"]).round(3)
        results[name] = r["recall"]
        print(f"{name}: overall recall {m.hit.mean():.3f}")
        print(r.to_string(), "\n")

    if results:
        tab = pd.DataFrame(results)
        tab.insert(0, "n_lesions", counts)
        tab.to_csv(OUT_CSV)
        print(f"Wrote {OUT_CSV}\n")
        print(tab.to_string())

    # Figure: histogram of lesion size with median marked
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gt.side, bins=40, color="#4A7EBB", edgecolor="white", linewidth=0.5)
    ax.axvline(gt.side.median(), color="#D85A30", linewidth=2,
               label=f"median {gt.side.median():.0f} px")
    ax.set_xlabel("Lesion size (square root of box area, pixels)")
    ax.set_ylabel("Annotated instances")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200, facecolor="white")
    print(f"\nWrote {OUT_FIG}")


if __name__ == "__main__":
    main()

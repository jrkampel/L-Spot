"""
IoU matching-threshold sweep
============================
Re-evaluates fixed detection sets under varying IoU matching criteria.

The detections themselves are unchanged; only the threshold at which a
detection is counted as matching a ground-truth box varies. This isolates
the effect of the matching criterion from any change in the detector.

Run from ./ so that
`import evaluate_postprocessing` resolves.

    cd .
    python iou_sweep.py

CPU only -- this is CSV matching, no GPU or SLURM job required.
"""

import evaluate_postprocessing as ev

THRESHOLDS = [0.30, 0.40, 0.50]

SOURCES = {
    "YOLOv8m unfiltered": ev.BASE_DIR / "runs" / "raw_detections.csv",
    "YOLOv10m unfiltered": ev.BASE_DIR / "runs" / "raw_detections_yolo10.csv",
    "L-Spot (>=2 of 3)": ev.BASE_DIR / "runs" / "three_way_ensemble" / "ensemble_2of3_detections.csv",
    "L-Spot (all 3)": ev.BASE_DIR / "runs" / "three_way_ensemble" / "ensemble_all3_detections.csv",
}

gt = ev.load_gt_boxes()
n_gt = sum(len(v) for v in gt.values())
print(f"Ground-truth boxes loaded: {n_gt}\n")

print(f"{'Source':<22} {'IoU':>5} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
print("-" * 72)

rows = []

for name, path in SOURCES.items():
    if not path.exists():
        print(f"{name:<22}  MISSING: {path}")
        continue

    dets = ev.load_detections(path)

    for t in THRESHOLDS:
        ev.IOU_THRESHOLD = t
        r = ev.evaluate(dets, gt)
        print(f"{name:<22} {t:>5.2f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>7.3f} {r['tp']:>6d} {r['fp']:>6d} {r['fn']:>6d}")
        rows.append({
            "source": name,
            "iou_threshold": t,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "tp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
        })
    print()

# Write results alongside the other run outputs
import csv

out_path = ev.BASE_DIR / "runs" / "iou_threshold_sweep.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Written: {out_path}")
print("\nSanity check: the IoU 0.30 rows should reproduce Table 6 "
      "(YOLOv8m 0.566/0.674/0.615, TP 781, FP 599, FN 378) and the "
      "three-source consensus table (>=2 of 3: 0.820/0.664/0.733).")

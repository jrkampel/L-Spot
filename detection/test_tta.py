"""
Test-Time Augmentation (TTA) Evaluation
==========================================
Runs YOLOv8m baseline inference on the test set WITH test-time augmentation
enabled (augment=True), which runs inference on flipped/scaled versions of
each image and merges the predictions.

Compares against the standard (non-TTA) baseline to see if TTA recovers
additional true positives (improves recall) without excessive precision loss.
"""

from ultralytics import YOLO
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(".")
MODEL_PATH  = BASE_DIR / "runs" / "liver_tumour_yolov8" / "weights" / "best.pt"
DATA_YAML   = BASE_DIR / "dataset.yaml"

print(f"Loading model from {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

# ── Standard validation (no TTA) — for reference ────────────────────────────
print("\n=== Standard inference (no TTA) ===")
results_standard = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=512,
    batch=16,
    conf=0.10,
    augment=False,
    project=str(BASE_DIR / "runs" / "detect"),
    name="yolov8m_standard_testeval",
    exist_ok=True
)
print(f"Precision: {results_standard.box.mp:.3f}")
print(f"Recall:    {results_standard.box.mr:.3f}")
print(f"mAP50:     {results_standard.box.map50:.3f}")
print(f"mAP50-95:  {results_standard.box.map:.3f}")

# ── TTA validation ───────────────────────────────────────────────────────────
print("\n=== Test-Time Augmentation (TTA) inference ===")
results_tta = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=512,
    batch=16,
    conf=0.10,
    augment=True,
    project=str(BASE_DIR / "runs" / "detect"),
    name="yolov8m_tta_testeval",
    exist_ok=True
)
print(f"Precision: {results_tta.box.mp:.3f}")
print(f"Recall:    {results_tta.box.mr:.3f}")
print(f"mAP50:     {results_tta.box.map50:.3f}")
print(f"mAP50-95:  {results_tta.box.map:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Config':<20}{'Precision':>12}{'Recall':>12}{'mAP50':>12}{'mAP50-95':>12}")
print(f"{'Standard':<20}{results_standard.box.mp:>12.3f}{results_standard.box.mr:>12.3f}"
      f"{results_standard.box.map50:>12.3f}{results_standard.box.map:>12.3f}")
print(f"{'TTA':<20}{results_tta.box.mp:>12.3f}{results_tta.box.mr:>12.3f}"
      f"{results_tta.box.map50:>12.3f}{results_tta.box.map:>12.3f}")

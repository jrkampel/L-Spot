"""
Create a 3-row grid figure for the paper
Row 1: P0004_C1_slice_037 - false positive reduction
Row 2: P0004_C1_slice_018 - good detection
Row 3: P0082_C1_slice_016 - limitation case
"""

import cv2
import numpy as np
from pathlib import Path

VIZ_DIR = Path("./runs/visualizations_v2")
OUTPUT  = Path("./runs/figure_grid.png")

images = [
    ("P0004_C1_slice_037.png", "(a) False positive reduction"),
    ("P0004_C1_slice_018.png", "(b) Correct detection"),
    ("P0082_C1_slice_016.png", "(c) Limitation: missed tumour"),
]

rows = []
for filename, label in images:
    img = cv2.imread(str(VIZ_DIR / filename))
    if img is None:
        print(f"Could not load {filename}")
        continue

    # Add label below each row
    h, w = img.shape[:2]
    label_bar = np.ones((30, w, 3), dtype=np.uint8) * 40
    cv2.putText(label_bar, label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    row = np.vstack([img, label_bar])
    rows.append(row)

# Add small gap between rows
gap = np.ones((6, rows[0].shape[1], 3), dtype=np.uint8) * 255
grid = np.vstack([rows[0], gap, rows[1], gap, rows[2]])

cv2.imwrite(str(OUTPUT), grid)
print(f"Grid saved to {OUTPUT}")

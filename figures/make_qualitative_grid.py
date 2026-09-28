"""
make_qualitative_grid.py

Builds the qualitative results figure with two categories
(False positive reduction, Correct detection), each shown as its
own 2x2 grid of images, stacked vertically.

Designed to be included at \\textwidth in LaTeX (NOT 0.5\\textwidth).

Usage (on the cluster, via Termius):
    python make_qualitative_grid.py

Edit IMG_DIR and CATEGORIES below to point at your actual files.
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

# ---- EDIT THESE ----------------------------------------------------

IMG_DIR = "./runs/visualizations_v2"

# Each category is its own 2x2 block (2 rows x 2 cols).
CATEGORIES = {
    "False positive reduction": [
        "P0014_C1_slice_009.png",
        "P0014_C1_slice_012.png",
        "P0014_C1_slice_016.png",
        "P0015_C1_slice_016.png",
    ],
    "Correct detection": [
        "P0029_C1_slice_032.png",
        "P0028_C1_slice_010.png",
        "P0018_C1_slice_027.png",
        "P0028_C1_slice_008.png",
    ],
}

OUTPUT_PATH = "./qualitative_grid.png"

# --- Sizing / quality -------------------------------------------------
# FIG_WIDTH_IN is the width of the *source canvas*. Making it larger than
# the final on-page width (LNCS \textwidth is ~4.8in) means LaTeX scales
# the PNG DOWN, which supersamples and keeps the baked-in panel labels and
# confidence numbers crisp instead of blurry. Bigger source + high DPI =
# sharper result once included at \textwidth.
FIG_WIDTH_IN   = 10.0    # was 6.75 -- larger, higher-res source
DPI            = 400     # was 300
HEADER_FONTSIZE = 20     # was 9 -- category titles, scaled for the 10in canvas
TITLE_PAD_IN   = 0.50    # was 0.28 -- more room for the bigger headers

# ---------------------------------------------------------------------

n_categories = len(CATEGORIES)
n_rows = n_categories * 2   # 2 rows per category
n_cols = 2

# Detect the actual aspect ratio from the first available image so each
# cell can be sized to match exactly -- removes the letterbox gaps caused
# by these being wide (Baseline | Ensemble) composite images.
sample_img = None
for filenames in CATEGORIES.values():
    for fname in filenames:
        fpath = os.path.join(IMG_DIR, fname)
        if os.path.exists(fpath):
            sample_img = Image.open(fpath)
            break
    if sample_img:
        break

if sample_img is not None:
    img_aspect = sample_img.width / sample_img.height  # W/H
else:
    img_aspect = 2.0  # fallback guess if no files found yet

cell_width_in = FIG_WIDTH_IN / n_cols
cell_height_in = cell_width_in / img_aspect
fig_height_in = cell_height_in * n_rows + TITLE_PAD_IN * n_categories

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(FIG_WIDTH_IN, fig_height_in),
    dpi=DPI,
)

for cat_idx, (cat_label, filenames) in enumerate(CATEGORIES.items()):
    base_row = cat_idx * 2
    for i in range(4):
        r = base_row + (i // 2)
        c = i % 2
        ax = axes[r, c]
        ax.axis("off")
        if i < len(filenames):
            fname = filenames[i]
            fpath = os.path.join(IMG_DIR, fname)
            if os.path.exists(fpath):
                img = Image.open(fpath)
                # aspect='auto' fills the cell exactly (cell sized to image
                # aspect above); 'lanczos' gives clean supersampled downscaling
                # so baked-in text stays legible.
                ax.imshow(img, aspect="auto", interpolation="lanczos")
            else:
                ax.text(0.5, 0.5, "missing:\n" + fname,
                        ha="center", va="center", fontsize=11, color="red",
                        transform=ax.transAxes)
    # Category label above the top-left axis of its block
    axes[base_row, 0].set_title(cat_label, loc="left",
                                fontsize=HEADER_FONTSIZE,
                                fontweight="bold", pad=6)

plt.subplots_adjust(wspace=0.01, hspace=0.06, left=0.01, right=0.99,
                     top=0.97, bottom=0.01)
plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
print(f"Saved grid figure to {OUTPUT_PATH}")

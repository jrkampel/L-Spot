"""
Patch Generation Script for Liver Tumour Detection
====================================================
Cuts full 512x512 CT slices into overlapping patches for YOLO training.
Applies to train and val sets only — test set is left untouched.

Usage:
    python generate_patches.py --patch_size 256 --overlap 0.25 --output_suffix exp1
    python generate_patches.py --patch_size 256 --overlap 0.50 --output_suffix exp2
    python generate_patches.py --patch_size 384 --overlap 0.25 --output_suffix exp3
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import shutil

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Generate patches from CT slices for YOLO training.")
parser.add_argument("--patch_size",   type=int,   default=256,   help="Patch width and height in pixels (e.g. 256 or 384)")
parser.add_argument("--overlap",      type=float, default=0.25,  help="Fractional overlap between patches (e.g. 0.25 or 0.50)")
parser.add_argument("--output_suffix",type=str,   default="exp1",help="Suffix for output dataset folder (e.g. exp1, exp2, exp3)")
parser.add_argument("--neg_ratio",    type=float, default=0.3,   help="Fraction of negative patches (no tumour) to keep per image")
parser.add_argument("--base_dir",     type=str,   default="./dataset",
                    help="Path to the original dataset root folder")
args = parser.parse_args()

# ── Derived settings ──────────────────────────────────────────────────────────

PATCH_SIZE  = args.patch_size
STRIDE      = int(PATCH_SIZE * (1 - args.overlap))   # step between patch starts
NEG_RATIO   = args.neg_ratio
BASE_DIR    = Path(args.base_dir)
OUT_DIR     = BASE_DIR.parent / f"dataset_patches_{args.output_suffix}"
SPLITS      = ["train", "val"]   # test set is never patched

print(f"\n{'='*60}")
print(f"Patch size : {PATCH_SIZE}x{PATCH_SIZE}")
print(f"Overlap    : {args.overlap*100:.0f}%  (stride = {STRIDE}px)")
print(f"Input dir  : {BASE_DIR}")
print(f"Output dir : {OUT_DIR}")
print(f"{'='*60}\n")

# ── Helper functions ──────────────────────────────────────────────────────────

def read_yolo_labels(label_path):
    """
    Read a YOLO label file and return a list of boxes.
    Each box: [class_id, cx, cy, w, h]  (all normalised 0-1)
    Returns empty list if file missing or empty.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, w, h = parts
                boxes.append([int(cls), float(cx), float(cy), float(w), float(h)])
    return boxes


def yolo_to_pixel(box, img_w, img_h):
    """
    Convert a YOLO normalised box [cls, cx, cy, w, h] to pixel coords.
    Returns [cls, x1, y1, x2, y2]
    """
    cls, cx, cy, bw, bh = box
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return [cls, x1, y1, x2, y2]


def clip_box_to_patch(box_pixel, px1, py1, px2, py2, min_area_ratio=0.25):
    """
    Clip a pixel-space box to a patch region.
    Returns None if the intersection is too small to be useful.
    min_area_ratio: discard if intersection < this fraction of original box area.
    """
    cls, bx1, by1, bx2, by2 = box_pixel

    # Intersection with patch
    ix1 = max(bx1, px1)
    iy1 = max(by1, py1)
    ix2 = min(bx2, px2)
    iy2 = min(by2, py2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None  # no intersection

    orig_area  = (bx2 - bx1) * (by2 - by1)
    inter_area = (ix2 - ix1) * (iy2 - iy1)

    if orig_area == 0 or inter_area / orig_area < min_area_ratio:
        return None  # intersection too small

    # Convert back to YOLO normalised coords relative to patch
    patch_w = px2 - px1
    patch_h = py2 - py1

    new_cx = (ix1 + ix2) / 2 - px1
    new_cy = (iy1 + iy2) / 2 - py1
    new_w  = ix2 - ix1
    new_h  = iy2 - iy1

    # Normalise
    new_cx /= patch_w
    new_cy /= patch_h
    new_w  /= patch_w
    new_h  /= patch_h

    # Clamp to [0, 1]
    new_cx = np.clip(new_cx, 0, 1)
    new_cy = np.clip(new_cy, 0, 1)
    new_w  = np.clip(new_w,  0, 1)
    new_h  = np.clip(new_h,  0, 1)

    return [cls, new_cx, new_cy, new_w, new_h]


def get_patch_coords(img_h, img_w, patch_size, stride):
    """
    Generate all (y1, x1, y2, x2) patch coordinates for an image.
    Ensures the right and bottom edges are always covered.
    """
    coords = []
    y_starts = list(range(0, img_h - patch_size, stride)) + [img_h - patch_size]
    x_starts = list(range(0, img_w - patch_size, stride)) + [img_w - patch_size]
    y_starts = sorted(set(max(0, y) for y in y_starts))
    x_starts = sorted(set(max(0, x) for x in x_starts))
    for y1 in y_starts:
        for x1 in x_starts:
            coords.append((y1, x1, y1 + patch_size, x1 + patch_size))
    return coords


# ── Main processing loop ──────────────────────────────────────────────────────

total_pos_patches = 0
total_neg_patches = 0
total_images      = 0

for split in SPLITS:
    img_dir   = BASE_DIR / split / "images"
    lbl_dir   = BASE_DIR / split / "labels"
    out_img   = OUT_DIR  / split / "images"
    out_lbl   = OUT_DIR  / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    image_files = sorted(img_dir.glob("*.png"))
    print(f"[{split}] Processing {len(image_files)} images...")

    for img_path in image_files:
        stem      = img_path.stem
        lbl_path  = lbl_dir / (stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Could not read {img_path}, skipping.")
            continue

        img_h, img_w = img.shape[:2]
        boxes_norm   = read_yolo_labels(lbl_path)
        boxes_pixel  = [yolo_to_pixel(b, img_w, img_h) for b in boxes_norm]
        patch_coords = get_patch_coords(img_h, img_w, PATCH_SIZE, STRIDE)

        pos_patches  = []
        neg_patches  = []

        for idx, (py1, px1, py2, px2) in enumerate(patch_coords):
            patch      = img[py1:py2, px1:px2]
            patch_boxes = []

            for bp in boxes_pixel:
                clipped = clip_box_to_patch(bp, px1, py1, px2, py2)
                if clipped is not None:
                    patch_boxes.append(clipped)

            if patch_boxes:
                pos_patches.append((patch, patch_boxes, idx))
            else:
                neg_patches.append((patch, [], idx))

        # Keep all positive patches
        kept_patches = pos_patches[:]

        # Keep a random subset of negative patches
        if neg_patches:
            n_neg = max(1, int(len(pos_patches) * NEG_RATIO / (1 - NEG_RATIO + 1e-6)))
            n_neg = min(n_neg, len(neg_patches))
            chosen_neg = np.random.choice(len(neg_patches), size=n_neg, replace=False)
            kept_patches += [neg_patches[i] for i in chosen_neg]

        # Save patches
        for patch_img, patch_boxes, idx in kept_patches:
            patch_name = f"{stem}_p{idx:03d}"

            cv2.imwrite(str(out_img / f"{patch_name}.png"), patch_img)

            with open(out_lbl / f"{patch_name}.txt", "w") as f:
                for pb in patch_boxes:
                    f.write(f"{pb[0]} {pb[1]:.6f} {pb[2]:.6f} {pb[3]:.6f} {pb[4]:.6f}\n")

        total_pos_patches += len(pos_patches)
        total_neg_patches += len([p for p in kept_patches if not p[1]])
        total_images += 1

    print(f"[{split}] Done.")

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"Total source images processed : {total_images}")
print(f"Total positive patches saved  : {total_pos_patches}")
print(f"Total negative patches saved  : {total_neg_patches}")
print(f"Output written to             : {OUT_DIR}")
print(f"{'='*60}\n")

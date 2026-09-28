"""
Plot a single CT test slice with its ground-truth bounding box (green),
and optionally a predicted bounding box, for use in the methods diagram.

Usage:
    python plot_box_for_diagram.py --slice <slice_filename_without_extension>

If no slice is specified, the script will list the first 10 tumour-containing
test slices it finds so you can pick one.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def yolo_to_xyxy(x_center, y_center, w, h, img_w, img_h):
    """Convert normalized YOLO format to pixel xyxy coordinates."""
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h
    return int(x1), int(y1), int(x2), int(y2)


def load_yolo_boxes(label_path, img_w, img_h):
    """Load all bounding boxes from a YOLO-format label file."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = parts[:5]
            boxes.append(yolo_to_xyxy(float(xc), float(yc), float(w), float(h), img_w, img_h))
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./dataset",
        help="Root of the dataset folder containing test/images and test/labels",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Filename (without extension) of the slice to plot, e.g. patient012_slice045",
    )
    parser.add_argument(
        "--pred_label",
        type=str,
        default=None,
        help="Optional path to a predicted YOLO label .txt file to overlay in a second colour",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/user-data/outputs/diagram_box_example.png",
        help="Output path for the saved image",
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        default=3,
        help="Bounding box line thickness in pixels",
    )
    args = parser.parse_args()

    images_dir = Path(args.dataset_root) / "test" / "images"
    labels_dir = Path(args.dataset_root) / "test" / "labels"

    if args.slice is None:
        # List tumour-containing slices to help pick one
        print("No --slice specified. Here are some tumour-containing test slices:\n")
        count = 0
        for label_file in sorted(labels_dir.glob("*.txt")):
            if label_file.stat().st_size > 0:  # non-empty = has a tumour box
                print(f"  {label_file.stem}")
                count += 1
            if count >= 15:
                break
        print("\nRe-run with --slice <name> to plot one of these.")
        return

    # Find the image file (try common extensions)
    image_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = images_dir / f"{args.slice}{ext}"
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        print(f"Could not find image for slice '{args.slice}' in {images_dir}")
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image at {image_path}")
        return

    img_h, img_w = img.shape[:2]

    # Ground truth boxes (green)
    gt_label_path = labels_dir / f"{args.slice}.txt"
    gt_boxes = load_yolo_boxes(gt_label_path, img_w, img_h)

    if not gt_boxes:
        print(f"Warning: no ground truth boxes found for slice '{args.slice}'")

    GREEN = (0, 200, 0)  # BGR
    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, args.line_thickness)

    # Optional predicted boxes (blue)
    if args.pred_label is not None:
        pred_boxes = load_yolo_boxes(Path(args.pred_label), img_w, img_h)
        BLUE = (255, 120, 0)  # BGR
        for (x1, y1, x2, y2) in pred_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), BLUE, args.line_thickness)

    cv2.imwrite(args.output, img)
    print(f"Saved annotated image to: {args.output}")
    print(f"Ground truth boxes drawn: {len(gt_boxes)}")


if __name__ == "__main__":
    main()

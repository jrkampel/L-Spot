"""
Plot a single CT test slice showing detections from all three L-Spot
sources (S1: YOLOv8m standard, S2: YOLOv8m+TTA, S3: YOLOv10m) overlaid
in different colours, alongside the ground truth box.

Useful for the "Multi-source consensus voting" panel of the methods
diagram, since it visually shows multiple sources proposing boxes for
the same lesion.

Usage:
    python plot_multisource_for_diagram.py --slice P0004_C1_slice_012
"""

import argparse
import os
from pathlib import Path

import cv2
import pandas as pd


def yolo_to_xyxy(x_center, y_center, w, h, img_w, img_h):
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h
    return int(x1), int(y1), int(x2), int(y2)


def load_gt_boxes(label_path, img_w, img_h):
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


def load_source_boxes(csv_path, slice_name, conf_thresh):
    """Load predicted boxes for one slice from a raw detections CSV."""
    df = pd.read_csv(csv_path)
    rows = df[(df["filename"] == slice_name) & (df["confidence"] >= conf_thresh)]
    boxes = []
    for _, row in rows.iterrows():
        boxes.append((int(row.x1), int(row.y1), int(row.x2), int(row.y2), row.confidence))
    return boxes


def draw_boxes(img, boxes, color, thickness, label_prefix=None):
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label_prefix is not None and len(box) > 4:
            conf = box[4]
            cv2.putText(
                img, f"{label_prefix} {conf:.2f}", (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", type=str,
        default="./dataset",
    )
    parser.add_argument(
        "--runs_root", type=str,
        default="./runs",
    )
    parser.add_argument("--slice", type=str, required=True)
    parser.add_argument(
        "--output", type=str,
        default="./diagram_multisource_example.png",
    )
    parser.add_argument("--line_thickness", type=int, default=2)
    parser.add_argument("--show_labels", action="store_true",
                         help="Draw confidence score labels next to each box")
    args = parser.parse_args()

    images_dir = Path(args.dataset_root) / "test" / "images"
    labels_dir = Path(args.dataset_root) / "test" / "labels"

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
    img_h, img_w = img.shape[:2]

    # Ground truth - green
    gt_boxes = load_gt_boxes(labels_dir / f"{args.slice}.txt", img_w, img_h)
    draw_boxes(img, gt_boxes, (0, 200, 0), args.line_thickness + 1)

    # S1: YOLOv8m standard - purple, conf >= 0.10
    s1_boxes = load_source_boxes(
        Path(args.runs_root) / "raw_detections.csv", args.slice, conf_thresh=0.10
    )
    draw_boxes(img, s1_boxes, (200, 80, 160), args.line_thickness,
               label_prefix="S1" if args.show_labels else None)

    # S2: YOLOv8m + TTA - teal, conf >= 0.30
    s2_boxes = load_source_boxes(
        Path(args.runs_root) / "raw_detections_tta.csv", args.slice, conf_thresh=0.30
    )
    draw_boxes(img, s2_boxes, (180, 160, 0), args.line_thickness,
               label_prefix="S2" if args.show_labels else None)

    # S3: YOLOv10m standard - coral/orange, conf >= 0.10
    s3_boxes = load_source_boxes(
        Path(args.runs_root) / "raw_detections_yolo10.csv", args.slice, conf_thresh=0.10
    )
    draw_boxes(img, s3_boxes, (60, 120, 255), args.line_thickness,
               label_prefix="S3" if args.show_labels else None)

    cv2.imwrite(args.output, img)
    print(f"Saved to: {args.output}")
    print(f"GT boxes: {len(gt_boxes)} | S1: {len(s1_boxes)} | S2: {len(s2_boxes)} | S3: {len(s3_boxes)}")


if __name__ == "__main__":
    main()

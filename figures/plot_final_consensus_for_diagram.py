"""
Compute and plot the FINAL L-Spot detection for a single slice: the box
that survives both slice-consistency (Rule B, checked separately in the
evaluation pipeline) and multi-source >=2/3 agreement (IoU >= 0.30).

This is meant as the "after" image for the diagram, to contrast with
the messy multi-box "before" image from the three raw sources.

Usage:
    python plot_final_consensus_for_diagram.py --slice P0004_C1_slice_012
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd


def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def yolo_to_xyxy(x_center, y_center, w, h, img_w, img_h):
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h
    return int(x1), int(y1), int(x2), int(y2)


def load_gt_boxes(label_path, img_w, img_h):
    boxes = []
    if not Path(label_path).exists():
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
    df = pd.read_csv(csv_path)
    rows = df[(df["filename"] == slice_name) & (df["confidence"] >= conf_thresh)]
    boxes = []
    for _, row in rows.iterrows():
        boxes.append((row.x1, row.y1, row.x2, row.y2, row.confidence))
    return boxes


def consensus_filter(s1_boxes, s2_boxes, s3_boxes, iou_thresh=0.30, min_agree=2):
    """
    For each candidate box (taken from the union of all sources), count how
    many of the OTHER sources have a box with IoU >= iou_thresh against it.
    Keep candidates where total agreement (including itself) >= min_agree.
    Among kept candidates, merge overlapping ones and keep the highest
    confidence box as the representative final box.
    """
    all_candidates = []
    for src_idx, boxes in enumerate([s1_boxes, s2_boxes, s3_boxes]):
        for b in boxes:
            all_candidates.append((src_idx, b))

    kept = []
    for src_idx, cand in all_candidates:
        cand_box = cand[:4]
        agree_count = 1  # counts itself
        for other_idx, other_boxes in enumerate([s1_boxes, s2_boxes, s3_boxes]):
            if other_idx == src_idx:
                continue
            for ob in other_boxes:
                if iou(cand_box, ob[:4]) >= iou_thresh:
                    agree_count += 1
                    break
        if agree_count >= min_agree:
            kept.append(cand)

    if not kept:
        return []

    # Merge kept boxes that overlap into clusters, keep highest-confidence box per cluster
    kept = sorted(kept, key=lambda b: b[4], reverse=True)
    final = []
    used = [False] * len(kept)
    for i, b in enumerate(kept):
        if used[i]:
            continue
        cluster = [b]
        used[i] = True
        for j in range(i + 1, len(kept)):
            if used[j]:
                continue
            if iou(b[:4], kept[j][:4]) >= iou_thresh:
                cluster.append(kept[j])
                used[j] = True
        # representative = highest confidence in cluster (already sorted, so b)
        final.append(b)

    return final


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
        default="./diagram_final_consensus_example.png",
    )
    parser.add_argument("--iou_thresh", type=float, default=0.30)
    parser.add_argument("--min_agree", type=int, default=2)
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

    gt_boxes = load_gt_boxes(labels_dir / f"{args.slice}.txt", img_w, img_h)

    s1_boxes = load_source_boxes(Path(args.runs_root) / "raw_detections.csv", args.slice, 0.10)
    s2_boxes = load_source_boxes(Path(args.runs_root) / "raw_detections_tta.csv", args.slice, 0.30)
    s3_boxes = load_source_boxes(Path(args.runs_root) / "raw_detections_yolo10.csv", args.slice, 0.10)

    final_boxes = consensus_filter(
        s1_boxes, s2_boxes, s3_boxes,
        iou_thresh=args.iou_thresh, min_agree=args.min_agree,
    )

    GREEN = (0, 200, 0)   # ground truth
    BLUE = (255, 120, 0)  # final L-Spot detection

    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 3)

    for box in final_boxes:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), BLUE, 3)

        label = f"L-Spot {conf:.2f}"
        label_x, label_y = int(x1), max(int(y1) - 10, 18)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        # filled background box behind the text for contrast
        cv2.rectangle(
            img,
            (label_x - 2, label_y - text_h - 4),
            (label_x + text_w + 2, label_y + baseline),
            (255, 255, 255), -1,
        )
        cv2.putText(img, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(args.output, img)
    print(f"Saved to: {args.output}")
    print(f"GT boxes: {len(gt_boxes)} | Final L-Spot detections kept: {len(final_boxes)}")
    print(f"(S1={len(s1_boxes)}, S2={len(s2_boxes)}, S3={len(s3_boxes)} candidates before filtering)")


if __name__ == "__main__":
    main()

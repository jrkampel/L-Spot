"""
YOLO + CNN-ViT Hybrid Integration: False Positive Reduction
================================================================
Takes YOLO's raw detections on the test set, crops a patch around each
detection, and passes it through the trained CNN+ViT hybrid classifier.
Detections the hybrid labels as "not tumour" are discarded as false positives.

Outputs a filtered detection CSV and evaluation metrics (Precision, Recall, F1)
compared against the unfiltered YOLO baseline.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import csv
from pathlib import Path
from collections import defaultdict

# Config
BASE_DIR         = Path(".")
RAW_DETECTIONS   = BASE_DIR / "runs" / "raw_detections.csv"
TEST_IMAGES_DIR  = BASE_DIR / "dataset" / "test" / "images"
TEST_LABELS_DIR  = BASE_DIR / "dataset" / "test" / "labels"
MODEL_PATH       = BASE_DIR / "runs" / "cnn_vit_hybrid" / "best_cnn_vit_hybrid.pt"
OUTPUT_DIR       = BASE_DIR / "runs" / "cnn_vit_filtering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE     = 64
IMG_SIZE       = 512
IOU_THRESHOLD  = 0.30
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

print(f"Using device: {DEVICE}")

# CNN+ViT Hybrid Model definition (must match training script exactly)
class CNNViTHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_out_dim = resnet.fc.in_features

        vit = models.vit_b_16(weights=None)
        self.vit_backbone = vit
        self.vit_backbone.heads = nn.Identity()
        self.vit_out_dim = 768

        fusion_dim = self.cnn_out_dim + self.vit_out_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        cnn_feat = self.cnn_backbone(x)
        cnn_feat = cnn_feat.flatten(1)
        vit_feat = self.vit_backbone(x)
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        out = self.fusion_head(combined)
        return out


# Load model
print("Loading CNN+ViT hybrid model...")
model = CNNViTHybrid()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Load raw detections
print(f"Loading raw detections from {RAW_DETECTIONS}")
detections = []
with open(RAW_DETECTIONS, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        detections.append(row)

print(f"Loaded {len(detections)} detections")

# Classify each detection
def extract_patch(img, cx, cy, patch_size):
    h, w = img.shape[:2]
    half = patch_size // 2
    x1 = int(cx - half); y1 = int(cy - half)
    x2 = x1 + patch_size; y2 = y1 + patch_size
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > w: x1 -= (x2 - w); x2 = w
    if y2 > h: y1 -= (y2 - h); y2 = h
    x1, y1 = max(0, x1), max(0, y1)
    patch = img[y1:y2, x1:x2]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size))
    return patch


print("Running CNN+ViT hybrid on each detection...")

image_cache = {}
kept_detections = []
removed_detections = []

with torch.no_grad():
    for i, det in enumerate(detections):
        filename = det["filename"]
        if filename not in image_cache:
            img_path = TEST_IMAGES_DIR / f"{filename}.png"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_cache[filename] = img
        img = image_cache[filename]

        cx, cy = float(det["cx"]), float(det["cy"])
        patch = extract_patch(img, cx, cy, PATCH_SIZE)

        patch_tensor = transform(patch).unsqueeze(0).to(DEVICE)
        output = model(patch_tensor)
        pred = output.argmax(dim=1).item()
        confidence_clf = torch.softmax(output, dim=1)[0, pred].item()

        det["hybrid_pred"] = pred
        det["hybrid_conf"] = round(confidence_clf, 4)

        if pred == 1:
            kept_detections.append(det)
        else:
            removed_detections.append(det)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(detections)} detections...")

print(f"\nHybrid filtering complete.")
print(f"Kept (classified as tumour)    : {len(kept_detections)}")
print(f"Removed (classified as healthy): {len(removed_detections)}")

# Save filtered detections
fieldnames = list(detections[0].keys()) + ["hybrid_pred", "hybrid_conf"]

with open(OUTPUT_DIR / "hybrid_filtered_detections.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept_detections)

with open(OUTPUT_DIR / "hybrid_removed_detections.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(removed_detections)

print(f"Saved filtered results to {OUTPUT_DIR}")

# Evaluate against ground truth
def load_gt_boxes():
    gt = {}
    for label_file in TEST_LABELS_DIR.glob("*.txt"):
        stem = label_file.stem
        boxes = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, w, h = map(float, parts)
                    x1 = (cx - w / 2) * IMG_SIZE
                    y1 = (cy - h / 2) * IMG_SIZE
                    x2 = (cx + w / 2) * IMG_SIZE
                    y2 = (cy + h / 2) * IMG_SIZE
                    boxes.append([x1, y1, x2, y2])
        gt[stem] = boxes
    for img_file in TEST_IMAGES_DIR.glob("*.png"):
        if img_file.stem not in gt:
            gt[img_file.stem] = []
    return gt


def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate(det_list, gt_boxes):
    by_image = defaultdict(list)
    for d in det_list:
        by_image[d["filename"]].append([float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])])

    tp, fp, fn = 0, 0, 0
    all_stems = set(gt_boxes.keys()) | set(by_image.keys())

    for stem in all_stems:
        gts  = list(gt_boxes.get(stem, []))
        dets = list(by_image.get(stem, []))
        matched_gt = set()
        for det_box in dets:
            best_iou, best_idx = 0.0, -1
            for j, gt_box in enumerate(gts):
                if j in matched_gt:
                    continue
                iou_val = iou(det_box, gt_box)
                if iou_val > best_iou:
                    best_iou, best_idx = iou_val, j
            if best_iou >= IOU_THRESHOLD:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn += len(gts) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


print("\nEvaluating against ground truth...")
gt_boxes = load_gt_boxes()

baseline_metrics = evaluate(detections, gt_boxes)
filtered_metrics = evaluate(kept_detections, gt_boxes)

print(f"\n{'Config':<35}{'Precision':>10}{'Recall':>10}{'F1':>10}{'TP':>6}{'FP':>6}{'FN':>6}")
print("-" * 85)
print(f"{'Unfiltered YOLO (baseline)':<35}{baseline_metrics['precision']:>10}{baseline_metrics['recall']:>10}"
      f"{baseline_metrics['f1']:>10}{baseline_metrics['tp']:>6}{baseline_metrics['fp']:>6}{baseline_metrics['fn']:>6}")
print(f"{'YOLO + CNN-ViT Hybrid filtering':<35}{filtered_metrics['precision']:>10}{filtered_metrics['recall']:>10}"
      f"{filtered_metrics['f1']:>10}{filtered_metrics['tp']:>6}{filtered_metrics['fp']:>6}{filtered_metrics['fn']:>6}")

with open(OUTPUT_DIR / "hybrid_evaluation_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["config", "precision", "recall", "f1", "tp", "fp", "fn"])
    writer.writerow(["Unfiltered YOLO (baseline)", baseline_metrics["precision"], baseline_metrics["recall"],
                      baseline_metrics["f1"], baseline_metrics["tp"], baseline_metrics["fp"], baseline_metrics["fn"]])
    writer.writerow(["YOLO + CNN-ViT Hybrid filtering", filtered_metrics["precision"], filtered_metrics["recall"],
                      filtered_metrics["f1"], filtered_metrics["tp"], filtered_metrics["fp"], filtered_metrics["fn"]])

print(f"\nSummary saved to {OUTPUT_DIR / 'hybrid_evaluation_summary.csv'}")
print("Done.")

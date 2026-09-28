"""
Extract Classifier Patches from CT Test Set and Evaluate Classifier v1
=======================================================================
Extracts tumour and healthy patches from the held-out CT test images
(completely unseen during training) and evaluates the best saved
ResNet18 classifier on them.

This gives a genuine test set result for healthy vs tumour classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
from pathlib import Path
import random

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path(".")
DATASET_DIR   = BASE_DIR / "dataset"
MODEL_PATH    = BASE_DIR / "runs" / "classifier" / "best_classifier.pt"
OUTPUT_DIR    = BASE_DIR / "classifier_dataset_test"
PATCH_SIZE    = 64
SEARCH_RADIUS = 100
MAX_ATTEMPTS  = 30
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
torch.manual_seed(42)

print(f"Using device: {DEVICE}")
print(f"Extracting patches from CT test set...")

# ── Helper functions (same as original extraction script) ─────────────────────
def read_yolo_labels(label_path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, w, h = map(float, parts)
                boxes.append((cx, cy, w, h))
    return boxes

def yolo_to_pixel_box(box, img_w, img_h):
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2

def boxes_overlap(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    return x2 > x1 and y2 > y1

def extract_patch(img, cx, cy, patch_size):
    h, w = img.shape[:2]
    half = patch_size // 2
    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > w: x1 -= (x2 - w); x2 = w
    if y2 > h: y1 -= (y2 - h); y2 = h
    x1, y1 = max(0, x1), max(0, y1)
    return img[y1:y2, x1:x2]

def find_healthy_patch(img, tumour_boxes_pixel, img_w, img_h):
    anchor = random.choice(tumour_boxes_pixel)
    anchor_cx = (anchor[0] + anchor[2]) / 2
    anchor_cy = (anchor[1] + anchor[3]) / 2
    half = PATCH_SIZE // 2
    for _ in range(MAX_ATTEMPTS):
        angle = random.uniform(0, 2 * np.pi)
        dist  = random.uniform(PATCH_SIZE, SEARCH_RADIUS)
        cx = anchor_cx + dist * np.cos(angle)
        cy = anchor_cy + dist * np.sin(angle)
        if cx - half < 0 or cx + half > img_w or cy - half < 0 or cy + half > img_h:
            continue
        candidate_box = [cx - half, cy - half, cx + half, cy + half]
        if any(boxes_overlap(candidate_box, tb) for tb in tumour_boxes_pixel):
            continue
        return cx, cy
    return None, None

# ── Extract patches from CT test set ─────────────────────────────────────────
img_dir = DATASET_DIR / "test" / "images"
lbl_dir = DATASET_DIR / "test" / "labels"

out_tumour_dir  = OUTPUT_DIR / "tumour"
out_healthy_dir = OUTPUT_DIR / "healthy"
out_tumour_dir.mkdir(parents=True, exist_ok=True)
out_healthy_dir.mkdir(parents=True, exist_ok=True)

image_files   = sorted(img_dir.glob("*.png"))
total_tumour  = 0
total_healthy = 0
total_skipped = 0

print(f"Found {len(image_files)} test images. Extracting patches...\n")

for img_path in image_files:
    stem     = img_path.stem
    lbl_path = lbl_dir / (stem + ".txt")

    boxes_norm = read_yolo_labels(lbl_path)
    if not boxes_norm:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    img_h, img_w  = img.shape[:2]
    boxes_pixel   = [yolo_to_pixel_box(b, img_w, img_h) for b in boxes_norm]

    for i, box in enumerate(boxes_pixel):
        tx1, ty1, tx2, ty2 = box
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2

        tumour_patch = extract_patch(img, tcx, tcy, PATCH_SIZE)
        if tumour_patch.shape[0] != PATCH_SIZE or tumour_patch.shape[1] != PATCH_SIZE:
            total_skipped += 1
            continue

        cv2.imwrite(str(out_tumour_dir / f"{stem}_t{i:02d}.png"), tumour_patch)
        total_tumour += 1

        hcx, hcy = find_healthy_patch(img, boxes_pixel, img_w, img_h)
        if hcx is None:
            total_skipped += 1
            continue

        healthy_patch = extract_patch(img, hcx, hcy, PATCH_SIZE)
        if healthy_patch.shape[0] != PATCH_SIZE or healthy_patch.shape[1] != PATCH_SIZE:
            total_skipped += 1
            continue

        cv2.imwrite(str(out_healthy_dir / f"{stem}_h{i:02d}.png"), healthy_patch)
        total_healthy += 1

print(f"Extraction complete.")
print(f"  Tumour patches:  {total_tumour}")
print(f"  Healthy patches: {total_healthy}")
print(f"  Skipped:         {total_skipped}\n")

# ── Dataset ───────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, tumour_dir, healthy_dir, transform=None):
        self.samples  = []
        self.transform = transform
        for f in sorted(tumour_dir.glob("*.png")):
            self.samples.append((f, 1))
        for f in sorted(healthy_dir.glob("*.png")):
            self.samples.append((f, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_dataset = PatchDataset(out_tumour_dir, out_healthy_dir,
                            transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=32,
                          shuffle=False, num_workers=4)

n_tumour  = sum(1 for _, l in test_dataset.samples if l == 1)
n_healthy = sum(1 for _, l in test_dataset.samples if l == 0)
print(f"Test patch dataset: {len(test_dataset)} patches "
      f"(tumour: {n_tumour}, healthy: {n_healthy})\n")

# ── Load model ────────────────────────────────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded. Running evaluation...\n")

# ── Evaluate ──────────────────────────────────────────────────────────────────
tp, fp, fn, tn = 0, 0, 0, 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds   = outputs.argmax(dim=1)
        for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            if   p == 1 and l == 1: tp += 1
            elif p == 1 and l == 0: fp += 1
            elif p == 0 and l == 1: fn += 1
            else:                   tn += 1

# ── Metrics ───────────────────────────────────────────────────────────────────
total    = tp + fp + fn + tn
accuracy = (tp + tn) / total

prec_tumour  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec_tumour   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_tumour    = (2 * prec_tumour * rec_tumour / (prec_tumour + rec_tumour)
                if (prec_tumour + rec_tumour) > 0 else 0.0)

prec_healthy = tn / (tn + fn) if (tn + fn) > 0 else 0.0
rec_healthy  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
f1_healthy   = (2 * prec_healthy * rec_healthy / (prec_healthy + rec_healthy)
                if (prec_healthy + rec_healthy) > 0 else 0.0)

macro_prec = (prec_tumour + prec_healthy) / 2
macro_rec  = (rec_tumour  + rec_healthy)  / 2
macro_f1   = (f1_tumour   + f1_healthy)   / 2

print("=" * 58)
print("  Classifier v1 (ResNet18) — Genuine Test Set Results")
print("  (patches extracted from held-out CT test images)")
print("=" * 58)
print(f"\n  Total patches evaluated: {total}")
print(f"  Overall Accuracy:        {accuracy*100:.2f}%")
print(f"\n  Confusion matrix:")
print(f"    TP (tumour  → tumour):  {tp:>5}")
print(f"    TN (healthy → healthy): {tn:>5}")
print(f"    FP (healthy → tumour):  {fp:>5}")
print(f"    FN (tumour  → healthy): {fn:>5}")
print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"  {'-'*46}")
print(f"  {'Tumour':<12} {prec_tumour*100:>9.2f}% "
      f"{rec_tumour*100:>9.2f}% {f1_tumour*100:>9.2f}%")
print(f"  {'Healthy':<12} {prec_healthy*100:>9.2f}% "
      f"{rec_healthy*100:>9.2f}% {f1_healthy*100:>9.2f}%")
print(f"  {'-'*46}")
print(f"  {'Macro avg':<12} {macro_prec*100:>9.2f}% "
      f"{macro_rec*100:>9.2f}% {macro_f1*100:>9.2f}%")
print("=" * 58)

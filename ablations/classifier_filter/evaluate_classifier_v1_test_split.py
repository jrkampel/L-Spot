"""
Split Validation Set and Evaluate Classifier v1 on Held-Out Test Portion
=========================================================================
Takes 50% of the validation patches as a test set and evaluates the
best saved classifier v1 (ResNet18, random negatives) on them.

Note: this test split is derived from the original validation set.
The model did not directly train on these patches, but the val set
was used for early stopping checkpoint selection.
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
BASE_DIR    = Path(".")
DATASET_DIR = BASE_DIR / "classifier_dataset" / "val"
MODEL_PATH  = BASE_DIR / "runs" / "classifier" / "best_classifier.pt"

BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42
TEST_FRAC   = 0.5   # 50% of val used as test set

random.seed(SEED)
torch.manual_seed(SEED)

print(f"Using device: {DEVICE}")
print(f"Model:        {MODEL_PATH}")
print(f"Val set:      {DATASET_DIR}")
print(f"Test fraction: {TEST_FRAC*100:.0f}% of val patches\n")

# ── Collect all patches ───────────────────────────────────────────────────────
tumour_files  = sorted((DATASET_DIR / "tumour").glob("*.png"))
healthy_files = sorted((DATASET_DIR / "healthy").glob("*.png"))

# Shuffle with fixed seed for reproducibility
random.shuffle(tumour_files)
random.shuffle(healthy_files)

# Split
n_tumour_test  = int(len(tumour_files)  * TEST_FRAC)
n_healthy_test = int(len(healthy_files) * TEST_FRAC)

test_files  = ([(f, 1) for f in tumour_files[:n_tumour_test]] +
               [(f, 0) for f in healthy_files[:n_healthy_test]])
val_files   = ([(f, 1) for f in tumour_files[n_tumour_test:]] +
               [(f, 0) for f in healthy_files[n_healthy_test:]])

print(f"Original val set:  {len(tumour_files)} tumour, {len(healthy_files)} healthy")
print(f"New test split:    {n_tumour_test} tumour, {n_healthy_test} healthy "
      f"({len(test_files)} total)")
print(f"Remaining val:     {len(tumour_files)-n_tumour_test} tumour, "
      f"{len(healthy_files)-n_healthy_test} healthy ({len(val_files)} total)\n")

# ── Dataset ───────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_dataset = PatchDataset(test_files, transform=val_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4)

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully. Running evaluation on test split...\n")

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

# Tumour class
prec_tumour = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec_tumour  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_tumour   = (2 * prec_tumour * rec_tumour / (prec_tumour + rec_tumour)
               if (prec_tumour + rec_tumour) > 0 else 0.0)

# Healthy class
prec_healthy = tn / (tn + fn) if (tn + fn) > 0 else 0.0
rec_healthy  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
f1_healthy   = (2 * prec_healthy * rec_healthy / (prec_healthy + rec_healthy)
                if (prec_healthy + rec_healthy) > 0 else 0.0)

# Macro average
macro_prec = (prec_tumour + prec_healthy) / 2
macro_rec  = (rec_tumour  + rec_healthy)  / 2
macro_f1   = (f1_tumour   + f1_healthy)   / 2

print("=" * 58)
print("  Classifier v1 (ResNet18) — Test Split Results")
print("  (50% held-out portion of original validation set)")
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

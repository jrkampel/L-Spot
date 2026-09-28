"""
Train Binary Classifier: Tumour vs Healthy Tissue
====================================================
Fine-tunes a pretrained ResNet18 on the extracted patch dataset to classify
small image crops as tumour (1) or healthy (0).

This classifier will later be used to filter YOLO false positives.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
from pathlib import Path
import random

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(".")
DATASET_DIR = BASE_DIR / "classifier_dataset"
OUTPUT_DIR  = BASE_DIR / "runs" / "classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-4
PATCH_SIZE   = 64
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE     = 7   # early stopping

random.seed(42)
torch.manual_seed(42)

print(f"Using device: {DEVICE}")

# ── Dataset ─────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        tumour_dir  = root_dir / "tumour"
        healthy_dir = root_dir / "healthy"

        for f in sorted(tumour_dir.glob("*.png")):
            self.samples.append((f, 1))
        for f in sorted(healthy_dir.glob("*.png")):
            self.samples.append((f, 0))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label


# ── Transforms ──────────────────────────────────────────────────────────────
# ImageNet normalisation since we're using pretrained ResNet18
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),   # ResNet expects 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Load datasets ───────────────────────────────────────────────────────────
train_dataset = PatchDataset(DATASET_DIR / "train", transform=train_transform)
val_dataset   = PatchDataset(DATASET_DIR / "val",   transform=val_transform)

print(f"Train samples: {len(train_dataset)}  (tumour: {sum(1 for _, l in train_dataset.samples if l == 1)}, "
      f"healthy: {sum(1 for _, l in train_dataset.samples if l == 0)})")
print(f"Val samples:   {len(val_dataset)}  (tumour: {sum(1 for _, l in val_dataset.samples if l == 1)}, "
      f"healthy: {sum(1 for _, l in val_dataset.samples if l == 0)})")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ── Model ───────────────────────────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)   # binary classification: healthy vs tumour
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# ── Training loop ───────────────────────────────────────────────────────────
best_val_loss = float("inf")
epochs_no_improve = 0

print(f"\n{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<14}{'Val Acc':<12}{'Val Prec':<12}{'Val Recall':<12}")
print("-" * 70)

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)

    train_loss /= len(train_dataset)

    # --- Validate ---
    model.eval()
    val_loss = 0.0
    tp, fp, fn, tn = 0, 0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                if p == 1 and l == 1: tp += 1
                elif p == 1 and l == 0: fp += 1
                elif p == 0 and l == 1: fn += 1
                else: tn += 1

    val_loss /= len(val_dataset)
    val_acc  = (tp + tn) / (tp + tn + fp + fn)
    val_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    val_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    scheduler.step(val_loss)

    print(f"{epoch:<8}{train_loss:<14.4f}{val_loss:<14.4f}{val_acc:<12.4f}{val_prec:<12.4f}{val_rec:<12.4f}")

    # --- Early stopping & checkpoint ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), OUTPUT_DIR / "best_classifier.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print(f"\nTraining complete. Best model saved to {OUTPUT_DIR / 'best_classifier.pt'}")
print(f"Best validation loss: {best_val_loss:.4f}")

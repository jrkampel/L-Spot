"""
CNN + ViT Hybrid Model (Original Architecture Contribution)
================================================================
Two parallel branches process the same input patch:
  - CNN branch: ResNet18 (pretrained ImageNet) - captures local texture/edges
  - ViT branch: ViT-B/16 (pretrained ImageNet)  - captures global context

Their feature embeddings are concatenated and passed through a small
fusion classifier head to produce the final tumour vs not-tumour prediction.

Trained on classifier_dataset_v2 (tumour vs healthy+hard_negative merged),
the same dataset used for classifier v2, for a fair comparison.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
from pathlib import Path
import random

# Config
BASE_DIR    = Path(".")
DATASET_DIR = BASE_DIR / "classifier_dataset_v2"
OUTPUT_DIR  = BASE_DIR / "runs" / "cnn_vit_hybrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE   = 7

random.seed(42)
torch.manual_seed(42)

print(f"Using device: {DEVICE}")

# Dataset
class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        tumour_dir   = root_dir / "tumour"
        healthy_dir  = root_dir / "healthy"
        hard_neg_dir = root_dir / "hard_negative"

        for f in sorted(tumour_dir.glob("*.png")):
            self.samples.append((f, 1))
        for f in sorted(healthy_dir.glob("*.png")):
            self.samples.append((f, 0))
        for f in sorted(hard_neg_dir.glob("*.png")):
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


# Transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
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

# Load datasets
train_dataset = PatchDataset(DATASET_DIR / "train", transform=train_transform)
val_dataset   = PatchDataset(DATASET_DIR / "val",   transform=val_transform)

n_train_tumour = sum(1 for _, l in train_dataset.samples if l == 1)
n_train_neg    = sum(1 for _, l in train_dataset.samples if l == 0)
n_val_tumour   = sum(1 for _, l in val_dataset.samples if l == 1)
n_val_neg      = sum(1 for _, l in val_dataset.samples if l == 0)

print(f"Train samples: {len(train_dataset)}  (tumour: {n_train_tumour}, not-tumour: {n_train_neg})")
print(f"Val samples:   {len(val_dataset)}  (tumour: {n_val_tumour}, not-tumour: {n_val_neg})")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# CNN + ViT Hybrid Model
class CNNViTHybrid(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_out_dim = resnet.fc.in_features

        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
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


print("\nBuilding CNN+ViT hybrid model...")
model = CNNViTHybrid().to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# Training loop
best_val_loss = float("inf")
epochs_no_improve = 0

print(f"\n{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<14}{'Val Acc':<12}{'Val Prec':<12}{'Val Recall':<12}")
print("-" * 70)

for epoch in range(1, EPOCHS + 1):
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), OUTPUT_DIR / "best_cnn_vit_hybrid.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print(f"\nTraining complete. Best model saved to {OUTPUT_DIR / 'best_cnn_vit_hybrid.pt'}")
print(f"Best validation loss: {best_val_loss:.4f}")

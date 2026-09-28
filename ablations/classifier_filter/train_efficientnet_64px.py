import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
from pathlib import Path
import random

BASE_DIR    = Path(".")
DATASET_DIR = BASE_DIR / "classifier_dataset_v2"
OUTPUT_DIR  = BASE_DIR / "runs" / "efficientnet_64px"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE   = 10

random.seed(42)
torch.manual_seed(42)
print(f"Using device: {DEVICE}")

class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for f in sorted((root_dir / "tumour").glob("*.png")):
            self.samples.append((f, 1))
        for f in sorted((root_dir / "healthy").glob("*.png")):
            self.samples.append((f, 0))
        random.shuffle(self.samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        return img, label

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_dataset = PatchDataset(DATASET_DIR / "train", transform=train_transform)
val_dataset   = PatchDataset(DATASET_DIR / "val",   transform=val_transform)
print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features, 2))
model = model.to(DEVICE)
print(f"Model: EfficientNetV2-S 64px ({sum(p.numel() for p in model.parameters()):,} params)")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

best_val_loss     = float("inf")
epochs_no_improve = 0

print(f"\n{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<14}{'Val Acc':<12}{'Val Prec':<12}{'Val Recall':<12}")
print("-" * 72)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
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
            val_loss += criterion(outputs, labels).item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                if   p==1 and l==1: tp+=1
                elif p==1 and l==0: fp+=1
                elif p==0 and l==1: fn+=1
                else:               tn+=1

    val_loss /= len(val_dataset)
    val_acc  = (tp+tn)/(tp+tn+fp+fn)
    val_prec = tp/(tp+fp) if (tp+fp)>0 else 0
    val_rec  = tp/(tp+fn) if (tp+fn)>0 else 0

    scheduler.step(val_loss)
    print(f"{epoch:<8}{train_loss:<14.4f}{val_loss:<14.4f}{val_acc:<12.4f}{val_prec:<12.4f}{val_rec:<12.4f}")

    if val_loss < best_val_loss:
        best_val_loss     = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), OUTPUT_DIR / "best_efficientnet_64px.pt")
        print(f"  --> saved best model (val loss: {best_val_loss:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print(f"\nDone. Best model: {OUTPUT_DIR / 'best_efficientnet_64px.pt'}")

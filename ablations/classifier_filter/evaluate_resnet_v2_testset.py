import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
from pathlib import Path

BASE_DIR   = Path(".")
MODEL_PATH = BASE_DIR / "runs" / "classifier_v2" / "best_classifier_v2.pt"
TEST_DIR   = BASE_DIR / "classifier_dataset_test"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatchDataset(Dataset):
    def __init__(self, tumour_dir, healthy_dir, transform=None):
        self.samples = [(f, 1) for f in sorted(tumour_dir.glob("*.png"))] + \
                       [(f, 0) for f in sorted(healthy_dir.glob("*.png"))]
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        return img, label

transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

dataset = PatchDataset(TEST_DIR/"tumour", TEST_DIR/"healthy", transform=transform)
loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
print(f"Test patches: {len(dataset)}")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

tp, fp, fn, tn = 0, 0, 0, 0
with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            if   p==1 and l==1: tp+=1
            elif p==1 and l==0: fp+=1
            elif p==0 and l==1: fn+=1
            else:               tn+=1

total        = tp+fp+fn+tn
accuracy     = (tp+tn)/total
prec_tumour  = tp/(tp+fp) if (tp+fp)>0 else 0
rec_tumour   = tp/(tp+fn) if (tp+fn)>0 else 0
f1_tumour    = 2*prec_tumour*rec_tumour/(prec_tumour+rec_tumour) if (prec_tumour+rec_tumour)>0 else 0
prec_healthy = tn/(tn+fn) if (tn+fn)>0 else 0
rec_healthy  = tn/(tn+fp) if (tn+fp)>0 else 0
f1_healthy   = 2*prec_healthy*rec_healthy/(prec_healthy+rec_healthy) if (prec_healthy+rec_healthy)>0 else 0

print(f"\n{'='*55}")
print(f"  ResNet18 v2 (hard negatives) — Genuine Test Set Results")
print(f"{'='*55}")
print(f"  Total: {total}  |  Accuracy: {accuracy*100:.2f}%")
print(f"\n  Confusion matrix:")
print(f"    TP: {tp}  FN: {fn}")
print(f"    FP: {fp}  TN: {tn}")
print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"  {'-'*44}")
print(f"  {'Tumour':<12} {prec_tumour*100:>9.2f}% {rec_tumour*100:>9.2f}% {f1_tumour*100:>9.2f}%")
print(f"  {'Healthy':<12} {prec_healthy*100:>9.2f}% {rec_healthy*100:>9.2f}% {f1_healthy*100:>9.2f}%")
print(f"  {'Macro avg':<12} {(prec_tumour+prec_healthy)*50:>9.2f}% {(rec_tumour+rec_healthy)*50:>9.2f}% {(f1_tumour+f1_healthy)*50:>9.2f}%")
print(f"{'='*55}")

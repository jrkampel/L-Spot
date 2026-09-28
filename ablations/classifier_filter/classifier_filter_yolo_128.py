import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR        = Path(".")
RAW_DETECTIONS  = BASE_DIR / "runs" / "raw_detections.csv"
TEST_IMAGES_DIR = BASE_DIR / "dataset" / "test" / "images"
TEST_LABELS_DIR = BASE_DIR / "dataset" / "test" / "labels"
CLASSIFIER_PATH = BASE_DIR / "runs" / "resnet_128_native" / "best_resnet_128_native.pt"
OUTPUT_DIR      = BASE_DIR / "runs" / "classifier_filtering_128"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE    = 128
IMG_SIZE      = 512
IOU_THRESHOLD = 0.30
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

print(f"Using device: {DEVICE}")
print("Loading ResNet18 128px native classifier...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

print(f"Loading raw detections from {RAW_DETECTIONS}")
detections = []
with open(RAW_DETECTIONS, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        detections.append(row)
print(f"Loaded {len(detections)} detections")

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

print("Running classifier on each detection...")
image_cache = {}
kept_detections = []
removed_detections = []

with torch.no_grad():
    for i, det in enumerate(detections):
        filename = det["filename"]
        if filename not in image_cache:
            img = cv2.imread(str(TEST_IMAGES_DIR / f"{filename}.png"))
            if img is None: continue
            image_cache[filename] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = image_cache[filename]

        cx, cy = float(det["cx"]), float(det["cy"])
        patch = extract_patch(img, cx, cy, PATCH_SIZE)
        output = model(transform(patch).unsqueeze(0).to(DEVICE))
        pred   = output.argmax(dim=1).item()
        conf   = torch.softmax(output, dim=1)[0, pred].item()

        det["classifier_pred"] = pred
        det["classifier_conf"] = round(conf, 4)

        if pred == 1:
            kept_detections.append(det)
        else:
            removed_detections.append(det)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(detections)} detections...")

print(f"\nKept:    {len(kept_detections)}")
print(f"Removed: {len(removed_detections)}")

fieldnames = list(detections[0].keys()) + ["classifier_pred", "classifier_conf"]
with open(OUTPUT_DIR / "filtered_detections.csv", "w", newline="") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    csv.DictWriter(f, fieldnames=fieldnames).writerows(kept_detections)

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
                    x1 = (cx - w/2) * IMG_SIZE; y1 = (cy - h/2) * IMG_SIZE
                    x2 = (cx + w/2) * IMG_SIZE; y2 = (cy + h/2) * IMG_SIZE
                    boxes.append([x1, y1, x2, y2])
        gt[stem] = boxes
    for img_file in TEST_IMAGES_DIR.glob("*.png"):
        if img_file.stem not in gt:
            gt[img_file.stem] = []
    return gt

def iou(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    if x2<=x1 or y2<=y1: return 0.0
    inter=(x2-x1)*(y2-y1)
    union=(b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter
    return inter/union if union>0 else 0.0

def evaluate(det_list, gt_boxes):
    by_image = defaultdict(list)
    for d in det_list:
        by_image[d["filename"]].append([float(d["x1"]),float(d["y1"]),float(d["x2"]),float(d["y2"])])
    tp, fp, fn = 0, 0, 0
    for stem in set(gt_boxes.keys()) | set(by_image.keys()):
        gts  = list(gt_boxes.get(stem, []))
        dets = list(by_image.get(stem, []))
        matched = set()
        for det_box in dets:
            best_iou, best_idx = 0.0, -1
            for j, gt_box in enumerate(gts):
                if j in matched: continue
                v = iou(det_box, gt_box)
                if v > best_iou: best_iou, best_idx = v, j
            if best_iou >= IOU_THRESHOLD:
                tp += 1; matched.add(best_idx)
            else:
                fp += 1
        fn += len(gts) - len(matched)
    p = tp/(tp+fp) if (tp+fp)>0 else 0
    r = tp/(tp+fn) if (tp+fn)>0 else 0
    f = 2*p*r/(p+r) if (p+r)>0 else 0
    return {"precision":round(p,3),"recall":round(r,3),"f1":round(f,3),"tp":tp,"fp":fp,"fn":fn}

print("\nEvaluating against ground truth...")
gt_boxes = load_gt_boxes()
baseline = evaluate(detections, gt_boxes)
filtered = evaluate(kept_detections, gt_boxes)

print(f"\n{'Config':<40}{'Precision':>10}{'Recall':>10}{'F1':>10}{'TP':>6}{'FP':>6}{'FN':>6}")
print("-" * 90)
print(f"{'Unfiltered YOLO (baseline)':<40}{baseline['precision']:>10}{baseline['recall']:>10}{baseline['f1']:>10}{baseline['tp']:>6}{baseline['fp']:>6}{baseline['fn']:>6}")
print(f"{'YOLO + ResNet18 128px native':<40}{filtered['precision']:>10}{filtered['recall']:>10}{filtered['f1']:>10}{filtered['tp']:>6}{filtered['fp']:>6}{filtered['fn']:>6}")
print("\nDone.")

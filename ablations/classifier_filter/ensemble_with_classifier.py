import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR        = Path(".")
TEST_IMAGES_DIR = BASE_DIR / "dataset" / "test" / "images"
TEST_LABELS_DIR = BASE_DIR / "dataset" / "test" / "labels"
OUTPUT_DIR      = BASE_DIR / "runs" / "ensemble_with_classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STANDARD_CSV    = BASE_DIR / "runs" / "raw_detections.csv"
TTA_CSV         = BASE_DIR / "runs" / "raw_detections_tta.csv"
YOLO10_CSV      = BASE_DIR / "runs" / "raw_detections_yolo10.csv"
CLASSIFIER_PATH = BASE_DIR / "runs" / "resnet_128_native" / "best_resnet_128_native.pt"

PATCH_SIZE    = 128
IMG_SIZE      = 512
IOU_THRESHOLD = 0.30
TTA_CONF_GATE = 0.30
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

print(f"Using device: {DEVICE}")

print("Loading classifier...")
classifier = models.resnet18(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 2)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
classifier = classifier.to(DEVICE)
classifier.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

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
    patch = img[y1:y2, x1:x2]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size))
    return patch

def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0.0

def classify_detections(detections, image_cache):
    kept = []
    with torch.no_grad():
        for det in detections:
            filename = det["filename"]
            if filename not in image_cache:
                img = cv2.imread(str(TEST_IMAGES_DIR / f"{filename}.png"))
                if img is None:
                    continue
                image_cache[filename] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = image_cache[filename]
            cx = (float(det["x1"]) + float(det["x2"])) / 2
            cy = (float(det["y1"]) + float(det["y2"])) / 2
            patch = extract_patch(img, cx, cy, PATCH_SIZE)
            output = classifier(transform(patch).unsqueeze(0).to(DEVICE))
            pred = output.argmax(dim=1).item()
            if pred == 1:
                kept.append(det)
    return kept

def load_csv(path):
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

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
                    x1 = (cx - w/2) * IMG_SIZE
                    y1 = (cy - h/2) * IMG_SIZE
                    x2 = (cx + w/2) * IMG_SIZE
                    y2 = (cy + h/2) * IMG_SIZE
                    boxes.append([x1, y1, x2, y2])
        gt[stem] = boxes
    for img_file in TEST_IMAGES_DIR.glob("*.png"):
        if img_file.stem not in gt:
            gt[img_file.stem] = []
    return gt

def evaluate(det_list, gt_boxes):
    by_image = defaultdict(list)
    for d in det_list:
        by_image[d["filename"]].append(
            [float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])])
    tp, fp, fn = 0, 0, 0
    for stem in set(gt_boxes.keys()) | set(by_image.keys()):
        gts  = list(gt_boxes.get(stem, []))
        dets = list(by_image.get(stem, []))
        matched = set()
        for det_box in dets:
            best_iou, best_idx = 0.0, -1
            for j, gt_box in enumerate(gts):
                if j in matched:
                    continue
                v = iou(det_box, gt_box)
                if v > best_iou:
                    best_iou, best_idx = v, j
            if best_iou >= IOU_THRESHOLD:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1
        fn += len(gts) - len(matched)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"precision": round(p, 3), "recall": round(r, 3),
            "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}

def group_by_image(dets):
    by_img = defaultdict(list)
    for d in dets:
        by_img[d["filename"]].append(
            [float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])])
    return by_img

print("Loading raw detections...")
standard_dets = load_csv(STANDARD_CSV)
tta_dets      = [d for d in load_csv(TTA_CSV) if float(d["confidence"]) >= TTA_CONF_GATE]
yolo10_dets   = load_csv(YOLO10_CSV)
print(f"Standard: {len(standard_dets)}")
print(f"TTA (conf>={TTA_CONF_GATE}): {len(tta_dets)}")
print(f"YOLOv10m: {len(yolo10_dets)}")

image_cache = {}
print("Classifying Standard YOLOv8m detections...")
standard_filtered = classify_detections(standard_dets, image_cache)
print(f"  Kept: {len(standard_filtered)}/{len(standard_dets)}")

print("Classifying TTA YOLOv8m detections...")
tta_filtered = classify_detections(tta_dets, image_cache)
print(f"  Kept: {len(tta_filtered)}/{len(tta_dets)}")

print("Classifying YOLOv10m detections...")
yolo10_filtered = classify_detections(yolo10_dets, image_cache)
print(f"  Kept: {len(yolo10_filtered)}/{len(yolo10_dets)}")

print("Running 3-way majority vote...")
std_by_img = group_by_image(standard_filtered)
tta_by_img = group_by_image(tta_filtered)
y10_by_img = group_by_image(yolo10_filtered)

all_images = set(std_by_img.keys()) | set(tta_by_img.keys()) | set(y10_by_img.keys())

ensemble_2of3 = []
ensemble_3of3 = []

for img in all_images:
    sources = [std_by_img.get(img, []), tta_by_img.get(img, []), y10_by_img.get(img, [])]
    all_boxes = [box for src in sources for box in src]
    matched = set()
    for i, box in enumerate(all_boxes):
        if i in matched:
            continue
        votes = 1
        matched.add(i)
        for j, other in enumerate(all_boxes):
            if j <= i or j in matched:
                continue
            if iou(box, other) >= IOU_THRESHOLD:
                votes += 1
                matched.add(j)
        if votes >= 2:
            ensemble_2of3.append({"filename": img, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
        if votes >= 3:
            ensemble_3of3.append({"filename": img, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})

print(f"Ensemble (>=2 of 3): {len(ensemble_2of3)} detections")
print(f"Ensemble (all 3):    {len(ensemble_3of3)} detections")

print("Evaluating against ground truth...")
gt_boxes = load_gt_boxes()

results = [
    ("Unfiltered YOLO (baseline)",    evaluate(standard_dets,    gt_boxes)),
    ("Standard YOLO + classifier",    evaluate(standard_filtered, gt_boxes)),
    ("3-way + classifier (>=2 of 3)", evaluate(ensemble_2of3,    gt_boxes)),
    ("3-way + classifier (all 3)",    evaluate(ensemble_3of3,    gt_boxes)),
]

print(f"\n{'Config':<45} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
print("-" * 80)
for name, m in results:
    print(f"{name:<45} {m['precision']:>6} {m['recall']:>6} {m['f1']:>6} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")
print("Done.")

# L-Spot

Code for L-Spot, a retraining-free inference-time ensemble for hepatocellular carcinoma (HCC) detection in arterial-phase contrast-enhanced CT. Presented at the CaPTion Workshop, MICCAI 2026.

L-Spot combines three detection sources and keeps detections that at least two of them agree on:

1. YOLOv8m
2. YOLOv8m with test-time augmentation (TTA)
3. YOLOv10m, trained independently

Detections are then filtered for consistency across consecutive axial slices.

Final test-set result: F1 = 0.733, precision = 0.820, recall = 0.664.

## Repository layout

| Folder | Contents |
|---|---|
| `configs/` | Dataset config and the saved YOLOv8m and YOLOv10m training arguments |
| `detection/` | Raw detection extraction for each source, TTA evaluation |
| `ensemble/` | 2-of-3 voting and agreement-confidence sweep |
| `postprocessing/` | Slice-consistency rules, threshold and IoU sweeps |
| `evaluation/` | Stage-by-stage evaluation, mAP50, lesion-size analysis |
| `figures/` | Scripts for the paper and thesis figures |
| `ablations/` | Experiments that did not improve on the baseline: classifier filtering, CNN+ViT hybrid, patch-based training |
| `slurm/` | Example SLURM job script |

## Data

The CT dataset is not included. Scripts expect 512x512 PNG axial slices with YOLO-format labels, arranged as described in `configs/dataset.yaml`, with `dataset/` and `runs/` in the directory the scripts are run from.

## Training

Both detectors were trained for 50 epochs at 512 px with batch size 16. The full arguments are in `configs/yolov8m_train_args.yaml` and `configs/yolov10m_train_args.yaml`.

## Environment

Install with `pip install -r requirements.txt`. The original runs used PyTorch built for CUDA 13.0 on an NVIDIA A100. Install the PyTorch build that matches your CUDA version.

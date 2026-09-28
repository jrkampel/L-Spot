#!/bin/bash
# Example SLURM job. Originally run on QMUL Apocrita (A100). Set partition and account for your cluster.
#SBATCH --job-name=raw_detect
#SBATCH --partition=<your-partition>
#SBATCH --account=<your-account>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/raw_detect_%j.out
#SBATCH --error=./logs/raw_detect_%j.err

echo Job started

python detection/extract_raw_detections.py

echo Job finished

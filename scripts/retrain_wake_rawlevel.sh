#!/usr/bin/env bash
# Retrain the wake-word BC-ResNet2d on the AGC-free (random presentation
# level) dataset. Mel cache + checkpoints land beside the dataset / in
# weights/wake as usual; the fused 'BWK2' checkpoint overwrites
# weights/wake/computer.bw (peak-norm model backed up as
# computer.bw.peaknorm-backup).
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0
./build-cuda/Release/brosoundml_wake_train.exe \
    --dataset ../brosoundml-data/wake/computer-rawlevel \
    --out-checkpoint weights/wake/computer.bw \
    --device cuda \
    --seed 42

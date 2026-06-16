#!/usr/bin/env bash
# The promising pairing: LOW lr (reaches the ~2 floor) + beta2 0.95 (kills the
# post-descent divergence that wrecked low-lr runs at beta2 0.999). Want a combo
# that descends low AND stays there across 8 epochs on the cached 600-clip subset.
set -euo pipefail
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin:$PATH"
cd /d/projects/brosoundml
EXE=./build-cuda/Release/brosoundml_supertonic_encoder_train.exe
DATA=/d/projects/kws_data/vctk_44k
CACHE=/d/projects/kws_data/enc_cache
for LR in 7e-6 1e-5 1.5e-5; do
  echo "==================== lr $LR  beta2 0.95 ===================="
  "$EXE" --data "$DATA" --max-clips 600 --epochs 8 --accum 16 --lr "$LR" \
         --beta2 0.95 --cache "$CACHE" \
         --out /d/projects/brosoundml-data/supertonic/enc_scan.safetensors 2>&1 \
    | grep -E 'epoch [0-9]+:|lr schedule'
done
echo "==================== low-lr scan done ===================="

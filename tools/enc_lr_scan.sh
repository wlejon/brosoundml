#!/usr/bin/env bash
# Scan encoder-training lr on a small CACHED subset (600 clips ⊂ the 1500-clip
# seed-1234 set, so the clip cache already covers them — fast warm epochs). For
# each lr, print the per-epoch mean loss so we can see which lr actually DESCENDS
# below the ~20 zero-init baseline without diverging. accum 16, zero-init proj_out.
set -euo pipefail
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin:$PATH"
cd /d/projects/brosoundml
EXE=./build-cuda/Release/brosoundml_supertonic_encoder_train.exe
DATA=/d/projects/kws_data/vctk_44k
CACHE=/d/projects/kws_data/enc_cache
for LR in 1e-5 2e-5 5e-5 1e-4; do
  echo "==================== lr $LR ===================="
  "$EXE" --data "$DATA" --max-clips 600 --epochs 5 --accum 16 --lr "$LR" \
         --cache "$CACHE" \
         --out /d/projects/brosoundml-data/supertonic/enc_scan.safetensors 2>&1 \
    | grep -E 'epoch [0-9]+:|lr schedule'
done
echo "==================== scan done ===================="

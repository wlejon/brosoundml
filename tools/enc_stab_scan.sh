#!/usr/bin/env bash
# At a fixed, well-behaved lr (2e-5), sweep the two stability levers that target
# the descend-then-climb instability: Adam beta2 (lower = faster variance
# adaptation on the sharp landscape) and decoupled weight decay (opposes the slow
# unbounded weight growth). 8 epochs on the cached 600-clip subset so the climb
# (if any) has room to show. We want a combo whose per-epoch loss SETTLES near the
# ~2 plateau instead of climbing back out.
set -euo pipefail
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin:$PATH"
cd /d/projects/brosoundml
EXE=./build-cuda/Release/brosoundml_supertonic_encoder_train.exe
DATA=/d/projects/kws_data/vctk_44k
CACHE=/d/projects/kws_data/enc_cache
run() {  # $1=beta2 $2=wd
  echo "==================== lr 2e-5  beta2 $1  wd $2 ===================="
  "$EXE" --data "$DATA" --max-clips 600 --epochs 8 --accum 16 --lr 2e-5 \
         --beta2 "$1" --wd "$2" --cache "$CACHE" \
         --out /d/projects/brosoundml-data/supertonic/enc_scan.safetensors 2>&1 \
    | grep -E 'epoch [0-9]+:|lr schedule'
}
run 0.999 0       # reference: expect the climb
run 0.95  0       # faster variance adaptation
run 0.999 1e-2    # weight decay only
run 0.95  1e-2    # both
echo "==================== stab scan done ===================="

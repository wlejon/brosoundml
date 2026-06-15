#!/usr/bin/env bash
# Download NVIDIA Streaming Sortformer diarizer weights for brosoundml.
#
# Mirrors download-parakeet.sh: curl straight from the Hugging Face `resolve`
# endpoint — no huggingface_hub dependency.
#
# diar_streaming_sortformer_4spk-v2.1 (NVIDIA, 2025) is a streaming speaker
# diarization model: a NEST (Fast-Conformer) encoder + an 18-layer Transformer
# with an Arrival-Order Speaker Cache, emitting per-80ms-frame activity for up
# to 4 speakers. The checkpoint ships as a native NeMo `.nemo` archive (a tar of
# model_config.yaml + model_weights.ckpt); run scripts/convert-sortformer.py to
# turn it into the config.json + model.safetensors brosoundml::Sortformer loads.
#
# Usage:
#   scripts/download-sortformer.sh [--size 4spk-v2.1] [--out-dir D] [--force]
#
# Files fetched:
#   diar_streaming_sortformer_<size>.nemo   ~471 MB
#
# Auth: the repo is public (NVIDIA Open Model License); HF_TOKEN is honoured but
# optional.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SIZE="4spk-v2.1"
OUT_DIR=""
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="${2:?--size needs a value}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

case "$SIZE" in
    4spk-v2.1|4spk-v2) ;;
    *) echo "error: unsupported --size '$SIZE' (want 4spk-v2.1 or 4spk-v2)" >&2; exit 2 ;;
esac

REPO="nvidia/diar_streaming_sortformer_${SIZE}"
NEMO="diar_streaming_sortformer_${SIZE}.nemo"
[ -z "$OUT_DIR" ] && OUT_DIR="$REPO_ROOT/weights/sortformer/${SIZE}"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "Repo:    $REPO"
echo "Target:  $OUT_DIR"
[ -n "${HF_TOKEN:-}" ] && echo "Auth:    HF_TOKEN (bearer)"
echo

dest="$OUT_DIR/$NEMO"
if [ "$FORCE" -eq 0 ] && [ -s "$dest" ]; then
    echo "==> $NEMO  (cached, skipping)"
else
    echo "==> $NEMO"
    url="https://huggingface.co/$REPO/resolve/main/$NEMO"
    auth=()
    [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
    code="$(curl -fL --retry 3 --retry-delay 2 "${auth[@]}" \
                 -o "$dest.part" -w '%{http_code}' "$url")"
    if [ "$code" = "200" ]; then
        mv "$dest.part" "$dest"
    else
        rm -f "$dest.part"
        echo "    HTTP $code for $url" >&2
        exit 1
    fi
fi

echo
echo "Done. Now run:  scripts/convert-sortformer.py --src $OUT_DIR"

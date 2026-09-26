#!/usr/bin/env bash
# Download the LAION CLAP checkpoint (laion/larger_clap_general) for brosoundml.
#
# Mirrors download-nllb.sh: curl straight from the Hugging Face `resolve`
# endpoint — no huggingface_hub dependency, no Python.
#
# CLAP (Contrastive Language-Audio Pretraining) embeds a sound clip and a text
# prompt into one 512-d joint space: an HTSAT Swin audio tower (unfused
# variant, 48 kHz, 10 s window) plus a RoBERTa text tower, each followed by a
# two-layer projection. bro.ear.loadClap scores a clip against text prompts.
#
# Like NLLB, laion/larger_clap_general ships ONLY a pickled pytorch_model.bin,
# so a conversion step is REQUIRED afterwards: scripts/convert-clap.py writes
# model.safetensors (FP32) and validates the layout.
#
# Usage:
#   scripts/download-clap.sh [--out-dir D] [--force]
#
#   --out-dir D   where to drop files (default: <repo>/weights/clap)
#   --force       re-download even if a file already exists
#
# Files fetched:
#   config.json                ~5 KB
#   preprocessor_config.json   ~0.5 KB  (ClapFeatureExtractor: mel + padding)
#   tokenizer.json             ~2 MB    (RoBERTa byte-level BPE, brolm loader)
#   vocab.json / merges.txt    ~1.3 MB  (same BPE, classic form)
#   tokenizer_config.json, special_tokens_map.json
#   pytorch_model.bin          ~776 MB  (FP32)
#
# Auth: public (Apache-2.0); HF_TOKEN is honoured but optional.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="$REPO_ROOT/weights/clap"
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

REPO="laion/larger_clap_general"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "Repo:    $REPO"
echo "Target:  $OUT_DIR"
[ -n "${HF_TOKEN:-}" ] && echo "Auth:    HF_TOKEN (bearer)"
echo

fetch() {
    local rel="$1" dest="$2"
    local url="https://huggingface.co/$REPO/resolve/main/$rel"
    local auth=()
    [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")

    mkdir -p "$(dirname "$dest")"
    local code
    code="$(curl -fL --retry 3 --retry-delay 2 \
                 "${auth[@]}" \
                 -o "$dest.part" -w '%{http_code}' "$url" 2>/dev/null)" || {
        rm -f "$dest.part"
        echo "    curl failed for $url" >&2
        return 1
    }
    if [ "$code" = "200" ]; then
        mv "$dest.part" "$dest"
        return 0
    fi
    rm -f "$dest.part"
    echo "    HTTP $code for $url" >&2
    return 1
}

download() {
    local rel="$1" dest="$OUT_DIR/$1"
    if [ "$FORCE" -eq 0 ] && [ -s "$dest" ]; then
        echo "==> $rel  (cached, skipping)"
        return 0
    fi
    echo "==> $rel"
    fetch "$rel" "$dest"
}

download config.json
download preprocessor_config.json
download tokenizer.json
download tokenizer_config.json
download special_tokens_map.json
download vocab.json
download merges.txt
download pytorch_model.bin

echo
echo "Done. Now run: scripts/convert-clap.py   (pytorch_model.bin -> model.safetensors)"

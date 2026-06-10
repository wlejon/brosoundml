#!/usr/bin/env bash
# Download NVIDIA Parakeet-TDT weights for brosoundml.
#
# Mirrors download-qwen-asr.sh: curl straight from the Hugging Face `resolve`
# endpoint — no huggingface_hub dependency, no Python.
#
# Parakeet-TDT-0.6B-v3 (NVIDIA, 2025) is a multilingual (25 European languages)
# FastConformer-TDT speech-to-text model: a FastConformer encoder feeding a
# Token-and-Duration Transducer decoder. brosoundml loads the HuggingFace
# `transformers` ParakeetForTDT checkpoint (config.json + model.safetensors)
# and the unified SentencePiece tokenizer.json.
#
# Usage:
#   scripts/download-parakeet.sh [--size 0.6b-v3] [--out-dir D] [--force]
#
#   --size SIZE   model variant (default: 0.6b-v3 -> nvidia/parakeet-tdt-0.6b-v3)
#   --out-dir D   where to drop files (default: <repo>/weights/parakeet/<size>)
#   --force       re-download even if a file already exists
#
# Files fetched:
#   config.json                ~2 KB
#   tokenizer.json             ~1-4 MB  (brolm::t5::Tokenizer, SentencePiece)
#   model.safetensors          ~2.4 GB  (encoder + TDT decoder, FP32)
#   preprocessor_config.json   ~optional (feature-extractor params, if present)
#   tokenizer_config.json      ~optional
#   generation_config.json     ~optional
#
# Auth: the Parakeet repos are public (CC-BY-4.0); HF_TOKEN is honoured but
# optional.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SIZE="0.6b-v3"
OUT_DIR=""
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="${2:?--size needs a value}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

case "$SIZE" in
    0.6b-v3) ;;
    *) echo "error: unsupported --size '$SIZE' (want 0.6b-v3)" >&2; exit 2 ;;
esac

REPO="nvidia/parakeet-tdt-${SIZE}"
[ -z "$OUT_DIR" ] && OUT_DIR="$REPO_ROOT/weights/parakeet/${SIZE}"

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

# Required file: abort the script if it cannot be fetched.
download() {
    local rel="$1" dest="$OUT_DIR/$1"
    if [ "$FORCE" -eq 0 ] && [ -s "$dest" ]; then
        echo "==> $rel  (cached, skipping)"
        return 0
    fi
    echo "==> $rel"
    fetch "$rel" "$dest"
}

# Optional file: a 404 is fine (not every checkpoint ships it).
download_optional() {
    local rel="$1" dest="$OUT_DIR/$1"
    if [ "$FORCE" -eq 0 ] && [ -s "$dest" ]; then
        echo "==> $rel  (cached, skipping)"
        return 0
    fi
    echo "==> $rel  (optional)"
    fetch "$rel" "$dest" || echo "    (not present — skipping)"
}

download          config.json
download          tokenizer.json
download_optional tokenizer_config.json
download_optional special_tokens_map.json
download_optional preprocessor_config.json
download_optional generation_config.json
download          model.safetensors

echo
echo "Done. Model at: $OUT_DIR"

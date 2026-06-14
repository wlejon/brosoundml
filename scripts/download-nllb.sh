#!/usr/bin/env bash
# Download the NLLB-200 (M2M-100) translation checkpoint for brolm.
#
# Mirrors download-whisper.sh: curl straight from the Hugging Face `resolve`
# endpoint — no huggingface_hub dependency, no Python.
#
# Unlike Whisper, facebook/nllb-200-* ships ONLY a pickled pytorch_model.bin
# (no safetensors), so a conversion step is REQUIRED afterwards:
# scripts/convert-nllb.py turns pytorch_model.bin into model.safetensors and
# validates the layout against brolm's loader contract.
#
# Usage:
#   scripts/download-nllb.sh [--variant distilled-600M|distilled-1.3B|1.3B|3.3B]
#                            [--out-dir D] [--force]
#
#   --variant V    NLLB variant (default: distilled-600M). Maps to repo
#                  facebook/nllb-200-<variant> on Hugging Face.
#   --out-dir D    where to drop the files (default: <repo>/weights/nllb)
#   --force        re-download even if a file already exists
#
# Files fetched (what convert-nllb.py + brolm need):
#   config.json                  ~850 B
#   generation_config.json       ~190 B
#   pytorch_model.bin            ~2.46 GB (distilled-600M)
#   sentencepiece.bpe.model      ~4.85 MB
#   tokenizer.json               ~17.3 MB  (carries the 200+ language codes)
#   tokenizer_config.json        ~560 B
#   special_tokens_map.json      ~3.5 KB
#
# Auth: facebook/nllb-200-* repos are public; HF_TOKEN is honoured but optional.
#
# After this script, run scripts/convert-nllb.py to emit model.safetensors,
# then publish the directory to the brosoundml-data dataset repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANT="distilled-600M"
OUT_DIR="$REPO_ROOT/weights/nllb"
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --variant) VARIANT="${2:?--variant needs a value}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

case "$VARIANT" in
    distilled-600M|distilled-1.3B|1.3B|3.3B) ;;
    *) echo "error: unsupported --variant '$VARIANT'" >&2; exit 2 ;;
esac

REPO="facebook/nllb-200-$VARIANT"

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

# --- model + config ---------------------------------------------------------
download "config.json"
download "generation_config.json"
download "pytorch_model.bin"

# --- tokenizer (Unigram SPM + fast tokenizer.json with the language codes) ---
download "sentencepiece.bpe.model"
download "tokenizer.json"
download "tokenizer_config.json"
download "special_tokens_map.json"

echo
echo "Done. Files in $OUT_DIR :"
find "$OUT_DIR" -maxdepth 1 -type f | sort | while read -r p; do
    sz="$(wc -c < "$p" | tr -d ' ')"
    printf '  %12s  %s\n' "$sz" "${p#$OUT_DIR/}"
done

echo
echo "Next: run scripts/convert-nllb.py to emit model.safetensors from"
echo "pytorch_model.bin (NLLB ships no safetensors), then publish the dir."

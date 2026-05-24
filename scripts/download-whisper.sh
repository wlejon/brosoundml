#!/usr/bin/env bash
# Download Whisper weights + tokenizer files for brosoundml end-to-end testing.
#
# Mirrors download-kokoro.sh: curl straight from the Hugging Face `resolve`
# endpoint — no huggingface_hub dependency, no Python.
#
# Usage:
#   scripts/download-whisper.sh [--size tiny|base|small|medium|large-v3]
#                               [--out-dir D] [--force]
#
#   --size SIZE    Whisper model size (default: tiny). Maps to repo
#                  openai/whisper-<size> on Hugging Face.
#   --out-dir D    where to drop the files (default: <repo>/weights/whisper)
#   --force        re-download even if a file already exists
#
# Files fetched (the four brosoundml + brolm need at runtime):
#   config.json            ~1.5 KB
#   model.safetensors      ~150 MB (tiny) ... ~3 GB (large-v3)
#   vocab.json             ~1.1 MB  (brolm::whisper::Tokenizer)
#   merges.txt             ~500 KB  (brolm::whisper::Tokenizer)
#
# Auth: openai/whisper-* repos are public; HF_TOKEN is honoured but optional.
#
# After this script, run scripts/convert-whisper.py to validate the layout
# and (if needed) re-cast model.safetensors tensors to FP32 in place.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SIZE="tiny"
OUT_DIR="$REPO_ROOT/weights/whisper"
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="${2:?--size needs a value}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

case "$SIZE" in
    tiny|base|small|medium|large|large-v2|large-v3|large-v3-turbo) ;;
    *) echo "error: unsupported --size '$SIZE'" >&2; exit 2 ;;
esac

REPO="openai/whisper-$SIZE"

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
download "model.safetensors"

# --- tokenizer (brolm::whisper::Tokenizer reads vocab.json + merges.txt) ----
#
# Note: HF splits Whisper's vocabulary across two files — vocab.json holds the
# 50257 regular BPE entries, added_tokens.json holds the ~1600 special tokens
# (<|startoftranscript|>, language codes, timestamp tokens, ...). Brolm's
# tokenizer only reads vocab.json, so scripts/convert-whisper.py merges the
# added_tokens entries into vocab.json in place after this script runs.
download "vocab.json"
download "merges.txt"
download "added_tokens.json"

# --- generation_config.json: optional sanity reference (suppression lists,
# language token map, no-speech threshold). brosoundml doesn't read it today
# but it's useful for the user to inspect.
download "generation_config.json" || echo "  (generation_config.json absent — fine)"

echo
echo "Done. Files in $OUT_DIR :"
find "$OUT_DIR" -maxdepth 1 -type f | sort | while read -r p; do
    sz="$(wc -c < "$p" | tr -d ' ')"
    printf '  %12s  %s\n' "$sz" "${p#$OUT_DIR/}"
done

echo
echo "Next: run scripts/convert-whisper.py to validate the layout"
echo "(no .pth conversion needed — HF already distributes model.safetensors)."

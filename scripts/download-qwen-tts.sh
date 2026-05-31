#!/usr/bin/env bash
# Download Qwen3-TTS weights for brosoundml.
#
# Mirrors download-whisper.sh / download-kokoro.sh: curl straight from the
# Hugging Face `resolve` endpoint — no huggingface_hub dependency, no Python.
#
# Qwen3-TTS (Alibaba, Jan 2026) is the 12 Hz multi-codebook TTS series: an
# autoregressive Qwen3 "Talker" + a small "Code Predictor" depth-transformer
# emit 16 RVQ codes/frame @ 12.5 Hz, which a bundled Mimi-style codec decodes
# to a 24 kHz waveform. The codec ships *inside* each model repo under
# speech_tokenizer/ — identical across variants — so no separate tokenizer
# repo is needed.
#
# Usage:
#   scripts/download-qwen-tts.sh [--size 0.6B|1.7B]
#                                [--variant customvoice|base|voicedesign]
#                                [--out-dir D] [--force]
#
#   --size SIZE       parameter count (default: 0.6B)
#   --variant V       customvoice = 9 preset speakers, no ref audio (default;
#                                   the simplest forward path — no speaker
#                                   encoder, no codec encoder)
#                     base        = 3-second zero-shot voice cloning
#                     voicedesign = natural-language voice control (1.7B only)
#   --out-dir D       where to drop files
#                     (default: <repo>/weights/qwen-tts/<size>-<variant>)
#   --force           re-download even if a file already exists
#
# Files fetched (root — the Talker model + Qwen BPE tokenizer files):
#   config.json                ~5 KB
#   generation_config.json     ~250 B
#   tokenizer_config.json      ~7 KB
#   vocab.json                 ~2.7 MB  (brolm::qwen_tokenizer)
#   merges.txt                 ~1.6 MB  (brolm::qwen_tokenizer)
#   preprocessor_config.json   ~130 B
#   model.safetensors          ~1.8 GB  (Talker + Code Predictor + embeddings)
#
# Files fetched (speech_tokenizer/ — the bundled 12 Hz codec):
#   config.json                ~2.3 KB
#   configuration.json         ~80 B
#   preprocessor_config.json   ~230 B
#   model.safetensors          ~680 MB  (RVQ codec encoder+decoder, 0.2B)
#
# Auth: the Qwen3-TTS repos are public (Apache-2.0); HF_TOKEN is honoured but
# optional.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SIZE="0.6B"
VARIANT="customvoice"
OUT_DIR=""
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="${2:?--size needs a value}"; shift 2 ;;
        --variant) VARIANT="${2:?--variant needs a value}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,49p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

case "$SIZE" in
    0.6B|1.7B) ;;
    *) echo "error: unsupported --size '$SIZE' (want 0.6B or 1.7B)" >&2; exit 2 ;;
esac

case "$VARIANT" in
    customvoice) REPO_SUFFIX="CustomVoice" ;;
    base)        REPO_SUFFIX="Base" ;;
    voicedesign) REPO_SUFFIX="VoiceDesign" ;;
    *) echo "error: unsupported --variant '$VARIANT'" >&2; exit 2 ;;
esac

REPO="Qwen/Qwen3-TTS-12Hz-${SIZE}-${REPO_SUFFIX}"
[ -z "$OUT_DIR" ] && OUT_DIR="$REPO_ROOT/weights/qwen-tts/${SIZE}-${VARIANT}"

mkdir -p "$OUT_DIR/speech_tokenizer"
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

# --- Talker model + config --------------------------------------------------
download "config.json"
download "generation_config.json"
download "model.safetensors"

# --- Qwen BPE tokenizer (brolm::qwen_tokenizer reads vocab.json + merges.txt)-
download "tokenizer_config.json"
download "vocab.json"
download "merges.txt"
download "preprocessor_config.json" || echo "  (preprocessor_config.json absent — fine)"

# --- bundled 12 Hz codec (speech_tokenizer/) --------------------------------
download "speech_tokenizer/config.json"
download "speech_tokenizer/configuration.json"
download "speech_tokenizer/preprocessor_config.json" \
    || echo "  (speech_tokenizer/preprocessor_config.json absent — fine)"
download "speech_tokenizer/model.safetensors"

echo
echo "Done. Files in $OUT_DIR :"
find "$OUT_DIR" -type f | sort | while read -r p; do
    sz="$(wc -c < "$p" | tr -d ' ')"
    printf '  %12s  %s\n' "$sz" "${p#$OUT_DIR/}"
done

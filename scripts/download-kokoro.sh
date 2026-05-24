#!/usr/bin/env bash
# Download Kokoro-82M weights for brosoundml end-to-end testing.
#
# Bash port of the brodiffusion download pattern, narrowed to the one model
# brosoundml currently targets. Pulls files straight from the Hugging Face
# `resolve` endpoint with `curl` — no huggingface_hub dependency, no Python.
#
# Usage:
#   scripts/download-kokoro.sh [--out-dir D] [--force] [--voice NAME]...
#
#   --out-dir D    where to drop the files (default: <repo>/weights/kokoro)
#   --force        re-download even if a file already exists
#   --voice NAME   only fetch the named voice (e.g. af_bella). Repeat to add
#                  more. By default ALL voices in voices/ are fetched.
#
# Files fetched:
#   config.json            ~2.4 KB
#   kokoro-v1_0.pth        ~327 MB  (pickled PyTorch checkpoint)
#   voices/<name>.pt       ~520 KB each (pickled voice packs)
#
# Auth: the repo is public; HF_TOKEN is honoured but not required.
#
# After this script, run scripts/convert-kokoro.py to produce
# model.safetensors + voices/*.bin in brosoundml's expected layout.

set -euo pipefail

REPO="hexgrad/Kokoro-82M"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/weights/kokoro"
FORCE=0
VOICES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        --voice)   VOICES+=("${2:?--voice needs a value}"); shift 2 ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT_DIR/voices"
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
download "kokoro-v1_0.pth"

# --- voices -----------------------------------------------------------------
# If the user didn't pass --voice, fetch every voice file under voices/. The
# Hugging Face tree API gives us the canonical list; we fall back to a hard-
# coded set if the tree endpoint is unreachable (offline mirror, etc.).
if [ "${#VOICES[@]}" -eq 0 ]; then
    if list="$(curl -fsL "https://huggingface.co/api/models/$REPO/tree/main/voices" 2>/dev/null)"; then
        # Each entry is {"type":"file","path":"voices/af_bella.pt",...}
        VOICES_REL=( $(printf '%s' "$list" | grep -oE '"path":"voices/[^"]+\.pt"' \
                                          | sed 's|.*voices/||; s|\.pt"||' \
                                          | sort -u) )
    else
        echo "warning: could not list voices/ via HF API; falling back to a default subset" >&2
        VOICES_REL=( af_heart af_bella am_adam bf_emma bm_george )
    fi
else
    VOICES_REL=( "${VOICES[@]}" )
fi

echo
echo "Voices: ${#VOICES_REL[@]} files"
for v in "${VOICES_REL[@]}"; do
    download "voices/${v}.pt"
done

echo
echo "Done. Files in $OUT_DIR :"
find "$OUT_DIR" -type f | sort | while read -r p; do
    sz="$(wc -c < "$p" | tr -d ' ')"
    printf '  %12s  %s\n' "$sz" "${p#$OUT_DIR/}"
done

echo
echo "Next: run scripts/convert-kokoro.py to produce model.safetensors + voices/*.bin"

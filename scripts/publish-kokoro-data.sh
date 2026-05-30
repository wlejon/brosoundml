#!/usr/bin/env bash
# Publish the converted Kokoro artifacts to the wlejon/brosoundml-data dataset.
#
# This is a MAINTAINER tool, not a runtime/download step. The Kokoro synth
# weights the loader needs — model.safetensors + raw-f32 voices/<name>.bin —
# are *converted* from upstream's pickled checkpoint (hexgrad/Kokoro-82M) by
# scripts/convert-kokoro.py and have no upstream download URL of their own. To
# let packaged apps download-and-speak (e.g. broworkshop's voice-pipeline), a
# maintainer uploads them once into a kokoro/ tree in the dataset repo; from
# then on the app's models.js resolves them from the per-user cache.
#
# Uploaded layout (under the dataset root):
#   kokoro/config.json
#   kokoro/model.safetensors
#   kokoro/voices/<name>.bin        (af_heart by default; --voice to add more)
#
# Usage:
#   HF_TOKEN=hf_... scripts/publish-kokoro-data.sh [--src DIR] [--voice NAME]...
#
#   --src DIR      converted-artifacts dir (default: <repo>/weights/kokoro,
#                  i.e. the output of convert-kokoro.py)
#   --voice NAME   extra voice to publish (repeat to add more; af_heart always
#                  included). Each must exist as <src>/voices/<NAME>.bin
#
# Auth + tooling: needs write access to the dataset (HF_TOKEN with write scope)
# and the huggingface_hub CLI for the LFS upload (`pip install huggingface_hub`).
# Unlike the download-* scripts (curl-only, no-Python by design), publishing is
# a one-off maintainer action, so leaning on huggingface-cli for the multi-step
# LFS protocol is the pragmatic choice.

set -euo pipefail

DATASET="wlejon/brosoundml-data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_ROOT/weights/kokoro"
VOICES=(af_heart)

while [ $# -gt 0 ]; do
    case "$1" in
        --src)   SRC="${2:?--src needs a value}"; shift 2 ;;
        --voice) VOICES+=("${2:?--voice needs a value}"); shift 2 ;;
        -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

if [ -z "${HF_TOKEN:-}" ]; then
    echo "error: HF_TOKEN is required (a token with write access to $DATASET)" >&2
    exit 2
fi
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "error: huggingface-cli not found — install with 'pip install huggingface_hub'" >&2
    exit 2
fi

SRC="$(cd "$SRC" && pwd)"
echo "Dataset: $DATASET (dataset repo)"
echo "Source:  $SRC"
echo

# De-dup the voice list (af_heart may be passed explicitly too).
mapfile -t VOICES < <(printf '%s\n' "${VOICES[@]}" | awk '!seen[$0]++')

# Build the upload list: local path -> path-in-repo. Verify each exists first so
# we fail before any partial upload.
declare -a SRCS DSTS
add() {
    local local_path="$1" repo_path="$2"
    if [ ! -s "$local_path" ]; then
        echo "error: missing or empty: $local_path" >&2
        echo "       run scripts/convert-kokoro.py first (see download-kokoro.sh)" >&2
        exit 1
    fi
    SRCS+=("$local_path"); DSTS+=("$repo_path")
}

add "$SRC/config.json"        "kokoro/config.json"
add "$SRC/model.safetensors"  "kokoro/model.safetensors"
for v in "${VOICES[@]}"; do
    add "$SRC/voices/$v.bin" "kokoro/voices/$v.bin"
done

echo "Will upload ${#SRCS[@]} file(s) to $DATASET:"
for i in "${!SRCS[@]}"; do
    sz="$(wc -c < "${SRCS[$i]}" | tr -d ' ')"
    printf '  %12s  %s\n' "$sz" "${DSTS[$i]}"
done
echo

for i in "${!SRCS[@]}"; do
    echo "==> ${DSTS[$i]}"
    huggingface-cli upload "$DATASET" "${SRCS[$i]}" "${DSTS[$i]}" \
        --repo-type dataset \
        --commit-message "Add ${DSTS[$i]} for download-and-speak"
done

echo
echo "Done. The voice-pipeline app's models.js resolves these from the cache at"
echo "  <cache>/datasets/$DATASET/kokoro/ ."
echo "Verify the tree with scripts/download-brosoundml-data.sh --kokoro --force."

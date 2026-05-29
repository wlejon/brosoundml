#!/usr/bin/env bash
# Download brosoundml's packed data artifacts (lexicon, POS tagger, wake word).
#
# Mirrors download-whisper.sh / download-kokoro.sh: curl straight from the
# Hugging Face `resolve` endpoint — no huggingface_hub dependency, no Python.
# Unlike those two, these are FINAL artifacts (no convert step afterwards):
# they are brosoundml's own trained/packed outputs, hosted at the dataset repo
# wlejon/brosoundml-data.
#
# Usage:
#   scripts/download-brosoundml-data.sh [--out-dir D] [--force]
#
#   --out-dir D    where to drop the files (default: <repo>/../brosoundml-data,
#                  the sibling layout bro's loaders resolve by default — see
#                  BROSOUNDML_DATA_DIR in the dataset README)
#   --force        re-download even if a file already exists
#
# Files fetched (what brosoundml needs at runtime):
#   g2p/lexicon_en_us.bin   ~6.8 MB  brosoundml::g2p::Lexicon::load()
#   pos_tagger/model.bin    ~7.6 MB  brosoundml::g2p::PosTagger::load()
#   wake/computer.bw        ~64 KB   brosoundml::WakeWord::load()
#
# Provenance/licence files (NOTICE, LICENSE-*) are fetched too — best-effort —
# because the g2p lexicon (Apache 2.0, derived from misaki) ships an attribution
# NOTICE that must travel with it.
#
# Auth: the dataset repo is public; HF_TOKEN is honoured but not required.

set -euo pipefail

REPO="wlejon/brosoundml-data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/../brosoundml-data"
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "Repo:    $REPO (dataset)"
echo "Target:  $OUT_DIR"
[ -n "${HF_TOKEN:-}" ] && echo "Auth:    HF_TOKEN (bearer)"
echo

fetch() {
    local rel="$1" dest="$2"
    local url="https://huggingface.co/datasets/$REPO/resolve/main/$rel"
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

# --- runtime artifacts (required) -------------------------------------------
download "g2p/lexicon_en_us.bin"
download "pos_tagger/model.bin"
download "wake/computer.bw"

# --- provenance / licence (best-effort: never block a checkout on these) ----
for doc in LICENSE README.md \
           g2p/NOTICE g2p/LICENSE-APACHE-2.0 g2p/README.md \
           pos_tagger/README.md wake/README.md; do
    download "$doc" || echo "  ($doc absent — fine)"
done

echo
echo "Done. Files in $OUT_DIR :"
find "$OUT_DIR" -type f | sort | while read -r p; do
    sz="$(wc -c < "$p" | tr -d ' ')"
    printf '  %12s  %s\n' "$sz" "${p#$OUT_DIR/}"
done

echo
echo "These are final artifacts — no conversion step. bro resolves them via"
echo "BROSOUNDML_DATA_DIR or the ../brosoundml-data sibling (see dataset README)."

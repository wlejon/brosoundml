#!/usr/bin/env bash
# SLURP (spoken assistant commands with intents and entities), EVAL ONLY.
#
# Licences (checked at github.com/pswietojanski/slurp, 2026-09-24): the text
# annotations are CC BY 4.0, but the AUDIO on Zenodo is CC BY-NC 4.0. That
# fails the adapter's Apache-2.0 release policy, so SLURP audio is used for
# evaluation only and never for training (DATA_LICENSES.md).
#
# Only the test split is kept: one recording per test utterance (the first
# listed, usually the close-talk headset take). The 3.9 GB slurp_real archive
# streams past and only those files are extracted.
#
# Usage: fetch_slurp.sh [DEST=D:/datasets/slurp]
set -euo pipefail
DEST=${1:-D:/datasets/slurp}
GH=https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp
ZEN=https://zenodo.org/record/4274930/files
mkdir -p "$DEST"
cd "$DEST"
for s in test devel; do
    [ -f $s.jsonl ] || curl -sfL -o $s.jsonl "$GH/$s.jsonl"
done
curl -sfL -o LICENSE.txt https://raw.githubusercontent.com/pswietojanski/slurp/master/LICENSE.txt || true
sed -E 's/.*"recordings": \[\{"file": "([^"]+)".*/slurp_real\/\1/' test.jsonl > keep.list
echo "slurp: extracting $(wc -l < keep.list) test recordings"
if [ ! -d slurp_real ] || [ "$(ls slurp_real | wc -l)" -lt "$(wc -l < keep.list)" ]; then
    curl -sfL "$ZEN/slurp_real.tar.gz" | tar -xzf - -T keep.list
fi
echo "slurp: $(ls slurp_real | wc -l) files, $(du -sh "$DEST" | cut -f1)"

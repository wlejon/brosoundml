#!/usr/bin/env bash
# CREMA-D (acted emotional speech: 91 actors, 12 fixed sentences, six
# emotions), EVAL ONLY.
#
# Licence (checked at github.com/CheyneyComputerScience/CREMA-D, 2026-09-24):
# Open Database License + Database Contents License. The ODbL is share-alike
# for derived databases; that is not the public-domain / CC0 / CC BY class the
# adapter's Apache-2.0 release admits, so CREMA-D is never trained on. It is
# the paralinguistic test: every emotion is spoken over the same 12
# sentences, so the words carry no emotion and a transcript-only model is at
# chance by construction.
#
# The 16 kHz WAVs are fetched one by one from the repository's Git LFS
# store (about 0.55 GB), with the crowd ratings (processedResults).
#
# Usage: fetch_cremad.sh [DEST=D:/datasets/cremad] [JOBS=16]
set -euo pipefail
DEST=${1:-D:/datasets/cremad}
JOBS=${2:-16}
RAW=https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master
LFS=https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master
mkdir -p "$DEST/wav"
cd "$DEST"
curl -sfL -o LICENSE.txt "$RAW/LICENSE.txt" || true
curl -sfL -o SentenceFilenames.csv "$RAW/SentenceFilenames.csv"
curl -sfL -o summaryTable.csv "$LFS/processedResults/summaryTable.csv" ||
    curl -sfL -o summaryTable.csv "$RAW/processedResults/summaryTable.csv"
tail -n +2 SentenceFilenames.csv | tr -d '\r' | cut -d, -f2 | sed 's/\.flv$//' > clips.list
fetch_one() {
    [ -s "wav/$1.wav" ] && return 0
    curl -sfL -o "wav/$1.wav.part" "$LFS/AudioWAV/$1.wav" && mv "wav/$1.wav.part" "wav/$1.wav" || rm -f "wav/$1.wav.part"
}
export -f fetch_one
export LFS
xargs -P "$JOBS" -I{} bash -c 'fetch_one {}' < clips.list
echo "cremad: $(ls wav | wc -l) of $(wc -l < clips.list) clips, $(du -sh "$DEST" | cut -f1)"

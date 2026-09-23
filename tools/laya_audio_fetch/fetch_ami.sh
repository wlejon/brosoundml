#!/usr/bin/env bash
# AMI meeting corpus (CC BY 4.0): single distant microphone (Array1-01) of
# every meeting that has one, as 16 kHz mono Opus, plus the manual
# annotations' word files (word-level start/end times) and the licence.
# Each WAV is transcoded and deleted as soon as it lands, so at most a few
# ~100 MB WAVs exist at once.
#
# Usage: fetch_ami.sh [DEST=D:/datasets/ami] [JOBS=4]
set -euo pipefail
DEST=${1:-D:/datasets/ami}
JOBS=${2:-4}
BASE=https://groups.inf.ed.ac.uk/ami
mkdir -p "$DEST/sdm"
cd "$DEST"
if [ ! -d words ]; then
    curl -s -o ami_public_manual_1.6.2.zip $BASE/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
    unzip -o -q ami_public_manual_1.6.2.zip 'words/*' LICENCE.txt 00README_MANUAL.txt 'corpusResources/meetings.xml'
    rm ami_public_manual_1.6.2.zip
fi
ls words | sed 's/\..*//' | sort -u > meetings.txt

fetch_one() {
    local m=$1 out=$DEST/sdm/$1.opus wav=$DEST/sdm/$1.wav
    [ -f "$out" ] && return 0
    if ! curl -sf -o "$wav" "$BASE/AMICorpusMirror/amicorpus/$m/audio/$m.Array1-01.wav"; then
        rm -f "$wav"
        echo "no Array1-01 for $m"
        return 0
    fi
    ffmpeg -nostdin -hide_banner -loglevel error -y -i "$wav" -ac 1 -ar 16000 -c:a libopus -b:a 32k -f ogg "$out.part"
    mv "$out.part" "$out"
    rm -f "$wav"
    echo "$m done"
}
export -f fetch_one
export DEST BASE
xargs -P "$JOBS" -I{} bash -c 'fetch_one {}' < meetings.txt
echo "ami: $(ls sdm/*.opus | wc -l) meetings, $(du -sh "$DEST" | cut -f1)"

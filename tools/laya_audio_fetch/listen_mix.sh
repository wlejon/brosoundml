#!/usr/bin/env bash
# Streaming evaluation of one adapter (brosoundml_laya_audio_listen): hold
# operating points and the single-hop boop / unboop sweep on LibriTTS
# dev-clean, AMI test and VoxPopuli en test; false alarms per keyword-hour on
# the held-out noise and music. The same fixed vocabulary and seeds for every
# adapter. Per-hop timing is a separate listen run (without --no-timing) on
# an idle GPU.
#
#   listen_mix.sh TAG PROJ LAYA_DIR TEMPERATURE [UTTS=100]
#
# Writes $A/listen_TAG.log.
set -euo pipefail
TAG=$1
PROJ=$2
LAYA=$3
TEMP=$4
UTTS=${5:-100}
A=${A:-D:/projects/brosoundml-data/laya-audio2}
L=${L:-D:/projects/brosoundml-data/laya-audio}
B=${B:-build-cuda/Release}
VOCAB=$L/align_train.tsv+$A/align_ami_train.tsv
for lg in en de es fr; do VOCAB=$VOCAB+$A/align_vp_${lg}_train.tsv+$A/align_mswc_${lg}_train.tsv; done
run() {  # run NAME STREAM [extra...]
    local name=$1 stream=$2
    shift 2
    echo "===== $name"
    $B/brosoundml_laya_audio_listen --vocab-align "$VOCAB" --stream "$stream" --proj "$PROJ" --laya "$LAYA" \
        --temperature "$TEMP" --utts "$UTTS" --no-timing "$@" 2>&1
}
{
    run libri-dev "$L/align_dev.tsv"
    run ami-test "$A/align_ami_test.tsv"
    run vp-en-test "$A/align_vp_en_test.tsv"
    run noise-test "$A/align_noise_test.tsv" --nonspeech
    run music-test "$A/align_music_test.tsv" --nonspeech
} > "$A/listen_$TAG.log"
echo "listen $TAG done"

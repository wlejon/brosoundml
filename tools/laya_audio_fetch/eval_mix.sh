#!/usr/bin/env bash
# Per-domain evaluation of one adapter on the broadened mix's held-out sets.
# Every adapter is evaluated with the same fixed negative vocabulary (all
# training alignments of the mix), the same sets and the same seed, so the
# probes are identical across adapters; only "seen" (asked in training) and
# the examples-per-word counts depend on the adapter's own training data.
#
#   eval_mix.sh TAG PROJ LAYA_DIR TEMPERATURE TRAIN_ALIGN[+ALIGN...]
#
# Writes $A/eval_TAG.log (all sets, full vocabulary), $A/eval_TAG_libri.log
# (LibriTTS dev-clean with the first adapter's vocabulary, comparable with
# the numbers in docs/laya-audio.md) and $A/eval_TAG_mswc.log (the MSWC test
# sets with examples-per-word counted over the MSWC training clips only, the
# frequency-controlled curve).
set -euo pipefail
TAG=$1
PROJ=$2
LAYA=$3
TEMP=$4
TRAIN=$5
A=${A:-D:/projects/brosoundml-data/laya-audio2}
L=${L:-D:/projects/brosoundml-data/laya-audio}
B=${B:-build-cuda/Release}
LANGS=${LANGS:-"en de es fr"}

VOCAB=$L/align_train.tsv+$A/align_ami_train.tsv
MSWC_TRAIN=""
SETS=(--set "libri-dev,$L/align_dev.tsv,$L/dev_w3.lac" --set "ami-test,$A/align_ami_test.tsv,$A/ami_test.lac")
MSWC_SETS=()
for lg in $LANGS; do
    VOCAB=$VOCAB+$A/align_vp_${lg}_train.tsv
    SETS+=(--set "vp-$lg-test,$A/align_vp_${lg}_test.tsv,$A/vp_${lg}_test.lac")
    if [ -f "$A/align_mswc_${lg}_train.tsv" ]; then
        VOCAB=$VOCAB+$A/align_mswc_${lg}_train.tsv
        MSWC_TRAIN=${MSWC_TRAIN:+$MSWC_TRAIN+}$A/align_mswc_${lg}_train.tsv
        MSWC_SETS+=(--set "mswc-$lg-test,$A/align_mswc_${lg}_test.tsv,$A/mswc_${lg}_test.lac")
    fi
done
SETS+=("${MSWC_SETS[@]}")
SETS+=(--set "noise-test,$A/align_noise_test.tsv,$A/noise_test.lac" --set "music-test,$A/align_music_test.tsv,$A/music_test.lac")

$B/brosoundml_laya_audio_eval --vocab-align "$VOCAB" --train-align "$TRAIN" "${SETS[@]}" \
    --proj "$PROJ" --laya "$LAYA" --temperature "$TEMP" > "$A/eval_$TAG.log" 2>&1
$B/brosoundml_laya_audio_eval --vocab-align "$L/align_train.tsv" --train-align "$TRAIN" \
    --set "libri-dev,$L/align_dev.tsv,$L/dev_w3.lac" \
    --proj "$PROJ" --laya "$LAYA" --temperature "$TEMP" > "$A/eval_${TAG}_libri.log" 2>&1
$B/brosoundml_laya_audio_eval --vocab-align "$VOCAB" --train-align "$TRAIN" --count-align "$MSWC_TRAIN" \
    "${MSWC_SETS[@]}" --proj "$PROJ" --laya "$LAYA" --temperature "$TEMP" > "$A/eval_${TAG}_mswc.log" 2>&1
echo "eval $TAG done"

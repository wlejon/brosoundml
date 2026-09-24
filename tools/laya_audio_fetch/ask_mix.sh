#!/usr/bin/env bash
# Free-form question evaluation of one adapter (brosoundml_laya_audio_ask):
# the adapter vs text Laya on the gold transcript (ceiling) vs text Laya on
# Parakeet's transcript (baseline), with real labels where they exist.
#
#   [TEXT_LAYA=DIR TEXT_T=T] ask_mix.sh TAG PROJ LAYA_DIR TEMPERATURE [WINDOWS=600]
#
# Sets: LibriTTS dev-clean, AMI test (dialogue acts, named entities, topics),
# VoxPopuli en / de test, Timers and Such real test (intents), SLURP test
# (intents, entities; eval only), CREMA-D (emotions; eval only), held-out
# noise. Writes $A/ask_TAG.log and the per-answer dump $A/ask_TAG_*.tsv.
set -euo pipefail
TAG=$1
PROJ=$2
LAYA=$3
TEMP=$4
WIN=${5:-600}
A=${A:-D:/projects/brosoundml-data/laya-audio3}
A2=${A2:-D:/projects/brosoundml-data/laya-audio2}
L=${L:-D:/projects/brosoundml-data/laya-audio}
B=${B:-build-cuda/Release}
S=(--set "libri-dev,$L/align_dev.tsv,$L/dev_w3.lac"
   --set "ami-test,$A2/align_ami_test.tsv,$A2/ami_test.lac,$A/labels_ami_test.tsv"
   --set "vp-en-test,$A2/align_vp_en_test.tsv,$A2/vp_en_test.lac"
   --set "vp-de-test,$A2/align_vp_de_test.tsv,$A2/vp_de_test.lac"
   --set "timers-test,$A/align_timers_test.tsv,$A/timers_test.lac,$A/labels_timers_test.tsv"
   --set "slurp-test,$A/align_slurp_test.tsv,$A/slurp_test.lac,$A/labels_slurp_test.tsv"
   --set "cremad,$A/align_cremad.tsv,$A/cremad.lac,$A/labels_cremad.tsv"
   --set "noise-test,$A2/align_noise_test.tsv,$A2/noise_test.lac")
# TEXT_LAYA / TEXT_T: run the text methods on another checkpoint (the teacher).
TX=()
if [ -n "${TEXT_LAYA:-}" ]; then TX=(--text-laya "$TEXT_LAYA" --text-temperature "$TEXT_T"); fi
$B/brosoundml_laya_audio_ask --bank tools/laya_audio_questions.tsv --proj "$PROJ" --laya "$LAYA" --temperature "$TEMP" \
    "${TX[@]}" "${S[@]}" --windows "$WIN" --asr --dump "$A/ask_$TAG" > "$A/ask_$TAG.log" 2>&1
echo "ask $TAG done"

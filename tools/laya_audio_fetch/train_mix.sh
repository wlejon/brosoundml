#!/usr/bin/env bash
# Train the projector on the broadened mix (the caches build_mix.sh made).
#
#   train_mix.sh TAG LAYA_DIR [STEPS=24000]
#
# Domain weights (probability that a batch window comes from the domain):
#   speech, English   LibriTTS 0.22 (clean + one augmented copy), AMI 0.18,
#                     VoxPopuli en 0.14, MSWC en 0.10
#   speech, other     VoxPopuli de/es/fr 0.06 each, MSWC de/es/fr 0.03 each
#   non-speech        noise 0.045, music 0.045 (speaking = no, every keyword = no)
# Dev domains are reported separately every 4000 steps.
set -euo pipefail
TAG=$1
LAYA=$2
STEPS=${3:-24000}
A=${A:-D:/projects/brosoundml-data/laya-audio2}
L=${L:-D:/projects/brosoundml-data/laya-audio}
B=${B:-build-cuda/Release}
T=()
T+=(--train "libri,0.22,$L/align_train.tsv,$L/train_w3_a.lac+$L/train_w3_b.lac+$A/libri_train_aug.lac")
T+=(--train "ami,0.18,$A/align_ami_train.tsv,$A/ami_train.lac")
T+=(--train "vp-en,0.14,$A/align_vp_en_train.tsv,$A/vp_en_train.lac")
T+=(--train "mswc-en,0.10,$A/align_mswc_en_train.tsv,$A/mswc_en_train.lac")
for lg in de es fr; do
    T+=(--train "vp-$lg,0.06,$A/align_vp_${lg}_train.tsv,$A/vp_${lg}_train.lac")
    T+=(--train "mswc-$lg,0.03,$A/align_mswc_${lg}_train.tsv,$A/mswc_${lg}_train.lac")
done
T+=(--train "noise,0.045,$A/align_noise_train.tsv,$A/noise_train.lac")
T+=(--train "music,0.045,$A/align_music_train.tsv,$A/music_train.lac")
D=(--dev "libri-dev,$L/align_dev.tsv,$L/dev_w3.lac" --dev "ami-dev,$A/align_ami_dev.tsv,$A/ami_dev.lac"
   --dev "vp-en-dev,$A/align_vp_en_dev.tsv,$A/vp_en_dev.lac" --dev "vp-de-dev,$A/align_vp_de_dev.tsv,$A/vp_de_dev.lac")
$B/brosoundml_laya_audio_train --laya "$LAYA" "${T[@]}" "${D[@]}" \
    --steps "$STEPS" --eval-every 4000 --out "$A/proj_$TAG.mlp" > "$A/train_$TAG.log" 2>&1
echo "train $TAG done"

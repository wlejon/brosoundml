#!/usr/bin/env bash
# Free-form distillation round: fine-tune a projector on the broadened mix
# (train_mix.sh's domains and weights) plus Timers and Such real commands,
# with free-form questions distilled from a text-Laya teacher on the gold
# transcript of every training window, next to the keyword / speaking /
# word-end tasks.
#
#   train_free.sh TAG STUDENT_LAYA INIT_PROJ TEACHER_LAYA TEACHER_T STUDENT_T [STEPS=12000]
#   e.g. train_free.sh free_mm D:/projects/laya/multilingual $A2/proj_mm.mlp D:/projects/laya 1.9834 1.0
#
# Only the bank's train families / train phrasings are ever asked (see
# tools/laya_audio_questions.tsv); held-out families, held-out phrasings and
# the tone-of-voice questions measure generalisation.
set -euo pipefail
TAG=$1
LAYA=$2
INIT=$3
TEACHER=$4
TT=$5
TS=$6
STEPS=${7:-12000}
A=${A:-D:/projects/brosoundml-data/laya-audio3}
A2=${A2:-D:/projects/brosoundml-data/laya-audio2}
L=${L:-D:/projects/brosoundml-data/laya-audio}
B=${B:-build-cuda/Release}
T=()
T+=(--train "libri,0.21,$L/align_train.tsv,$L/train_w3_a.lac+$L/train_w3_b.lac+$A2/libri_train_aug.lac")
T+=(--train "ami,0.17,$A2/align_ami_train.tsv,$A2/ami_train.lac")
T+=(--train "vp-en,0.14,$A2/align_vp_en_train.tsv,$A2/vp_en_train.lac")
T+=(--train "mswc-en,0.09,$A2/align_mswc_en_train.tsv,$A2/mswc_en_train.lac")
for lg in de es fr; do
    T+=(--train "vp-$lg,0.06,$A2/align_vp_${lg}_train.tsv,$A2/vp_${lg}_train.lac")
    T+=(--train "mswc-$lg,0.03,$A2/align_mswc_${lg}_train.tsv,$A2/mswc_${lg}_train.lac")
done
T+=(--train "timers,0.04,$A/align_timers_train.tsv,$A/timers_train.lac")
T+=(--train "noise,0.045,$A2/align_noise_train.tsv,$A2/noise_train.lac")
T+=(--train "music,0.045,$A2/align_music_train.tsv,$A2/music_train.lac")
D=(--dev "libri-dev,$L/align_dev.tsv,$L/dev_w3.lac" --dev "ami-dev,$A2/align_ami_dev.tsv,$A2/ami_dev.lac"
   --dev "vp-en-dev,$A2/align_vp_en_dev.tsv,$A2/vp_en_dev.lac" --dev "vp-de-dev,$A2/align_vp_de_dev.tsv,$A2/vp_de_dev.lac")
$B/brosoundml_laya_audio_train --laya "$LAYA" --init "$INIT" "${T[@]}" "${D[@]}" \
    --bank tools/laya_audio_questions.tsv --teacher "$TEACHER" --teacher-temperature "$TT" --student-temperature "$TS" \
    --free-cand 8 --free-keep 4 --free-dev-windows 80 \
    --steps "$STEPS" --lr 2e-4 --eval-every 3000 --out "$A/proj_$TAG.mlp" > "$A/train_$TAG.log" 2>&1
echo "train $TAG done"

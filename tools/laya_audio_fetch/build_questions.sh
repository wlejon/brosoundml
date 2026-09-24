#!/usr/bin/env bash
# Free-form question round: labels, alignments and window caches for the new
# sets (docs/laya-audio.md, "Free-form questions"). Outputs in $A.
#
#   build_questions.sh timers | cremad | slurp | ami
#
#   timers  Timers and Such real speech (CC0): train (trainable) and test.
#           Train: 4 random windows per utterance; test: one window ending
#           0-0.3 s after the last word.
#   cremad  CREMA-D (eval only): one window per clip at the end of speech.
#   slurp   SLURP test (eval only): one window per utterance at the end of speech.
#   ami     dialogue-act / named-entity / topic labels on the AMI dev and test chunks.
set -euo pipefail
WHAT=$1
A=${A:-D:/projects/brosoundml-data/laya-audio3}
A2=${A2:-D:/projects/brosoundml-data/laya-audio2}
B=${B:-build-cuda/Release}
mkdir -p "$A"
LAB=$B/brosoundml_laya_audio_labels
align() {  # manifest subset out
    [ -s "$3" ] || $B/brosoundml_laya_audio_align --manifest "$1" --subset "$2" --out "$3" > "${3%.tsv}.log" 2>&1
}
case $WHAT in
timers)
    for s in train test; do
        $LAB timers --csv D:/datasets/timers/$s-real.csv --audio-dir D:/datasets/timers --subset timers-$s \
            --manifest "$A/timers_${s}_manifest.tsv" --out "$A/labels_timers_$s.tsv"
        align "$A/timers_${s}_manifest.tsv" timers-$s "$A/align_timers_$s.tsv"
    done
    $B/brosoundml_laya_audio_cache --align "$A/align_timers_train.tsv" --out "$A/timers_train.lac" --per-utt 4 --seed 31 \
        > "$A/cache_timers_train.log" 2>&1
    $B/brosoundml_laya_audio_cache --align "$A/align_timers_test.tsv" --out "$A/timers_test.lac" --per-utt 1 \
        --after-word 0.3 --seed 32 > "$A/cache_timers_test.log" 2>&1
    ;;
cremad)
    $LAB cremad --dir D:/datasets/cremad --manifest "$A/cremad_manifest.tsv" --out "$A/labels_cremad.tsv"
    align "$A/cremad_manifest.tsv" cremad "$A/align_cremad.tsv"
    $B/brosoundml_laya_audio_cache --align "$A/align_cremad.tsv" --out "$A/cremad.lac" --per-utt 1 --after-word 0.3 \
        --seed 33 > "$A/cache_cremad.log" 2>&1
    ;;
slurp)
    $LAB slurp --jsonl D:/datasets/slurp/test.jsonl --audio-dir D:/datasets/slurp/slurp_real --subset slurp-test \
        --manifest "$A/slurp_test_manifest.tsv" --out "$A/labels_slurp_test.tsv"
    align "$A/slurp_test_manifest.tsv" slurp-test "$A/align_slurp_test.tsv"
    $B/brosoundml_laya_audio_cache --align "$A/align_slurp_test.tsv" --out "$A/slurp_test.lac" --per-utt 1 \
        --after-word 0.3 --seed 34 > "$A/cache_slurp_test.log" 2>&1
    ;;
ami)
    for s in dev test; do
        $LAB ami --dir D:/datasets/ami --align "$A2/align_ami_$s.tsv" --out "$A/labels_ami_$s.tsv"
    done
    ;;
*) echo "unknown: $WHAT"; exit 2 ;;
esac
echo "build_questions $WHAT done"

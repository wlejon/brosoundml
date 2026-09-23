#!/usr/bin/env bash
# The broadened laya-audio training mix: word timings and window caches for
# every domain, from the data the fetch_*.sh scripts leave in D:/datasets.
# Run from the brosoundml checkout; each step is skipped when its output
# exists. Steps can be run one domain at a time as downloads finish:
#
#   build_mix.sh nonspeech | libri | ami | vp LANG | mswc LANG
#
# Outputs go to $A (default D:/projects/brosoundml-data/laya-audio2); the
# first adapter's LibriTTS alignments and caches are reused from $L.
set -euo pipefail
A=${A:-D:/projects/brosoundml-data/laya-audio2}
L=${L:-D:/projects/brosoundml-data/laya-audio}
B=${B:-build-cuda/Release}
DS=${DS:-D:/datasets}
mkdir -p "$A"
AUG="--aug-rirs $DS/rirs_noises/rirs.list --aug-noises $A/aug_noise.list --aug-music $A/aug_music.list"

run() {  # run OUT LOG cmd... : skip when OUT exists
    local out=$1 log=$2
    shift 2
    if [ -e "$out" ]; then echo "have $out"; return 0; fi
    echo "make $out"
    "$@" > "$log" 2>&1 || { echo "FAILED: see $log"; tail -3 "$log"; exit 1; }
    tail -2 "$log"
}

case "${1:-}" in
nonspeech)
    # Noise: OpenSLR 28 point-source (public domain) + MUSAN sound-bible
    # (CC BY 3.0 / PD); music: MUSAN tracks kept by musan_filter_licenses.sh,
    # instrumental only. 15 % of the files (by name hash) are held out; the
    # augmentation lists are the training files only.
    (cat "$DS/rirs_noises/noises.list"; ls "$DS"/musan/noise/sound-bible/*.opus) > "$A/noises_all.list"
    awk -F'\t' '$3 == "N" {print $1}' "$DS/musan/music_kept.tsv" > "$A/music_novocals.list"
    run "$A/align_noise_train.tsv" "$A/prep_noise.log" $B/brosoundml_laya_audio_prep nonspeech \
        --list "$A/noises_all.list" --subset noise --out-prefix "$A/align_noise"
    run "$A/align_music_train.tsv" "$A/prep_music.log" $B/brosoundml_laya_audio_prep nonspeech \
        --list "$A/music_novocals.list" --subset music --out-prefix "$A/align_music"
    for s in noise music; do cut -f2 "$A/align_${s}_train.tsv" | sed 's/@.*//' | sort -u > "$A/aug_$s.list"; done
    run "$A/noise_train.lac" "$A/cache_noise_train.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_noise_train.tsv" --out "$A/noise_train.lac" --per-utt 3 --seed 41 \
        --aug-rirs "$DS/rirs_noises/rirs.list" --p-rir 0.3
    run "$A/music_train.lac" "$A/cache_music_train.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_music_train.tsv" --out "$A/music_train.lac" --per-utt 2 --seed 42 \
        --aug-rirs "$DS/rirs_noises/rirs.list" --p-rir 0.3
    run "$A/noise_test.lac" "$A/cache_noise_test.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_noise_test.tsv" --out "$A/noise_test.lac" --per-utt 2 --seed 43
    run "$A/music_test.lac" "$A/cache_music_test.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_music_test.tsv" --out "$A/music_test.lac" --per-utt 1 --seed 44
    ;;
libri)
    # One augmented window per train-clean-100 utterance, beside the first
    # adapter's clean caches.
    run "$A/libri_train_aug.lac" "$A/cache_libri_aug.log" $B/brosoundml_laya_audio_cache \
        --align "$L/align_train.tsv" --out "$A/libri_train_aug.lac" --per-utt 1 --seed 31 \
        $AUG --p-rir 0.5 --p-noise 0.5 --p-music 0.2
    ;;
ami)
    run "$A/align_ami_train.tsv" "$A/prep_ami.log" $B/brosoundml_laya_audio_prep ami \
        --dir "$DS/ami" --out-prefix "$A/align_ami"
    # Already far-field and noisy: light extra noise only.
    run "$A/ami_train.lac" "$A/cache_ami_train.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_ami_train.tsv" --out "$A/ami_train.lac" --per-utt 3 --seed 51 \
        $AUG --p-rir 0 --p-noise 0.2 --p-music 0.1
    run "$A/ami_dev.lac" "$A/cache_ami_dev.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_ami_dev.tsv" --out "$A/ami_dev.lac" --per-utt 1 --seed 52
    run "$A/ami_test.lac" "$A/cache_ami_test.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_ami_test.tsv" --out "$A/ami_test.lac" --per-utt 2 --seed 53
    ;;
vp)
    lg=$2
    for split in train dev test; do
        run "$A/align_vp_${lg}_$split.tsv" "$A/align_vp_${lg}_$split.log" $B/brosoundml_laya_audio_align \
            --manifest "$DS/voxpopuli/$lg/manifest.tsv" --subset "voxpopuli-$lg-$split" \
            --out "$A/align_vp_${lg}_$split.tsv"
    done
    run "$A/vp_${lg}_train.lac" "$A/cache_vp_${lg}_train.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_vp_${lg}_train.tsv" --out "$A/vp_${lg}_train.lac" --per-utt 2 --seed 61 \
        $AUG --p-rir 0.3 --p-noise 0.4 --p-music 0.15
    run "$A/vp_${lg}_dev.lac" "$A/cache_vp_${lg}_dev.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_vp_${lg}_dev.tsv" --out "$A/vp_${lg}_dev.lac" --per-utt 1 --seed 62
    run "$A/vp_${lg}_test.lac" "$A/cache_vp_${lg}_test.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_vp_${lg}_test.tsv" --out "$A/vp_${lg}_test.lac" --per-utt 2 --seed 63
    ;;
mswc)
    lg=$2
    run "$A/align_mswc_${lg}.tsv" "$A/prep_mswc_$lg.log" $B/brosoundml_laya_audio_prep clips \
        --manifest "$DS/mswc/$lg/manifest.tsv" --out "$A/align_mswc_${lg}.tsv"
    grep -P '\tmswc-[a-z]+-train\t' "$A/align_mswc_${lg}.tsv" > "$A/align_mswc_${lg}_train.tsv"
    grep -P '\tmswc-[a-z]+-test\t' "$A/align_mswc_${lg}.tsv" > "$A/align_mswc_${lg}_test.tsv"
    run "$A/mswc_${lg}_train.lac" "$A/cache_mswc_${lg}_train.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_mswc_${lg}_train.tsv" --out "$A/mswc_${lg}_train.lac" --per-utt 1 --seed 71 \
        $AUG --p-rir 0.3 --p-noise 0.4 --p-music 0.15
    # Test: the window ends 0-0.5 s after the word (a word just finished).
    run "$A/mswc_${lg}_test.lac" "$A/cache_mswc_${lg}_test.log" $B/brosoundml_laya_audio_cache \
        --align "$A/align_mswc_${lg}_test.tsv" --out "$A/mswc_${lg}_test.lac" --per-utt 1 --seed 72 \
        --after-word 0.5
    ;;
*)
    echo "usage: build_mix.sh nonspeech | libri | ami | vp LANG | mswc LANG" >&2
    exit 2
    ;;
esac

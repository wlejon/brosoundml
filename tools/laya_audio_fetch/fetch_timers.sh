#!/usr/bin/env bash
# Timers and Such v1.0 (SpeechBrain; real spoken assistant commands: set a
# timer, set an alarm, simple math, unit conversion). CC0 per the dataset
# paper and SpeechBrain recipe (the v0.1 Zenodo record states CC BY 4.0), so
# it is admitted for training.
#
# Only the REAL recordings (train/dev/test-real, about 0.2 GB) and the CSVs
# are taken. The 13.1 GB zip is mostly synthetic speech, so instead of
# streaming it whole this reads its central directory with HTTP range
# requests (brosoundml_laya_audio_labels zipdir) and fetches the byte range
# of each real subset, which is contiguous in the archive; a zip stream that
# starts at a local header extracts with bsdtar (Windows' System32 tar).
#
# Usage: fetch_timers.sh [DEST=D:/datasets/timers] [BUILD=build-cuda/Release]
set -euo pipefail
DEST=${1:-D:/datasets/timers}
B=$(cd "${2:-build-cuda/Release}" && pwd)
U="https://zenodo.org/records/4623772/files/timers-and-such-v1.0.zip?download=1"
BSDTAR=${BSDTAR:-/c/Windows/System32/tar.exe}
mkdir -p "$DEST"
cd "$DEST"
SIZE=$(curl -sIL "$U" | tr -d '\r' | awk 'tolower($1)=="content-length:" {v=$2} END {print v}')
# Zip64 end record: the last 22 + 20 + 56 bytes; its central directory size
# and offset sit at bytes 40 and 48 of the zip64 record.
curl -s -r $((SIZE - 98))-$((SIZE - 1)) -L "$U" -o _end.bin
CD_SIZE=$(od -An -tu8 -j 40 -N 8 _end.bin | tr -d ' ')
CD_OFF=$(od -An -tu8 -j 48 -N 8 _end.bin | tr -d ' ')
curl -s -r "$CD_OFF"-$((CD_OFF + CD_SIZE - 1)) -L "$U" -o _cd.bin
"$B/brosoundml_laya_audio_labels" zipdir --cd _cd.bin | sort -n > _dir.tsv
# Byte range [first entry of PREFIX, first entry after it) for each subset.
fetch_range() {
    local prefix=$1 from to
    from=$(awk -F'\t' -v p="$prefix" 'index($3, p) == 1 {print $1; exit}' _dir.tsv)
    to=$(awk -F'\t' -v f="$from" '$1 > f {n = $1} $1 > f && index($3, p) != 1 {print $1; exit}' p="$prefix" _dir.tsv)
    echo "timers: $prefix bytes $from..$to"
    # Via a file: from a pipe, bsdtar drops a lone last entry when the
    # stream ends without the next header.
    curl -s -r "$from"-$((to - 1)) -L "$U" -o _part.zip
    "$BSDTAR" -xf _part.zip 2> /dev/null || true
    rm -f _part.zip
}
for s in train-real/ dev-real/ test-real/ LICENSE train-real.csv dev-real.csv test-real.csv; do
    fetch_range "$s"
done
rm -f _end.bin _cd.bin
echo "timers: $(find train-real dev-real test-real -name '*.wav' | wc -l) real recordings, $(du -sh "$DEST" | cut -f1)"

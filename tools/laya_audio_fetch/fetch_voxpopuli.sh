#!/usr/bin/env bash
# VoxPopuli transcribed speech (CC0; raw audio (c) European Parliament,
# reproduction authorised with source acknowledged) from the ORIGINAL release:
# the per-language ASR annotation TSV plus the raw per-year session archives
# (dl.fbaipublicfiles.com/voxpopuli). The session archives (5-8 GB each) are
# streamed, never stored: each session .ogg is decoded once and cut at the
# annotated segment boundaries into 16 kHz mono Opus files; unwanted sessions
# and the gaps between segments are discarded.
#
# Kept per language: every dev/test segment of the chosen years and a
# deterministic random sample of train segments up to CAP_H hours. The
# "invalid" split is never used.
#
# Usage: fetch_voxpopuli.sh LANG "YEARS" CAP_H [DEST=D:/datasets/voxpopuli]
#   e.g. fetch_voxpopuli.sh en "2018" 50
set -euo pipefail
LANG_=$1
YEARS=$2
CAP_H=$3
DEST=${4:-D:/datasets/voxpopuli}
BASE=https://dl.fbaipublicfiles.com/voxpopuli
D=$DEST/$LANG_
mkdir -p "$D/_segs"
cd "$D"
[ -f asr_$LANG_.tsv.gz ] || curl -s -o asr_$LANG_.tsv.gz $BASE/annotations/asr/asr_$LANG_.tsv.gz

# Columns: id|paragraph_id|session_id|speaker_id|original_text|normed_text|decoded|start|end|cer|wer|vad|split|gender
yre=$(echo "$YEARS" | sed 's/ /|/g')
zcat asr_$LANG_.tsv.gz | awk -F'|' -v yre="^($yre)" 'NR>1 && $13!="invalid" && substr($3,1,4) ~ yre' > cand.tsv
awk -F'|' '$13=="train"' cand.tsv | shuf --random-source=<(yes) > train_shuf.tsv
{
    awk -F'|' '$13!="train"' cand.tsv
    awk -F'|' -v cap="$CAP_H" '{ h += ($9-$8)/3600; if (h <= cap) print }' train_shuf.tsv
} | sort -t'|' -k3,3 -k8,8g > selected.tsv
rm cand.tsv train_shuf.tsv
awk -F'|' '{n[$13]++; h[$13]+=($9-$8)/3600} END {for (s in n) printf "selected %s: %d segments, %.1f h\n", s, n[s], h[s]}' selected.tsv

# Per session: "start end id split" in time order, overlapping or abutting
# segments dropped so every kept segment is its own piece of the cut.
rm -f _segs/*.txt
awk -F'|' '{
    if ($3 != sess) { sess = $3; last = -1 }
    if ($8 < last + 0.05) next
    id = $1; gsub(":", "-", id)
    printf "%.3f %.3f %s %s\n", $8, $9, id, $13 >> ("_segs/" $3 ".txt")
    last = $9
}' selected.tsv

cut_session() {  # stdin: the session .ogg
    local sess=$1 list=$D/_segs/$1.txt tmp=$D/_cut_$1
    if [ ! -f "$list" ]; then cat > /dev/null; return 0; fi
    rm -rf "$tmp" && mkdir -p "$tmp"
    local times
    times=$(awk '{printf "%s%s,%s", (NR>1?",":""), $1, $2}' "$list")
    ffmpeg -nostdin -hide_banner -loglevel error -i pipe:0 -ac 1 -ar 16000 -c:a libopus -b:a 32k \
        -f segment -segment_times "$times" -segment_format ogg "$tmp/p%05d.opus"
    # Piece 2k+1 is segment k (piece 0 is the audio before the first one).
    awk '{printf "%05d %s %s\n", 2*(NR-1)+1, $3, $4}' "$list" | while read -r p id split; do
        mkdir -p "$D/$split"
        [ -f "$tmp/p$p.opus" ] && mv "$tmp/p$p.opus" "$D/$split/$id.opus"
    done
    rm -rf "$tmp"
    echo "$sess: $(wc -l < "$list") segments"
}
export -f cut_session
export D
for y in $YEARS; do
    curl -s $BASE/audios/${LANG_}_$y.tar | tar -xf - --to-command='
        case "$TAR_FILENAME" in
          *.ogg) s=$(basename "$TAR_FILENAME"); cut_session "${s%_*.ogg}" ;;
          *) cat > /dev/null ;;
        esac'
done

# Manifest: id \t audio \t speaker \t subset \t text (normalised text).
awk -F'|' -v D="$D" '{
    id = $1; gsub(":", "-", id)
    f = D "/" $13 "/" id ".opus"
    if ((getline line < f) < 0) next; close(f)
    spk = ($4 == "None" || $4 == "") ? $3 : $4
    printf "%s\t%s\t%s\tvoxpopuli-'"$LANG_"'-%s\t%s\n", id, f, spk, $13, $6
}' selected.tsv > manifest.tsv
rmdir _segs 2>/dev/null || true
echo "voxpopuli $LANG_: $(wc -l < manifest.tsv) segments in the manifest, $(du -sh "$D" | cut -f1)"

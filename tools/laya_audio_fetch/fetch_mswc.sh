#!/usr/bin/env bash
# Multilingual Spoken Words (MLCommons, CC BY 4.0; derived from Common Voice,
# CC0): a controlled keyword subset, Opus clips kept as distributed.
#
# MSWC's own train/dev/test splits share speakers (45k of the 47k en test
# speakers also occur in train), so this builds its own speaker-disjoint
# split out of the train pool: speakers whose hashed id starts with 0 or 1
# (1/8 of them) are TEST speakers and never contribute a training clip.
#
# Words (>= 3 characters, enough valid clips) are drawn at random and each
# gets a training budget from BUCKETS (examples per word); UNSEEN more words
# get no training clips at all. Every chosen word gets up to TEST_PER clips
# from test speakers. The random bucket assignment over one eligible pool
# keeps word frequency from confounding the examples-per-word curve.
#
# Only the chosen clips are extracted while the language archive streams
# past; the archive itself is never stored.
#
# Usage: fetch_mswc.sh LANG WORDS_PER_BUCKET "BUCKETS" UNSEEN TEST_PER [DEST=D:/datasets/mswc]
#   e.g. fetch_mswc.sh en 150 "1 2 4 8 16 32 64 128 256" 150 8
set -euo pipefail
LANG_=$1
PER=$2
BUCKETS=$3
UNSEEN=$4
TEST_PER=$5
DEST=${6:-D:/datasets/mswc}
# The original release (MLCommons' bucket): audio/<lang>.tar.gz holds
# <lang>/clips/<word>/<clip>.opus; splits/<lang>.tar.gz the split CSVs.
GCS=https://storage.googleapis.com/public-datasets-mswc
D=$DEST/$LANG_
mkdir -p "$D/clips" "$D/_tmp"
cd "$D"
if [ ! -f _tmp/train.csv ]; then
    curl -s "$GCS/splits/$LANG_.tar.gz" | tar -xzf - -C _tmp "${LANG_}_train.csv" version.txt
    mv "_tmp/${LANG_}_train.csv" _tmp/train.csv
    mv _tmp/version.txt version.txt
fi

# LINK,WORD,VALID,SPEAKER,GENDER -> selection.tsv: file word speaker split bucket
awk -F, -v per="$PER" -v buckets="$BUCKETS" -v unseen="$UNSEEN" -v test_per="$TEST_PER" '
BEGIN { srand(7); nb = split(buckets, B, " ") }
NR > 1 && $3 == "True" && length($2) >= 3 {
    w = $2; t = (substr($4, 1, 1) == "0" || substr($4, 1, 1) == "1")
    f = $1
    s = substr($4, 1, 16)
    if (t) { nt[w]++; tf[w, nt[w]] = f; ts[w, nt[w]] = s }
    else   { nr[w]++; rf[w, nr[w]] = f; rs[w, nr[w]] = s }
}
END {
    maxb = 0; for (i = 1; i <= nb; ++i) if (B[i] + 0 > maxb) maxb = B[i] + 0
    # Eligible: enough training clips for the largest bucket and a full test set.
    n = 0
    for (w in nr) if (nr[w] >= maxb && nt[w] >= test_per) E[++n] = w
    for (i = n; i > 1; --i) { j = int(rand() * i) + 1; tmp = E[i]; E[i] = E[j]; E[j] = tmp }
    k = 0
    for (b = 1; b <= nb; ++b) for (i = 0; i < per && k < n; ++i) { k++; W[k] = E[k]; budget[E[k]] = B[b] }
    for (i = 0; i < unseen && k < n; ++i) { k++; W[k] = E[k]; budget[E[k]] = 0 }
    printf "%d eligible words, %d chosen\n", n, k > "/dev/stderr"
    for (i = 1; i <= k; ++i) {
        w = W[i]
        for (c = 1; c <= nr[w]; ++c) idx[c] = c
        for (c = nr[w]; c > 1; --c) { j = int(rand() * c) + 1; tmp = idx[c]; idx[c] = idx[j]; idx[j] = tmp }
        for (c = 1; c <= budget[w]; ++c) printf "%s\t%s\t%s\ttrain\t%d\n", rf[w, idx[c]], w, rs[w, idx[c]], budget[w]
        for (c = 1; c <= test_per; ++c) printf "%s\t%s\t%s\ttest\t%d\n", tf[w, c], w, ts[w, c], budget[w]
    }
}' _tmp/train.csv > selection.tsv
awk -F'\t' -v L="$LANG_" '{print L "/clips/" $1}' selection.tsv > _tmp/wanted.txt
echo "selected $(wc -l < selection.tsv) clips"

# One pass over the language archive; members not in the list are skipped.
curl -s "$GCS/audio/$LANG_.tar.gz" | tar -xzf - -C clips --strip-components=2 -T _tmp/wanted.txt 2>/dev/null || true

# Manifest: id \t audio \t speaker \t subset \t text (the word).
awk -F'\t' -v D="$D" -v L="$LANG_" '{
    f = D "/clips/" $1
    if ((getline line < f) < 0) next; close(f)
    id = $1; sub(/\.opus$/, "", id); gsub("/", "_", id)
    printf "%s\t%s\t%s\tmswc-%s-%s\t%s\n", id, f, $3, L, $4, $2
}' selection.tsv > manifest.tsv
rm -rf _tmp
echo "mswc $LANG_: $(wc -l < manifest.tsv) clips in the manifest, $(du -sh "$D" | cut -f1)"

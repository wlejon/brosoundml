#!/usr/bin/env bash
# Keep only the MUSAN music tracks whose per-file licence allows training and
# releasing a model under Apache-2.0: public domain, CC0, CC BY (any
# version). Share-alike, no-derivatives, non-commercial and unlabelled
# tracks are DELETED. Each music/<dir>/LICENSE lists blocks of track ids
# followed by their licence line, separated by "=====" rules; rfm's LICENSE
# states one licence (CC BY 3.0, Kevin MacLeod / incompetech) for the whole
# folder in its header.
#
# Writes <DEST>/music_kept.tsv: file \t licence \t vocals(Y/N) \t artist.
#
# Usage: musan_filter_licenses.sh [DEST=D:/datasets/musan]
set -euo pipefail
DEST=${1:-D:/datasets/musan}
cd "$DEST/music"
: > ../music_kept.tsv
for d in */; do
    d=${d%/}
    [ -f "$d/LICENSE" ] || continue
    awk -v dir="$d" '
        function flush(   ok, i, l) {
            l = lic
            if (l == "" && head != "") l = head
            ok = (l ~ /[Pp]ublic [Dd]omain|CC0/ || l ~ /Attribution|CC BY/) &&
                 l !~ /Share[- ]?[Aa]like|BY-SA|NoDeriv|BY-ND|NonCommercial|Non-Commercial|BY-NC/
            for (i = 1; i <= n; ++i) printf "%s\t%s\t%s\n", ids[i], ok ? "keep" : "drop", l
            n = 0; lic = ""
        }
        /^=====/ { if (n == 0 && !seen_id) head = lic; flush(); next }
        /^music-[a-z-]+-[0-9]+/ { ids[++n] = $1; seen_id = 1; next }
        /CC|[Pp]ublic [Dd]omain|[Ll]icen|Attribution/ { gsub(/\r/, ""); lic = lic (lic == "" ? "" : " ; ") $0 }
        END { flush() }' "$d/LICENSE" > "$d/.verdict"
    while IFS=$'\t' read -r id verdict lic; do
        f="$d/$id.opus"
        [ -f "$f" ] || continue
        if [ "$verdict" = keep ]; then
            vocals=$(awk -v id="$id" '$1 == id {print $3}' "$d/ANNOTATIONS")
            artist=$(awk -v id="$id" '$1 == id {print $4}' "$d/ANNOTATIONS")
            printf "%s\t%s\t%s\t%s\n" "$DEST/music/$f" "$lic" "${vocals:-?}" "${artist:-?}" >> ../music_kept.tsv
        else
            rm -f "$f"
        fi
    done < "$d/.verdict"
    # Tracks the LICENSE never names have no licence: delete them too.
    for f in "$d"/*.opus; do
        id=$(basename "$f" .opus)
        grep -q "^$id	" "$d/.verdict" || rm -f "$f"
    done
    rm -f "$d/.verdict"
done
echo "kept $(wc -l < ../music_kept.tsv) music tracks ($(awk -F'\t' '$3=="N"' ../music_kept.tsv | wc -l) without vocals); $(find . -name '*.opus' | wc -l) opus files remain"

# noise/sound-bible: every block names one file and its licence line
# ("Attribution 3.0" or "Public Domain"). Files the LICENSE never names, or
# names with any other licence, are deleted.
cd "$DEST/noise/sound-bible"
awk '
    { sub(/[ \t\r]+$/, "") }
    /^noise-sound-bible-[0-9]+$/ { id = $0; next }
    /^License:/ && id != "" { if (!(id in lic)) lic[id] = $0 }
    END { for (i in lic) if (lic[i] ~ /Attribution 3\.0|Public Domain/ && lic[i] !~ /NonCommercial|NoDeriv|ShareAlike/) print i }
' LICENSE | sort > .keep
for f in *.opus; do
    grep -qx "$(basename "$f" .opus)" .keep || { echo "sound-bible: no usable licence for $f, deleted"; rm -f "$f"; }
done
rm -f .keep
echo "kept $(ls *.opus | wc -l) sound-bible noises"

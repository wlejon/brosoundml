#!/usr/bin/env bash
# Per-file attribution table for the MUSAN recordings the adapter was
# trained on, from MUSAN's own LICENSE blocks. A block lists one or more
# "<file id>" lines, each optionally followed by a title line, then the
# block's licence, recorder and source URL, ending in a "=====" rule.
#
# Usage: attributions.sh USED_LIST... > ATTRIBUTIONS.tsv
#   USED_LIST: files with one audio path per line (e.g. aug_music.list);
#   only those files are listed.
# Output: file id \t title \t author \t licence \t url
set -euo pipefail
MUSAN=${MUSAN:-D:/datasets/musan}
printf 'file\ttitle\tauthor\tlicence\turl\n'
cat "$@" | sed 's/@.*//; s#.*/##; s/\.[a-z0-9]*$//' | sort -u > "${TMPDIR:-/tmp}/attr_used.$$"
find "$MUSAN" -name LICENSE | while read -r lic; do
    awk -v used="${TMPDIR:-/tmp}/attr_used.$$" '
    BEGIN { while ((getline f < used) > 0) want[f] = 1 }
    # A block with a licence but no file ids is a file-wide header (rfm):
    # its licence applies to every block that names none.
    function flush(   i, a, t, l) {
        if (n == 0 && licence != "") deflic = licence
        for (i = 1; i <= n; ++i) if (ids[i] in want) {
            a = author; t = title[i]; l = licence != "" ? licence : deflic
            if (a == "" && match(t, /\(by [^)]*\)/)) a = substr(t, RSTART + 4, RLENGTH - 5)
            if (a == "" && match(l, / by .* is licensed/)) a = substr(l, RSTART + 4, RLENGTH - 16)
            printf "%s\t%s\t%s\t%s\t%s\n", ids[i], t, a, l, url
        }
        n = 0; author = ""; licence = ""; url = ""
    }
    { sub(/[ \t\r]+$/, "") }
    /^=+$/ { flush(); next }
    /^(music|noise)-[a-z-]+-[0-9]+$/ { ids[++n] = $0; title[n] = ""; next }
    /^http/ { url = $0; next }
    /^License:/ { l = $0; sub(/^License: */, "", l); sub(/ *$/, "", l); if (licence == "") licence = l; next }
    /^(CC |Attribution|Public Domain|CC0)/ { if (licence == "") licence = $0; next }
    /^Recorded by / { author = $0; sub(/^Recorded by */, "", author); sub(/ *$/, "", author); next }
    /^Performer: / { author = $0; sub(/^Performer: */, "", author); next }
    /^Title: / { if (n > 0) { title[n] = $0; sub(/^Title: */, "", title[n]) } next }
    / is licensed under / { if (licence == "") licence = $0; next }
    { if (n > 0 && title[n] == "") title[n] = $0 }
    END { flush() }' "$lic"
done
rm -f "${TMPDIR:-/tmp}/attr_used.$$"

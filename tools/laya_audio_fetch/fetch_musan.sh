#!/usr/bin/env bash
# MUSAN (OpenSLR 17, CC BY 4.0) music + sound-bible noise, streamed: the 11 GB
# archive is never stored. Each wanted member is transcoded on the fly to
# 16 kHz mono Opus; licence/annotation text files are kept as-is. The speech
# part is skipped (not needed), and noise/free-sound is skipped because the
# same public-domain recordings come with OpenSLR 28 (fetch_rirs_noises.sh).
# Afterwards musan_filter_licenses.sh deletes every file whose per-file
# licence is not CC0 / CC BY / public domain.
#
# Usage: fetch_musan.sh [DEST=D:/datasets/musan]
set -euo pipefail
DEST=${1:-D:/datasets/musan}
URL=https://openslr.elda.org/resources/17/musan.tar.gz
mkdir -p "$DEST"
export DEST
curl -s "$URL" | tar -xzf - --to-command='
    set -e
    case "$TAR_FILENAME" in
      musan/music/*|musan/noise/sound-bible/*) ;;
      *) cat > /dev/null; exit 0 ;;
    esac
    rel=${TAR_FILENAME#musan/}
    case "$rel" in
      *.wav)
        out="$DEST/${rel%.wav}.opus"
        mkdir -p "$(dirname "$out")"
        ffmpeg -nostdin -hide_banner -loglevel error -y -f wav -i pipe:0 -ac 1 -ar 16000 -c:a libopus -b:a 48k "$out" ;;
      *)
        mkdir -p "$(dirname "$DEST/$rel")"
        cat > "$DEST/$rel" ;;
    esac'
echo "musan: $(find "$DEST" -name '*.opus' | wc -l) files, $(du -sh "$DEST" | cut -f1)"

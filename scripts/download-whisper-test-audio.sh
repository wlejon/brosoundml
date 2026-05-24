#!/usr/bin/env bash
# Fetch a short English test clip + transcode to 16 kHz mono WAV.
#
# Drops <repo>/weights/whisper/test_audio_en.wav into place so the opt-in
# real-weights block in tests/test_whisper.cpp fires, and gives you a known
# clip to point brosoundml_transcribe at.
#
# Source: openai/whisper's `tests/jfk.flac` — a public-domain JFK speech
# excerpt that's been the canonical Whisper smoke clip since release.
#
# Requires curl + ffmpeg on PATH. Idempotent: skips if the WAV already
# exists (--force to overwrite).
#
# Usage:
#   scripts/download-whisper-test-audio.sh [--out-dir D] [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="$REPO_ROOT/weights/whisper"
FORCE=0
SRC_URL="https://github.com/openai/whisper/raw/main/tests/jfk.flac"

while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

command -v curl   >/dev/null || { echo "error: curl not on PATH"   >&2; exit 2; }
command -v ffmpeg >/dev/null || { echo "error: ffmpeg not on PATH" >&2; exit 2; }

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

WAV="$OUT_DIR/test_audio_en.wav"
FLAC="$OUT_DIR/.test_audio_en.flac"

if [ "$FORCE" -eq 0 ] && [ -s "$WAV" ]; then
    echo "$WAV already exists (use --force to overwrite)"
    exit 0
fi

echo "Fetching $SRC_URL ..."
curl -fL --retry 3 --retry-delay 2 -o "$FLAC.part" "$SRC_URL"
mv "$FLAC.part" "$FLAC"

echo "Transcoding to 16 kHz mono PCM16 WAV ..."
# -y overwrite, -ar 16k sample rate, -ac 1 mono, -sample_fmt s16 little-endian
# 16-bit PCM (what brosoundml::read_wav expects).
ffmpeg -y -loglevel error -i "$FLAC" -ar 16000 -ac 1 -sample_fmt s16 "$WAV"
rm -f "$FLAC"

dur="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$WAV" 2>/dev/null || echo '?')"
sz="$(wc -c < "$WAV" | tr -d ' ')"
printf 'Wrote %s  (%s bytes, %s s)\n' "$WAV" "$sz" "$dur"

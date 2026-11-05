#!/usr/bin/env bash
# Round 3: builds the SILENT film on the record's clock. The owner muxes the original recording later.
#   ./build.sh                       # 140.000 bpm, first click at 0.000 s  -> 4783 frames
#   BPM=139.6 OFFSET=0.12 ./build.sh # re-time for a measured tempo / first-click offset
# Needs: node + playwright + chromium (preinstalled here), python3 + Pillow, ffmpeg.
set -euo pipefail
cd "$(dirname "$0")"
FPS="${FPS:-60}"
CRF="${CRF:-21}"
BPM="${BPM:-140.0}"
OFFSET="${OFFSET:-0.0}"
OUT=art_of_dying_rhythm_game_record_silent.mp4
mkdir -p out stills

echo "== 1/4 timeline (rhythm.py -> timeline.js)"
python3 rhythm.py --bpm "$BPM" --offset "$OFFSET"

echo "== 2/4 frames (Playwright, stepped clock) -> silent H.264: about 8 minutes"
node render.js video "$OUT" "$FPS" "$CRF"

echo "== 3/4 stills and a contact sheet"
node render.js stills stills 0.6 16.0 19.4 38.7 57.9 72.0 77.3 79.5
python3 contact_sheet.py "$OUT" contact_sheet.png 3.2

echo "== 4/4 checks (expect 4783 frames at 140.000 bpm / 0.000 s; no audio stream)"
ffprobe -v error -count_frames -show_entries stream=codec_type,width,height,r_frame_rate,pix_fmt,nb_read_frames:format=duration,size \
  -of default=noprint_wrappers=1 "$OUT"

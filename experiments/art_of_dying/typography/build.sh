#!/usr/bin/env bash
# Build the film: soundtrack (numpy + piper words) and frames (Pillow) -> MP4.
# Usage: ./build.sh [out.mp4]      (about 2-3 minutes on 4 CPUs)
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-art_of_dying_typography.mp4}"
TMP="${TMPDIR:-/tmp}/aod_typo"
mkdir -p "$TMP"

[ -d words ] && [ -f words/five.wav ] || python3 voices.py
python3 audio.py "$TMP/soundtrack.wav"

python3 render.py pipe | ffmpeg -y -hide_banner -loglevel error \
  -f rawvideo -pix_fmt rgb24 -s 1080x1920 -r 60 -i - \
  -i "$TMP/soundtrack.wav" \
  -map 0:v -map 1:a -shortest \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -profile:v high -level 4.2 \
  -g 60 -movflags +faststart \
  -af "loudnorm=I=-14:TP=-1.5:LRA=11" -c:a aac -b:a 192k -ar 48000 \
  "$OUT"
ffprobe -v error -select_streams v -show_entries stream=width,height,r_frame_rate,nb_frames,pix_fmt \
  -show_entries format=duration,size -of default=nw=1 "$OUT"

#!/usr/bin/env bash
# Builds the film from scratch: soundtrack -> frames -> mux -> stills -> checks.
# Needs: node + playwright + chromium (preinstalled here), python3 + numpy, ffmpeg.
set -euo pipefail
cd "$(dirname "$0")"
FPS="${FPS:-60}"
CRF="${CRF:-21}"
mkdir -p out stills

echo "== 1/4 soundtrack (numpy)"
python3 synth.py out/soundtrack.wav

echo "== 2/4 frames (Playwright, stepped clock) -> silent H.264"
node render.js video out/video_silent.mp4 "$FPS" "$CRF"

echo "== 3/4 mux (peak limiter only: about -16 LUFS, true peak about -1.5 dBFS, dynamics kept)"
ffmpeg -y -loglevel error -i out/video_silent.mp4 -i out/soundtrack.wav \
  -af "alimiter=limit=0.7:attack=2:release=40:level=false" -ar 48000 \
  -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart art_of_dying_rhythm_game.mp4

echo "== 4/4 stills"
node render.js stills stills 0.9 7.2 15.3 20.95 25.0 40.4

echo "== checks"
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,r_frame_rate,pix_fmt,nb_read_frames:format=duration,size \
  -of default=noprint_wrappers=1 art_of_dying_rhythm_game.mp4
ffmpeg -hide_banner -i art_of_dying_rhythm_game.mp4 -af ebur128 -f null - 2>&1 | grep -E "^\s+(I|LRA):"

#!/usr/bin/env bash
# Builds the film from scratch: timeline -> soundtrack -> frames -> mux -> stills -> checks.
# Needs: node + playwright + chromium (preinstalled here), python3 + numpy + Pillow, ffmpeg.
set -euo pipefail
cd "$(dirname "$0")"
FPS="${FPS:-60}"
CRF="${CRF:-21}"
mkdir -p out stills

echo "== 1/5 timeline (rhythm.py -> timeline.js)"
python3 rhythm.py

echo "== 2/5 soundtrack (numpy): about 10 s"
python3 soundtrack.py out/soundtrack.wav

echo "== 3/5 frames (Playwright, stepped clock) -> silent H.264: about 4 minutes"
node render.js video out/video_silent.mp4 "$FPS" "$CRF"

echo "== 4/5 mux (the soundtrack is already mastered: -14 LUFS, -2 dBTP; AAC adds < 1 dB)"
ffmpeg -y -loglevel error -i out/video_silent.mp4 -i out/soundtrack.wav \
  -c:v copy -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart art_of_dying_rhythm_game.mp4

echo "== 5/5 stills and a contact sheet"
node render.js stills stills 0.6 7.0 13.0 19.45 26.0 31.0
python3 contact_sheet.py art_of_dying_rhythm_game.mp4 out/contact_sheet.png

echo "== checks"
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,r_frame_rate,pix_fmt,nb_read_frames:format=duration,size \
  -of default=noprint_wrappers=1 art_of_dying_rhythm_game.mp4
ffmpeg -hide_banner -i art_of_dying_rhythm_game.mp4 -af ebur128=peak=true -f null - 2>&1 | grep -E "^\s+(I|LRA|Peak):"
python3 loop_check.py art_of_dying_rhythm_game.mp4

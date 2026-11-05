#!/usr/bin/env bash
# Build the film: frames (Pillow, 4 workers) -> soundtrack (numpy, pre-mastered to -14 LUFS)
# -> MP4 (H.264 yuv420p + AAC 256k) -> stills. About 15 min on 4 CPUs.
# Usage: ./build.sh [workdir]   (defaults to ./build; the MP4 and stills land next to this script)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-$HERE/build}"
mkdir -p "$WORK/frames"
cd "$HERE"
python3 rhythm.py
python3 render.py --out "$WORK/frames" --workers 4
python3 soundtrack.py "$WORK/soundtrack.wav"
ffmpeg -y -v error -framerate 60 -i "$WORK/frames/f%05d.png" -i "$WORK/soundtrack.wav" \
  -c:v libx264 -preset slow -crf 27 -pix_fmt yuv420p -r 60 -movflags +faststart \
  -c:a aac -b:a 256k -shortest "$HERE/art_of_dying_gears.mp4"
python3 render.py --still 0 310 1170 1560 2060 2650 --out "$HERE/stills"
ffprobe -v error -select_streams v -count_frames \
  -show_entries stream=nb_read_frames,r_frame_rate,pix_fmt,width,height -show_entries format=duration,size \
  "$HERE/art_of_dying_gears.mp4"
ffmpeg -v info -i "$HERE/art_of_dying_gears.mp4" -af ebur128=peak=true -f null - 2>&1 | grep -E "^\s+(I|Peak):"

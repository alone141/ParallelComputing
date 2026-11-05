#!/usr/bin/env bash
# Build the film: frames (Pillow, 4 workers) -> audio (numpy) -> MP4 (ffmpeg, H.264 yuv420p, AAC).
# Usage: ./build.sh [workdir]   (defaults to ./build; final MP4 and stills land next to this script)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-$HERE/build}"
mkdir -p "$WORK/frames"
cd "$HERE"
python3 render.py --out "$WORK/frames" --workers 4
python3 audio.py "$WORK/audio.wav"
ffmpeg -y -v error -framerate 60 -i "$WORK/frames/f%05d.png" -i "$WORK/audio.wav" \
  -af "loudnorm=I=-14:TP=-1.5:LRA=9" \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -r 60 -movflags +faststart \
  -c:a aac -b:a 192k -shortest "$HERE/art_of_dying_gears.mp4"
python3 render.py --still 0 300 760 1120 1156 --out "$HERE/stills"
ffprobe -v error -select_streams v -count_frames -show_entries stream=nb_read_frames,r_frame_rate,pix_fmt,width,height \
  -show_entries format=duration,size "$HERE/art_of_dying_gears.mp4"

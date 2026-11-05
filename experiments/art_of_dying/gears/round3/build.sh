#!/usr/bin/env bash
# Round 3: render the SILENT picture for the original recording.
#   ./build.sh                       # 140.000 bpm, first click at 0.000 s -> 4783 frames
#   AOD_BPM=139.85 AOD_OFFSET=0.12 ./build.sh    # re-render on a measured tempo / offset
# Frames (Pillow, 4 workers, ~12 min on 4 CPUs) -> MP4 (H.264 yuv420p, no audio track) -> stills.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-$HERE/build}"
mkdir -p "$WORK/frames"
cd "$HERE"
python3 rhythm.py
rm -f "$WORK"/frames/f*.png
python3 render.py --out "$WORK/frames" --workers 4
ffmpeg -y -v error -framerate 60 -i "$WORK/frames/f%05d.png" \
  -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p -r 60 -movflags +faststart -an \
  "$HERE/art_of_dying_gears_record_silent_round3.mp4"
python3 render.py --still 0 1159 2320 3473 3700 4640 --out "$HERE/stills"
ffprobe -v error -select_streams v -count_frames \
  -show_entries stream=nb_read_frames,r_frame_rate,pix_fmt,width,height -show_entries format=duration,size \
  "$HERE/art_of_dying_gears_record_silent_round3.mp4"

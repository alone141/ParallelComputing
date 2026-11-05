#!/usr/bin/env bash
# Round 4/5: render the SILENT picture for the original recording (one gear per instrument).
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
  "$HERE/art_of_dying_gears_record_silent.mp4"
python3 render.py --still 0 1130 1159 2025 3415 3462 3473 3482 3700 4650 4782 --out "$HERE/stills"
ffprobe -v error -select_streams v -count_frames \
  -show_entries stream=nb_read_frames,r_frame_rate,pix_fmt,width,height -show_entries format=duration,size \
  "$HERE/art_of_dying_gears_record_silent.mp4"
python3 - "$HERE/stills" <<'PY'
import sys, glob, os
from PIL import Image
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "still_*.png")))
ims = [Image.open(f).resize((405, 720)) for f in files]
cols = 4
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (405 * cols, 720 * rows), (0, 0, 0))
for i, im in enumerate(ims):
    sheet.paste(im, ((i % cols) * 405, (i // cols) * 720))
sheet.save(os.path.join(d, "contact_sheet.png"))
print("contact sheet", len(ims), "stills")
PY

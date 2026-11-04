#!/usr/bin/env bash
# The Art of Dying, intro: render the picture and mux the soundtrack.
#   ./render.sh                     final:   1080x1920, 60 fps -> out/the_art_of_dying_intro.mp4
#   ./render.sh preview             preview:  540x960,  30 fps -> out/the_art_of_dying_intro_preview.mp4
#   START=19 DUR=5 ./render.sh preview     a time range in seconds (snapped to whole frames);
#                                   the file name then carries the range
# Environment: JOBS (parallel slices, default 4) · PYTHON · PRESET (x264, default slow)
#   GRAIN (luma grain added at encode time, default 3, 0 = off)
#   AUDIO=build (default: python3 soundtrack.py out/soundtrack.wav) | reuse (use the existing wav) | none
#
# The frames are rendered as JOBS frame-exact slices in parallel, each piped as raw RGB straight into
# its own ffmpeg; the slices are joined without re-encoding, then the soundtrack is muxed in.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
JOBS=${JOBS:-4}
PRESET=${PRESET:-slow}
GRAIN=${GRAIN:-3}
AUDIO=${AUDIO:-build}

case "${1:-final}" in
  final)   FPS=60; HALF=;       SIZE=1080x1920; TAG=intro ;;
  preview) FPS=30; HALF=--half; SIZE=540x960;   TAG=intro_preview ;;
  *) echo "usage: $0 [final|preview]   (START=<s> DUR=<s> for a range)" >&2; exit 2 ;;
esac

# Frame range [F0, F1) at FPS. The full film is round(TOTAL * FPS) frames (rhythm.N_FRAMES at 60).
read -r F0 F1 LABEL SS TT < <("$PY" -c '
import sys, rhythm as R
fps, start, dur = int(sys.argv[1]), sys.argv[2], sys.argv[3]
total = round(R.TOTAL * fps)
f0 = max(0, min(total, round(float(start or 0) * fps)))
f1 = total if not dur else max(f0, min(total, round((float(start or 0) + float(dur)) * fps)))
label = f"_{f0 / fps:.2f}-{f1 / fps:.2f}s" if (f0, f1) != (0, total) else "-"
print(f0, f1, label, f"{f0 / fps:.6f}", f"{(f1 - f0) / fps:.6f}")
' "$FPS" "${START:-}" "${DUR:-}")
[[ "$LABEL" == - ]] && LABEL=
FRAMES=$((F1 - F0))
((FRAMES > 0)) || { echo "empty frame range" >&2; exit 2; }
((JOBS > FRAMES)) && JOBS=$FRAMES
OUT="out/the_art_of_dying_${TAG}${LABEL}.mp4"
WORK="out/work/${TAG}${LABEL}"
mkdir -p "$WORK"
echo "rendering frames $F0-$((F1 - 1)) ($FRAMES @ $FPS fps, $SIZE) in $JOBS slices -> $OUT"

# ---- soundtrack (first, so a failure shows before the long render) -----------------------------
WAV=out/soundtrack.wav
case "$AUDIO" in
  build)
    if [[ -f soundtrack.py ]]; then
      "$PY" soundtrack.py "$WAV"
    else
      echo "warning: soundtrack.py not found, rendering silent" >&2; WAV=
    fi ;;
  reuse) [[ -f "$WAV" ]] || { echo "warning: $WAV not found, rendering silent" >&2; WAV=; } ;;
  none) WAV= ;;
  *) echo "AUDIO must be build, reuse or none" >&2; exit 2 ;;
esac

# ---- picture: parallel frame-exact slices -----------------------------------------------------
# BT.709 conversion, then grain on luma only (temporal, so it never freezes into a pattern).
VF="scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p,noise=c0s=${GRAIN}:c0f=t"
start_s=$(date +%s)
pids=()
for ((i = 0; i < JOBS; i++)); do
  a=$((F0 + FRAMES * i / JOBS)); b=$((F0 + FRAMES * (i + 1) / JOBS))
  (
    "$PY" film.py frames "$a" "$b" --fps "$FPS" $HALF |
      ffmpeg -y -hide_banner -loglevel error -f rawvideo -pix_fmt rgb24 -s "$SIZE" -framerate "$FPS" -i - \
        -vf "$VF" -c:v libx264 -preset "$PRESET" -crf 18 -profile:v high -pix_fmt yuv420p \
        -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
        "$WORK/part$i.mp4"
  ) > "$WORK/part$i.log" 2>&1 &
  pids+=($!)
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "slice $i failed, see $WORK/part$i.log" >&2; exit 1; }
done
echo "slices done in $(($(date +%s) - start_s)) s"

LIST="$WORK/parts.txt"
: > "$LIST"
for ((i = 0; i < JOBS; i++)); do echo "file 'part$i.mp4'" >> "$LIST"; done
ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i "$LIST" -c copy "$WORK/video.mp4"

# ---- mux --------------------------------------------------------------------------------------
if [[ -n "$WAV" ]]; then
  ffmpeg -y -hide_banner -loglevel error -i "$WORK/video.mp4" -ss "$SS" -t "$TT" -i "$WAV" \
    -map 0:v -map 1:a -c:v copy -c:a aac -b:a 256k -ar 48000 -movflags +faststart "$OUT"
else
  ffmpeg -y -hide_banner -loglevel error -i "$WORK/video.mp4" -map 0:v -c copy -movflags +faststart "$OUT"
fi

N=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$OUT")
[[ "$N" == "$FRAMES" ]] || { echo "error: $OUT has $N frames, expected $FRAMES" >&2; exit 1; }
echo "done: $OUT ($N frames$([[ -n "$WAV" ]] || echo ', silent'))"

#!/usr/bin/env bash
# The Long Number: render the film.
#
#   bash render.sh preview    540 x 960, 30 fps (every second frame), with sound  -> out/the_long_number_preview.mp4
#   bash render.sh final      1080 x 1920, 60 fps, 2208 frames, with sound        -> out/the_shuffle_final.mp4
#                                                                                   out/the_shuffle_final_silent.mp4
#                                                                                   out/the_shuffle_cover_hook.jpg
#
# Both modes: story.py -> build/story.json; four picture slices rendered in parallel by render.mjs (PNG frames
# piped into lossless x264); ffmpeg concat without re-encoding; soundtrack.py -> out/soundtrack.wav; the final
# encode (grain at encode time, yuv420p, AAC 256 kbps, +faststart); then checks.py on the result. The final
# mode also writes the silent copy (stream copy, no audio) and the cover (frame 144 of the final MP4), and
# deletes the lossless slices once the encode has succeeded.
#
# Environment overrides:
#   PYTHON   Python 3 with numpy (default: python3)
#   NODE     Node 22 that can require('playwright') (default: node)
#   GRAIN    luma grain strength at encode time (default 3; 0 turns it off)
#   PLAYWRIGHT_BROWSERS_PATH   where Playwright's browsers live (default /opt/pw-browsers when that folder exists).
#            This script never installs a browser; if Chromium is missing it says so and stops.
set -euo pipefail
cd "$(dirname "$0")"

MODE=${1:-final}
PY=${PYTHON:-python3}
NODE=${NODE:-node}
GRAIN=${GRAIN:-3}
if [[ -z "${PLAYWRIGHT_BROWSERS_PATH:-}" && -d /opt/pw-browsers ]]; then export PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers; fi

case "$MODE" in
  preview) TAG=preview; SLICE_FLAG=--preview; SLICE_FILE=preview_slice; OUT_MP4=out/the_long_number_preview.mp4 ;;
  final)   TAG=final;   SLICE_FLAG=--slice;   SLICE_FILE=slice;         OUT_MP4=out/the_shuffle_final.mp4 ;;
  *) echo "usage: bash render.sh preview|final" >&2; exit 2 ;;
esac

T_START=$SECONDS
stage_t0=$SECONDS
stage() {  # stage <name>: print the previous stage's elapsed time and start a new one
  local now=$SECONDS
  if [[ -n "${stage_name:-}" ]]; then printf '  [%s] %s: %d s\n' "$TAG" "$stage_name" $((now - stage_t0)); fi
  stage_name=$1; stage_t0=$now
  if [[ -n "$1" ]]; then printf '[%s] %s\n' "$TAG" "$1"; fi
  return 0
}
die() { echo "render.sh: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------- tooling
stage "tooling"
command -v ffmpeg >/dev/null || die "ffmpeg not found"
command -v ffprobe >/dev/null || die "ffprobe not found"
FILTERS=$(ffmpeg -hide_banner -filters 2>/dev/null || true)   # captured first: grep -q on a pipe trips pipefail
ENCODERS=$(ffmpeg -hide_banner -encoders 2>/dev/null || true)
grep -q ' ebur128 ' <<<"$FILTERS" || die "ffmpeg lacks the ebur128 filter"
grep -q 'libx264' <<<"$ENCODERS" || die "ffmpeg lacks libx264"
"$PY" -c "import numpy" 2>/dev/null || die "$PY cannot import numpy (pip install -r requirements.txt, or set PYTHON)"
"$NODE" -e "require('playwright')" 2>/dev/null || die "$NODE cannot require('playwright') (npm install, or set NODE / NODE_PATH)"
CHROME=$("$NODE" -e "console.log(require('playwright').chromium.executablePath())")
[[ -x "$CHROME" ]] || die "Playwright's Chromium is not at $CHROME; set PLAYWRIGHT_BROWSERS_PATH to the folder that holds it (this script does not run 'playwright install')"
echo "  python: $("$PY" --version 2>&1), numpy $("$PY" -c 'import numpy; print(numpy.__version__)')"
echo "  node: $("$NODE" --version), playwright $("$NODE" -e "console.log(require('playwright/package.json').version)"), chromium: $CHROME"
echo "  ffmpeg: $(ffmpeg -version | head -1 | cut -d' ' -f3)"
mkdir -p out out/logs build

# ------------------------------------------------------------------------------------------------ story
stage "story.py"
"$PY" story.py
SLICES=$("$PY" -c "import json; print(json.load(open('build/story.json'))['meta']['slices'])")
FRAMES=$("$PY" -c "import json; print(json.load(open('build/story.json'))['meta']['frames'])")
FPS=$("$PY" -c "import json; print(json.load(open('build/story.json'))['meta']['fps'])")
echo "  $FRAMES frames at $FPS fps in $SLICES slices"

# ----------------------------------------------------------------------------------------------- slices
stage "picture: $SLICES slices in parallel ($MODE)"
rm -f out/${SLICE_FILE}_*.mkv
pids=()
for ((i = 0; i < SLICES; i++)); do
  "$NODE" render.mjs $SLICE_FLAG "$i/$SLICES" > "out/logs/${TAG}_slice_$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "  slice $i FAILED (out/logs/${TAG}_slice_$i.log):" >&2; tail -5 "out/logs/${TAG}_slice_$i.log" >&2; fail=1; fi
done
[[ $fail == 0 ]] || die "a slice failed"
for ((i = 0; i < SLICES; i++)); do tail -1 "out/logs/${TAG}_slice_$i.log" | sed 's/^/  /'; done

# ----------------------------------------------------------------------------------------------- concat
stage "concat (stream copy)"
LIST=out/${TAG}_slices.txt
: > "$LIST"
for ((i = 0; i < SLICES; i++)); do
  f="out/${SLICE_FILE}_$i.mkv"; [[ -s "$f" ]] || die "missing slice $f"
  echo "file '$(realpath "$f")'" >> "$LIST"
done
LOSSLESS=out/${TAG}_lossless.mkv
ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "$LOSSLESS"
N_LOSSLESS=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$LOSSLESS")
echo "  $LOSSLESS: $N_LOSSLESS frames"

# ------------------------------------------------------------------------------------------------ sound
stage "soundtrack.py"
"$PY" soundtrack.py > out/logs/soundtrack.log 2>&1 || { tail -20 out/logs/soundtrack.log >&2; die "soundtrack.py failed (out/logs/soundtrack.log)"; }
grep -E "soundtrack.wav:|passed|FAILED" out/logs/soundtrack.log | sed 's/^/  /'
[[ -s out/soundtrack.wav ]] || die "out/soundtrack.wav was not written"

# ----------------------------------------------------------------------------------------------- encode
stage "encode -> $OUT_MP4"
if [[ "$MODE" == final ]]; then
  VF="noise=c0s=${GRAIN}:c0f=t,format=yuv420p"
  [[ "$GRAIN" == 0 ]] && VF="format=yuv420p"
  ffmpeg -y -loglevel error -i "$LOSSLESS" -i out/soundtrack.wav -map 0:v:0 -map 1:a:0 \
    -vf "$VF" -c:v libx264 -preset slow -crf 18 -movflags +faststart \
    -c:a aac -b:a 256k -shortest "$OUT_MP4"
else
  # the preview has no grain (it would be twice as coarse at half size) and a faster preset
  ffmpeg -y -loglevel error -i "$LOSSLESS" -i out/soundtrack.wav -map 0:v:0 -map 1:a:0 \
    -vf "format=yuv420p" -c:v libx264 -preset medium -crf 18 -movflags +faststart \
    -c:a aac -b:a 256k -shortest "$OUT_MP4"
fi
ffprobe -v error -show_entries format=duration -show_entries stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels -of default=nw=1 "$OUT_MP4" | grep -v '=0/0$' | sed 's/^/  /'

if [[ "$MODE" == final ]]; then
  stage "silent copy and cover"
  ffmpeg -y -loglevel error -i "$OUT_MP4" -map 0:v:0 -c copy -an -movflags +faststart out/the_shuffle_final_silent.mp4
  COVER_FRAME=$("$PY" -c "import json; print(json.load(open('build/story.json'))['meta']['loop']['cover_frame'])")
  ffmpeg -y -loglevel error -i "$OUT_MP4" -vf "select=eq(n\,${COVER_FRAME})" -fps_mode passthrough -frames:v 1 -q:v 2 -update 1 out/the_shuffle_cover_hook.jpg
  echo "  out/the_shuffle_final_silent.mp4: $(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 out/the_shuffle_final_silent.mp4) frames, no audio"
  echo "  out/the_shuffle_cover_hook.jpg: frame $COVER_FRAME, $(ffprobe -v error -show_entries stream=width,height -of csv=p=0:s=x out/the_shuffle_cover_hook.jpg)"
fi

# ---------------------------------------------------------------------------------------------- clean up
stage "delete the lossless slices"
rm -f out/${SLICE_FILE}_*.mkv "$LOSSLESS" "$LIST"

# ----------------------------------------------------------------------------------------------- checks
stage "checks.py"
if [[ "$MODE" == final ]]; then "$PY" checks.py "$OUT_MP4"; else "$PY" checks.py --preview "$OUT_MP4"; fi
stage ""
printf '[%s] done: %s in %d s\n' "$TAG" "$OUT_MP4" $((SECONDS - T_START))

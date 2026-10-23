#!/usr/bin/env bash
# Render the film and mux the soundtrack.
#   ./render.sh            final:   1080x1920, 60 fps
#   ./render.sh preview    preview:  540x960,  30 fps
# The film is rendered as JOBS frame-exact slices in parallel, then joined
# without re-encoding.  Set MANIM / PYTHON to use a specific environment.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
MANIM=${MANIM:-manim}
JOBS=${JOBS:-4}
if [[ "${1:-final}" == preview ]]; then RES=540,960; FPS=30; TAG=preview; else RES=1080,1920; FPS=60; TAG=final; fi

mkdir -p media
TOTAL_FRAMES=$("$PY" -c "from story import TIMELINE; print(round(TIMELINE.total * $FPS))")
"$PY" -c "from story import TIMELINE, MILESTONES; from visuals import ORDER
[c(m, TIMELINE) for c, m in zip(ORDER, MILESTONES)]" >/dev/null  # warm the texture cache once

pids=()
for ((i = 0; i < JOBS; i++)); do
  a=$((TOTAL_FRAMES * i / JOBS)); b=$((TOTAL_FRAMES * (i + 1) / JOBS))
  FILM_RANGE="$("$PY" -c "print(f'{$a/$FPS!r},{$b/$FPS!r}')")" \
    "$MANIM" -r "$RES" --fps "$FPS" --disable_caching --progress_bar none --media_dir "media/slices/$TAG$i" \
    -o "slice" film.py DoublingFilm > "media/render_${TAG}_part$i.log" 2>&1 &
  pids+=($!)
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "slice $i failed, see media/render_${TAG}_part$i.log" >&2; exit 1; }
done

LIST=media/parts_$TAG.txt
: > "$LIST"
for ((i = 0; i < JOBS; i++)); do
  echo "file '$(realpath "$(find "media/slices/$TAG$i/videos" -name slice.mp4 | head -n1)")'" >> "$LIST"
done
"$PY" soundtrack.py media/soundtrack.wav

mkdir -p out
GRAIN=${GRAIN:-3}  # luma film grain added at encode time (0 = off)
ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "media/doubling_$TAG.mp4"
ffmpeg -y -loglevel error -i "media/doubling_$TAG.mp4" -i media/soundtrack.wav -map 0:v -map 1:a \
  -vf "noise=c0s=${GRAIN}:c0f=t" -c:v libx264 -preset slow -crf 18 -profile:v high -pix_fmt yuv420p \
  -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart "out/the_doubling_${TAG}.mp4"
ffmpeg -y -loglevel error -i "out/the_doubling_${TAG}.mp4" -map 0:v -c copy -movflags +faststart \
  "out/the_doubling_${TAG}_silent.mp4"
echo "done: out/the_doubling_${TAG}.mp4 (+ _silent)"

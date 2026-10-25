#!/usr/bin/env bash
# Render The Shuffle and mux its score.
#   ./render.sh                final:   1080x1920, 60 fps → out/the_shuffle_final.mp4 (+ _silent)
#   ./render.sh preview        preview:  540x960,  30 fps → out/the_shuffle_preview.mp4
#   ./render.sh range A B      one 1080x1920 60 fps slice from A to B seconds, no sound → media/range.mp4
# The film is rendered as JOBS frame-exact slices in parallel, joined without re-encoding,
# then checked (frame count, loudness, loop).  Set MANIM / PYTHON to use a specific environment.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
MANIM=${MANIM:-manim}
JOBS=${JOBS:-4}
export PYTHONWARNINGS=ignore
mkdir -p media out

if [[ "${1:-final}" == range ]]; then
  FILM_RANGE="$2,$3" "$MANIM" -r 1080,1920 --fps 60 --disable_caching --progress_bar none --media_dir media/range \
    -o range film.py ShuffleFilm > media/render_range.log 2>&1
  cp "$(find media/range/videos -name range.mp4 | head -n1)" media/range.mp4
  echo "done: media/range.mp4 ($2 → $3 s)"; exit 0
fi
if [[ "${1:-final}" == preview ]]; then RES=540,960; FPS=30; TAG=preview; else RES=1080,1920; FPS=60; TAG=final; fi

TOTAL_FRAMES=$("$PY" -c "from story import TIMELINE; print(round(TIMELINE.total * $FPS))")
"$PY" -c "import textures; textures.point_disc(); textures.spiral(); textures.dust()" >/dev/null  # warm the cache once

pids=()
for ((i = 0; i < JOBS; i++)); do
  a=$((TOTAL_FRAMES * i / JOBS)); b=$((TOTAL_FRAMES * (i + 1) / JOBS))
  FILM_RANGE="$("$PY" -c "print(f'{$a/$FPS!r},{$b/$FPS!r}')")" \
    "$MANIM" -r "$RES" --fps "$FPS" --disable_caching --progress_bar none --media_dir "media/slices/$TAG$i" \
    -o "slice" film.py ShuffleFilm > "media/render_${TAG}_part$i.log" 2>&1 &
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
"$PY" soundtrack.py "media/soundtrack_$TAG.wav"

GRAIN=${GRAIN:-3}  # luma film grain added at encode time (0 = off)
ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "media/shuffle_$TAG.mp4"
ffmpeg -y -loglevel error -i "media/shuffle_$TAG.mp4" -i "media/soundtrack_$TAG.wav" -map 0:v -map 1:a \
  -vf "noise=c0s=${GRAIN}:c0f=t" -c:v libx264 -preset slow -crf 18 -profile:v high -pix_fmt yuv420p \
  -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart "out/the_shuffle_${TAG}.mp4"
ffmpeg -y -loglevel error -i "out/the_shuffle_${TAG}.mp4" -map 0:v -c copy -movflags +faststart \
  "out/the_shuffle_${TAG}_silent.mp4"
if [[ "$TAG" == final ]]; then  # the cover: the hook frame (1.9 s), taken unchanged from the final MP4
  ffmpeg -y -loglevel error -i "out/the_shuffle_final.mp4" -vf "select=eq(n\,$((19 * FPS / 10)))" -vsync 0 -frames:v 1 -q:v 2 \
    "out/the_shuffle_cover_hook.jpg"
fi
echo "done: out/the_shuffle_${TAG}.mp4 (+ _silent)"
"$PY" checks.py "out/the_shuffle_${TAG}.mp4" --fps "$FPS"

#!/usr/bin/env bash
# Mux the film audio into the engine's silent render and verify the result.
#
#   ./mux_verify.sh <silent.mp4> <film_audio.wav> <out.mp4> <record_map.json>
#
# Checks: the video stream is copied untouched and keeps every frame (no -shortest: the audio may
# be a few ms shorter than the picture); integrated loudness and true peak of the finished file; and
# the audio decoded back out of the MP4 is checked against the picture's step grid with
# edit_record.py verify, so an AAC encoder delay that a player would not compensate shows up here.
set -euo pipefail
silent=$1; audio=$2; out=$3; map=$4
here=$(cd "$(dirname "$0")" && pwd)
ffmpeg -v error -y -i "$silent" -i "$audio" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 256k -ar 48000 \
       -movflags +faststart "$out"
echo "== $out"
ffprobe -v error -count_frames -show_entries stream=codec_type,codec_name,nb_read_frames,duration:format=duration,size \
        -of compact "$out"
ffmpeg -hide_banner -nostats -i "$out" -af ebur128=peak=true -f null - 2>&1 | sed -n '/Summary/,$p' | grep -E "I:|Peak:"
tmp=$(mktemp --suffix=.wav)
ffmpeg -v error -y -i "$out" -vn -ac 2 -ar 48000 -c:a pcm_s24le "$tmp"
python3 "$here/edit_record.py" verify "$tmp" "$map"
rm -f "$tmp"

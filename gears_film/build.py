"""Frame rendering in parallel slices, the silent MP4, stills, the record map."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from multiprocessing import Pool

from .song import FPS, Song, load_song

_A = None
_SONG_PATH = None


def _init(song_path):
    global _A, _SONG_PATH
    from .render import Assets
    _SONG_PATH = song_path
    _A = Assets(load_song(song_path))


def _work(args):
    from .render import render_frame
    f, out = args
    render_frame(_A, f).save(out, compress_level=1)
    return f


def out_dir(song: Song) -> str:
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gears_film", "out", song.slug)
    os.makedirs(d, exist_ok=True)
    return d


def render_frames(song_path: str, song: Song, frames_dir: str, workers: int = 4, lo: int = 0, hi: int | None = None):
    os.makedirs(frames_dir, exist_ok=True)
    hi = song.n_frames if hi is None else hi
    jobs = [(f, os.path.join(frames_dir, f"f{f:05d}.png")) for f in range(lo, hi)]
    with Pool(workers, initializer=_init, initargs=(song_path,)) as p:
        for i, f in enumerate(p.imap_unordered(_work, jobs, chunksize=8)):
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    return len(jobs)


def encode(frames_dir: str, mp4: str, crf: int = 28):
    cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", os.path.join(frames_dir, "f%05d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-movflags", "+faststart", "-an", mp4]
    subprocess.run(cmd, check=True)


def count_frames(mp4: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", mp4],
                         check=True, capture_output=True, text=True).stdout.strip()
    return int(out)


def write_record_map(song: Song, path: str):
    with open(path, "w") as fh:
        json.dump(song.record_map(), fh, indent=1)
    return path


def render_film(song_path: str, workers: int = 4, crf: int = 28):
    song = load_song(song_path)
    problems = song.check()
    if problems:
        print("song data problems:\n  " + "\n  ".join(problems))
        sys.exit(1)
    d = out_dir(song)
    frames = os.path.join(d, "frames")
    print(f"{song.title}: {song.total_steps} steps, {song.total_s:.4f} s, {song.n_frames} frames -> {frames}")
    n = render_frames(song_path, song, frames, workers)
    mp4 = os.path.join(d, f"{song.slug}_silent.mp4")
    encode(frames, mp4, crf)
    got = count_frames(mp4)
    ok = got == song.n_frames == n
    print(f"wrote {mp4}: {got} frames (expected {song.n_frames}) {'OK' if ok else 'MISMATCH'}, "
          f"{os.path.getsize(mp4) / 1e6:.1f} MB")
    rm = write_record_map(song, os.path.join(d, "record_map.json"))
    print(f"wrote {rm}")
    if not ok:
        sys.exit(2)
    return mp4

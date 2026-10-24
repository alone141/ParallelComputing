"""Definition-of-done checks on a rendered film (film-style §8).

    python checks.py out/the_shuffle_final.mp4 [--fps 60]

Prints and checks: resolution, frame count (= round(total × fps) ± 1), integrated loudness
(-16 to -14 LUFS) and true peak (≤ -1 dBTP) on the final MP4, and the loop (mean absolute
pixel difference between the last frame and the first).  Exits non-zero on failure.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from soundtrack import measure  # noqa: E402
from story import TIMELINE  # noqa: E402


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_read_frames", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    w, h, rate, frames = r.stdout.strip().split(",")
    return int(w), int(h), rate, int(frames)


def frame(path, index, total):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = Path(f.name)
    sel = f"select=eq(n\\,{index})"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vf", sel, "-vsync", "0", "-frames:v", "1",
                    str(tmp)], check=True)
    img = np.asarray(Image.open(tmp).convert("RGB")).astype(np.float32)
    tmp.unlink()
    return img


def main(argv):
    path = Path(argv[0])
    fps = int(argv[argv.index("--fps") + 1]) if "--fps" in argv else 60
    silent = "--silent" in argv
    ok = True
    w, h, rate, frames = probe(path)
    want = round(TIMELINE.total * fps)
    good = abs(frames - want) <= 1
    ok &= good
    print(f"video: {w}x{h} @ {rate} fps, {frames} frames (want {want} ± 1) {'OK' if good else 'FAIL'}")
    if not silent:
        lufs, peak = measure(path)
        good = -16.0 <= lufs <= -14.0 and peak <= -1.0
        ok &= good
        print(f"audio: {lufs:.1f} LUFS integrated, true peak {peak:.1f} dBTP (want -16..-14, ≤ -1) {'OK' if good else 'FAIL'}")
    first, last = frame(path, 0, frames), frame(path, frames - 1, frames)
    diff = np.abs(first - last).mean()
    good = diff < 2.0
    ok &= good
    print(f"loop: mean |last - first| = {diff:.3f}/255 (want < 2) {'OK' if good else 'FAIL'}")
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

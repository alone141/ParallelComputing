#!/usr/bin/env python3
"""Render, inspect and assemble the reel.

  python reel/build.py render  b03_cpu [b04_gpu ...] [--final]
  python reel/build.py contact b03_cpu [--frames 12]      # preview + labelled contact sheet PNG
  python reel/build.py frames  b03_cpu 0.5 1.25 3.0       # preview + single frames at local times
  python reel/build.py handoff b03_cpu b04_gpu            # last frame of A next to first frame of B
  python reel/build.py film    [--final]                  # all beats -> concat -> grade -> audio -> out/
  python reel/build.py --reuse contact b03_cpu            # inspect the last preview without re-rendering

Beats are listed in timeline.py. Preview = 540x960 @ 30 fps, final = 1080x1920 @ 60 fps.
Run with the venv that has manim (default: /home/user/.venvs/manim/bin/python).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REEL = Path(__file__).resolve().parent
sys.path.insert(0, str(REEL))
from timeline import BEATS, FILM_SECONDS  # noqa: E402

BUILD = REEL / "build"
MEDIA = BUILD / "media"
PREVIEW_DIR = BUILD / "preview"
OUT = REEL / "out"
MANIM = os.environ.get("MANIM_BIN", "/home/user/.venvs/manim/bin/manim")
PYTHON = os.environ.get("REEL_PYTHON", "/home/user/.venvs/manim/bin/python")

QUALITY = {
    "preview": {"res": (540, 960), "fps": 30},
    "final": {"res": (1080, 1920), "fps": 60},
}

# Two-scale bloom, gentle vignette, fine grain. The bloom works on fixed-size
# intermediates and the grain is generated at 540x960 and upscaled, so the final
# 1080x1920 master looks exactly like the approved 540x960 previews.
GRADE = (
    "[1:v]format=gbrp,scale={w}:{h}[bgl];"
    "[0:v]format=gbrp[src];[src][bgl]blend=all_mode=lighten:shortest=1,split=3[base][b1][b2];"
    "[b1]scale=270:480,gblur=sigma=4,scale={w}:{h}:flags=bicubic[g1];"
    "[b2]scale=90:160,gblur=sigma=6,scale={w}:{h}:flags=bicubic[g2];"
    "[base][g1]blend=all_mode=screen:all_opacity=0.45[t1];"
    "[t1][g2]blend=all_mode=screen:all_opacity=0.65,"
    "vignette=angle=PI/7:aspect={w}/{h}[graded];"
    "color=c=gray:s=540x960:r={fps},format=gbrp,noise=alls=4:allf=t,"
    "scale={w}:{h}:flags=bicubic[grain];"
    "[graded][grain]blend=all_mode=grainmerge:all_opacity=1:shortest=1,format=yuv420p[v]"
)


def beat_lookup(key: str) -> tuple[str, str, float]:
    for cls, mod, dur in BEATS:
        if key in (cls, mod, Path(mod).stem):
            return cls, mod, dur
    raise SystemExit(f"unknown beat {key!r}; known: {[Path(m).stem for _, m, _ in BEATS]}")


def run(cmd: list[str], **kw) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    res = subprocess.run(cmd, **kw)
    if res.returncode != 0:
        raise SystemExit(f"command failed ({res.returncode}): {cmd[0]} ... (see output above)")


REUSE = False  # set by --reuse: skip rendering when the preview video already exists


def render(key: str, quality: str = "preview") -> Path:
    cls, mod, _ = beat_lookup(key)
    q = QUALITY[quality]
    w, h = q["res"]
    out = MEDIA / "videos" / Path(mod).stem / f"{h}p{q['fps']}" / f"{cls}.mp4"
    if REUSE and out.exists():
        return out
    env = dict(os.environ, PYTHONPATH=str(REEL))
    # Caching off: manim's cache key ignores updater code, so it can serve stale clips.
    run([MANIM, "render", str(REEL / mod), cls, "-r", f"{w},{h}", "--fps", str(q["fps"]),
         "--media_dir", str(MEDIA), "--disable_caching", "-v", "WARNING", "--progress_bar", "none"],
        env=env)
    if not out.exists():
        raise SystemExit(f"render output for {cls} not found at {out}")
    return out


def probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def frame_info(video: Path) -> tuple[int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
                          "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "csv=p=0",
                          str(video)], capture_output=True, text=True, check=True).stdout.strip()
    rate, n = out.split(",")[:2]
    num, den = rate.split("/")
    return int(n), float(num) / float(den)


def grab(video: Path, t: float, dest: Path) -> Path:
    """Exact frame at local time t (clamped; t < 0 counts from the end, -0.0001 = last frame)."""
    n, fps = frame_info(video)
    idx = int(round(t * fps)) if t >= 0 else n - 1 + int(round(t * fps))
    idx = min(max(idx, 0), n - 1)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{idx})",
         "-frames:v", "1", "-fps_mode", "passthrough", str(dest)])
    return dest


def contact(key: str, frames: int = 12) -> Path:
    """Grid of frames sampled across the beat, each stamped with its local time."""
    video = render(key)
    cls, mod, _ = beat_lookup(key)
    dur = probe_duration(video)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PREVIEW_DIR / f"_{cls}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    times = [dur * (i + 0.5) / frames for i in range(frames)]
    times[-1] = -0.0001               # always include the exit frame
    times[0] = 0.0                    # and the entry frame
    for i, t in enumerate(times):
        p = grab(video, t, tmp / f"raw{i:02d}.png")
        run(["ffmpeg", "-y", "-v", "error", "-i", str(p), "-vf",
             f"scale=270:480,drawtext=text='{max(t, 0) if t >= 0 else dur:.2f}s':x=8:y=8:fontsize=22:fontcolor=yellow:"
             f"box=1:boxcolor=black@0.6", str(tmp / f"f{i:02d}.png")])
    cols = 6 if frames > 8 else 4
    rows = -(-frames // cols)
    sheet = PREVIEW_DIR / f"{Path(mod).stem}_contact.png"
    run(["ffmpeg", "-y", "-v", "error", "-framerate", "1", "-i", str(tmp / "f%02d.png"),
         "-vf", f"tile={cols}x{rows}:padding=4:color=gray", "-frames:v", "1", str(sheet)])
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"contact sheet: {sheet}  (duration {dur:.2f}s)")
    return sheet


def frames(key: str, times: list[float]) -> list[Path]:
    video = render(key)
    cls, mod, _ = beat_lookup(key)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    outs = [grab(video, t, PREVIEW_DIR / f"{Path(mod).stem}_{t:05.2f}s.png") for t in times]
    for o in outs:
        print("frame:", o)
    return outs


def handoff(a: str, b: str) -> Path:
    """Exit frame of beat A beside the entry frame of beat B, plus a difference image."""
    va, vb = render(a), render(b)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fa = grab(va, -0.0001, PREVIEW_DIR / "_ha.png")
    fb = grab(vb, 0.0, PREVIEW_DIR / "_hb.png")
    out = PREVIEW_DIR / f"handoff_{Path(beat_lookup(a)[1]).stem}__{Path(beat_lookup(b)[1]).stem}.png"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(fa), "-i", str(fb), "-filter_complex",
         "[0]format=gbrp[a];[1]format=gbrp[b];[a][b]blend=all_mode=difference,"
         "lutrgb=r=val*4:g=val*4:b=val*4[d];[0]format=gbrp[a2];[1]format=gbrp[b2];[a2][b2][d]hstack=3",
         str(out)])
    print(f"handoff sheet (A exit | B entry | 4x difference): {out}")
    return out


def film(quality: str) -> None:
    q = QUALITY[quality]
    w, h = q["res"]
    parts = [render(cls, quality) for cls, _mod, _dur in BEATS]
    BUILD.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lst = BUILD / f"concat_{quality}.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    raw = BUILD / f"raw_{quality}.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(raw)])
    total = probe_duration(raw)
    print(f"raw film: {total:.3f}s (timeline says {FILM_SECONDS:.3f}s)")

    suffix = "" if quality == "final" else "_preview"
    silent = OUT / f"parallel_scales_silent{suffix}.mp4"
    bg = subprocess.run([PYTHON, "-c", "from style import background_png; print(background_png())"],
                        capture_output=True, text=True, check=True, cwd=REEL,
                        env=dict(os.environ, PYTHONPATH=str(REEL))).stdout.strip().splitlines()[-1]
    run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-loop", "1", "-i", bg,
         "-filter_complex", GRADE.format(w=w, h=h, fps=q["fps"]),
         "-map", "[v]", "-c:v", "libx264", "-preset", "slow" if quality == "final" else "veryfast",
         # Final: high quality but capped at ~10 Mb/s (~60 MB), under GitHub's 100 MB file limit and
         # phone upload caps; the platforms re-encode to far less anyway. Grain is the bit-hungry part.
         *(["-crf", "18", "-maxrate", "10M", "-bufsize", "20M"] if quality == "final" else ["-crf", "23"]),
         "-profile:v", "high", "-pix_fmt", "yuv420p",
         "-r", str(q["fps"]), "-movflags", "+faststart", str(silent)])

    wav = BUILD / "soundtrack.wav"
    # The timeline length is exact (frame-counted); the probed container duration can
    # come up a frame short, and trimming the tail would break the loop seam.
    run([PYTHON, str(REEL / "audio" / "synth.py"), "--out", str(wav), "--seconds", f"{FILM_SECONDS:.4f}"])
    scored = OUT / f"parallel_scales_scored{suffix}.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(wav), "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-t", f"{FILM_SECONDS:.4f}",
         "-movflags", "+faststart", str(scored)])
    print(f"done:\n  {silent}\n  {scored}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render"); r.add_argument("beats", nargs="+"); r.add_argument("--final", action="store_true")
    c = sub.add_parser("contact"); c.add_argument("beat"); c.add_argument("--frames", type=int, default=12)
    f = sub.add_parser("frames"); f.add_argument("beat"); f.add_argument("times", nargs="+", type=float)
    h = sub.add_parser("handoff"); h.add_argument("a"); h.add_argument("b")
    fm = sub.add_parser("film"); fm.add_argument("--final", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="inspect existing preview videos instead of re-rendering")
    a = ap.parse_args()
    global REUSE
    REUSE = a.reuse
    if a.cmd == "render":
        for b in a.beats:
            print(render(b, "final" if a.final else "preview"))
    elif a.cmd == "contact":
        contact(a.beat, a.frames)
    elif a.cmd == "frames":
        frames(a.beat, a.times)
    elif a.cmd == "handoff":
        handoff(a.a, a.b)
    elif a.cmd == "film":
        film("final" if a.final else "preview")


if __name__ == "__main__":
    main()

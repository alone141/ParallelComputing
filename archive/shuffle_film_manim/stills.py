"""Stills and layout checks for The Shuffle.

Every still is a real frame of the film (``film.py`` posed at a time), so nothing here
can drift from the build.

    python stills.py hook payoff_late beat1        # render named frames to media/stills/*.png
    python stills.py 9.8 21.0                      # or any time in seconds
    python stills.py --all                         # every named frame
    python stills.py --check [names|times]         # no render: print the collision / safe-zone report
    python stills.py --list                        # the named frames and their times

Set MANIM to use a specific manim (default: the one next to this Python).  ``--res W,H``
changes the resolution (default 1080,1920).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from story import BEATS, TIMELINE as tl  # noqa: E402

FPS = 60


def named_frames():
    a6, a11, a52 = tl.arrive["everyone"], tl.arrive["bigbang"], tl.arrive["deck"]
    b = {k.key: tl.beat_arrive[k.key] for k in BEATS}
    o = tl.outro_start
    frames = {
        "frame0": 0.0,
        "hook": 1.9,  # the hook frame: question and sub-line fully in, not yet leaving (the cover)
        "start": 2.7,  # the question has gone; the deck and its label
        "card1": tl.card_time(1) + 0.08,
        "card2": tl.card_time(2) + 0.08,
        "card3": tl.card_time(3) + 0.08,
        "everyone_arrive": a6 + 0.35,
        "everyone_mid": a6 + 1.0,
        "everyone_end": a6 + 2.0,
        "everyone_late": a6 + 2.6,
        "payoff_arrive": a11 + 0.35,
        "payoff_mid": a11 + 0.85,
        "payoff_end": a11 + 2.0,
        "payoff_late": a11 + 3.6,
        "sprint_early": tl.sprint_start + 1.0,
        "sprint_late": tl.sprint_start + 1.9,
        "final_arrive": a52 + 0.35,
        "final_note": a52 + 2.2,
        "beat1": b["second"] + 2.0,
        "beat2": b["everyone"] + 2.0,
        "beat3": b["planck"] + 2.0,
        "outro": o + 1.8,
        "glide": tl.total - 0.6,
        "last": tl.total - 1 / FPS,
    }
    return frames


def resolve(args):
    frames = named_frames()
    out = []
    for a in args:
        if a in frames:
            out.append((a, frames[a]))
        else:
            t = float(a)
            out.append((f"t{t:06.2f}".replace(".", "_"), t))
    return out


def render(items, res="1080,1920"):
    manim = os.environ.get("MANIM") or shutil.which("manim", path=str(Path(sys.executable).parent)) or "manim"
    tmp = HERE / "media" / "stills_tmp"
    out_dir = HERE / "media" / "stills"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in items:
        start = max(t - 1 / FPS, 0.0)
        env = dict(os.environ, FILM_RANGE=f"{start!r},{start + 2 / FPS!r}", PYTHONWARNINGS="ignore")
        subprocess.run([manim, "-r", res, "--fps", str(FPS), "--disable_caching", "--progress_bar", "none",
                        "--format", "png", "--media_dir", str(tmp), "-o", name, "film.py", "ShuffleFilm"],
                       cwd=HERE, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pngs = sorted(tmp.glob(f"images/film/{name}*.png"), key=lambda p: p.stat().st_mtime)
        if not pngs:
            raise SystemExit(f"no PNG produced for {name}")
        dest = out_dir / f"{name}.png"
        shutil.move(str(pngs[-1]), dest)
        for p in pngs[:-1]:
            p.unlink()
        print(f"{name:16s} t = {t:6.2f} s  → {dest.relative_to(HERE)}")


def check(items):
    from manim import config

    config.frame_rate = FPS
    import film

    scene = film.ShuffleFilm()
    scene.setup_film()
    bad = 0
    for name, t in items:
        problems, report = scene.report(t, hook_frame=(name == "hook"))
        print(f"[{name}] {report}")
        bad += len(problems)
    print(f"\n{len(items)} frames checked, {bad} problem(s)")
    return bad


def main(argv):
    if "--list" in argv:
        for k, v in named_frames().items():
            print(f"{k:16s} {v:6.2f} s")
        return 0
    res = "1080,1920"
    if "--res" in argv:
        i = argv.index("--res")
        res = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    names = [a for a in argv if not a.startswith("--")]
    if "--all" in argv or not names:
        names = list(named_frames())
    items = resolve(names)
    if "--check" in argv:
        return 1 if check(items) else 0
    render(items, res)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

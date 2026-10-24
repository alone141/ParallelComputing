"""THE SHUFFLE: a timed storyboard (animatic) of the ≈ 32 s cut, with a temporary click track.

Holds each key frame from stills.py for its real duration, with short crossfades and a burned-in
timecode and beat label, so the pacing can be judged before the film is built.  It is a review
tool, not the film: nothing moves inside a frame, and the sound is a click track, not the score.

    python animatic.py            # renders the frames it needs, then writes the MP4
    FORCE=1 python animatic.py    # re-renders every frame first

Output: media/animatic/the_shuffle_animatic.mp4 (540×960, 30 fps).  Set MANIM to use a specific
manim; by default it uses the one next to this Python.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media" / "animatic"
OUT = MEDIA / "the_shuffle_animatic.mp4"
W, H, FPS = 540, 960, 30
XFADE = 0.2  # seconds
SR = 48_000

# The timeline (README.md, "Structure"): (time the frame takes over, scene in stills.py, label)
HOOK, TEACH, TEACH_STEPS, SETTLE = 2.75, 2.75, (0.70, 0.60, 0.46), 0.06
C1 = TEACH + TEACH_STEPS[0]
C2 = C1 + SETTLE + TEACH_STEPS[1]
C3 = C2 + SETTLE + TEACH_STEPS[2]
TEACH_END = C3 + SETTLE
CARD6 = TEACH_END + 0.5 + 0.028 * 3  # an ordinary transition (film-style.md §2)
CARD11 = CARD6 + 3.2 + 0.5 + 0.028 * 5
SPRINT = CARD11 + 5.4
CARD52 = SPRINT + 2.5
BEATS = (CARD52 + 3.0, CARD52 + 5.6, CARD52 + 8.2)
OUTRO = BEATS[2] + 3.0
TOTAL = OUTRO + 4.2

FRAMES = [
    (0.00, "LoopFrame", "frame 0"),
    (0.30, "HookFrame", "HOOK"),
    (2.10, "StartFrame", "HOOK leaves · HUD fades in"),
    (C1, "TeachCard1", "TEACH · card 1 · ×52"),
    (C2, "TeachCard2", "TEACH · card 2 · ×51"),
    (C3, "TeachCard3", "TEACH · card 3 · ×50"),
    (CARD6, "EveryoneArrival", "STOP · card 6"),
    (CARD6 + 1.5, "EveryoneFrame", "STOP · card 6 · row"),
    (CARD11, "PayoffArrival", "PAYOFF · card 11"),
    (CARD11 + 0.85, "PayoffMidRow", "PAYOFF · row"),
    (CARD11 + 2.0, "PayoffRowEnd", "PAYOFF · row ends"),
    (CARD11 + 2.5, "PayoffFrame", "PAYOFF · note"),
    (SPRINT, "SprintEarly", "SPRINT · cards 12 → 52"),
    (SPRINT + 1.25, "SprintLate", "SPRINT · cards 12 → 52"),
    (CARD52, "FinalArrival", "FINAL · card 52"),
    (CARD52 + 1.2, "FinalFrame", "FINAL · note"),
    (BEATS[0], "ImagineSecond", "IMAGINE 1 · a shuffle a second"),
    (BEATS[1], "ImagineEveryone", "IMAGINE 2 · everyone, every second"),
    (BEATS[2], "ImaginePlanck", "IMAGINE 3 · every Planck time"),
    (OUTRO, "OutroFrame", "OUTRO"),
    (TOTAL - 1.2, "LoopFrame", "glide back to frame 0"),
]


def smoothstep(u):
    return u * u * (3 - 2 * u)


def card_times():
    """When each card lands: one at a time while teaching, then spread over each transition."""
    times = [C1, C2, C3]
    times += [TEACH_END + (CARD6 - TEACH_END) * k / 3 for k in (1, 2, 3)]  # cards 4–6
    start = CARD6 + 3.2
    times += [start + (CARD11 - start) * k / 5 for k in range(1, 6)]  # cards 7–11
    for k in range(1, 42):  # cards 12–52, eased over the 2.5 s sprint
        p, lo, hi = k / 41, 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if smoothstep(mid) < p else (lo, mid)
        times.append(SPRINT + 2.5 * hi)
    return times


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def render_frames():
    manim = os.environ.get("MANIM") or shutil.which("manim", path=str(Path(sys.executable).parent)) or "manim"
    for name in sorted({name for _, name, _ in FRAMES}):
        png = MEDIA / "images" / "stills" / f"{name}.png"
        if png.exists() and not os.environ.get("FORCE"):
            continue
        print(f"rendering {name}")
        subprocess.run([manim, "-r", f"{W},{H}", "-s", "--disable_caching", "--progress_bar", "none",
                        "--media_dir", str(MEDIA), "-o", name, "stills.py", name],
                       cwd=HERE, check=True, stdout=subprocess.DEVNULL)


def load_frames():
    cache = {}
    for _, name, _ in FRAMES:
        if name not in cache:
            img = Image.open(MEDIA / "images" / "stills" / f"{name}.png").convert("RGB").resize((W, H), Image.LANCZOS)
            cache[name] = np.asarray(img, np.float32)
    return cache


def overlay(img, t, label, fonts):
    """Burn in the timecode and beat label, top left, on a dark tab."""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([10, 10, 330, 58], radius=8, fill=(0, 0, 0, 150))
    d.text((20, 15), f"{t:05.2f} s", font=fonts[0], fill=(245, 247, 255, 255))
    d.text((20, 36), label, font=fonts[1], fill=(201, 209, 230, 255))
    d.rectangle([0, H - 4, int(W * t / TOTAL), H], fill=(185, 166, 255, 200))  # progress
    return np.asarray(im)


# ---------------------------------------------------------------------------
# Click track (a stand-in for the score of film-style.md §7)
# ---------------------------------------------------------------------------

NOTE = {"F2": 87.31, "A2": 110.0, "C3": 130.81, "E3": 164.81, "A3": 220.0, "B3": 246.94, "C4": 261.63,
        "E4": 329.63, "G4": 392.0, "A4": 440.0, "B4": 493.88, "C5": 523.25, "C#5": 554.37, "D5": 587.33,
        "E5": 659.25, "A5": 880.0, "B5": 987.77, "E6": 1318.51}


def soundtrack(path):
    n = int(TOTAL * SR) + SR
    out = np.zeros(n)
    rng = np.random.default_rng(3)

    def add(t, sig):
        i = int(t * SR)
        out[i:i + len(sig)] += sig[: max(0, n - i)]

    def env(dur, decay):
        x = np.arange(int(dur * SR)) / SR
        return x, np.exp(-x / decay)

    def tick(gain):
        x, e = env(0.03, 0.006)
        return gain * np.diff(rng.normal(0, 1, len(x) + 1)) * e

    def pluck(f, gain=0.22):
        x, e = env(0.8, 0.22)
        return gain * (np.sin(2 * np.pi * f * x) + 0.3 * np.sin(4 * np.pi * f * x)) * e

    def bell(f, gain=0.16):
        x, _ = env(2.5, 1)
        return gain * sum(a * np.sin(2 * np.pi * f * r * x) * np.exp(-x / d)
                          for r, a, d in ((1, 1, 0.9), (2.76, 0.4, 0.4), (5.4, 0.2, 0.2)))

    def boom(gain=0.55, f0=95, f1=34):
        x, e = env(1.6, 0.45)
        f = f1 + (f0 - f1) * np.exp(-x / 0.25)
        return gain * np.sin(2 * np.pi * np.cumsum(f) / SR) * e

    def chord(names, gain=0.07):
        x, e = env(3.0, 1.1)
        att = np.minimum(x / 0.03, 1)
        return gain * att * e * sum(np.sin(2 * np.pi * NOTE[k] * x) for k in names)

    def noise_swell(dur, gain=0.08):
        x = np.arange(int(dur * SR)) / SR
        s = np.convolve(rng.normal(0, 1, len(x)), np.ones(24) / 24, "same")
        return gain * s * (x / dur) ** 2

    add(0.0, boom(0.35, 70, 40))  # hook: a soft bloom and a bell
    add(0.0, bell(NOTE["A4"], 0.12))
    add(HOOK - 1.4, noise_swell(1.4))  # riser into the first card
    times = card_times()
    for k, t in enumerate(times, start=1):
        add(t, tick(0.10 if k <= 11 else 0.05))
    for t, f in zip(times[:3], ("E4", "G4", "A4")):
        add(t, pluck(NOTE[f]))
    add(CARD6, boom(0.4))  # card 6, the YOU moment: the "You" chord
    add(CARD6, chord(("A3", "E4", "A4", "C#5")))
    add(CARD11, boom(0.55))  # the payoff: A minor add9, then the answer shimmer with the note
    add(CARD11, chord(("A2", "E3", "A3", "C4", "E4", "B4")))
    for i, f in enumerate(("E5", "A5", "B5", "E6")):
        add(CARD11 + 2.5 + 0.09 * i, bell(NOTE[f], 0.06))
    add(SPRINT, noise_swell(2.5, 0.05))
    add(CARD52, boom(0.55))  # the finale: F major 7/9
    add(CARD52, chord(("F2", "C3", "A3", "E4", "C5"), 0.06))
    for t, f in zip(BEATS, ("C5", "D5", "E5")):  # a quieter bell per beat
        add(t, bell(NOTE[f], 0.1))
    add(OUTRO - 0.6, noise_swell(0.6, 0.06))  # outro: swell, boom and A minor
    add(OUTRO, boom(0.5))
    add(OUTRO, chord(("A2", "E3", "B3", "C4", "E4", "A4")))

    out = out[: int(TOTAL * SR)]
    out *= 10 ** (-1.5 / 20) / max(np.abs(out).max(), 1e-9)  # peak at -1.5 dBFS
    pcm = (np.repeat(out[:, None], 2, axis=1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble(wav):
    frames = load_frames()
    fonts = (ImageFont.truetype("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf", 17),
             ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Medium.otf", 14))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(OUT)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    starts = [s for s, _, _ in FRAMES]
    for f in range(round(TOTAL * FPS)):
        t = f / FPS
        i = max(k for k, s in enumerate(starts) if s <= t + 1e-9)
        img = frames[FRAMES[i][1]]
        a = min((t - starts[i]) / XFADE, 1.0) if i > 0 else 1.0
        if a < 1.0:
            img = img * a + frames[FRAMES[i - 1][1]] * (1 - a)
        proc.stdin.write(overlay(img.astype(np.uint8), t, FRAMES[i][2], fonts).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise SystemExit("ffmpeg failed")


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    render_frames()
    wav = MEDIA / "click_track.wav"
    soundtrack(wav)
    assemble(wav)
    print(f"done: {OUT} ({TOTAL:.2f} s, {len(FRAMES)} key frames)")
    for s, name, label in FRAMES:
        print(f"  {s:6.2f} s  {label:38s} {name}")


if __name__ == "__main__":
    main()

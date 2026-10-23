"""Shared design system for the vertical "Parallel computing at every scale" reel.

Every beat file imports * from here. It provides:
  * the vertical frame (9 x 16 units, 1 unit = 120 px at 1080x1920)
  * palette, typography and easing curves
  * the platform safe zone + a checker
  * BeatScene: background, a film-continuous dust field, audio cue logging,
    duration checking
  * small building blocks (glow, token, counter text, chips)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from manim import *  # noqa: F401,F403

# --------------------------------------------------------------------------
# Frame
# --------------------------------------------------------------------------
config.frame_height = 16.0
config.frame_width = 9.0
config.background_color = "#05060A"

FRAME_W = 9.0
FRAME_H = 16.0
PX_PER_UNIT = 120.0

# Region not covered by TikTok / Reels UI (top tabs, right action rail,
# bottom caption + music bar). Text and key action stay inside.
SAFE_LEFT, SAFE_RIGHT = -4.0, 3.3
SAFE_BOTTOM, SAFE_TOP = -4.3, 6.3
SAFE_CENTER = np.array([(SAFE_LEFT + SAFE_RIGHT) / 2, (SAFE_BOTTOM + SAFE_TOP) / 2, 0.0])

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
BG = ManimColor("#05060A")
BG_LIFT = ManimColor("#0B1020")
CYAN = ManimColor("#16E0FF")
MAGENTA = ManimColor("#FF2BD6")
AMBER = ManimColor("#FFB020")
LIME = ManimColor("#7CFF6B")
INK = ManimColor("#F5F7FF")  # "white" text
DIM = ManimColor("#5A6275")
DIMMER = ManimColor("#262B38")
RED_HOT = ManimColor("#FF3B5C")

ACCENTS = [CYAN, MAGENTA, AMBER, LIME]

# --------------------------------------------------------------------------
# Typography
# --------------------------------------------------------------------------
FONT_DISPLAY = "Inter Display"
FONT_BODY = "Inter"
FONT_MONO = "JetBrains Mono"

MIN_FONT_SIZE = 34  # smallest size that stays readable on a phone


def headline(text: str, size: float = 96, color=INK, weight=HEAVY, **kw) -> Text:
    """Big confident display type."""
    return Text(text, font=FONT_DISPLAY, font_size=size, color=color, weight=weight, **kw)


def body(text: str, size: float = 44, color=INK, weight=MEDIUM, **kw) -> Text:
    return Text(text, font=FONT_BODY, font_size=size, color=color, weight=weight, **kw)


def mono(text: str, size: float = 48, color=INK, weight=BOLD, **kw) -> Text:
    """Numbers, counters, code."""
    return Text(text, font=FONT_MONO, font_size=size, color=color, weight=weight, **kw)


def kicker(text: str, size: float = 34, color=CYAN, tracking: int = 300) -> MarkupText:
    """Small tracked-out uppercase label ("SCALE 03 · GPU"). tracking is in Pango units."""
    return MarkupText(
        f'<span letter_spacing="{tracking}">{text.upper()}</span>',
        font=FONT_BODY, font_size=size, color=color, weight=BOLD,
    )


def fit_width(mob: Mobject, max_width: float = SAFE_RIGHT - SAFE_LEFT - 0.4) -> Mobject:
    """Shrink (never grow) so it fits inside the safe width. Note: font_size 100
    is ~1 unit (120 px) tall here and the safe zone is only 7.3 units wide."""
    if mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}"


def counter_text(value_getter, size: float = 96, color=INK, font=FONT_MONO,
                 fmt=fmt_int, anchor=None, align=ORIGIN) -> Mobject:
    """A live number (DecimalNumber needs LaTeX, which we don't have).

    value_getter: callable -> float. anchor: point (or callable -> point) the
    text is aligned to; align: which edge of the text sits on the anchor.
    """
    def build():
        t = Text(fmt(value_getter()), font=font, font_size=size, color=color, weight=BOLD)
        a = anchor() if callable(anchor) else (anchor if anchor is not None else ORIGIN)
        if np.any(align):
            t.move_to(a, aligned_edge=align)
        else:
            t.move_to(a)
        return t
    return always_redraw(build)


# --------------------------------------------------------------------------
# Easing (premium motion = strong ease-out, rare linear)
# --------------------------------------------------------------------------
def ease_out_expo(t: float) -> float:
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def ease_in_expo(t: float) -> float:
    return 0.0 if t <= 0 else 2 ** (10 * t - 10)


def ease_in_out_expo(t: float) -> float:
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return 2 ** (20 * t - 10) / 2 if t < 0.5 else (2 - 2 ** (-20 * t + 10)) / 2


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_back(t: float, s: float = 1.70158) -> float:
    t -= 1
    return t * t * ((s + 1) * t + s) + 1


def ease_in_back(t: float, s: float = 1.70158) -> float:
    return t * t * ((s + 1) * t - s)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def glow(mob: VMobject, color=None, layers: int = 5, spread: float = 5.0,
         opacity: float = 0.07, base_width: float | None = None) -> VGroup:
    """Stack of widening, faint strokes behind `mob` (the post-process bloom
    does the rest). Returns VGroup(halo..., mob)."""
    color = color or mob.get_stroke_color()
    w = base_width if base_width is not None else max(mob.get_stroke_width(), 2)
    halo = VGroup()
    for i in range(layers, 0, -1):
        h = mob.copy()
        h.set_fill(opacity=0)
        h.set_stroke(color, width=w + i * spread, opacity=opacity)
        halo.add(h)
    return VGroup(halo, mob)


def glow_dot(point=ORIGIN, radius: float = 0.08, color=CYAN, halo: float = 4.0,
             layers: int = 6, core_opacity: float = 1.0) -> VGroup:
    """A bright dot with a soft circular halo."""
    g = VGroup()
    for i in range(layers, 0, -1):
        r = radius * (1 + (halo - 1) * i / layers)
        g.add(Circle(radius=r, stroke_width=0, fill_color=color,
                     fill_opacity=0.10 * (1 - i / (layers + 1)) ** 1.5))
    g.add(Circle(radius=radius, stroke_width=0, fill_color=color, fill_opacity=core_opacity))
    g.move_to(point)
    return g


def token(size: float = 1.0, color=CYAN, fill_opacity: float = 0.18,
          stroke_width: float = 4, corner: float = 0.22) -> RoundedRectangle:
    """The handoff unit: every scale collapses into one of these."""
    return RoundedRectangle(
        width=size, height=size, corner_radius=corner * size,
        stroke_color=color, stroke_width=stroke_width,
        fill_color=color, fill_opacity=fill_opacity,
    )


def chip(label: str, color=CYAN, size: float = 34, pad: float = 0.22) -> VGroup:
    """Pill-shaped label."""
    t = Text(label, font=FONT_MONO, font_size=size, color=color, weight=BOLD)
    box = RoundedRectangle(
        width=t.width + 2 * pad, height=t.height + 1.4 * pad, corner_radius=(t.height + 1.4 * pad) / 2,
        stroke_color=color, stroke_width=2.5, fill_color=color, fill_opacity=0.12,
    )
    t.move_to(box)
    return VGroup(box, t)


def in_safe_zone(mob: Mobject, tol: float = 1e-6) -> bool:
    l, r = mob.get_left()[0], mob.get_right()[0]
    b, t = mob.get_bottom()[1], mob.get_top()[1]
    return (l >= SAFE_LEFT - tol and r <= SAFE_RIGHT + tol
            and b >= SAFE_BOTTOM - tol and t <= SAFE_TOP + tol)


def assert_safe(*mobs: Mobject) -> None:
    for m in mobs:
        if not in_safe_zone(m):
            raise AssertionError(
                f"{type(m).__name__} outside safe zone: x[{m.get_left()[0]:.2f},{m.get_right()[0]:.2f}] "
                f"y[{m.get_bottom()[1]:.2f},{m.get_top()[1]:.2f}] vs x[{SAFE_LEFT},{SAFE_RIGHT}] "
                f"y[{SAFE_BOTTOM},{SAFE_TOP}]")


# --------------------------------------------------------------------------
# Background + film-continuous dust
# --------------------------------------------------------------------------
REEL_DIR = Path(__file__).resolve().parent
BUILD_DIR = REEL_DIR / "build"
ASSET_DIR = BUILD_DIR / "assets"
CUE_DIR = BUILD_DIR / "cues"


def background_png() -> str:
    """Radial lift in the middle of the safe area, generated once."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / "bg_1080x1920.png"
    if path.exists():
        return str(path)
    from PIL import Image
    w, h = 1080, 1920
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.47, h * 0.45
    d = np.sqrt(((xx - cx) / (w * 0.75)) ** 2 + ((yy - cy) / (h * 0.62)) ** 2)
    k = np.clip(1 - d, 0, 1) ** 1.8
    bg = np.array(BG.to_rgb()) * 255
    lift = np.array(BG_LIFT.to_rgb()) * 255
    img = bg[None, None, :] * (1 - k[..., None]) + lift[None, None, :] * k[..., None]
    rng = np.random.default_rng(7)  # dither to avoid gradient banding
    img = np.clip(img + rng.uniform(-0.6, 0.6, img.shape), 0, 255).astype(np.uint8)
    alpha = np.full(img.shape[:2] + (1,), 255, np.uint8)
    Image.fromarray(np.concatenate([img, alpha], axis=2), "RGBA").save(path)
    return str(path)


class DustField(VGroup):
    """Slowly rising specks whose positions depend only on *film* time, so
    they continue seamlessly across beat cuts and loop at FILM_SECONDS."""

    def __init__(self, scene: "BeatScene", n: int = 70, seed: int = 11, opacity: float = 0.35):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.scene_ref = scene
        self.x0 = rng.uniform(-FRAME_W / 2, FRAME_W / 2, n)
        self.y0 = rng.uniform(-FRAME_H / 2, FRAME_H / 2, n)
        self.laps = rng.integers(1, 3, n)  # whole laps per film => periodic
        self.sway = rng.uniform(0.05, 0.25, n)
        self.phase = rng.uniform(0, TAU, n)
        r = rng.uniform(0.012, 0.035, n)
        colors = [CYAN, INK, MAGENTA]
        for i in range(n):
            c = colors[int(rng.integers(0, len(colors)))]
            self.add(Dot(radius=r[i], color=c, fill_opacity=opacity * rng.uniform(0.35, 1.0)))
        self.add_updater(lambda m, dt: m._place())
        self._place()

    def _place(self) -> None:
        T = FILM_SECONDS
        t = self.scene_ref.film_time() % T
        y = (self.y0 + FRAME_H * self.laps * t / T + FRAME_H / 2) % FRAME_H - FRAME_H / 2
        x = self.x0 + self.sway * np.sin(TAU * self.laps * t / T * 3 + self.phase)
        for d, xi, yi in zip(self.submobjects, x, y):
            d.move_to([xi, yi, 0])


# --------------------------------------------------------------------------
# Beat registry + base scene
# --------------------------------------------------------------------------
from timeline import BEATS, FILM_SECONDS  # noqa: E402  (single source of truth)


def beat_start(name: str) -> float:
    t = 0.0
    for cls, _mod, dur in BEATS:
        if cls == name:
            return t
        t += dur
    return 0.0


def beat_duration(name: str) -> float | None:
    for cls, _mod, dur in BEATS:
        if cls == name:
            return dur
    return None


class BeatScene(Scene):
    """Base class for every beat.

    Subclasses implement `beat()` (not construct). Use `self.cue("hit")`
    right before the play() it should land on, or `self.cue("hit", dt=0.4)`
    to land 0.4 s into the next animation. Cues are written to
    build/cues/<SceneName>.json with film-absolute times for the audio mixer.

    Timing is frame-exact: the scene counts frames exactly as manim writes
    them and pads the end with a hold so the beat fills its timeline slot.
    Keep the exit state on screen at the end (the padding is that hold).
    """

    with_dust = True
    # Waits must render frame by frame (never freeze), so the dust keeps
    # drifting and frame counts stay exact.
    always_update_mobjects = True

    def setup(self):
        self._cues: list[dict] = []
        self._frames = 0
        # The background is flat BG here; build.py lifts it with a radial
        # gradient (bg PNG, "lighten" blend) in the grade, which is cheaper
        # than compositing an ImageMobject on every frame.
        if self.with_dust:
            self.dust = DustField(self)
            self.add(self.dust)

    @staticmethod
    def _frames_for(run_time: float) -> int:
        # identical to manim's own frame loop: np.arange(0, run_time, 1/fps)
        return len(np.arange(0, run_time, 1 / config.frame_rate))

    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        self._frames += self._frames_for(self.duration)

    @property
    def t(self) -> float:
        """Frame-exact local time (seconds) at the end of what has been played."""
        return self._frames / config.frame_rate

    def film_time(self) -> float:
        """Film-absolute time of the frame being drawn (used by DustField)."""
        return beat_start(type(self).__name__) + self.time

    def cue(self, kind: str, dt: float = 0.0, gain: float = 1.0, note: str = "", **params) -> None:
        self._cues.append({
            "t": round(beat_start(type(self).__name__) + self.t + dt, 4),
            "local_t": round(self.t + dt, 4),
            "type": kind, "gain": gain, "note": note, **params,
        })

    def hold_until(self, local_t: float) -> None:
        """Wait (frame-exact) until the given local time; no-op if already past."""
        n = int(round(local_t * config.frame_rate)) - self._frames
        if n > 0:
            self.wait((n - 0.5) / config.frame_rate)

    def construct(self):
        self.beat()
        name = type(self).__name__
        target = beat_duration(name)
        if target is not None:
            n_target = int(round(target * config.frame_rate))
            if self._frames < n_target:
                self.hold_until(target)
            elif self._frames > n_target:
                print(f"[beat] WARNING {name} runs {(self._frames - n_target) / config.frame_rate:.3f}s "
                      f"over its {target:.2f}s slot")
        CUE_DIR.mkdir(parents=True, exist_ok=True)
        out = {"scene": name, "start": beat_start(name), "duration": self.t, "frames": self._frames,
               "fps": config.frame_rate, "cues": self._cues}
        (CUE_DIR / f"{name}.json").write_text(json.dumps(out, indent=1))

    def beat(self):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

"""THE SHUFFLE: test stills for the treatment's open layout questions.

Poses single frames at full resolution with The Doubling's engine, so the
4-line hook (and later the new accents and the ring) can be checked before
the film is built (README.md).  Only the hook frame exists so far, and
nothing here animates.

    manim -r 1080,1920 -s --disable_caching stills.py HookFrame
    GUIDES=1 manim -r 1080,1920 -s --disable_caching stills.py HookFrame   # safe-zone overlay

The scene prints the measured extents of the hook so they can be checked
against film-style.md §1 and §5.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from manim import BOLD, MEDIUM, Circle, DashedLine, Rectangle, RoundedRectangle, Scene, Square, VGroup, VMobject, config

# Reuse The Doubling's engine until this film has its own.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "doubling_film"))
from engine import INTER_DISPLAY, MONO, FilmCamera, PostFX, fit_width, hex_rgb, mix, text  # noqa: E402
from visuals import D0, R0, polyline_points, sprite  # noqa: E402

config.frame_width = 9.0
config.frame_height = 16.0
config.background_color = "#04050B"

Y0 = 0.22  # where the featured object sits
WHITE_TXT = "#F5F7FF"
MUTED = "#8A93A8"
BODY = "#C9D1E6"
LAVENDER = "#B9A6FF"
CYAN = "#6CF2FF"
SAFE_W = 6.8

# The 4-line hook (film-style.md §5).  Size, pitch and centre are what this still checks.
HOOK_LINES = ["How many cards", "have more orders", "than seconds since", "the Big Bang?"]
HOOK_SIZE = float(os.environ.get("HOOK_SIZE", 42))
HOOK_PITCH = float(os.environ.get("HOOK_PITCH", 1.04))  # baseline to baseline, in em
HOOK_Y = float(os.environ.get("HOOK_Y", 4.78))
HOOK_STAGGER, HOOK_DUR_IN = 0.009, 0.32
SUB_Y = 3.3
FLAT = set("ABDEFHIKLMNPRTXZhiklmnrxz")  # glyphs whose bottom sits on the baseline


def em(size):
    """One em in scene units at a Manim font size (film-style.md: size 60 ≈ 100 px em)."""
    return 100 / 60 * size / 120


def baseline(mob, s):
    chars = [c for c in s if not c.isspace()]
    assert len(chars) == len(mob.submobjects), s
    return min(g.get_bottom()[1] for c, g in zip(chars, mob.submobjects) if c in FLAT)


def baseline_lines(rows, size, pitch, y, t2c=None, **kw):
    """Centred lines spaced by baseline, so lines with and without descenders keep an even
    rhythm (stacking by ink boxes packs a line without descenders closer to the next)."""
    g = VGroup()
    for i, s in enumerate(rows):
        r = text(s, size, t2c={k: v for k, v in (t2c or {}).items() if k in s}, **kw)
        r.shift([-r.get_center()[0], -i * pitch * em(size) - baseline(r, s), 0])
        g.add(r)
    return g.shift([0, y - g.get_center()[1], 0])


def markup(s, size, accent, **kw):
    """Text with [highlighted] spans, as in The Doubling."""
    hl = re.findall(r"\[(.+?)\]", s)
    return text(s.replace("[", "").replace("]", ""), size, t2c={h: accent for h in hl}, **kw)


def label_block(size_text, title, line, accent):
    """The start's label block: size text in the accent, no chip (film-style.md §2, §5)."""
    a = text(size_text, 25, font=MONO, weight=MEDIUM, color=accent).move_to([0, -2.86, 0])
    ti = fit_width(text(title, 60, font=INTER_DISPLAY, weight=BOLD, color=WHITE_TXT), SAFE_W).move_to([0, -3.51, 0])
    c = fit_width(markup(line, 31, accent, color=BODY, weight=MEDIUM), SAFE_W)
    c.move_to([0, -4.21 - (c.height - 0.32) / 2, 0])
    return [a, ti, c]


def card(height, accent, at=(0.0, Y0)):
    """One playing card (63 x 88 mm) drawn as lavender line art with a soft halo."""
    w = height * 63 / 88
    x, y = at
    halo = sprite("glow", D0 * 1.15, "#8F6BFF", alpha=0.5, at=at)
    body = RoundedRectangle(width=w, height=height, corner_radius=0.06 * height, stroke_color=accent,
                            stroke_width=3.2, fill_color=accent, fill_opacity=0.08).move_to([x, y, 0])
    inset = 0.055 * height
    inner = RoundedRectangle(width=w - 2 * inset, height=height - 2 * inset, corner_radius=0.04 * height,
                             stroke_color=accent, stroke_width=1.4, stroke_opacity=0.45, fill_opacity=0).move_to([x, y, 0])
    diamond = Square(side_length=0.2 * height, stroke_color=accent, stroke_width=2.2, fill_opacity=0).rotate(math.pi / 4)
    diamond.move_to([x, y, 0])
    core = sprite("star", 0.3 * height, "#E4DCFF", spikes=0.0, at=at)
    return [halo, body, inner, diamond, core]


def idle_warp(n, t, accent, count=150):
    """The Doubling's warp streaks and rings at rest (speed 0)."""
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, count)
    phase = rng.uniform(0, 1, count)
    bright = rng.uniform(0.3, 1.0, count)
    widths = rng.uniform(1.3, 2.8, count)
    pcol = mix(hex_rgb(accent), np.ones(3), 0.55)
    u = (((n + 0.06 * t) / 6.0) + phase) % 1.0
    r_head = 10.5 * 2.0 ** (-6.0 * u)
    r_tail = r_head * 2.0 ** 0.05
    alpha = np.sin(np.pi * u) ** 1.3 * bright * 0.36
    out = []
    for i in range(count):
        p = VMobject(stroke_color="#FFFFFF", stroke_width=widths[i], fill_opacity=0)
        c, s = math.cos(theta[i]), math.sin(theta[i])
        p.points = polyline_points([(c * r_tail[i], Y0 + s * r_tail[i]), (c * r_head[i], Y0 + s * r_head[i])])
        p.stroke_rgbas[:, :3] = pcol
        p.stroke_rgbas[:, 3] = alpha[i]
        out.append(p)
    f = n - math.floor(n)
    for j in range(9):
        r = R0 * 2.0 ** (1 - j - f)
        ring = Circle(radius=r, stroke_width=1.5, stroke_color="#FFFFFF").move_to([0, Y0, 0])
        ring.stroke_rgbas[:, :3] = pcol
        ring.stroke_rgbas[:, 3] = 0.05 * math.exp(-((math.log2(r / R0) / 2.2) ** 2))
        out.append(ring)
    return out


def guides():
    """Safe zones from film-style.md §1, drawn over the frame for checking."""
    red, amber, grey = "#FF4D5E", "#FFB547", "#8A93A8"
    g = [
        DashedLine([-4.5, 6.7, 0], [4.5, 6.7, 0], stroke_width=2, color=red),  # tab bar
        DashedLine([-4.5, 6.0, 0], [4.5, 6.0, 0], stroke_width=2, color=amber),  # Instagram 3:4 crop
        DashedLine([-4.5, -6.0, 0], [4.5, -6.0, 0], stroke_width=2, color=amber),
        DashedLine([-4.5, -4.8, 0], [4.5, -4.8, 0], stroke_width=2, color=red),  # caption bar
        DashedLine([-3.4, 8, 0], [-3.4, -8, 0], stroke_width=1.2, color=grey),  # centred-text width
        DashedLine([3.4, 8, 0], [3.4, -8, 0], stroke_width=1.2, color=grey),
        Rectangle(width=1.1, height=6.3, stroke_width=2, color=red).move_to([3.95, -1.65, 0]),  # button column
    ]
    return g


class HookFrame(Scene):
    """The hook frame (≈ 1.9 s, also the cover): the card, the question, the sub-line and the start label."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=FilmCamera, **kwargs)

    def construct(self):
        t, n = 1.9, 1.0
        bg = Rectangle(width=9.4, height=16.4, stroke_width=0, fill_color="#08061A", fill_opacity=1)
        nebula = sprite("glow", 14.6, LAVENDER, alpha=0.2, at=(0.0, Y0))
        q = baseline_lines(HOOK_LINES, HOOK_SIZE, HOOK_PITCH, HOOK_Y, font=INTER_DISPLAY, weight=BOLD,
                           color=WHITE_TXT, t2c={"orders": LAVENDER, "Big Bang?": CYAN})
        sub = text("Guess before it ends.", 27, color=MUTED).move_to([0, SUB_Y, 0])
        block = label_block("1 order", "ONE CARD", "It can only lie [one] way.", LAVENDER)
        mobs = [bg, nebula, *idle_warp(n, t, LAVENDER), *card(2.9, LAVENDER), q, sub, *block]
        if os.environ.get("GUIDES"):
            mobs += guides()
        self.add(*mobs, PostFX())

        glyphs = sum(len(re.sub(r"\s", "", s)) for s in HOOK_LINES)
        bases = [baseline(r, s) for r, s in zip(q, HOOK_LINES)]
        print(f"hook: size {HOOK_SIZE}, pitch {HOOK_PITCH} em, centre {HOOK_Y}")
        print(f"  baselines {', '.join(f'{b:.3f}' for b in bases)}")
        print(f"  top {q.get_top()[1]:.3f} (cover crop +6.0, tab bar +6.7), bottom {q.get_bottom()[1]:.3f}")
        print(f"  widest line {max(r.width for r in q):.3f} (limit {SAFE_W})")
        print(f"  sub-line top {sub.get_top()[1]:.3f}, gap to hook {q.get_bottom()[1] - sub.get_top()[1]:.3f}")
        print(f"  {glyphs} glyphs, last lands at {(glyphs - 1) * HOOK_STAGGER + HOOK_DUR_IN:.2f} s")

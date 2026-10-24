"""THE SHUFFLE: test stills for the treatment's open layout questions.

Poses single frames at full resolution with The Doubling's engine, so the
layout, the new accents and the ring can be checked before the film is
built (README.md).  Nothing here animates.

    manim -r 1080,1920 -s --disable_caching stills.py HookFrame        # the cover
    manim -r 1080,1920 -s --disable_caching stills.py PayoffFrame      # card 11, late in the hold
    manim -r 1080,1920 -s --disable_caching stills.py FinalFrame       # card 52
    manim -r 1080,1920 -s --disable_caching stills.py ImagineSecond    # the first "imagine" beat
    GUIDES=1 manim ... stills.py PayoffFrame                           # safe zones drawn over it

PayoffArrival, PayoffMidRow, PayoffRowEnd and FinalArrival pose the other
moments film-style.md §8 asks to check, and ImagineEveryone and
ImaginePlanck the other two beats.  The scenes print what they measure: the
hook's extents, and any text that overlaps other text, the ring, the meter
or the payoff brackets.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from manim import (
    BOLD, DOWN, MEDIUM, RIGHT, UP, Circle, DashedLine, DashedVMobject, Line, Rectangle, RoundedRectangle, Scene,
    Square, VGroup, VMobject, config,
)

# Reuse The Doubling's engine until this film has its own.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "doubling_film"))
from engine import (  # noqa: E402
    INTER, INTER_DISPLAY, MONO, FilmCamera, MonoFont, MonoLine, PostFX, clamp01, ease_in_out, ease_out, fit_width,
    hex_rgb, mix, text,
)
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
TIME = "#D4FF5C"  # time / the Big Bang (film-style.md §3)
DECK = "#E08BFF"  # cards (the full deck)
SAFE_W = 6.8

CARDS = 52
STOPS = [(6, CYAN), (11, TIME), (52, DECK)]
RING_R = 2.55  # the ring of 52 card slots, centred on the featured object
MX, MY0, MY1 = -3.95, -2.45, 3.45  # progress meter (film-style.md §4)
ROW_Y, ROW_X0, ROW_X1 = 3.75, -2.6, 2.6  # the row's track, where The Doubling's ruler sat

SECONDS = 13.787e9 * 365.25 * 86400  # since the Big Bang (Planck 2018)
PEOPLE = 8.3e9  # alive in 2026 (UN WPP 2024)
PLANCK_TIME = 5.391247e-44  # seconds (CODATA 2018)

# The 4-line hook (film-style.md §5).  Size, pitch and centre are what this still checks.
HOOK_LINES = ["How many cards dealt", "have more orders than", "seconds since", "the Big Bang?"]
HOOK_SIZE = float(os.environ.get("HOOK_SIZE", 42))
HOOK_PITCH = float(os.environ.get("HOOK_PITCH", 1.04))  # baseline to baseline, in em
HOOK_Y = float(os.environ.get("HOOK_Y", 4.78))
HOOK_STAGGER, HOOK_DUR_IN = 0.0085, 0.32
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
    """Text with [highlighted] spans, as in The Doubling; ``\\n`` makes extra centred lines."""
    rows = []
    for row in s.split("\n"):
        hl = re.findall(r"\[(.+?)\]", row)
        rows.append(text(row.replace("[", "").replace("]", ""), size, t2c={h: accent for h in hl}, **kw))
    return rows[0] if len(rows) == 1 else VGroup(*rows).arrange(DOWN, buff=0.1)


def label_block(size_text, title, line, accent, chip=None):
    """A label block (film-style.md §5): with a chip at a stop, without one at the start or a phase
    label, where the size text takes the accent instead."""
    sup = re.search("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]", size_text)
    a = text(size_text, 27 if sup else 25, font=INTER if sup else MONO, weight=MEDIUM,
             color=BODY if chip else accent)
    if chip:
        chip_txt = text(chip, 22, font=MONO, weight=BOLD, color=accent)
        pill = RoundedRectangle(width=chip_txt.width + 0.38, height=0.44, corner_radius=0.22, fill_color=accent,
                                fill_opacity=0.16, stroke_color=accent, stroke_width=1.6)
        chip_txt.move_to(pill)
        a = VGroup(VGroup(pill, chip_txt), a).arrange(RIGHT, buff=0.26)
    a.move_to([0, -2.86, 0])
    ti = fit_width(text(title, 60, font=INTER_DISPLAY, weight=BOLD, color=WHITE_TXT), SAFE_W).move_to([0, -3.51, 0])
    two = "\n" in line
    c = fit_width(markup(line, 29 if two else 31, accent, color=BODY, weight=MEDIUM), SAFE_W)
    # a two-line note grows downward; nudged as in The Doubling so its last baseline stays at or above -4.8
    c.move_to([0, -4.21 - (c.height - 0.32) / 2 + (0.05 if two else 0.0), 0])
    return a, ti, c


def card(height, accent, at=(0.0, Y0), halo=True):
    """One playing card (63 x 88 mm) drawn as line art, with a soft halo."""
    w = height * 63 / 88
    x, y = at
    out = [sprite("glow", D0 * 1.15, "#8F6BFF", alpha=0.5, at=at)] if halo else []
    # a lone card is lit from within; the top card of a deck is opaque, so the edges beneath stay hidden
    body = RoundedRectangle(width=w, height=height, corner_radius=0.06 * height, stroke_color=accent, stroke_width=3.2,
                            fill_color=accent if halo else "#0C0920", fill_opacity=0.08 if halo else 0.92)
    body.move_to([x, y, 0])
    inset = 0.055 * height
    inner = RoundedRectangle(width=w - 2 * inset, height=height - 2 * inset, corner_radius=0.04 * height,
                             stroke_color=accent, stroke_width=1.4, stroke_opacity=0.45, fill_opacity=0).move_to([x, y, 0])
    diamond = Square(side_length=0.2 * height, stroke_color=accent, stroke_width=2.2, fill_opacity=0).rotate(math.pi / 4)
    diamond.move_to([x, y, 0])
    return out + [body, inner, diamond, sprite("star", 0.3 * height, "#E4DCFF", spikes=0.0, at=at)]


def deck(height, accent, at=(0.0, Y0), depth=4):
    """A full deck: the top card over a few edges stepping down and right."""
    x, y = at
    out = [sprite("glow", D0 * 1.15, "#8F6BFF", alpha=0.5, at=at)]
    w = height * 63 / 88
    for k in range(depth, 0, -1):
        edge = RoundedRectangle(width=w, height=height, corner_radius=0.06 * height, stroke_color=accent,
                                stroke_width=1.6, stroke_opacity=0.25 + 0.1 * (depth - k), fill_color="#0C0920",
                                fill_opacity=0.92).move_to([x + 0.06 * k, y - 0.06 * k, 0])
        out.append(edge)
    return out + card(height, accent, at, halo=False)


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
    return [
        DashedLine([-4.5, 6.7, 0], [4.5, 6.7, 0], stroke_width=2, color=red),  # tab bar
        DashedLine([-4.5, 6.0, 0], [4.5, 6.0, 0], stroke_width=2, color=amber),  # Instagram 3:4 crop
        DashedLine([-4.5, -6.0, 0], [4.5, -6.0, 0], stroke_width=2, color=amber),
        DashedLine([-4.5, -4.8, 0], [4.5, -4.8, 0], stroke_width=2, color=red),  # caption bar
        DashedLine([-3.4, 8, 0], [-3.4, -8, 0], stroke_width=1.2, color=grey),  # centred-text width
        DashedLine([3.4, 8, 0], [3.4, -8, 0], stroke_width=1.2, color=grey),
        Rectangle(width=1.1, height=6.3, stroke_width=2, color=red).move_to([3.95, -1.65, 0]),  # button column
    ]


class HookFrame(Scene):
    """The hook frame (≈ 1.9 s, also the cover): the full deck, the question, the sub-line and the start label."""

    def __init__(self, **kwargs):
        super().__init__(camera_class=FilmCamera, **kwargs)

    def construct(self):
        t, n = 1.9, 0.0
        bg = Rectangle(width=9.4, height=16.4, stroke_width=0, fill_color="#08061A", fill_opacity=1)
        nebula = sprite("glow", 14.6, LAVENDER, alpha=0.2, at=(0.0, Y0))
        q = baseline_lines(HOOK_LINES, HOOK_SIZE, HOOK_PITCH, HOOK_Y, font=INTER_DISPLAY, weight=BOLD,
                           color=WHITE_TXT, t2c={"dealt": LAVENDER, "Big Bang?": CYAN})
        sub = text("Guess before it ends.", 27, color=MUTED).move_to([0, SUB_Y, 0])
        block = label_block("52 cards", "A FULL DECK", "Deal it [one card] at a time.", LAVENDER)
        mobs = [bg, nebula, *idle_warp(n, t, LAVENDER), *deck(2.9, LAVENDER), q, sub, *block]
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


# ---------------------------------------------------------------------------
# Stops
# ---------------------------------------------------------------------------


def orders(dealt):
    """The possible orders of ``dealt`` cards dealt from a full deck: 52 × 51 × … (one factor a card)."""
    return math.prod(range(CARDS - dealt + 1, CARDS + 1))


def log2_orders(dealt):
    return (math.lgamma(CARDS + 1) - math.lgamma(CARDS - dealt + 1)) / math.log(2)


def meter_y(dealt):
    return MY0 + (MY1 - MY0) * log2_orders(dealt) / log2_orders(CARDS)


def sci(value, digit_h, scale=1.0):
    """Three significant figures with a raised exponent, as a MonoLine spec."""
    e = int(math.floor(math.log10(value)))
    mant = value / 10**e
    if round(mant, 2) >= 10:
        mant, e = mant / 10, e + 1
    spec = [(c, scale, 0.0) for c in f"{mant:.2f}"]
    spec += [(" ", 0.45, 0), ("×", scale, 0.0), (" ", 0.45, 0), ("1", scale, 0.0), ("0", scale, 0.0)]
    return spec + [(c, 0.6 * scale, digit_h * 0.55) for c in str(e)]


def count_spec(value, digit_h, unit):
    """The count readout: exact with commas below 10⁷, then three significant figures with a
    raised exponent, so it matches a size text that shows the same count ("8.07 × 10⁶⁷")."""
    head = [(c, 1.0, 0.0) for c in f"{int(value):,}"] if value < 1e7 else sci(value, digit_h)
    return head + [(" ", 0.6, 0)] + [(c, 1.0, 0.0) for c in unit]


def counter_spec(value, digit_h, noun):
    """The live counter (film-style.md §4): commas, then words from ten million, then the readout's
    notation above 10²¹, where the words run out."""
    if value >= 1e21:
        head = sci(value, digit_h)
    else:
        words = f"{int(round(value)):,}"
        for div, name in ((1e18, "quintillion"), (1e15, "quadrillion"), (1e12, "trillion"), (1e9, "billion"),
                          (1e6, "million")):
            if value >= 1e7 and value >= div:
                words = f"{value / div:,.0f} {name}"
                break
        head = [(c, 1.0, 0.0) for c in words]
    return head + [(" ", 0.6, 0)] + [(c, 1.0, 0.0) for c in noun]


def deck_ring(dealt, accent=LAVENDER):
    """52 card slots clockwise from the top; the first ``dealt`` are filled."""
    glows, slots = [], []
    for i in range(CARDS):
        a = math.pi / 2 - 2 * math.pi * (i + 0.5) / CARDS
        x, y = RING_R * math.cos(a), Y0 + RING_R * math.sin(a)
        on = i < dealt
        slot = RoundedRectangle(width=0.19, height=0.27, corner_radius=0.035, stroke_color=accent,
                                stroke_width=1.8 if on else 1.1, stroke_opacity=0.95 if on else 0.22,
                                fill_color=accent, fill_opacity=0.28 if on else 0.0)
        slots.append(slot.rotate(a - math.pi / 2).move_to([x, y, 0]))
        if on:
            glows.append(sprite("glow", 0.55, accent, alpha=0.35, at=(x, y)))
    return glows, slots


def orders_cloud(count=260, radius=2.25, seed=4, colors=(LAVENDER,), brightness=1.0):
    """The orders: a faint dust of points behind the reference."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0, 1, count))
    th = rng.uniform(0, 2 * np.pi, count)
    cols = rng.choice(len(colors), count)
    return [Circle(radius=rad, stroke_width=0, fill_color=colors[c], fill_opacity=min(op * brightness, 1.0))
            .move_to([x, Y0 + y, 0])
            for x, y, rad, op, c in zip(r * np.cos(th), r * np.sin(th), rng.uniform(0.008, 0.02, count),
                                        rng.uniform(0.12, 0.5, count), cols)]


def big_bang(accent=TIME, r_max=1.35, turns=3):
    """Stand-in drawing: a bright point with time spiralling out from it to now."""
    th = np.linspace(0, 2 * np.pi * turns, 420)
    r = r_max * (th / th[-1]) ** 0.85
    pts = np.c_[r * np.cos(th + np.pi / 2), Y0 + r * np.sin(th + np.pi / 2)]
    out = [sprite("glow", 2.8, accent, alpha=0.45, at=(0.0, Y0))]
    for k, idx in enumerate(np.array_split(np.arange(len(pts)), 6)):
        v = VMobject(stroke_color=accent, stroke_width=2.6, stroke_opacity=0.35 + 0.1 * k, fill_opacity=0)
        v.points = polyline_points(pts[max(idx[0] - 1, 0): idx[-1] + 1])
        out.append(v)
    return out + [sprite("star", 0.85, "#F4FFD6", spikes=0.35, at=(0.0, Y0)),
                  sprite("star", 0.32, "#FFFFFF", spikes=0.0, at=tuple(pts[-1]))]


def crowd(radius=1.5, count=420, seed=9):
    """Stand-in drawing: everyone alive as a disc of cyan points."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0, 1, count))
    th = rng.uniform(0, 2 * np.pi, count)
    return [sprite("glow", 3.2, CYAN, alpha=0.3, at=(0.0, Y0))] + [
        Circle(radius=0.022, stroke_width=0, fill_color=CYAN, fill_opacity=op).move_to([x, Y0 + y, 0])
        for x, y, op in zip(r * np.cos(th), r * np.sin(th), rng.uniform(0.35, 0.9, count))]


def planck_dot(radius=1.3):
    """Stand-in drawing: the Planck dot inside The Doubling's dashed reticle."""
    circ = DashedVMobject(Circle(radius=radius, stroke_width=1.6, color=LAVENDER), num_dashes=70).move_to([0, Y0, 0])
    ticks = [Line([radius - 0.1, 0, 0], [radius + 0.1, 0, 0], stroke_width=1.8, color=LAVENDER)
             .rotate(a, about_point=np.zeros(3)).shift([0, Y0, 0]) for a in (0, np.pi / 2, np.pi, 3 * np.pi / 2)]
    return [sprite("glow", 2.6, "#8F6BFF", alpha=0.5, at=(0.0, Y0)), circ, *ticks,
            sprite("star", 0.3, "#E4DCFF", spikes=0.0, at=(0.0, Y0))]


class StopScene(Scene):
    """A frame ``AT`` seconds after a stop arrives, with the shared HUD, meter, track and overlap report."""

    AT, CARD, ACCENT = 0.0, 11, TIME

    def __init__(self, **kwargs):
        super().__init__(camera_class=FilmCamera, **kwargs)

    def hud(self):
        """The counter (it punches and flashes on arrival), its label and the count readout."""
        at = self.AT
        punch = math.sin(math.pi * at / 0.5) if at <= 0.5 else 0.0
        flash = math.exp(-at / 0.45)
        counter = MonoLine(MonoFont(size=104, weight=BOLD), 4)
        counter.show([(ch, 1 + 0.16 * punch, 0.0) for ch in str(self.CARD)], (0, 5.6))
        col = mix(np.ones(3), hex_rgb(self.ACCENT), 0.85 * flash)
        for slot in counter.submobjects:
            slot.fill_rgbas[:, :3] = col
        label = text("CARDS DEALT", 19, weight=MEDIUM, color=MUTED, tracking=0.09).move_to([0, 6.55, 0])
        small = MonoFont(size=30, weight=MEDIUM, letters=True)
        readout = MonoLine(small, 22, color="#9FB3D9")
        readout.show(count_spec(float(orders(self.CARD)), small.digit_h, "orders"), (0, 4.5))
        mobs = [label, readout]
        if flash > 0.01:
            mobs.append(sprite("glow", 3.2, self.ACCENT, alpha=0.35 * flash, at=(0.0, 5.6)))
        mobs.append(counter)
        return mobs, [("hud label", label), ("counter", counter), ("readout", readout)]

    def meter(self):
        """log2 of the orders, from no cards dealt to the full deck."""
        y = meter_y(self.CARD)
        fill = Line([MX, MY0, 0], [MX, y, 0], stroke_width=3.2)
        fill.set_stroke([LAVENDER, DECK]).set_sheen_direction(UP)
        mobs = [Line([MX, MY0, 0], [MX, MY1, 0], stroke_width=2, color="#27304A"), fill]
        for card_n, acc in STOPS:
            tk = Line([MX - 0.08, meter_y(card_n), 0], [MX + 0.08, meter_y(card_n), 0], stroke_width=2.2, color=acc)
            mobs.append(tk.set_stroke(opacity=1.0 if card_n <= self.CARD else 0.3))
        start = text("0", 17, font=MONO, color=MUTED).move_to([MX, MY0 - 0.26, 0])
        mobs += [start, sprite("star", 0.55, "#FFFFFF", spikes=0.0, at=(MX, y))]
        return mobs, [("meter start", start)]

    @staticmethod
    def track():
        """The row's track, under the count readout where The Doubling's ruler was."""
        out = [Line([ROW_X0, ROW_Y, 0], [ROW_X1, ROW_Y, 0], stroke_width=2, color="#27304A")]
        return out + [Line([x, ROW_Y - 0.1, 0], [x, ROW_Y + 0.1, 0], stroke_width=1.6, color="#5D6B8C")
                      for x in (ROW_X0, ROW_X1)]

    @staticmethod
    def glow_line(x0, x1, color):
        return [Line([x0, ROW_Y, 0], [x1, ROW_Y, 0], stroke_width=10, color=color, stroke_opacity=0.22),
                Line([x0, ROW_Y, 0], [x1, ROW_Y, 0], stroke_width=3.2, color=color)]

    def copies_row(self, progress, count, noun, color):
        """Copies of the smaller count lining up across the orders (film-style.md §4): a glowing line with
        ~90 beads and a star at the growing end, its counter climbing to ``count``."""
        x1 = mix(ROW_X0, ROW_X1, progress)
        mobs = self.glow_line(ROW_X0, x1, color)
        mobs += [Circle(radius=0.022, stroke_width=0, fill_color="#FFFFFF", fill_opacity=0.9).move_to([x, ROW_Y, 0])
                 for x in np.linspace(ROW_X0, ROW_X1, 90) if x <= x1]
        mobs.append(sprite("star", 0.4, "#FFFFFF", spikes=0.35, at=(x1, ROW_Y)))
        cnt, t = self.row_counter(count * progress, noun, (ROW_X0 + x1) / 2, color)
        return mobs + cnt, t

    @staticmethod
    def row_counter(value, noun, x_mid, color):
        """The live counter, 0.3 below the track; it starts no further left than the track, which
        keeps it clear of the meter and the payoff brackets."""
        font = MonoFont(size=21, weight=BOLD, letters=True)
        spec = counter_spec(value, font.digit_h, noun)
        w = sum(s for _, s, _ in spec) * font.adv
        x = min(max(x_mid, ROW_X0 + w / 2), 3.3 - w / 2)
        shadow = MonoLine(font, 44, color="#000000", stroke_width=8, stroke_color="#000000")
        line = MonoLine(font, 44)
        shadow.alpha = 0.78
        shadow.show(spec, (x, ROW_Y - 0.3))
        line.show(spec, (x, ROW_Y - 0.3))
        for slot in line.submobjects:
            slot.fill_rgbas[:, :3] = mix(hex_rgb(color), np.ones(3), 0.4)
        return [shadow, line], ("row counter", line)

    def finish(self, mobs, texts, slots):
        if os.environ.get("GUIDES"):
            mobs += guides()
        self.add(*mobs, PostFX())
        self.report(texts, slots)

    def report(self, texts, slots):
        """Print overlaps between texts, and between texts and the ring's card slots or the meter."""
        def box(m):
            (x0, y0, _), (x1, y1, _) = m.get_corner([-1, -1, 0]), m.get_corner([1, 1, 0])
            return x0, y0, x1, y1

        def hit(a, b, pad=0.03):
            return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad

        found = []
        boxes = [(n, box(m)) for n, m in texts if len(m.get_family()) > 0 and m.width > 0]
        for i, (na, a) in enumerate(boxes):
            for nb, b in boxes[i + 1:]:
                if hit(a, b):
                    found.append(f"{na} / {nb}")
            if any(hit(a, box(s)) for s in slots):
                found.append(f"{na} / ring")
            if na != "meter start" and hit(a, (MX - 0.1, MY0, MX + 0.1, MY1)):
                found.append(f"{na} / meter")
        name = type(self).__name__
        print(f"{name} (card {self.CARD}, +{self.AT:.2f} s): " + ("no overlaps" if not found else "OVERLAPS: " + ", ".join(found)))
        for n, b in boxes:
            print(f"  {n:15s} x {b[0]:+.2f} … {b[2]:+.2f}   y {b[1]:+.2f} … {b[3]:+.2f}")


class PayoffFrame(StopScene):
    """Card 11, the answer, late in its 5.4 s hold."""

    AT, CARD, ACCENT = 3.6, 11, TIME

    def construct(self):
        at = self.AT
        fill_ref = ease_in_out(clamp01((at - 0.5) / 0.7))  # the reference fills first...
        fill_count = ease_in_out(clamp01((at - 1.2) / 0.8))  # ...then the count overtakes it (1.5 s in all)
        noted = at >= 2.5

        bg = Rectangle(width=9.4, height=16.4, stroke_width=0, fill_color="#09090F", fill_opacity=1)
        nebula = sprite("glow", 14.6, TIME, alpha=0.12, at=(0.0, Y0))
        glows, slots = deck_ring(self.CARD)
        mobs = [bg, nebula, *idle_warp(self.CARD, at, TIME), *orders_cloud(), *big_bang(), *glows, *slots]
        hud, texts = self.hud()
        mobs += hud
        if at <= 1.2:  # answer ring
            hh = mix(1.4, 5.0, ease_out(at / 1.2))
            mobs.append(sprite("ring", hh, TIME, alpha=0.9 * (1 - at / 1.2) ** 1.5, at=(0.0, 5.6),
                               r0=0.8, inner=0.02, outer=0.06))

        # pass row: the reference fills in its accent, then the orders overtake it in lavender
        x_ref = ROW_X0 + (ROW_X1 - ROW_X0) * SECONDS / orders(self.CARD)
        mobs += self.track()
        for x0, x1, c in [(ROW_X0, mix(ROW_X0, x_ref, fill_ref), TIME), (x_ref, mix(x_ref, ROW_X1, fill_count), LAVENDER)]:
            if x1 - x0 > 0.005:
                mobs += self.glow_line(x0, x1, c)
        if fill_ref >= 1:
            mobs.append(Line([x_ref, ROW_Y - 0.13, 0], [x_ref, ROW_Y + 0.13, 0], stroke_width=2.2, color=TIME))
        if at >= 0.5:
            cnt, t = self.row_counter(SECONDS * fill_ref, "seconds", (ROW_X0 + mix(ROW_X0, x_ref, fill_ref)) / 2, TIME)
            mobs += cnt
            texts.append(t)

        meter, mt = self.meter()
        mobs += meter
        texts += mt
        if noted:  # payoff brackets: 11 cards to the answer (cyan), 41 still in the deck (Mist Light)
            y = meter_y(self.CARD)
            bx = MX + 0.74
            br_a = VGroup(Line([bx, MY0, 0], [bx, y - 0.04, 0], stroke_width=2.2, color=CYAN),
                          Line([bx - 0.07, y - 0.04, 0], [bx, y - 0.04, 0], stroke_width=2.2, color=CYAN))
            br_b = VGroup(Line([bx, y + 0.04, 0], [bx, MY1, 0], stroke_width=1.8, color=MUTED),
                          Line([bx - 0.07, MY1, 0], [bx, MY1, 0], stroke_width=1.8, color=MUTED))
            la = text("11", 20, font=MONO, weight=BOLD, color=CYAN).move_to([bx + 0.3, (MY0 + y) / 2, 0])
            lb = text("41", 20, font=MONO, weight=BOLD, color=BODY).move_to([bx + 0.3, (y + MY1) / 2, 0])
            mobs += [br_a, br_b, la, lb]
            texts += [("bracket 11", la), ("bracket 41", lb), ("bracket line 11", br_a), ("bracket line 41", br_b)]

        head, title, line = label_block("4.35 × 10¹⁷ seconds", "SINCE THE BIG BANG",
                                        "Not even [a quarter] of the deck." if noted else "More orders than [every second]",
                                        TIME, chip="CARD 11")
        mobs += [head, title, line]
        texts += [("chip + size", head), ("title", title), ("note" if noted else "line", line)]
        self.finish(mobs, texts, slots)


class PayoffArrival(PayoffFrame):
    AT = 0.35


class PayoffMidRow(PayoffFrame):
    AT = 0.85


class PayoffRowEnd(PayoffFrame):
    AT = 2.0


class FinalFrame(StopScene):
    """Card 52, the whole deck: the ring closes, then the note sets up the "imagine" beats, which
    count shuffles since the Big Bang."""

    AT, CARD, ACCENT = 2.0, 52, DECK

    def backdrop(self, drawing):
        bg = Rectangle(width=9.4, height=16.4, stroke_width=0, fill_color="#0A0514", fill_opacity=1)
        glows, slots = deck_ring(self.CARD, DECK)
        mobs = [bg, sprite("glow", 14.6, DECK, alpha=0.2, at=(0.0, Y0)), *idle_warp(self.CARD, self.AT, DECK),
                sprite("glow", 4.6, DECK, alpha=0.3, at=(0.0, Y0)),
                *orders_cloud(count=900, radius=2.3, seed=6, colors=(LAVENDER, DECK), brightness=1.6),
                *drawing, *glows, *slots]
        return mobs, slots

    def construct(self):
        mobs, slots = self.backdrop([])
        hud, texts = self.hud()
        meter, mt = self.meter()
        mobs += hud + self.track() + meter
        texts += mt
        noted = self.AT >= 1.2
        head, title, line = label_block("8.07 × 10⁶⁷ orders", "THE WHOLE DECK",
                                        "Imagine [shuffling] since the Big Bang." if noted else "52 × 51 × 50 × … × 2 × 1",
                                        DECK, chip="CARD 52")
        mobs += [head, title, line]
        texts += [("chip + size", head), ("title", title), ("note" if noted else "line", line)]
        self.finish(mobs, texts, slots)


class FinalArrival(FinalFrame):
    AT = 0.35


class ImagineSecond(FinalFrame):
    """The first "imagine" beat, 2.0 s in: one shuffle a second since the Big Bang still sees at most one
    order in 1.85 × 10⁵⁰.  Each beat is a phase label with a copies row: copies of all those shuffles
    lining up across the orders."""

    AT = 2.0
    SHUFFLES, ACC = SECONDS, TIME
    COUNT = 1.85e50  # 52 factorial / SHUFFLES, as the line rounds it: the row must end on the line's number
    SIZE, TITLE = "4.35 × 10¹⁷ shuffles", "A SHUFFLE A SECOND"
    LINE = "At most [1 in 1.85 × 10⁵⁰] of the orders"

    def drawing(self):
        return big_bang()

    def construct(self):
        mobs, slots = self.backdrop(self.drawing())
        hud, texts = self.hud()
        meter, mt = self.meter()
        mobs += hud + self.track() + meter
        texts += mt
        row, t = self.copies_row(ease_in_out(clamp01((self.AT - 0.5) / 1.5)), self.COUNT, "per shuffle", self.ACC)
        mobs += row
        texts.append(t)
        head, title, line = label_block(self.SIZE, self.TITLE, self.LINE, self.ACC)
        mobs += [head, title, line]
        texts += [("size text", head), ("title", title), ("line", line)]
        self.finish(mobs, texts, slots)
        exact = orders(CARDS) / self.SHUFFLES
        print(f"  row ends on {self.COUNT:.3g} per shuffle (exact {exact:.4g}, {abs(self.COUNT / exact - 1):.2%} off)")


class ImagineEveryone(ImagineSecond):
    SHUFFLES, ACC, COUNT = PEOPLE * SECONDS, CYAN, 2.23e40
    SIZE, TITLE = "3.61 × 10²⁷ shuffles", "EVERYONE, EVERY SECOND"
    LINE = "At most [1 in 2.23 × 10⁴⁰] of the orders"

    def drawing(self):
        return crowd()


class ImaginePlanck(ImagineSecond):
    SHUFFLES, ACC, COUNT = SECONDS / PLANCK_TIME, LAVENDER, 1e7
    SIZE, TITLE = "8.07 × 10⁶⁰ shuffles", "EVERY PLANCK TIME"
    LINE = "Still at most [1 in 10 million]"

    def drawing(self):
        return planck_dot()

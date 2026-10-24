"""THE SHUFFLE: test stills for the treatment's open layout questions.

Poses single frames at full resolution with The Doubling's engine, so the
layout, the new accents and the ring can be checked before the film is
built (README.md).  Nothing here animates.

    manim -r 1080,1920 -s --disable_caching stills.py HookFrame      # the cover
    manim -r 1080,1920 -s --disable_caching stills.py PayoffFrame    # card 20, late in the hold
    manim -r 1080,1920 -s --disable_caching stills.py FinalFrame     # card 52, late in the hold
    GUIDES=1 manim ... stills.py PayoffFrame                         # safe zones drawn over it

The *Arrival, *MidRow and *RowEnd scenes pose the other moments that
film-style.md §8 asks to check.  The scenes print what they measure: the
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
from manim import BOLD, DOWN, MEDIUM, RIGHT, UP, Circle, DashedLine, Line, Rectangle, RoundedRectangle, Scene, Square, VGroup, VMobject, config

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
STOPS = [(14, CYAN), (20, TIME), (27, CYAN), (42, "#4DA3FF"), (46, "#FFB13B"), (48, LAVENDER), (52, DECK)]
RING_R = 2.55  # the ring of 52 card slots, centred on the featured object
MX, MY0, MY1 = -3.95, -2.45, 3.45  # progress meter (film-style.md §4)
ROW_Y, ROW_X0, ROW_X1 = 3.75, -2.6, 2.6  # pass row, where The Doubling's ruler sat

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
    """Text with [highlighted] spans, as in The Doubling; ``\\n`` makes extra centred lines."""
    rows = []
    for row in s.split("\n"):
        hl = re.findall(r"\[(.+?)\]", row)
        rows.append(text(row.replace("[", "").replace("]", ""), size, t2c={h: accent for h in hl}, **kw))
    return rows[0] if len(rows) == 1 else VGroup(*rows).arrange(DOWN, buff=0.1)


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


# ---------------------------------------------------------------------------
# Stops
# ---------------------------------------------------------------------------


def log2_orders(cards):
    """log2 of the orders of a deck of ``cards`` cards (cards factorial)."""
    return math.lgamma(cards + 1) / math.log(2)


def meter_y(cards):
    return MY0 + (MY1 - MY0) * log2_orders(cards) / log2_orders(CARDS)


def count_text(value):
    """The live counter's words for big totals (film-style.md §4), as in The Doubling."""
    for div, name in ((1e18, "quintillion"), (1e15, "quadrillion"), (1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if value >= div:
            return f"{value / div:,.0f} {name}"
    return f"{int(round(value)):,}"


def count_spec(value, digit_h, unit):
    """The count readout: exact with commas below 10⁷, then three significant figures with a
    raised exponent, so it matches a size text that shows the same count ("8.07 × 10⁶⁷")."""
    if value < 1e7:
        return [(c, 1.0, 0.0) for c in f"{int(value):,} {unit}"]
    e = int(math.floor(math.log10(value)))
    mant = value / 10**e
    if round(mant, 2) >= 10:
        mant, e = mant / 10, e + 1
    spec = [(c, 1.0, 0.0) for c in f"{mant:.2f}"]
    spec += [(" ", 0.45, 0), ("×", 1.0, 0.0), (" ", 0.45, 0), ("1", 1.0, 0.0), ("0", 1.0, 0.0)]
    spec += [(c, 0.6, digit_h * 0.55) for c in str(e)]
    return spec + [(" ", 0.6, 0)] + [(c, 1.0, 0.0) for c in unit]


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


def chip_block(step, size_text, title, line, accent):
    """A stop's label block with its chip (film-style.md §4, §5)."""
    sup = re.search("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]", size_text)
    a = text(size_text, 27 if sup else 25, font=INTER if sup else MONO, weight=MEDIUM, color=BODY)
    chip_txt = text(step, 22, font=MONO, weight=BOLD, color=accent)
    chip = RoundedRectangle(width=chip_txt.width + 0.38, height=0.44, corner_radius=0.22, fill_color=accent,
                            fill_opacity=0.16, stroke_color=accent, stroke_width=1.6)
    chip_txt.move_to(chip)
    head = VGroup(VGroup(chip, chip_txt), a).arrange(RIGHT, buff=0.26).move_to([0, -2.86, 0])
    ti = fit_width(text(title, 60, font=INTER_DISPLAY, weight=BOLD, color=WHITE_TXT), SAFE_W).move_to([0, -3.51, 0])
    two = "\n" in line
    c = fit_width(markup(line, 29 if two else 31, accent, color=BODY, weight=MEDIUM), SAFE_W)
    # a two-line note grows downward; nudged as in The Doubling so its last baseline stays at or above -4.8
    c.move_to([0, -4.21 - (c.height - 0.32) / 2 + (0.05 if two else 0.0), 0])
    return head, ti, c


class StopScene(Scene):
    """A stop ``AT`` seconds after it arrives, with the shared HUD, meter, row track and overlap report."""

    AT, CARD, ACCENT = 0.0, 20, TIME

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
        label = text("CARDS", 19, weight=MEDIUM, color=MUTED, tracking=0.09).move_to([0, 6.55, 0])
        small = MonoFont(size=30, weight=MEDIUM, letters=True)
        readout = MonoLine(small, 22, color="#9FB3D9")
        readout.show(count_spec(float(math.factorial(self.CARD)), small.digit_h, "orders"), (0, 4.5))
        mobs = [label, readout]
        if flash > 0.01:
            mobs.append(sprite("glow", 3.2, self.ACCENT, alpha=0.35 * flash, at=(0.0, 5.6)))
        mobs.append(counter)
        return mobs, [("hud label", label), ("counter", counter), ("readout", readout)]

    def meter(self):
        """log2 of the orders, from one card to the full deck."""
        y = meter_y(self.CARD)
        fill = Line([MX, MY0, 0], [MX, y, 0], stroke_width=3.2)
        fill.set_stroke([LAVENDER, DECK]).set_sheen_direction(UP)
        mobs = [Line([MX, MY0, 0], [MX, MY1, 0], stroke_width=2, color="#27304A"), fill]
        for card_n, acc in STOPS:
            tk = Line([MX - 0.08, meter_y(card_n), 0], [MX + 0.08, meter_y(card_n), 0], stroke_width=2.2, color=acc)
            mobs.append(tk.set_stroke(opacity=1.0 if card_n <= self.CARD else 0.3))
        start = text("1", 17, font=MONO, color=MUTED).move_to([MX, MY0 - 0.26, 0])
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

    @staticmethod
    def row_counter(label, x_mid, color):
        """The live counter, 0.3 below the track; it starts no further left than the track, which
        keeps it clear of the meter and the payoff brackets."""
        font = MonoFont(size=21, weight=BOLD, letters=True)
        w = len(label) * font.adv
        x = min(max(x_mid, ROW_X0 + w / 2), 3.3 - w / 2)
        shadow = MonoLine(font, 40, color="#000000", stroke_width=8, stroke_color="#000000")
        line = MonoLine(font, 40)
        chars = [(ch, 1.0, 0.0) for ch in label]
        shadow.alpha = 0.78
        shadow.show(chars, (x, ROW_Y - 0.3))
        line.show(chars, (x, ROW_Y - 0.3))
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
    """Card 20, the answer, late in its 5.4 s hold."""

    AT, CARD, ACCENT = 3.6, 20, TIME
    REF, REF_NOUN = 13.787e9 * 365.25 * 86400, "seconds"  # seconds since the Big Bang

    def construct(self):
        at = self.AT
        orders = float(math.factorial(self.CARD))
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
        x_ref = ROW_X0 + (ROW_X1 - ROW_X0) * self.REF / orders
        mobs += self.track()
        for x0, x1, c in [(ROW_X0, mix(ROW_X0, x_ref, fill_ref), TIME), (x_ref, mix(x_ref, ROW_X1, fill_count), LAVENDER)]:
            if x1 - x0 > 0.005:
                mobs += self.glow_line(x0, x1, c)
        if fill_ref >= 1:
            mobs.append(Line([x_ref, ROW_Y - 0.13, 0], [x_ref, ROW_Y + 0.13, 0], stroke_width=2.2, color=TIME))
        if at >= 0.5:
            cnt, t = self.row_counter(f"{count_text(self.REF * fill_ref)} {self.REF_NOUN}",
                                      (ROW_X0 + mix(ROW_X0, x_ref, fill_ref)) / 2, TIME)
            mobs += cnt
            texts.append(t)

        meter, mt = self.meter()
        mobs += meter
        texts += mt
        if noted:  # payoff brackets: 20 cards to the answer (cyan), 32 still to add (Mist Light)
            y20, bx = meter_y(self.CARD), MX + 0.74
            br_a = VGroup(Line([bx, MY0, 0], [bx, y20 - 0.04, 0], stroke_width=2.2, color=CYAN),
                          Line([bx - 0.07, y20 - 0.04, 0], [bx, y20 - 0.04, 0], stroke_width=2.2, color=CYAN))
            br_b = VGroup(Line([bx, y20 + 0.04, 0], [bx, MY1, 0], stroke_width=1.8, color=MUTED),
                          Line([bx - 0.07, MY1, 0], [bx, MY1, 0], stroke_width=1.8, color=MUTED))
            la = text("20", 20, font=MONO, weight=BOLD, color=CYAN).move_to([bx + 0.3, (MY0 + y20) / 2, 0])
            lb = text("32", 20, font=MONO, weight=BOLD, color=BODY).move_to([bx + 0.3, (y20 + MY1) / 2, 0])
            mobs += [br_a, br_b, la, lb]
            texts += [("bracket 20", la), ("bracket 32", lb), ("bracket line 20", br_a), ("bracket line 32", br_b)]

        head, title, line = chip_block("CARD 20", "4.35 × 10¹⁷ seconds", "SINCE THE BIG BANG",
                                       "Not even [half] a deck." if noted else "More orders than [every second]", TIME)
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
    """Card 52, the whole deck, late in its 4.6 s hold: the ring closes, and the note's copies row
    lines up 10 million copies of every Planck time since the Big Bang across the orders."""

    AT, CARD, ACCENT = 4.0, 52, DECK
    COPIES = 1e7  # 52 factorial / 8.07e60 Planck times since the Big Bang = 9.99e6
    NOTE = "[10 million] orders for every\nPlanck time since the Big Bang."

    def construct(self):
        at = self.AT
        noted = at >= 1.3  # the note, and its row, enter 1.3 s into the hold
        progress = ease_in_out(clamp01((at - 1.3) / 1.5))

        bg = Rectangle(width=9.4, height=16.4, stroke_width=0, fill_color="#0A0514", fill_opacity=1)
        nebula = sprite("glow", 14.6, DECK, alpha=0.2, at=(0.0, Y0))
        glows, slots = deck_ring(self.CARD, DECK)
        mobs = [bg, nebula, *idle_warp(self.CARD, at, DECK), sprite("glow", 4.6, DECK, alpha=0.35, at=(0.0, Y0)),
                *orders_cloud(count=900, radius=2.3, seed=6, colors=(LAVENDER, DECK), brightness=1.6), *glows, *slots]
        hud, texts = self.hud()
        mobs += hud + self.track()

        # copies row: the Planck times since the Big Bang are too few to show on a pass row
        # (1e-7 of the track), so 10 million copies of them line up across the orders instead
        if noted and progress > 0:
            x1 = mix(ROW_X0, ROW_X1, progress)
            mobs += self.glow_line(ROW_X0, x1, LAVENDER)
            beads = np.linspace(ROW_X0, ROW_X1, 90)
            mobs += [Circle(radius=0.022, stroke_width=0, fill_color="#FFFFFF", fill_opacity=0.9).move_to([x, ROW_Y, 0])
                     for x in beads if x <= x1]
            mobs.append(sprite("star", 0.4, "#FFFFFF", spikes=0.35, at=(x1, ROW_Y)))
            cnt, t = self.row_counter(f"{count_text(self.COPIES * progress)} per Planck time", (ROW_X0 + x1) / 2, LAVENDER)
            mobs += cnt
            texts.append(t)

        meter, mt = self.meter()
        mobs += meter
        texts += mt
        head, title, line = chip_block("CARD 52", "8.07 × 10⁶⁷ orders", "THE WHOLE DECK",
                                       self.NOTE if noted else "1 × 2 × 3 × … × 52", DECK)
        mobs += [head, title, line]
        texts += [("chip + size", head), ("title", title), ("note" if noted else "line", line)]
        self.finish(mobs, texts, slots)
        if noted:
            last = self.NOTE.split("\n")[-1]
            print(f"  note's last baseline {baseline(line[-1], last):+.3f} (limit -4.8)")


class FinalArrival(FinalFrame):
    AT = 0.35


class FinalMidRow(FinalFrame):
    AT = 2.05


class FinalRowEnd(FinalFrame):
    AT = 2.8

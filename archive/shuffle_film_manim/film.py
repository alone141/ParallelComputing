"""THE SHUFFLE: how many cards dealt from a full deck have more orders than seconds since
the Big Bang?  A vertical (9:16) film for TikTok, Reels and Shorts.  See README.md.

    manim -r 1080,1920 --fps 60 --disable_caching film.py ShuffleFilm

The Doubling's engine and textures are imported as a package (``doubling_film.engine``,
``doubling_film.assets``).  Its story, visuals and soundtrack are never imported: they use
flat imports that would bind to this film's modules.
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from manim import BOLD, DOWN, MEDIUM, RIGHT, UP, Circle, Group, Line, Rectangle, RoundedRectangle, Scene, VGroup, VMobject, config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from doubling_film.engine import (  # noqa: E402
    INTER, INTER_DISPLAY, MONO, AnimText, FilmCamera, MonoFont, MonoLine, PostFX, Rig, clamp01, ease_in_out, ease_out,
    fit_width, hex_rgb, mix, sci_spec, smooth, text,
)
from story import (  # noqa: E402
    BEATS, BY_KEY, CARDS, CYAN, HOOK, HOOK_LINES, HOOK_OUT, HUD_IN, HUD_UNIT, LABEL_GAP, LAVENDER, LIME, LOG2_ALL, MAGENTA,
    MECHANIC_IN, MECHANIC_LABEL, NOTE_IN, OUTRO_LINES, ROW_DURATION, ROW_REF_PART, SPRINT_LABEL, START_LABEL, START_LABEL_T,
    STEP_TIMES, STOPS, SUB_IN, TIMELINE, VIOLET, log2_orders, orders, orders_f,
)
from visuals import (  # noqa: E402
    D0, DRAWINGS, RING_R, Y0, CardRing, Deck, Dust, RowLine, polyline_points, quad, sprite,
)

for _bad in ("doubling_film.story", "doubling_film.visuals", "doubling_film.soundtrack", "doubling_film.film"):
    assert _bad not in sys.modules, f"{_bad} must not be imported by The Shuffle"
assert Path(sys.modules["story"].__file__).resolve().parent == HERE, "story resolved to another film's module"

config.frame_width = 9.0
config.frame_height = 16.0
config.background_color = "#04050B"
config.video_encoder_options = {"crf": "15", "preset": "medium"}

WHITE_TXT = "#F5F7FF"
MUTED = "#8A93A8"
BODY = "#C9D1E6"
SAFE_W = 6.8  # centred text stays inside ±3.4 (the button column starts at +3.4)
ROW_Y, ROW_X0, ROW_X1 = 3.75, -D0 / 2, D0 / 2  # the pass row's track, where The Doubling's ruler sat
MX, MY0, MY1 = -3.95, -2.45, 3.45  # the progress meter
HOOK_SIZE, HOOK_PITCH, HOOK_Y, HOOK_STAGGER, HOOK_DUR_IN = 42, 1.04, 4.75, 0.0085, 0.32
FLAT = set("ABDEFHIKLMNPRTXZhiklmnrxz")  # glyphs whose bottom sits on the baseline


def em(size):
    """One em in scene units at a Manim font size (size 60 ≈ 100 px em at 1080 wide)."""
    return 100 / 60 * size / 120


def baseline(mob, s):
    chars = [c for c in s if not c.isspace()]
    assert len(chars) == len(mob.submobjects), s
    return min(g.get_bottom()[1] for c, g in zip(chars, mob.submobjects) if c in FLAT)


def baseline_lines(rows, size, pitch, y, t2c=None, **kw):
    """Centred lines spaced by baseline, so lines with and without descenders keep an even
    rhythm (stacking ink boxes packs a line without descenders closer to the next)."""
    g = VGroup()
    for i, s in enumerate(rows):
        r = text(s, size, t2c={k: v for k, v in (t2c or {}).items() if k in s}, **kw)
        r.shift([-r.get_center()[0], -i * pitch * em(size) - baseline(r, s), 0])
        g.add(r)
    return g.shift([0, y - g.get_center()[1], 0])


def markup(s, size, accent, **kw):
    """Text with [highlighted] spans; ``\\n`` makes extra centred lines."""
    rows = []
    for row in s.split("\n"):
        hl = re.findall(r"\[(.+?)\]", row)
        clean = row.replace("[", "").replace("]", "")
        rows.append(text(clean, size, t2c={h: accent for h in hl}, **kw))
    return VGroup(*rows).arrange(DOWN, buff=0.1)


def ramp(keys, t, width=0.6):
    """Piecewise value over time: ``keys`` is [(t_switch, value)], each switch crossfading
    over ``width`` seconds ending at t_switch."""
    v = keys[0][1]
    for ts, val in keys[1:]:
        v = mix(v, val, smooth(ts - width, ts, t))
    return v


def count_spec(value, digit_h, unit="orders"):
    """The count readout: exact with commas below 10⁷, then three significant figures with
    a raised exponent, so it agrees with a size text showing the same count."""
    if value < 1e7:
        return [(c, 1.0, 0.0) for c in f"{int(round(value)):,}"] + [(" ", 0.6, 0)] + [(c, 1.0, 0.0) for c in unit]
    return sci_spec(value, digit_h=digit_h, unit=unit, decimals=2)


def counter_spec(value, digit_h, noun, notation):
    """A live counter.  ``words``: commas, then unit words from ten million with one decimal
    below ten of a unit ("8.3 billion").  ``sci``: the readout's notation ("4.35 × 10¹⁷")."""
    if notation == "sci":
        if value < 1e7:
            head = [(c, 1.0, 0.0) for c in f"{int(round(value)):,}"]
        else:
            head = sci_spec(max(value, 1e-9), digit_h=digit_h, unit="", decimals=2)[:-1]
    else:
        words = f"{int(round(value)):,}"
        for div, name in ((1e18, "quintillion"), (1e15, "quadrillion"), (1e12, "trillion"), (1e9, "billion"),
                          (1e6, "million")):
            if value >= 1e7 and value >= div:
                q = value / div
                words = f"{q:.1f} {name}" if q < 10 else f"{q:,.0f} {name}"
                break
        head = [(c, 1.0, 0.0) for c in words]
    return head + [(" ", 0.6, 0)] + [(c, 1.0, 0.0) for c in noun]


def spec_width(spec, adv):
    return sum(s for _, s, _ in spec) * adv


class Words(AnimText):
    """AnimText whose exit waits for the last glyph's staggered fade, so no line is cut off mid-fade
    (the engine hides the whole text at t_out + dur_out, before late glyphs have finished)."""

    def update(self, t):
        tail = (len(self.glyphs) - 1) * self.stagger * 0.3
        if t < self.t_in or t > self.t_out + self.dur_out + tail or self.extra_alpha <= 0.002:
            self.visible = False
            return
        self.visible = True
        n = len(self.glyphs)
        for i, g in enumerate(self.glyphs):
            p_in = ease_out((t - self.t_in - i * self.stagger) / self.dur_in)
            p_out = ease_in_out((t - self.t_out - (n - 1 - i) * self.stagger * 0.3) / self.dur_out)
            a = p_in * (1 - p_out) * self.extra_alpha
            dy = -(1 - p_in) * self.rise + p_out * self.lift
            pts = g._rest
            if self.scale_in != 1.0:
                s = mix(self.scale_in, 1.0, p_in)
                pts = (pts - self.center) * s + self.center
            g.points = pts + np.array([0.0, dy, 0.0]) + self.offset
            g.fill_rgbas[:, 3] = g._fa * a
            g.stroke_rgbas[:, 3] = g._sa * a


class ShuffleFilm(Scene):
    def __init__(self, **kwargs):
        super().__init__(camera_class=FilmCamera, **kwargs)

    # ------------------------------------------------------------------ build
    def get_moving_and_static_mobjects(self, animations):
        if hasattr(self, "stage"):
            return [self.stage], []
        return super().get_moving_and_static_mobjects(animations)

    def construct(self):
        self.setup_film()
        start, end = 0.0, self.tl.total
        if os.environ.get("FILM_RANGE"):
            start, end = (float(x) for x in os.environ["FILM_RANGE"].split(","))
        self.t = start
        self.tick(self.stage, 0.0)
        self.wait(end - start - 0.25 / config.frame_rate, frozen_frame=False)

    def setup_film(self):
        self.tl = TIMELINE
        tl = self.tl
        self.o = tl.outro_start
        self.deck = Deck(DECK_H := 2.9, LAVENDER)
        self.ring = CardRing(self.deck)
        self.dust = Dust()
        self.drawings = {k: cls(acc, seed=i + 2) for i, (k, cls, acc) in enumerate(
            (("crowd", DRAWINGS["crowd"], CYAN), ("spiral", DRAWINGS["spiral"], LIME), ("planck", DRAWINGS["planck"], LAVENDER)))}
        self.windows = {k: [] for k in self.drawings}
        for s in STOPS:
            if s.drawing:
                self.windows[s.drawing].append((tl.arrive[s.key] - 0.45, tl.leave[s.key] + 0.05))
        for b in BEATS:
            t_out = tl.beat_leave[b.key] + 0.05 if b is not BEATS[-1] else self.o + 0.05
            self.windows[b.drawing].append((tl.beat_arrive[b.key] - 0.45, t_out))
        self.whole_glow = sprite("glow", 4.6, VIOLET, alpha=0.0, at=(0.0, Y0))
        self.build_background()
        self.build_particles()
        self.build_rings()
        self.build_pin()
        self.build_hud()
        self.build_rows()
        self.build_labels()
        self.build_hook_and_outro()
        self.fx = PostFX()
        self.stage = Group()
        self.t = 0.0
        self.stage.add_updater(self.tick)
        self.add(self.stage)
        self.tick(self.stage, 0.0)

    def build_background(self):
        tl = self.tl
        self.bg = Rectangle(width=9.4, height=16.4, stroke_width=0, fill_color="#08061A", fill_opacity=1)
        self.nebula = sprite("glow", D0 * 2.7, "#FFFFFF", alpha=0.0)
        a6, a11, a52 = tl.arrive["everyone"], tl.arrive["bigbang"], tl.arrive["deck"]
        b1, b2, b3 = (tl.beat_arrive[b.key] for b in BEATS)
        H = hex_rgb
        # every channel stays at or below 0x1A (film-style §3)
        self.bg_keys = [(0.0, H("#08061A")), (a6 - 0.2, H("#050C13")), (a11 - 0.2, H("#0A0D08")),
                        (tl.sprint_start + 0.4, H("#09071A")), (a52 - 0.1, H("#0B0414")), (b1, H("#0A0D08")),
                        (b2, H("#050C13")), (b3, H("#08061A")), (self.o + 0.8, H("#0B0414"))]
        self.accent_keys = [(0.0, H(LAVENDER)), (a6 - 0.3, H(CYAN)), (a11 - 0.3, H(LIME)),
                            (tl.sprint_start + 0.3, H(LAVENDER)), (a52 - 0.3, H(VIOLET)), (b1 - 0.3, H(LIME)),
                            (b2 - 0.3, H(CYAN)), (b3 - 0.3, H(LAVENDER)), (self.o + 0.7, H(VIOLET))]
        self.nebula_keys = [(0.0, 1.0), (tl.sprint_start + 0.3, 0.6), (a52 - 0.3, 1.0)]

    def build_particles(self, count=150):
        rng = np.random.default_rng(1)
        self.p_theta = rng.uniform(0, 2 * np.pi, count)
        self.p_phase = rng.uniform(0, 1, count)
        self.p_bright = rng.uniform(0.3, 1.0, count)
        self.particles = [VMobject(stroke_color="#FFFFFF", stroke_width=w, stroke_opacity=0, fill_opacity=0)
                          for w in rng.uniform(1.3, 2.8, count)]
        for p in self.particles:
            p.points = polyline_points([[0, 0], [0.01, 0]])

    def build_rings(self):
        self.rings = [Circle(radius=1, stroke_width=1.5, stroke_color="#FFFFFF", stroke_opacity=0) for _ in range(9)]
        for r in self.rings:
            r._unit = r.points.copy()

    def build_pin(self):
        cap = RoundedRectangle(width=0.9, height=0.4, corner_radius=0.2, fill_color=CYAN, fill_opacity=1,
                               stroke_width=0).move_to([0, 0.6, 0])
        lab = text("YOU", 21, font=INTER_DISPLAY, weight=BOLD, color="#03121A", tracking=0.035).move_to([0, 0.6, 0])
        stem = Line([0, 0.4, 0], [0, 0.1, 0], stroke_width=2.6, color=CYAN)
        ring = Circle(radius=0.075, stroke_color=CYAN, stroke_width=2.6, fill_color=CYAN, fill_opacity=0.25)
        self.pin_glow = sprite("glow", 0.9, CYAN, alpha=0.75)
        self.pin = Group(self.pin_glow, stem, ring, cap, lab)
        self.pin_rig = Rig(self.pin)

    def meter_y(self, n):
        return MY0 + (MY1 - MY0) * clamp01(n / CARDS)

    def build_hud(self):
        tl = self.tl
        self.font_big = MonoFont(size=104, weight=BOLD)
        self.font_small = MonoFont(size=30, weight=MEDIUM, letters=True)
        self.counter = MonoLine(self.font_big, 4, color="#FFFFFF")
        self.counter_glow = sprite("glow", 3.2, "#FFFFFF", alpha=0.0)
        self.answer_ring = sprite("ring", 2.0, LIME, alpha=0.0, r0=0.8, inner=0.02, outer=0.06)
        self.readout = MonoLine(self.font_small, 26, color="#9FB3D9")
        self.hud_label = text(HUD_UNIT, 19, weight=MEDIUM, color=MUTED, tracking=0.09).move_to([0, 6.55, 0])
        self.track = VGroup(
            Line([ROW_X0, ROW_Y, 0], [ROW_X1, ROW_Y, 0], stroke_width=2, color="#27304A"),
            Line([ROW_X0, ROW_Y - 0.1, 0], [ROW_X0, ROW_Y + 0.1, 0], stroke_width=1.6, color="#5D6B8C"),
            Line([ROW_X1, ROW_Y - 0.1, 0], [ROW_X1, ROW_Y + 0.1, 0], stroke_width=1.6, color="#5D6B8C"),
        )
        # multiplier pops while the mechanic is taught
        self.pops = []
        for k in range(1, len(STEP_TIMES) + 1):
            m = text(f"×{CARDS - k + 1}", 40, font=INTER_DISPLAY, weight=BOLD, color=LAVENDER).move_to([1.55, 5.62, 0])
            t_next = tl.card_time(k + 1) if k < len(STEP_TIMES) else 1e9
            self.pops.append((AnimText(m, 0), tl.card_time(k), t_next))
        self.pop_alpha = [0.0] * len(self.pops)
        # progress meter: linear in cards dealt
        my = self.meter_y
        self.meter_base = Line([MX, MY0, 0], [MX, MY1, 0], stroke_width=2, color="#27304A")
        self.meter_fill = Line([MX, MY0, 0], [MX, MY0 + 0.01, 0], stroke_width=3.2)
        self.meter_fill.set_stroke([LAVENDER, CYAN, LIME, VIOLET]).set_sheen_direction(UP)
        self.meter_dot = sprite("star", 0.55, "#FFFFFF", spikes=0.0)
        self.meter_ticks = []
        for s in STOPS:
            tk = Line([MX - 0.08, my(s.card), 0], [MX + 0.08, my(s.card), 0], stroke_width=2.2, color=s.accent)
            tk._n = s.card
            self.meter_ticks.append(tk)
        self.meter_zero = text("0", 17, font=MONO, color=MUTED).move_to([MX, MY0 - 0.26, 0])
        you = BY_KEY["everyone"]
        self.you_tag = Words(text("YOU", 16, font=INTER_DISPLAY, weight=BOLD, color=CYAN)
                             .move_to([MX + 0.36, my(you.card), 0]), tl.arrive["everyone"] + 0.4)
        ans = BY_KEY["bigbang"]
        t_ans = tl.note_time["bigbang"] + NOTE_IN  # with the punchline
        t_end = tl.arrive["deck"] + 0.6
        bx = MX + 0.74
        ya = my(ans.card)
        brA = VGroup(Line([bx, MY0, 0], [bx, ya - 0.04, 0], stroke_width=2.2, color=LIME),
                     Line([bx - 0.07, MY0, 0], [bx, MY0, 0], stroke_width=2.2, color=LIME),
                     Line([bx - 0.07, ya - 0.04, 0], [bx, ya - 0.04, 0], stroke_width=2.2, color=LIME))
        brB = VGroup(Line([bx, ya + 0.04, 0], [bx, MY1, 0], stroke_width=1.8, color=MUTED),
                     Line([bx - 0.07, MY1, 0], [bx, MY1, 0], stroke_width=1.8, color=MUTED))
        lA = text(str(ans.card), 20, font=MONO, weight=BOLD, color=LIME).move_to([bx + 0.3, (MY0 + ya) / 2, 0])
        lB = text(str(CARDS - ans.card), 20, font=MONO, weight=BOLD, color=BODY).move_to([bx + 0.3, (ya + MY1) / 2, 0])
        self.brackets = [Words(VGroup(brA, lA), t_ans, t_end, stagger=0.0, rise=0.0, lift=0.0),
                         Words(VGroup(brB, lB), t_ans + 0.35, t_end, stagger=0.0, rise=0.0, lift=0.0)]
        self.hud_static = Rig(VGroup(self.hud_label, self.track, self.meter_base, self.meter_zero, *self.meter_ticks))
        self.font_count = MonoFont(size=21, weight=BOLD, letters=True)
        self.count_shadow = MonoLine(self.font_count, 44, color="#000000", stroke_width=8, stroke_color="#000000")
        self.count_line = MonoLine(self.font_count, 44, color="#FFFFFF")

    def build_rows(self):
        self.row_ref = RowLine(LAVENDER)
        self.row_orders = RowLine(LAVENDER)
        self.row_mark = Line([0, ROW_Y - 0.13, 0], [0, ROW_Y + 0.13, 0], stroke_width=2.2, color=LAVENDER)
        self.row_star = sprite("star", 0.5, LAVENDER, alpha=0.0, spikes=0.35)

    def build_labels(self):
        tl = self.tl
        self.labels = []
        ya, yt, yc = -2.86, -3.51, -4.21

        def block(size_text, title, compare, accent, t_in, t_out, note=None, t_note=None, chip=None):
            mono = not re.search("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]", size_text)
            a = text(size_text, 25 if mono else 27, font=MONO if mono else INTER, weight=MEDIUM,
                     color=accent if chip is None else BODY)
            if chip is not None:
                chip_txt = text(chip, 22, font=MONO, weight=BOLD, color=accent)
                pill = RoundedRectangle(width=chip_txt.width + 0.38, height=0.44, corner_radius=0.22,
                                        fill_color=accent, fill_opacity=0.16, stroke_color=accent, stroke_width=1.6)
                chip_txt.move_to(pill)
                a = VGroup(VGroup(pill, chip_txt), a).arrange(RIGHT, buff=0.26)
            a.move_to([0, ya, 0])
            ti = fit_width(text(title, 60, font=INTER_DISPLAY, weight=BOLD, color=WHITE_TXT), SAFE_W).move_to([0, yt, 0])
            # the chip row rises a little less than the title, so a descender never dips into the title's cap line
            self.labels += [Words(a, t_in, t_out, rise=0.16), Words(ti, t_in + 0.06, t_out, stagger=0.018)]
            c_out = t_note if note else t_out
            c_in = t_in + (0.16 if chip is not None else 0.06)
            for s, ti_, to_ in [(compare, c_in, c_out)] + ([(note, t_note + NOTE_IN, t_out)] if note else []):
                two = "\n" in s
                size = 29 if two else 31
                c = markup(s, size, accent, color=BODY, weight=MEDIUM)
                effective = size * min(1.0, SAFE_W / c.width)
                assert effective >= 26 - 1e-6, f"line fits below size 26 ({effective:.1f}): {s!r}; shorten it"
                fit_width(c, SAFE_W)
                # a two-line note grows downward; nudged up so its last baseline stays at or above y = -4.8
                c.move_to([0, yc - (c.height - 0.32) / 2 + (0.12 if two else 0.0), 0])
                self.labels.append(Words(c, ti_, to_, stagger=0.008, rise=0.08 if two else 0.22))

        blocks = []  # (size_text, title, compare, accent, t_in, t_out, note, t_note, chip), in order
        blocks.append((*START_LABEL, LAVENDER, START_LABEL_T[0], START_LABEL_T[1], None, None, None))
        blocks.append((*MECHANIC_LABEL, LAVENDER, MECHANIC_IN, tl.arrive["everyone"] - 0.3, None, None, None))
        for s in STOPS:
            t_in = tl.arrive[s.key] - 0.3
            t_out = tl.leave[s.key] - 0.02 if s.key != "deck" else tl.beat_arrive[BEATS[0].key] - 0.3
            blocks.append((s.size_text, s.title, s.compare, s.accent, t_in, t_out, s.note, tl.note_time.get(s.key),
                           f"CARD {s.card}"))
            if s.key == "bigbang":
                blocks.append((*SPRINT_LABEL, LAVENDER, tl.sprint_start + 0.1, tl.arrive["deck"] - 0.3, None, None, None))
        for i, b in enumerate(BEATS):
            t_in = tl.beat_arrive[b.key] - 0.15  # a beat is a phase label: it may enter closer to its arrival
            t_out = tl.beat_leave[b.key] - 0.02 if i < len(BEATS) - 1 else self.o + 0.05
            blocks.append((b.size_text, b.title, b.compare, b.accent, t_in, t_out, None, None, None))
        # every block has left (its 0.28 s fade included) before the next one enters
        for i in range(len(blocks) - 1):
            b = blocks[i]
            blocks[i] = (*b[:5], min(b[5], blocks[i + 1][4] - LABEL_GAP), *b[6:])
        self.label_windows = [(b[4], b[5]) for b in blocks]
        for b in blocks:
            block(*b[:8], chip=b[8])

    def build_hook_and_outro(self):
        tl = self.tl
        q = baseline_lines(HOOK_LINES, HOOK_SIZE, HOOK_PITCH, HOOK_Y, font=INTER_DISPLAY, weight=BOLD, color=WHITE_TXT,
                           t2c={"dealt": LAVENDER, "Big Bang?": CYAN})
        sub = text("Guess before it ends.", 27, color=MUTED).move_to([0, 3.3, 0])
        self.hook_q = q
        self.hook = [Words(q, 0.0, HOOK_OUT, stagger=HOOK_STAGGER, dur_in=HOOK_DUR_IN), Words(sub, SUB_IN, HOOK_OUT)]
        o = self.o
        l1 = text(HUD_UNIT, 26, weight=MEDIUM, color=BODY, tracking=0.13).move_to([0, -0.78, 0])
        l2 = VGroup(*[text(r, 34, color=WHITE_TXT, weight=MEDIUM) for r in OUTRO_LINES]).arrange(DOWN, buff=0.14)
        l2.move_to([0, -1.95, 0])
        end = tl.total - 1.3
        self.outro = [Words(l1, o + 0.6, end), Words(l2, o + 0.75, end, stagger=0.01)]

    # ------------------------------------------------------------------ frame
    def gather_of(self, key, t):
        g = 0.0
        for t_in, t_out in self.windows[key]:
            g = max(g, smooth(t_in, t_in + 0.7, t) * (1 - smooth(t_out - 0.35, t_out, t)))
        return g

    def tick(self, mob, dt):
        self.t += dt
        t = self.t
        tl = self.tl
        o = self.o
        n = float(tl.n_at(t))
        v = abs(float(tl.speed_at(t)))
        loop = tl.loop(t)
        subs = [self.bg]
        centre = (0.0, Y0)
        accent = ramp(self.accent_keys, t)
        accent = mix(accent, hex_rgb(LAVENDER), loop)

        # background, nebula ------------------------------------------------
        self.bg.fill_rgbas[:, :3] = mix(ramp(self.bg_keys, t), self.bg_keys[0][1], loop)
        strength = ramp(self.nebula_keys, t) * (1 - 0.6 * smooth(o, o + 1.0, t))
        self.nebula.points = quad(0.0, Y0, D0 * 2.7)
        self.nebula.tint = accent
        self.nebula.alpha = mix(0.2 * strength, 0.2, loop)
        subs.append(self.nebula)

        # particles: the warp follows the rate of dealing -----------------------
        speed = clamp01(v / 18)
        fade_all = 1 - smooth(tl.total - 2.0, tl.total - 1.3, t) + smooth(tl.total - 0.9, tl.total - 0.05, t)
        P = 6.0
        phase = ((0.5 * n + 0.06 * t) / P) * (1 - loop)
        u = (phase + self.p_phase) % 1.0
        r_head = 10.5 * 2.0 ** (-P * u)
        r_tail = r_head * 2.0 ** (min(v * 0.03, 1.3) + 0.05)
        alpha = np.sin(np.pi * u) ** 1.3 * self.p_bright * (0.36 + 0.22 * speed) * fade_all
        cos, sin = np.cos(self.p_theta), np.sin(self.p_theta)
        pcol = mix(accent, np.ones(3), 0.55)
        for i, p in enumerate(self.particles):
            h = (cos[i] * r_head[i], Y0 + sin[i] * r_head[i])
            tt = (cos[i] * r_tail[i], Y0 + sin[i] * r_tail[i])
            p.points = polyline_points([tt, h])
            p.stroke_rgbas[:, :3] = pcol
            p.stroke_rgbas[:, 3] = alpha[i]
        subs += self.particles
        f = n - math.floor(n)
        base = (0.05 + 0.2 * speed) * fade_all
        for j, ring in enumerate(self.rings):
            r = (D0 / 2) * 2.0 ** (1 - j - f)
            a = base * math.exp(-((math.log2(r / (D0 / 2)) / 2.2) ** 2))
            ring.points = ring._unit * r + np.array([0, Y0, 0])
            ring.stroke_rgbas[:, :3] = pcol
            ring.stroke_rgbas[:, 3] = a
        subs += self.rings

        # the orders dust, the deck, the ring and the drawings ------------------
        a52 = tl.arrive["deck"]
        whole = smooth(a52, a52 + 0.6, t)
        dust_tint = mix(hex_rgb(LAVENDER), hex_rgb(VIOLET), whole)
        b1 = tl.beat_arrive[BEATS[0].key]
        dust_k = (0.6 + 0.4 * whole) * mix(1.0, 0.55, smooth(b1 - 0.4, b1, t))
        self.dust.pose(centre, (log2_orders(n) / LOG2_ALL) * dust_k, t, dust_tint, alpha=1 - loop)
        subs += self.dust.mobs
        # the deck steps back while a reference drawing holds the centre; its halo stays
        drawing_up = max(self.gather_of(k, t) for k in self.drawings)
        self.deck.pose(mix(CARDS - n, CARDS, loop), centre, alpha=mix(1 - smooth(0.0, 0.35, drawing_up), 1.0, loop))
        subs += self.deck.mobs
        ring_a = smooth(HOOK - 0.3, HOOK + 0.2, t) * (1 - smooth(o + 0.1, o + 0.6, t))
        if ring_a > 0.003:
            ring_col = mix(hex_rgb(LAVENDER), hex_rgb(VIOLET), whole)
            self.ring.pose(n, alpha=ring_a, color=ring_col, gather=smooth(o + 0.1, o + 0.6, t))
            subs += self.ring.mobs
        wg = smooth(a52 - 0.2, a52 + 0.5, t) * (1 - smooth(tl.beat_arrive[BEATS[0].key] - 0.5, tl.beat_arrive[BEATS[0].key], t))
        if wg > 0.003:
            self.whole_glow.points = quad(0.0, Y0, 4.6 * (1 + 0.03 * math.sin(2 * math.pi * 0.6 * t)))
            self.whole_glow.alpha = 0.3 * wg
            subs.append(self.whole_glow)
        for key, d in self.drawings.items():
            g = self.gather_of(key, t)
            if g > 0.003:
                d.pose(centre, 1.0, g, t)
                subs += d.mobs

        # the rows -----------------------------------------------------------
        for s in STOPS:
            if not s.row or t < tl.row_start[s.key]:
                continue
            t0, t_end = tl.row_start[s.key], tl.leave[s.key]
            ra = smooth(t0, t0 + 0.15, t) * (1 - smooth(t_end - 0.25, t_end, t))
            if ra < 0.003:
                continue
            p1 = ease_in_out((t - t0) / ROW_REF_PART)
            p2 = ease_in_out((t - t0 - ROW_REF_PART) / (ROW_DURATION - ROW_REF_PART))
            x_ref = ROW_X0 + (ROW_X1 - ROW_X0) * (s.ref_exact or s.row.ref) / orders(s.card)
            x1 = mix(ROW_X0, x_ref, p1)
            self.row_ref.set(ROW_X0, x1, ROW_Y, ra, s.row.color)
            subs += self.row_ref.mobs
            if p1 >= 1:
                self.row_mark.points = polyline_points([[x_ref, ROW_Y - 0.13], [x_ref, ROW_Y + 0.13]])
                self.row_mark.stroke_rgbas[:, :3] = hex_rgb(s.row.color)
                self.row_mark.stroke_rgbas[:, 3] = ra * smooth(0.0, 0.1, p2 + 1e-3)
                subs.append(self.row_mark)
            if p2 > 0:
                self.row_orders.set(x_ref, mix(x_ref, ROW_X1, p2), ROW_Y, ra, LAVENDER)
                subs += self.row_orders.mobs
            # the live counter, under the middle of the reference fill, never left of the track
            spec = counter_spec(s.row.ref * p1, self.font_count.digit_h, s.row.noun, s.row.notation)
            w = spec_width(spec, self.font_count.adv)
            x = min(max((ROW_X0 + x1) / 2, ROW_X0 + w / 2), 3.3 - w / 2)
            col = mix(hex_rgb(s.row.color), np.ones(3), 0.4)
            self.count_shadow.alpha = 0.78 * ra
            self.count_shadow.show(spec, (x, ROW_Y - 0.3))
            self.count_line.alpha = ra
            self.count_line.show(spec, (x, ROW_Y - 0.3))
            for slot in self.count_line.submobjects:
                slot.fill_rgbas[:, :3] = col
            subs += [self.count_shadow, self.count_line]
        for i, b in enumerate(BEATS):
            t0 = tl.beat_row_start[b.key]
            t_end = tl.beat_leave[b.key] if i < len(BEATS) - 1 else o + 0.05
            if t < t0:
                continue
            ra = smooth(t0, t0 + 0.3, t) * (1 - smooth(t_end - 0.25, t_end, t))
            if ra < 0.003:
                continue
            # all those shuffles: one point of light at the start of the track; the orders sweep past it
            self.row_star.points = quad(ROW_X0 + 0.08, ROW_Y, 0.5 * (1 + 0.08 * math.sin(2 * math.pi * 1.3 * t)))
            self.row_star.tint = mix(hex_rgb(b.accent), np.ones(3), 0.5)
            self.row_star.alpha = 0.95 * ra
            p = ease_in_out((t - t0 - 0.2) / 1.3)
            if p > 0:
                self.row_orders.set(ROW_X0 + 0.16, mix(ROW_X0 + 0.16, ROW_X1, p), ROW_Y, ra * 0.85, LAVENDER)
                subs += self.row_orders.mobs
            subs.append(self.row_star)

        # YOU pin: one of everyone alive -----------------------------------------
        s6 = BY_KEY["everyone"]
        pin_a = smooth(tl.arrive["everyone"], tl.arrive["everyone"] + 0.3, t) * (1 - smooth(tl.leave["everyone"] - 0.1, tl.leave["everyone"] + 0.2, t))
        if pin_a > 0.003:
            self.pin_glow._ra = 0.55 + 0.3 * math.sin(2 * math.pi * 1.2 * t)
            self.pin_rig.pose(1.0, [s6.you[0], Y0 + s6.you[1], 0], pin_a)
            subs.append(self.pin)

        # HUD ----------------------------------------------------------------
        hud_a = smooth(HUD_IN[0], HUD_IN[1], t) * (1 - smooth(o + 0.1, o + 0.6, t))
        if hud_a > 0.003:
            self.hud_static.pose(1.0, [0, 0, 0], hud_a)
            for tk in self.meter_ticks:
                tk.stroke_rgbas[:, 3] = hud_a * (0.3 + 0.7 * smooth(tk._n - 0.4, tk._n, n))
            subs += [self.hud_label, self.track, self.meter_base, self.meter_zero, *self.meter_ticks]
            y = self.meter_y(n)
            self.meter_fill.points = polyline_points([[MX, MY0], [MX, max(y, MY0 + 0.01)]])
            self.meter_fill.stroke_rgbas[:, 3] = hud_a
            self.meter_dot.points = quad(MX, y, 0.55)
            self.meter_dot.alpha = hud_a
            subs += [self.meter_fill, self.meter_dot]
            ct1 = tl.card_time(1)
            dim = mix(1.0, 0.4, smooth(tl.beat_arrive[BEATS[0].key] - 0.4, tl.beat_arrive[BEATS[0].key], t))
            ro_a = hud_a * smooth(ct1 - 0.05, ct1 + 0.25, t) * dim * (1 - smooth(o + 0.05, o + 0.24, t))
            if ro_a > 0.003:
                self.readout.alpha = ro_a
                # while the mechanic is taught the readout steps with the counter (52 → 2,652 → 132,600)
                val = float(orders(int(math.floor(n + 0.02)))) if t < tl.steps_end + 0.05 else orders_f(n)
                self.readout.show(count_spec(val, self.font_small.digit_h), (0, 4.5))
                subs.append(self.readout)
            for a in [self.you_tag, *self.brackets]:
                a.extra_alpha = hud_a
                a.update(t)
                if a.visible:
                    subs.append(a.mob)
            for j, (anim, tk, tk_next) in enumerate(self.pops):
                self.pop_alpha[j] = 0.0
                if t < tk:
                    continue
                pop = math.exp(-(t - tk) / 0.28) * (1 - smooth(tk_next - 0.1, tk_next, t))
                if pop > 0.01:
                    self.pop_alpha[j] = pop * hud_a
                    for g in anim.glyphs:
                        g.points = (g._rest - anim.center) * (1 + 0.25 * pop) + anim.center
                        g.fill_rgbas[:, 3] = g._fa * pop * hud_a
                    subs.append(anim.mob)

        # the counter (it flies to the centre for the outro) ------------------
        cnt_a = smooth(HUD_IN[0], HUD_IN[1], t) * (1 - smooth(tl.total - 1.45, tl.total - 1.1, t))
        if cnt_a > 0.003:
            shown = min(int(math.floor(n + 0.02)), CARDS) if t < o else CARDS
            punch = 0.0
            for s in STOPS:
                if tl.arrive[s.key] - 0.05 <= t <= tl.leave[s.key]:
                    shown = s.card
                dt_a = t - tl.arrive[s.key]
                if 0.0 <= dt_a <= 0.5:
                    punch = math.sin(math.pi * dt_a / 0.5)
            fly = ease_in_out((t - o - 0.05) / 0.85)
            sc = mix(1.0, 2.15, fly) * (1 + 0.16 * punch)
            cy = mix(5.6, 0.95, fly)
            flash, fcol = 0.0, np.ones(3)
            for s in STOPS:
                ta = tl.arrive[s.key]
                if ta - 0.05 <= t:
                    fl = math.exp(-(t - ta) / 0.45)
                    if fl > flash:
                        flash, fcol = fl, hex_rgb(s.accent)
            col = mix(np.ones(3), fcol, 0.85 * flash)
            col = mix(col, hex_rgb("#FFD1F3"), fly)
            self.counter.alpha = cnt_a
            self.counter.show([(ch, sc, 0.0) for ch in str(shown)], (0, cy))
            for slot in self.counter.submobjects:
                slot.fill_rgbas[:, :3] = col
            ga = 0.35 * flash + 0.45 * fly
            if ga > 0.01:
                g = self.counter_glow
                g.points = quad(0.0, cy, 3.2 * sc)
                g.tint = mix(fcol, hex_rgb(MAGENTA), fly)
                g.alpha = ga * cnt_a
                subs.append(g)
            subs.append(self.counter)
            ta = tl.arrive["bigbang"]
            if ta <= t <= ta + 1.2:
                p = (t - ta) / 1.2
                self.answer_ring.points = quad(0.0, 5.6, mix(1.4, 5.0, ease_out(p)))
                self.answer_ring.alpha = 0.9 * (1 - p) ** 1.5
                subs.append(self.answer_ring)

        # words ----------------------------------------------------------------
        for a in self.labels + self.hook + self.outro:
            a.update(t)
            if a.visible:
                subs.append(a.mob)

        subs.append(self.fx)
        mob.submobjects = subs

    # ------------------------------------------------------------------ checks
    def report(self, t, hook_frame=False):
        """Pose the frame at ``t`` and list text collisions and safe-zone breaches
        (film-style §1, §5, §8).  Returns a list of problem strings (empty = clean)."""
        self.t = t
        self.tick(self.stage, 0.0)
        shown = set(id(m) for m in self.stage.submobjects)

        def box(mob, min_alpha=0.02):
            pts = []
            for m in mob.get_family():
                if isinstance(m, VMobject) and len(m.points):
                    a = max(float(m.fill_rgbas[:, 3].max()) if len(m.fill_rgbas) else 0.0,
                            float(m.stroke_rgbas[:, 3].max()) if len(m.stroke_rgbas) else 0.0)
                    if a >= min_alpha:
                        pts.append(m.points[:, :2])
            if not pts:
                return None
            p = np.vstack(pts)
            return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())

        items = []  # (name, box, kind)
        for j, (p, _, _) in enumerate(self.pops):
            if self.pop_alpha[j] > 0.02 and id(p.mob) in shown:
                b = box(p.mob, min_alpha=0.0)
                if b:
                    items.append((p.mob.original_text if hasattr(p.mob, "original_text") else "pop", b, "text"))
        for a in self.labels + self.hook + self.outro + [self.you_tag] + self.brackets:
            if a.visible and id(a.mob) in shown and a.extra_alpha > 0.02:
                b = box(a.mob)
                if b:
                    kind = "meter" if a in self.brackets or a is self.you_tag else "text"
                    name = a.mob.original_text if hasattr(a.mob, "original_text") else next(
                        (m.original_text for m in a.mob.get_family() if hasattr(m, "original_text")), "text")
                    items.append((name[:28], b, kind))
        for name, m in (("HUD label", self.hud_label), ("meter 0", self.meter_zero)):
            if id(m) in shown:
                b = box(m)
                if b:
                    items.append((name, b, "meter" if name == "meter 0" else "text"))
        for name, m in (("counter", self.counter), ("readout", self.readout), ("row counter", self.count_line)):
            if id(m) in shown and m.alpha > 0.02:
                b = box(m)
                if b:
                    items.append((name, b, "text"))
        if id(self.pin) in shown:
            b = box(self.pin)
            if b:
                items.append(("YOU pin", b, "pin"))
        pad = 0.02

        def hit(a, b):
            return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad

        problems = []
        for i, (na, a, ka) in enumerate(items):
            for nb, b, kb in items[i + 1:]:
                if hit(a, b):
                    problems.append(f"overlap: {na} / {nb}")
            if ka == "text" and hit(a, (MX - 0.1, MY0, MX + 0.1, MY1)) and id(self.meter_base) in shown:
                problems.append(f"on meter: {na}")
            if a[1] < -4.9:
                problems.append(f"below y=-4.9 (bottom {a[1]:+.2f}): {na}")
            if a[2] > 3.4 and a[3] > -4.8 and a[1] < 1.5:
                problems.append(f"in button column (right {a[2]:+.2f}): {na}")
            if a[3] > 6.7:
                problems.append(f"in tab bar (top {a[3]:+.2f}): {na}")
            if ka == "text" and (a[2] - a[0]) > SAFE_W + 0.02 and abs((a[0] + a[2]) / 2) < 0.3:
                problems.append(f"wider than 6.8 ({a[2] - a[0]:.2f}): {na}")
        if hook_frame:
            b = box(self.hook_q)
            if b and b[3] > 6.0:
                problems.append(f"hook top {b[3]:+.2f} outside the 3:4 cover crop (+6.0)")
        ring_boxes = []
        if id(self.ring.slots[0]) in shown and self.ring.slots[0].stroke_rgbas[:, 3].max() > 0.05:
            for slot in self.ring.slots:
                p = slot.points[:, :2]
                ring_boxes.append((p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()))
        for na, a, ka in items:
            if ka == "text" and any(hit(a, rb) for rb in ring_boxes):
                problems.append(f"on the ring: {na}")
        lines = [f"t = {t:6.2f} s  n = {float(self.tl.n_at(t)):5.2f}: " + ("clean" if not problems else "; ".join(problems))]
        for na, a, ka in items:
            lines.append(f"    {na:30s} x {a[0]:+.2f} … {a[2]:+.2f}   y {a[1]:+.2f} … {a[3]:+.2f}  ({ka})")
        return problems, "\n".join(lines)

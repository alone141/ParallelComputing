"""THE DOUBLING: from the Planck length to the observable universe in 205 doublings.

A vertical (9:16) film for TikTok / Instagram Reels.  See README.md.

    manim -r 1080,1920 --fps 60 --disable_caching film.py DoublingFilm
"""

from __future__ import annotations

import math
import os
import re

import numpy as np
from manim import (
    BOLD,
    DOWN,
    RIGHT,
    MEDIUM,
    UP,
    Circle,
    Group,
    Line,
    Rectangle,
    RoundedRectangle,
    Scene,
    VGroup,
    VMobject,
    config,
)

from engine import (
    INTER,
    INTER_DISPLAY,
    MONO,
    AnimText,
    FilmCamera,
    MonoFont,
    MonoLine,
    PostFX,
    Rig,
    clamp01,
    ease_in_out,
    ease_out,
    fit_width,
    hex_rgb,
    lines,
    mix,
    rot2,
    sci_spec,
    smooth,
    text,
)
from story import (
    ARRIVE_BEFORE,
    BY_KEY,
    HOOK,
    LEAVE_AFTER,
    MILESTONES,
    N_END,
    N_GALAXY,
    N_HUMAN,
    PLANCK,
    STEP_TIMES,
    TIMELINE,
)
from visuals import D0, ORDER, R0, polyline_points, sprite

config.frame_width = 9.0
config.frame_height = 16.0
config.background_color = "#04050B"
config.video_encoder_options = {"crf": "15", "preset": "medium"}

Y0 = 0.22  # where the featured object sits
WHITE_TXT = "#F5F7FF"
MUTED = "#8A93A8"
BODY = "#C9D1E6"

NEBULA = {"star": 0.25, "solar": 0.6, "everest": 0.7, "earth": 0.8}  # per-scene glow strength

BG_KEYS = [  # (doublings, background colour)
    (0, "#08061A"), (40, "#060717"), (66, "#0E0710"), (82, "#040B19"), (92, "#0E0614"), (98, "#12060A"),
    (107, "#0D0905"), (116, "#050C13"), (128, "#050A15"), (139, "#030917"), (146, "#130A04"),
    (158, "#04070F"), (170, "#05050C"), (185, "#07051A"), (205, "#0B0414"), (215, "#0B0414"),
]


def bg_colour(n):
    for (n0, c0), (n1, c1) in zip(BG_KEYS, BG_KEYS[1:]):
        if n <= n1:
            return mix(hex_rgb(c0), hex_rgb(c1), smooth(n0, n1, n))
    return hex_rgb(BG_KEYS[-1][1])


def step_of(m):
    """The step (doubling) shown for a milestone: the nearest whole doubling."""
    return int(round(m.n))


def count_text(value, total):
    """Format a running count, switching to words for astronomically big totals."""
    if total >= 1e7:
        for div, name in ((1e18, "quintillion"), (1e15, "quadrillion"), (1e12, "trillion"),
                          (1e9, "billion"), (1e6, "million")):
            if total >= div:
                return f"{value / div:,.0f} {name}"
    return f"{int(round(value)):,}"


def markup(s, size, accent, **kw):
    """Text with [highlighted] spans; ``\\n`` makes extra centred lines."""
    rows = []
    for row in s.split("\n"):
        hl = re.findall(r"\[(.+?)\]", row)
        clean = row.replace("[", "").replace("]", "")
        rows.append(text(clean, size, t2c={h: accent for h in hl}, **kw))
    g = VGroup(*rows).arrange(DOWN, buff=0.1)
    return g


class DoublingFilm(Scene):
    def __init__(self, **kwargs):
        super().__init__(camera_class=FilmCamera, **kwargs)

    # ------------------------------------------------------------------ build
    def get_moving_and_static_mobjects(self, animations):
        # The stage swaps its children every frame, so hand the renderer the
        # stage itself (its family is re-read each frame) instead of letting
        # manim snapshot the family when the animation starts.
        if hasattr(self, "stage"):
            return [self.stage], []
        return super().get_moving_and_static_mobjects(animations)

    def construct(self):
        self.setup_film()
        # FILM_RANGE="start,end" renders one slice (render.sh runs slices in parallel).
        start, end = 0.0, self.tl.total
        if os.environ.get("FILM_RANGE"):
            start, end = (float(x) for x in os.environ["FILM_RANGE"].split(","))
        self.t = start
        self.tick(self.stage, 0.0)
        # a hair under the slice length so float rounding never adds a frame
        self.wait(end - start - 0.25 / config.frame_rate, frozen_frame=False)

    def setup_film(self):
        self.tl = TIMELINE
        self.visuals = [cls(m, self.tl) for cls, m in zip(ORDER, MILESTONES)]
        self.idx_human = [m.key for m in MILESTONES].index("human")
        self.build_background()
        self.build_particles()
        self.build_rings()
        self.build_pin()
        self.build_hud()
        self.build_labels()
        self.build_hook_and_outro()
        self.fx = PostFX()
        self.stage = Group()
        self.t = 0.0
        self.stage.add_updater(self.tick)
        self.add(self.stage)
        self.tick(self.stage, 0.0)

    def build_background(self):
        self.bg = Rectangle(width=9.4, height=16.4, stroke_width=0, fill_color="#08061A", fill_opacity=1)
        self.nebula = sprite("glow", D0 * 2.7, "#FFFFFF", alpha=0.0)

    def build_particles(self, count=150):
        rng = np.random.default_rng(1)
        self.p_theta = rng.uniform(0, 2 * np.pi, count)
        self.p_phase = rng.uniform(0, 1, count)
        self.p_bright = rng.uniform(0.3, 1.0, count)
        self.particles = [
            VMobject(stroke_color="#FFFFFF", stroke_width=w, stroke_opacity=0, fill_opacity=0)
            for w in rng.uniform(1.3, 2.8, count)
        ]
        for p in self.particles:
            p.points = polyline_points([[0, 0], [0.01, 0]])

    def build_rings(self):
        self.rings = [Circle(radius=1, stroke_width=1.5, stroke_color="#FFFFFF", stroke_opacity=0) for _ in range(9)]
        for r in self.rings:
            r._unit = r.points.copy()

    def build_pin(self):
        cyan = "#6CF2FF"
        cap = RoundedRectangle(width=0.9, height=0.4, corner_radius=0.2, fill_color=cyan, fill_opacity=1,
                               stroke_width=0).move_to([0, 0.6, 0])
        lab = text("YOU", 21, font=INTER_DISPLAY, weight=BOLD, color="#03121A", tracking=0.035).move_to([0, 0.6, 0])
        stem = Line([0, 0.4, 0], [0, 0.1, 0], stroke_width=2.6, color=cyan)
        ring = Circle(radius=0.075, stroke_color=cyan, stroke_width=2.6, fill_color=cyan, fill_opacity=0.25)
        self.pin_glow = sprite("glow", 0.9, cyan, alpha=0.75)
        self.pin = Group(self.pin_glow, stem, ring, cap, lab)
        self.pin_rig = Rig(self.pin)

    def build_hud(self):
        self.font_big = MonoFont(size=104, weight=BOLD)
        self.font_small = MonoFont(size=30, weight=MEDIUM)
        self.counter = MonoLine(self.font_big, 4, color="#FFFFFF")
        self.counter_glow = sprite("glow", 3.2, "#FFFFFF", alpha=0.0)
        self.answer_ring = sprite("ring", 2.0, "#C7A6FF", alpha=0.0, r0=0.8, inner=0.02, outer=0.06)
        self.sizeline = MonoLine(self.font_small, 18, color="#9FB3D9")
        self.hud_label = text("DOUBLINGS", 19, weight=MEDIUM, color=MUTED, tracking=0.09).move_to([0, 6.55, 0])
        y = 4.12
        self.ruler = VGroup(
            Line([-R0, y, 0], [R0, y, 0], stroke_width=1.6, color="#5D6B8C"),
            Line([-R0, y - 0.1, 0], [-R0, y + 0.1, 0], stroke_width=1.6, color="#5D6B8C"),
            Line([R0, y - 0.1, 0], [R0, y + 0.1, 0], stroke_width=1.6, color="#5D6B8C"),
            Line([0, y - 0.05, 0], [0, y + 0.05, 0], stroke_width=1.2, color="#5D6B8C"),
        )
        self.x2 = text("×2", 40, font=INTER_DISPLAY, weight=BOLD, color="#B9A6FF").move_to([1.55, 5.62, 0])
        self.x2_anim = AnimText(self.x2, 0)
        ts = np.linspace(HOOK, self.tl.steps_end + 2.5, 20000)
        ns = self.tl.n_at(ts)
        self.x2_times = [float(ts[np.argmax(ns >= k - 0.02)]) for k in range(1, 9)]
        # progress meter on the left edge
        self.mx, self.my0, self.my1 = -3.95, -2.45, 3.45
        mx, my = self.mx, self.meter_y
        self.meter_base = Line([mx, self.my0, 0], [mx, self.my1, 0], stroke_width=2, color="#27304A")
        self.meter_fill = Line([mx, self.my0, 0], [mx, self.my0 + 0.01, 0], stroke_width=3.2)
        self.meter_fill.set_stroke(["#B9A6FF", "#6CF2FF", "#FF6FD8"]).set_sheen_direction(UP)
        self.meter_dot = sprite("star", 0.55, "#FFFFFF", spikes=0.0)
        self.meter_ticks = []
        for m in MILESTONES[1:]:
            tk = Line([mx - 0.08, my(m.n), 0], [mx + 0.08, my(m.n), 0], stroke_width=2.2, color=m.accent)
            tk._n = m.n
            self.meter_ticks.append(tk)
        self.meter_zero = text("0", 17, font=MONO, color=MUTED).move_to([mx, self.my0 - 0.26, 0])
        self.you_tag = AnimText(text("YOU", 16, font=INTER_DISPLAY, weight=BOLD, color="#6CF2FF")
                                .move_to([mx + 0.36, my(N_HUMAN), 0]), self.tl.arrive["human"] + 0.4)
        mw = BY_KEY["milkyway"]
        t_ans = self.tl.arrive["milkyway"] + mw.note_delay
        t_end = self.tl.arrive["universe"] + 0.6
        bx = mx + 0.74  # right of the meter's YOU tag
        brA = VGroup(Line([bx, my(0), 0], [bx, my(N_HUMAN) - 0.04, 0], stroke_width=1.8, color=MUTED),
                     Line([bx - 0.07, my(0), 0], [bx, my(0), 0], stroke_width=1.8, color=MUTED))
        brB = VGroup(Line([bx, my(N_HUMAN) + 0.04, 0], [bx, my(N_GALAXY), 0], stroke_width=2.2, color="#6CF2FF"),
                     Line([bx - 0.07, my(N_GALAXY), 0], [bx, my(N_GALAXY), 0], stroke_width=2.2, color="#6CF2FF"))
        lA = text("116", 20, font=MONO, weight=BOLD, color=BODY).move_to([bx + 0.36, (my(0) + my(N_HUMAN)) / 2, 0])
        lB = text("69", 20, font=MONO, weight=BOLD, color="#6CF2FF").move_to([bx + 0.3, (my(N_HUMAN) + my(N_GALAXY)) / 2, 0])
        self.brackets = [AnimText(VGroup(brA, lA), t_ans, t_end, stagger=0.0),
                         AnimText(VGroup(brB, lB), t_ans + 0.35, t_end, stagger=0.0)]
        self.hud_static = Rig(VGroup(self.hud_label, self.ruler, self.meter_base, self.meter_zero, *self.meter_ticks))
        # live counter that rides the head of each comparison row
        self.font_count = MonoFont(size=21, weight=BOLD, letters=True)
        self.count_shadow = MonoLine(self.font_count, 40, color="#000000", stroke_width=8, stroke_color="#000000")
        self.count_line = MonoLine(self.font_count, 40, color="#FFFFFF")

    def meter_y(self, n):
        return self.my0 + (self.my1 - self.my0) * clamp01(n / N_END)

    def build_labels(self):
        tl = self.tl
        self.labels = []
        ya, yt, yc = -2.86, -3.51, -4.21
        safe_w = 6.8  # keep clear of the TikTok/Reels button column (x > 3.4)

        def block(size_text, title, compare, accent, t_in, t_out, note=None, t_note=None, step=None):
            mono = not re.search("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]", size_text)
            a = text(size_text, 25 if mono else 27, font=MONO if mono else INTER, weight=MEDIUM,
                     color=accent if step is None else BODY)
            if step is not None:
                chip_txt = text(step, 22, font=MONO, weight=BOLD, color=accent)
                chip = RoundedRectangle(width=chip_txt.width + 0.38, height=0.44, corner_radius=0.22,
                                        fill_color=accent, fill_opacity=0.16, stroke_color=accent, stroke_width=1.6)
                chip_txt.move_to(chip)
                a = VGroup(VGroup(chip, chip_txt), a).arrange(RIGHT, buff=0.26)
            a.move_to([0, ya, 0])
            ti = fit_width(text(title, 60, font=INTER_DISPLAY, weight=BOLD, color=WHITE_TXT), safe_w).move_to([0, yt, 0])
            self.labels += [AnimText(a, t_in, t_out), AnimText(ti, t_in + 0.06, t_out, stagger=0.018)]
            c_out = t_note if note else t_out
            for s, ti_, to_ in [(compare, t_in + 0.16, c_out)] + ([(note, t_note + 0.12, t_out)] if note else []):
                c = fit_width(markup(s, 31 if "\n" not in s else 29, accent, color=BODY, weight=MEDIUM), safe_w)
                c.move_to([0, yc - (c.height - 0.32) / 2, 0])
                self.labels.append(AnimText(c, ti_, to_, stagger=0.008))

        planck = MILESTONES[0]
        block(planck.size_text, planck.title, planck.compare, planck.accent, 0.45, 3.3)
        block("2 → 4 → 8 → 16 → … Planck lengths", "DOUBLE IT. AGAIN.", "Every step makes it [twice] as long.",
              "#B9A6FF", 3.45, tl.steps_end - 0.1)
        # Colliders resolve ~1e-20 m (about doubling 50): only claim "beyond any
        # experiment" while the counter is below that.  The exit fade takes ~0.3 s
        # at ~30 doublings/s, so start it around step 42.
        ts = np.linspace(tl.steps_end, tl.arrive["proton"], 4000)
        t50 = float(ts[np.argmax(tl.n_at(ts) >= 42.0)])
        block(f"STEPS {len(STEP_TIMES) + 1} → 50", "THE GREAT DESERT", "Smaller than [any experiment] can probe.",
              "#B9A6FF", tl.steps_end + 0.15, t50)
        for m in MILESTONES[1:]:
            t_in = tl.arrive[m.key] - 0.3
            t_out = tl.leave[m.key] - 0.02 if m.key != "universe" else tl.outro_start + 0.05
            t_note = tl.arrive[m.key] + m.note_delay if m.note else None
            block(m.size_text, m.title, m.compare, m.accent, t_in, t_out, m.note, t_note, step=f"STEP {step_of(m)}")

    def build_hook_and_outro(self):
        tl = self.tl
        q = lines(["How many times", "must you double this", "to reach a galaxy?"], size=50,
                  font=INTER_DISPLAY, weight=BOLD, color=WHITE_TXT, gap=0.16,
                  t2c={"double": "#B9A6FF", "galaxy?": "#6CF2FF"})
        q.move_to([0, 4.75, 0])
        sub = text("Guess before it ends.", 27, color=MUTED).move_to([0, 3.3, 0])
        self.hook = [AnimText(q, 0.0, 2.1, stagger=0.011, dur_in=0.32), AnimText(sub, 0.85, 2.1)]
        o = tl.outro_start
        self.outro_t = o
        self.big_glow = sprite("glow", 6.5, "#FF6FD8", alpha=0.0)
        l1 = text("DOUBLINGS", 26, weight=MEDIUM, color=BODY, tracking=0.13).move_to([0, -0.78, 0])
        l2 = lines(["From the Planck length", "to the edge of everything."], size=34, color=WHITE_TXT,
                   weight=MEDIUM, gap=0.14).move_to([0, -1.95, 0])
        end = tl.total - 1.45
        self.outro = [AnimText(l1, o + 0.75, end), AnimText(l2, o + 1.0, end, stagger=0.01)]

    # ------------------------------------------------------------------ frame
    def layout(self, n, t):
        ms = MILESTONES
        last = len(ms) - 1
        k = max(i for i in range(len(ms)) if ms[i].n <= n + 1e-9)
        ang = [v.angle(t) for v in self.visuals]

        def scale(j):
            return 2.0 ** min(ms[j].n - n, 900)

        def rotv(j, p):
            r = rot2(ang[j])
            return (r @ np.array([p[0], p[1], 0.0]))[:2]

        def off(j):  # centre of j relative to centre of j+1, in screen units
            return D0 * (rotv(j + 1, ms[j + 1].sit) * scale(j + 1) - rotv(j, ms[j].attach) * scale(j))

        if k < last:
            na, nb = ms[k].n + LEAVE_AFTER, ms[k + 1].n - ARRIVE_BEFORE
            beta = clamp01((2 ** (n - na) - 1) / (2 ** (nb - na) - 1)) if n > na else 0.0
            origin = k + 1
        else:
            beta, origin = 0.0, last
        pos = {origin: np.zeros(2)}
        for j in range(origin - 1, max(origin - 4, -1), -1):
            pos[j] = pos[j + 1] + off(j)
        for j in range(origin + 1, min(origin + 3, last + 1)):
            pos[j] = pos[j - 1] - off(j - 1)
        cam = (1 - beta) * pos[k] if k < last else np.zeros(2)
        centre = np.array([0.0, Y0])
        screen = {j: centre + p - cam for j, p in pos.items()}

        def you(j):
            if ms[j].you is None or j not in screen:
                return None
            return screen[j] + rotv(j, ms[j].you) * D0 * scale(j)

        if k < last:
            a, b = you(k), you(k + 1)
            pin = b if a is None else (a if b is None else (1 - beta) * a + beta * b)
        else:
            pin = you(last)
        return screen, pin

    def tick(self, mob, dt):
        self.t += dt
        t = self.t
        tl = self.tl
        n = float(tl.n_at(t))
        v = abs(float(tl.speed_at(t)))
        screen, pin = self.layout(n, t)
        subs = [self.bg]
        o = self.outro_t

        # background, nebula ------------------------------------------------
        loop = smooth(tl.total - 1.2, tl.total - 0.05, t)  # glide back to the first frame
        self.bg.fill_rgbas[:, :3] = mix(bg_colour(n), bg_colour(0.0), loop)
        ws = np.array([math.exp(-(((n - m.n) / 2.6) ** 2)) for m in MILESTONES])
        ws[0] = math.exp(-((n / 4.0) ** 2))
        if ws.sum() > 1e-4:
            j_near = int(np.argmax(ws))
            col = sum(w * hex_rgb(m.accent) for w, m in zip(ws, MILESTONES)) / ws.sum()
            c = screen.get(j_near, np.array([0.0, Y0]))
            strength = NEBULA.get(MILESTONES[j_near].key, 1.0)
            c = mix(c, np.array([0.0, Y0]), loop)
            self.nebula.points = np.array([[c[0] - 7.3, c[1] + 7.3, 0], [c[0] + 7.3, c[1] + 7.3, 0],
                                           [c[0] - 7.3, c[1] - 7.3, 0], [c[0] + 7.3, c[1] - 7.3, 0]])
            self.nebula.tint = mix(col, hex_rgb(MILESTONES[0].accent), loop)
            a_neb = 0.2 * strength * float(ws.max()) * (1 - 0.6 * smooth(o, o + 1.0, t))
            self.nebula.alpha = mix(a_neb, 0.2, loop)
            subs.append(self.nebula)
        accent = hex_rgb(MILESTONES[int(np.argmax(ws))].accent) if ws.sum() > 1e-4 else hex_rgb("#B9A6FF")

        # particles ----------------------------------------------------------
        speed = clamp01(v / 30)
        fade_all = 1 - smooth(tl.total - 2.0, tl.total - 1.3, t) + smooth(tl.total - 0.9, tl.total - 0.05, t)
        P = 6.0
        u = (((n + 0.06 * t) / P) + self.p_phase) % 1.0
        r_head = 10.5 * 2.0 ** (-P * u)
        r_tail = r_head * 2.0 ** (min(v * 0.04, 1.9) + 0.05)
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

        # doubling rings -----------------------------------------------------
        f = n - math.floor(n)
        base = (0.05 + 0.2 * speed) * fade_all
        for j, ring in enumerate(self.rings):
            r = R0 * 2.0 ** (1 - j - f)
            a = base * math.exp(-((math.log2(r / R0) / 2.2) ** 2))
            ring.points = ring._unit * r + np.array([0, Y0, 0])
            ring.stroke_rgbas[:, :3] = pcol
            ring.stroke_rgbas[:, 3] = a
        subs += self.rings

        # the drawings -------------------------------------------------------
        for j in range(len(self.visuals) - 1, -1, -1):
            vis = self.visuals[j]
            mult = 1.0
            if vis.key == "universe":
                mult = mix(1.0, 0.38, smooth(o, o + 0.9, t)) * (1 - smooth(tl.total - 1.6, tl.total - 0.9, t))
            if vis.key == "planck":
                if t > tl.total - 1.2:  # loop: end on the very first frame
                    vis.update(t, 0.0, (0.0, Y0), smooth(tl.total - 1.0, tl.total - 0.02, t))
                elif j in screen:
                    vis.update(t, n, screen[j])
                else:
                    vis.visible = False
            elif j in screen:
                vis.update(t, n, screen[j], mult)
            else:
                vis.visible = False
            if vis.visible:
                subs.append(vis.mob)

        # the comparison counter ------------------------------------------------
        for vis in self.visuals:
            row = vis.row
            if row is None or not vis.visible or t < row.t_start:
                continue
            a = smooth(row.t_start, row.t_start + 0.15, t) * (1 - smooth(row.t_end - 0.25, row.t_end, t))
            a *= vis.alpha(n, t)
            if a < 0.003:
                continue
            spec = row.spec
            label = f"{count_text(row.progress(t) * spec.count, spec.count)} {spec.noun}"
            hx, hy = row.head_point()[:2]
            width = len(label) * self.font_count.adv
            if spec.label_side in ("up", "down"):
                # ride the middle of the filled part, so it lands centred on the row
                sx, sy = row.glow.points[0][:2]
                mx, my = (sx + hx) / 2, (sy + hy) / 2
                x = min(max(mx, -3.4 + width / 2), 3.3 - width / 2)  # clear of the meter and the button column
                off = spec.label_offset
                pos, align = (x, my + (off if spec.label_side == "up" else -off)), "center"
            elif spec.label_side == "left":
                pos, align = (hx - 0.2, hy), "right"
            else:
                pos, align = (hx + 0.2, hy), "left"
            col = mix(hex_rgb(spec.color), np.ones(3), 0.4)
            chars = [(ch, 1.0, 0.0) for ch in label]
            self.count_shadow.alpha = 0.78 * a
            self.count_shadow.show(chars, pos, align)
            self.count_line.alpha = a
            self.count_line.show(chars, pos, align)
            for slot in self.count_line.submobjects:
                slot.fill_rgbas[:, :3] = col
            subs += [self.count_shadow, self.count_line]

        # YOU pin ------------------------------------------------------------
        pin_a = smooth(N_HUMAN - 0.6, N_HUMAN - 0.25, n) * (1 - smooth(o, o + 0.5, t))
        if pin is not None and pin_a > 0.003:
            self.pin_glow._ra = 0.55 + 0.3 * math.sin(2 * math.pi * 1.2 * t)
            self.pin_rig.pose(1.0, [pin[0], pin[1], 0], pin_a)
            subs.append(self.pin)

        # HUD ----------------------------------------------------------------
        hud_a = smooth(2.5, 2.95, t) * (1 - smooth(o + 0.1, o + 0.6, t))
        if hud_a > 0.003:
            self.hud_static.pose(1.0, [0, 0, 0], hud_a)
            for tk in self.meter_ticks:
                tk.stroke_rgbas[:, 3] = hud_a * (0.3 + 0.7 * smooth(tk._n - 0.4, tk._n, n))
            subs += [self.hud_label, self.ruler, self.meter_base, self.meter_zero, *self.meter_ticks]
            y = self.meter_y(n)
            self.meter_fill.points = polyline_points([[self.mx, self.my0], [self.mx, max(y, self.my0 + 0.01)]])
            self.meter_fill.stroke_rgbas[:, 3] = hud_a
            self.meter_dot.points = np.array([[self.mx - 0.275, y + 0.275, 0], [self.mx + 0.275, y + 0.275, 0],
                                              [self.mx - 0.275, y - 0.275, 0], [self.mx + 0.275, y - 0.275, 0]])
            self.meter_dot.alpha = hud_a
            subs += [self.meter_fill, self.meter_dot]
            self.sizeline.alpha = hud_a
            self.sizeline.show(sci_spec(PLANCK * 2.0**n, digit_h=self.font_small.digit_h), (0, 4.5))
            subs.append(self.sizeline)
            for a in [self.you_tag, *self.brackets]:
                a.extra_alpha = hud_a
                a.update(t)
                if a.visible:
                    subs.append(a.mob)
            # ×2 pops during the first doublings
            pop = max([math.exp(-(t - tk) / 0.28) for tk in self.x2_times if t >= tk] or [0.0])
            if pop > 0.01 and n < 9:
                self.x2_anim.offset = np.zeros(3)
                for g in self.x2_anim.glyphs:
                    g.points = (g._rest - self.x2_anim.center) * (1 + 0.25 * pop) + self.x2_anim.center
                    g.fill_rgbas[:, 3] = g._fa * pop * hud_a
                subs.append(self.x2)

        # the counter (it flies to the centre for the outro) ------------------
        cnt_a = smooth(2.5, 2.95, t) * (1 - smooth(tl.total - 1.45, tl.total - 1.1, t))
        if cnt_a > 0.003:
            shown = min(int(round(n)), 205) if t < o else 205
            if t < tl.steps_end + 0.05:  # counted steps: flip with the ×2 pop and the pluck
                shown = int(math.floor(n + 0.02))
            punch = 0.0
            for m in MILESTONES[1:]:
                if tl.arrive[m.key] - 0.05 <= t <= tl.leave[m.key]:
                    shown = step_of(m)
                dt_a = t - tl.arrive[m.key]
                if 0.0 <= dt_a <= 0.5:
                    punch = math.sin(math.pi * dt_a / 0.5)
            fly = ease_in_out((t - o - 0.05) / 0.85)
            s = mix(1.0, 2.15, fly) * (1 + 0.16 * punch)
            cy = mix(5.6, 0.95, fly)
            flash, fcol = 0.0, np.ones(3)
            for m in MILESTONES[1:]:
                ta = tl.arrive[m.key]
                if ta - 0.05 <= t:
                    fl = math.exp(-(t - ta) / 0.45)
                    if fl > flash:
                        flash, fcol = fl, hex_rgb(m.accent)
            col = mix(np.ones(3), fcol, 0.85 * flash)
            col = mix(col, hex_rgb("#FFD1F3"), fly)
            self.counter.alpha = cnt_a
            spec = [(ch, s, 0.0) for ch in str(shown)]
            self.counter.show(spec, (0, cy))
            for slot in self.counter.submobjects:
                slot.fill_rgbas[:, :3] = col
            ga = 0.35 * flash + 0.45 * fly
            if ga > 0.01:
                g = self.counter_glow
                hh = 3.2 * s
                g.points = np.array([[-hh / 2, cy + hh / 2, 0], [hh / 2, cy + hh / 2, 0],
                                     [-hh / 2, cy - hh / 2, 0], [hh / 2, cy - hh / 2, 0]])
                g.tint = mix(fcol, hex_rgb("#FF6FD8"), fly)
                g.alpha = ga * cnt_a
                subs.append(g)
            subs.append(self.counter)
            ta = tl.arrive["milkyway"]
            if ta <= t <= ta + 1.2:
                p = (t - ta) / 1.2
                hh = mix(1.4, 5.0, ease_out(p))
                self.answer_ring.points = np.array([[-hh / 2, 5.6 + hh / 2, 0], [hh / 2, 5.6 + hh / 2, 0],
                                                    [-hh / 2, 5.6 - hh / 2, 0], [hh / 2, 5.6 - hh / 2, 0]])
                self.answer_ring.alpha = 0.9 * (1 - p) ** 1.5
                subs.append(self.answer_ring)

        # words ----------------------------------------------------------------
        for a in self.labels + self.hook + self.outro:
            a.update(t)
            if a.visible:
                subs.append(a.mob)

        subs.append(self.fx)
        mob.submobjects = subs

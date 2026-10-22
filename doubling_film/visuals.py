"""The fourteen drawings.  Each one is authored at the reference size: its
characteristic size spans ``D0`` scene units, centred on the origin.  The film
re-poses them every frame (scale ``2**(n_i - n)``, pan, fade) through a Rig.
"""

from __future__ import annotations

import math

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    DashedLine,
    DashedVMobject,
    Dot,
    Ellipse,
    Group,
    Line,
    Polygon,
    RoundedRectangle,
    VGroup,
    VMobject,
)
from manim.constants import CapStyleType, LineJointType

import assets
from engine import (
    INTER,
    Rig,
    Sprite,
    Texture,
    ease_in_out,
    ease_out,
    hex_rgb,
    mix,
    rot2,
    set_rest,
    smooth,
    text,
)
from story import EVEREST_SUMMIT_X, GALAXY_SUN, SOLAR_EARTH

D0 = 5.2  # reference diameter in scene units
R0 = D0 / 2

_TEX: dict = {}


def tex(kind, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in _TEX:
        _TEX[key] = Texture(*getattr(assets, kind)(**kw))
    return _TEX[key]


def sprite(kind, height, color="#FFFFFF", alpha=1.0, blend="screen", at=(0.0, 0.0), width=None, **kw):
    s = Sprite(tex(kind, **kw), height, width=width, blend=blend, alpha=alpha, tint=hex_rgb(color))
    s.shift([at[0], at[1], 0])
    return s


def quad(x, y, h, w=None):
    w = h if w is None else w
    return np.array([[x - w / 2, y + h / 2, 0], [x + w / 2, y + h / 2, 0],
                     [x - w / 2, y - h / 2, 0], [x + w / 2, y - h / 2, 0]])


def polyline_points(p):
    """Straight-segment Bezier points for a polyline (N, 2|3) -> (4(N-1), 3)."""
    p = np.asarray(p, float)
    if p.shape[1] == 2:
        p = np.hstack([p, np.zeros((len(p), 1))])
    a, b = p[:-1], p[1:]
    out = np.empty((len(a) * 4, 3))
    out[0::4], out[1::4], out[2::4], out[3::4] = a, a + (b - a) / 3, a + 2 * (b - a) / 3, b
    return out


def stroke_path(points, color, width, opacity=1.0, closed=False):
    pts = list(points) + ([points[0]] if closed else [])
    v = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity, fill_opacity=0)
    v.points = polyline_points(pts)
    v.set_cap_style(CapStyleType.ROUND)
    v.joint_type = LineJointType.ROUND
    return v


def move_rest(m, p):
    """Translate a sub-drawing's rest pose (relative to where it was built)."""
    for g in m.get_family():
        if hasattr(g, "_rest"):
            if not hasattr(g, "_home"):
                g._home = g._rest.copy()
            set_rest(g, g._home + p)


def blob(points, fill, opacity=1.0, stroke=None, stroke_width=0.0, smooth_it=True):
    pts = [np.array([x, y, 0.0]) for x, y in points]
    v = VMobject(fill_color=fill, fill_opacity=opacity, stroke_width=stroke_width,
                 stroke_color=stroke or fill, stroke_opacity=1.0 if stroke else 0)
    if smooth_it:
        v.set_points_smoothly(pts + [pts[0]])
    else:
        v.set_points_as_corners(pts + [pts[0]])
    return v


class Visual:
    fade_in = (1.6, 0.55)
    fade_out = (3.4, 6.4)

    def __init__(self, ms, tl):
        self.ms, self.tl = ms, tl
        self.key = ms.key
        self.n0 = ms.n
        self.t_arrive = tl.arrive[ms.key]
        self.t_leave = tl.leave[ms.key]
        self.spinners: list = []
        self.mob = self.build()
        self.rig = Rig(self.mob)
        for group, _ in self.spinners:
            for m in group.get_family():
                if hasattr(m, "_rest"):
                    m._base = m._rest.copy()
        self.visible = False

    def build(self):
        raise NotImplementedError

    def alpha(self, n, t):
        a_in = smooth(self.n0 - self.fade_in[0], self.n0 - self.fade_in[1], n)
        a_out = 1 - smooth(self.n0 + self.fade_out[0], self.n0 + self.fade_out[1], n)
        return a_in * a_out

    def angle(self, t):
        return 0.0

    def animate(self, t, n):
        for group, omega in self.spinners:
            r = rot2(omega * t)
            for m in group.get_family():
                if hasattr(m, "_base"):
                    m._rest = m._base @ r.T

    def update(self, t, n, center, alpha_mult=1.0):
        a = self.alpha(n, t) * alpha_mult
        s = 2.0 ** (self.n0 - n)
        if a < 0.003 or s > 14 or s < 0.0015:
            self.visible = False
            return
        self.visible = True
        self.animate(t, n)
        self.rig.pose(s, [center[0], center[1], 0.0], a, self.angle(t))

    def stagger(self, mobs, t, start, step, dur=0.18):
        for i, m in enumerate(mobs):
            k = ease_out((t - start - i * step) / dur)
            for leaf in m.get_family():
                leaf.k = k


# ---------------------------------------------------------------------------


class Planck(Visual):
    """One Planck length, then 2, 4, 8... lined up across the reference width."""

    def build(self):
        self.halo = sprite("glow", D0 * 1.35, "#8F6BFF", alpha=0.55)
        self.dots = [sprite("star", 1.0, "#E4DCFF", spikes=0.0) for _ in range(64)]
        self.line = Line(LEFT * R0, RIGHT * R0, stroke_width=3.0, color="#D4C8FF")
        circ = DashedVMobject(Circle(radius=R0, stroke_width=1.6, color="#B9A6FF"), num_dashes=90)
        ticks = VGroup(*[Line(RIGHT * (R0 - 0.12), RIGHT * (R0 + 0.12), stroke_width=1.8, color="#B9A6FF")
                         .rotate(a, about_point=np.zeros(3)) for a in (0, np.pi / 2, np.pi, 3 * np.pi / 2)])
        self.reticle = VGroup(circ, ticks)
        self.line_rig = Rig(VGroup(self.line))
        self.ret_rig = Rig(self.reticle)
        return Group(self.halo, *self.dots, self.line, self.reticle)

    def update(self, t, n, center, alpha_mult=1.0):
        self.visible = n < 12 and alpha_mult > 0.003
        if not self.visible:
            return
        cx, cy = center
        m = int(math.floor(n + 1e-9))
        f = n - m
        p = ease_in_out(f)
        row_a = (1 - smooth(4.2, 6.6, n)) * alpha_mult
        breathe = 1 + 0.035 * math.sin(2 * math.pi * 0.7 * t) * (1 - smooth(0, 0.5, n))
        for d in self.dots:
            d.alpha = 0.0
        if m < 6 and row_a > 0.003:
            count = 2**m
            u_m, u_m1 = D0 / count, D0 / count / 2
            for j in range(2 * count):
                i = j // 2
                start = (i - (count - 1) / 2) * u_m
                end = (j - (2 * count - 1) / 2) * u_m1
                h = mix(u_m, u_m1, p) * 1.15 * breathe
                d = self.dots[j]
                d.points = quad(cx + mix(start, end, p), cy, h)
                d.alpha = row_a * (1.0 if j % 2 == 0 else smooth(0.0, 0.2, p))
        self.halo.points = quad(cx, cy, D0 * 1.35 * breathe)
        self.halo.alpha = 0.55 * (1 - smooth(0.6, 4.0, n)) * alpha_mult
        self.line_rig.pose(1.0, [cx, cy, 0], smooth(3.3, 4.6, n) * (1 - smooth(7.5, 11.5, n)) * 0.9 * alpha_mult)
        self.ret_rig.pose(1.0, [cx, cy, 0], 0.7 * (1 - smooth(0.25, 1.6, n)) * alpha_mult)


class Proton(Visual):
    QUARKS = ("#FF3D5A", "#3DFFA0", "#3D8BFF")

    def build(self):
        bag = sprite("glow", D0 * 1.35, "#FF6A33", alpha=0.6)
        body = sprite("disc", D0, "#FF8A5B", alpha=0.10, edge=0.3)
        edge = sprite("ring", D0 * 1.25, "#FFB08A", alpha=0.45, r0=0.8, inner=0.02, outer=0.07)
        self.gluons = [VMobject(stroke_color="#FFE6CC", stroke_width=2.6, stroke_opacity=0.6, fill_opacity=0)
                       for _ in range(3)]
        for g in self.gluons:
            g.points = polyline_points([[0, 0], [1, 0]])
        self.q_glow = [sprite("glow", 1.15 * R0, c, alpha=1.0) for c in self.QUARKS]
        self.q_core = [sprite("star", 0.62 * R0, "#FFFFFF", spikes=0.0) for _ in self.QUARKS]
        return Group(bag, body, edge, *self.gluons, *self.q_glow, *self.q_core)

    def animate(self, t, n):
        pos = []
        for i in range(3):
            a = math.radians(90 + 120 * i) + 0.25 * math.sin(0.6 * t + i)
            r = 0.36 * R0 + 0.07 * R0 * math.sin(1.9 * t + 2.1 * i)
            pos.append(np.array([r * math.cos(a), r * math.sin(a)]))
        for i in range(3):
            set_rest(self.q_glow[i], quad(*pos[i], 1.15 * R0))
            set_rest(self.q_core[i], quad(*pos[i], 0.62 * R0))
            a, b = pos[i], pos[(i + 1) % 3]
            s = np.linspace(0, 1, 70)[:, None]
            d = b - a
            nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
            wave = np.sin(2 * np.pi * (7 * s - 1.4 * t)) * np.sin(np.pi * s) * 0.05 * R0
            pts = a + s * d + wave * nrm
            set_rest(self.gluons[i], polyline_points(pts))


class Atom(Visual):
    def build(self):
        rng = np.random.default_rng(4)
        r = rng.gamma(3.0, 0.21 / 2, 600)
        v = rng.normal(size=(600, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        self.e_pos = v[:, :2] * r[:, None] * R0
        cloud = sprite("electron_cloud", D0)
        nuc_glow = sprite("glow", 0.5, "#FF8A5B", alpha=0.9)
        nucleus = sprite("star", 0.55, "#FFE6D6", spikes=0.25)
        self.electrons = [sprite("star", 0.42, "#C8F6FF", spikes=0.0) for _ in range(6)]
        return Group(cloud, nuc_glow, nucleus, *self.electrons)

    def animate(self, t, n):
        step = int(t / 0.085)
        for k, e in enumerate(self.electrons):
            p = self.e_pos[(step - k) % len(self.e_pos)]
            set_rest(e, quad(p[0], p[1], 0.42 * (1 - 0.1 * k)))
            e._ra = 0.95 * 0.55**k


class Virus(Visual):
    def build(self):
        rng = np.random.default_rng(9)
        br = 0.37 * D0
        glow = sprite("glow", D0 * 1.3, "#FF3F8E", alpha=0.45)
        spikes = VGroup()
        for i in range(30):
            a = 2 * np.pi * i / 30 + rng.normal(0, 0.04)
            L = 0.105 * D0 * rng.uniform(0.85, 1.1)
            u = np.array([math.cos(a), math.sin(a), 0])
            stem = Line(u * (br - 0.05), u * (br + L), stroke_width=5, color="#FF7DBA")
            club = Circle(radius=0.03 * D0, fill_color="#FFC2DF", fill_opacity=1, stroke_width=0).move_to(u * (br + L))
            spikes.add(stem, club)
        body = Circle(radius=br, fill_color="#3E0B2E", fill_opacity=1, stroke_color="#FF6FB0", stroke_width=3.5)
        texture = VGroup()
        pts = []
        while len(pts) < 42:
            p = rng.uniform(-0.88, 0.88, 2) * br
            if np.hypot(*p) < 0.88 * br and all(np.hypot(*(p - q)) > 0.2 for q in pts):
                pts.append(p)
        for p in pts:
            rr = rng.uniform(0.06, 0.13)
            texture.add(Circle(radius=rr, fill_color="#FF6FB0", fill_opacity=rng.uniform(0.12, 0.3),
                               stroke_width=0).move_to([p[0], p[1], 0]))
        shade = sprite("sphere_shade", 2 * br, blend="over")
        rim = sprite("rim_light", 2 * br * 1.02, "#FFB3D6", alpha=0.9)
        spec = sprite("glow", br * 0.9, "#FFFFFF", alpha=0.2, at=(-0.35 * br, 0.4 * br))
        self.spinners = [(spikes, 0.16), (texture, 0.16)]
        return Group(glow, spikes, body, texture, shade, rim, spec)


class RedBloodCell(Visual):
    def build(self):
        w, h = D0, 0.6 * D0
        glow = sprite("glow", D0 * 1.35, "#FF2440", alpha=0.35)
        outer = Ellipse(width=w, height=h, stroke_color="#FF8A98", stroke_width=2, stroke_opacity=0.6)
        outer.set_fill(["#FF6B7D", "#B3102F"], opacity=1)
        outer.set_sheen_direction(DOWN)
        dimple = VGroup()
        for i in range(7):
            k = i / 6
            e = Ellipse(width=w * mix(0.66, 0.4, k), height=h * mix(0.42, 0.22, k), stroke_width=0)
            e.set_fill("#6E0820", opacity=0.16).shift(DOWN * h * 0.04)
            dimple.add(e)
        hi = Ellipse(width=w * 0.94, height=h * 0.86)
        rim_hi = VMobject(stroke_color="#FFFFFF", stroke_width=5, stroke_opacity=0.32, fill_opacity=0)
        rim_hi.pointwise_become_partial(hi, 0.08, 0.42)
        dh = Ellipse(width=w * 0.6, height=h * 0.38).shift(DOWN * h * 0.04)
        inner_hi = VMobject(stroke_color="#FF9AAA", stroke_width=3.5, stroke_opacity=0.4, fill_opacity=0)
        inner_hi.pointwise_become_partial(dh, 0.6, 0.9)
        self.row = [Dot(radius=D0 / 150, color="#FF8FCB").move_to([-w / 2 + (i + 0.5) * w / 75, 0, 0])
                    for i in range(75)]
        return Group(glow, outer, dimple, rim_hi, inner_hi, *self.row)

    def animate(self, t, n):
        self.stagger(self.row, t, self.t_arrive + 0.3, 0.011)


class Ant(Visual):
    AMBER = "#FFB547"

    def build(self):
        U = D0
        glow = sprite("glow", D0 * 1.2, "#FF9A1F", alpha=0.25)

        def part(cx, cy, w, h, ang=0):
            e = Ellipse(width=w * U, height=h * U, stroke_color=self.AMBER, stroke_width=2.6)
            e.set_fill(["#B8741C", "#5A3208", "#2A1604"], opacity=1).set_sheen_direction(DOWN)
            return e.rotate(math.radians(ang)).move_to([cx * U, cy * U, 0])

        self.far_legs = [stroke_path([[0, 0], [0, 0.1], [0, 0.2]], "#9C6A25", 3.2) for _ in range(3)]
        self.near_legs = [stroke_path([[0, 0], [0, 0.1], [0, 0.2]], self.AMBER, 3.6) for _ in range(3)]
        body = VGroup(
            part(-0.30, 0.02, 0.36, 0.24, -8),
            part(-0.095, 0.03, 0.06, 0.07),
            part(0.04, 0.05, 0.26, 0.11, 12),
            part(0.26, 0.08, 0.17, 0.15, -10),
        )
        eye = Ellipse(width=0.045 * U, height=0.035 * U, fill_color="#FFD58A", fill_opacity=0.9,
                      stroke_width=0).move_to([0.29 * U, 0.10 * U, 0])
        mand = stroke_path(np.array([[0.33, 0.045], [0.39, 0.015], [0.36, -0.01]]) * U, self.AMBER, 2.6)
        ant1 = stroke_path(np.array([[0.30, 0.14], [0.36, 0.30], [0.47, 0.25]]) * U, self.AMBER, 2.4)
        ant2 = stroke_path(np.array([[0.28, 0.14], [0.31, 0.30], [0.44, 0.32]]) * U, "#9C6A25", 2.2)
        shadow = sprite("glow", D0 * 0.9, "#FFB547", alpha=0.18, width=D0 * 0.9, at=(0, -0.2 * U))
        shadow.stretch(0.12, 1)
        return Group(glow, shadow, ant2, *self.far_legs, body, eye, mand, ant1, *self.near_legs)

    def animate(self, t, n):
        U = D0
        hips = [(-0.03, 0.02), (0.04, 0.02), (0.11, 0.03)]
        knees = [(-0.11, 0.13), (0.02, 0.14), (0.15, 0.13)]
        feet = [(-0.17, -0.2), (0.04, -0.2), (0.24, -0.2)]
        for side, legs in ((0, self.near_legs), (1, self.far_legs)):
            for i, leg in enumerate(legs):
                phase = 2 * math.pi * 2.2 * t + (math.pi if (i % 2) ^ side else 0)
                dx = 0.035 * math.sin(phase)
                lift = 0.02 * max(0.0, math.cos(phase))
                off = 0.018 if side else 0.0
                hx, hy = hips[i]
                kx, ky = knees[i]
                fx, fy = feet[i]
                pts = np.array([[hx + off, hy], [kx + off + dx * 0.5, ky + lift], [fx + off + dx, fy + lift]]) * U
                set_rest(leg, polyline_points(pts))


class Human(Visual):
    SKIN = "#F4F1EA"

    def build(self):
        U = D0
        glow = sprite("glow", D0 * 1.25, "#38D9FF", alpha=0.3)
        floor = sprite("glow", D0 * 0.8, "#6CF2FF", alpha=0.35, at=(0, -0.5 * U))
        floor.stretch(0.1, 1)
        head = Circle(radius=0.085 * U, fill_color=self.SKIN, fill_opacity=1, stroke_width=0).move_to([0, 0.405 * U, 0])
        torso = RoundedRectangle(width=0.25 * U, height=0.34 * U, corner_radius=0.075 * U,
                                 fill_color=self.SKIN, fill_opacity=1, stroke_width=0).move_to([0, 0.17 * U, 0])
        legs = [stroke_path(np.array([[s * 0.068, 0.04], [s * 0.095, -0.45]]) * U, self.SKIN, 0.1 * U * 100)
                for s in (-1, 1)]
        arm_l = stroke_path(np.array([[-0.118, 0.29], [-0.175, 0.17], [-0.2, 0.05]]) * U, self.SKIN, 0.075 * U * 100)
        self.arm_r = stroke_path(np.array([[0.118, 0.29], [0.24, 0.33], [0.3, 0.5]]) * U, self.SKIN, 0.075 * U * 100)
        return Group(glow, floor, *legs, arm_l, self.arm_r, torso, head)

    def animate(self, t, n):
        U = D0
        ang = math.radians(72 + 16 * math.sin(2 * math.pi * 1.5 * (t - self.t_arrive)))
        el = np.array([0.235, 0.335])
        hand = el + 0.165 * np.array([math.cos(ang), math.sin(ang)])
        set_rest(self.arm_r, polyline_points(np.array([[0.118, 0.29], el, hand]) * U))


RIDGE = [(-0.95, -0.5), (-0.78, -0.30), (-0.68, -0.34), (-0.52, -0.10), (-0.42, -0.16), (-0.26, 0.16),
         (-0.16, 0.08), (EVEREST_SUMMIT_X, 0.5), (0.20, 0.20), (0.30, 0.27), (0.44, 0.02), (0.56, 0.08),
         (0.72, -0.22), (0.84, -0.17), (0.98, -0.5)]


class Everest(Visual):
    def build(self):
        U = D0
        glow = sprite("glow", D0 * 1.5, "#5C8DFF", alpha=0.3, at=(0, 0.1 * U))
        body = Polygon(*[np.array([x * U, y * U, 0]) for x, y in RIDGE], stroke_width=0)
        body.set_fill(["#DCEBFF", "#4E6E9E", "#0B1530"], opacity=1).set_sheen_direction(DOWN)
        snow = VGroup()
        shadow = VGroup()
        rng = np.random.default_rng(2)
        for i in range(1, len(RIDGE) - 1):
            (x0, y0), (x1, y1), (x2, y2) = RIDGE[i - 1], RIDGE[i], RIDGE[i + 1]
            if y1 > y0 and y1 > y2:  # a peak
                depth = 0.07 + 0.28 * max(y1 + 0.2, 0)
                lft = (x1 + (x0 - x1) * min(depth / (y1 - y0), 1), y1 - min(depth, y1 - y0))
                rgt = (x1 + (x2 - x1) * min(depth / (y1 - y2), 1), y1 - min(depth, y1 - y2))
                jag = [lft]
                for k in range(1, 6):
                    s = k / 6
                    jag.append((mix(lft[0], rgt[0], s), mix(lft[1], rgt[1], s) - rng.uniform(0.0, 0.06)))
                jag.append(rgt)
                snow.add(Polygon(*[np.array([x * U, y * U, 0]) for x, y in [(x1, y1)] + jag[::-1]],
                                 fill_color="#F4F8FF", fill_opacity=0.92, stroke_width=0))
                foot = (x1 + 0.3 * (x2 - x1), max(-0.5, y2 - 0.32))
                sh = [(x1, y1), (x2, y2), foot]
                shadow.add(Polygon(*[np.array([x * U, y * U, 0]) for x, y in sh],
                                   fill_color="#02060F", fill_opacity=0.36, stroke_width=0))
        ridge = stroke_path(np.array(RIDGE[1:-1]) * U, "#EAF4FF", 2.2, opacity=0.75)
        return Group(glow, body, snow, shadow, ridge)


# Rough coastlines (lon, lat) for the visible hemisphere of the Earth.
CONTINENTS = {
    "africa": [(-17, 21), (-16, 12), (-12, 7), (-8, 4.5), (-3, 5), (5, 6), (9, 4), (9.5, 2), (9, -1), (12, -5),
               (13, -11), (12, -17), (15, -27), (18, -34.5), (22, -34), (27, -33.5), (32, -29), (35, -24),
               (35.5, -19), (40, -15), (40.5, -11), (39, -7), (40, -3), (43, 2), (51, 11.5), (44, 11),
               (43, 12.5), (39, 16), (37, 20), (35, 24), (34, 28), (32.5, 31), (29, 31), (25, 32), (20, 32),
               (19, 30.5), (15.5, 31.5), (11, 33), (10.5, 37), (9.5, 37.3), (3, 36.8), (-2, 35), (-5.5, 35.8),
               (-9.5, 32), (-10, 29), (-13, 27.5)],
    "madagascar": [(44, -25), (47, -25), (50, -15.5), (49.5, -12), (48, -13.5), (44, -17), (43.5, -22)],
    "eurasia": [(-9, 37), (-9.5, 43), (-2, 43.5), (-1, 46), (-4.5, 48.5), (1.5, 50), (4, 51.5), (8, 54),
                (8.5, 57), (10.5, 57.5), (12, 56), (10.5, 59), (5, 58.5), (5, 62), (10, 64), (14, 67.5),
                (18, 69.5), (25, 71), (31, 70), (41, 67), (44, 68.5), (53, 68.5), (60, 69.5), (69, 73),
                (80, 73), (90, 75.5), (100, 77), (113, 73.5), (130, 71), (140, 72), (160, 70), (160, 60),
                (142, 59), (138, 54), (141, 48), (135, 43), (130, 42.5), (129, 35), (126, 37), (122, 40),
                (118, 38), (122, 31), (120, 25), (113, 22), (108, 21.5), (106, 18), (109, 12), (105, 8.6),
                (103, 10.5), (100, 13.5), (100, 6.5), (103.5, 1.5), (100.5, 3.5), (98, 8), (98, 16),
                (94, 17), (92, 22), (90, 22), (87, 21.5), (80, 15.5), (80, 10), (77.5, 8), (76, 10.5),
                (73, 17), (72.5, 21), (68.5, 23.5), (66.5, 25.4), (61.5, 25.2), (57, 25.7), (56.3, 27),
                (51.5, 27.9), (50, 30), (48, 30), (48.5, 28.5), (50.5, 26), (51.6, 24.5), (56.3, 26.2),
                (59.5, 22.5), (57, 18.5), (52, 16), (45, 13), (43, 12.7), (43, 16.5), (39, 21.5), (35, 28),
                (34.3, 28), (33, 31), (35.5, 33.5), (36, 36.5), (30, 36.5), (27, 37), (26, 40), (23, 40.5),
                (24, 38), (22, 36.5), (21, 39), (19.5, 41.5), (16, 43.5), (13.5, 45.5), (12.5, 44), (16, 41),
                (18.5, 40), (16, 38), (15.6, 38), (12, 42), (10, 44), (7, 43.5), (3, 43.3), (3, 41.8),
                (0, 39.5), (-0.5, 38), (-2, 36.7), (-5.5, 36), (-6.5, 37)],
    "britain": [(-5.5, 50), (1.5, 51), (1.7, 52.8), (0, 53.5), (-1.5, 55), (-2, 57.6), (-3.3, 58.6),
                (-5, 58.6), (-6.2, 56.5), (-4.8, 54.8), (-3, 53.4), (-4.5, 52.8), (-5, 51.6)],
    "ireland": [(-10, 51.5), (-6, 52), (-6, 54.5), (-8, 55.3), (-10, 54)],
    "sumatra": [(95.5, 5.5), (98, 4), (104, -2.5), (106, -6), (102, -4), (100, -1), (96, 3)],
    "borneo": [(109, 1.5), (117, 7), (119, 5), (116, -4), (110, -3)],
    "srilanka": [(80, 6), (82, 7.5), (80.5, 9.7), (79.8, 8)],
    "australia": [(113, -22), (114, -26), (115, -34), (118, -35), (124, -33.5), (129, -31.6), (134, -32.5),
                  (138, -35), (140, -38), (146, -39), (150, -37.5), (153, -31), (153, -25), (146, -19),
                  (142.5, -10.7), (141.5, -17), (136, -15), (137, -12), (130, -11.2), (126, -14), (122, -17)],
}
LAKES = {
    "black": [(28, 41.3), (28, 44), (30.5, 46.5), (33.5, 44.5), (37, 45), (41.5, 41.5), (36, 41.7), (32, 41.2)],
    "caspian": [(47, 44.5), (50, 46.5), (53, 47), (54, 45), (51, 43), (52.5, 41.5), (54, 40), (53.5, 37),
                (50.5, 37), (49, 38.5), (49.5, 40.5), (47.5, 43)],
}


def ortho(lon, lat, lon0=45.0, lat0=18.0):
    lon, lat = np.radians(lon), np.radians(lat)
    l0, p0 = np.radians(lon0), np.radians(lat0)
    x = np.cos(lat) * np.sin(lon - l0)
    y = np.cos(p0) * np.sin(lat) - np.sin(p0) * np.cos(lat) * np.cos(lon - l0)
    vis = np.sin(p0) * np.sin(lat) + np.cos(p0) * np.cos(lat) * np.cos(lon - l0)
    if vis < 0:  # behind the horizon: push onto the limb
        r = math.hypot(x, y) + 1e-9
        x, y = x / r, y / r
    return x, y


def _densify(poly, step=2.0):
    out = []
    for (a, b), (c, d) in zip(poly, poly[1:] + poly[:1]):
        k = max(1, int(max(abs(c - a), abs(d - b)) / step))
        for i in range(k):
            out.append((a + (c - a) * i / k, b + (d - b) * i / k))
    return out


class Earth(Visual):
    fade_out = (7.4, 8.6)  # stays visible beside the Sun

    def build(self):
        R = R0
        atmo = sprite("ring", D0 * 1.3, "#4FB6FF", alpha=0.75, r0=0.77, inner=0.015, outer=0.09)
        glow = sprite("glow", D0 * 1.5, "#2E7BFF", alpha=0.35)
        ocean = Circle(radius=R, stroke_width=0)
        ocean.set_fill(["#2C8CF0", "#0B3A8C", "#061B4A"], opacity=1).set_sheen_direction(np.array([1, -1, 0]))
        land = VGroup()
        for name, poly in CONTINENTS.items():
            pts = [ortho(lo, la) for lo, la in _densify(poly)]
            v = blob([(x * R, y * R) for x, y in pts], "#3DBE7A", 0.95, smooth_it=False)
            v.set_fill(["#7FD88A", "#2E9A5C", "#1F6E46"], opacity=0.96).set_sheen_direction(np.array([1, -1, 0]))
            land.add(v)
        for name, poly in LAKES.items():
            pts = [ortho(lo, la) for lo, la in _densify(poly)]
            land.add(blob([(x * R, y * R) for x, y in pts], "#164E9E", 1.0, smooth_it=False))
        ice = blob([(x * R, y * R) for x, y in [ortho(lo, 76 + 3 * math.sin(lo / 9)) for lo in range(-180, 180, 6)]],
                   "#EEF6FF", 0.85, smooth_it=False)
        rng = np.random.default_rng(12)
        clouds = VGroup()
        for _ in range(9):
            c = rng.uniform(-0.6, 0.6, 2) * R
            e = Ellipse(width=rng.uniform(0.5, 1.2), height=rng.uniform(0.08, 0.16), stroke_width=0)
            e.set_fill("#FFFFFF", opacity=0.16).rotate(rng.uniform(-0.4, 0.4)).move_to([c[0], c[1], 0])
            clouds.add(e)
        shade = sprite("sphere_shade", D0, blend="over", strength=0.9)
        spec = sprite("glow", R * 0.9, "#FFFFFF", alpha=0.16, at=(-0.38 * R, 0.42 * R))
        rim = sprite("rim_light", D0 * 1.02, "#9ED8FF", alpha=0.8)
        return Group(glow, atmo, ocean, land, ice, clouds, shade, spec, rim)


class Sun(Visual):
    def build(self):
        corona = sprite("glow", D0 * 2.5, "#FF7A14", alpha=0.75)
        inner = sprite("glow", D0 * 1.45, "#FFC86B", alpha=0.65, sigma=0.4)
        disc = sprite("sun_disc", D0, blend="over")
        bloom = sprite("glow", D0 * 1.12, "#FFF1C9", alpha=0.22)
        self.row = [Dot(radius=D0 / 218 * 0.88, color="#8CCBFF").move_to([-R0 + (i + 0.5) * D0 / 109, 0, 0])
                    for i in range(109)]
        return Group(corona, inner, disc, bloom, *self.row)

    def animate(self, t, n):
        self.stagger(self.row, t, self.t_arrive + 0.25, 0.0062)


PLANETS = [  # (name, AU, radius px at 1080p, colour, start angle)
    ("Mercury", 0.387, 3, "#B9B2A8", 2.2), ("Venus", 0.723, 4, "#F0D9A8", 4.1), ("Earth", 1.0, 4.5, "#4DA3FF", 0.0),
    ("Mars", 1.524, 3.5, "#FF7A50", 1.2), ("Jupiter", 5.2, 8, "#E8C9A0", 2.6), ("Saturn", 9.54, 7, "#F2DDA8", 4.9),
    ("Uranus", 19.2, 5.5, "#A8F0F2", 1.4), ("Neptune", 30.07, 5.5, "#6F8CFF", 5.6),
]
TILT = 0.42


class SolarSystem(Visual):
    def build(self):
        self.orbits = VGroup()
        self.planets = []
        self.labels = []
        for name, au, rpx, col, a0 in PLANETS:
            a = R0 * au / 30.07
            e = Ellipse(width=2 * a, height=2 * a * TILT, stroke_color="#8FB8FF", fill_opacity=0,
                        stroke_width=1.6 if name != "Neptune" else 2.0, stroke_opacity=0.32 if name != "Earth" else 0.6)
            self.orbits.add(e)
            dot = Dot(radius=rpx / 120, color=col)
            self.planets.append((dot, a, au, a0, name))
            if name in ("Uranus", "Neptune"):
                self.labels.append((text(name, 17, color="#AFC0E0"), name))
        ring = Ellipse(width=0.17, height=0.05, stroke_color="#F2DDA8", stroke_width=1.6, fill_opacity=0)
        self.saturn_ring = ring
        belts = sprite("belts", D0 * 1.6, "#FFFFFF", alpha=0.32)
        sun = sprite("star", 0.75, "#FFE7A8", spikes=0.25)
        sun_glow = sprite("glow", 1.1, "#FFB13B", alpha=0.8)
        return Group(belts, self.orbits, sun_glow, sun, *[p[0] for p in self.planets], ring,
                     *[lab for lab, _ in self.labels])

    def animate(self, t, n):
        pos = {}
        for dot, a, au, a0, name in self.planets:
            ang = a0 if name == "Earth" else a0 + 0.55 * au**-0.75 * t
            pos[name] = np.array([a * math.cos(ang), a * TILT * math.sin(ang), 0])
            move_rest(dot, pos[name])
        move_rest(self.saturn_ring, pos["Saturn"])
        for lab, name in self.labels:
            move_rest(lab, pos[name] + np.array([0, -0.2, 0]))


class NearestStar(Visual):
    def build(self):
        sx, px = -R0, R0
        sun_glow = sprite("glow", 1.0, "#FFC56B", alpha=0.8, at=(sx, 0))
        sun = sprite("star", 0.9, "#FFF2D0", spikes=0.35, at=(sx, 0))
        prox_glow = sprite("glow", 0.9, "#FF5A3C", alpha=0.85, at=(px, 0))
        prox = sprite("star", 0.75, "#FFB49C", spikes=0.35, at=(px, 0))
        ab = [sprite("star", 0.62, "#FFF0C8", spikes=0.3, at=(px - 0.2, 0.22)),
              sprite("star", 0.5, "#FFD9A0", spikes=0.3, at=(px - 0.12, 0.27))]
        line = DashedLine([sx + 0.25, 0, 0], [px - 0.25, 0, 0], dash_length=0.12, stroke_width=2.2,
                          color="#FFFFFF", stroke_opacity=0.5)
        l1 = text("Sun", 20, color="#C9D1E6").move_to([sx, -0.42, 0])
        l2 = text("Proxima Centauri", 20, color="#C9D1E6").move_to([px - 0.55, -0.42, 0])
        return Group(sun_glow, prox_glow, line, sun, prox, *ab, l1, l2)


class MilkyWay(Visual):
    fade_out = (4.0, 6.5)

    def build(self):
        halo = sprite("glow", D0 * 1.6, "#6A4CFF", alpha=0.25)
        gal = sprite("galaxy", D0 * 1.15)
        return Group(halo, gal)

    def angle(self, t):
        return -0.035 * (t - self.t_arrive)


class Universe(Visual):
    fade_out = (50, 60)

    def build(self):
        web = sprite("cosmic_web", D0 * 1.1)
        home = sprite("star", 0.35, "#E2D4FF", spikes=0.0)
        return Group(web, home)


ORDER = [Planck, Proton, Atom, Virus, RedBloodCell, Ant, Human, Everest, Earth, Sun, SolarSystem,
         NearestStar, MilkyWay, Universe]

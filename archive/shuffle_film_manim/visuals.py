"""The Shuffle's drawings: the deck, the ring of 52 card slots, the three reference
drawings (everyone alive, time since the Big Bang, the Planck time), the orders dust,
the rows and the points of light that gather into each arrival.

Counts are not sizes, so nothing here zooms by scale.  Each drawing is posed every frame
from absolute time and ``n`` (cards dealt) by ``film.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from manim import Circle, DashedVMobject, Line, RoundedRectangle, Square, VGroup, VMobject
from manim.constants import CapStyleType, LineJointType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from doubling_film import assets  # noqa: E402
from doubling_film.engine import Sprite, Texture, clamp01, ease_in_out, ease_out, hex_rgb, mix, smooth  # noqa: E402

import textures  # noqa: E402

Y0 = 0.22  # where the featured object sits
D0 = 5.2  # the reference width (the row's track, the nebula)
RING_R = 2.55  # the ring of 52 card slots (2.7 ran into the label block's chip)
SLOT_W, SLOT_H = 0.19, 0.27
DECK_H = 2.9
CARD_ASPECT = 63 / 88

_TEX: dict = {}


def tex(kind, **kw):
    key = (kind, tuple(sorted(kw.items())))
    if key not in _TEX:
        fn = getattr(textures, kind, None) or getattr(assets, kind)
        _TEX[key] = Texture(*fn(**kw))
    return _TEX[key]


def sprite(kind, height, color="#FFFFFF", alpha=1.0, blend="screen", at=(0.0, 0.0), width=None, **kw):
    s = Sprite(tex(kind, **kw), height, width=width, blend=blend, alpha=alpha, tint=hex_rgb(color))
    s.shift([at[0], at[1], 0])
    return s


def quad(x, y, h, w=None, angle=0.0):
    w = h if w is None else w
    c, s = math.cos(angle), math.sin(angle)
    pts = np.array([[-w / 2, h / 2], [w / 2, h / 2], [-w / 2, -h / 2], [w / 2, -h / 2]])
    pts = pts @ np.array([[c, s], [-s, c]])
    return np.c_[pts[:, 0] + x, pts[:, 1] + y, np.zeros(4)]


def polyline_points(p):
    """Straight-segment Bezier points for a polyline (N, 2|3) -> (4(N-1), 3)."""
    p = np.asarray(p, float)
    if p.shape[1] == 2:
        p = np.hstack([p, np.zeros((len(p), 1))])
    a, b = p[:-1], p[1:]
    out = np.empty((len(a) * 4, 3))
    out[0::4], out[1::4], out[2::4], out[3::4] = a, a + (b - a) / 3, a + 2 * (b - a) / 3, b
    return out


def line_mob(color, width, opacity=1.0):
    v = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity, fill_opacity=0)
    v.points = polyline_points([[0, 0], [1e-3, 0]])
    v.set_cap_style(CapStyleType.ROUND)
    v.joint_type = LineJointType.ROUND
    return v


def set_alpha(m, a):
    m.stroke_rgbas[:, 3] = a


def set_rgb(m, col):
    m.stroke_rgbas[:, :3] = col
    if len(m.fill_rgbas):
        m.fill_rgbas[:, :3] = col


# ---------------------------------------------------------------------------
# The deck
# ---------------------------------------------------------------------------


class Deck:
    """A full deck: the top card (line art, lit from within) over stepped edges.  The edges
    thin as cards are dealt, and the top card leaves with the last one."""

    def __init__(self, height=DECK_H, accent="#B9A6FF", depth=4):
        self.h, self.w = height, height * CARD_ASPECT
        self.accent = accent
        self.halo = sprite("glow", D0 * 1.15, "#8F6BFF", alpha=0.5)
        self.edges = []
        for k in range(depth, 0, -1):
            e = RoundedRectangle(width=self.w, height=self.h, corner_radius=0.06 * height, stroke_color=accent,
                                 stroke_width=1.6, stroke_opacity=0.25 + 0.1 * (depth - k), fill_color="#0C0920",
                                 fill_opacity=0.92)
            e._k, e._off = k, np.array([0.06 * k, -0.06 * k, 0.0])
            e._unit = e.points.copy()
            self.edges.append(e)
        self.body = RoundedRectangle(width=self.w, height=self.h, corner_radius=0.06 * height, stroke_color=accent,
                                     stroke_width=3.2, fill_color="#0C0920", fill_opacity=0.92)
        inset = 0.055 * height
        self.inner = RoundedRectangle(width=self.w - 2 * inset, height=self.h - 2 * inset, corner_radius=0.04 * height,
                                      stroke_color=accent, stroke_width=1.4, stroke_opacity=0.45, fill_opacity=0)
        self.diamond = Square(side_length=0.2 * height, stroke_color=accent, stroke_width=2.2, fill_opacity=0)
        self.diamond.rotate(math.pi / 4)
        self.pip = sprite("star", 0.3 * height, "#E4DCFF", spikes=0.0)
        self.parts = [self.body, self.inner, self.diamond]
        for m in self.parts:
            m._unit = m.points.copy()
            m._sa = m.stroke_rgbas[:, 3].copy()
            m._fa = m.fill_rgbas[:, 3].copy()
        for e in self.edges:
            e._sa = e.stroke_rgbas[:, 3].copy()
            e._fa = e.fill_rgbas[:, 3].copy()
        self.mobs = [self.halo, *self.edges, self.body, self.inner, self.diamond, self.pip]

    def pose(self, remaining, center=(0.0, Y0), alpha=1.0, scale=1.0, halo=1.0):
        """``remaining`` cards left in the deck (52 − n); ``alpha`` fades the cards, ``halo`` the glow."""
        c = np.array([center[0], center[1], 0.0])
        self.halo.points = quad(c[0], c[1], D0 * 1.15 * scale)
        self.halo.alpha = 0.5 * halo
        for e in self.edges:
            k = e._k
            a = alpha * smooth((k - 1) * 13 + 2, (k - 1) * 13 + 7, remaining)
            e.points = e._unit * scale + c + e._off * scale
            e.stroke_rgbas[:, 3] = e._sa * a
            e.fill_rgbas[:, 3] = e._fa * a
        top = alpha * smooth(0.0, 0.9, remaining)
        for m in self.parts:
            m.points = m._unit * scale + c
            m.stroke_rgbas[:, 3] = m._sa * top
            m.fill_rgbas[:, 3] = m._fa * top
            m.stroke_width = {self.body: 3.2, self.inner: 1.4, self.diamond: 2.2}[m] * scale
        self.pip.points = quad(c[0], c[1], 0.3 * self.h * scale)
        self.pip.alpha = top


# ---------------------------------------------------------------------------
# The ring of 52 slots
# ---------------------------------------------------------------------------


class CardRing:
    """52 card slots clockwise from the top.  Slot ``i`` fills when card ``i + 1`` is dealt:
    the card flies out of the deck, shrinking into its slot."""

    def __init__(self, deck: Deck):
        self.deck = deck
        self.slots, self.sockets, self.glows, self.slot_rest, self.deck_rest, self.angles = [], [], [], [], [], []
        for i in range(52):
            a = math.pi / 2 - 2 * math.pi * (i + 0.5) / 52
            x, y = RING_R * math.cos(a), Y0 + RING_R * math.sin(a)
            slot = RoundedRectangle(width=SLOT_W, height=SLOT_H, corner_radius=0.035, stroke_color="#B9A6FF",
                                    stroke_width=1.1, stroke_opacity=1.0, fill_color="#B9A6FF", fill_opacity=1.0)
            slot.rotate(a - math.pi / 2).move_to([x, y, 0])
            rest = slot.points.copy()
            # the same shape as it sits in the deck: card-sized, unrotated, at the deck's centre
            big = RoundedRectangle(width=deck.w * 0.5, height=deck.h * 0.5, corner_radius=0.03 * deck.h,
                                   stroke_width=1.1).move_to([0, Y0, 0])
            self.slots.append(slot)
            sock = RoundedRectangle(width=SLOT_W, height=SLOT_H, corner_radius=0.035, stroke_color="#B9A6FF",
                                    stroke_width=1.1, stroke_opacity=1.0, fill_opacity=0.0)
            sock.points = rest.copy()
            self.sockets.append(sock)
            self.slot_rest.append(rest)
            self.deck_rest.append(big.points.copy())
            self.angles.append(a)
            self.glows.append(sprite("glow", 0.55, "#B9A6FF", alpha=0.35, at=(x, y)))
            self.glows[-1]._at = (x, y)
        self.mobs = [*self.sockets, *self.glows, *self.slots]

    def pose(self, n, alpha=1.0, color=None, empty_alpha=0.3, gather=0.0):
        """``gather`` 1 spirals every card into the centre at slot size, fading (the outro).  An
        empty socket stays on the ring until its card has landed; a card fades in as it leaves the
        deck, so nothing pops."""
        col = hex_rgb(color) if isinstance(color, str) else (np.asarray(color) if color is not None else hex_rgb("#B9A6FF"))
        centre = np.array([0.0, Y0, 0.0])
        for i, (slot, sock, glow) in enumerate(zip(self.slots, self.sockets, self.glows)):
            dealt = n - i  # >= 1: in its slot; 0..1: flying out of the deck; <= 0: an empty socket
            gi = ease_in_out(clamp01(gather * 1.35 - 0.35 * i / 51)) if gather > 0 else 0.0
            on = smooth(0.0, 0.35, dealt) if dealt > 1e-6 else 0.0
            set_rgb(sock, col)
            sock.stroke_rgbas[:, 3] = alpha * empty_alpha * (1 - on) * (1 - gi)
            if dealt <= 1e-6:
                slot.stroke_rgbas[:, 3] = 0.0
                slot.fill_rgbas[:, 3] = 0.0
                glow.alpha = 0.0
                continue
            p = ease_out(min(dealt, 1.0))
            pts = mix(self.deck_rest[i], self.slot_rest[i], p)
            a = alpha * (1 - gi) * (smooth(0.0, 0.12, dealt) if dealt < 1.0 else 1.0)
            sc = self.slot_rest[i].mean(axis=0)
            if gi > 0:
                pts = (pts - sc) * (1 - 0.85 * gi) + sc + (centre - sc) * gi
            slot.points = pts
            set_rgb(slot, col)
            slot.stroke_rgbas[:, 3] = a * 0.95
            slot.fill_rgbas[:, 3] = a * 0.28 * max(on, 0.5)
            slot.stroke_width = mix(1.1, 1.8, on)
            if on > 0.003 and a > 0.003:
                gc = sc + (centre - sc) * gi
                glow.points = quad(gc[0], gc[1], 0.55 * (1 - 0.85 * gi))
                glow.tint = col
                glow.alpha = 0.35 * on * a * (1 if p >= 1 else p)
            else:
                glow.alpha = 0.0


# ---------------------------------------------------------------------------
# Reference drawings, each gathering from points of light
# ---------------------------------------------------------------------------


class Drawing:
    """A drawing inside the ring.  ``pose(center, alpha, gather, t)``: ``gather`` runs 0 → 1 as
    it arrives (points of light converge, the drawing brightens) and back to 0 as it leaves."""

    RADIUS = 1.5
    POINTS = 36

    def __init__(self, accent, seed=1):
        self.accent = accent
        rng = np.random.default_rng(seed)
        self.pt_dir = rng.uniform(0, 2 * math.pi, self.POINTS)
        self.pt_r = self.RADIUS * np.sqrt(rng.uniform(0.05, 1.0, self.POINTS))
        self.pt_th = rng.uniform(0, 2 * math.pi, self.POINTS)
        self.pt_far = rng.uniform(2.6, 3.6, self.POINTS)
        self.pt_size = rng.uniform(0.16, 0.3, self.POINTS)
        self.points = [sprite("star", 0.2, accent, alpha=0.0, spikes=0.0) for _ in range(self.POINTS)]
        self.mobs = []
        self.build()
        self.mobs = self.mobs + self.points

    def build(self):
        raise NotImplementedError

    def pose_points(self, c, gather):
        g = gather
        for k, s in enumerate(self.points):
            tx = c[0] + self.pt_r[k] * math.cos(self.pt_th[k])
            ty = c[1] + self.pt_r[k] * math.sin(self.pt_th[k])
            d = self.pt_far[k] * (1 - ease_out(g))
            x, y = tx + d * math.cos(self.pt_dir[k]), ty + d * math.sin(self.pt_dir[k])
            s.points = quad(x, y, self.pt_size[k])
            s.alpha = 0.95 * math.sin(math.pi * min(g * 1.08, 1.0)) ** 0.8 if 0 < g < 1 else 0.0

    def body_alpha(self, gather):
        return smooth(0.3, 1.0, gather)


class Crowd(Drawing):
    """Everyone alive: a disc of cyan points of light."""

    def build(self):
        self.glow = sprite("glow", 3.4, self.accent, alpha=0.3)
        self.disc = sprite("point_disc", 2 * self.RADIUS, self.accent, alpha=0.95)
        self.mobs = [self.glow, self.disc]

    def pose(self, c, alpha, gather, t):
        a = alpha * self.body_alpha(gather)
        self.glow.points = quad(c[0], c[1], 3.4)
        self.glow.alpha = 0.3 * a
        breathe = 1 + 0.012 * math.sin(2 * math.pi * 0.35 * t)
        self.disc.points = quad(c[0], c[1], 2 * self.RADIUS * breathe)
        self.disc.alpha = 0.95 * a
        self.pose_points(c, gather if alpha > 0.003 else 0.0)


class Spiral(Drawing):
    """Time since the Big Bang: a lime spiral unwinding from a bright point to now."""

    RADIUS = 1.45

    def build(self):
        self.glow = sprite("glow", 3.2, self.accent, alpha=0.4)
        self.arm = sprite("spiral", 2 * self.RADIUS * 1.08, self.accent, alpha=0.9)
        self.core = sprite("star", 0.85, "#F4FFD6", alpha=1.0, spikes=0.35)
        self.now = sprite("star", 0.32, "#FFFFFF", alpha=1.0, spikes=0.0)
        self.mobs = [self.glow, self.arm, self.core, self.now]

    def pose(self, c, alpha, gather, t):
        a = alpha * self.body_alpha(gather)
        ang = 0.06 * t
        self.glow.points = quad(c[0], c[1], 3.2)
        self.glow.alpha = 0.4 * a
        self.arm.points = quad(c[0], c[1], 2 * self.RADIUS * 1.08, angle=ang)
        self.arm.alpha = 0.9 * a
        self.core.points = quad(c[0], c[1], 0.85 * (1 + 0.05 * math.sin(2 * math.pi * 0.9 * t)))
        self.core.alpha = a
        # "now" sits at the end of the spiral: 2.4 turns from the top, radius 0.92 of the arm
        th = 2 * math.pi * 2.4 + math.pi / 2 - ang
        r = 0.92 * self.RADIUS * 1.08
        self.now.points = quad(c[0] + r * math.cos(th), c[1] + r * math.sin(th), 0.32)
        self.now.alpha = a
        self.pose_points(c, gather if alpha > 0.003 else 0.0)


class PlanckDot(Drawing):
    """The Planck time: the point inside The Doubling's dashed reticle."""

    RADIUS = 1.3

    def build(self):
        self.glow = sprite("glow", 2.6, "#8F6BFF", alpha=0.5)
        circ = DashedVMobject(Circle(radius=self.RADIUS, stroke_width=1.6, color=self.accent), num_dashes=70)
        ticks = [Line([self.RADIUS - 0.1, 0, 0], [self.RADIUS + 0.1, 0, 0], stroke_width=1.8, color=self.accent)
                 .rotate(a, about_point=np.zeros(3)) for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)]
        self.ret = [*circ.submobjects, *ticks]
        for m in self.ret:
            m._unit = m.points.copy()
        self.dot = sprite("star", 0.3, "#E4DCFF", alpha=1.0, spikes=0.0)
        self.mobs = [self.glow, *self.ret, self.dot]

    def pose(self, c, alpha, gather, t):
        a = alpha * self.body_alpha(gather)
        cc = np.array([c[0], c[1], 0.0])
        self.glow.points = quad(c[0], c[1], 2.6)
        self.glow.alpha = 0.5 * a
        for m in self.ret:
            m.points = m._unit + cc
            m.stroke_rgbas[:, 3] = 0.8 * a
        self.dot.points = quad(c[0], c[1], 0.3 * (1 + 0.08 * math.sin(2 * math.pi * 1.1 * t)))
        self.dot.alpha = a
        self.pose_points(c, gather if alpha > 0.003 else 0.0)


DRAWINGS = {"crowd": Crowd, "spiral": Spiral, "planck": PlanckDot}


class Dust:
    """The orders: a faint lavender dust behind everything inside the ring, thickening as
    the count climbs and turning violet when the deck is whole."""

    def __init__(self):
        self.a = sprite("dust", 4.5, "#B9A6FF", alpha=0.0)
        self.b = sprite("dust", 3.6, "#B9A6FF", alpha=0.0, seed=7)
        self.mobs = [self.a, self.b]

    def pose(self, c, strength, t, tint="#B9A6FF", alpha=1.0):
        col = hex_rgb(tint) if isinstance(tint, str) else tint
        for s, size, w, k in ((self.a, 4.5, 0.028, 1.0), (self.b, 3.6, -0.041, 0.7)):
            s.points = quad(c[0], c[1], size, angle=w * t)
            s.tint = col
            s.alpha = 0.55 * strength * k * alpha


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


class RowLine:
    """A glowing line segment (10 px glow at 22%, 3.2 px core) with endpoints set per frame."""

    def __init__(self, color):
        self.glow = line_mob(color, 10, 0.22)
        self.core = line_mob(color, 3.2, 0.95)
        self.mobs = [self.glow, self.core]

    def set(self, x0, x1, y, alpha, color=None):
        pts = polyline_points([[x0, y], [max(x1, x0 + 1e-3), y]])
        for m, base in ((self.glow, 0.22), (self.core, 0.95)):
            m.points = pts
            m.stroke_rgbas[:, 3] = base * alpha
            if color is not None:
                m.stroke_rgbas[:, :3] = hex_rgb(color) if isinstance(color, str) else color

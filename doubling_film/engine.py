"""Rendering helpers: glow sprites, a faster camera, rigs and animated text.

Manim's Cairo camera composites every ImageMobject over the *whole* frame,
which is far too slow for dozens of glows per frame at 1080x1920.  ``Sprite``
plus ``FilmCamera`` composite only the pixels a sprite covers, add light with a
"screen" blend (so glows add up like real light), pick a mip level so big
textures don't shimmer when they shrink, and finish each frame with a
vignette (film grain is added at encode time, see render.sh).
"""

from __future__ import annotations

import math

import numpy as np
from manim import (
    BOLD,
    DOWN,
    ImageMobject,
    MEDIUM,
    NORMAL,
    Text,
    VGroup,
    VMobject,
)
from manim.camera.camera import Camera
from manim.constants import LineJointType
from PIL import Image

INTER = "Inter"
INTER_DISPLAY = "Inter Display"
MONO = "JetBrains Mono"


# ---------------------------------------------------------------------------
# Small maths helpers
# ---------------------------------------------------------------------------


def clamp01(x):
    return min(1.0, max(0.0, x))


def smooth(e0, e1, x):
    """Smoothstep from 0 at ``e0`` to 1 at ``e1`` (works for e0 > e1 too)."""
    if e0 == e1:
        return 1.0 if x >= e1 else 0.0
    t = clamp01((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = clamp01(t)
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def mix(a, b, t):
    return a + (b - a) * t


def rot2(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Sprites
# ---------------------------------------------------------------------------


class Texture:
    """Mip pyramid of a premultiplied texture, shareable between sprites.

    ``rgb`` is HxWx3 premultiplied colour and ``alpha`` HxW coverage, both
    floats in [0, 1].
    """

    def __init__(self, rgb, alpha):
        rgb8 = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
        a8 = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
        col, alp = Image.fromarray(rgb8, "RGB"), Image.fromarray(a8, "L")
        self.aspect = col.size[0] / col.size[1]
        self.levels = [(col, alp)]
        while min(col.size) > 8:
            col, alp = col.reduce(2), alp.reduce(2)
            self.levels.append((col, alp))


class Sprite(ImageMobject):
    """A textured quad drawn by ``FilmCamera``.  ``blend`` is ``"screen"``
    (adds light) or ``"over"``; ``tint`` multiplies the colour."""

    def __init__(self, tex: Texture, height, width=None, blend="screen", alpha=1.0, tint=None):
        super().__init__(np.zeros((2, 2, 4), dtype=np.uint8))
        if width is None:
            width = height * tex.aspect
        self.levels = tex.levels
        self.points = np.array(
            [
                [-width / 2, height / 2, 0],
                [width / 2, height / 2, 0],
                [-width / 2, -height / 2, 0],
                [width / 2, -height / 2, 0],
            ]
        )
        self.blend = blend
        self.alpha = alpha
        self.tint = np.ones(3) if tint is None else np.asarray(tint, float)


class PostFX(ImageMobject):
    """Placed last in the scene: the camera applies vignette and grain."""

    def __init__(self):
        super().__init__(np.zeros((2, 2, 4), dtype=np.uint8))
        self.points = np.zeros((4, 3))
        self.frame = 0


class FilmCamera(Camera):
    # Grain is added at encode time (render.sh); per-frame noise baked in here
    # makes files huge and turns to mush after TikTok/Instagram re-encoding.
    grain_strength = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vignette = None
        self._grain = None

    # -- sprites ------------------------------------------------------------
    def _to_px(self, pts):
        fw, fh = self.frame_width, self.frame_height
        pw, ph = self.pixel_width, self.pixel_height
        c = self.frame_center
        x = (pts[:, 0] - c[0]) * (pw / fw) + pw / 2
        y = -(pts[:, 1] - c[1]) * (ph / fh) + ph / 2
        return np.stack([x, y], axis=1)

    def display_image_mobject(self, image_mobject, pixel_array):
        if isinstance(image_mobject, PostFX):
            self._post(image_mobject, pixel_array)
            return
        if not isinstance(image_mobject, Sprite):
            return super().display_image_mobject(image_mobject, pixel_array)
        sp = image_mobject
        if sp.alpha <= 0.002:
            return
        ul, ur, dl = self._to_px(sp.points[:3])
        right, down = ur - ul, dl - ul
        corners = np.array([ul, ur, dl, ul + right + down])
        ph, pw = pixel_array.shape[:2]
        x0 = max(int(math.floor(corners[:, 0].min())), 0)
        x1 = min(int(math.ceil(corners[:, 0].max())), pw)
        y0 = max(int(math.floor(corners[:, 1].min())), 0)
        y1 = min(int(math.ceil(corners[:, 1].max())), ph)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return
        on_screen_w = max(np.hypot(*right), 1e-6)
        w0 = sp.levels[0][0].size[0]
        lvl = int(np.clip(math.floor(math.log2(max(w0 / on_screen_w, 1.0))), 0, len(sp.levels) - 1))
        col, alp = sp.levels[lvl]
        wk, hk = col.size
        m = np.array([[right[0], down[0]], [right[1], down[1]]])
        det = np.linalg.det(m)
        if abs(det) < 1e-9:
            return
        mi = np.linalg.inv(m)
        ox, oy = x0 - ul[0], y0 - ul[1]
        data = (
            wk * mi[0, 0], wk * mi[0, 1], wk * (mi[0, 0] * ox + mi[0, 1] * oy),
            hk * mi[1, 0], hk * mi[1, 1], hk * (mi[1, 0] * ox + mi[1, 1] * oy),
        )
        size = (x1 - x0, y1 - y0)
        c = np.asarray(col.transform(size, Image.AFFINE, data, resample=Image.BILINEAR), np.float32)
        a = np.asarray(alp.transform(size, Image.AFFINE, data, resample=Image.BILINEAR), np.float32)
        k = sp.alpha / 255.0
        c *= (k * sp.tint).astype(np.float32)
        dst = pixel_array[y0:y1, x0:x1, :3].astype(np.float32)
        if sp.blend == "screen":
            out = dst + c * (255.0 - dst)
        else:
            a *= k
            out = dst * (1.0 - a[..., None]) + c * 255.0
        pixel_array[y0:y1, x0:x1, :3] = np.clip(out, 0, 255).astype(np.uint8)

    # -- finishing ------------------------------------------------------------
    def _post(self, fx, pixel_array):
        ph, pw = pixel_array.shape[:2]
        if self._vignette is None or self._vignette.shape != (ph, pw):
            yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
            nx = (xx / pw - 0.5) * 2
            ny = (yy / ph - 0.5) * 2
            r = np.sqrt(nx**2 * 0.9 + ny**2 * 0.55)
            v = 1.0 - 0.42 * np.clip(r - 0.35, 0, None) ** 1.6
            self._vignette = (np.clip(v, 0, 1) * 256).astype(np.uint16)[..., None]
            rng = np.random.default_rng(7)
            self._grain = [
                (rng.normal(0, self.grain_strength, (ph, pw, 1))).astype(np.int16) for _ in range(6)
            ]
        rgb = pixel_array[..., :3].astype(np.uint16)
        rgb = (rgb * self._vignette) >> 8
        if self.grain_strength:
            g = self._grain[fx.frame % len(self._grain)]
            rgb = np.clip(rgb.astype(np.int16) + g, 0, 255)
        pixel_array[..., :3] = rgb.astype(np.uint8)
        fx.frame += 1


# ---------------------------------------------------------------------------
# Rigs: drawings authored at reference size, re-posed every frame
# ---------------------------------------------------------------------------


class Rig:
    """Holds the rest pose of every leaf so a drawing can be scaled/moved/faded
    without accumulating error.  Leaves may carry ``k`` (extra alpha)."""

    def __init__(self, mob):
        self.mob = mob
        self.leaves = []
        self.refresh()

    def refresh(self):
        self.leaves = []
        for m in self.mob.get_family():
            if isinstance(m, Sprite):
                m._rest = m.points.copy()
                m._ra = m.alpha
                self.leaves.append(m)
            elif isinstance(m, VMobject) and len(m.points):
                m._rest = m.points.copy()
                m._fa = m.fill_rgbas[:, 3].copy()
                m._sa = m.stroke_rgbas[:, 3].copy()
                m._sw = m.stroke_width
                self.leaves.append(m)
            if not hasattr(m, "k"):
                m.k = 1.0

    def pose(self, scale, center, alpha, angle=0.0):
        center = np.asarray(center, float)
        r = rot2(angle) if angle else None
        for m in self.leaves:
            pts = m._rest if r is None else m._rest @ r.T
            m.points = pts * scale + center
            a = alpha * m.k
            if isinstance(m, Sprite):
                m.alpha = m._ra * a
            else:
                m.stroke_width = m._sw * scale
                m.fill_rgbas[:, 3] = m._fa * a
                m.stroke_rgbas[:, 3] = m._sa * a


def set_rest(m, pts):
    m._rest = np.asarray(pts, float)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def text(s, size=32, font=INTER, weight=NORMAL, color="#FFFFFF", t2c=None, tracking=0.0):
    t = Text(s, font=font, weight=weight, font_size=size, color=color, t2c=t2c or {})
    if tracking:
        for i, g in enumerate(t.submobjects):
            g.shift([i * tracking, 0, 0])
        t.center()
    return t


def lines(rows, size=32, gap=0.18, **kw):
    """Stack centred text lines."""
    g = VGroup(*[text(r, size=size, **kw) for r in rows])
    g.arrange(DOWN, buff=gap)
    return g


def fit_width(mob, max_w):
    if mob.width > max_w:
        mob.scale(max_w / mob.width)
    return mob


class AnimText:
    """Per-glyph rise-and-fade in, fade-and-lift out, driven by absolute time."""

    def __init__(self, mob, t_in, t_out=1e9, stagger=0.014, dur_in=0.42, dur_out=0.28,
                 rise=0.22, lift=0.16, scale_in=1.0):
        self.mob = mob
        self.t_in, self.t_out = t_in, t_out
        self.stagger, self.dur_in, self.dur_out = stagger, dur_in, dur_out
        self.rise, self.lift, self.scale_in = rise, lift, scale_in
        self.glyphs = [m for m in mob.get_family() if isinstance(m, VMobject) and len(m.points)]
        for g in self.glyphs:
            g._rest = g.points.copy()
            g._fa = g.fill_rgbas[:, 3].copy()
            g._sa = g.stroke_rgbas[:, 3].copy()
        self.center = mob.get_center()
        self.visible = False
        self.offset = np.zeros(3)
        self.extra_alpha = 1.0

    def update(self, t):
        if t < self.t_in or t > self.t_out + self.dur_out or self.extra_alpha <= 0.002:
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


class MonoFont:
    """Glyph cache for a monospaced font so numbers can change every frame
    without re-running Pango."""

    CHARSET = "0123456789.,-−×+#:mkly"
    LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, size=48, weight=BOLD, font=MONO, letters=False):
        ref = Text("0" * 8, font=font, weight=weight, font_size=size, disable_ligatures=True)
        self.adv = ref.submobjects[1].get_center()[0] - ref.submobjects[0].get_center()[0]
        # Each character is rendered next to a reference "0" so its baseline
        # and side bearing are kept (whole-string rendering trips ligatures).
        self.glyphs = {}
        for ch in self.CHARSET + (self.LETTERS if letters else ""):
            t = Text("0" + ch, font=font, weight=weight, font_size=size, disable_ligatures=True)
            zero, g = t.submobjects
            self.glyphs[ch] = g.points - (zero.get_center() + np.array([self.adv, 0, 0]))
        self.digit_h = ref.submobjects[0].height


class MonoLine(VGroup):
    """A fixed pool of glyph slots that can display any short string."""

    def __init__(self, font: MonoFont, slots=16, color="#FFFFFF", stroke_width=0.0, stroke_color="#000000"):
        super().__init__(*[VMobject(fill_color=color, fill_opacity=1, stroke_width=stroke_width,
                                    stroke_color=stroke_color, stroke_opacity=1) for _ in range(slots)])
        for slot in self.submobjects:
            slot.joint_type = LineJointType.ROUND  # a wide halo stroke would grow miter needles
        self.font = font
        self.alpha = 1.0

    def show(self, spec, center, align="center"):
        """``spec`` is a list of (char, scale, dy); a space just advances."""
        f = self.font
        xs, x = [], 0.0
        for ch, s, _ in spec:
            xs.append(x + f.adv * s / 2)
            x += f.adv * s
        width = x
        x_off = {"center": -width / 2, "left": 0.0, "right": -width}[align]
        k = 0
        for (ch, s, dy), xc in zip(spec, xs):
            if ch == " " or ch not in f.glyphs:
                continue
            slot = self.submobjects[k]
            slot.points = f.glyphs[ch] * s + np.array([center[0] + x_off + xc, center[1] + dy, 0.0])
            k += 1
        for slot in self.submobjects[k:]:
            slot.points = np.zeros((0, 3))
        for slot in self.submobjects:
            slot.fill_rgbas[:, 3] = self.alpha
            slot.stroke_rgbas[:, 3] = self.alpha
        return width


def sci_spec(value, exp_scale=0.6, exp_dy=None, digit_h=0.3, unit="m"):
    """``value`` metres as ``1.6 × 10⁻³⁵ m`` glyph spec."""
    e = int(math.floor(math.log10(value)))
    mant = value / 10**e
    if round(mant, 1) >= 10:
        mant /= 10
        e += 1
    if exp_dy is None:
        exp_dy = digit_h * 0.55
    spec = [(c, 1.0, 0.0) for c in f"{mant:.1f}"]
    spec += [(" ", 0.45, 0), ("×", 1.0, 0.0), (" ", 0.45, 0), ("1", 1.0, 0.0), ("0", 1.0, 0.0)]
    es = str(e).replace("-", "−")
    spec += [(c, exp_scale, exp_dy) for c in es]
    spec += [(" ", 0.6, 0)] + [(c, 1.0, 0.0) for c in unit]
    return spec

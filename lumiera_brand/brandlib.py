"""Lumiera brand primitives: palette, fonts, the glowing mark, text and backgrounds.

Everything is rendered procedurally (numpy + Pillow) so each asset can be rebuilt
at any size with ``python build_brand.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- palette
PALETTE = {
    "deep_space": "#07061A",  # backgrounds
    "night": "#120D2E",  # raised backgrounds, gradients
    "space_glow": "#1C1450",  # soft centre glow of brand stills only, never films
    "starlight": "#F5F7FF",  # main text, the core of the mark
    "lavender": "#B9A6FF",  # primary accent; the start of every journey
    "cyan": "#6CF2FF",  # means you: the you-are-here dot, YOU tags; never a general highlight
    "magenta": "#FF6FD8",  # the observable universe, the outro counter's glow
    "amber": "#FFB547",  # warm accent (life, warmth); the films' Sun is #FFB13B
    "mist_light": "#C9D1E6",  # readable secondary lines
    "mist": "#8A93A8",  # small labels and hints
}

FONT_DIR = Path("/usr/share/fonts/opentype/inter")
MONO_DIR = Path("/usr/share/fonts/truetype/jetbrains-mono")
FONTS = {
    "display": FONT_DIR / "InterDisplay-Bold.otf",  # titles and the hook (brand §4)
    "display_medium": FONT_DIR / "InterDisplay-Medium.otf",
    "display_light": FONT_DIR / "InterDisplay-Light.otf",
    "body": FONT_DIR / "Inter-Medium.otf",
    "body_regular": FONT_DIR / "Inter-Regular.otf",
    "mono": MONO_DIR / "JetBrainsMono-Medium.ttf",
    "mono_bold": MONO_DIR / "JetBrainsMono-Bold.ttf",
}


def rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)], dtype=np.float32)


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS[kind]), size)


# --------------------------------------------------------------------------- light buffers
class Light:
    """Premultiplied RGBA float buffer where shapes add up like light."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.a = np.zeros((h, w), np.float32)

    def grid(self, cx: float, cy: float, scale: float):
        """Coordinates in units of ``scale`` pixels around (cx, cy), y up."""
        ys, xs = np.mgrid[0 : self.h, 0 : self.w].astype(np.float32)
        return (xs + 0.5 - cx) / scale, (cy - ys - 0.5) / scale

    def add(self, coverage: np.ndarray, color, strength: float = 1.0):
        c = np.clip(coverage * strength, 0, None)
        col = np.asarray(color, np.float32)
        self.rgb += c[..., None] * col[None, None, :]
        self.a = 1 - (1 - self.a) * (1 - np.clip(c, 0, 1))

    def over(self, coverage: np.ndarray, color, opacity: float = 1.0):
        """Opaque paint (for text and solid shapes)."""
        a = np.clip(coverage * opacity, 0, 1)
        col = np.asarray(color, np.float32)
        self.rgb = self.rgb * (1 - a[..., None]) + a[..., None] * col[None, None, :]
        self.a = self.a * (1 - a) + a

    def fade_edges(self, cx: float, cy: float, inner: float, outer: float):
        """Fade light to zero between radii ``inner`` and ``outer`` (px) so no glow
        is ever clipped by the image's square edge."""
        x, y = self.grid(cx, cy, 1.0)
        k = np.clip((outer - np.hypot(x, y)) / max(outer - inner, 1e-6), 0, 1)
        k = k * k * (3 - 2 * k)
        self.rgb *= k[..., None]
        self.a *= k

    def image(self, background=None) -> Image.Image:
        rgb_ = 1 - np.exp(-self.rgb * 1.15)  # soft tone-map so stacked glows never clip harshly
        if background is not None:
            bg = np.asarray(background, np.float32)
            out = bg + rgb_ * (1 - bg)  # screen blend onto the background
            return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
        a = np.clip(np.maximum(self.a, rgb_.max(axis=2)), 1e-6, 1)
        col = np.clip(rgb_ / a[..., None], 0, 1)
        arr = np.dstack([col, a])
        return Image.fromarray((arr * 255 + 0.5).astype(np.uint8), "RGBA")


def _aa(d: np.ndarray, px: float) -> np.ndarray:
    """Antialiased coverage from a signed distance (negative inside), px = pixel size."""
    return np.clip(0.5 - d / px, 0, 1)


# --------------------------------------------------------------------------- the mark
MARK = {
    # radii double, in units of the mark radius (outer ring = 1)
    "rings": (0.25, 0.5, 1.0),
    "widths": (0.075, 0.068, 0.06),
    "gap": (math.radians(-38), math.radians(8)),  # opening in the outer ring
    "you_angle": math.radians(8),  # small dot that sits at the end of the opening
}


def draw_mark(light: Light, cx: float, cy: float, radius: float, variant: str = "gradient",
              glow: float = 1.0, core: float = 1.0, mono: str | None = None):
    """The Lumiera mark: a light point with rings whose radii double.

    ``radius`` is the outer ring radius in pixels.  ``variant`` picks the colouring:
    "gradient" (lavender to cyan, by angle) or "white".
    ``mono`` forces a single colour (for watermarks).
    """
    x, y = light.grid(cx, cy, radius)
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    px = 1.0 / radius
    lav, cyan, white = rgb(PALETTE["lavender"]), rgb(PALETTE["cyan"]), rgb(PALETTE["starlight"])
    # light comes from the upper left: blend lavender -> cyan across the mark
    t = np.clip((np.cos(th - math.radians(135)) * -0.5 + 0.5), 0, 1)[..., None]
    grad = lav[None, None, :] * (1 - t) + cyan[None, None, :] * t

    def colour(i):
        if mono is not None:
            return rgb(mono)[None, None, :]
        if variant == "white":
            return white[None, None, :]
        return grad

    g0, g1 = MARK["gap"]
    for i, (R, w) in enumerate(zip(MARK["rings"], MARK["widths"])):
        dist = np.abs(r - R)  # distance to the ring's centre line
        if i == 2:  # the outer ring is an open arc: inside the gap, measure to its ends
            in_gap = (th > g0) & (th < g1)
            ends = [np.hypot(x - R * math.cos(a), y - R * math.sin(a)) for a in (g0, g1)]
            dist = np.where(in_gap, np.minimum(*ends), dist)
        cov = _aa(dist - w / 2, px)  # rounded caps come for free from the distance
        halo = np.exp(-(dist / (w * 2.6)) ** 2) * 0.55 * glow
        col = colour(i)
        light.rgb += (cov[..., None] * col * 1.25) + (halo[..., None] * col * 0.6)
        light.a = np.maximum(light.a, cov)
    # the "you are here" dot at the end of the opening
    if mono is None:
        ya = MARK["you_angle"]
        ex, ey = 1.0 * math.cos(ya), 1.0 * math.sin(ya)
        dd = np.hypot(x - ex, y - ey)
        light.rgb += _aa(dd - 0.07, px)[..., None] * cyan[None, None, :] * 1.6
        light.rgb += (np.exp(-(dd / 0.16) ** 2) * 0.7 * glow)[..., None] * cyan[None, None, :]
        light.a = np.maximum(light.a, _aa(dd - 0.07, px))
    # the light point in the middle
    core_cov = _aa(r - 0.085 * core, px)
    light.rgb += core_cov[..., None] * white[None, None, :] * 2.2
    light.rgb += (np.exp(-(r / 0.2) ** 2) * 1.1 * glow)[..., None] * (white * 0.4 + lav * 0.6)[None, None, :]
    light.rgb += (np.exp(-(r / 0.55) ** 2) * 0.22 * glow)[..., None] * lav[None, None, :]
    light.a = np.maximum(light.a, core_cov)


# --------------------------------------------------------------------------- backgrounds
def space_background(w: int, h: int, cx=None, cy=None, stars: int = 140, seed: int = 7,
                     glow_color=PALETTE["space_glow"], glow_radius=None) -> np.ndarray:
    """Deep-space gradient with a soft central glow and faint stars (float RGB)."""
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    glow_radius = glow_radius or 0.75 * min(w, h)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot(xs - cx, ys - cy) / glow_radius
    base = rgb(PALETTE["deep_space"])[None, None, :]
    glow = rgb(glow_color)[None, None, :]
    img = base + (glow - base) * np.exp(-(d**2) * 1.6)[..., None]
    rng = np.random.default_rng(seed)
    for _ in range(stars):
        sx, sy = rng.uniform(0, w), rng.uniform(0, h)
        b = rng.uniform(0.08, 0.5) ** 1.5
        s = rng.uniform(0.6, 1.6) * max(w, h) / 1080
        x0, x1 = int(max(sx - 4 * s, 0)), int(min(sx + 4 * s + 1, w))
        y0, y1 = int(max(sy - 4 * s, 0)), int(min(sy + 4 * s + 1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        img[y0:y1, x0:x1] += (np.exp(-((gx - sx) ** 2 + (gy - sy) ** 2) / (2 * s * s)) * b)[..., None]
    return np.clip(img, 0, 1)


def to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- text
def tracked_text_size(text: str, f: ImageFont.FreeTypeFont, tracking: float) -> tuple[int, int, int]:
    """Width, ascent-to-baseline offset and height of text with extra letter spacing (em)."""
    em = f.size
    width = sum(f.getlength(ch) for ch in text) + tracking * em * (len(text) - 1)
    l, t, r, b = f.getbbox(text)
    return int(math.ceil(width)), t, b - t


def draw_tracked(img: Image.Image, xy, text: str, f, fill, tracking: float = 0.0, anchor: str = "mm"):
    """Draw text with letter spacing.  ``anchor`` is "mm" (centre) or "lm" (left-middle)."""
    w, top, height = tracked_text_size(text, f, tracking)
    x, y = xy
    if anchor == "mm":
        x -= w / 2
    y -= top + height / 2
    d = ImageDraw.Draw(img)
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill)
        x += f.getlength(ch) + tracking * f.size
    return w


def wordmark_size(height_px: int):
    """Font size and tracking for the LUMIERA wordmark at a given cap height."""
    f = font("display_medium", 100)
    cap = f.getbbox("L")[3] - f.getbbox("L")[1]
    size = int(round(100 * height_px / cap))
    return font("display_medium", size), 0.32

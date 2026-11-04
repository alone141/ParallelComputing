"""Drawing helpers for film.py: typesetting with glyph fallback, antialiased mark sprites and the
glow composite. Pillow + numpy only (fontTools, if installed, is used for the cmap check).

The picture is built in two passes, as in the approved mock-up:
    light   an RGB layer on black. Everything that glows (lanes, marks, playhead, brackets) is drawn
            here; it is blurred twice and screen-blended over the background like real light.
    text    drawn last, straight onto the composited frame, without glow.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920

# ---- palette ----------------------------------------------------------------------------------
BG = (7, 6, 26)                  # deep space
STAR = (245, 247, 255)           # Starlight
MIST_L = (201, 209, 230)         # Mist Light
MIST = (138, 147, 168)           # Mist
WHITE = (255, 255, 255)


def hexc(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lerp_col(a, b, x):
    return tuple(int(round(p + (q - p) * x)) for p, q in zip(a, b))


def scale_col(col, a):
    return tuple(int(round(v * a)) for v in col)


# ---- fonts ------------------------------------------------------------------------------------
_INTER = "/usr/share/fonts/opentype/inter/"
_JBM = "/usr/share/fonts/truetype/jetbrains-mono/"
DISPLAY_BOLD = _INTER + "InterDisplay-Bold.otf"
DISPLAY_SEMI = _INTER + "InterDisplay-SemiBold.otf"
INTER_MED = _INTER + "Inter-Medium.otf"
MONO_BOLD = _JBM + "JetBrainsMono-Bold.ttf"
MONO_MED = _JBM + "JetBrainsMono-Medium.ttf"
# PIL never falls back on its own: a glyph missing from the primary font comes from the first of
# these that has it, scaled to the same cap height and set on the same baseline.
FALLBACKS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
)


@functools.lru_cache(maxsize=None)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@functools.lru_cache(maxsize=None)
def _cmap(path: str):
    """The set of code points the font maps, or None if fontTools is not installed."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None
    return frozenset(TTFont(path, lazy=True).getBestCmap())


@functools.lru_cache(maxsize=None)
def has_glyph(path: str, ch: str) -> bool:
    cm = _cmap(path)
    if cm is not None:
        return ord(ch) in cm
    # Without fontTools: a missing glyph renders as .notdef, so compare against a code point that
    # no font here maps (a private-use one).
    f = font(path, 40)

    def probe(c):
        im = Image.new("L", (100, 80), 0)
        ImageDraw.Draw(im).text((20, 60), c, font=f, fill=255, anchor="ls")
        return im.tobytes()

    return probe(ch) != probe("\U000F0000")


@functools.lru_cache(maxsize=None)
def _cap_height(path: str, size: int) -> float:
    return -font(path, size).getbbox("H", anchor="ls")[1]


@functools.lru_cache(maxsize=None)
def _fallback(path: str, size: int, ch: str):
    for fb in FALLBACKS:
        if has_glyph(fb, ch):
            fsize = max(1, round(size * _cap_height(path, 200) / _cap_height(fb, 200)))
            return font(fb, fsize)
    raise ValueError(f"no installed font has {ch!r} (U+{ord(ch):04X})")


SUB_TIGHTEN = 0.32                       # of a mono cell, between consecutive subscripts
# Glyphs taken from the fallback fonts even though the primary has them: JetBrains Mono's union
# sign is drawn exactly like its capital U ("S U {4, ...}"); DejaVu's sits at x-height, as an operator.
PREFER_FALLBACK = frozenset("\u222a")


def _primary_has(path: str, ch: str) -> bool:
    return ch not in PREFER_FALLBACK and has_glyph(path, ch)


def _is_sub(ch: str) -> bool:
    return "\u2080" <= ch <= "\u209c"


@functools.lru_cache(maxsize=4096)
def typeset(text: str, path: str, size: int, tracking: float = 0.0):
    """Lay out one line. Returns (pieces, width): pieces are (dx, string, font) on a shared
    baseline, dx measured from the line's left edge."""
    prim = font(path, size)
    mono = "Mono" in path
    cell = prim.getlength("0") if mono else None
    pieces, x, run = [], 0.0, ""

    def flush():
        nonlocal x, run
        if run:
            pieces.append((x, run, prim))
            x += prim.getlength(run) + tracking * len(run)
            run = ""

    prev = ""
    for ch in text:
        tight = mono and _is_sub(ch) and _is_sub(prev)
        prev = ch
        if tight and _primary_has(path, ch):  # a run of subscripts (Z45): close the mono gap
            flush()
            x -= cell * SUB_TIGHTEN
            pieces.append((x, ch, prim))
            x += cell + tracking
            continue
        if _primary_has(path, ch):
            if tracking:                      # tracked text is set glyph by glyph
                pieces.append((x, ch, prim))
                x += prim.getlength(ch) + tracking
            else:
                run += ch
            continue
        flush()
        fb = _fallback(path, size, ch)
        adv = fb.getlength(ch)
        if mono:                              # keep the monospaced grid: centre it in a cell
            pieces.append((x + (cell - adv) / 2, ch, fb))
            x += cell + tracking
        else:
            pieces.append((x, ch, fb))
            x += adv + tracking
    flush()
    width = x - (tracking if tracking and text else 0.0)
    return tuple(pieces), width


def text_width(text, path, size, tracking=0.0) -> float:
    return typeset(text, path, size, tracking)[1]


class TextLayer:
    """Draws text onto the composited frame and records every element for the layout JSON."""

    def __init__(self, img: Image.Image):
        self.d = ImageDraw.Draw(img, "RGBA")
        self.records = []

    def text(self, x, y, s, path, size, col, alpha=1.0, anchor="m", tracking=0.0, kind="text"):
        """Draw `s` with its baseline at y. anchor: 'l', 'm' or 'r' (horizontal)."""
        a = max(0.0, min(1.0, alpha))
        if a <= 1.0 / 255 or not s:
            return None
        pieces, width = typeset(s, path, size, tracking)
        x0 = x - (width / 2 if anchor == "m" else width if anchor == "r" else 0.0)
        fill = tuple(col) + (int(round(255 * a)),)
        box = None
        for dx, piece, f in pieces:
            px = x0 + dx
            self.d.text((px, y), piece, font=f, fill=fill, anchor="ls")
            b = f.getbbox(piece, anchor="ls")
            b = (px + b[0], y + b[1], px + b[2], y + b[3])
            box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                         max(box[2], b[2]), max(box[3], b[3]))
        fonts = sorted({f.path.rsplit("/", 1)[-1] for _, _, f in pieces})
        rec = {"text": s, "kind": kind, "bbox": [round(v, 1) for v in box], "baseline": round(y, 1),
               "font": fonts[0] if len(fonts) == 1 else fonts, "size": size, "alpha": round(a, 3)}
        self.records.append(rec)
        return rec


# ---- mark sprites -----------------------------------------------------------------------------
SS = 4                                   # supersampling factor for the sprites
Q = 4                                    # sub-pixel positions and sizes are quantised to 1/Q px


def _sprite(draw_fn, extent: float, fx: float, fy: float) -> tuple[Image.Image, int]:
    """Render an antialiased L-mode mask. The mark's centre sits at (c + fx, c + fy) inside it."""
    c = int(math.ceil(extent)) + 2
    n = 2 * c + 1
    im = Image.new("L", (n * SS, n * SS), 0)
    draw_fn(ImageDraw.Draw(im), ((c + fx) * SS, (c + fy) * SS))
    return im.reduce(SS), c


@functools.lru_cache(maxsize=8192)
def _disc(rq: int, fxq: int, fyq: int):
    r = rq / Q * SS
    return _sprite(lambda d, p: d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255),
                   rq / Q, fxq / Q, fyq / Q)


@functools.lru_cache(maxsize=8192)
def _ring(rq: int, wq: int, fxq: int, fyq: int):
    r, w = rq / Q * SS, wq / Q * SS
    return _sprite(lambda d, p: d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=255,
                                          width=max(1, round(w))), rq / Q + 1, fxq / Q, fyq / Q)


@functools.lru_cache(maxsize=8192)
def _stroke(sq: int, fxq: int, fyq: int):
    """A short slanted stroke (the guitar's palm-muted chug), 18 px tall at scale 1."""
    s = sq / Q

    def draw(d, p):
        dx, dy, w = 3 * s * SS, 9 * s * SS, 3 * s * SS
        ux, uy = 2 * dx, 2 * dy
        ln = math.hypot(ux, uy)
        nx, ny = -uy / ln * w / 2, ux / ln * w / 2             # half-width normal
        a, b = (p[0] - dx, p[1] - dy), (p[0] + dx, p[1] + dy)
        d.polygon([(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny),
                   (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)], fill=255)
        for q in (a, b):                                       # round caps
            d.ellipse([q[0] - w / 2, q[1] - w / 2, q[0] + w / 2, q[1] + w / 2], fill=255)

    return _sprite(draw, 10 * s, fxq / Q, fyq / Q)


def _split(v: float):
    i = math.floor(v)
    f = round((v - i) * Q)
    if f == Q:
        i, f = i + 1, 0
    return i, f


class LightLayer:
    """The glowing layer. Marks are antialiased sprites combined by 'lighter' (per-channel max),
    so a dim or fading mark never punches a dark hole in the rail or a halo beneath it."""

    def __init__(self):
        self.im = Image.new("RGB", (W, H), (0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def _paste(self, col, sprite_fn, x, y, *size_args):
        if max(col) <= 0:
            return
        ix, fx = _split(x)
        iy, fy = _split(y)
        mask, c = sprite_fn(*size_args, fx, fy)
        box = (ix - c, iy - c, ix - c + mask.width, iy - c + mask.height)
        under = self.im.crop(box)
        top = under.copy()
        top.paste(col, (0, 0), mask)
        self.im.paste(ImageChops.lighter(under, top), box)

    def disc(self, x, y, r, col):
        self._paste(col, _disc, x, y, max(1, round(r * Q)))

    def ring(self, x, y, r, col, w=2.0):
        self._paste(col, _ring, x, y, max(1, round(r * Q)), max(1, round(w * Q)))

    def stroke(self, x, y, s, col):
        self._paste(col, _stroke, x, y, max(1, round(s * Q)))

    def mask(self, mask: Image.Image, x: int, y: int, col):
        if max(col) > 0:
            self.im.paste(col, (x, y), mask)

    def line(self, pts, col, w=2):
        if max(col) > 0:
            self.d.line(pts, fill=col, width=w)

    def vline_max(self, x: float, y0: int, y1: int, col, width=2.0, fade=40):
        """A vertical line at fractional x, antialiased across columns, faded at both ends, and
        combined with what is already there by 'lighter' (max), so it never darkens a mark."""
        c0 = math.floor(x - width / 2)
        cols = np.arange(c0, c0 + int(math.ceil(width)) + 2)
        cov = np.clip(np.minimum(cols + 1, x + width / 2) - np.maximum(cols, x - width / 2), 0, 1)
        ys = np.arange(y0, y1)
        env = np.clip(np.minimum(ys - y0, y1 - 1 - ys) / fade, 0, 1)
        env = env * env * (3 - 2 * env)
        a = env[:, None, None] * cov[None, :, None] * np.asarray(col, float)[None, None, :]
        patch = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8), "RGB")
        box = (int(cols[0]), y0, int(cols[-1]) + 1, y1)
        self.im.paste(ImageChops.lighter(self.im.crop(box), patch), box)


# ---- background and glow composite ------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def background() -> Image.Image:
    """Deep-space background with the mock's soft vignette, dithered once (fixed seed) so the
    gradient does not band; the same pattern every frame, so it never shimmers."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = np.hypot((xx - W / 2) / W, (yy - H * 0.48) / H)
    base = np.asarray(BG, np.float32)[None, None, :] * (1.15 - 0.55 * rr[..., None])
    base += np.random.default_rng(7).uniform(-0.5, 0.5, base.shape).astype(np.float32)
    return Image.fromarray(np.clip(np.round(base), 0, 255).astype(np.uint8), "RGB")


GLOW_DOWN = 3                            # the blur runs on a 1/3-scale copy of the light layer
_LUT_NEAR = [min(255, round(v * 1.1)) for v in range(256)] * 3
_LUT_FAR = [min(255, round(v * 0.9)) for v in range(256)] * 3


def composite(light: Image.Image) -> Image.Image:
    """light + 1.1 * blur(10 px) + 0.9 * blur(34 px), screen-blended over the background."""
    small = light.reduce(GLOW_DOWN)
    near = small.filter(ImageFilter.GaussianBlur(10 / GLOW_DOWN)).point(_LUT_NEAR)
    far = small.filter(ImageFilter.GaussianBlur(34 / GLOW_DOWN)).point(_LUT_FAR)
    glow = ImageChops.add(light, ImageChops.add(near, far).resize((W, H), Image.BILINEAR))
    return ImageChops.screen(background(), glow)

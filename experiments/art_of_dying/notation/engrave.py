"""A small drum-notation engraver on top of Pillow.

It is deliberately *not* a general engraver: it knows how to lay out beamed
sixteenth-note groups on a five-line percussion staff, with stems up (hands)
and stems down (feet), cross / diamond / normal noteheads, stemlets over
rests, grace notes, ledger lines, clef, time signature and repeat barlines.
Every notehead, stem and beam is returned as geometry, so the film can
animate them (draw-on, cursor, highlights).

All coordinates are in a 1080 x 1920 design space; `S` is the supersampling
factor used when rasterising.
"""
import math
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/opentype/inter/"
MONO_DIR = "/usr/share/fonts/truetype/jetbrains-mono/"
_fonts = {}

def font(name, size):
    key = (name, size)
    if key not in _fonts:
        path = {
            "display": FONT_DIR + "InterDisplay-Bold.otf",
            "display-x": FONT_DIR + "InterDisplay-ExtraBold.otf",
            "text": FONT_DIR + "Inter-Medium.otf",
            "text-semi": FONT_DIR + "Inter-SemiBold.otf",
            "mono": MONO_DIR + "JetBrainsMono-Bold.ttf",
            "mono-r": MONO_DIR + "JetBrainsMono-Medium.ttf",
            "serif": "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
        }[name]
        _fonts[key] = ImageFont.truetype(path, int(round(size)))
    return _fonts[key]


# ---- staff positions (line 1 = bottom, 5 = top; spaces between) ----------
# value = number of half-spaces above line 1
POS = {
    "kick": 1,        # F  (space 1)
    "floortom": 3,    # A  (space 2)
    "snare": 5,       # C  (space 3)
    "sidestick": 5,   # cross on the snare space
    "hightom": 7,     # E  (space 4)
    "ride": 8,        # F  (line 5)
    "bell": 8,        # diamond on the ride line
    "hihat": 9,       # G  (space above)
    "crash": 10,      # A  (first ledger line)
    "china": 10,
}
HEAD = {               # notehead shape per instrument
    "kick": "normal", "floortom": "normal", "snare": "normal", "hightom": "normal",
    "sidestick": "cross", "ride": "cross", "hihat": "cross", "crash": "cross",
    "china": "cross", "bell": "diamond",
}


class Staff:
    """One system of a drum staff."""

    def __init__(self, x0, x1, y_top, sp=22.0, n_slots=24, lead=92.0):
        self.x0, self.x1, self.y_top, self.sp = x0, x1, y_top, sp
        self.lead = lead                      # room for clef + time signature
        self.n = n_slots
        self.y_bot = y_top + 4 * sp
        self.y_mid = y_top + 2 * sp
        first = x0 + lead
        self.slot_w = (x1 - first - 0.5 * sp) / n_slots
        self.xs = [first + (i + 0.5) * self.slot_w for i in range(n_slots)]

    def y_of(self, pos):
        return self.y_bot - pos * self.sp / 2.0

    def x_of(self, slot):
        return self.xs[slot]


# ---- geometric primitives -------------------------------------------------

def _ellipse_poly(cx, cy, rx, ry, angle_deg, n=40):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        x, y = rx * math.cos(th), ry * math.sin(th)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


class Canvas:
    """Thin wrapper over ImageDraw that applies the supersample factor."""

    def __init__(self, img, S):
        self.S = S
        self.d = ImageDraw.Draw(img, "RGBA")

    def line(self, p0, p1, fill, w):
        S = self.S
        self.d.line([(p0[0] * S, p0[1] * S), (p1[0] * S, p1[1] * S)], fill=fill, width=max(1, int(round(w * S))))

    def poly(self, pts, fill, outline=None, w=0):
        S = self.S
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=fill, outline=outline, width=int(round(w * S)))

    def rect(self, x0, y0, x1, y1, fill):
        S = self.S
        self.d.rectangle([x0 * S, y0 * S, x1 * S, y1 * S], fill=fill)

    def disc(self, cx, cy, r, fill):
        S = self.S
        self.d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=fill)

    def text(self, xy, s, fnt_name, size, fill, anchor="la"):
        S = self.S
        self.d.text((xy[0] * S, xy[1] * S), s, font=font(fnt_name, size * S), fill=fill, anchor=anchor)

    def text_width(self, s, fnt_name, size):
        return font(fnt_name, size * self.S).getlength(s) / self.S


# ---- notation glyphs ------------------------------------------------------

def notehead(c, x, y, sp, kind, color, scale=1.0, lw=None):
    """Draw a notehead centred at (x, y).  Returns (left, right) x extent."""
    sp = sp * scale
    lw = lw or 0.14 * sp / scale
    if kind == "normal":
        c.poly(_ellipse_poly(x, y, 0.64 * sp, 0.46 * sp, -22), fill=color)
        return x - 0.62 * sp, x + 0.62 * sp
    if kind == "cross":
        hw, hh = 0.52 * sp, 0.5 * sp
        lw = lw * 1.35
        c.line((x - hw, y - hh), (x + hw, y + hh), color, lw)
        c.line((x - hw, y + hh), (x + hw, y - hh), color, lw)
        return x - hw, x + hw
    if kind == "diamond":
        pts = [(x, y - 0.55 * sp), (x + 0.6 * sp, y), (x, y + 0.55 * sp), (x - 0.6 * sp, y)]
        c.poly(pts, fill=None, outline=color, w=lw)
        return x - 0.6 * sp, x + 0.6 * sp
    raise ValueError(kind)


def rest16(c, x, y, sp, color):
    """A sixteenth rest, drawn geometrically (centre y = middle line)."""
    lw = 0.11 * sp
    # main slanted stroke
    top = (x + 0.28 * sp, y - 0.75 * sp)
    bot = (x - 0.3 * sp, y + 1.0 * sp)
    c.line(top, bot, color, lw)
    # two hooks with dots
    for k, frac in enumerate((0.04, 0.44)):
        px = top[0] + (bot[0] - top[0]) * frac
        py = top[1] + (bot[1] - top[1]) * frac
        c.line((px, py), (px - 0.42 * sp, py + 0.2 * sp), color, lw * 0.9)
        c.disc(px - 0.47 * sp, py + 0.19 * sp, 0.15 * sp, color)


def grace_note(c, x, y, sp, color, kind="cross"):
    """Small slashed grace note with its own flag, stem up."""
    s = 0.62
    notehead(c, x, y, sp, kind, color, scale=s, lw=0.14 * sp)
    sx = x + 0.5 * sp * s
    top = y - 2.3 * sp * s
    c.line((sx, y), (sx, top), color, 0.09 * sp)
    # slash through the stem
    c.line((sx - 0.45 * sp * s, top + 0.9 * sp * s), (sx + 0.45 * sp * s, top + 0.3 * sp * s), color, 0.1 * sp)
    # flag
    c.line((sx, top), (sx + 0.55 * sp * s, top + 0.7 * sp * s), color, 0.17 * sp)


def percussion_clef(c, x, st, color):
    sp = st.sp
    w = 0.32 * sp
    for dx in (0, 0.62 * sp):
        c.rect(x + dx, st.y_of(6), x + dx + w, st.y_of(2), color)


def time_signature(c, x, st, num, den, color, fnt="serif"):
    sp = st.sp
    size = 2.55 * sp
    c.text((x, st.y_of(4) + 0.08 * sp), str(num), fnt, size, color, anchor="ms")  # upper
    c.text((x, st.y_of(0) + 0.08 * sp), str(den), fnt, size, color, anchor="ms")  # lower


def staff_lines(c, st, color, w=1.6, x_to=None):
    x1 = st.x1 if x_to is None else x_to
    for i in range(5):
        y = st.y_of(2 * i)
        c.line((st.x0, y), (x1, y), color, w)


def barline(c, x, st, color, kind="single"):
    sp = st.sp
    if kind == "single":
        c.line((x, st.y_top), (x, st.y_bot), color, 1.8)
    elif kind == "repeat-start":
        c.rect(x - 0.22 * sp, st.y_top, x + 0.22 * sp, st.y_bot, color)
        c.line((x + 0.6 * sp, st.y_top), (x + 0.6 * sp, st.y_bot), color, 1.8)
        c.disc(x + 1.15 * sp, st.y_of(3), 0.17 * sp, color)
        c.disc(x + 1.15 * sp, st.y_of(5), 0.17 * sp, color)
    elif kind == "repeat-end":
        c.rect(x - 0.22 * sp, st.y_top, x + 0.22 * sp, st.y_bot, color)
        c.line((x - 0.6 * sp, st.y_top), (x - 0.6 * sp, st.y_bot), color, 1.8)
        c.disc(x - 1.15 * sp, st.y_of(3), 0.17 * sp, color)
        c.disc(x - 1.15 * sp, st.y_of(5), 0.17 * sp, color)
    elif kind == "final":
        c.line((x - 0.6 * sp, st.y_top), (x - 0.6 * sp, st.y_bot), color, 1.8)
        c.rect(x - 0.22 * sp, st.y_top, x + 0.22 * sp, st.y_bot, color)


# ---- a beamed sixteenth-note voice ----------------------------------------

class Voice:
    """Events on a sixteenth grid for one stem direction.

    events: dict slot -> list of instrument names (a chord) or [] for a rest.
    Slots absent from `events` draw nothing.

    secondary="full": every slot is a sixteenth (stemlets over rests), the
    secondary beam runs across the whole group.
    secondary="segments": notes are eighths unless two consecutive notes are
    one slot apart, in which case a secondary beam joins them (how an engraver
    would beam a hands voice of eighths with a few sixteenths).
    """

    def __init__(self, st, groups, events, stem_up=True, secondary="full", grace=None,
                 rest_pos=4):
        self.st, self.groups, self.events, self.up = st, groups, events, stem_up
        self.secondary = secondary
        self.grace = grace or set()
        self.rest_pos = rest_pos
        sp = st.sp
        poss = [POS[i] for ev in events.values() for i in ev]
        if stem_up:
            top = max(poss) if poss else 8
            self.y_beam = min(st.y_of(top) - 2.7 * sp, st.y_top - 2.4 * sp)
        else:
            bot = min(poss) if poss else 1
            self.y_beam = max(st.y_of(bot) + 2.7 * sp, st.y_bot + 2.4 * sp)

    def stem_x(self, slot):
        sp = self.st.sp
        x = self.st.x_of(slot)
        return x + 0.56 * sp if self.up else x - 0.56 * sp

    def _beam(self, c, xa, xb, level, col):
        sp = self.st.sp
        th, gap = 0.48 * sp, 0.28 * sp
        off = level * (th + gap)
        if self.up:
            c.rect(xa - 0.065 * sp, self.y_beam + off, xb + 0.065 * sp, self.y_beam + off + th, col)
        else:
            c.rect(xa - 0.065 * sp, self.y_beam - off - th, xb + 0.065 * sp, self.y_beam - off, col)

    def draw(self, c, color, p_visible=None, slot_color=None, group_color=None, ghost=None):
        """Draw the voice.

        p_visible: float; only slots < p_visible are drawn and the beam of the
        group containing p_visible grows with it (the draw-on animation).
        slot_color(slot) -> colour for that slot's noteheads/stem.
        group_color(gi) -> colour for group gi's beams.
        """
        st, sp = self.st, self.st.sp
        start = 0
        for gi, g in enumerate(self.groups):
            slots = list(range(start, start + g))
            start += g
            vis = [s for s in slots if (p_visible is None or s < p_visible)]
            if not vis:
                continue
            gcol = group_color(gi) if group_color else color
            stems = []                       # (slot, x) of every stem in the group
            for s in vis:
                if s not in self.events:
                    continue
                ev = self.events[s]
                col = slot_color(s) if slot_color else color
                x = st.x_of(s)
                sx = self.stem_x(s)
                if ev:
                    ys = []
                    for inst in ev:
                        y = st.y_of(POS[inst])
                        if POS[inst] >= 10:
                            c.line((x - 0.95 * sp, st.y_of(10)), (x + 0.95 * sp, st.y_of(10)), color, 1.6)
                        notehead(c, x, y, sp, HEAD[inst], col)
                        ys.append(y)
                    y_from = min(ys) if self.up else max(ys)
                    c.line((sx, y_from), (sx, self.y_beam), col, 0.13 * sp)
                    stems.append((s, sx))
                    if s in self.grace:
                        grace_note(c, x - 1.05 * sp, st.y_of(POS["sidestick"]), sp, col)
                else:
                    rest16(c, x, st.y_of(self.rest_pos), sp, col)
                    if self.secondary == "full":
                        ly = self.y_beam + (1.1 * sp if self.up else -1.1 * sp)
                        c.line((sx, ly), (sx, self.y_beam), gcol, 0.13 * sp)
                        stems.append((s, sx))
            if not stems:
                continue
            xa = stems[0][1]
            xb = stems[-1][1]
            if p_visible is not None and slots[0] < p_visible < slots[-1] + 1:
                # growing beam: interpolate between stems
                f = p_visible - math.floor(p_visible)
                s0 = int(math.floor(p_visible))
                s1 = min(s0 + 1, slots[-1])
                xb = self.stem_x(s0) + f * (self.stem_x(s1) - self.stem_x(s0))
                xb = max(xb, xa)
            if xb > xa + 0.5:
                self._beam(c, xa, xb, 0, gcol)
                if self.secondary == "full":
                    self._beam(c, xa, xb, 1, gcol)
                else:
                    for (sa, xa2), (sb, xb2) in zip(stems, stems[1:]):
                        if sb - sa == 1:
                            self._beam(c, xa2, min(xb2, xb), 1, gcol)
            else:
                self._beam(c, xa, xa + 0.6 * sp, 0, gcol)

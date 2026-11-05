"""Frame renderer, round 4: one gear per instrument on a common rack (silent film, 60 fps).

The drive is a double-sided rack running across the middle of the screen, one tooth per
sixteenth on the record's clock. Each instrument is a gear that sits on the rack (above) or
hangs from it (below). Tall teeth are hits: they reach the rack's floor and strike; stub teeth
are rests: they still mesh, they strike nothing. A gear slides in from the right, lifted, and
drops into mesh exactly on its first sixteenth; when its part stops it lifts and slides out to
the left. A camera pans and zooms to keep the engaged gears framed. See rhythm.GEARS.

Usage: python3 render.py [--frames a:b] [--still N ...] [--out DIR] [--workers 4]
"""
import argparse
import math
import os
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

import rhythm as R

W, H = 1080, 1920
SS = 2
SW, SH = W * SS, H * SS
RACK_Y = 1000.0                # screen y of the rack's centre line
RH = R.RACK_HALF
PITCH = R.PITCH
MODULE = PITCH / math.pi
ADDENDUM = {"tall": 18.0, "wide": 18.0, "medium": 11.0, "stub": 6.0}
DEDENDUM = 20.0
LIFT = 70.0                    # how far a disengaged gear sits off the rack (world units)
GLIDE_D = 1400.0               # how far a gear travels when gliding in (from off screen)
EXIT_D = 700.0                 # how far it glides out (it is out of frame long before the end)
FIT_X0, FIT_X1 = 60, 860       # screen box the engaged set is framed into (right column stays free)

BG = (9, 9, 12)
STEEL, STEEL_EDGE = (58, 61, 70), (122, 128, 142)
RACK_COL, RACK_EDGE = (34, 36, 42), (96, 100, 112)
HOME_WHITE = (238, 246, 255)
GOLD = (255, 225, 160)
TEXT, DIM, DIMMER = (236, 236, 240), (150, 150, 160), (96, 96, 106)
AMBER = R.BY_KEY["pulse"].colour

FONT_I = "/usr/share/fonts/opentype/inter/"
FONT_M = "/usr/share/fonts/truetype/jetbrains-mono/"


def font(name, size):
    path = {"display": FONT_I + "InterDisplay-Bold.otf", "display_sb": FONT_I + "InterDisplay-SemiBold.otf",
            "text": FONT_I + "Inter-Medium.otf", "mono": FONT_M + "JetBrainsMono-Medium.ttf",
            "mono_b": FONT_M + "JetBrainsMono-Bold.ttf"}[name]
    return ImageFont.truetype(path, int(size * SS))


def polar(cx, cy, r, th):
    return (cx + r * math.cos(th), cy + r * math.sin(th))


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def mix(c1, c2, a):
    return tuple(int(round(c1[i] * (1 - a) + c2[i] * a)) for i in range(3))


def scale(c, a):
    return tuple(int(round(v * a)) for v in c)


def r_pitch(g):
    return g.n * MODULE / 2


def r_tip(g):
    return r_pitch(g) + ADDENDUM["tall"]


def rest_pos(g):
    """World (x, y) of the gear's centre when in mesh."""
    return g.x, -g.side * (r_pitch(g) + RH)


# ---- camera -------------------------------------------------------------------------------------
def fit(keys, zoom_cap=0.95):
    """(cam_x, zoom) framing the given gears into the screen box [FIT_X0, FIT_X1]."""
    x0 = min(R.BY_KEY[k].x - r_tip(R.BY_KEY[k]) for k in keys)
    x1 = max(R.BY_KEY[k].x + r_tip(R.BY_KEY[k]) for k in keys)
    zoom = min(zoom_cap, (FIT_X1 - FIT_X0) / (x1 - x0))
    centre = (x0 + x1) / 2
    cam_x = centre + (W / 2 - (FIT_X0 + FIT_X1) / 2) / zoom
    return cam_x, zoom


F_RIM = (0.0, 0.9)
F_PULSE = (0.0, 0.8)
F_BREATH = fit(("rim", "snare", "kick", "bass", "gtr", "ride"))
F_BAND = fit(("snare", "kick", "bass", "gtr", "ride"))
F_TURN = fit(("ride", "tbass", "tkick", "tgtr", "tsnare"))
F_TURN_WIDE = fit(("snare", "kick", "bass", "gtr", "ride", "tbass", "tkick", "tgtr", "tsnare"))
# (k_start, k_end, frame): the camera eases (smoothstep) to `frame` over the window and holds it.
# Each move happens while the arriving gears glide, so they are seen coming; the settle after a
# landing waits until the leaving gears have been seen going.
CAM_MOVES = (
    (R.K_PULSE - 12.0, R.K_PULSE - 6.0, F_PULSE),
    (R.K_BAND - 13.0, R.K_BAND - 7.0, F_BREATH),
    (R.K_BAND + 3.0, R.K_BAND + 8.0, F_BAND),
    (R.K_TURN - 13.0, R.K_TURN - 7.0, F_TURN_WIDE),
    (R.K_TURN + 1.5, R.K_TURN + 5.5, F_TURN),
)


def camera(k):
    cx, z = F_RIM
    for k0, k1, (fx, fz) in CAM_MOVES:
        if k <= k0:
            break
        u = ease((k - k0) / (k1 - k0))
        cx, z = cx + (fx - cx) * u, z + (fz - z) * u
    return cx, z


# ---- sprites ------------------------------------------------------------------------------------
def gear_sprite(g):
    """RGBA body of a gear at world scale x SS, position p at angle +p*d (ccw gears: -p*d)."""
    rp = r_pitch(g)
    r_root = rp - DEDENDUM
    size = int((r_tip(g) + 12) * 2 * SS)
    c = size / 2
    d = 2 * math.pi / g.n
    sgn = 1 if g.side > 0 else -1
    hw_root, hw_tip = 0.44 * d, 0.26 * d
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    pts = []
    painted = []
    for p in range(g.n):
        kind = R.tooth_kind(g.key, p)
        add = ADDENDUM[kind]
        a = sgn * p * d
        ext = 0.6 * d if kind == "wide" else 0.0          # a held note: the tooth runs on into the rest
        seq = [(r_root, -0.5 * d), (r_root, -hw_root), (rp + add, -hw_tip), (rp + add, hw_tip + ext),
               (r_root, hw_root + ext), (r_root, 0.5 * d + ext)]
        if kind == "wide":
            seq[-1] = (r_root, 0.5 * d + ext)
        if sgn < 0:
            seq = [(r, -t) for (r, t) in seq][::-1]
        for r, th in seq:
            pts.append(polar(c, c, r * SS, a + th))
        if kind != "stub":
            col = g.colour
            if g.colour2 is not None and ((p % 6) // 3) % 2 == 1:
                col = g.colour2
            if g.key == "rim" and p == R.HOME_WHITE_TOOTH:
                col = HOME_WHITE
            tseq = [(r_root + 3, -hw_root * 0.9), (rp + add - 1, -hw_tip), (rp + add - 1, hw_tip + ext), (r_root + 3, hw_root * 0.9 + ext)]
            if sgn < 0:
                tseq = [(r, -t) for (r, t) in tseq][::-1]
            painted.append(([polar(c, c, r * SS, a + th) for r, th in tseq], col))
    dr.polygon(pts, fill=STEEL + (255,))
    arr = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.hypot(xx - c, yy - c) / (r_tip(g) * SS)
    ang = np.arctan2(yy - c, xx - c)
    radial = 0.74 + 0.5 * np.clip(rr, 0, 1) ** 2.2
    sheen = 1.0 + 0.2 * np.cos(ang + 2.3) * np.clip(rr, 0, 1)
    arr[..., :3] = np.clip(arr[..., :3] * (radial * sheen)[..., None], 0, 255)
    img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    dr = ImageDraw.Draw(img)
    for poly, col in painted:
        dr.polygon(poly, fill=col + (255,))

    def ring(r, col, w):
        dr.ellipse([c - r * SS, c - r * SS, c + r * SS, c + r * SS], outline=col, width=int(w * SS))

    ring(r_root, STEEL_EDGE + (110,), 1.5)
    if g.n >= 40:
        ring(r_root - 40, (0, 0, 0, 80), 22)
        ring(r_root - 29, STEEL_EDGE + (70,), 1.2)
        hub = 118
        ring(hub, STEEL_EDGE + (120,), 2)
        ring(hub - 8, (0, 0, 0, 90), 6)
        for i in range(6):
            p = polar(c, c, (hub - 28) * SS, i * math.pi / 3 + 0.2)
            dr.ellipse([p[0] - 5 * SS, p[1] - 5 * SS, p[0] + 5 * SS, p[1] + 5 * SS], fill=(18, 18, 22, 255))
    else:
        hub = 26
        ring(hub, STEEL_EDGE + (120,), 2)
        for i in range(3):
            p = polar(c, c, 44 * SS, i * 2 * math.pi / 3 + 0.4)
            dr.ellipse([p[0] - 9 * SS, p[1] - 9 * SS, p[0] + 9 * SS, p[1] + 9 * SS], fill=(18, 18, 22, 255))
    return img


def glow_sprite(radius, colour, power=1.6):
    s = int(radius * 2 * SS)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    rr = np.hypot(xx - s / 2, yy - s / 2) / (radius * SS)
    a = np.clip(1 - rr, 0, 1) ** power
    return Image.fromarray(np.stack([a * colour[i] for i in range(3)], axis=-1).astype(np.uint8), "RGB")


def ring_sprite(radius, width, colour):
    s = int((radius + width) * 2 * SS) + 4
    img = Image.new("RGB", (s, s), (0, 0, 0))
    d = ImageDraw.Draw(img)
    c = s / 2
    d.ellipse([c - radius * SS, c - radius * SS, c + radius * SS, c + radius * SS], outline=colour, width=int(width * SS))
    return img.filter(ImageFilter.GaussianBlur(1.2 * SS))


def add_sprite(canvas, sprite, x, y, gain=1.0, size_scale=1.0):
    if gain <= 0.01:
        return
    if abs(size_scale - 1.0) > 0.02:
        w0, h0 = sprite.size
        sprite = sprite.resize((max(2, int(w0 * size_scale)), max(2, int(h0 * size_scale))), Image.BILINEAR)
    if gain < 0.999:
        sprite = sprite.point(lambda v: int(v * gain))
    w, h = sprite.size
    x0, y0 = int(x * SS - w / 2), int(y * SS - h / 2)
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(SW, x0 + w), min(SH, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    region = canvas.crop((cx0, cy0, cx1, cy1))
    sp = sprite.crop((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0))
    canvas.paste(ImageChops.add(region, sp), (cx0, cy0))


class Assets:
    def __init__(self):
        self.gears = {g.key: gear_sprite(g) for g in R.GEARS}
        self.glow = {k: glow_sprite(90, R.BY_KEY[k].colour, 2.0) for k in R.KEYS}
        self.glow_big = {"pulse": glow_sprite(150, AMBER, 2.0), "kick": glow_sprite(120, R.BY_KEY["kick"].colour, 2.0),
                         "snare": glow_sprite(130, (255, 255, 255), 2.2)}
        self.glow_crash = glow_sprite(260, R.BY_KEY["ride"].colour, 2.4)
        self.glow_crash_core = glow_sprite(110, (255, 255, 255), 1.6)
        self.glow_home = glow_sprite(420, GOLD, 1.6)
        self.rings = {r: ring_sprite(r, 5, AMBER) for r in range(40, 340, 10)}
        self.rings_home = {r: ring_sprite(r, 7, (255, 236, 190)) for r in range(40, 700, 10)}
        self.bg = self.make_bg()
        self.f_title = font("display", 62)
        self.f_sub = font("text", 32)
        self.f_lap = font("display", 34)
        self.f_mono = font("mono", 24)
        self.f_mono_s = font("mono", 21)
        self.f_mono_xs = font("mono", 18)
        self.f_hub = font("mono_b", 40)
        self.f_hub_s = font("display_sb", 22)
        self.f_group = font("display", 44)
        self.f_chip = font("display_sb", 26)
        self.f_card1 = font("display", 78)
        self.f_card2 = font("display_sb", 40)
        self.f_card3 = font("text", 30)
        self.f_lap = font("display", 40)
        self.f_mono = font("mono", 30)
        self.f_mono_s = font("mono", 26)
        self.f_group = font("display", 52)
        self.f_label = font("display_sb", 36)
        self.f_label_s = font("mono", 24)
        self.f_bracket = font("mono", 32)
        self._resized = {}
        self._fonts = {}

    def f(self, name, size):
        size = int(round(size))
        key = (name, size)
        if key not in self._fonts:
            self._fonts[key] = font(name, size)
        return self._fonts[key]

    def make_bg(self):
        yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
        v = np.hypot((xx - SW / 2) / SW, (yy - RACK_Y * SS) / SH)
        shade = np.clip(1.0 - v * 1.1, 0.0, 1.0)
        arr = np.array(BG, np.float32)[None, None, :] * (0.55 + 0.9 * shade[..., None])
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    def gear_at_zoom(self, key, zoom):
        zq = round(zoom, 3)
        kk = (key, zq)
        if kk not in self._resized:
            if len(self._resized) > 64:
                self._resized.clear()
            sp = self.gears[key]
            n = max(8, int(sp.width * zq))
            self._resized[kk] = sp.resize((n, n), Image.LANCZOS)
        return self._resized[kk]


def wrap(text, f, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.getlength(t) <= max_w * SS:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ---- the frame ----------------------------------------------------------------------------------
def gear_state(g, k):
    """(world x, world y, T_eff, moving) for gear g at sixteenth k."""
    phase, u = R.motion(g.key, k)
    x0, y0 = rest_pos(g)
    lift_dir = -g.side            # above the rack lifts up (negative y), below lifts down
    if phase == "away":
        return None
    if phase == "in":
        return x0, y0, k, False
    if phase == "enter":
        return x0 + GLIDE_D * (1 - ease(u)), y0 + lift_dir * LIFT, float(g.k_in), True
    if phase == "hang":
        return x0, y0 + lift_dir * LIFT, float(g.k_in), True
    if phase == "drop":
        return x0, y0 + lift_dir * LIFT * (1 - ease(u)), float(g.k_in), True
    if phase == "lift":
        return x0, y0 + lift_dir * LIFT * ease(u), float(g.k_out), True
    if phase == "exit":
        return x0 - EXIT_D * ease(u), y0 + lift_dir * LIFT, float(g.k_out), True
    raise ValueError(phase)


def render_frame(A, f):
    T = R.k_at(f)
    time_s = f / R.FPS
    n_now = int(math.floor(T))
    end_dim = ease((time_s - (R.TOTAL - R.END_FADE_S)) / R.END_FADE_S)
    live = 1.0 - 0.85 * end_dim
    cam_x, zoom = camera(T)

    def to_screen(x, y):
        return (W / 2 + (x - cam_x) * zoom, RACK_Y + y * zoom)

    canvas = A.bg.copy()
    dr = ImageDraw.Draw(canvas)

    # --- the rack: a rail across the screen with teeth on both faces, moving one pitch per sixteenth
    top, bot = RACK_Y - RH * zoom, RACK_Y + RH * zoom
    dr.rectangle([0, top * SS, SW, bot * SS], fill=RACK_COL)
    rows = max(3, int((bot - top) * SS / 3))
    for i in range(rows):                                    # brushed surface: fine horizontal rows
        yy = top * SS + (i + 0.5) * (bot - top) * SS / rows
        v = 34 + int(10 * ((i * 7919) % 13) / 12.0)
        dr.line([(0, yy), (SW, yy)], fill=(v, v + 2, v + 8), width=1)
    dr.line([(0, top * SS), (SW, top * SS)], fill=(130, 134, 146), width=int(2 * SS))        # lit top edge
    dr.line([(0, (top + 3 * zoom) * SS), (SW, (top + 3 * zoom) * SS)], fill=(62, 64, 74), width=int(1 * SS))
    dr.line([(0, bot * SS), (SW, bot * SS)], fill=(18, 18, 22), width=int(2 * SS))           # shadowed bottom edge
    dr.line([(0, (bot - 3 * zoom) * SS), (SW, (bot - 3 * zoom) * SS)], fill=(74, 78, 90), width=int(1 * SS))
    th_h = 16 * zoom
    x_world0 = cam_x - W / 2 / zoom - PITCH
    n0 = int(math.floor(x_world0 / PITCH - T - 0.5))
    count = int(W / zoom / PITCH) + 4
    hw_r, hw_t = 0.30 * PITCH * zoom, 0.18 * PITCH * zoom
    for n in range(n0, n0 + count):
        xw = (n + 0.5 + T) * PITCH
        sx = W / 2 + (xw - cam_x) * zoom
        lap_mark = (n % R.CYCLE) == 0                        # tooth 0 of a lap: a lighter tooth
        lit = (168, 172, 186) if lap_mark else (122, 126, 140)
        dark = (112, 116, 130) if lap_mark else (66, 70, 82)
        tip = (210, 214, 226) if lap_mark else (150, 154, 168)
        for face, sgn in ((top, -1), (bot, 1)):
            ty = face + sgn * th_h
            # left flank catches the light, right flank is in shadow
            dr.polygon([((sx - hw_r) * SS, face * SS), ((sx - hw_t) * SS, ty * SS), (sx * SS, ty * SS), (sx * SS, face * SS)], fill=lit)
            dr.polygon([(sx * SS, face * SS), (sx * SS, ty * SS), ((sx + hw_t) * SS, ty * SS), ((sx + hw_r) * SS, face * SS)], fill=dark)
            dr.line([((sx - hw_t) * SS, ty * SS), ((sx + hw_t) * SS, ty * SS)], fill=tip, width=max(1, int(1.2 * SS)))
        if lap_mark:                                         # and a notch on the rail
            dr.rectangle([(sx - 2 * zoom) * SS, (top + 4 * zoom) * SS, (sx + 2 * zoom) * SS, (bot - 4 * zoom) * SS], fill=(150, 154, 168))
        if n % 6 == 3:                                       # rivets at a regular pitch
            rr = max(2.0, 4.5 * zoom)
            cy = RACK_Y
            dr.ellipse([(sx - rr) * SS, (cy - rr) * SS, (sx + rr) * SS, (cy + rr) * SS], fill=(22, 22, 28))
            dr.ellipse([(sx - rr * 0.7) * SS, (cy - rr * 0.7) * SS, (sx + rr * 0.7) * SS, (cy + rr * 0.7) * SS], fill=(88, 92, 104))
            dr.ellipse([(sx - rr * 0.35) * SS, (cy - rr * 0.6) * SS, (sx + rr * 0.1) * SS, (cy - rr * 0.1) * SS], fill=(150, 154, 168))
    # wear marks and glints where engaged gears strike the rail
    for g in R.GEARS:
        if not R.engaged(g.key, max(0, n_now)):
            continue
        cx_s = W / 2 + (g.x - cam_x) * zoom
        if cx_s < -60 or cx_s > W + 60:
            continue
        face = top if g.side > 0 else bot
        inner = face + (1 if g.side > 0 else -1) * 5 * zoom
        ww = 26 * zoom
        dr.ellipse([(cx_s - ww) * SS, min(face, inner) * SS, (cx_s + ww) * SS, max(face, inner) * SS], fill=(26, 26, 32))
        gl = scale((190, 194, 206), live)
        dr.line([((cx_s - 9 * zoom) * SS, face * SS), ((cx_s + 9 * zoom) * SS, face * SS)], fill=gl, width=int(2 * SS))

    # --- gears
    states = {}
    for g in R.GEARS:
        st = gear_state(g, T)
        if st is None:
            continue
        x, y, Te, moving = st
        states[g.key] = st
        d = 2 * math.pi / g.n
        rot = (Te * d - math.pi / 2) if g.side > 0 else (math.pi / 2 - Te * d)
        sp = A.gear_at_zoom(g.key, zoom)
        img = sp.rotate(math.degrees(rot), resample=Image.BICUBIC)
        sx, sy = to_screen(x, y)
        if sx * SS + img.width / 2 < 0 or sx * SS - img.width / 2 > SW:
            continue
        if end_dim > 0:
            pass
        canvas.paste(img, (int(sx * SS - img.width / 2), int(sy * SS - img.height / 2)), img)
    dr = ImageDraw.Draw(canvas)

    # --- hits: flashes at each engaged gear's contact with the rack
    size_scale = min(1.3, max(0.6, zoom / 0.65))
    for n in range(max(0, n_now - 5), n_now + 1):
        age = (T - n) * R.SIXTEENTH
        for g in R.GEARS:
            if not R.engaged(g.key, n) or g.key not in states:
                continue
            kinds = R.hits_at(g.key, n)
            if not kinds:
                continue
            kind = kinds[0]
            fx, fy = to_screen(g.x, -g.side * RH)
            if kind == "pulse":
                dur = 0.36
                if age < dur:
                    e = (1 - age / dur) ** 1.6 * live
                    add_sprite(canvas, A.glow_big["pulse"], fx, fy + 6, 0.75 * e, size_scale)
                    rr = 40 + int(age / dur * 300) // 10 * 10
                    if rr in A.rings:
                        add_sprite(canvas, A.rings[rr], fx, fy, 0.8 * e, size_scale)
            elif kind == "crash":
                if age < 0.45:
                    e = (1 - age / 0.45) ** 1.5
                    add_sprite(canvas, A.glow_crash, fx, fy, 0.95 * e, size_scale)
                    add_sprite(canvas, A.glow_crash_core, fx, fy, e ** 0.7, size_scale)
                    rr = 40 + int(age / 0.45 * 300) // 10 * 10
                    if rr in A.rings_home:
                        add_sprite(canvas, A.rings_home[rr], fx, fy, 0.7 * e, size_scale)
            elif kind in ("kick", "snare"):
                dur = 0.22 if kind == "snare" else 0.28
                if age < dur:
                    e = (1 - age / dur) ** 1.5 * live
                    add_sprite(canvas, A.glow_big[kind], fx, fy, 0.8 * e, size_scale * (0.75 if g.n < 40 else 1.0))
            else:
                dur = {"click": 0.14, "bell": 0.3, "ride": 0.16, "note": 0.22, "chug": 0.09}[kind]
                lvl = {"click": 0.9, "bell": 0.8, "ride": 0.35, "note": 0.7, "chug": 0.3}[kind]
                if age < dur:
                    e = (1 - age / dur) ** 1.5 * live
                    add_sprite(canvas, A.glow[g.key], fx, fy, lvl * e, size_scale)
    for hn in R.HOMES:
        age = (T - hn) * R.SIXTEENTH
        if 0 <= age < 1.1:
            e = (1 - age / 1.1) ** 1.3
            hx, hy = to_screen(0.0, 0.0)
            add_sprite(canvas, A.glow_home, hx, hy, 0.8 * e, size_scale)
            rr = 40 + int(age / 1.1 * 640) // 10 * 10
            if rr in A.rings_home:
                add_sprite(canvas, A.rings_home[rr], hx, hy, 0.9 * e, size_scale)

    # --- gear labels: hub number and name for the big gears, a two-line label beside the small ones
    LABEL_DX = {"gtr": -44, "ride": 44}                      # neighbours step aside instead of overlapping
    LABEL_DY = {"ride": -40}
    for g in R.GEARS:
        if g.key not in states:
            continue
        x, y, Te, moving = states[g.key]
        sx, sy = to_screen(x, y)
        if sx < -200 or sx > W + 200:
            continue
        a = (0.55 if moving else 1.0) * live
        if g.n >= 40:
            f_num = A.f("mono_b", min(70, max(38, 95 * zoom)))
            f_name = A.f("display_sb", min(34, max(22, 44 * zoom)))
            gap = 18 * min(1.4, max(0.8, zoom / 0.55))
            dr.text((sx * SS, (sy - gap) * SS), str(g.n), font=f_num, fill=scale(TEXT, a), anchor="ms")
            dr.text((sx * SS, (sy + gap) * SS), g.short, font=f_name, fill=scale(g.colour, a), anchor="mt")
        else:
            up = g.side > 0
            base = sy + (r_tip(g) * zoom + 30) * (-1 if up else 1) + LABEL_DY.get(g.key, 0) * (1 if up else -1)
            lx_ = sx + LABEL_DX.get(g.key, 0)
            name = g.short.replace("GUITARS", "GTRS")
            if up:
                dr.text((lx_ * SS, base * SS), name, font=A.f_label, fill=scale(g.colour, a), anchor="ms")
                dr.text((lx_ * SS, (base - 42) * SS), f"{g.n} TEETH", font=A.f_label_s, fill=scale(DIM, a), anchor="ms")
            else:
                dr.text((lx_ * SS, base * SS), name, font=A.f_label, fill=scale(g.colour, a), anchor="mt")
                dr.text((lx_ * SS, (base + 42) * SS), f"{g.n} TEETH", font=A.f_label_s, fill=scale(DIM, a), anchor="mt")

    # --- slip bracket at each cycle start once the beat is in play
    tl = T % R.CYCLE
    t_in_lap = tl * R.SIXTEENTH
    sl = R.slip(T)
    if R.K_PULSE + R.CYCLE <= T < R.K_TURN and t_in_lap < 1.9 and sl > 0:
        key = "pulse" if T < R.K_BAND else "ride"
        g = R.BY_KEY[key]
        if key in states and not states[key][3]:
            a = min(ease(t_in_lap / 0.25), ease((1.9 - t_in_lap) / 0.4)) * live
            col = scale((255, 255, 255), a)
            gx, gy = to_screen(*rest_pos(g))
            r_arc = (r_tip(g) + 16) * zoom
            d = 2 * math.pi / g.n
            th0 = -math.pi / 2 if g.side < 0 else math.pi / 2
            th1 = th0 + (sl * d if g.side < 0 else -sl * d)
            pts = [polar(gx * SS, gy * SS, r_arc * SS, th0 + (th1 - th0) * i / 24) for i in range(25)]
            dr.line(pts, fill=col, width=int(4 * SS), joint="curve")
            for th in (th0, th1):
                p0 = polar(gx * SS, gy * SS, (r_arc - 8) * SS, th)
                p1 = polar(gx * SS, gy * SS, (r_arc + 8) * SS, th)
                dr.line([p0, p1], fill=col, width=int(4 * SS))
            word = "1 tooth ahead" if sl == 1 else f"{sl} teeth ahead"
            dr.text(((gx + r_arc + 22) * SS, gy * SS), word, font=A.f_bracket, fill=col, anchor="lm")

    # --- captions (top)
    idx = R.caption_index(T)
    ka, kb = R.caption_window(idx)
    fade = 0.35 / R.SIXTEENTH
    fade_in = 0.0 if idx in (0, 4, 8, 12) else 0.15 / R.SIXTEENTH
    alpha = min(ease((T - ka) / fade_in) if fade_in else 1.0, ease((kb - T) / fade))
    if idx == R.N_LAPS_CYCLE:
        alpha = min(ease((T - ka) / fade_in) if fade_in else 1.0, 1.0 - end_dim)
    title, sub = R.CAPTIONS[idx]
    title_col = TEXT
    if title.startswith("HOME"):
        title_col = mix(TEXT, GOLD, 0.8)
    elif idx == 4:
        title_col = mix(TEXT, AMBER, 0.85)
    elif idx in (13, 15, 16):
        title_col = mix(TEXT, R.BY_KEY["kick"].colour, 0.7)
    elif idx == 14:
        title_col = mix(TEXT, R.BY_KEY["ride"].colour, 0.8)
    kick_px = 0.0
    nb = (n_now // R.BEAT) * R.BEAT
    age_b = (T - nb) * R.SIXTEENTH
    if (R.engaged("pulse", n_now) or R.engaged("ride", n_now)) and age_b < 0.18:
        kick_px = (1 - age_b / 0.18) * 4
    y = 112 - kick_px
    for line in wrap(title, A.f_title, 940):
        dr.text((W / 2 * SS, y * SS), line, font=A.f_title, fill=scale(title_col, alpha), anchor="mm")
        y += 70
    y += 8
    for line in wrap(sub, A.f_sub, 900):
        dr.text((W / 2 * SS, y * SS), line, font=A.f_sub, fill=scale(DIM, alpha), anchor="mm")
        y += 42

    # --- readout strip: row A group counter + lap, row B slip (+ the cut note), row C the stack
    lx, ry = 60, 1392
    dot_r, dot_dx = 12, 38
    if T < R.K_TURN:
        gi, gs, gsize = R.group_of(n_now % R.CYCLE)
        p_in = n_now % R.CYCLE - gs
        dr.text((lx * SS, ry * SS), str(gsize), font=A.f_group, fill=scale(TEXT, live), anchor="lm")
        for q in range(gsize):
            x0 = lx + 50 + q * dot_dx
            reached = q <= p_in
            if gs + q in R.RIM_RESTS:
                dr.ellipse([(x0 - dot_r) * SS, (ry - dot_r) * SS, (x0 + dot_r) * SS, (ry + dot_r) * SS], outline=(TEXT if reached else DIMMER), width=int(2.5 * SS))
            else:
                dr.ellipse([(x0 - dot_r) * SS, (ry - dot_r) * SS, (x0 + dot_r) * SS, (ry + dot_r) * SS], fill=(R.BY_KEY["rim"].colour if reached else DIMMER))
    else:
        c_in = (n_now - R.K_TURN) % 6
        dr.text((lx * SS, ry * SS), "6", font=A.f_group, fill=scale(TEXT, live), anchor="lm")
        for q, cell in enumerate(R.TURN_CELL):
            x0 = lx + 50 + q * dot_dx
            reached = q <= c_in
            if cell == ".":
                dr.ellipse([(x0 - dot_r) * SS, (ry - dot_r) * SS, (x0 + dot_r) * SS, (ry + dot_r) * SS], outline=(TEXT if reached else DIMMER), width=int(2.5 * SS))
            else:
                colr = R.BY_KEY["kick"].colour if cell == "K" else (255, 255, 255)
                dr.ellipse([(x0 - dot_r) * SS, (ry - dot_r) * SS, (x0 + dot_r) * SS, (ry + dot_r) * SS], fill=(scale(colr, live) if reached else DIMMER))
    la = ease((T - R.K_PULSE) * R.SIXTEENTH / 0.5) * live
    if la > 0:
        lap_txt = f"LAP {min(int(T // R.CYCLE), R.N_LAPS_CYCLE - 1) + 1} / {R.N_LAPS_CYCLE}" if T < R.K_TURN else "6/4 TURN"
        dr.text((330 * SS, ry * SS), lap_txt, font=A.f_lap, fill=scale(TEXT, la), anchor="lm")
        sy_b = 1446
        dr.text((lx * SS, sy_b * SS), "SLIP", font=A.f_mono, fill=scale(DIM, la), anchor="lm")
        for i in range(4):
            x0 = lx + 90 + i * 36
            box = [x0 * SS, (sy_b - 13) * SS, (x0 + 26) * SS, (sy_b + 13) * SS]
            if i < sl:
                dr.rectangle(box, fill=scale(AMBER, la))
            else:
                dr.rectangle(box, outline=scale(DIM, la), width=int(2 * SS))
        label = "HOME" if sl == 0 else (f"{sl} TOOTH" if sl == 1 else f"{sl} TEETH")
        dr.text(((lx + 90 + 4 * 36 + 10) * SS, sy_b * SS), label, font=A.f_mono, fill=scale(AMBER if sl else HOME_WHITE, la), anchor="lm")
    sy = 1502
    xcur = lx
    for key in R.STACK_KEYS:
        g = R.BY_KEY[key]
        on = any(R.engaged(k2, n_now) for k2 in R.KEYS if (R.BY_KEY[k2].stack or k2) == key)
        col = mix(DIMMER, g.colour, (1.0 if on else 0.0) * live)
        lit = 0.0
        for n in range(max(0, n_now - 2), n_now + 1):
            for k2 in R.KEYS:
                if (R.BY_KEY[k2].stack or k2) == key and R.engaged(k2, n) and R.hits_at(k2, n):
                    lit = max(lit, max(0.0, 1 - (T - n) * R.SIXTEENTH / 0.16))
        dcol = mix(col, (255, 255, 255), lit * 0.8 * live) if on else DIMMER
        rad = 7 + 3 * lit
        dr.ellipse([(xcur + 8 - rad) * SS, (sy - rad) * SS, (xcur + 8 + rad) * SS, (sy + rad) * SS], fill=dcol)
        dr.text(((xcur + 24) * SS, sy * SS), g.name, font=A.f_mono, fill=(col if on else DIMMER), anchor="lm")
        xcur += 24 + A.f_mono.getlength(g.name) / SS + 20
    age_cut = (T - R.K_BAND) * R.SIXTEENTH
    if 1.9 <= age_cut < 4.6:
        a = min(ease((age_cut - 1.9) / 0.3), ease((4.6 - age_cut) / 0.4))
        dr.text((440 * SS, 1446 * SS), "cut: bars 34-41 removed", font=A.f_mono_s, fill=scale(DIM, a), anchor="lm")

    # --- closing card over the audio fade
    if end_dim > 0:
        canvas = Image.blend(canvas, Image.new("RGB", canvas.size, (0, 0, 0)), 0.72 * end_dim)
        dr = ImageDraw.Draw(canvas)
        a = ease((end_dim - 0.15) / 0.7)
        cy = 980
        dr.text((W / 2 * SS, (cy - 90) * SS), "GOJIRA", font=A.f_card1, fill=scale(TEXT, a), anchor="mm")
        dr.text((W / 2 * SS, (cy - 24) * SS), "THE ART OF DYING", font=A.f_card2, fill=scale(TEXT, a), anchor="mm")
        dr.text((W / 2 * SS, (cy + 24) * SS), "The Way of All Flesh, 2008", font=A.f_card3, fill=scale(DIM, a), anchor="mm")
        dr.text((W / 2 * SS, (cy + 96) * SS), "45 against 4: it drifts one tooth a lap and comes home in four.",
                font=A.f_card3, fill=scale(GOLD, a), anchor="mm")
        dr.text((W / 2 * SS, (cy + 138) * SS), "An independent study of the rhythm. Not affiliated with the band or label.",
                font=A.f_mono_s, fill=scale(DIM, a * 0.8), anchor="mm")

    return canvas.resize((W, H), Image.LANCZOS)


_ASSETS = None


def _work(args):
    global _ASSETS
    f, out = args
    if _ASSETS is None:
        _ASSETS = Assets()
    render_frame(_ASSETS, f).save(out, compress_level=1)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=f"0:{R.N_FRAMES}")
    ap.add_argument("--still", type=int, nargs="*")
    ap.add_argument("--out", default="build/frames")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.still:
        A = Assets()
        for f in a.still:
            render_frame(A, f).save(os.path.join(a.out, f"still_{f:05d}.png"))
            print("still", f, "k=%.2f" % R.k_at(f), "t=%.2fs" % (f / R.FPS), "cam", tuple(round(v, 2) for v in camera(R.k_at(f))))
        return
    lo, hi = (int(x) for x in a.frames.split(":"))
    jobs = [(f, os.path.join(a.out, f"f{f:05d}.png")) for f in range(lo, hi)]
    with Pool(a.workers) as p:
        for i, f in enumerate(p.imap_unordered(_work, jobs, chunksize=8)):
            if i % 200 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print("done", len(jobs))


if __name__ == "__main__":
    main()

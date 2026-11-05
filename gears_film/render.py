"""The frame renderer: a rack, one gear per instrument, a camera. Reads only a Song.

World units: the rack pitch is 42 (one tooth = one step). The rack's centre line is screen y 1000.
Supersampled 2x, 1080x1920 at 60 fps. See gears_film.song for the data it draws.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

from .song import FPS, W, H, TOOTH_TYPES, Song

SS = 2
SW, SH = W * SS, H * SS
RACK_Y = 1000.0
RH = 20.0                      # rack half-thickness
PITCH = 42.0
MODULE = PITCH / math.pi
DEDENDUM = 20.0
LIFT = 70.0
GLIDE_D, EXIT_D = 1400.0, 700.0
GLIDE_S, HANG_S, DROP_S, EXIT_S = 1.0, 0.38, 0.08, 1.0
FIT_X0, FIT_X1 = 60, 860

BG = (9, 9, 12)
STEEL, STEEL_EDGE = (58, 61, 70), (122, 128, 142)
RACK_COL = (34, 36, 42)
HOME_WHITE = (238, 246, 255)
GOLD = (255, 225, 160)
TEXT, DIM, DIMMER = (236, 236, 240), (150, 150, 160), (96, 96, 106)

FONT_I = "/usr/share/fonts/opentype/inter/"
FONT_M = "/usr/share/fonts/truetype/jetbrains-mono/"
FONT_FILES = {"display": FONT_I + "InterDisplay-Bold.otf", "display_sb": FONT_I + "InterDisplay-SemiBold.otf",
              "text": FONT_I + "Inter-Medium.otf", "mono": FONT_M + "JetBrainsMono-Medium.ttf",
              "mono_b": FONT_M + "JetBrainsMono-Bold.ttf"}


def font(name, size):
    return ImageFont.truetype(FONT_FILES[name], int(size * SS))


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
    return g.teeth * MODULE / 2


def r_tip(g):
    return r_pitch(g) + TOOTH_TYPES["tall"]["add"]


def rest_pos(g):
    return g.x, -g.side * (r_pitch(g) + RH)


# ---- camera -------------------------------------------------------------------------------------
def fit(song, keys, zoom_cap=0.95):
    x0 = min(song.by_key[k].x - r_tip(song.by_key[k]) for k in keys)
    x1 = max(song.by_key[k].x + r_tip(song.by_key[k]) for k in keys)
    zoom = min(zoom_cap, (FIT_X1 - FIT_X0) / (x1 - x0))
    centre = (x0 + x1) / 2
    return centre + (W / 2 - (FIT_X0 + FIT_X1) / 2) / zoom, zoom


def auto_camera(song):
    """Camera moves derived from the gears when the song gives none: wide while gears arrive,
    settle after the leavers have gone."""
    events = sorted({a for g in song.gears for a, b in g.ranges} | {b for g in song.gears for a, b in g.ranges if b < song.total_steps})
    moves = []
    first = [g.key for g in song.gears if g.engaged(0)]
    if first:
        cx, z = fit(song, first, song.opening_zoom)
        moves.append((0.0, 0.0, cx, z))
    for e in events:
        if e == 0:
            continue
        before = [g.key for g in song.gears if g.engaged(e - 1e-6)]
        after = [g.key for g in song.gears if g.engaged(e + 1e-6)]
        step_s = song.step_s_at(e)
        both = sorted(set(before) | set(after))
        if both:
            cx, z = fit(song, both)
            moves.append((e - 1.4 / step_s, e - 0.75 / step_s, cx, z))
        if after and set(after) != set(both):
            cx, z = fit(song, after)
            moves.append((e + 0.3 / step_s, e + 0.85 / step_s, cx, z))
    return moves


def camera_moves(song):
    if not song.camera:
        return auto_camera(song)
    out = []
    for m in song.camera:
        if m.keys:
            cx, z = fit(song, m.keys, m.zoom_cap)
        else:
            cx, z = m.cam_x, m.zoom
        out.append((m.step_from, m.step_to, cx, z))
    return out


def camera_at(moves, k):
    cx, z = moves[0][2], moves[0][3]
    for k0, k1, fx, fz in moves[1:]:
        if k <= k0:
            break
        u = ease((k - k0) / (k1 - k0)) if k1 > k0 else 1.0
        cx, z = cx + (fx - cx) * u, z + (fz - z) * u
    return cx, z


# ---- motion -------------------------------------------------------------------------------------
def motion(song, g, k):
    """("in"|"enter"|"hang"|"drop"|"lift"|"exit"|"away", u) for gear g at film step k, in seconds."""
    t = song.step_time(k)
    best = ("away", 0.0)
    for a, b in g.ranges:
        if a == 0 and k < a:
            return ("in", 1.0)
        t_in, t_out = song.step_time(a), song.step_time(b)
        hang = HANG_S + g.lead
        t0 = t_in - DROP_S - hang - GLIDE_S
        if t < t0:
            continue
        if t < t0 + GLIDE_S:
            return ("enter", (t - t0) / GLIDE_S, a)
        if t < t_in - DROP_S:
            return ("hang", (t - t0 - GLIDE_S) / hang, a)
        if t < t_in:
            return ("drop", (t - (t_in - DROP_S)) / DROP_S, a)
        if k < b or b >= song.total_steps:
            return ("in", 1.0, a)
        if t < t_out + DROP_S:
            return ("lift", (t - t_out) / DROP_S, b)
        if t < t_out + DROP_S + EXIT_S:
            return ("exit", (t - t_out - DROP_S) / EXIT_S, b)
    return best


def gear_state(song, g, k):
    m = motion(song, g, k)
    phase, u = m[0], m[1]
    if phase == "away":
        return None
    x0, y0 = rest_pos(g)
    lift_dir = -g.side
    if phase == "in":
        return x0, y0, k, False
    ref = float(m[2])
    if phase == "enter":
        return x0 + GLIDE_D * (1 - ease(u)), y0 + lift_dir * LIFT, ref, True
    if phase == "hang":
        return x0, y0 + lift_dir * LIFT, ref, True
    if phase == "drop":
        return x0, y0 + lift_dir * LIFT * (1 - ease(u)), ref, True
    if phase == "lift":
        return x0, y0 + lift_dir * LIFT * ease(u), ref, True
    if phase == "exit":
        return x0 - EXIT_D * ease(u), y0 + lift_dir * LIFT, ref, True
    raise ValueError(phase)


# ---- sprites ------------------------------------------------------------------------------------
def gear_sprite(g):
    rp = r_pitch(g)
    r_root = rp - DEDENDUM
    size = int((r_tip(g) + 12) * 2 * SS)
    c = size / 2
    d = 2 * math.pi / g.teeth
    sgn = 1 if g.side > 0 else -1
    hw_root, hw_tip = 0.44 * d, 0.26 * d
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    pts, painted = [], []
    for p in range(g.teeth):
        kind = g.tooth_type(p)
        spec = TOOTH_TYPES[kind]
        add, ext, tips = spec["add"], spec["ext"] * d, spec["tips"]
        a = sgn * p * d
        if tips == 1:
            seq = [(r_root, -0.5 * d), (r_root, -hw_root), (rp + add, -hw_tip), (rp + add, hw_tip + ext),
                   (r_root, hw_root + ext), (r_root, 0.5 * d + ext)]
        else:   # a compound tooth: `tips` narrow tips inside one position
            seq = [(r_root, -0.5 * d), (r_root, -hw_root)]
            span = 2 * hw_root
            for i in range(tips):
                t0 = -hw_root + span * i / tips
                t1 = -hw_root + span * (i + 1) / tips
                mid = (t0 + t1) / 2
                seq += [(rp + add, mid - (t1 - t0) * 0.25), (rp + add, mid + (t1 - t0) * 0.25), (r_root + add * 0.45, t1 - (t1 - t0) * 0.08)]
            seq += [(r_root, hw_root), (r_root, 0.5 * d)]
        if sgn < 0:
            seq = [(r, -t) for (r, t) in seq][::-1]
        for r, th in seq:
            pts.append(polar(c, c, r * SS, a + th))
        if kind != "stub":
            col = g.colour
            if g.colour2 is not None and p in g.colour2_positions:
                col = g.colour2
            if g.white_position is not None and p == g.white_position:
                col = HOME_WHITE
            if tips == 1:
                tseqs = [[(r_root + 3, -hw_root * 0.9), (rp + add - 1, -hw_tip), (rp + add - 1, hw_tip + ext), (r_root + 3, hw_root * 0.9 + ext)]]
            else:   # paint each tip of a compound tooth on its own
                tseqs = []
                span = 2 * hw_root
                for i in range(tips):
                    t0 = -hw_root + span * i / tips
                    t1 = -hw_root + span * (i + 1) / tips
                    mid = (t0 + t1) / 2
                    tseqs.append([(r_root + 3, t0 + (t1 - t0) * 0.12), (rp + add - 1, mid - (t1 - t0) * 0.25),
                                  (rp + add - 1, mid + (t1 - t0) * 0.25), (r_root + 3, t1 - (t1 - t0) * 0.12)])
            for tseq in tseqs:
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
    if g.teeth >= 40:
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


BIG_GLOW = {"pulse": (150, None, 2.0), "kick": (120, None, 2.0), "snare": (130, (255, 255, 255), 2.2)}
SMALL_GLOW = {"click": (0.14, 0.9), "bell": (0.3, 0.8), "ride": (0.16, 0.35), "note": (0.22, 0.7), "chug": (0.09, 0.3)}


class Assets:
    def __init__(self, song: Song):
        self.song = song
        self.gears = {g.key: gear_sprite(g) for g in song.gears}
        self.glow = {g.key: glow_sprite(90, g.colour, 2.0) for g in song.gears}
        self.glow_big = {}
        self.rings = {}
        for g in song.gears:
            if g.sound in BIG_GLOW:
                rad, col, pw = BIG_GLOW[g.sound]
                self.glow_big[g.key] = glow_sprite(rad, col or g.colour, pw)
            if g.sound == "pulse":
                self.rings[g.key] = {r: ring_sprite(r, 5, g.colour) for r in range(40, 340, 10)}
            if g.first_hit == "crash":
                self.glow_big["crash:" + g.key] = glow_sprite(260, g.colour, 2.4)
        self.glow_crash_core = glow_sprite(110, (255, 255, 255), 1.6)
        self.glow_home = glow_sprite(420, GOLD, 1.6)
        self.rings_home = {r: ring_sprite(r, 7, (255, 236, 190)) for r in range(40, 700, 10)}
        self.bg = self.make_bg()
        self.f_title = font("display", 62)
        self.f_sub = font("text", 32)
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
        self.moves = camera_moves(song)

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
def render_frame(A: Assets, f: int):
    song = A.song
    T = song.k_at(f)
    time_s = f / FPS
    n_now = int(math.floor(T))
    step_s = song.step_s_at(T)
    end_dim = ease((time_s - (song.total_s - song.end_fade_s)) / song.end_fade_s)
    live = 1.0 - 0.85 * end_dim
    cam_x, zoom = camera_at(A.moves, T)
    sp = song.slip_params()
    beat_steps = song.slip.beat_steps if song.slip else None
    k_ref = sp[3] if sp else None
    cyc = song.cycle_steps

    def to_screen(x, y):
        return (W / 2 + (x - cam_x) * zoom, RACK_Y + y * zoom)

    def age_of(n):
        return time_s - song.step_time(n)

    canvas = A.bg.copy()
    dr = ImageDraw.Draw(canvas)

    # --- the rack
    top, bot = RACK_Y - RH * zoom, RACK_Y + RH * zoom
    dr.rectangle([0, top * SS, SW, bot * SS], fill=RACK_COL)
    rows = max(3, int((bot - top) * SS / 3))
    for i in range(rows):
        yy = top * SS + (i + 0.5) * (bot - top) * SS / rows
        v = 34 + int(10 * ((i * 7919) % 13) / 12.0)
        dr.line([(0, yy), (SW, yy)], fill=(v, v + 2, v + 8), width=1)
    dr.line([(0, top * SS), (SW, top * SS)], fill=(130, 134, 146), width=int(2 * SS))
    dr.line([(0, (top + 3 * zoom) * SS), (SW, (top + 3 * zoom) * SS)], fill=(62, 64, 74), width=int(1 * SS))
    dr.line([(0, bot * SS), (SW, bot * SS)], fill=(18, 18, 22), width=int(2 * SS))
    dr.line([(0, (bot - 3 * zoom) * SS), (SW, (bot - 3 * zoom) * SS)], fill=(74, 78, 90), width=int(1 * SS))
    th_h = 16 * zoom
    x_world0 = cam_x - W / 2 / zoom - PITCH
    n0 = int(math.floor(x_world0 / PITCH - T - 0.5))
    count = int(W / zoom / PITCH) + 4
    hw_r, hw_t = 0.30 * PITCH * zoom, 0.18 * PITCH * zoom
    for n in range(n0, n0 + count):
        xw = (n + 0.5 + T) * PITCH
        sx = W / 2 + (xw - cam_x) * zoom
        lap_mark = bool(cyc) and (n % cyc) == 0
        lit = (168, 172, 186) if lap_mark else (122, 126, 140)
        dark = (112, 116, 130) if lap_mark else (66, 70, 82)
        tip = (210, 214, 226) if lap_mark else (150, 154, 168)
        for face, sgn in ((top, -1), (bot, 1)):
            ty = face + sgn * th_h
            dr.polygon([((sx - hw_r) * SS, face * SS), ((sx - hw_t) * SS, ty * SS), (sx * SS, ty * SS), (sx * SS, face * SS)], fill=lit)
            dr.polygon([(sx * SS, face * SS), (sx * SS, ty * SS), ((sx + hw_t) * SS, ty * SS), ((sx + hw_r) * SS, face * SS)], fill=dark)
            dr.line([((sx - hw_t) * SS, ty * SS), ((sx + hw_t) * SS, ty * SS)], fill=tip, width=max(1, int(1.2 * SS)))
        if lap_mark:
            dr.rectangle([(sx - 2 * zoom) * SS, (top + 4 * zoom) * SS, (sx + 2 * zoom) * SS, (bot - 4 * zoom) * SS], fill=(150, 154, 168))
        if n % 6 == 3:
            rr = max(2.0, 4.5 * zoom)
            cy = RACK_Y
            dr.ellipse([(sx - rr) * SS, (cy - rr) * SS, (sx + rr) * SS, (cy + rr) * SS], fill=(22, 22, 28))
            dr.ellipse([(sx - rr * 0.7) * SS, (cy - rr * 0.7) * SS, (sx + rr * 0.7) * SS, (cy + rr * 0.7) * SS], fill=(88, 92, 104))
            dr.ellipse([(sx - rr * 0.35) * SS, (cy - rr * 0.6) * SS, (sx + rr * 0.1) * SS, (cy - rr * 0.1) * SS], fill=(150, 154, 168))
    for g in song.gears:
        if not g.engaged(max(0, n_now)):
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
    for g in song.gears:
        st = gear_state(song, g, T)
        if st is None:
            continue
        x, y, Te, moving = st
        states[g.key] = st
        d = 2 * math.pi / g.teeth
        rot = (Te * d - math.pi / 2) if g.side > 0 else (math.pi / 2 - Te * d)
        spr = A.gear_at_zoom(g.key, zoom)
        img = spr.rotate(math.degrees(rot), resample=Image.BICUBIC)
        sx, sy = to_screen(x, y)
        if sx * SS + img.width / 2 < 0 or sx * SS - img.width / 2 > SW:
            continue
        canvas.paste(img, (int(sx * SS - img.width / 2), int(sy * SS - img.height / 2)), img)
    dr = ImageDraw.Draw(canvas)

    # --- hits: flashes at each engaged gear's contact with the rack
    size_scale = min(1.3, max(0.6, zoom / 0.65))
    for n in range(max(0, n_now - 5), n_now + 1):
        for g in song.gears:
            if not g.engaged(n) or g.key not in states:
                continue
            kinds = song.hits_at(g.key, n)
            if not kinds:
                continue
            kind = kinds[0]
            fx, fy = to_screen(g.x, -g.side * RH)
            small = g.teeth < 40
            for frac in song.subhits(g.key, n) or (0.0,):
                age = time_s - song.step_time(n + frac)
                if age < 0:
                    continue
                if kind == "pulse":
                    dur = 0.36
                    if age < dur:
                        e = (1 - age / dur) ** 1.6 * live
                        add_sprite(canvas, A.glow_big[g.key], fx, fy + 6, 0.75 * e, size_scale)
                        rr = 40 + int(age / dur * 300) // 10 * 10
                        if rr in A.rings.get(g.key, {}):
                            add_sprite(canvas, A.rings[g.key][rr], fx, fy, 0.8 * e, size_scale)
                elif kind == "crash":
                    if age < 0.45:
                        e = (1 - age / 0.45) ** 1.5
                        add_sprite(canvas, A.glow_big["crash:" + g.key], fx, fy, 0.95 * e, size_scale)
                        add_sprite(canvas, A.glow_crash_core, fx, fy, e ** 0.7, size_scale)
                        rr = 40 + int(age / 0.45 * 300) // 10 * 10
                        if rr in A.rings_home:
                            add_sprite(canvas, A.rings_home[rr], fx, fy, 0.7 * e, size_scale)
                elif kind in ("kick", "snare"):
                    dur = 0.22 if kind == "snare" else 0.28
                    if age < dur:
                        e = (1 - age / dur) ** 1.5 * live
                        add_sprite(canvas, A.glow_big[g.key], fx, fy, 0.8 * e, size_scale * (0.75 if small else 1.0))
                else:
                    dur, lvl = SMALL_GLOW.get(kind, (0.16, 0.6))
                    if age < dur:
                        e = (1 - age / dur) ** 1.5 * live
                        add_sprite(canvas, A.glow[g.key], fx, fy, lvl * e, size_scale)
    if song.slip:
        hx, hy = to_screen(song.by_key[song.slip.cycle_gear].x, 0.0)
        for hn in song.homes:
            age = age_of(hn)
            if 0 <= age < 1.1:
                e = (1 - age / 1.1) ** 1.3
                add_sprite(canvas, A.glow_home, hx, hy, 0.8 * e, size_scale)
                rr = 40 + int(age / 1.1 * 640) // 10 * 10
                if rr in A.rings_home:
                    add_sprite(canvas, A.rings_home[rr], hx, hy, 0.9 * e, size_scale)

    # --- gear labels
    for g in song.gears:
        if g.key not in states:
            continue
        x, y, Te, moving = states[g.key]
        sx, sy = to_screen(x, y)
        if sx < -200 or sx > W + 200:
            continue
        a = (0.55 if moving else 1.0) * live
        if g.teeth >= 40:
            f_num = A.f("mono_b", min(70, max(38, 95 * zoom)))
            f_name = A.f("display_sb", min(34, max(22, 44 * zoom)))
            gap = 18 * min(1.4, max(0.8, zoom / 0.55))
            dr.text((sx * SS, (sy - gap) * SS), str(g.teeth), font=f_num, fill=scale(TEXT, a), anchor="ms")
            dr.text((sx * SS, (sy + gap) * SS), g.short, font=f_name, fill=scale(g.colour, a), anchor="mt")
        else:
            up = g.side > 0
            base = sy + (r_tip(g) * zoom + 30 + g.label_dy) * (-1 if up else 1)
            lx_ = sx + g.label_dx
            if up:
                dr.text((lx_ * SS, base * SS), g.short, font=A.f_label, fill=scale(g.colour, a), anchor="ms")
                dr.text((lx_ * SS, (base - 42) * SS), f"{g.teeth} TEETH", font=A.f_label_s, fill=scale(DIM, a), anchor="ms")
            else:
                dr.text((lx_ * SS, base * SS), g.short, font=A.f_label, fill=scale(g.colour, a), anchor="mt")
                dr.text((lx_ * SS, (base + 42) * SS), f"{g.teeth} TEETH", font=A.f_label_s, fill=scale(DIM, a), anchor="mt")

    # --- slip bracket at each cycle start once the beat is in play
    sl = song.slip_at(T) if song.slip else 0
    if song.slip and cyc:
        tl = T % cyc
        t_in_lap = tl * step_s
        in_region = k_ref + cyc <= T < (song.cycle_end if song.cycle_end is not None else song.total_steps)
        if in_region and t_in_lap < 1.9 and sl > 0:
            key = next((kk for kk in song.slip.beat_gears if song.by_key[kk].engaged(T)), None)
            if key and key in states and not states[key][3]:
                g = song.by_key[key]
                a = min(ease(t_in_lap / 0.25), ease((1.9 - t_in_lap) / 0.4)) * live
                col = scale((255, 255, 255), a)
                gx, gy = to_screen(*rest_pos(g))
                r_arc = (r_tip(g) + 16) * zoom
                d = 2 * math.pi / g.teeth
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
    idx = song.moment_index(T)
    ka, kb = song.moment_window(idx)
    mo = song.moments[idx]
    fade_out = 0.35 / step_s
    fade_in = 0.0 if mo.kind in ("hook", "big", "home") else 0.15 / step_s
    alpha = min(ease((T - ka) / fade_in) if fade_in else 1.0, ease((kb - T) / fade_out))
    if idx == len(song.moments) - 1:
        alpha = min(ease((T - ka) / fade_in) if fade_in else 1.0, 1.0 - end_dim)
    title_col = TEXT
    if mo.kind == "home":
        title_col = mix(TEXT, GOLD, 0.8)
    elif mo.colour is not None:
        col = song.by_key[mo.colour].colour if isinstance(mo.colour, str) else mo.colour
        title_col = mix(TEXT, col, 0.85 if mo.kind == "big" else (0.8 if mo.colour == "ride" else 0.7))
    kick_px = 0.0
    if beat_steps:
        nb = (n_now // beat_steps) * beat_steps
        age_b = age_of(nb)
        if any(song.by_key[kk].engaged(n_now) for kk in song.slip.beat_gears) and age_b < 0.18:
            kick_px = (1 - age_b / 0.18) * 4
    y = 112 - kick_px
    for line in wrap(mo.title, A.f_title, 940):
        dr.text((W / 2 * SS, y * SS), line, font=A.f_title, fill=scale(title_col, alpha), anchor="mm")
        y += 70
    y += 8
    for line in wrap(mo.subtitle, A.f_sub, 900):
        dr.text((W / 2 * SS, y * SS), line, font=A.f_sub, fill=scale(DIM, alpha), anchor="mm")
        y += 42

    # --- readout strip: row A counter + lap, row B slip (+ the cut note), row C the stack
    lx, ry = 60, 1392
    dot_r, dot_dx = 12, 38
    ctr = song.counter_at(max(0, n_now))
    lap_x = 330                                   # LAP sits right of the counter's widest bar
    for c in song.counters:
        if c.k_from <= max(0, n_now) < c.k_to:
            lap_x = lx + 50 + max(c.bars) * dot_dx + 30
    if ctr:
        blen, pos, slots = ctr
        dr.text((lx * SS, ry * SS), str(blen), font=A.f_group, fill=scale(TEXT, live), anchor="lm")
        for q, colour in enumerate(slots):
            x0 = lx + 50 + q * dot_dx
            reached = q <= pos
            if colour is None:
                dr.ellipse([(x0 - dot_r) * SS, (ry - dot_r) * SS, (x0 + dot_r) * SS, (ry + dot_r) * SS], outline=(TEXT if reached else DIMMER), width=int(2.5 * SS))
            else:
                dr.ellipse([(x0 - dot_r) * SS, (ry - dot_r) * SS, (x0 + dot_r) * SS, (ry + dot_r) * SS], fill=(scale(colour, live) if reached else DIMMER))
    la = ease((time_s - song.step_time(k_ref)) / 0.5) * live if k_ref is not None else 0.0
    if la > 0:
        dr.text((lap_x * SS, ry * SS), song.lap_label(T), font=A.f_lap, fill=scale(TEXT, la), anchor="lm")
        sy_b = 1446
        dr.text((lx * SS, sy_b * SS), "SLIP", font=A.f_mono, fill=scale(DIM, la), anchor="lm")
        laps = sp[2]
        for i in range(laps):
            x0 = lx + 90 + i * 36
            box = [x0 * SS, (sy_b - 13) * SS, (x0 + 26) * SS, (sy_b + 13) * SS]
            if i < sl:
                dr.rectangle(box, fill=scale(song.by_key[song.slip.beat_gears[0]].colour, la))
            else:
                dr.rectangle(box, outline=scale(DIM, la), width=int(2 * SS))
        label = "HOME" if sl == 0 else (f"{sl} TOOTH" if sl == 1 else f"{sl} TEETH")
        dr.text(((lx + 90 + laps * 36 + 10) * SS, sy_b * SS), label, font=A.f_mono,
                fill=scale(song.by_key[song.slip.beat_gears[0]].colour if sl else HOME_WHITE, la), anchor="lm")
    sy = 1502
    xcur = lx
    for key in song.stack:
        g = song.by_key[key]
        members = [g2 for g2 in song.gears if (g2.stack or g2.key) == key]
        on = any(g2.engaged(n_now) for g2 in members)
        col = mix(DIMMER, g.colour, (1.0 if on else 0.0) * live)
        lit = 0.0
        for n in range(max(0, n_now - 2), n_now + 1):
            for g2 in members:
                if g2.engaged(n) and song.hits_at(g2.key, n):
                    lit = max(lit, max(0.0, 1 - age_of(n) / 0.16))
        dcol = mix(col, (255, 255, 255), lit * 0.8 * live) if on else DIMMER
        rad = 7 + 3 * lit
        dr.ellipse([(xcur + 8 - rad) * SS, (sy - rad) * SS, (xcur + 8 + rad) * SS, (sy + rad) * SS], fill=dcol)
        dr.text(((xcur + 24) * SS, sy * SS), g.name, font=A.f_mono, fill=(col if on else DIMMER), anchor="lm")
        xcur += 24 + A.f_mono.getlength(g.name) / SS + 20
    if song.cut_note:
        cut_k = next((song.seg_film_start[i] for i in range(1, len(song.segments))
                      if song.segments[i].record_start != song.segments[i - 1].record_start + song.segments[i - 1].steps), None)
        if cut_k is not None:
            age_cut = age_of(cut_k)
            if 1.9 <= age_cut < 4.6:
                a = min(ease((age_cut - 1.9) / 0.3), ease((4.6 - age_cut) / 0.4))
                dr.text((440 * SS, 1446 * SS), song.cut_note, font=A.f_mono_s, fill=scale(DIM, a), anchor="lm")

    # --- closing card over the audio fade
    if end_dim > 0:
        canvas = Image.blend(canvas, Image.new("RGB", canvas.size, (0, 0, 0)), 0.72 * end_dim)
        dr = ImageDraw.Draw(canvas)
        a = ease((end_dim - 0.15) / 0.7)
        cy = 980
        dr.text((W / 2 * SS, (cy - 90) * SS), song.artist.upper(), font=A.f_card1, fill=scale(TEXT, a), anchor="mm")
        dr.text((W / 2 * SS, (cy - 24) * SS), song.title.upper(), font=A.f_card2, fill=scale(TEXT, a), anchor="mm")
        dr.text((W / 2 * SS, (cy + 24) * SS), f"{song.album}, {song.year}", font=A.f_card3, fill=scale(DIM, a), anchor="mm")
        yy = cy + 96
        for line in song.card_lines:
            dr.text((W / 2 * SS, yy * SS), line, font=A.f_card3, fill=scale(GOLD, a), anchor="mm")
            yy += 42
        for line in song.credit_lines:
            dr.text((W / 2 * SS, yy * SS), line, font=A.f_mono_s, fill=scale(DIM, a * 0.8), anchor="mm")
            yy += 36

    return canvas.resize((W, H), Image.LANCZOS)

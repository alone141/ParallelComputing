"""Frame renderer, round 3: the picture for Gojira's own recording (silent film, 60 fps).

Riff wheel (top, steel, 45 slots): teeth = rim clicks (11 missing: the rests); pin track 1 =
    kick (red) / snare (white); pin track 2 = bass (violet) with held arcs through the rests.
Beat wheel (bottom, bronze, 44 teeth): every fourth tooth amber = the tribal pulse; track 1 =
    ride (bell / ghost pins); track 2 = guitars (every tooth).
Turnaround wheel (small, upper right of the riff wheel, 12 teeth): teeth = the turnaround bass
    (C1 dark violet, C2 light violet, in threes); one pin track = K K K S . S twice round.
All three advance one tooth per sixteenth on the record's clock (rhythm.k_at). A follower on a
track is engaged only while the record plays that part: laps 1-4 clicks, laps 5-12 clicks +
pulse, laps 13-16 the band on the inner tracks, then the turnaround wheel for 24 sixteenths.

Usage: python3 render.py [--frames a:b] [--still N ...] [--out DIR] [--workers 4]
"""
import argparse
import math
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

import rhythm as R

W, H = 1080, 1920
SS = 2
SW, SH = W * SS, H * SS

# ---- geometry (1x units) ------------------------------------------------------------------------
PITCH = 42.0
MODULE = PITCH / math.pi
R_PITCH_A = R.CYCLE * MODULE / 2          # 300.8
R_PITCH_B = 44 * MODULE / 2               # 294.1
R_PITCH_C = 12 * MODULE / 2               # 80.2
ADDENDUM, DEDENDUM = 18.0, 20.0
CX = 600.0
CY_A = 690.0
CY_B = CY_A + R_PITCH_A + R_PITCH_B
MESH = (CX, CY_A + R_PITCH_A)
TH_C = math.radians(-40.0)                # direction from the riff wheel's centre to the turnaround wheel
CX_C = CX + (R_PITCH_A + R_PITCH_C) * math.cos(TH_C)
CY_C = CY_A + (R_PITCH_A + R_PITCH_C) * math.sin(TH_C)
MESH_C = (CX + R_PITCH_A * math.cos(TH_C), CY_A + R_PITCH_A * math.sin(TH_C))
DA, DB, DC = 2 * math.pi / R.CYCLE, 2 * math.pi / 44, 2 * math.pi / 12
PHASE = 0.25                              # A centred at T=k+.25, B at T=j-.25; C at T=s exactly
TRACK_R = {1: 246.0, 2: 204.0}
TRACK_R_C = {1: 46.0}
HUB_R, HUB_R_C = 118.0, 24.0
PIN_RL, PIN_HW = 11.0, 6.5

# ---- palette ------------------------------------------------------------------------------------
BG = (9, 9, 12)
STEEL, STEEL_EDGE = (58, 61, 70), (122, 128, 142)
BRONZE, BRONZE_EDGE = (72, 58, 44), (150, 122, 92)
AMBER = R.BY_KEY["pulse"].colour
HOME_WHITE = (238, 246, 255)
GOLD = (255, 225, 160)
TEXT, DIM, DIMMER = (236, 236, 240), (150, 150, 160), (96, 96, 106)

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


# ---- gear sprites (bodies only; colour is drawn per frame) --------------------------------------
def gear_sprite(n, present, r_pitch, body, edge, tracks, hub_r, ccw_index=False):
    r_tip, r_root = r_pitch + ADDENDUM, r_pitch - DEDENDUM
    size = int((r_tip + 12) * 2 * SS)
    c = size / 2
    d = 2 * math.pi / n
    sgn = -1 if ccw_index else 1
    hw_root, hw_tip = 0.44 * d, 0.26 * d
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    pts = []
    for k in range(n):
        a = sgn * k * d
        if k in present:
            seq = [(r_root, -0.5 * d), (r_root, -hw_root), (r_tip, -hw_tip), (r_tip, hw_tip),
                   (r_root, hw_root), (r_root, 0.5 * d)]
        else:
            seq = [(r_root, -0.5 * d), (r_root + 2.5, -0.3 * d), (r_root + 2.5, 0.3 * d), (r_root, 0.5 * d)]
        if ccw_index:
            seq = [(r, -t) for (r, t) in seq][::-1]
        for r, th in seq:
            pts.append(polar(c, c, r * SS, a + th))
    dr.polygon(pts, fill=body + (255,))
    arr = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.hypot(xx - c, yy - c) / (r_tip * SS)
    ang = np.arctan2(yy - c, xx - c)
    radial = 0.74 + 0.5 * np.clip(rr, 0, 1) ** 2.2
    sheen = 1.0 + 0.2 * np.cos(ang + 2.3) * np.clip(rr, 0, 1)
    arr[..., :3] = np.clip(arr[..., :3] * (radial * sheen)[..., None], 0, 255)
    img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    dr = ImageDraw.Draw(img)

    def ring(r, col, w):
        dr.ellipse([c - r * SS, c - r * SS, c + r * SS, c + r * SS], outline=col, width=int(w * SS))

    ring(r_root, edge + (110,), 1.5)
    for tr in tracks:
        gw = 30 if tr > 100 else 20
        ring(tr, (0, 0, 0, 70), gw)
        ring(tr + gw / 2, edge + (80,), 1.2)
        ring(tr - gw / 2, (0, 0, 0, 120), 1.2)
    ring(hub_r, edge + (120,), 2)
    if hub_r > 60:
        ring(hub_r - 8, (0, 0, 0, 90), 6)
        for i in range(6):
            p = polar(c, c, (hub_r - 28) * SS, i * math.pi / 3 + 0.2)
            dr.ellipse([p[0] - 5 * SS, p[1] - 5 * SS, p[0] + 5 * SS, p[1] + 5 * SS], fill=(18, 18, 22, 255))
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
    d.ellipse([c - radius * SS, c - radius * SS, c + radius * SS, c + radius * SS], outline=colour,
              width=int(width * SS))
    return img.filter(ImageFilter.GaussianBlur(1.2 * SS))


def add_sprite(canvas, sprite, x, y, gain=1.0):
    if gain <= 0.01:
        return
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
        self.gear_a = gear_sprite(R.CYCLE, set(R.RIM_HITS), R_PITCH_A, STEEL, STEEL_EDGE, TRACK_R.values(), HUB_R)
        self.gear_b = gear_sprite(44, set(range(44)), R_PITCH_B, BRONZE, BRONZE_EDGE, TRACK_R.values(), HUB_R, ccw_index=True)
        self.gear_c = gear_sprite(12, set(range(12)), R_PITCH_C, STEEL, STEEL_EDGE, TRACK_R_C.values(), HUB_R_C, ccw_index=True)
        self.glow = {k: glow_sprite(90, R.BY_KEY[k].colour, 2.0) for k in R.KEYS}
        self.glow["snare"] = glow_sprite(90, R.SNARE_COLOUR, 2.0)
        self.glow_big = {"pulse": glow_sprite(150, AMBER, 2.0), "kick": glow_sprite(120, R.BY_KEY["ks"].colour, 2.0),
                         "snare": glow_sprite(130, (255, 255, 255), 2.2), "crash": glow_sprite(520, (255, 250, 240), 1.4)}
        self.glow_small = {"kick": glow_sprite(70, R.BY_KEY["ks"].colour, 2.0), "snare": glow_sprite(75, (255, 255, 255), 2.2)}
        self.glow_home = glow_sprite(420, GOLD, 1.6)
        self.rings = {r: ring_sprite(r, 5, AMBER) for r in range(40, 340, 10)}
        self.rings_home = {r: ring_sprite(r, 7, (255, 236, 190)) for r in range(40, 700, 10)}
        self.bg = self.make_bg()
        self.f_title = font("display", 62)
        self.f_sub = font("text", 32)
        self.f_lap = font("display", 36)
        self.f_mono = font("mono", 24)
        self.f_mono_s = font("mono", 21)
        self.f_hub = font("mono_b", 58)
        self.f_hub_s = font("display_sb", 24)
        self.f_hub_c = font("mono_b", 22)
        self.f_group = font("display", 46)
        self.f_chip = font("display_sb", 26)
        self.f_card1 = font("display", 78)
        self.f_card2 = font("display_sb", 40)
        self.f_card3 = font("text", 30)

    def make_bg(self):
        yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
        v = np.hypot((xx - SW / 2) / SW, (yy - SH * 0.53) / SH)
        shade = np.clip(1.0 - v * 1.1, 0.0, 1.0)
        arr = np.array(BG, np.float32)[None, None, :] * (0.55 + 0.9 * shade[..., None])
        m = np.exp(-(((xx - MESH[0] * SS) / (420 * SS)) ** 2 + ((yy - MESH[1] * SS) / (300 * SS)) ** 2))
        arr += (m * 14)[..., None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


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


# ---- wheel coordinates ----------------------------------------------------------------------------
def angle_a(k, T):
    return math.pi / 2 + (k + PHASE - T) * DA


def angle_b(j, T):
    return -math.pi / 2 + (T - j + PHASE) * DB


def angle_c(s, T):
    return (TH_C + math.pi) + (T - s) * DC


def pin_poly(cx, cy, r, th, rl=PIN_RL, hw=PIN_HW):
    dth = hw / r
    return [polar(cx * SS, cy * SS, (r + s * rl) * SS, th + w * dth) for s, w in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def tooth_poly(cx, cy, r_pitch, th, d, inset=2.0):
    r_tip, r_root = r_pitch + ADDENDUM - 1.0, r_pitch - DEDENDUM + inset
    hw_root, hw_tip = 0.44 * d * 0.92, 0.26 * d * 0.92
    seq = [(r_root, -hw_root), (r_tip, -hw_tip), (r_tip, hw_tip), (r_root, hw_root)]
    return [polar(cx * SS, cy * SS, r * SS, th + t) for r, t in seq]


def arc_points(cx, cy, r, th0, th1, steps=12):
    return [polar(cx * SS, cy * SS, r * SS, th0 + (th1 - th0) * i / steps) for i in range(steps + 1)]


def follower_pos(key):
    L = R.BY_KEY[key]
    if L.wheel == "turn":
        if L.track == 0:
            return MESH_C
        return polar(CX_C, CY_C, TRACK_R_C[1], TH_C + math.pi)
    if L.track == 0:
        return MESH
    r = TRACK_R[L.track]
    return (CX, CY_A + r) if L.wheel == "riff" else (CX, CY_B - r)


# ---- the frame ----------------------------------------------------------------------------------
def render_frame(A, f):
    T = R.k_at(f)                                  # continuous sixteenth on the record's clock
    time_s = f / R.FPS
    n_now = int(math.floor(T))
    end_dim = ease((time_s - (R.TOTAL - R.END_FADE_S)) / R.END_FADE_S)   # 0 .. 1 over the last 1.2 s
    live = 1.0 - 0.85 * end_dim                    # lights go down with the audio fade

    canvas = A.bg.copy()
    rot_a = math.degrees((T - PHASE) * DA - math.pi / 2)
    rot_b = math.degrees(math.pi / 2 - (T + PHASE) * DB)
    rot_c = math.degrees(-(TH_C + math.pi) - T * DC)
    for sprite, rot, cx, cy in ((A.gear_a, rot_a, CX, CY_A), (A.gear_b, rot_b, CX, CY_B), (A.gear_c, rot_c, CX_C, CY_C)):
        g = sprite.rotate(rot, resample=Image.BICUBIC)
        canvas.paste(g, (int(cx * SS - g.width / 2), int(cy * SS - g.height / 2)), g)
    dr = ImageDraw.Draw(canvas)

    pres = {k: R.presence(k, T) for k in R.KEYS}

    def pin_colour(key, base, slot_T, floor=0.22):
        b = floor + (1 - floor) * pres[key]
        col = mix((40, 40, 46), base, b * live)
        age = (T - slot_T) * R.SIXTEENTH
        if 0 <= age < 0.12 and pres[key] > 0.05:
            col = mix(col, (255, 255, 255), (1 - age / 0.12) * 0.9 * live)
        return col

    k0 = n_now - R.CYCLE // 2
    for k in range(k0, k0 + R.CYCLE):
        th = angle_a(k, T)
        t = k % R.CYCLE
        if t == 0:
            dr.polygon(tooth_poly(CX, CY_A, R_PITCH_A, th, DA), fill=pin_colour("rim", HOME_WHITE, k, floor=0.6))
        if t in R.SNARES:
            dr.polygon(pin_poly(CX, CY_A, TRACK_R[1], th, rl=12.5, hw=7.5), fill=pin_colour("ks", R.SNARE_COLOUR, k))
        elif t in R.KICKS:
            dr.polygon(pin_poly(CX, CY_A, TRACK_R[1], th), fill=pin_colour("ks", R.BY_KEY["ks"].colour, k))
        if t not in R.RIM_RESTS:
            dr.polygon(pin_poly(CX, CY_A, TRACK_R[2], th), fill=pin_colour("bass", R.BY_KEY["bass"].colour, k))
            if (t + 1) % R.CYCLE in R.RIM_RESTS:
                hold = mix((40, 40, 46), R.BY_KEY["bass"].colour, (0.22 + 0.78 * pres["bass"]) * live)
                dr.line(arc_points(CX, CY_A, TRACK_R[2], th, th - DA * 0.95), fill=hold, width=int(5 * SS))
    j0 = n_now - 22
    for j in range(j0, j0 + 44):
        th = angle_b(j, T)
        if j % R.BEAT == 0:
            dr.polygon(tooth_poly(CX, CY_B, R_PITCH_B, th, DB), fill=pin_colour("pulse", AMBER, j, floor=0.28))
            dr.polygon(pin_poly(CX, CY_B, TRACK_R[1], th, rl=12.5, hw=7.5), fill=pin_colour("ride", R.BY_KEY["ride"].colour, j))
        elif j % R.BEAT == 2:
            dr.polygon(pin_poly(CX, CY_B, TRACK_R[1], th, rl=7, hw=4.5), fill=pin_colour("ride", R.BY_KEY["ride"].colour, j))
        dr.polygon(pin_poly(CX, CY_B, TRACK_R[2], th, rl=8, hw=5), fill=pin_colour("gtr", R.BY_KEY["gtr"].colour, j))
    s0 = n_now - 6
    for s in range(s0, s0 + 12):
        th = angle_c(s, T)
        c6 = (s - R.K_TURN) % 6
        base = R.BY_KEY["tbass"].colour if ((s - R.K_TURN) // 3) % 2 == 0 else R.BASS_C2_COLOUR
        dr.polygon(tooth_poly(CX_C, CY_C, R_PITCH_C, th, DC, inset=3.0), fill=pin_colour("tbass", base, s, floor=0.3))
        if c6 in R.TURN_KICKS:
            dr.polygon(pin_poly(CX_C, CY_C, TRACK_R_C[1], th, rl=7, hw=4.5), fill=pin_colour("tks", R.BY_KEY["tks"].colour, s))
        elif c6 in R.TURN_SNARES:
            dr.polygon(pin_poly(CX_C, CY_C, TRACK_R_C[1], th, rl=8, hw=5), fill=pin_colour("tks", R.SNARE_COLOUR, s))

    # --- hits: flashes at the followers
    for n in range(max(0, n_now - 5), n_now + 1):
        age = (T - n) * R.SIXTEENTH
        for key in R.KEYS:
            if not R.engaged(key, n):
                continue
            kinds = R.hits_at(key, n)
            if not kinds:
                continue
            kind = kinds[0]
            fx, fy = follower_pos(key)
            small = R.BY_KEY[key].wheel == "turn"
            if kind == "pulse":
                dur = 0.36
                if age < dur:
                    e = (1 - age / dur) ** 1.6 * live
                    add_sprite(canvas, A.glow_big["pulse"], fx, fy + 6, 0.75 * e)
                    rr = 40 + int(age / dur * 300) // 10 * 10
                    if rr in A.rings:
                        add_sprite(canvas, A.rings[rr], fx, fy, 0.8 * e)
            elif kind == "crash":
                if age < 0.9:
                    e = (1 - age / 0.9) ** 1.8
                    add_sprite(canvas, A.glow_big["crash"], MESH[0], MESH[1], 0.9 * e)
                    add_sprite(canvas, A.glow["ride"], fx, fy, e)
            elif kind in ("kick", "snare"):
                dur = 0.22 if kind == "snare" else 0.28
                if age < dur:
                    e = (1 - age / dur) ** 1.5 * live
                    add_sprite(canvas, A.glow_small[kind] if small else A.glow_big[kind], fx, fy, 0.8 * e)
            else:
                dur = {"click": 0.14, "bell": 0.3, "ride": 0.16, "note": 0.22, "chug": 0.09}[kind]
                lvl = {"click": 0.9, "bell": 0.8, "ride": 0.35, "note": 0.7, "chug": 0.3}[kind]
                if age < dur:
                    e = (1 - age / dur) ** 1.5 * live
                    add_sprite(canvas, A.glow[key], fx, fy - (6 if key == "rim" else 0), lvl * e * (0.7 if small else 1))
    for hn in R.HOMES:
        age = (T - hn) * R.SIXTEENTH
        if 0 <= age < 1.1:
            e = (1 - age / 1.1) ** 1.3
            add_sprite(canvas, A.glow_home, MESH[0], MESH[1], 0.8 * e)
            rr = 40 + int(age / 1.1 * 640) // 10 * 10
            if rr in A.rings_home:
                add_sprite(canvas, A.rings_home[rr], MESH[0], MESH[1], 0.9 * e)

    # --- followers: engaged = on the track in colour; lifted = a step back, grey
    for key in R.KEYS:
        L = R.BY_KEY[key]
        p = pres[key]
        col = mix(DIMMER, L.colour, p * live)
        fx, fy = follower_pos(key)
        if L.wheel == "turn":
            # points along the line toward the turnaround wheel's centre
            d = TH_C          # from the wheel centre toward the mesh: the follower sits on the mesh side
            lift = (1 - p) * 8
            if L.track == 0:
                base_r, tip_r = -(30 + lift), -(14 + lift)      # outside the teeth, toward the riff wheel... (negative = toward A)
                b = polar(fx, fy, 0, 0)
                ux, uy = math.cos(TH_C), math.sin(TH_C)          # unit vector A -> C
                base = (fx - ux * (30 + lift), fy - uy * (30 + lift))
                tip = (fx - ux * (14 + lift), fy - uy * (14 + lift))
            else:
                ux, uy = math.cos(TH_C), math.sin(TH_C)
                base = (fx - ux * (22 + lift), fy - uy * (22 + lift))
                tip = (fx - ux * (8 + lift), fy - uy * (8 + lift))
            px, py = -uy, ux
            dr.polygon([((base[0] + px * 8) * SS, (base[1] + py * 8) * SS), ((base[0] - px * 8) * SS, (base[1] - py * 8) * SS),
                        (tip[0] * SS, tip[1] * SS)], fill=col)
        elif L.track == 0:
            up = 1 if L.wheel == "riff" else -1
            lift = (1 - p) * 10
            base_y, tip_y = fy - up * (54 + lift), fy - up * (36 + lift)
            dr.polygon([((fx - 13) * SS, base_y * SS), ((fx + 13) * SS, base_y * SS), (fx * SS, tip_y * SS)], fill=col)
        else:
            up = -1 if L.wheel == "riff" else 1
            lift = (1 - p) * 10
            base_y, tip_y = fy - up * (26 + lift), fy - up * (12 + lift)
            dr.polygon([((fx - 10) * SS, base_y * SS), ((fx + 10) * SS, base_y * SS), (fx * SS, tip_y * SS)], fill=col)
            dr.line([(fx * SS, base_y * SS), (fx * SS, (base_y - up * 10) * SS)], fill=col, width=int(3 * SS))
        # name chip for 2.2 s after the part engages
        age_in = (T - L.k_in) * R.SIXTEENTH
        if key not in ("rim", "tbass") and 0 <= age_in < 2.2:
            a = min(ease(age_in / 0.2), ease((2.2 - age_in) / 0.4))
            label = L.name
            tw = A.f_chip.getlength(label) / SS
            if L.wheel == "turn":
                x0, y0 = fx - tw - 60, fy + 30
            else:
                x0, y0 = fx + 28, fy - 17
            dr.rounded_rectangle([x0 * SS, y0 * SS, (x0 + tw + 24) * SS, (y0 + 34) * SS], radius=8 * SS, fill=(14, 14, 18))
            dr.text(((x0 + 12) * SS, (y0 + 17) * SS), label, font=A.f_chip, fill=scale(L.colour, a), anchor="lm")

    # --- hub labels
    dr.text((CX * SS, (CY_A - 12) * SS), "45", font=A.f_hub, fill=scale(TEXT, live), anchor="mm")
    dr.text((CX * SS, (CY_A + 30) * SS), "THE RIFF", font=A.f_hub_s, fill=scale(DIM, live), anchor="mm")
    pb = (0.35 + 0.65 * max(pres["pulse"], pres["ride"])) * live
    dr.text((CX * SS, (CY_B - 12) * SS), "44", font=A.f_hub, fill=scale(AMBER, pb), anchor="mm")
    dr.text((CX * SS, (CY_B + 30) * SS), "THE BEAT", font=A.f_hub_s, fill=scale(DIM, pb), anchor="mm")
    pc = (0.35 + 0.65 * pres["tks"]) * live
    dr.text((CX_C * SS, CY_C * SS), "12", font=A.f_hub_c, fill=scale(TEXT, pc), anchor="mm")

    # --- captions (top)
    idx = R.caption_index(T)
    ka, kb = R.caption_window(idx)
    fade = 0.35 / R.SIXTEENTH
    fade_in = 0.0 if idx in (0, 4, 8, 12) else 0.15 / R.SIXTEENTH   # the big moments land at once
    alpha = min(ease((T - ka) / fade_in) if fade_in else 1.0, ease((kb - T) / fade))
    if idx == R.N_LAPS_CYCLE:
        alpha = min(ease((T - ka) / fade), 1.0 - end_dim)
    title, sub = R.CAPTIONS[idx]
    title_col = TEXT
    if title.startswith("HOME"):
        title_col = mix(TEXT, GOLD, 0.8)
    elif idx == 4:
        title_col = mix(TEXT, AMBER, 0.85)
    elif idx in (13, 15, 16):
        title_col = mix(TEXT, R.BY_KEY["ks"].colour, 0.7)
    elif idx == 14:
        title_col = mix(TEXT, R.BY_KEY["ride"].colour, 0.8)
    kick_px = 0.0
    nb = (n_now // R.BEAT) * R.BEAT
    age_b = (T - nb) * R.SIXTEENTH
    if (pres["pulse"] > 0.5 or pres["ride"] > 0.5) and age_b < 0.18:
        kick_px = (1 - age_b / 0.18) * 4
    y = 112 - kick_px
    for line in wrap(title, A.f_title, 940):
        dr.text((W / 2 * SS, y * SS), line, font=A.f_title, fill=scale(title_col, alpha), anchor="mm")
        y += 70
    y += 8
    for line in wrap(sub, A.f_sub, 900):
        dr.text((W / 2 * SS, y * SS), line, font=A.f_sub, fill=scale(DIM, alpha), anchor="mm")
        y += 42

    # --- left column
    lx = 56
    gy = 790
    if T < R.K_TURN:
        gi, gs, gsize = R.group_of(n_now % R.CYCLE)
        p_in = n_now % R.CYCLE - gs
        dr.text((lx * SS, gy * SS), str(gsize), font=A.f_group, fill=scale(TEXT, live), anchor="lm")
        for q in range(gsize):
            x0 = lx + 44 + q * 30
            slot = gs + q
            reached = q <= p_in
            if slot in R.RIM_RESTS:
                dr.ellipse([(x0 - 9) * SS, (gy - 9) * SS, (x0 + 9) * SS, (gy + 9) * SS],
                           outline=(TEXT if reached else DIMMER), width=int(2 * SS))
            else:
                dr.ellipse([(x0 - 9) * SS, (gy - 9) * SS, (x0 + 9) * SS, (gy + 9) * SS],
                           fill=(R.BY_KEY["rim"].colour if reached else DIMMER))
    else:
        c_in = (n_now - R.K_TURN) % 6
        dr.text((lx * SS, gy * SS), "6", font=A.f_group, fill=scale(TEXT, live), anchor="lm")
        for q, cell in enumerate(R.TURN_CELL):
            x0 = lx + 44 + q * 30
            reached = q <= c_in
            if cell == ".":
                dr.ellipse([(x0 - 9) * SS, (gy - 9) * SS, (x0 + 9) * SS, (gy + 9) * SS],
                           outline=(TEXT if reached else DIMMER), width=int(2 * SS))
            else:
                colr = R.BY_KEY["tks"].colour if cell == "K" else R.SNARE_COLOUR
                dr.ellipse([(x0 - 9) * SS, (gy - 9) * SS, (x0 + 9) * SS, (gy + 9) * SS],
                           fill=(scale(colr, live) if reached else DIMMER))
    la = ease((T - R.K_PULSE) * R.SIXTEENTH / 0.5) * live
    if la > 0:
        ry = 846
        lap_txt = f"LAP {min(int(T // R.CYCLE), R.N_LAPS_CYCLE - 1) + 1} / {R.N_LAPS_CYCLE}" if T < R.K_TURN else "6/4 TURN"
        dr.text((lx * SS, ry * SS), lap_txt, font=A.f_lap, fill=scale(TEXT, la), anchor="lm")
        sy = ry + 40
        sl = R.slip(T)
        dr.text((lx * SS, sy * SS), "SLIP", font=A.f_mono_s, fill=scale(DIM, la), anchor="lm")
        for i in range(4):
            x0 = lx + 64 + i * 32
            box = [x0 * SS, (sy - 10) * SS, (x0 + 22) * SS, (sy + 10) * SS]
            if i < sl:
                dr.rectangle(box, fill=scale(AMBER, la))
            else:
                dr.rectangle(box, outline=scale(DIM, la), width=int(1.5 * SS))
        label = "HOME" if sl == 0 else (f"{sl} TOOTH" if sl == 1 else f"{sl} TEETH")
        dr.text(((lx + 64 + 4 * 32 + 6) * SS, sy * SS), label, font=A.f_mono_s,
                fill=scale(AMBER if sl else HOME_WHITE, la), anchor="lm")
    sy0 = 930
    for i, key in enumerate(R.STACK_KEYS):
        L = R.BY_KEY[key]
        p = pres[key]
        yy = sy0 + i * 29
        col = mix(DIMMER, L.colour, p * live)
        lit = 0.0
        for n in range(max(0, n_now - 2), n_now + 1):
            if R.engaged(key, n) and R.hits_at(key, n):
                age = (T - n) * R.SIXTEENTH
                lit = max(lit, max(0.0, 1 - age / 0.16))
        dcol = mix(col, (255, 255, 255), lit * 0.8 * live) if p > 0.02 else DIMMER
        rad = 6 + 3 * lit
        dr.ellipse([(lx + 8 - rad) * SS, (yy - rad) * SS, (lx + 8 + rad) * SS, (yy + rad) * SS], fill=dcol)
        age_in = (T - L.k_in) * R.SIXTEENTH
        hot = 0 <= age_in < 2.2 and key != "rim"
        fw = A.f_mono if hot else A.f_mono_s
        tcol = mix(col, (255, 255, 255), 0.35 if hot else 0.0) if p > 0.02 else DIMMER
        dr.text(((lx + 28) * SS, yy * SS), L.name, font=fw, fill=tcol, anchor="lm")
        if hot:
            dr.polygon([((lx - 10) * SS, (yy - 6) * SS), ((lx - 10) * SS, (yy + 6) * SS), ((lx - 2) * SS, yy * SS)],
                       fill=scale(L.colour, live))

    # --- slip bracket at each cycle start once the beat is in play, and the cut note
    note_y = 1138
    tl = T % R.CYCLE
    t_in_lap = tl * R.SIXTEENTH
    sl = R.slip(T)
    if R.K_PULSE + R.CYCLE <= T < R.K_TURN and t_in_lap < 1.9 and sl > 0:
        a = min(ease(t_in_lap / 0.25), ease((1.9 - t_in_lap) / 0.4)) * live
        col = scale((255, 255, 255), a)
        r_arc = R_PITCH_B + ADDENDUM + 20
        th0 = -math.pi / 2
        th1 = th0 + sl * DB
        dr.line(arc_points(CX, CY_B, r_arc, th0, th1, 30), fill=col, width=int(4 * SS), joint="curve")
        for th in (th0, th1):
            p0 = polar(CX * SS, CY_B * SS, (r_arc - 10) * SS, th)
            p1 = polar(CX * SS, CY_B * SS, (r_arc + 10) * SS, th)
            dr.line([p0, p1], fill=col, width=int(4 * SS))
        word = "beat 1 tooth ahead" if sl == 1 else f"beat {sl} teeth ahead"
        dr.text((lx * SS, note_y * SS), word, font=A.f_mono_s, fill=col, anchor="lm")
    for hn in R.HOMES:
        age = (T - hn) * R.SIXTEENTH
        if 0 <= age < 1.9:
            a = min(ease(age / 0.2), ease((1.9 - age) / 0.5))
            dr.text((lx * SS, note_y * SS), "home: 4 laps, 45 beats", font=A.f_mono_s, fill=scale(HOME_WHITE, a), anchor="lm")
    age_cut = (T - R.K_BAND) * R.SIXTEENTH
    if 1.9 <= age_cut < 4.6:
        a = min(ease((age_cut - 1.9) / 0.3), ease((4.6 - age_cut) / 0.4))
        dr.text((lx * SS, note_y * SS), "cut: bars 34-41 (pulse alone)", font=A.f_mono_s, fill=scale(DIM, a), anchor="lm")

    # --- closing card over the audio fade
    if end_dim > 0:
        overlay = Image.new("RGB", canvas.size, (0, 0, 0))
        canvas = Image.blend(canvas, overlay, 0.72 * end_dim)
        dr = ImageDraw.Draw(canvas)
        a = ease((end_dim - 0.15) / 0.7)
        cy = 980
        dr.text((W / 2 * SS, (cy - 90) * SS), "GOJIRA", font=A.f_card1, fill=scale(TEXT, a), anchor="mm")
        dr.text((W / 2 * SS, (cy - 24) * SS), "THE ART OF DYING", font=A.f_card2, fill=scale(TEXT, a), anchor="mm")
        dr.text((W / 2 * SS, (cy + 24) * SS), "The Way of All Flesh, 2008", font=A.f_card3, fill=scale(DIM, a), anchor="mm")
        dr.text((W / 2 * SS, (cy + 96) * SS), "45 against 44: it drifts one tooth a lap and comes home in four.",
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
            print("still", f, "k=%.2f" % R.k_at(f), "t=%.2fs" % (f / R.FPS), "bar", R.record_bar(max(0, int(R.k_at(f)))))
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

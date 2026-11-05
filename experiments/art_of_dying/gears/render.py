"""Frame renderer, round 2: two meshing wheels that are also music-box discs.

Riff wheel (top, steel, 45 slots): outer teeth = rim clicks (11 teeth missing: the rests);
    pin track 1 = kick (red) and snare (white); pin track 2 = bass (violet), with each group's
    last note drawn as a held arc through the rest.
Beat wheel (bottom, bronze, 44 teeth): every fourth tooth painted amber = the tribal pulse;
    pin track 1 = ride (bell on the beat, a ghost pin between); pin track 2 = guitars (every tooth).
Both wheels advance one tooth per sixteenth. Each track has a follower on the line between the
two centres; a pin or tooth passing under an engaged follower is a hit (a flash, and a sound).
45 = 44 + 1, so the beat wheel gains one tooth per lap on the riff wheel and comes home every four.

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
ADDENDUM, DEDENDUM = 18.0, 20.0
CX = 600.0                                # wheels sit a little right: the left column is for readouts
CY_A = 690.0
CY_B = CY_A + R_PITCH_A + R_PITCH_B
MESH = (CX, CY_A + R_PITCH_A)
DA = 2 * math.pi / R.CYCLE
DB = 2 * math.pi / 44
PHASE = 0.25                              # teeth interleave: A centred at T=k+.25, B at T=j-.25
TRACK_R = {1: 246.0, 2: 204.0}            # pin track radii (same on both wheels)
HUB_R = 118.0
PIN_RL, PIN_HW = 11.0, 6.5                # pin: radial half-length, tangential half-width

# ---- palette ------------------------------------------------------------------------------------
BG = (9, 9, 12)
STEEL, STEEL_EDGE = (58, 61, 70), (122, 128, 142)
BRONZE, BRONZE_EDGE = (72, 58, 44), (150, 122, 92)
AMBER = R.BY_KEY["pulse"].colour
HOME_WHITE = (238, 246, 255)
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


# ---- gear sprites (bodies only; all colour is drawn per frame) ----------------------------------
def gear_sprite(n, present, r_pitch, body, edge, ccw_index=False):
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
        else:   # a missing tooth: the root, with a shallow stump where it was
            seq = [(r_root, -0.5 * d), (r_root + 2.5, -0.3 * d), (r_root + 2.5, 0.3 * d), (r_root, 0.5 * d)]
        if ccw_index:
            seq = [(r, -t) for (r, t) in seq][::-1]
        for r, th in seq:
            pts.append(polar(c, c, r * SS, a + th))
    dr.polygon(pts, fill=body + (255,))
    # brushed-metal shading
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
    # engraved pin tracks: a darker groove with a light upper edge
    for tr in TRACK_R.values():
        ring(tr, (0, 0, 0, 70), 30)
        ring(tr + 15, edge + (80,), 1.2)
        ring(tr - 15, (0, 0, 0, 120), 1.2)
    ring(HUB_R, edge + (120,), 2)
    ring(HUB_R - 8, (0, 0, 0, 90), 6)
    for i in range(6):
        p = polar(c, c, (HUB_R - 28) * SS, i * math.pi / 3 + 0.2)
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
    """Additive blend of an RGB sprite centred at (x, y) in 1x coordinates."""
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
        self.gear_a = gear_sprite(R.CYCLE, set(R.RIM_HITS), R_PITCH_A, STEEL, STEEL_EDGE)
        self.gear_b = gear_sprite(44, set(range(44)), R_PITCH_B, BRONZE, BRONZE_EDGE, ccw_index=True)
        self.glow = {k: glow_sprite(90, R.BY_KEY[k].colour, 2.0) for k in R.KEYS}
        self.glow["snare"] = glow_sprite(90, R.SNARE_COLOUR, 2.0)
        self.glow_big = {"pulse": glow_sprite(150, AMBER, 2.0), "kick": glow_sprite(120, R.BY_KEY["ks"].colour, 2.0),
                         "snare": glow_sprite(130, (255, 255, 255), 2.2)}
        self.glow_home = glow_sprite(420, (255, 225, 160), 1.6)
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
        self.f_group = font("display", 46)
        self.f_chip = font("display_sb", 26)

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
    """Display angle of riff-wheel slot k (screen coords, y down) at sixteenth T."""
    return math.pi / 2 + (k + PHASE - T) * DA


def angle_b(j, T):
    return -math.pi / 2 + (T - j + PHASE) * DB


def pin_poly(cx, cy, r, th, rl=PIN_RL, hw=PIN_HW):
    """A pin: a rounded-ish block at radius r, angle th (4 corners, in SS pixels)."""
    dth = hw / r
    return [polar(cx * SS, cy * SS, (r + s * rl) * SS, th + w * dth) for s, w in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def tooth_poly(cx, cy, r_pitch, th, d, inset=2.0):
    r_tip, r_root = r_pitch + ADDENDUM - 1.0, r_pitch - DEDENDUM + inset
    hw_root, hw_tip = 0.44 * d * 0.92, 0.26 * d * 0.92
    seq = [(r_root, -hw_root), (r_tip, -hw_tip), (r_tip, hw_tip), (r_root, hw_root)]
    return [polar(cx * SS, cy * SS, r * SS, th + t) for r, t in seq]


def arc_points(cx, cy, r, th0, th1, steps=12):
    return [polar(cx * SS, cy * SS, r * SS, th0 + (th1 - th0) * i / steps) for i in range(steps + 1)]


# ---- the frame ----------------------------------------------------------------------------------
def render_frame(A, f):
    T = R.frame_to_n(f)
    time = f / R.FPS
    n_now = int(math.floor(T))
    lap = min(int(T // R.CYCLE), R.N_LAPS - 1)
    tl = T % R.CYCLE
    t_in_lap = tl * R.SIXTEENTH
    pulse_on = R.presence("pulse", time) > 0.02
    slip = lap % 4

    canvas = A.bg.copy()
    rot_a = math.degrees((T - PHASE) * DA - math.pi / 2)
    rot_b = math.degrees(math.pi / 2 - (T + PHASE) * DB)
    ga = A.gear_a.rotate(rot_a, resample=Image.BICUBIC)
    gb = A.gear_b.rotate(rot_b, resample=Image.BICUBIC)
    canvas.paste(ga, (int(CX * SS - ga.width / 2), int(CY_A * SS - ga.height / 2)), ga)
    canvas.paste(gb, (int(CX * SS - gb.width / 2), int(CY_B * SS - gb.height / 2)), gb)
    dr = ImageDraw.Draw(canvas)

    bright = {k: R.pin_brightness(k, time) for k in R.KEYS}
    focus = {k: R.focus_weight(k, time) for k in R.KEYS}

    def pin_colour(key, base, slot_T, floor=0.22):
        """Colour of a pin for slot time slot_T (the sixteenth at which it passes the follower)."""
        b = floor + (1 - floor) * bright[key]
        col = mix((40, 40, 46), base, b)
        age = (T - slot_T) * R.SIXTEENTH
        if 0 <= age < 0.12 and bright[key] > 0.05:
            col = mix(col, (255, 255, 255), (1 - age / 0.12) * 0.9 * min(1.0, bright[key] + 0.3))
        return col

    # visible slot ranges (only draw what is near the screen: the whole wheel is on screen anyway)
    k0 = n_now - R.CYCLE // 2
    slots_a = range(k0, k0 + R.CYCLE)                    # 45 consecutive slots cover the wheel once
    j0 = n_now - 22
    slots_b = range(j0, j0 + 44)

    # --- riff wheel: home tooth, kick/snare pins, bass pins + held arcs
    for k in slots_a:
        th = angle_a(k, T)
        t = k % R.CYCLE
        if t == 0:
            col = pin_colour("rim", HOME_WHITE, k, floor=0.6)
            dr.polygon(tooth_poly(CX, CY_A, R_PITCH_A, th, DA), fill=col)
        if t in R.SNARES:
            dr.polygon(pin_poly(CX, CY_A, TRACK_R[1], th, rl=12.5, hw=7.5), fill=pin_colour("ks", R.SNARE_COLOUR, k))
        elif t in R.KICKS:
            dr.polygon(pin_poly(CX, CY_A, TRACK_R[1], th), fill=pin_colour("ks", R.BY_KEY["ks"].colour, k))
        if t not in R.RIM_RESTS:
            col = pin_colour("bass", R.BY_KEY["bass"].colour, k)
            dr.polygon(pin_poly(CX, CY_A, TRACK_R[2], th), fill=col)
            if (t + 1) % R.CYCLE in R.RIM_RESTS:        # the group's last note holds through the rest
                hold = mix((40, 40, 46), R.BY_KEY["bass"].colour, 0.22 + 0.78 * bright["bass"])
                dr.line(arc_points(CX, CY_A, TRACK_R[2], th, th - DA * 0.95), fill=hold, width=int(5 * SS))
    # --- beat wheel: pulse paint, ride pins, guitar pins
    for j in slots_b:
        th = angle_b(j, T)
        if j % R.BEAT == R.BEAT_PHASE:
            col = pin_colour("pulse", AMBER, j, floor=0.28)
            dr.polygon(tooth_poly(CX, CY_B, R_PITCH_B, th, DB), fill=col)
            dr.polygon(pin_poly(CX, CY_B, TRACK_R[1], th, rl=12.5, hw=7.5), fill=pin_colour("ride", R.BY_KEY["ride"].colour, j))
        elif j % R.BEAT == (R.BEAT_PHASE + 2) % R.BEAT:
            dr.polygon(pin_poly(CX, CY_B, TRACK_R[1], th, rl=7, hw=4.5), fill=pin_colour("ride", R.BY_KEY["ride"].colour, j))
        dr.polygon(pin_poly(CX, CY_B, TRACK_R[2], th, rl=8, hw=5), fill=pin_colour("gtr", R.BY_KEY["gtr"].colour, j))

    # --- hits: flashes at the followers
    def follower_pos(key):
        L = R.BY_KEY[key]
        if L.track == 0:
            return MESH
        r = TRACK_R[L.track]
        return (CX, CY_A + r) if L.wheel == "riff" else (CX, CY_B - r)

    for n in range(max(0, n_now - 5), n_now + 1):
        age = (T - n) * R.SIXTEENTH
        for key in R.KEYS:
            g = R.layer_gain(key, n * R.SIXTEENTH)
            if g < 0.03:
                continue
            kinds = R.hits_at(key, n)
            if not kinds:
                continue
            kind = kinds[0]
            fx, fy = follower_pos(key)
            boost = 0.6 + 0.4 * g + 0.5 * focus[key]
            if kind == "pulse":
                dur = 0.36
                if age < dur:
                    e = (1 - age / dur) ** 1.6
                    add_sprite(canvas, A.glow_big["pulse"], fx, fy + 6, 0.75 * e * boost)
                    rr = 40 + int(age / dur * 300) // 10 * 10
                    if rr in A.rings:
                        add_sprite(canvas, A.rings[rr], fx, fy, 0.8 * e * boost)
            elif kind in ("kick", "snare"):
                dur = 0.22 if kind == "snare" else 0.28
                if age < dur:
                    e = (1 - age / dur) ** 1.5
                    add_sprite(canvas, A.glow_big[kind], fx, fy, 0.8 * e * boost)
            else:
                dur = {"click": 0.14, "bell": 0.3, "ride": 0.16, "note": 0.22, "chug": 0.09}[kind]
                lvl = {"click": 0.9, "bell": 0.8, "ride": 0.35, "note": 0.7, "chug": 0.3}[kind]
                if age < dur:
                    e = (1 - age / dur) ** 1.5
                    add_sprite(canvas, A.glow[key], fx, fy - (6 if key == "rim" else 0), lvl * e * boost)
    # home: n = 180 k (k >= 1): the beat and the cycle start meet again
    for hn in (180, 360):
        age = (T - hn) * R.SIXTEENTH
        if 0 <= age < 1.1:
            e = (1 - age / 1.1) ** 1.3
            add_sprite(canvas, A.glow_home, MESH[0], MESH[1], 0.8 * e)
            rr = 40 + int(age / 1.1 * 640) // 10 * 10
            if rr in A.rings_home:
                add_sprite(canvas, A.rings_home[rr], MESH[0], MESH[1], 0.9 * e)

    # --- followers (engaged: on the track and coloured; lifted: a step back and grey)
    for key in R.KEYS:
        L = R.BY_KEY[key]
        p = R.presence(key, time)
        col = mix(DIMMER, L.colour, p)
        fx, fy = follower_pos(key)
        if L.track == 0:
            up = 1 if L.wheel == "riff" else -1          # riff follower points down to the mesh, beat up
            lift = (1 - p) * 10
            base_y = fy - up * (54 + lift)
            tip_y = fy - up * (36 + lift)
            dr.polygon([((fx - 13) * SS, base_y * SS), ((fx + 13) * SS, base_y * SS), (fx * SS, tip_y * SS)], fill=col)
        else:
            up = -1 if L.wheel == "riff" else 1          # points toward the wheel centre
            lift = (1 - p) * 10
            base_y = fy - up * (26 + lift)
            tip_y = fy - up * (12 + lift)
            dr.polygon([((fx - 10) * SS, base_y * SS), ((fx + 10) * SS, base_y * SS), (fx * SS, tip_y * SS)], fill=col)
            dr.line([(fx * SS, base_y * SS), (fx * SS, (base_y - up * 10) * SS)], fill=col, width=int(3 * SS))
        # focus chip: the part's name beside its follower during its focus lap
        if focus[key] > 0.01 and key != "rim":
            a = focus[key]
            label = L.name
            tw = A.f_chip.getlength(label) / SS
            x0, y0 = fx + 28, fy - 17
            dr.rounded_rectangle([x0 * SS, y0 * SS, (x0 + tw + 24) * SS, (y0 + 34) * SS], radius=8 * SS,
                                 fill=scale((14, 14, 18), 1.0))
            dr.text(((x0 + 12) * SS, (y0 + 17) * SS), label, font=A.f_chip, fill=scale(L.colour, a), anchor="lm")

    # --- hub labels
    dr.text((CX * SS, (CY_A - 12) * SS), "45", font=A.f_hub, fill=TEXT, anchor="mm")
    dr.text((CX * SS, (CY_A + 30) * SS), "THE RIFF", font=A.f_hub_s, fill=DIM, anchor="mm")
    pb = 0.35 + 0.65 * R.presence("pulse", time)
    dr.text((CX * SS, (CY_B - 12) * SS), "44", font=A.f_hub, fill=scale(AMBER, pb), anchor="mm")
    dr.text((CX * SS, (CY_B + 30) * SS), "THE BEAT", font=A.f_hub_s, fill=scale(DIM, pb), anchor="mm")

    # --- captions (top): crossfade at each lap start; the hook is up from frame 0
    fade_s = 0.35
    t_to_end = (R.CYCLE - tl) * R.SIXTEENTH
    alpha = min(ease(t_in_lap / fade_s), ease(t_to_end / fade_s)) if lap > 0 else ease(t_to_end / fade_s)
    title, sub = R.CAPTIONS[lap]
    title_col = TEXT
    fk = R.focused_layer(time)
    if fk and fk != "rim":
        title_col = mix(TEXT, R.BY_KEY[fk].colour, 0.85)
    if title.startswith("HOME"):
        title_col = mix(TEXT, (255, 225, 160), 0.8)
    kick_px = 0.0
    nb = (n_now // R.BEAT) * R.BEAT
    age_b = (T - nb) * R.SIXTEENTH
    if pulse_on and age_b < 0.18:
        kick_px = (1 - age_b / 0.18) * 4
    y = 112 - kick_px
    for line in wrap(title, A.f_title, 940):
        dr.text((W / 2 * SS, y * SS), line, font=A.f_title, fill=scale(title_col, alpha), anchor="mm")
        y += 70
    y += 8
    for line in wrap(sub, A.f_sub, 900):
        dr.text((W / 2 * SS, y * SS), line, font=A.f_sub, fill=scale(DIM, alpha), anchor="mm")
        y += 42

    # --- left column: group counter, lap + slip, the stack
    lx = 56
    # group counter: the current 5 or 3, filled as the clicks land; the rest is a hollow dot
    gi, gs, gsize = R.group_of(n_now % R.CYCLE)
    p_in = n_now % R.CYCLE - gs
    gy = 790
    dr.text((lx * SS, gy * SS), str(gsize), font=A.f_group, fill=TEXT, anchor="lm")
    for q in range(gsize):
        x0 = lx + 44 + q * 30
        slot = gs + q
        is_rest = slot in R.RIM_RESTS
        reached = q <= p_in
        if is_rest:
            dr.ellipse([(x0 - 9) * SS, (gy - 9) * SS, (x0 + 9) * SS, (gy + 9) * SS],
                       outline=(TEXT if reached else DIMMER), width=int(2 * SS))
        else:
            dr.ellipse([(x0 - 9) * SS, (gy - 9) * SS, (x0 + 9) * SS, (gy + 9) * SS],
                       fill=(R.BY_KEY["rim"].colour if reached else DIMMER))
    # lap + slip (from the pulse's entry)
    la = ease((time - R.entry_time("pulse")) / 0.5)
    if la > 0:
        ry = 846
        dr.text((lx * SS, ry * SS), f"LAP {lap + 1} / {R.N_LAPS}", font=A.f_lap, fill=scale(TEXT, la), anchor="lm")
        sy = ry + 40
        dr.text((lx * SS, sy * SS), "SLIP", font=A.f_mono_s, fill=scale(DIM, la), anchor="lm")
        for i in range(4):
            x0 = lx + 64 + i * 32
            box = [x0 * SS, (sy - 10) * SS, (x0 + 22) * SS, (sy + 10) * SS]
            if i < slip:
                dr.rectangle(box, fill=scale(AMBER, la))
            else:
                dr.rectangle(box, outline=scale(DIM, la), width=int(1.5 * SS))
        label = "HOME" if slip == 0 else (f"{slip} TOOTH" if slip == 1 else f"{slip} TEETH")
        dr.text(((lx + 64 + 4 * 32 + 6) * SS, sy * SS), label, font=A.f_mono_s,
                fill=scale(AMBER if slip else HOME_WHITE, la), anchor="lm")
    # the stack: six parts, lit when present, a dot that flashes on each hit
    sy0 = 930
    for i, key in enumerate(R.KEYS):
        L = R.BY_KEY[key]
        p = R.presence(key, time)
        yy = sy0 + i * 31
        col = mix(DIMMER, L.colour, p)
        # hit dot
        lit = 0.0
        for n in range(max(0, n_now - 2), n_now + 1):
            if R.hits_at(key, n) and R.layer_gain(key, n * R.SIXTEENTH) > 0.03:
                age = (T - n) * R.SIXTEENTH
                lit = max(lit, max(0.0, 1 - age / 0.16))
        dcol = mix(col, (255, 255, 255), lit * 0.8) if p > 0.02 else DIMMER
        rad = 6 + 3 * lit
        dr.ellipse([(lx + 8 - rad) * SS, (yy - rad) * SS, (lx + 8 + rad) * SS, (yy + rad) * SS], fill=dcol)
        fw = A.f_mono if focus[key] > 0.3 else A.f_mono_s
        tcol = mix(col, (255, 255, 255), 0.35 * focus[key]) if p > 0.02 else DIMMER
        dr.text(((lx + 28) * SS, yy * SS), L.name, font=fw, fill=tcol, anchor="lm")
        if focus[key] > 0.3:
            dr.polygon([((lx - 10) * SS, (yy - 6) * SS), ((lx - 10) * SS, (yy + 6) * SS), ((lx - 2) * SS, yy * SS)],
                       fill=scale(L.colour, focus[key]))

    # --- slip bracket at each cycle start while the pulse plays: the amber tooth is already ahead
    if pulse_on and lap >= 1 and t_in_lap < 1.9 and slip > 0:
        a = min(ease(t_in_lap / 0.25), ease((1.9 - t_in_lap) / 0.4))
        col = scale((255, 255, 255), a)
        r_arc = R_PITCH_B + ADDENDUM + 20
        th0 = -math.pi / 2
        th1 = th0 + slip * DB
        dr.line(arc_points(CX, CY_B, r_arc, th0, th1, 30), fill=col, width=int(4 * SS), joint="curve")
        for th in (th0, th1):
            p0 = polar(CX * SS, CY_B * SS, (r_arc - 10) * SS, th)
            p1 = polar(CX * SS, CY_B * SS, (r_arc + 10) * SS, th)
            dr.line([p0, p1], fill=col, width=int(4 * SS))
        word = "beat already 1 tooth ahead" if slip == 1 else f"beat already {slip} teeth ahead"
        dr.text((lx * SS, 1125 * SS), word, font=A.f_mono_s, fill=col, anchor="lm")
    for hn in (180, 360):
        age = (T - hn) * R.SIXTEENTH
        if 0 <= age < 1.9:
            a = min(ease(age / 0.2), ease((1.9 - age) / 0.5))
            dr.text((lx * SS, 1125 * SS), "home: 4 laps, 45 beats", font=A.f_mono_s,
                    fill=scale(HOME_WHITE, a), anchor="lm")

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
            print("still", f, "n=%.2f" % R.frame_to_n(f), "t=%.1fs" % (f / R.FPS))
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

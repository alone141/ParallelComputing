"""Frame renderer: two meshing gears play the intro of "The Art of Dying".

Top wheel  = the riff: 45 tooth slots, 11 teeth missing (the rests). A tooth striking the
             mesh point is a rim click; a gap is silence.
Bottom wheel = the beat: 44 teeth, every fourth one painted amber. A painted tooth at the
             mesh point is the big tribal hit.
Both advance one tooth per sixteenth. 45 = 44 + 1, so every lap of the riff wheel the beat
wheel is one tooth ahead; after four laps the slip is a whole beat and the wheels come home.

Usage: python3 render.py [--frames a:b] [--still N ...] [--out DIR] [--workers 4]
"""
import argparse
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

import rhythm as R

W, H = 1080, 1920
SS = 2                      # supersampling
SW, SH = W * SS, H * SS

# ---- geometry (1x units) -------------------------------------------------------------
PITCH = 46.0                              # circular pitch in px: one tooth = 46 px of rim
MODULE = PITCH / math.pi
R_PITCH_A = R.CYCLE * MODULE / 2          # 329.4
R_PITCH_B = 44 * MODULE / 2               # 322.1
ADDENDUM, DEDENDUM = 19.0, 21.0
CX = W / 2
CY_A = 700.0
CY_B = CY_A + R_PITCH_A + R_PITCH_B       # 1351.5
MESH = (CX, CY_A + R_PITCH_A)             # where teeth meet
DA = 2 * math.pi / R.CYCLE
DB = 2 * math.pi / 44
PHASE = 0.25                              # teeth interleave: A centred at t=k+.25, B at t=j-.25

# ---- palette ---------------------------------------------------------------------------
BG = (9, 9, 12)
STEEL = (58, 61, 70)
STEEL_EDGE = (120, 126, 140)
BRONZE = (74, 60, 46)
BRONZE_EDGE = (150, 122, 92)
AMBER = (255, 170, 60)
HOME_WHITE = (235, 245, 255)
RIM_FLASH = (255, 240, 200)
TEXT = (236, 236, 240)
DIM = (150, 150, 160)

FONT_DIR_I = "/usr/share/fonts/opentype/inter/"
FONT_DIR_M = "/usr/share/fonts/truetype/jetbrains-mono/"


def font(name, size):
    path = {
        "display": FONT_DIR_I + "InterDisplay-Bold.otf",
        "display_sb": FONT_DIR_I + "InterDisplay-SemiBold.otf",
        "text": FONT_DIR_I + "Inter-Medium.otf",
        "mono": FONT_DIR_M + "JetBrainsMono-Medium.ttf",
        "mono_b": FONT_DIR_M + "JetBrainsMono-Bold.ttf",
    }[name]
    return ImageFont.truetype(path, int(size * SS))


def polar(cx, cy, r, th):
    return (cx + r * math.cos(th), cy + r * math.sin(th))


# ---- gear sprites (pre-rendered once, rotated per frame) ------------------------------
def gear_sprite(n, present, r_pitch, body, edge, marks, ccw_index=False, spokes=5,
                engrave=None):
    """RGBA sprite of a gear, tooth k centred at angle k*2pi/n (or -k*2pi/n if ccw_index).

    present: set of tooth indices that exist. marks: {index: colour} for painted teeth.
    """
    r_tip = r_pitch + ADDENDUM
    r_root = r_pitch - DEDENDUM
    size = int((r_tip + 12) * 2 * SS)
    c = size / 2
    d = 2 * math.pi / n
    sgn = -1 if ccw_index else 1
    hw_root = 0.44 * d
    hw_tip = 0.26 * d

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)

    # body outline with teeth
    pts = []
    for k in range(n):
        a = sgn * k * d
        if k in present:
            seq = [(r_root, a - 0.5 * d), (r_root, a - hw_root), (r_tip, a - hw_tip),
                   (r_tip, a + hw_tip), (r_root, a + hw_root), (r_root, a + 0.5 * d)]
        else:
            seq = [(r_root, a - 0.5 * d), (r_root - 3, a - 0.3 * d), (r_root - 3, a + 0.3 * d),
                   (r_root, a + 0.5 * d)]
        if ccw_index:
            seq = [(r, a - (t - a)) for (r, t) in seq][::-1]
            seq = [(r, t) for (r, t) in seq]
        for r, th in seq:
            pts.append(polar(c, c, r * SS, th))
    dr.polygon(pts, fill=body + (255,))

    # metal shading: radial gradient + directional highlight, applied through numpy
    arr = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.hypot(xx - c, yy - c) / (r_tip * SS)
    ang = np.arctan2(yy - c, xx - c)
    radial = 0.72 + 0.55 * np.clip(rr, 0, 1) ** 2.2          # brighter toward the rim
    sheen = 1.0 + 0.22 * np.cos(ang + 2.3) * np.clip(rr, 0, 1)  # soft light from upper-left
    shade = (radial * sheen)[..., None]
    arr[..., :3] = np.clip(arr[..., :3] * shade, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    dr = ImageDraw.Draw(img)

    # rim bands and edge
    dr.ellipse([c - r_root * SS, c - r_root * SS, c + r_root * SS, c + r_root * SS],
               outline=edge + (110,), width=int(1.5 * SS))
    r_band = r_root - 46
    dr.ellipse([c - r_band * SS, c - r_band * SS, c + r_band * SS, c + r_band * SS],
               outline=edge + (90,), width=int(2 * SS))

    # painted teeth
    for k, col in marks.items():
        a = sgn * k * d
        seq = [(r_root + 2, a - hw_root * 0.95), (r_tip, a - hw_tip), (r_tip, a + hw_tip),
               (r_root + 2, a + hw_root * 0.95)]
        poly = [polar(c, c, r * SS, th) for r, th in seq]
        dr.polygon(poly, fill=col + (255,))

    # windows between spokes (cut out of alpha)
    mask = Image.new("L", (size, size), 255)
    md = ImageDraw.Draw(mask)
    r_in, r_out, sw = 92.0, r_band - 16, 26.0
    for i in range(spokes):
        th1 = i * 2 * math.pi / spokes + 0.3
        th2 = th1 + 2 * math.pi / spokes
        gi, go = math.asin(sw / r_in), math.asin(sw / r_out)
        poly = []
        for j in range(25):
            th = th1 + gi + (th2 - gi - th1 - gi) * j / 24
            poly.append(polar(c, c, r_in * SS, th))
        for j in range(40):
            th = th2 - go - (th2 - go - th1 - go) * j / 39
            poly.append(polar(c, c, r_out * SS, th))
        md.polygon(poly, fill=0)
    alpha = ImageChops.multiply(img.getchannel("A"), mask)
    img.putalpha(alpha)
    dr = ImageDraw.Draw(img)
    # window edges
    for i in range(spokes):
        pass
    # hub
    dr.ellipse([c - 88 * SS, c - 88 * SS, c + 88 * SS, c + 88 * SS], outline=edge + (120,),
               width=int(2 * SS))
    for i in range(6):
        p = polar(c, c, 62 * SS, i * math.pi / 3 + 0.2)
        dr.ellipse([p[0] - 5 * SS, p[1] - 5 * SS, p[0] + 5 * SS, p[1] + 5 * SS],
                   fill=(20, 20, 24, 255))

    # engraved group sizes on the rim band
    if engrave:
        f = font("mono_b", 30)
        for centre_slot, label in engrave:
            a = sgn * centre_slot * d
            tw = Image.new("RGBA", (int(60 * SS), int(60 * SS)), (0, 0, 0, 0))
            td = ImageDraw.Draw(tw)
            td.text((30 * SS, 30 * SS), str(label), font=f, fill=edge + (230,), anchor="mm")
            # text baseline tangent to the rim: rotate so "up" points outward
            tw = tw.rotate(-math.degrees(a) - 90, resample=Image.BICUBIC)
            p = polar(c, c, (r_root - 24) * SS, a)
            img.alpha_composite(tw, (int(p[0] - 30 * SS), int(p[1] - 30 * SS)))
    return img


def glow_sprite(radius, colour, power=1.6):
    """Additive radial glow sprite (RGB)."""
    s = int(radius * 2 * SS)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    rr = np.hypot(xx - s / 2, yy - s / 2) / (radius * SS)
    a = np.clip(1 - rr, 0, 1) ** power
    arr = np.stack([a * colour[i] for i in range(3)], axis=-1)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def ring_sprite(radius, width, colour):
    s = int((radius + width) * 2 * SS) + 4
    img = Image.new("RGB", (s, s), (0, 0, 0))
    d = ImageDraw.Draw(img)
    c = s / 2
    d.ellipse([c - radius * SS, c - radius * SS, c + radius * SS, c + radius * SS],
              outline=colour, width=int(width * SS))
    return img.filter(ImageFilter.GaussianBlur(1.2 * SS))


def add_sprite(canvas, sprite, x, y, gain=1.0):
    """Additive blend of an RGB sprite centred at (x, y) in 1x coordinates."""
    if gain <= 0.01:
        return
    if gain < 0.999:
        sprite = sprite.point(lambda v: int(v * gain))
    w, h = sprite.size
    x0, y0 = int(x * SS - w / 2), int(y * SS - h / 2)
    x1, y1 = x0 + w, y0 + h
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(SW, x1), min(SH, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    region = canvas.crop((cx0, cy0, cx1, cy1))
    sp = sprite.crop((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0))
    canvas.paste(ImageChops.add(region, sp), (cx0, cy0))


# ---- static assets -------------------------------------------------------------------
class Assets:
    def __init__(self):
        present_a = set(R.RIM_HITS)
        self.gear_a = gear_sprite(R.CYCLE, present_a, R_PITCH_A, STEEL, STEEL_EDGE,
                                  {0: HOME_WHITE}, ccw_index=False, spokes=5,
                                  engrave=[(cs, g) for cs, g in R.group_centres()])
        marks_b = {j: AMBER for j in range(0, 44, 4)}
        self.gear_b = gear_sprite(44, set(range(44)), R_PITCH_B, BRONZE, BRONZE_EDGE, marks_b,
                                  ccw_index=True, spokes=4)
        self.glow_rim = glow_sprite(70, RIM_FLASH, 2.2)
        self.glow_beat = glow_sprite(150, AMBER, 2.0)
        self.glow_home = glow_sprite(420, (255, 225, 160), 1.6)
        self.glow_tooth = glow_sprite(34, (255, 255, 255), 1.4)
        self.rings = {r: ring_sprite(r, 5, AMBER) for r in range(40, 420, 10)}
        self.rings_home = {r: ring_sprite(r, 7, (255, 236, 190)) for r in range(40, 700, 10)}
        self.bg = self.make_bg()
        self.f_title = font("display", 66)
        self.f_sub = font("text", 34)
        self.f_lap = font("display", 42)
        self.f_mono = font("mono", 30)
        self.f_mono_s = font("mono", 24)
        self.f_hub = font("mono_b", 64)
        self.f_hub_s = font("display_sb", 26)

    def make_bg(self):
        yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
        v = np.hypot((xx - SW / 2) / SW, (yy - SH * 0.53) / SH)
        shade = np.clip(1.0 - v * 1.1, 0.0, 1.0)
        base = np.array(BG, np.float32)
        arr = base[None, None, :] * (0.55 + 0.9 * shade[..., None])
        # faint spotlight on the mesh
        m = np.exp(-(((xx - MESH[0] * SS) / (420 * SS)) ** 2 + ((yy - MESH[1] * SS) / (300 * SS)) ** 2))
        arr += (m * 14)[..., None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


# ---- captions --------------------------------------------------------------------------
CAPTIONS = [
    ("WHY DOES THIS BEAT DRIFT?",
     "Gojira, The Art of Dying. Two wheels, one tooth apart."),
    ("45 TEETH AGAINST 44",
     "The riff wheel ends each lap one tooth late. The beat slips."),
    ("ONE TOOTH EVERY LAP",
     "45 = 11 × 4 + 1. The spare tooth piles up."),
    ("FOUR TEETH IS ONE BEAT",
     "After four laps the slip is a whole beat, and the wheels come home."),
]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def draw_text_centered(dr, xy, text, f, fill, max_w=None):
    dr.text((xy[0] * SS, xy[1] * SS), text, font=f, fill=fill, anchor="mm")


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


# ---- the frame -------------------------------------------------------------------------
def render_frame(A, f):
    T = R.frame_to_sixt(f)                 # 0 .. 180
    canvas = A.bg.copy()

    # beat/home flash brightens the whole scene a touch (through the background copy)
    lap = min(int(T // R.CYCLE), R.LAPS - 1)   # 0..3
    tl = T % R.CYCLE                       # position in the cycle
    slip = lap                             # teeth the beat is ahead of the cycle start

    # gears
    rot_a = math.degrees((T - PHASE) * DA - math.pi / 2)
    rot_b = math.degrees(math.pi / 2 - (T + PHASE) * DB)
    ga = A.gear_a.rotate(rot_a, resample=Image.BICUBIC)
    gb = A.gear_b.rotate(rot_b, resample=Image.BICUBIC)
    canvas.paste(ga, (int(CX * SS - ga.width / 2), int(CY_A * SS - ga.height / 2)), ga)
    canvas.paste(gb, (int(CX * SS - gb.width / 2), int(CY_B * SS - gb.height / 2)), gb)

    dr = ImageDraw.Draw(canvas)

    # ---- hits at the mesh point ----
    # rim clicks: tooth k of A is at the mesh at T = k (+PHASE visually)
    k = int(math.floor(T))
    for kk in (k - 1, k):               # current and previous sixteenth
        if kk < 0:
            continue
        age = (T - kk) * R.SIXT_S_FILM  # seconds since the hit
        if R.is_rim(kk) and age < 0.16:
            g = (1 - age / 0.16) ** 1.5
            add_sprite(canvas, A.glow_rim, MESH[0], MESH[1] - 6, 0.9 * g)
            add_sprite(canvas, A.glow_tooth, MESH[0], MESH[1] - 14, g)
    # the big hit: every 4 sixteenths
    kb = int(math.floor(T / R.BEAT)) * R.BEAT
    for bb in (kb - R.BEAT, kb):
        if bb < 0:
            continue
        age = (T - bb) * R.SIXT_S_FILM
        if age < 0.36:
            g = (1 - age / 0.36) ** 1.6
            add_sprite(canvas, A.glow_beat, MESH[0], MESH[1] + 8, 0.75 * g)
            rr = 40 + int(age / 0.36 * 330) // 10 * 10
            if rr in A.rings:
                add_sprite(canvas, A.rings[rr], MESH[0], MESH[1], 0.9 * (1 - age / 0.36) ** 1.2)
    # home: T = 0 (= 180 on the loop)
    age_home = T * R.SIXT_S_FILM
    if age_home < 1.1:
        g = (1 - age_home / 1.1) ** 1.3
        add_sprite(canvas, A.glow_home, MESH[0], MESH[1], 0.8 * g)
        rr = 40 + int(age_home / 1.1 * 640) // 10 * 10
        if rr in A.rings_home:
            add_sprite(canvas, A.rings_home[rr], MESH[0], MESH[1], 0.9 * g)

    # ---- the striker: a fixed pointer marking the mesh point ----
    px, py = MESH
    dr.polygon([((px - 14) * SS, (py - 70) * SS), ((px + 14) * SS, (py - 70) * SS),
                (px * SS, (py - 52) * SS)], fill=(255, 255, 255))
    dr.polygon([((px - 14) * SS, (py + 70) * SS), ((px + 14) * SS, (py + 70) * SS),
                (px * SS, (py + 52) * SS)], fill=(255, 255, 255))

    # ---- hub labels (static, over the rotating hubs) ----
    dr.text((CX * SS, (CY_A - 14) * SS), "45", font=A.f_hub, fill=TEXT, anchor="mm")
    dr.text((CX * SS, (CY_A + 34) * SS), "THE RIFF", font=A.f_hub_s, fill=DIM, anchor="mm")
    dr.text((CX * SS, (CY_B - 14) * SS), "44", font=A.f_hub, fill=AMBER, anchor="mm")
    dr.text((CX * SS, (CY_B + 34) * SS), "THE BEAT", font=A.f_hub_s, fill=DIM, anchor="mm")

    # ---- captions (top) ----
    # crossfade at every cycle start (T = 0, 45, 90, 135), 0.35 s out / in
    fade_s = 0.35
    t_in_lap = tl * R.SIXT_S_FILM
    t_to_end = (R.CYCLE - tl) * R.SIXT_S_FILM
    alpha = min(ease(t_in_lap / fade_s), ease(t_to_end / fade_s))
    if lap == 0:
        alpha = ease(t_to_end / fade_s)
    title, sub = CAPTIONS[lap]
    col_t = tuple(int(v * alpha) for v in TEXT)
    col_s = tuple(int(v * alpha) for v in DIM)
    # a tiny push on the title at every beat
    kick = 0.0
    age_b = (T - kb) * R.SIXT_S_FILM
    if age_b < 0.18:
        kick = (1 - age_b / 0.18) * 4
    y = 120 - kick
    for line in wrap(title, A.f_title, 940):
        dr.text((CX * SS, y * SS), line, font=A.f_title, fill=col_t, anchor="mm")
        y += 74
    y += 6
    for line in wrap(sub, A.f_sub, 900):
        dr.text((CX * SS, y * SS), line, font=A.f_sub, fill=col_s, anchor="mm")
        y += 44

    # ---- readouts (left of the mesh; right column stays free) ----
    rx = 64
    ry = MESH[1] - 118
    dr.text((rx * SS, ry * SS), f"LAP {lap + 1} / 4", font=A.f_lap, fill=TEXT, anchor="lm")
    beats_done = int(math.floor(T / R.BEAT)) % 45
    dr.text((rx * SS, (ry + 46) * SS), f"BEAT {beats_done + 1:02d}/45", font=A.f_mono,
            fill=DIM, anchor="lm")
    # slip meter: four slots, one per tooth of slip
    sy = ry + 100
    dr.text((rx * SS, sy * SS), "SLIP", font=A.f_mono_s, fill=DIM, anchor="lm")
    for i in range(4):
        x0 = rx + 86 + i * 40
        box = [(x0) * SS, (sy - 12) * SS, (x0 + 26) * SS, (sy + 12) * SS]
        filled = i < slip
        if filled:
            dr.rectangle(box, fill=AMBER)
        else:
            dr.rectangle(box, outline=DIM, width=int(1.5 * SS))
    label = "HOME" if slip == 0 else (f"{slip} TOOTH" if slip == 1 else f"{slip} TEETH")
    dr.text(((rx + 86 + 4 * 40 + 6) * SS, sy * SS), label, font=A.f_mono_s,
            fill=(AMBER if slip else HOME_WHITE), anchor="lm")

    # ---- slip bracket: at each cycle start show how far ahead the painted tooth already is ----
    if lap >= 1 and t_in_lap < 1.9:
        a = min(ease(t_in_lap / 0.25), ease((1.9 - t_in_lap) / 0.4))
        col = tuple(int(v * a) for v in (255, 255, 255))
        # arc along the top of B from the mesh point to the painted tooth, `slip` teeth clockwise
        r_arc = R_PITCH_B + ADDENDUM + 22
        th0 = -math.pi / 2
        th1 = th0 + slip * DB
        pts = [polar(CX * SS, CY_B * SS, r_arc * SS, th0 + (th1 - th0) * i / 30) for i in range(31)]
        dr.line(pts, fill=col, width=int(4 * SS), joint="curve")
        for th in (th0, th1):
            p0 = polar(CX * SS, CY_B * SS, (r_arc - 10) * SS, th)
            p1 = polar(CX * SS, CY_B * SS, (r_arc + 10) * SS, th)
            dr.line([p0, p1], fill=col, width=int(4 * SS))
        word = "the beat is already 1 tooth ahead" if slip == 1 else f"the beat is already {slip} teeth ahead"
        dr.text((rx * SS, (sy + 44) * SS), word, font=A.f_mono_s, fill=col, anchor="lm")
    if lap == 0 and age_home < 1.6:
        a = ease((1.6 - age_home) / 0.5)
        col = tuple(int(v * a) for v in HOME_WHITE)
        dr.text((rx * SS, (sy + 44) * SS), "4 laps · 45 beats · 19.3 s · home", font=A.f_mono_s,
                fill=col, anchor="lm")

    return canvas.resize((W, H), Image.LANCZOS)


_ASSETS = None


def _work(args):
    global _ASSETS
    f, out = args
    if _ASSETS is None:
        _ASSETS = Assets()
    img = render_frame(_ASSETS, f)
    img.save(out, compress_level=1)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=f"0:{R.FRAMES}")
    ap.add_argument("--still", type=int, nargs="*")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.still:
        A = Assets()
        for f in a.still:
            render_frame(A, f).save(os.path.join(a.out, f"still_{f:05d}.png"))
            print("still", f, "T=%.2f" % R.frame_to_sixt(f))
        return
    lo, hi = (int(x) for x in a.frames.split(":"))
    jobs = [(f, os.path.join(a.out, f"f{f:05d}.png")) for f in range(lo, hi)]
    with Pool(a.workers) as p:
        for i, f in enumerate(p.imap_unordered(_work, jobs, chunksize=8)):
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print("done", len(jobs))


if __name__ == "__main__":
    main()

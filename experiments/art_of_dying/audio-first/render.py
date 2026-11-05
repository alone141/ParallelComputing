#!/usr/bin/env python3
"""Render the film from build/analysis.npz (+ build/mix.wav) with Pillow and ffmpeg.

Every number on screen comes from analysis.npz, i.e. from the audio. The only things
written by hand are the captions (timed to the sections of the synthesised mix) and the
arithmetic line "45 = 11 x 4 + 1", which is shown only while the detector itself reports
loop = 45 and hit = 4.
"""
import numpy as np, os, sys, subprocess, shutil
from multiprocessing import Pool
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 30
BG = (10, 10, 16)
AMBER = (255, 184, 77)      # click stream
CORAL = (255, 86, 118)      # thump stream
WHITE = (242, 242, 247)
DIM = (128, 128, 146)
GRID = (36, 36, 48)
FONT_DIR = "/usr/share/fonts"
def font(name, size):
    paths = {"disp": "opentype/inter/InterDisplay-Bold.otf", "inter": "opentype/inter/Inter-Regular.otf",
             "inter_m": "opentype/inter/Inter-Medium.otf", "mono": "truetype/jetbrains-mono/JetBrainsMono-Bold.ttf",
             "mono_r": "truetype/jetbrains-mono/JetBrainsMono-Regular.ttf"}
    return ImageFont.truetype(os.path.join(FONT_DIR, paths[name]), size)

F_H1 = font("disp", 70); F_SUB = font("inter", 38); F_LAB = font("mono_r", 28); F_BIG = font("mono", 96)
F_MID = font("mono", 44); F_SMALL = font("mono_r", 24); F_TOP = font("inter_m", 30)

D = None
def load():
    global D
    D = dict(np.load("build/analysis.npz"))

def wrap(draw, text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= maxw:
            cur = trial
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines

def stable(arr, k, n=12):
    seg = arr[max(0, k - n + 1):k + 1]
    return Counter(seg.tolist()).most_common(1)[0][0]

# ---------------------------------------------------------------- captions
SEC_A_END, SEC_B_END = 9.643, 28.929          # from listening: where the texture changes
def captions(t, tatum_ms, loop, hit, loop_str):
    if t < 2.4:
        return ("Can a computer HEAR this riff's hidden number?",
                "Gojira · The Art of Dying · intro. No score, just the sound.")
    if t < SEC_A_END:
        if tatum_ms > 0:
            return ("Step 1: find the pulse.",
                    f"A click every {tatum_ms:.0f} ms. Step 2: how long until the pattern repeats?")
        return ("Step 1: find the pulse.", "Listening…")
    if t < 14.0:
        if loop == 45:
            return ("It had to hear the loop twice.", f"Loop found: {loop} clicks long. (Its first guess, 24, was the first half.)")
        return ("It had to hear the loop twice.", f"Best guess so far: {loop} clicks.")
    if t < SEC_B_END:
        if loop == 45 and hit == 4:
            return (f"The big hit repeats every {hit}. The clicks every {loop}.",
                    "45 = 11 × 4 + 1, so the hit lands one click earlier every lap.")
        return (f"The big hit repeats every {hit or '?'}. The clicks every {loop or '?'}.", "Watch the red dots drift in the fold below.")
    return ("Full band in. The machine loses the thread.",
            f"Kick, bass and guitars blur the clicks; it now hears {loop or '?'}. Postmortem in the README.")

# ---------------------------------------------------------------- panels
def panel_label(draw, x, y, text, color=DIM):
    draw.text((x, y), text, font=F_LAB, fill=color)

def draw_hears(draw, k, t):
    x0, x1, y0, y1 = 40, 1000, 470, 690
    panel_label(draw, x0, 436, "WHAT IT HEARS · last 3.5 s")
    hop = float(D["hop_s"]); span = 3.5
    n_now = int(t / hop); n_start = n_now - int(span / hop)
    flux = D["total_flux"]
    idx = np.arange(max(0, n_start), n_now)
    draw.line([(x0, y1), (x1, y1)], fill=GRID, width=2)
    if len(idx) > 2:
        xs = x0 + (idx - n_start) / (n_now - n_start) * (x1 - x0)
        ys = y1 - np.clip(flux[idx], 0, 1.2) * (y1 - y0) * 0.75
        pts = [(x0 + (idx[0] - n_start) / (n_now - n_start) * (x1 - x0), y1)] + list(zip(xs, ys)) + [(xs[-1], y1)]
        draw.polygon(pts, fill=(30, 30, 42))
    on = D["onsets"]; th = D["thumpness"]; st = D["strength"]
    sel = (on >= n_start) & (on < n_now)
    for i, tn, s in zip(on[sel], th[sel], st[sel]):
        x = x0 + (i - n_start) / (n_now - n_start) * (x1 - x0)
        if tn >= 0.4:
            r = 10 + 10 * min(1, s)
            draw.ellipse([x - r, y1 - 70 - r, x + r, y1 - 70 + r], fill=CORAL)
        else:
            hgt = 60 + 90 * min(1, s)
            draw.line([(x, y1), (x, y1 - hgt)], fill=AMBER, width=5)
    draw.line([(x1, y0 - 10), (x1, y1 + 6)], fill=WHITE, width=3)
    draw.text((640, 438), "▮ click", font=F_SMALL, fill=AMBER)
    draw.text((780, 438), "● thump", font=F_SMALL, fill=CORAL)

def draw_acf(draw, k, t, tatum_ms, loop, hit):
    x0, x1, y0, y1 = 40, 1000, 790, 1090
    hop = float(D["hop_s"])
    if t < 2.6 or tatum_ms <= 0:
        panel_label(draw, x0, 740, "HOW IT REPEATS · lag in ms")
        r = D["acf_hop"][k]; n = 260                 # 0..1.5 s
        draw.line([(x0, y1), (x1, y1)], fill=GRID, width=2)
        xs = x0 + np.arange(n) / n * (x1 - x0); ys = y1 - np.clip(r[:n], 0, 1) * (y1 - y0)
        if np.abs(r[:n]).sum() > 0:
            draw.line(list(zip(xs, ys)), fill=WHITE, width=3)
        for ms in (0, 250, 500, 750, 1000, 1250, 1500):
            x = x0 + ms / 1000 / hop / n * (x1 - x0)
            draw.text((x - 10, y1 + 8), f"{ms}", font=F_SMALL, fill=DIM)
        if tatum_ms > 0:
            x = x0 + (tatum_ms / 1000 / hop) / n * (x1 - x0)
            draw.line([(x, y0), (x, y1)], fill=AMBER, width=2)
            draw.text((x + 12, y0), f"pulse {tatum_ms:.0f} ms", font=F_MID, fill=AMBER)
        return
    panel_label(draw, x0, 740, "HOW IT REPEATS · lag in pulses")
    KSHOW = 100
    draw.line([(x0, y1), (x1, y1)], fill=GRID, width=2)
    for lag in range(0, KSHOW + 1, 10):
        x = x0 + lag / KSHOW * (x1 - x0)
        draw.line([(x, y1), (x, y1 + 6)], fill=DIM, width=2)
        draw.text((x - 12, y1 + 8), f"{lag}", font=F_SMALL, fill=DIM)
    for j, col in ((0, CORAL), (1, AMBER)):
        r = D["acf_tat"][k, j]
        ok = np.isfinite(r[:KSHOW + 1])
        if ok.sum() < 3 or not D["active"][k, j]:
            continue
        lags = np.arange(KSHOW + 1)[ok]
        xs = x0 + lags / KSHOW * (x1 - x0); ys = y1 - np.clip(r[:KSHOW + 1][ok], 0, 1) * (y1 - y0) * 0.9
        draw.line(list(zip(xs, ys)), fill=col, width=4)
        P = int(D["fund"][k, j])
        if 0 < P <= KSHOW and ok[P]:
            x = x0 + P / KSHOW * (x1 - x0); y = y1 - np.clip(r[P], 0, 1) * (y1 - y0) * 0.9
            draw.ellipse([x - 11, y - 11, x + 11, y + 11], fill=col, outline=BG, width=3)
            lab = f"{P}"
            tw = draw.textlength(lab, font=F_MID)
            draw.text((min(x - tw / 2, x1 - tw), y - 64), lab, font=F_MID, fill=col)
    # big readouts, top right of the panel (above the apps' button column)
    rx = 700
    if loop:
        draw.text((rx, 736), "LOOP", font=F_SMALL, fill=AMBER)
        draw.text((rx + 90, 718), f"{loop}", font=F_MID, fill=AMBER)
    if hit:
        draw.text((rx + 190, 736), "HIT", font=F_SMALL, fill=CORAL)
        draw.text((rx + 250, 718), f"{hit}", font=F_MID, fill=CORAL)

def draw_fold(draw, k, t, loop):
    x0, x1, y0, y1 = 40, 1000, 1200, 1520
    P = int(loop) if loop and loop >= 2 else 0
    g = D["grid"][k]
    valid = np.isfinite(g[0]) & np.isfinite(g[1])
    M = int(valid.sum())
    if P == 0 or M < P:
        panel_label(draw, x0, 1160, "FOLDED · waiting for a loop length")
        return
    panel_label(draw, x0, 1160, f"FOLDED EVERY {P} · one row per lap")
    gc, gt = g[1][valid], g[0][valid]
    rows = M // P
    show = min(rows, 6 if P >= 20 else 8)
    cw = (x1 - x0) / P
    rh = (y1 - y0) / show
    first = rows - show
    # align columns so that the latest row ends at the most recent complete lap
    off = M - rows * P
    for r in range(first, rows):
        yy = y0 + (r - first) * rh
        for c in range(P):
            m = off + r * P + c
            x = x0 + c * cw
            sc, stp = gc[m], gt[m]
            if stp > 0:
                a = int(120 + 135 * min(1, stp))
                rad = min(cw * 0.42, rh * 0.36)
                draw.ellipse([x + cw / 2 - rad, yy + rh / 2 - rad, x + cw / 2 + rad, yy + rh / 2 + rad],
                             fill=(255, a // 3 + 10, a // 2))
            elif sc > 0:
                a = int(110 + 145 * min(1, sc))
                pad = max(1, cw * 0.18)
                draw.rectangle([x + pad, yy + rh * 0.22, x + cw - pad, yy + rh * 0.78], fill=(a, int(a * 0.72), int(a * 0.3)))
        draw.text((x1 + 8, yy + rh / 2 - 12), f"{r - first + 1}", font=F_SMALL, fill=DIM)
    draw.line([(x0, y1 + 4), (x1, y1 + 4)], fill=GRID, width=2)

def render_frame(k):
    t = k / FPS
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    tatum_ms = float(D["tatum"][k]) * 1000
    loop = int(stable(D["cycle"], k)); hit = int(stable(D["beat"], k))
    loop_str = float(D["fstr"][k].max())
    d.text((60, 96), "GOJIRA · THE ART OF DYING · intro, synthesised · 140 bpm", font=F_TOP, fill=DIM)
    h1, sub = captions(t, tatum_ms, loop, hit, loop_str)
    y = 150
    for line in wrap(d, h1, F_H1, 960):
        d.text((60, y), line, font=F_H1, fill=WHITE); y += 80
    y += 6
    for line in wrap(d, sub, F_SUB, 940):
        d.text((60, y), line, font=F_SUB, fill=(190, 190, 205)); y += 46
    draw_hears(d, k, t)
    draw_acf(d, k, t, tatum_ms, loop, hit)
    draw_fold(d, k, t, loop)
    # progress line (non-essential, in the reserved bottom zone)
    dur = float(D["dur"])
    d.line([(60, 1570), (60 + 960 * min(1, t / dur), 1570)], fill=(70, 70, 86), width=4)
    return img

def worker(args):
    ks, out_dir = args
    load()
    for k in ks:
        render_frame(k).save(os.path.join(out_dir, f"{k:05d}.png"), compress_level=1)
    return len(ks)

def main(out_dir, mp4, stills_dir, still_times):
    load()
    n = len(D["cycle"])
    os.makedirs(out_dir, exist_ok=True)
    chunks = [(list(range(i, n, 4)), out_dir) for i in range(4)]
    with Pool(4) as p:
        p.map(worker, chunks)
    os.makedirs(stills_dir, exist_ok=True)
    for i, st in enumerate(still_times):
        shutil.copy(os.path.join(out_dir, f"{int(st * FPS):05d}.png"), os.path.join(stills_dir, f"still_{i+1}_{st:04.1f}s.png"))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(out_dir, "%05d.png"),
           "-i", "build/mix.wav", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
           "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", mp4]
    subprocess.run(cmd, check=True)
    print("wrote", mp4)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "frame":
        load(); k = int(float(sys.argv[2]) * FPS); render_frame(k).save(sys.argv[3]); sys.exit()
    main("build/frames", "art_of_dying_audio_first.mp4", "stills", [1.0, 6.0, 11.6, 22.5, 31.5])

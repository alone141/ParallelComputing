"""The Art of Dying, intro: the picture.

Every frame is posed from the time in seconds alone, through rhythm.py (the single source of rhythm
and timing). Nothing here re-types rhythm data; this file only decides how it looks.

    python3 film.py still <seconds> <out.png>     # one frame (snapped to 60 fps) + <out>.layout.json
    python3 film.py frames <f0> <f1> [--fps 60] [--half]
                                                  # raw RGB frames f0..f1-1 to stdout (render.sh)
    python3 film.py loopcheck                     # frame 0 vs the last frame, outside counter/playhead
    python3 film.py bench [frames]                # seconds per frame on one core

Picture, top to bottom (px on 1080x1920):
    header      GOJIRA · THE ART OF DYING / INTRO · 140 BPM, kept below the TikTok tab bar (y > 160)
    counter     t = <t>, and n = <n> · lap <k> of 13
    strip       the grouping 5 5 5 3 3 3 | 5 5 5 3 3 (24 + 21); the playhead's group lights up
    lanes       one per layer in rhythm.LAYERS order, x = 90..930, sixteenth t at the centre of cell t
    playhead    continuous: it crosses each mark at the instant the hit sounds
    equation    during a layer's focus lap: its name and Layer.equation, inside y = 1370..1536
"""

from __future__ import annotations

import json
import math
import sys
import time as clock
from pathlib import Path

import numpy as np
from PIL import Image

import rhythm as R
from film_gfx import (DISPLAY_BOLD, DISPLAY_SEMI, H, INTER_MED, MIST, MIST_L, MONO_BOLD, MONO_MED,
                      STAR, W, WHITE, LightLayer, TextLayer, composite, hexc, lerp_col, scale_col,
                      text_width)

# ---- layout (px) ------------------------------------------------------------------------------
X0, X1 = 90, 930                         # lane span
CW = (X1 - X0) / R.CYCLE                 # one sixteenth
TITLE_Y, SUB_Y = 212, 258                # baselines
COUNTER_Y, NLINE_Y = 380, 438
STRIP_Y, BRACKET_Y0, BRACKET_Y1, HALF_Y = 545, 556, 564, 597
LANE_TOP, LANE_GAP = 690, 122
LANE_Y = {layer.key: LANE_TOP + i * LANE_GAP for i, layer in enumerate(R.LAYERS)}
LABEL_DY = -30                           # lane label baseline above its line
PLAYHEAD_Y0, PLAYHEAD_Y1 = 614, LANE_Y[R.LAYERS[-1].key] + 40
EQ_NAME_Y, EQ_LINE_Y0, EQ_LEADING = 1400, 1444, 38
EQ_SIZE, EQ_MIN_SIZE = 30, 24
SAFE_RIGHT = 948                         # no text right of this between y = 780 and 1536
EQ_MAX_W = min(900, 2 * (SAFE_RIGHT - W / 2) - 8)   # centred lines: 808 px
EQ_RISE = 12                             # px; the lowest line still ends at or above y = 1536

# ---- look and motion --------------------------------------------------------------------------
AMBER = hexc(R.BY_KEY["rim"].colour)     # the grouping strip is the rim's own pattern
SNARE = hexc(R.SNARE_COLOUR)
DIM_UPCOMING = 0.30                      # marks still to come in this lap
FLASH, POP, HALO, HALO_A = 0.25, 0.35, 2.3, 0.35   # hit flash: decay (s), size pop, halo size, halo level
STROKE_HALO = 14.0                       # halo radius of a guitar stroke (18 px tall) at scale 1
FOCUS_GROW = 0.15                        # focused lane: marks this much larger
AFTERGLOW = 0.30                         # s: at each lap start the last lap's lit marks fade out
GROUP_FADE = 0.20                        # s: the group just left fades back to dim
GROUP_DIM = 0.40
RING_A = 0.28                            # previous lap's beats (pulse, ride), as outline rings
RAIL_IDLE, RAIL_LIVE, TICK_A = 0.10, 0.18, 0.12
LABEL_LEAD, LABEL_DUR, LABEL_RISE = 0.30, 0.45, 12   # label rises in from 0.3 s before entry
DOTS_IN = 0.35                           # s: a new lane's dim marks fade in after entry
EQ_STAGGER = 0.07                        # s between the equation's name and each line
PLAYHEAD_A = 0.55
SLIP_KEYS = ("pulse", "ride")            # the 4/4 layers: they slip one sixteenth per lap

# (layer key, hit kind) -> (shape, size, dy, colour or None for the lane colour)
MARKS = {
    ("rim", "click"): ("disc", 7.0, 0, None),
    ("pulse", "pulse"): ("disc", 9.5, 0, None),
    ("ks", "kick"): ("disc", 7.0, 0, None),
    ("ks", "snare"): ("disc", 7.5, -11, SNARE),
    ("ride", "ride"): ("disc", 4.5, 0, None),
    ("ride", "bell"): ("disc", 8.5, 0, None),
    ("bass", "note"): ("disc", 7.0, 0, None),
    ("gtr", "chug"): ("stroke", 1.0, 0, None),
}


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def cell_x(t: float) -> float:
    return X0 + (t + 0.5) * CW


def playhead_x(time: float) -> float:
    """Continuous: (time mod LAP) / LAP across the lane width, on the hit grid."""
    return cell_x((time % R.LAP) / R.LAP * R.CYCLE)


def flash_k(dt: float) -> float:
    """1 at the instant of a hit, decaying to 0 over FLASH seconds."""
    return (1.0 - dt / FLASH) ** 2 if 0.0 <= dt < FLASH else 0.0


def focus_look(key: str, time: float) -> float:
    """How focused the lane looks. rhythm.focus_weight, except that the rim clicks ease back into
    their opening (frame 0) focus while the other layers peel away, so the loop closes."""
    fw = R.focus_weight(key, time)
    if key == R.LAYERS[0].key:
        fw = max(fw, smooth((time - (R.TOTAL - R.PEEL)) / R.PEEL))
    return fw


def peel_factor(key: str, time: float) -> float:
    """rhythm.presence without the step at entry (1 until the layer peels away at the end), so a
    label can rise in before its layer enters."""
    return R.presence(key, max(time, R.entry_time(key)))


def arrival(key: str, time: float) -> tuple[float, float]:
    """(label rise 0..1, dim-marks fade-in 0..1). The rim clicks are there from frame 0."""
    if key == R.LAYERS[0].key:
        return 1.0, 1.0
    e = R.entry_time(key)
    return smooth((time - (e - LABEL_LEAD)) / LABEL_DUR), smooth((time - e) / DOTS_IN)


# ---- cached static pieces ---------------------------------------------------------------------
def _static_masks():
    rail = Image.new("L", (X1 - X0, 2), 255)
    ticks = Image.new("L", (X1 - X0, 9), 0)
    px = ticks.load()
    for t in range(R.CYCLE):
        x = int(round(cell_x(t) - X0))
        for y in range(9):
            px[x, y] = 255
    return rail, ticks


RAIL_MASK, TICK_MASK = _static_masks()


# ---- the frame --------------------------------------------------------------------------------
class Clock:
    """Where the music is at `time`."""

    def __init__(self, time: float):
        self.time = time
        self.n = int(math.floor(time / R.SIXTEENTH + 1e-9))           # sixteenths since the start
        self.lap = self.n // R.CYCLE
        self.t = self.n % R.CYCLE
        self.lap_n0 = self.lap * R.CYCLE
        self.since_lap = time - self.lap_n0 * R.SIXTEENTH
        self.group = max(i for i, s in enumerate(R.GROUP_STARTS) if s <= self.t)


def lane_marks(key: str, ck: Clock) -> list[dict]:
    """The marks to draw in one lane: this lap's hits (lit once played, flashing as they sound),
    plus the previous lap's, which fade out at the lap start (and stay as rings for the 4/4
    layers, so the slip shows). Only hits where the layer is present are shown. Lap 1's previous
    lap is the film's last lap, so frame 0 continues the last frame."""
    _, dots_in = arrival(key, ck.time)
    after = 1.0 - smooth(ck.since_lap / AFTERGLOW)
    marks: dict[tuple[int, str], dict] = {}
    for t in range(R.CYCLE):
        for lap_ofs in (-1, 0):
            n = ck.lap_n0 + lap_ofs * R.CYCLE + t
            nw = n % R.N_SIXTEENTHS                                  # wrapped into the film
            p = R.presence(key, nw * R.SIXTEENTH)
            if p <= 1e-4:
                continue
            for kind in R.hits_at(key, nw):
                m = marks.setdefault((t, kind), {"t": t, "kind": kind, "bright": 0.0, "flash": 0.0,
                                                 "ring": 0.0, "current": False})
                played = n <= ck.n
                k = flash_k(ck.time - n * R.SIXTEENTH) if played else 0.0
                if lap_ofs < 0:
                    m["bright"] = max(m["bright"], after * p)
                    m["ring"] = p if key in SLIP_KEYS else 0.0
                else:
                    m["bright"] = max(m["bright"], (1.0 if played else DIM_UPCOMING * dots_in) * p)
                    m["current"] = True
                m["flash"] = max(m["flash"], k * p)
    return list(marks.values())


def draw_lane(light: LightLayer, key: str, ck: Clock) -> dict:
    layer = R.BY_KEY[key]
    col = hexc(layer.colour)
    y = LANE_Y[key]
    op = R.lane_opacity(key, ck.time)
    _, dots_in = arrival(key, ck.time)
    live = op * dots_in
    fw = focus_look(key, ck.time)

    light.mask(TICK_MASK, X0, y - 4, scale_col(col, TICK_A * live))
    light.mask(RAIL_MASK, X0, y - 1, scale_col(col, RAIL_IDLE + (RAIL_LIVE - RAIL_IDLE) * live))
    if op <= 1e-4:
        return {"key": key, "y": y, "opacity": round(op, 4)}

    grow = 1.0 + FOCUS_GROW * fw
    marks = lane_marks(key, ck)
    for pass_ in ("ring", "halo", "mark"):
        for m in marks:
            shape, size, dy, mcol = MARKS[(key, m["kind"])]
            c = mcol or col
            x, my = cell_x(m["t"]), y + dy
            s = size * grow * (1.0 + POP * m["flash"])
            if pass_ == "ring" and m["ring"] > 0 and not m["current"]:
                light.ring(x, my, size * grow, scale_col(c, RING_A * m["ring"] * op))
            elif pass_ == "halo" and m["flash"] > 0:
                r = s * (HALO if shape == "disc" else STROKE_HALO)
                light.disc(x, my, r, scale_col(c, HALO_A * m["flash"] * op))
            elif pass_ == "mark" and m["bright"] > 0:
                cc = scale_col(c, m["bright"] * op)
                if shape == "disc":
                    light.disc(x, my, s, cc)
                else:
                    light.stroke(x, my, s, cc)
    return {"key": key, "y": y, "opacity": round(op, 4), "focus": round(fw, 4)}


def strip_groups(ck: Clock):
    """(group index, x from, x to, brightness) for the grouping strip. The group the playhead is
    in lights up; the one it just left fades back to dim."""
    prev = (ck.group - 1) % len(R.GROUPS)
    entered = (ck.lap_n0 + R.GROUP_STARTS[ck.group]) * R.SIXTEENTH
    out = []
    for i, (s, g) in enumerate(zip(R.GROUP_STARTS, R.GROUPS)):
        a = GROUP_DIM
        if i == ck.group:
            a = 1.0
        elif i == prev:
            a = GROUP_DIM + (1 - GROUP_DIM) * (1 - smooth((ck.time - entered) / GROUP_FADE))
        out.append((i, X0 + s * CW + 3, X0 + (s + g) * CW - 3, a))
    return out


SPLIT_X = X0 + R.HALVES[0] * CW          # between the 24 and the 21


def draw_strip_light(light: LightLayer, ck: Clock):
    for _, xa, xb, a in strip_groups(ck):
        light.line([(xa, BRACKET_Y0), (xa, BRACKET_Y1), (xb, BRACKET_Y1), (xb, BRACKET_Y0)],
                   scale_col(AMBER, 0.9 * a), 2)
    light.line([(SPLIT_X, STRIP_Y - 25), (SPLIT_X, BRACKET_Y1 + 16)], scale_col(MIST, 0.35), 1)


def draw_strip_text(txt: TextLayer, ck: Clock):
    for i, xa, xb, a in strip_groups(ck):
        txt.text((xa + xb) / 2, STRIP_Y, str(R.GROUPS[i]), MONO_BOLD, 24, AMBER, a,
                 kind="group_number")
    first = ck.t < R.HALVES[0]
    for (xa, xb), h, cur in (((X0, SPLIT_X), R.HALVES[0], first),
                             ((SPLIT_X, X1), R.HALVES[1], not first)):
        txt.text((xa + xb) / 2, HALF_Y, str(h), MONO_MED, 20, MIST_L if cur else MIST,
                 kind="half_total")


def draw_counter(txt: TextLayer, ck: Clock):
    lab, gap = "t =", 24
    w_lab = text_width(lab, MONO_MED, 46)
    w_val = text_width("44", MONO_BOLD, 110)                    # the widest value fixes the label
    x = W / 2 - (w_lab + gap + w_val) / 2
    txt.text(x, COUNTER_Y, lab, MONO_MED, 46, MIST_L, anchor="l", kind="counter_label")
    txt.text(x + w_lab + gap, COUNTER_Y, str(ck.t), MONO_BOLD, 110, WHITE, anchor="l",
             kind="counter_value")
    txt.text(W / 2, NLINE_Y, f"n = {ck.n} · lap {ck.lap + 1} of {R.N_LAPS}", MONO_MED, 24, MIST_L,
             0.9, kind="counter_line")


def draw_labels(txt: TextLayer, ck: Clock):
    for layer in R.LAYERS:
        key = layer.key
        rise, _ = arrival(key, ck.time)
        dim = 1.0 - (1.0 - R.FOCUS_DIM) * R.background_factor(key, ck.time)
        a = rise * dim * peel_factor(key, ck.time)
        col = lerp_col(MIST_L, hexc(layer.colour), focus_look(key, ck.time))
        y = LANE_Y[key] + LABEL_DY + LABEL_RISE * (1 - ease_out(rise))
        txt.text(X0, y, layer.name, INTER_MED, 24, col, a, anchor="l", tracking=2, kind="lane_label")


def _equation_size() -> int:
    """One size for every equation in the film: the largest (<= EQ_SIZE) at which every line
    fits EQ_MAX_W, never below EQ_MIN_SIZE."""
    lines = [s for layer in R.LAYERS for s in layer.equation]
    size = EQ_SIZE
    while size > EQ_MIN_SIZE and max(text_width(s, MONO_MED, size) for s in lines) > EQ_MAX_W:
        size -= 1
    return size


EQ_FONT_SIZE = _equation_size()


def draw_equation(txt: TextLayer, ck: Clock):
    key = R.focused_layer(ck.time)
    if key is None:
        return
    layer = R.BY_KEY[key]
    a0, a1, b0, b1 = R.equation_window(key)
    if not a0 <= ck.time < b1:
        return
    out = 1.0 - smooth((ck.time - b0) / (b1 - b0))
    items = [(layer.name, DISPLAY_BOLD, 34, hexc(layer.colour), 3, EQ_NAME_Y, "equation_name")]
    for i, line in enumerate(layer.equation):
        items.append((line, MONO_MED, EQ_FONT_SIZE, STAR if i == 0 else MIST_L, 0,
                      EQ_LINE_Y0 + i * EQ_LEADING, "equation_line"))
    dur = (a1 - a0) - EQ_STAGGER * (len(items) - 1)
    for k, (s, path, sz, col, tr, y, kind) in enumerate(items):
        x = (ck.time - (a0 + k * EQ_STAGGER)) / dur
        txt.text(W / 2, y + EQ_RISE * (1 - ease_out(x)), s, path, sz, col, smooth(x) * out,
                 tracking=tr, kind=kind)


def render(time: float):
    """The frame at `time` seconds: (RGB image 1080x1920, layout dict)."""
    ck = Clock(time)
    light = LightLayer()
    lanes = [draw_lane(light, layer.key, ck) for layer in R.LAYERS]
    px = playhead_x(time)
    light.vline_max(px, PLAYHEAD_Y0, PLAYHEAD_Y1, scale_col(STAR, PLAYHEAD_A))
    draw_strip_light(light, ck)

    img = composite(light.im)
    txt = TextLayer(img)
    txt.text(W / 2, TITLE_Y, "GOJIRA · THE ART OF DYING", DISPLAY_SEMI, 34, STAR, tracking=5,
             kind="title")
    txt.text(W / 2, SUB_Y, f"INTRO  ·  {R.BPM} BPM", MONO_MED, 24, MIST, tracking=2, kind="subtitle")
    draw_counter(txt, ck)
    draw_strip_text(txt, ck)
    draw_labels(txt, ck)
    draw_equation(txt, ck)
    layout = {
        "time": round(time, 4), "n": ck.n, "t": ck.t, "lap": ck.lap + 1, "size": [W, H],
        "focused": R.focused_layer(time),
        "playhead": {"x": round(px, 2), "y0": PLAYHEAD_Y0, "y1": PLAYHEAD_Y1},
        "lanes": lanes, "texts": txt.records,
    }
    return img, layout


# ---- command line -----------------------------------------------------------------------------
def frame_time(f: int, fps: int = R.FPS) -> float:
    return f / fps


def cmd_still(seconds: str, out: str):
    f = round(float(seconds) * R.FPS)
    img, layout = render(frame_time(f))
    layout["frame"] = f
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    lpath = out.with_suffix(".layout.json")
    lpath.write_text(json.dumps(layout, indent=1, ensure_ascii=False))
    print(f"{out}  {lpath}")


def cmd_frames(f0: int, f1: int, fps: int, half: bool):
    outb = sys.stdout.buffer
    for f in range(f0, f1):
        img, _ = render(frame_time(f, fps))
        if half:
            img = img.reduce(2)
        outb.write(img.tobytes())
    outb.flush()


def _mask_regions(layouts, pad=8, ph_half=64):
    """Pixels to ignore in the loop check: the counter texts, and a column around the playhead
    wide enough for the glow of the hit flashing under it."""
    keep = np.ones((H, W), bool)
    for lay in layouts:
        for rec in lay["texts"]:
            if rec["kind"].startswith("counter"):
                x0, y0, x1, y1 = rec["bbox"]
                keep[max(0, int(y0) - pad):int(y1) + pad, max(0, int(x0) - pad):int(x1) + pad] = False
        x = lay["playhead"]["x"]
        keep[PLAYHEAD_Y0 - 30:PLAYHEAD_Y1 + 30, max(0, int(x - ph_half)):int(x + ph_half)] = False
    return keep


def cmd_loopcheck():
    a, la = render(frame_time(0))
    b, lb = render(frame_time(R.N_FRAMES - 1))
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).astype(np.float32)
    keep = _mask_regions([la, lb])
    strip = keep.copy()
    strip[STRIP_Y - 40:HALF_Y + 12, :] = False
    print(f"frame 0 vs frame {R.N_FRAMES - 1}: mean |diff| outside counter + playhead "
          f"= {d[keep].mean():.4f} / 255 (max {d[keep].max():.0f}); also outside the strip "
          f"= {d[strip].mean():.4f} (max {d[strip].max():.0f}); whole frame = {d.mean():.4f}")


def cmd_bench(n: int):
    times = np.linspace(0, R.TOTAL, n, endpoint=False) + 0.37
    render(1.0)                                                    # warm the caches
    t0 = clock.perf_counter()
    for t in times:
        render(float(t))
    print(f"{(clock.perf_counter() - t0) / n:.4f} s/frame over {n} frames")


def main(argv):
    if len(argv) >= 3 and argv[0] == "still":
        return cmd_still(argv[1], argv[2])
    if len(argv) >= 3 and argv[0] == "frames":
        fps = int(argv[argv.index("--fps") + 1]) if "--fps" in argv else R.FPS
        return cmd_frames(int(argv[1]), int(argv[2]), fps, "--half" in argv)
    if argv[:1] == ["loopcheck"]:
        return cmd_loopcheck()
    if argv[:1] == ["bench"]:
        return cmd_bench(int(argv[1]) if len(argv) > 1 else 40)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Kinetic-typography film of the intro rhythm of Gojira's "The Art of Dying".

Nothing on screen but type.  Each rim click drops a counted digit (1..5 / 1..3)
into a stack of rows shaped like the groups 5 5 5 3 3 3 / 5 5 5 3 3 of the
45-sixteenth cycle.  The quarter-note beat is a second voice: a huge ghost
digit counting 1 2 3 4 behind the stack, and the digit it lands on turns
orange.  Because 45 = 11*4 + 1 the orange digits drift one column per lap.

Usage:
    python3 render.py still 7.3 out.png      # one frame at 7.3 s
    python3 render.py frames out_dir [fps]   # all frames as PNG (parallel)
    python3 render.py pipe | ffmpeg ...      # raw rgb24 frames on stdout
"""
import math
import os
import sys
from multiprocessing import Pool

from PIL import Image, ImageDraw, ImageFont

import rhythm as R
import timeline as T

W, H = 1080, 1920
FPS = 60
BG = (9, 9, 13)

MONO_B = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf"
MONO_XB = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-ExtraBold.ttf"
MONO_L = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Light.ttf"
DISP_BLACK = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
DISP_BOLD = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
DISP_MED = "/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf"
TEXT_MED = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"

_fonts = {}


def font(path, size):
    key = (path, int(size))
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, int(size))
    return _fonts[key]


# ----- layout -------------------------------------------------------------
# Grid: 11 rows (the groups), up to 5 columns.  Right-hand button column and
# bottom 20% of the frame stay free of essential type.
COL_X0, COL_DX = 190, 150          # glyph centres x = 190 .. 790
ROW_Y0, ROW_DY = 350, 106          # row centres y = 350 .. 1410
GLYPH = 118                        # base digit size
ACCENT = (255, 106, 43)            # orange-red: the beat
WHITE = (242, 240, 236)
GREY = (150, 148, 150)
DIM = (82, 80, 86)
RESIDUE = (52, 50, 56)
RESIDUE_BEAT = (96, 52, 32)
CAPTION_Y = 1478                   # one line, just above the UI zone


def lerp(a, b, u):
    return a + (b - a) * u


def mix(c1, c2, u):
    u = max(0.0, min(1.0, u))
    return tuple(int(round(lerp(a, b, u))) for a, b in zip(c1, c2))


def ease_out(u):
    u = max(0.0, min(1.0, u))
    return 1 - (1 - u) ** 3


def draw_centred(draw, xy, text, fnt, fill):
    x, y = xy
    draw.text((x, y), text, font=fnt, fill=fill, anchor="mm")


def ghost_layer(img, digit, alpha, scale=1.0):
    """Huge translucent digit behind everything: the beat's voice."""
    if alpha <= 0.002:
        return
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    fnt = font(DISP_BLACK, 1150 * scale)
    d.text((540, 905), digit, font=fnt, fill=ACCENT + (int(255 * alpha),), anchor="mm")
    img.alpha_composite(layer)


def render_frame(i, fps=FPS):
    time = i / fps
    g_exact = time / R.SIX                    # global sixteenth, fractional
    g = int(math.floor(g_exact))
    lap, t = divmod(g, R.CYCLE)
    frac = g_exact - g                        # position inside the sixteenth
    mode = T.lap_mode(lap)

    img = Image.new("RGBA", (W, H), BG + (255,))

    # --- second voice: the beat as a ghost digit -----------------------------
    if mode in ("beat", "band"):
        beats_on = True
    else:
        beats_on = False
    if beats_on:
        gb = (g // 4) * 4                     # last beat at or before now
        since = (g_exact - gb) * R.SIX        # seconds since that beat
        if gb >= T.FIRST_BEAT:
            n = R.beat_count(gb, T.FIRST_BEAT)
            a = 0.09 + 0.30 * math.exp(-since / 0.16)
            sc = 1.0 + 0.045 * math.exp(-since / 0.10)
            ghost_layer(img, str(n), a, sc)
        # grace note a 32nd before the next beat: a faint pre-flash
        nb = gb + 4
        to_next = (nb - g_exact) * R.SIX
        if 0 < to_next <= R.SIX / 2 and nb >= T.FIRST_BEAT:
            ghost_layer(img, str(R.beat_count(nb, T.FIRST_BEAT)), 0.05, 0.97)

    draw = ImageDraw.Draw(img)

    # --- the counted grid ------------------------------------------------------
    if mode != "none":
        # residue of the previous lap, very dim, so the eye knows where digits land
        if lap > 0 and T.lap_mode(lap - 1) != "none":
            for tt in range(R.CYCLE):
                if tt <= t:
                    continue                  # overwritten slots: nothing yet
                glyph = T.glyph_for(lap - 1, tt)
                r, c, size = R.slot_of(tt)
                x, y = COL_X0 + c * COL_DX, ROW_Y0 + r * ROW_DY
                if glyph:
                    gg = (lap - 1) * R.CYCLE + tt
                    was_beat = T.lap_mode(lap - 1) in ("beat", "band") and R.is_beat(gg) and gg >= T.FIRST_BEAT
                    draw_centred(draw, (x, y), glyph, font(MONO_XB if was_beat else MONO_B, GLYPH * (1.18 if was_beat else 1.0)),
                                 RESIDUE_BEAT if was_beat else RESIDUE)
                elif tt in R.RESTS and T.lap_mode(lap - 1) != "band":
                    draw_centred(draw, (x, y), str(size), font(MONO_L, GLYPH * 0.92), RESIDUE)

        for tt in range(0, t + 1):
            glyph = T.glyph_for(lap, tt)
            if not glyph:
                continue
            r, c, size = R.slot_of(tt)
            x, y = COL_X0 + c * COL_DX, ROW_Y0 + r * ROW_DY
            age = (g_exact - (lap * R.CYCLE + tt)) * R.SIX     # seconds since onset
            on_beat = beats_on and R.is_beat(lap * R.CYCLE + tt) and (lap * R.CYCLE + tt) >= T.FIRST_BEAT
            if mode == "band" and glyph == "S":
                on_beat = False
                heavy = True
            else:
                heavy = False
            u = ease_out(age / 0.09)
            if on_beat:
                sc = lerp(1.55, 1.18, u)
                col = mix(WHITE, ACCENT, ease_out(age / 0.12))
                fnt = font(MONO_XB, GLYPH * sc)
            elif heavy:
                sc = lerp(1.5, 1.15, u)
                col = WHITE
                fnt = font(MONO_XB, GLYPH * sc)
            else:
                sc = lerp(1.45, 1.0, u)
                col = mix(WHITE, GREY, ease_out((age - 0.08) / 0.35))
                fnt = font(MONO_B, GLYPH * sc)
            draw_centred(draw, (x, y), glyph, fnt, col)

        # the rest: counted but not played.  A thin, dim digit: shape, no weight.
        for r, (s0, size) in enumerate(zip(R.GROUP_START, R.GROUPS)):
            rest = s0 + size - 1
            if t >= rest and mode != "band":
                age = (g_exact - (lap * R.CYCLE + rest)) * R.SIX
                col = mix(GREY, DIM, ease_out(age / 0.4))
                x, y = COL_X0 + (size - 1) * COL_DX, ROW_Y0 + r * ROW_DY
                draw_centred(draw, (x, y), str(size), font(MONO_L, GLYPH * 0.92), col)

        # running tallies in the empty cells of the 3-rows: 24, then 21, then 45
        if t >= 23:
            age = (g_exact - (lap * R.CYCLE + 23)) * R.SIX
            col = mix(WHITE, GREY, ease_out(age / 0.6))
            draw_centred(draw, (COL_X0 + 3.5 * COL_DX, ROW_Y0 + 4 * ROW_DY), "24",
                         font(DISP_BOLD, 150), col)
        if t >= 44:
            age = (g_exact - (lap * R.CYCLE + 44)) * R.SIX
            col = mix(WHITE, GREY, ease_out(age / 0.6))
            draw_centred(draw, (COL_X0 + 3.5 * COL_DX, ROW_Y0 + 9.5 * ROW_DY), "21",
                         font(DISP_BOLD, 150), col)

    # --- words -------------------------------------------------------------------
    for item in T.captions_at(time):
        kind, text, alpha = item
        if alpha <= 0.01:
            continue
        col = mix(BG, WHITE, alpha)
        if kind == "title":
            draw_centred(draw, (540, 190), text, font(DISP_BLACK, 112), col)
        elif kind == "headline":
            draw_centred(draw, (540, 200), text, font(DISP_BOLD, 72), col)
        elif kind == "kicker":
            draw_centred(draw, (540, 200), text, font(DISP_MED, 44), mix(BG, GREY, alpha))
        elif kind == "caption":
            draw_centred(draw, (540, CAPTION_Y), text, font(DISP_BOLD, 50), col)
        elif kind == "sub":
            draw_centred(draw, (540, CAPTION_Y + 58), text, font(TEXT_MED, 36), mix(BG, GREY, alpha))
        elif kind == "big":
            draw_centred(draw, (540, 905), text, font(DISP_BLACK, 300), col)
        elif kind == "equation":
            draw_centred(draw, (540, 905), text, font(DISP_BLACK, 150), col)

    return img.convert("RGB")


def _save(args):
    i, out, fps = args
    render_frame(i, fps).save(out)
    return i


def main():
    cmd = sys.argv[1]
    if cmd == "still":
        tsec = float(sys.argv[2])
        render_frame(int(round(tsec * FPS))).save(sys.argv[3])
    elif cmd == "frames":
        out = sys.argv[2]
        os.makedirs(out, exist_ok=True)
        n = T.total_frames(FPS)
        jobs = [(i, os.path.join(out, f"f{i:05d}.png"), FPS) for i in range(n)]
        with Pool(4) as p:
            for k, _ in enumerate(p.imap_unordered(_save, jobs, chunksize=8)):
                if k % 200 == 0:
                    print(f"{k}/{n}", file=sys.stderr, flush=True)
    elif cmd == "pipe":
        n = T.total_frames(FPS)
        out = sys.stdout.buffer
        with Pool(4) as p:
            for img in p.imap(_rgb, range(n), chunksize=8):
                out.write(img)
    else:
        raise SystemExit(__doc__)


def _rgb(i):
    return render_frame(i).tobytes()


if __name__ == "__main__":
    main()

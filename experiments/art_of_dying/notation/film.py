"""The film: timeline, frame renderer, audio track and ffmpeg mux.

    python3 film.py            # full render  -> out/art_of_dying_notation.mp4
    python3 film.py --still 7.3 out/still.png   # one frame
    python3 film.py --preview  # 10 fps, first 12 s, quick look

Vertical 1080x1920, 30 fps.  Everything is timed in sixteenths at 140 bpm.
"""
import math, os, sys, textwrap
from multiprocessing import Pool
import numpy as np
from PIL import Image

import engrave as E
import pattern as P
import synth as SY

W, H, FPS, S = 1080, 1920, 30, 2
SX = P.SIXTEENTH                    # 0.10714 s
LAP = 45 * SX                       # 4.8214 s
BEAT = 4 * SX
T_RIFF = 4 * BEAT                   # the riff starts after a four-click count-in
N_LAPS = 8                          # laps 0-1 rim, 2-5 + tribal hit, 6-7 full kit
T_OUTRO = T_RIFF + N_LAPS * LAP     # 40.29 s
T_END = T_OUTRO + 4.6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# ---- palette -----------------------------------------------------------------
BG = (13, 13, 17)
INK = (240, 236, 226)
DIM = (120, 118, 128)
MUTED = (170, 168, 176)
AMBER = (255, 152, 64)
TEAL = (84, 220, 208)
HOT = (255, 236, 200)
RED = (255, 92, 92)


def lerp(a, b, f):
    return a + (b - a) * f


def mix(c1, c2, f):
    return tuple(int(round(lerp(a, b, f))) for a, b in zip(c1, c2))


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# ---- layout -------------------------------------------------------------------
SP = 24
X0, X1 = 44, 1048
SYS_Y_RIM = (560, 900)               # staff top of the two systems (rim chart)
SYS_Y_KIT = (568, 985)               # the two-voice kit chart is taller
LEAD = (166, 44)


def make_staves(ys):
    out = []
    for (a, b), y, lead in zip(P.SYSTEMS, ys, LEAD):
        out.append(E.Staff(X0, X1, y, sp=SP, n_slots=b - a, lead=lead))
    return out


STAVES_RIM = make_staves(SYS_Y_RIM)
STAVES_KIT = make_staves(SYS_Y_KIT)


def sys_of(t):
    """(system index, local slot) for cycle slot t."""
    for i, (a, b) in enumerate(P.SYSTEMS):
        if a <= t < b:
            return i, t - a
    raise ValueError(t)


def groups_of(i):
    a, b = P.SYSTEMS[i]
    return [g for g, s in zip(P.GROUPS, P.GROUP_START) if a <= s < b]


def rim_voices():
    vs = []
    for i, (a, b) in enumerate(P.SYSTEMS):
        ev = {t - a: (["sidestick"] if P.RIM[t] else []) for t in range(a, b)}
        vs.append(E.Voice(STAVES_RIM[i], groups_of(i), ev, stem_up=True, secondary="full"))
    return vs


def kit_voices(beat_off, crash_first=False):
    """Upper (hands) and lower (feet) voices for the full-band groove with the
    quarter-note beat at slots beat_off, beat_off+4, ..."""
    ups, lows = [], []
    for i, (a, b) in enumerate(P.SYSTEMS):
        up, low = {}, {}
        for t in range(a, b):
            hands = []
            if (t - beat_off) % 4 == 0:
                hands.append("crash" if (crash_first and t == 0) else "bell")
            elif (t - beat_off) % 4 == 2:
                hands.append("ride")
            if t in P.SNARE_SLOTS:
                hands.append("snare")
            if hands:
                up[t - a] = hands
            low[t - a] = ["kick"] if t in P.KICK_SLOTS else []
        ups.append(E.Voice(STAVES_KIT[i], groups_of(i), up, stem_up=True, secondary="segments"))
        lows.append(E.Voice(STAVES_KIT[i], groups_of(i), low, stem_up=False, secondary="full", rest_pos=2))
    return ups, lows


RIM_V = rim_voices()
KIT_V = {off: kit_voices(off, crash_first=(off == 0)) for off in range(4)}


# ---- timeline helpers -----------------------------------------------------------

def phase(t):
    """Return (name, lap, slot_float) for time t."""
    if t < T_RIFF:
        return "hook", -1, 0.0
    if t >= T_OUTRO:
        return "outro", N_LAPS, 0.0
    u = (t - T_RIFF) / SX
    lap = int(u // 45)
    slot = u - 45 * lap
    if lap < 2:
        return "rim", lap, slot
    if lap < 6:
        return "beat", lap, slot
    return "kit", lap, slot


def beat_off_for_lap(lap):
    """The tribal/kit beat grid starts at lap 2, slot 0."""
    return (-(45 * (lap - 2))) % 4


HEADLINES = {
    "hook": ("THIS DRUM INTRO\nIS IN 45/16", "Gojira · The Art of Dying · 140 bpm"),
    "rim0": ("THIS DRUM INTRO\nIS IN 45/16", "Gojira · The Art of Dying · 140 bpm"),
    "rim1": ("FIVES AND THREES", "5+5+5+3+3+3  +  5+5+5+3+3  =  45"),
    "beat": ("THE BEAT SLIPS", "45 = 11 × 4 + 1 · one sixteenth late per lap"),
    "kit": ("BAR 42:\nTHE WHOLE BAND", "kick · snare · ride · bell on the beat"),
    "outro": ("SCREENSHOT THIS", "45/16 · 140 bpm · the whole intro is this one bar"),
}

CAPTIONS = {
    "rim0": "Count every group out loud. A five is 1 2 3 4 (rest). A three is 1 2 (rest). "
            "The last sixteenth of every group is silent: that silence is the riff.",
    "rim1": "24 sixteenths on the top line, 21 on the bottom: 45 in all. Fans call it 21/16 "
            "or 45/16. Both are half right. Mario Duplantier plays it on the rim.",
    "beat2": "Lap 1 of 4. A tribal hit lands on every quarter note, every 4 sixteenths. "
             "The tab writes this as ten bars of 4/4 and one bar of 5/4.",
    "beat3": "Lap 2. 45 is not divisible by 4, so the beat now lands one sixteenth "
             "later in the pattern than it did last lap.",
    "beat4": "Lap 3. Two sixteenths late. The rim pattern never changes; "
             "only the beat drifts across it.",
    "beat5": "Lap 4. Three late. After this lap the beat and the pattern meet again: "
             "4 laps = 45 beats = 19.3 seconds.",
    "kit6": "A five is kick kick kick snare rest. A three is kick kick snare. The kick "
            "alternates feet, R L R. The ride plays eighths; the bell marks the beat.",
    "kit7": "Second lap: the beat has slipped one sixteenth, so the bell and the ride "
            "move one slot across the chart while kick and snare stay put.",
    "outro": "The whole 2:16 intro is this one bar of 45/16, repeated. Independent study "
             "of Gojira, The Art of Dying (2008). Drums synthesised, not sampled.",
}


def wrap(c, text, fnt, size, width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if c.text_width(trial, fnt, size) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ---- frame renderer ------------------------------------------------------------

def render_frame(t):
    img = Image.new("RGB", (W * S, H * S), BG)
    c = E.Canvas(img, S)
    name, lap, slot = phase(t)

    # --- headline
    if name == "hook":
        hk = "hook"
    elif name == "rim":
        hk = "rim%d" % lap
    else:
        hk = name
    head, sub = HEADLINES[hk]
    # fade headline in at phase starts
    if name == "hook":
        a = ease((t - 0.05) / 0.5)
    else:
        a = 1.0
    ink = mix(BG, INK, a)
    y = 170
    for line in head.split("\n"):
        c.text((X0 + 14, y), line, "display-x", 86, ink)
        y += 96
    c.text((X0 + 16, y + 6), sub, "text", 34, mix(BG, MUTED, a))

    # --- hit flashes: which slots were hit recently
    def flash(slot_hit):
        """0..1 intensity of the flash on slot_hit (float position slot)."""
        d = slot - slot_hit
        if d < 0:
            return 0.0
        return max(0.0, 1.0 - d / 1.6)

    # --- draw the score
    draw_p = None
    if name == "hook":
        draw_p = 45 * ease((t - 0.35) / 1.25)
        wipe = ease(t / 0.45)
    else:
        wipe = 1.0

    show_kit = name in ("kit", "outro")
    STAVES = STAVES_KIT if show_kit else STAVES_RIM
    for i, st in enumerate(STAVES):
        E.staff_lines(c, st, INK, x_to=st.x0 + (st.x1 - st.x0) * wipe)
        if wipe >= 1.0:
            if i == 0:
                E.percussion_clef(c, st.x0 + 16, st, INK)
                E.time_signature(c, st.x0 + 86, st, 45, 16, INK)
                E.barline(c, st.x0 + 138, st, INK, "repeat-start")
            else:
                E.barline(c, st.x1 - 6, st, INK, "repeat-end")

    def gcol_factory(i):
        a, b = P.SYSTEMS[i]
        gs = groups_of(i)
        first_g = P.SLOT_GROUP[a]

        def gcol(gi):
            g_abs = first_g + gi
            base = AMBER if gs[gi] == 5 else TEAL
            if name in ("hook", "outro"):
                return base
            active = (P.SLOT_GROUP[int(slot) % 45] == g_abs)
            return base if active else mix(BG, base, 0.62)
        return gcol

    def scol_factory(i):
        a, b = P.SYSTEMS[i]

        def scol(s):
            if name in ("hook", "outro"):
                return INK
            f = flash(s + a)
            return mix(INK, HOT, f) if f > 0 else mix(BG, INK, 0.78)
        return scol

    if show_kit:
        off = beat_off_for_lap(lap) if name == "kit" else 0
        ups, lows = KIT_V[off]
        for i in range(2):
            ups[i].draw(c, INK, slot_color=scol_factory(i), group_color=gcol_factory(i))
            lows[i].draw(c, INK, slot_color=scol_factory(i), group_color=gcol_factory(i))
        # R L R under the first group's kicks
        st = STAVES[0]
        yy = lows[0].y_beam + 0.3 * SP
        for s, lab in zip((0, 1, 2), ("R", "L", "R")):
            c.text((st.x_of(s), yy), lab, "text-semi", 22, DIM, anchor="mt")
    else:
        for i in range(2):
            a, b = P.SYSTEMS[i]
            pv = None if draw_p is None else (draw_p - a)
            if pv is not None and pv <= 0:
                continue
            RIM_V[i].draw(c, INK, p_visible=pv, slot_color=scol_factory(i), group_color=gcol_factory(i))

    # --- counts under each system
    if name != "hook":
        for i, st in enumerate(STAVES):
            a, b = P.SYSTEMS[i]
            ybase = (KIT_V[0][1][i].y_beam + 1.7 * SP) if show_kit else (st.y_bot + 2.3 * SP)
            for tt in range(a, b):
                n = P.SLOT_IN_GROUP[tt] + 1
                rest = tt in P.REST_SLOTS
                # in lap 0 the numbers appear as they are hit
                if name == "rim" and lap == 0 and tt > slot:
                    continue
                f = flash(tt)
                col = DIM if rest else MUTED
                size = 27
                if f > 0:
                    col = mix(col, HOT, f)
                    size = 27 + 10 * f
                c.text((st.x_of(tt - a), ybase), str(n), "mono", size, col, anchor="mt")

    # --- cursor
    if name in ("rim", "beat", "kit"):
        i, ls = sys_of(int(slot) % 45)
        st = STAVES[i]
        x = st.x_of(ls) + (slot - int(slot)) * st.slot_w - 0.5 * st.slot_w
        if show_kit:
            ytop = KIT_V[0][0][i].y_beam - 0.8 * SP
            ybot = KIT_V[0][1][i].y_beam + 3.4 * SP
        else:
            ytop = RIM_V[i].y_beam - 0.8 * SP
            ybot = st.y_bot + 4.0 * SP
        c.rect(x, ytop, x + st.slot_w, ybot, AMBER + (34,))
        c.rect(x - 1.5, ytop, x + 1.5, ybot, HOT + (200,))

    # --- beat markers (the tribal hit) above the staff, laps 2-5 (and kit laps)
    if name in ("beat", "kit"):
        off = beat_off_for_lap(lap)
        for tt in range(off, 45, 4):
            i, ls = sys_of(tt)
            st = STAVES[i]
            ybeam = (KIT_V[0][0][i].y_beam if show_kit else RIM_V[i].y_beam)
            yb = ybeam - 1.5 * SP
            f = flash(tt)
            r = 7 + 8 * f
            col = mix(RED, HOT, f)
            if tt >= slot and name == "beat":
                col = mix(BG, RED, 0.75)
            c.disc(st.x_of(ls), yb, r, col)
            if f > 0:
                c.disc(st.x_of(ls), yb, r + 10, col + (int(90 * f),))
        # label
        st = STAVES[0]
        lbl = "beat" if name == "beat" else "beat (bell)"
        c.text((st.x0 + 2, RIM_V[0].y_beam - 1.5 * SP), lbl, "text-semi", 22, RED, anchor="lm") if not show_kit else None

    # --- the tab's view: ten bars of 4/4 then one of 5/4, as chips, laps 2-5
    if name == "beat":
        y_chip = 1150
        beats_done = (t - (T_RIFF + 2 * LAP)) / BEAT         # 0..45
        x = X0 + 16
        cw, gap = 66, 9
        bar_start = 0
        for bi, nb in enumerate(P.TAB_BARS):
            active = bar_start <= beats_done < bar_start + nb
            done = beats_done >= bar_start + nb
            col = INK if (active or done) else mix(BG, INK, 0.35)
            c.rect(x, y_chip, x + cw, y_chip + 54, mix(BG, AMBER if active else INK, 0.16 if not active else 0.32))
            c.text((x + cw / 2, y_chip + 13), "%d/4" % nb, "mono", 21, col, anchor="mt")
            for k in range(nb):
                filled = beats_done >= bar_start + k + 1 - 1e-6 or (active and beats_done >= bar_start + k)
                dcol = (RED if (active and int(beats_done) == bar_start + k) else (col if filled else mix(BG, col, 0.3)))
                c.disc(x + cw / 2 + (k - (nb - 1) / 2) * 11, y_chip + 44, 3.6, dcol)
            x += cw + gap
            bar_start += nb
        c.text((X0 + 16, y_chip - 34), "the tab's bars  ·  lap %d of 4" % (lap - 1), "text-semi", 24, MUTED, anchor="lt")

    # --- caption
    if name != "hook":
        if name == "rim":
            key = "rim%d" % lap
        elif name == "beat":
            key = "beat%d" % lap
        elif name == "kit":
            key = "kit%d" % lap
        else:
            key = "outro"
        cap = CAPTIONS[key]
        ycap = 1250 if name == "beat" else 1180
        if show_kit:
            ycap = 1270
        lines = wrap(c, cap, "text", 36, 850)
        for line in lines:
            c.text((X0 + 16, ycap), line, "text", 36, INK)
            ycap += 50

    # --- count-in numbers during the hook
    if name == "hook":
        k = int(t // BEAT)
        for n in range(4):
            col = HOT if (n == k) else (INK if n < k else mix(BG, INK, 0.3))
            c.text((X0 + 16 + n * 70, 1180), str(n + 1), "mono", 54, col, anchor="lt")
        c.text((X0 + 16, 1250), "count-in · four stick clicks", "text", 30, MUTED, anchor="lt")

    # --- tiny footer credit (above the bottom 20%)
    c.text((X0 + 16, 1470), "Gojira — The Art of Dying (The Way of All Flesh, 2008) · independent study",
           "text", 20, DIM, anchor="lt")

    return img.resize((W, H), Image.LANCZOS)


# ---- audio ------------------------------------------------------------------

def build_audio(length):
    m = SY.Mix(length)
    st = SY.sticks(); ss = SY.side_stick(); th = SY.tribal_hit()
    kk = SY.kick(); sn = SY.snare(); rg = SY.ride_ghost(); rb = SY.ride_bell(); cr = SY.crash()
    g1 = SY.guitar_chug(); g2 = SY.guitar_chug(65.41 * 1.003)
    # count-in
    for n in range(4):
        m.add(st, n * BEAT, 2.0)
    # choir pad under laps 0-1
    pad = SY.choir_pad(2 * LAP + 1.5)
    fade = np.ones(len(pad)); nf = int(1.4 * SY.SR); fade[-nf:] = np.linspace(1, 0, nf)
    fade[: int(0.8 * SY.SR)] = np.linspace(0, 1, int(0.8 * SY.SR))
    m.add(pad * fade, T_RIFF, 0.22)
    for lap in range(N_LAPS + 1):           # lap 8 is the fade-out under the outro
        t0 = T_RIFF + lap * LAP
        for t in range(45):
            at = t0 + t * SX
            if lap < 6:
                if P.RIM[t]:
                    m.add(ss, at, 2.6)
            if 2 <= lap:
                off = beat_off_for_lap(lap)
                on_beat = (t - off) % 4 == 0
                if lap < 6 and on_beat:
                    m.add(st, at - SX / 2, 0.9)       # the grace note
                    m.add(th, at, 0.55)
                if lap >= 6:
                    if t in P.KICK_SLOTS:
                        m.add(kk, at, 0.8)
                    if t in P.SNARE_SLOTS:
                        m.add(sn, at, 0.7)
                    if on_beat:
                        if lap == 6 and t == 0:
                            m.add(cr, at, 0.8)
                        else:
                            m.add(rb, at, 0.55)
                    elif (t - off) % 4 == 2:
                        m.add(rg, at, 0.5)
                    m.add(g1, at, 0.33)
                    m.add(g2, at + 0.004, 0.33)
                    if P.RIM[t]:
                        hold = 2 if (t + 1) in P.REST_SLOTS else 1
                        m.add(SY.bass_note(length=hold * SX), at, 0.5)
    x = m.render(length, peak=0.9)
    # fade out the final lap under the outro card
    n0 = int(T_OUTRO * SY.SR); n1 = int((T_OUTRO + 3.6) * SY.SR)
    x[n0:n1] *= np.linspace(1, 0, n1 - n0) ** 1.5
    x[n1:] = 0
    return x


# ---- driver -------------------------------------------------------------------

def _frame_job(args):
    k, t, path = args
    render_frame(t).save(path, compress_level=1)
    return k


def main():
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) >= 3 and sys.argv[1] == "--still":
        t = float(sys.argv[2])
        path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(OUT, "still_%05.2f.png" % t)
        render_frame(t).save(path)
        print(path)
        return
    preview = "--preview" in sys.argv
    fps = 10 if preview else FPS
    length = 12.0 if preview else T_END
    frames_dir = os.path.join(os.environ.get("AOD_FRAMES", os.path.join(OUT, "frames")))
    os.makedirs(frames_dir, exist_ok=True)
    n = int(length * fps)
    jobs = [(k, k / fps, os.path.join(frames_dir, "f%05d.png" % k)) for k in range(n)]
    with Pool(4) as pool:
        for i, _ in enumerate(pool.imap_unordered(_frame_job, jobs, chunksize=8)):
            if i % 100 == 0:
                print("frames %d/%d" % (i, n), flush=True)
    wav = os.path.join(OUT, "audio.wav")
    SY.write_wav(wav, build_audio(length))
    mp4 = os.path.join(OUT, "preview.mp4" if preview else "art_of_dying_notation.mp4")
    cmd = ("ffmpeg -y -loglevel error -framerate %d -i %s/f%%05d.png -i %s -t %.3f "
           "-c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -r %d "
           "-af loudnorm=I=-14:TP=-1.5:LRA=11 -c:a aac -b:a 160k -movflags +faststart %s" % (fps, frames_dir, wav, length, fps, mp4))
    print(cmd)
    os.system(cmd)
    print("wrote", mp4)


if __name__ == "__main__":
    main()

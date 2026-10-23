"""b08 - THE CATCH: Amdahl's law (8.0 s, film 36.0-44.0).

Frame 0 is E07 (the '350,000+ / VOLUNTEER GPUs · 2020' HUD). The HUD cracks out (glitch)
and 'So… infinite speed?' pays off the hook's open loop. One CYAN job bar wipes in and
shimmers, the camera leans in, and at 1.50 'No.' hard-cuts in: the bar splits, its left 10%
turns into the AMBER serial block, and the HUD comes back as a SPEEDUP meter. The serial
block tries to split twice (crack, snap back). Four doublings of the GPU count halve the
parallel rows (1.8×, 3.0×, 4.7×, 6.4×) while an Amdahl curve grows on a log axis; then the
GPU count races to 1,000,000, the speedup turns AMBER and stalls at 9.9×, and the curve
flattens onto THE WALL. 5.90: 'GPUs: ∞', the HUD SLAMs '10×' with its assumption
('MAX SPEEDUP · 10% SERIAL'). The serial block heartbeats, then turns into the amber token
(E08) that b09 opens back into the lone hook worker.

Everything that moves is a pure function of self.now (one driver updater); the scene only
holds until the canonical exit swap at 7.95.
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# World layout (storyboard b08 visuals)
# --------------------------------------------------------------------------
N_XY = np.array([X_C, 2.95, 0.0])              # N readout (cap band centre)
BAR_X0, BAR_X1 = -3.70, 3.00                   # job bar
BAR_Y0, BAR_Y1 = 1.65, 2.45
BAR_C = np.array([(BAR_X0 + BAR_X1) / 2, (BAR_Y0 + BAR_Y1) / 2, 0.0])
BAR_R = 0.08
SER_X1 = -3.03                                 # serial block x[-3.70, -3.03]
SER_R = 0.06
SER_C = np.array([(BAR_X0 + SER_X1) / 2, BAR_C[1], 0.0])
SER_W, SER_H = SER_X1 - BAR_X0, BAR_Y1 - BAR_Y0
PAR_X0 = -2.97                                 # parallel part left edge (after the 0.06 gap)
PAR_W = BAR_X1 - PAR_X0                        # 5.97
ROW_GAP = 0.03
SLIVER_MIN = 0.03                              # storyboard 0.012 (1.4 px): too thin to read on a phone
FIN_W = 3.0                                    # finish-line stroke (0.03 units); its left edge hugs the rows
FIN_Y0, FIN_Y1 = 1.55, 2.55                    # finish line
LAB_Y = 1.35                                   # bar labels

GX0, GX1, GY0, GY1 = -3.40, 2.80, -2.20, 0.65  # graph axes (y top 0.85 in the storyboard)
Y_PER_S = 0.25                                 # storyboard 0.27: 0.25 frees 0.2 above the wall label
X_PER_DEC = (GX1 - GX0) / 6.0                  # 1.0333 per decade: N = 1e6 lands on x 2.80
LOG_MAX = 6.0


def gx(logn):
    return GX0 + X_PER_DEC * np.asarray(logn, float)


def gy(s):
    return GY0 + Y_PER_S * np.asarray(s, float)


WALL_Y = float(gy(10.0))                       # 0.30 (storyboard 0.50)
WALL_DASH = 0.15
N_DASH = 21
WALL_GAP = (GX1 - GX0 - N_DASH * WALL_DASH) / (N_DASH - 1)
WALL_LAB = (2.80, WALL_Y + 0.32)               # right edge, centre y (0.32 above the wall)
HEAD_R = 0.07

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
T_GLITCH1 = 0.30                               # HUD cracks out 0.00-0.30
T_Q_IN = 0.10                                  # 'So… infinite / speed?'
T_WIPE0, T_WIPE1 = 0.30, 0.75                  # bar wipe (ease_out_cubic)
T_N_IN = 0.45                                  # 'GPUs: 1' POP
T_SHIM = 0.75                                  # shimmer
T_PUSH0, T_PUSH1 = 1.00, 1.50                  # push-in 1.00 -> 1.02
T_NO = 1.50                                    # 'No.' HIT: split, HUD back
T_GAP = 0.12
T_LABELS = 1.60
T_CRACKS = (1.95, 2.35)
T_NO_GONE = 2.60
T_AXES0, T_AXES1 = 2.60, 3.00
T_HEAD = 2.60
T_DBL = (2.70, 3.10, 3.50, 3.90)               # N 1 -> 2 -> 4 -> 8 -> 16
DBL_DUR = 0.30
T_SOME_IN, T_SOME_GONE = 2.60, 4.45
T_RAMP0, T_RAMP1 = 4.30, 5.90
T_DIM0, T_DIM1 = 4.45, 4.60
T_WALL0, T_WALL1 = 4.90, 5.15
T_HIT = 5.90                                   # 'GPUs: ∞', '10×'
T_TAKE = 5.95                                  # '10% serial / = max 10×.'
T_HB1 = 7.50                                   # heartbeat lands on exactly 1.0
T_CLEAR0, T_CLEAR1 = 7.55, 7.80
T_MORPH0, T_MORPH1 = 7.55, 7.95
T_TEXT_OUT = 7.85
T_EXIT = 7.95                                  # canonical E08 for the last 3 frames

LOG16 = math.log10(16)

# serial-block crack geometry (zigzag down the middle)
CRACK_DX = np.array([0.0, 0.028, -0.022, 0.030, -0.026, 0.018, 0.0])
CRACK_Y = np.linspace(BAR_Y1, BAR_Y0, len(CRACK_DX))
CRACK_SEP = 0.035
CRACK_SHAKE = (0.05, -0.05, 0.035, -0.02)      # x offsets on the 4 snap-back frames
CRACK_ALPHA = (0.75, 0.5, 0.28, 0.12)

C_CY = np.array(CYAN.to_rgb())
C_AM = np.array(AMBER.to_rgb())
C_INK = np.array(INK.to_rgb())
C_DIM = np.array(DIM.to_rgb())

EMPTY = np.zeros((0, 3))
KAPPA = 0.5522847498
_CIRCLE = Circle(radius=1.0).points.copy()


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def seg(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    return np.array([a, a + d / 3, a + 2 * d / 3, b])


def corner(a, b, c) -> np.ndarray:
    """Quarter-arc cubic from a to c with the (sharp) corner at b."""
    a, b, c = (np.asarray(p, float) for p in (a, b, c))
    return np.array([a, a + KAPPA * (b - a), c + KAPPA * (b - c), c])


def P(x, y):
    return np.array([x, y, 0.0])


def rr_pts(x0, y0, x1, y1, r) -> np.ndarray:
    """Rounded rectangle as 8 cubic segments (fixed structure, r clamped)."""
    if x1 - x0 < 1e-5 or y1 - y0 < 1e-5:
        return EMPTY.copy()
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    return np.concatenate([
        seg(P(x0 + r, y0), P(x1 - r, y0)), corner(P(x1 - r, y0), P(x1, y0), P(x1, y0 + r)),
        seg(P(x1, y0 + r), P(x1, y1 - r)), corner(P(x1, y1 - r), P(x1, y1), P(x1 - r, y1)),
        seg(P(x1 - r, y1), P(x0 + r, y1)), corner(P(x0 + r, y1), P(x0, y1), P(x0, y1 - r)),
        seg(P(x0, y1 - r), P(x0, y0 + r)), corner(P(x0, y0 + r), P(x0, y0), P(x0 + r, y0)),
    ])


def rr_c(c, w, h, r) -> np.ndarray:
    return rr_pts(c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2, r)


def polyline(pts) -> np.ndarray:
    pts = np.asarray(pts, float)
    a, b = pts[:-1], pts[1:]
    d = b - a
    return np.stack([a, a + d / 3, a + 2 * d / 3, b], axis=1).reshape(-1, 3)


def _half_pts(side: int) -> np.ndarray:
    """Left (-1) / right (+1) half of the serial block, split along the zigzag crack."""
    x0, x1, y0, y1, r = BAR_X0, SER_X1, BAR_Y0, BAR_Y1, SER_R
    zz = [P(SER_C[0] + dx, y) for dx, y in zip(CRACK_DX, CRACK_Y)]   # top -> bottom
    out = []
    if side < 0:
        out.append(seg(zz[0], P(x0 + r, y1)))
        out.append(corner(P(x0 + r, y1), P(x0, y1), P(x0, y1 - r)))
        out.append(seg(P(x0, y1 - r), P(x0, y0 + r)))
        out.append(corner(P(x0, y0 + r), P(x0, y0), P(x0 + r, y0)))
        out.append(seg(P(x0 + r, y0), zz[-1]))
        for a, b in zip(zz[::-1][:-1], zz[::-1][1:]):
            out.append(seg(a, b))
    else:
        for a, b in zip(zz[:-1], zz[1:]):
            out.append(seg(a, b))
        out.append(seg(zz[-1], P(x1 - r, y0)))
        out.append(corner(P(x1 - r, y0), P(x1, y0), P(x1, y0 + r)))
        out.append(seg(P(x1, y0 + r), P(x1, y1 - r)))
        out.append(corner(P(x1, y1 - r), P(x1, y1), P(x1 - r, y1)))
        out.append(seg(P(x1 - r, y1), zz[0]))
    return np.concatenate(out)


HALF_L, HALF_R = _half_pts(-1), _half_pts(1)
CRACK_PTS = polyline([P(SER_C[0] + dx, y) for dx, y in zip(CRACK_DX, CRACK_Y)])


def log_n_ramp(now: float) -> float:
    """log10 N during the ramp: 16 -> 1,000,000 (ease_in_out_cubic, 4.30-5.90)."""
    return LOG16 + (LOG_MAX - LOG16) * ease_in_out_cubic(ramp(now, T_RAMP0, T_RAMP1))


def dbl_index(now: float) -> int:
    """Index of the latest doubling started at or before now (-1 before the first)."""
    k = -1
    for i, t in enumerate(T_DBL):
        if now >= t - EPS:
            k = i
    return k


def log_n_curve(now: float) -> float:
    """log10 N the curve (and its head) has reached."""
    if now >= T_HIT - EPS:
        return LOG_MAX
    if now >= T_RAMP0 - EPS:
        return log_n_ramp(now)
    k = dbl_index(now)
    if k < 0:
        return 0.0
    return math.log10(2) * (k + ease_out_cubic(ramp(now, T_DBL[k], T_DBL[k] + DBL_DUR)))


def n_display(now: float):
    """Integer GPU count shown in the readout (None = hidden, inf = '∞')."""
    if now < T_N_IN - EPS:
        return None
    if now >= T_HIT - EPS:
        return math.inf
    if now >= T_RAMP0 - EPS:
        return int(round(10 ** log_n_ramp(now)))
    return 2 ** (dbl_index(now) + 1)


def n_str(n) -> str:
    return "GPUs: ∞" if n == math.inf else f"GPUs: {n:,}"


def heartbeat(now: float) -> float:
    """0..1 'lub-dub' envelope, 3 beats over 5.90-7.50 (~1.9 Hz), exactly 0 at both ends."""
    if now < T_HIT - EPS or now >= T_HB1 - EPS:
        return 0.0
    per = (T_HB1 - T_HIT) / 3
    q = ((now - T_HIT) % per) / per

    def bump(a, b):
        return math.sin(math.pi * (q - a) / (b - a)) ** 2 if a <= q <= b else 0.0
    return min(1.0, bump(0.0, 0.30) + 0.55 * bump(0.28, 0.66))


def row_h(n: int) -> float:
    return (SER_H - ROW_GAP * (n - 1)) / n


def row_r(h: float, w: float) -> float:
    return min(SER_R, 0.3 * h, 0.3 * w)


def row_rect(n: int, k: int):
    """Row k (top = 0) of the n stacked parallel rows at rest: (x0, y0, x1, y1, r)."""
    h, w = row_h(n), PAR_W / n
    top = BAR_Y1 - k * (h + ROW_GAP)
    return np.array([PAR_X0, top - h, PAR_X0 + w, top, row_r(h, w)])


def curve_rgb(s):
    """Curve colour: CYAN, heating through a pale INK-lifted mix into AMBER as S -> 10."""
    w = np.clip((np.asarray(s, float) - 7.0) / 2.7, 0.0, 1.0)
    w = w * w * (3 - 2 * w)
    rgb = C_CY[None, :] * (1 - w[:, None]) + C_AM[None, :] * w[:, None]
    lift = 0.35 * np.sin(np.pi * w)[:, None]
    return rgb * (1 - lift) + C_INK[None, :] * lift


def vm(**kw) -> VMobject:
    m = VMobject(stroke_width=0, fill_opacity=0, **kw)
    m.points = EMPTY.copy()
    return m


def set_style(m: VMobject, fill=None, stroke=None, sw=None) -> None:
    m.fill_rgbas = np.zeros((1, 4)) if fill is None else np.atleast_2d(np.asarray(fill, float))
    m.stroke_rgbas = np.zeros((1, 4)) if stroke is None else np.atleast_2d(np.asarray(stroke, float))
    if sw is not None:
        m.stroke_width = sw


def stage_text(text, font, size, weight, color):
    t = Text(text, font=font, font_size=size, color=color, weight=weight)
    return t


class B08Amdahl(BeatScene):
    def beat(self):
        # ---------------- entry: E07 (HUD only) ----------------
        st = enter(self)
        hud = self.hud = st.hud

        # ---------------- world (pooled VMobjects, drawn by the driver) ----------------
        self.axes = VGroup(vm(), vm())
        self.wall = vm()
        self.curve = vm()
        self.curve.sheen_direction = np.array([1.0, 0.0, 0.0])
        self.curve.joint_type = LineJointType.ROUND
        self.curve.cap_style = CapStyleType.ROUND
        self.head = vm()
        self.bar = vm()
        self.bar.sheen_direction = np.array([1.0, 0.0, 0.0])
        self.rows = VGroup(*[vm() for _ in range(16)])
        self.finish = vm()
        self.serial = vm()
        self.halves = VGroup(vm(), vm())
        self.crack = vm()
        self.crack.joint_type = LineJointType.ROUND
        self.graph = VGroup(self.axes, self.wall, self.curve, self.head)
        self.barg = VGroup(self.bar, self.rows, self.finish, self.halves, self.crack, self.serial)

        # ---------------- stage text (TextSlots, drawn by the driver) ----------------
        lab_a = stage_text("10% can't split", FONT_BODY, 34, MEDIUM, AMBER)
        lab_b = stage_text("90% can split", FONT_BODY, 34, MEDIUM, CYAN)
        for t, edge in ((lab_a, "left"), (lab_b, "right")):
            b, top = cap_box(t)
            dx = (BAR_X0 - t.get_left()[0]) if edge == "left" else (BAR_X1 - t.get_right()[0])
            t.shift([dx, LAB_Y - (b + top) / 2, 0])
            assert_safe(t)
        assert lab_b.get_left()[0] - lab_a.get_right()[0] > 0.3
        wl = stage_text("THE WALL: 10×", FONT_DISPLAY, 40, BOLD, AMBER)
        b, top = cap_box(wl)
        wl.shift([WALL_LAB[0] - wl.get_right()[0], WALL_LAB[1] - (b + top) / 2, 0])
        assert_safe(wl)
        self.t_lab_a, self.t_lab_b, self.t_wall = lab_a, lab_b, wl
        self.s_lab_a, self.s_lab_b, self.s_wall = TextSlot(), TextSlot(), TextSlot()
        self.s_n, self.s_nc, self.s_nm = TextSlot(), TextSlot(), TextSlot()
        self._ncache: dict = {}
        self.stage_txt = VGroup(self.s_lab_a, self.s_lab_b, self.s_wall, self.s_nc, self.s_nm, self.s_n)

        # ---------------- cards + kicker (frame-exact schedule) ----------------
        tr = TextTrack(self)
        q = card([("So… infinite", "H84", INK), ("speed?", "H84", INK)])
        for ln in q:
            tr.on(ln, T_Q_IN, T_NO, exit="cut")
        no = card([("No.", "H150", AMBER)])[0]
        # Integration fix: the H150 'No.' sits ~0.1 under the kicker, so a centre SLAM (1.3x)
        # and CARD_OUT's 0.15 upward drift both pushed it into 'THE CATCH · AMDAHL'S LAW'.
        # SLAM about the top edge (grows downward) and fade out in place.
        no_out = T_NO_GONE - 0.20
        tr.on(no, T_NO, no_out, enter="slam", about=no.get_top(),
              shift=lambda now: DOWN * 0.15 * smooth(clip01((now - no_out) / 0.20)))
        some = card([("Some work", "H84", INK), ("can't be split.", "H84", INK, {"can't": AMBER})])
        tr.on(some[0], T_SOME_IN, T_SOME_GONE - 0.20)
        tr.on(some[1], T_SOME_IN + 0.08, T_SOME_GONE - 0.20)
        take = card([("10% serial", "H84", INK, {"10% serial": AMBER}),
                     ("= max 10×.", "H84", INK, {"10×": AMBER})])
        tr.on(take[0], T_TAKE, T_TEXT_OUT, exit_dur=T_EXIT - T_TEXT_OUT)
        tr.on(take[1], T_TAKE + 0.08, T_TEXT_OUT, exit_dur=T_EXIT - T_TEXT_OUT)
        tr.on(kick("THE CATCH · AMDAHL'S LAW", AMBER, tracking=80), T_NO, T_TEXT_OUT,
              exit_dur=T_EXIT - T_TEXT_OUT)

        # ---------------- HUD schedule ----------------
        hud.show_at(0.0)
        hud.glitch(0.0, T_GLITCH1, amp=0.06, tint=AMBER, split_frames=(6, 14), seed=8)
        hud.hide_at(T_GLITCH1)
        hud.show_at(T_NO)
        hud.cut(T_NO, "1.0×", INK)
        hud.pop(T_NO)
        hud.label(T_NO, "SPEEDUP", INK, 0.70)
        for t, n in zip(T_DBL, (2, 4, 8, 16)):
            hud.pop(t, speedup_str(amdahl(n)))

        def meter(now):
            s = amdahl(10 ** log_n_ramp(now))
            return speedup_str(s), (AMBER if s >= 9.5 else INK)
        hud.drive(T_RAMP0, meter)
        hud.slam(T_HIT, "10×", AMBER)
        hud.label(T_HIT, "MAX SPEEDUP · 10% SERIAL", AMBER, 0.85)

        # ---------------- z-order: dust, graph, bar, stage text, cards, HUD, flash ----------------
        add_behind(self, self.graph, self.barg, self.stage_txt)
        self.add(tr)
        self.remove(hud)
        self.add(hud)
        self.add(FrameFlash(self, T_HIT, 0.12, 0.30, rise=0.15, color=AMBER))
        self._draw(0.0)
        driver = Mobject()
        driver.add_updater(lambda m, dt: self._draw(self.now))
        self.add(driver)

        # ---------------- audio (storyboard cue list) ----------------
        self.cue("glitch", dt=0.0, gain=0.8, dur=0.30, note="the counter cracks out of the silence")
        self.cue("whoosh", dt=0.52, dur=0.45, direction="up", note="bar wipe")
        self.cue("silence", dt=1.00, dur=0.50, note="duck the bed under 'So… infinite speed?'")
        self.cue("hit", dt=T_NO, gain=1.0, note="'No.'")
        self.cue("sub_drop", dt=T_NO, dur=1.6, note="dark")
        for t in T_CRACKS:
            self.cue("glitch", dt=t, gain=0.5, dur=0.12, note="failed split")
        for t, fr in zip(T_DBL, (2400, 3000, 3600, 4400)):
            self.cue("tick", dt=t, gain=0.9, freq=fr, pan=0, note="doubling")
        self.cue("ticks_burst", dt=T_RAMP0, dur=0.72, rate0=30, rate1=4,
                 note="speedup ticks slowing, silent at ~5.02 (the stall)")
        self.cue("riser", dt=T_RAMP0, dur=1.60, note="into the wall")
        self.cue("hit", dt=T_WALL1, gain=0.6, note="THE WALL label")
        self.cue("hit", dt=T_HIT, gain=1.0, note="'10×' and '∞'")
        self.cue("sub_drop", dt=T_HIT, dur=2.0)
        self.cue("pad_swell", dt=T_TAKE, dur=1.6, chord=1, note="dark chord under the takeaway")
        self.cue("whoosh", dt=7.75, dur=0.40, direction="down", note="serial block becomes the token")

        # ---------------- run ----------------
        self.hold_until(T_EXIT)
        exit_to(self)                              # canonical E08, static 7.95-8.00

    # ======================================================================
    # Driver
    # ======================================================================
    def _draw(self, now: float) -> None:
        f = story_frame(now)
        sh = shake(now - T_NO, 0.06, 0.15, seed=81) + shake(now - T_HIT, 0.06, 0.20, seed=82)
        clear = 1.0 - smooth(ramp(now, T_CLEAR0, T_CLEAR1))
        self._draw_bar(now, f, sh, clear)
        self._draw_graph(now, f, sh, clear)
        self._draw_labels(now, f, sh, clear)
        self._draw_readout(now, f, sh, clear)

    # ---------------- job bar / serial block / parallel rows ----------------
    def _draw_bar(self, now, f, sh, clear) -> None:
        if now < T_NO - EPS:
            for m in (self.serial, *self.halves, self.crack, self.finish, *self.rows):
                m.points = EMPTY.copy()
            w = (BAR_X1 - BAR_X0) * ease_out_cubic(ramp(now, T_WIPE0, T_WIPE1))
            if w < 1e-4:
                self.bar.points = EMPTY.copy()
                return
            k = 1.0 + 0.02 * ease_in_out_sine(ramp(now, T_PUSH0, T_PUSH1))
            x0, x1 = BAR_X0, BAR_X0 + w
            pts = rr_pts(x0, BAR_Y0, x1, BAR_Y1, BAR_R)
            self.bar.points = (pts - BAR_C) * k + BAR_C
            # fill gradient across x: wipe head (INK-lifted leading edge) + travelling shimmer
            K = 40
            u = np.linspace(0.0, 1.0, K)
            x = x0 + u * w
            op = np.full(K, 0.75)
            amp = 0.15 * ramp(now, T_SHIM, T_SHIM + 0.20)
            if amp > 0:
                op = op + amp * np.sin(TAU * (1.5 * now - x / 3.0))
            head = (1.0 - ramp(now, T_WIPE1 - 0.10, T_WIPE1 + 0.12)) * np.clip((x - (x1 - 0.6)) / 0.6, 0, 1) ** 2
            rgb = C_CY[None, :] * (1 - 0.6 * head[:, None]) + C_INK[None, :] * 0.6 * head[:, None]
            op = np.clip(op + 0.25 * head, 0, 1)
            self.bar.fill_rgbas = np.column_stack([rgb, op])
            self.bar.stroke_rgbas = np.zeros((1, 4))
            self.bar.stroke_width = 0
            return
        self.bar.points = EMPTY.copy()

        # ---- serial block
        hb = heartbeat(now)
        flash = max(1.0 - ease_out_cubic(ramp(now, T_NO, T_NO + 0.22)),
                    1.0 - ease_out_cubic(ramp(now, T_HIT, T_HIT + 0.30)) if now >= T_HIT - EPS else 0.0)
        s_fill = np.array([*lerp(C_AM, C_INK, 0.65 * flash), 0.95])
        s_stroke = np.array([*C_INK, 1.0])
        crack_df = None
        for t in T_CRACKS:
            df = f - story_frame(t)
            if 0 <= df <= 6:
                crack_df = df
        dx = 0.0
        if crack_df is not None and crack_df >= 3:
            dx = CRACK_SHAKE[crack_df - 3]
        off = sh + np.array([dx, 0.0, 0.0])

        if now >= T_MORPH0 - EPS:
            e = ease_in_out_cubic(ramp(now, T_MORPH0, T_MORPH1))
            c = lerp(SER_C, AC, e)
            w, h = lerp(SER_W, TOKEN_SIZE, e), lerp(SER_H, TOKEN_SIZE, e)
            r = lerp(SER_R, TOKEN_CORNER * TOKEN_SIZE, e)
            self.serial.points = rr_c(c, w, h, r)
            set_style(self.serial, [*C_AM, lerp(0.95, TOKEN_FILL, e)],
                      [*lerp(C_INK, C_AM, e), 1.0], lerp(2.0, TOKEN_STROKE, e))
            for m in (*self.halves, self.crack):
                m.points = EMPTY.copy()
        elif crack_df is not None and crack_df <= 2:
            self.serial.points = EMPTY.copy()
            for m, base, sgn in ((self.halves[0], HALF_L, -1), (self.halves[1], HALF_R, 1)):
                m.points = base + off + np.array([sgn * CRACK_SEP, 0.0, 0.0])
                set_style(m, [*lerp(C_AM, C_INK, 0.25), 0.95], s_stroke, 2.0)
            self.crack.points = CRACK_PTS + off
            set_style(self.crack, None, [*C_INK, 1.0], 3.0)
        else:
            sc = 1.0 + 0.08 * hb
            self.serial.points = rr_c(SER_C + off, SER_W * sc, SER_H * sc, SER_R * sc)
            set_style(self.serial, s_fill, s_stroke, 2.0)
            for m in self.halves:
                m.points = EMPTY.copy()
            if crack_df is not None:
                self.crack.points = CRACK_PTS + off
                set_style(self.crack, None, [*C_INK, CRACK_ALPHA[crack_df - 3]], 2.5)
            else:
                self.crack.points = EMPTY.copy()

        # ---- parallel rows
        rects, gl = self._rows(now)
        flick = 1.0
        if T_HIT - EPS <= now < T_HB1 - EPS:
            flick = 0.8 + 0.2 * jitter(f // 3, 1.0, seed=58)          # 0.6 .. 1.0 (/0.8)
            flick /= 0.8
        rgb = lerp(C_CY, C_INK, 0.45 * gl)
        op = min(1.0, (0.8 + 0.2 * gl) * flick) * clear
        for i, m in enumerate(self.rows):
            if i < len(rects):
                x0, y0, x1, y1, r = rects[i]
                m.points = rr_pts(x0, y0, x1, y1, r) + sh
                set_style(m, [*rgb, op])
            else:
                m.points = EMPTY.copy()
        # ---- finish line at the parallel part's right edge
        xf = max(r[2] for r in rects) + FIN_W * 0.005
        self.finish.points = seg(P(xf, FIN_Y0), P(xf, FIN_Y1)) + sh
        set_style(self.finish, None, [*C_INK, clear], FIN_W)

    def _rows(self, now):
        """Parallel-part rectangles [(x0, y0, x1, y1, r)] and the doubling glow (0..1)."""
        if now < T_DBL[0] - EPS:
            x0 = lerp(SER_X1, PAR_X0, ease_out_cubic(ramp(now, T_NO, T_NO + T_GAP)))
            return [np.array([x0, BAR_Y0, BAR_X1, BAR_Y1, SER_R])], 0.0
        if now < T_RAMP0 - EPS:
            k = dbl_index(now)
            n0 = 2 ** k
            p = ease_out_back(ramp(now, T_DBL[k], T_DBL[k] + DBL_DUR))
            gl = 1.0 - ease_out_cubic(ramp(now, T_DBL[k], T_DBL[k] + 0.28))
            out = []
            for c in range(2 * n0):
                out.append(lerp(row_rect(n0, c // 2), row_rect(2 * n0, c), p))
            for r in out:                       # overshoot safety (heights/widths stay > 0)
                r[3] = max(r[3], r[1] + 1e-3)
                r[2] = max(r[2], r[0] + 1e-3)
            return out, gl
        n = 10 ** log_n_ramp(now) if now < T_HIT - EPS else math.inf
        w = max(PAR_W / n, SLIVER_MIN)
        m = ease_in_out_sine(clip01(math.log2(n) - 4.0)) if n < math.inf else 1.0
        if m < 1.0:
            g = ROW_GAP * (1 - m)
            h = (SER_H - 15 * g) / 16
            r = row_r(row_h(16), PAR_W / 16) * (1 - m)
            return [np.array([PAR_X0, BAR_Y1 - k * (h + g) - h, PAR_X0 + w, BAR_Y1 - k * (h + g), r])
                    for k in range(16)], 0.0
        return [np.array([PAR_X0, BAR_Y0, PAR_X0 + w, BAR_Y1, min(SER_R, 0.3 * w)])], 0.0

    # ---------------- graph: axes, wall, curve, head ----------------
    def _draw_graph(self, now, f, sh, clear) -> None:
        pa = ease_out_cubic(ramp(now, T_AXES0, T_AXES1))
        xa, ya = self.axes
        if pa <= 0 or clear <= 0:
            xa.points = EMPTY.copy()
            ya.points = EMPTY.copy()
        else:
            o = P(GX0, GY0) + sh
            xa.points = seg(o, o + np.array([(GX1 - GX0) * pa, 0, 0]))
            ya.points = seg(o, o + np.array([0, (GY1 - GY0) * pa, 0]))
            for m in (xa, ya):
                set_style(m, None, [*C_DIM, clear], 3.0)
        # wall
        pw = ease_out_cubic(ramp(now, T_WALL0, T_WALL1))
        if pw <= 0 or clear <= 0:
            self.wall.points = EMPTY.copy()
        else:
            xd = GX0 + (GX1 - GX0) * pw
            parts = []
            for k in range(N_DASH):
                a = GX0 + k * (WALL_DASH + WALL_GAP)
                if a >= xd - 1e-6:
                    break
                b = min(a + WALL_DASH, xd)
                parts.append(seg(P(a, WALL_Y), P(max(b, a + 1e-3), WALL_Y)))
            self.wall.points = np.concatenate(parts) + sh
            hb = heartbeat(now)
            set_style(self.wall, None, [*C_AM, (0.85 + 0.15 * hb) * clear], 3.0 + 1.0 * hb)
        # curve
        L = log_n_curve(now)
        if now < T_HEAD - EPS or clear <= 0:
            self.curve.points = EMPTY.copy()
            self.head.points = EMPTY.copy()
            return
        if L > 1e-4:
            n = int(np.clip(L / LOG_MAX * 199 + 2, 2, 200))
            u = np.linspace(0.0, L, n)
            s = amdahl(10 ** u)
            pts = np.stack([gx(u), gy(s), np.zeros(n)], axis=1) + sh
            self.curve.points = polyline(pts)
            K = 32
            uk = np.linspace(0.0, L, K)
            rgb = curve_rgb(amdahl(10 ** uk))
            set_style(self.curve, None, np.column_stack([rgb, np.full(K, clear)]), 5.0)
        else:
            self.curve.points = EMPTY.copy()
        hs = ease_out_back(ramp(now, T_HEAD, T_HEAD + 0.20))
        hr = (HEAD_R + 0.03 * heartbeat(now)) * hs
        if hr <= 1e-4:
            self.head.points = EMPTY.copy()
        else:
            c = P(float(gx(L)), float(gy(amdahl(10 ** L)))) + sh
            self.head.points = _CIRCLE * hr + c
            set_style(self.head, [*C_INK, clear])

    # ---------------- bar labels + wall label ----------------
    def _draw_labels(self, now, f, sh, clear) -> None:
        if now < T_LABELS - EPS:
            self.s_lab_a.clear_slot()
            self.s_lab_b.clear_slot()
        else:
            dim = 1.0 - 0.45 * ramp(now, T_DIM0, T_DIM1) if now < T_HIT - EPS else 1.0
            a = dim * clear
            sc = pop_scale(now - T_LABELS)
            sh_a, tint_a = sh, None
            for t in T_CRACKS:                     # '10% can't split' flinches with the block
                df = f - story_frame(t)
                if 0 <= df <= 2:
                    tint_a = INK
                elif 3 <= df <= 6:
                    sh_a = sh + np.array([CRACK_SHAKE[df - 3], 0.0, 0.0])
            # POP about the inner edges so the 0.37 gap between the labels never closes
            self.s_lab_a.display(self.t_lab_a, a, sc, self.t_lab_a.get_right(), sh_a, tint_a)
            self.s_lab_b.display(self.t_lab_b, a, sc, self.t_lab_b.get_left(), sh)
        if now < T_WALL1 - EPS:
            self.s_wall.clear_slot()
        else:
            ab = np.array([self.t_wall.get_right()[0], self.t_wall.get_center()[1], 0.0])
            self.s_wall.display(self.t_wall, clear, slam_scale(now - T_WALL1, s0=1.2), ab, sh)

    # ---------------- N readout ----------------
    def _ntmpl(self, s: str) -> Text:
        t = self._ncache.get(s)
        if t is None:
            t = mono(s, 44, CYAN, BOLD)
            b, top = cap_box(t)
            if s.endswith("∞"):                   # mono '∞' is x-height small: give it digit size
                g = t.family_members_with_points()[-1]
                g.scale(1.55).move_to([g.get_center()[0] + 0.04, (b + top) / 2, 0])
            t.shift([X_C - t.get_center()[0], N_XY[1] - (b + top) / 2, 0])
            assert t.width <= 5.45, s
            assert_safe(t)
            t._glyphs = t.family_members_with_points()
            self._ncache[s] = t
        return t

    def _draw_readout(self, now, f, sh, clear) -> None:
        n = n_display(now)
        if n is None or clear <= 0:
            for sl in (self.s_n, self.s_nc, self.s_nm):
                sl.clear_slot()
            return
        tm = self._ntmpl(n_str(n))
        ct = tm.get_center()
        # pops on every change of N (before the ramp), push-in with the bar
        pop = 1.0
        for t in (T_N_IN, *T_DBL):
            pop *= pop_scale(now - t)
        pop *= slam_scale(now - T_HIT, s0=1.15)
        k = 1.0 + 0.02 * ease_in_out_sine(ramp(now, T_PUSH0, T_PUSH1)) if now < T_NO - EPS else 1.0
        shift = (ct - BAR_C) * (k - 1) + sh
        # glitch-snap to '∞' at 5.90: decaying x-jitter + CYAN/MAGENTA split copies
        df = f - story_frame(T_HIT)
        if 0 <= df <= 6:
            shift = shift + np.array([jitter(df, 0.07 * (1 - df / 7), seed=59), 0.0, 0.0])
        self.s_n.display(tm, clear, pop * k, ct, shift)
        if 0 <= df <= 4:
            self.s_nc.display(tm, 0.55 * clear, pop * k, ct, shift + np.array([-0.06, 0, 0]), CYAN)
            self.s_nm.display(tm, 0.55 * clear, pop * k, ct, shift + np.array([0.06, 0, 0]), MAGENTA)
        else:
            self.s_nc.clear_slot()
            self.s_nm.clear_slot()

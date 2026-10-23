"""b08 - THE CATCH: Amdahl's law (8.0 s, film 36.0-44.0).

Frame 0 is E07 (the '350,000+ / VOLUNTEER GPUs · 2020' HUD). The HUD cracks out (glitch)
and 'So… infinite speed?' POPs in to pay off the hook's open loop. One big CYAN job bar,
captioned 'TIME TO FINISH', wipes in at frame centre and shimmers, and the camera leans in.
At 1.50 'No.' hard-cuts in: the bar splits, its left 10% turns into the AMBER serial block,
and the HUD comes back as a SPEEDUP meter. The serial block tries to split twice (an INK
crack, the halves fly apart, then snap back with a thunk). At 2.45 the bar group whips up
to the top and the speedup graph ('SPEEDUP' over 'GPUs →') draws in under it. Four doublings
of the GPU count halve the parallel rows (1.8×, 3.0×, 4.7×, 6.4×) while the Amdahl curve
grows on a log axis; then the GPU count races to 1,000,000, the speedup turns AMBER and
stalls at 9.9×, and the curve flattens onto THE WALL. 5.90: 'GPUs: ∞', the HUD SLAMs '10×'
with its assumption ('MAX SPEEDUP · 10% SERIAL'), and an AMBER burst + shock ring fire from
the curve head at the wall. Then a heartbeat: each LUB swells the serial block (1.25x, AMBER
ring), a light pulse runs up the curve, and the DUB lands on the wall (head flare + ring, wall
flash, the meter pulses). On the fourth beat the block grows into the amber token (E08) that
b09 opens back into the lone hook worker.

Everything that moves is a pure function of self.now (one driver), so 30-fps previews and
the 60-fps final show the same story frames.
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# World layout
# --------------------------------------------------------------------------
# The bar group has two poses: SETUP (0.30-2.45: big, at frame centre, so the question,
# 'No.' and the failed splits play on a balanced frame) and GRAPH (from 2.70: compact at the
# top, the speedup graph below). It whips between them over 2.45-2.70 (ease_in_out_quart: expo
# read as a jump cut at 30 fps).
BAR_X0, BAR_X1 = -3.70, 3.00                   # job bar
BAR_CX = (BAR_X0 + BAR_X1) / 2
BAR_R = 0.08
BAR_YC0, BAR_H0 = 0.45, 1.40                   # SETUP pose: centre y, height
BAR_YC1, BAR_H1 = 2.22, 0.80                   # GRAPH pose -> bar y[1.82, 2.62]
BAR_Y0F, BAR_Y1F = BAR_YC1 - BAR_H1 / 2, BAR_YC1 + BAR_H1 / 2
SER_X1 = -3.03                                 # serial block x[-3.70, -3.03]
SER_CX = (BAR_X0 + SER_X1) / 2
SER_W = SER_X1 - BAR_X0
SER_R = 0.06
PAR_X0 = -2.97                                 # parallel part left edge (after the 0.06 gap)
PAR_W = BAR_X1 - PAR_X0                        # 5.97
ROW_GAP = 0.03
SLIVER_MIN = 0.03                              # storyboard 0.012 (1.4 px): too thin to read on a phone
FIN_W = 3.0                                    # finish-line stroke; its left edge hugs the rows
FIN_EXT = 0.10                                 # finish line overhang above / below the bar
ROW_ABOVE = 0.24                               # bar top -> caption / N readout baseline
LAB_BELOW = 0.20                               # bar bottom -> bar-label cap top

# Graph: S from 1x (one GPU, the axes' corner) to the 10x wall, N on a log axis 1 .. 1e6.
GX0, GX1 = -3.30, 3.00
GY0 = -2.24                                    # x-axis ('10×' slammed 1.3x tops out at -2.37)
WALL_Y = 0.20                                  # S = 10
GY1 = WALL_Y + 0.32                            # y-axis top
Y_PER_S = (WALL_Y - GY0) / 9.0                 # 0.271 per 1x
X_PER_DEC = (GX1 - GX0) / 6.0                  # 1.05 per decade: N = 1e6 lands on x 3.00
LOG_MAX = 6.0
AX_W = 4.0
CURVE_W = 7.0
AREA_N = 40                                    # glow under the curve: vertical-fade colour strips ...
AREA_OP = (0.0, 0.03, 0.32)                    # ... alpha at the axis / halfway / under the curve (a slow
                                               # fade turned amber-over-navy into a mauve haze)
HEAD_R = 0.10


def gx(logn):
    return GX0 + X_PER_DEC * np.asarray(logn, float)


def gy(s):
    return GY0 + Y_PER_S * (np.asarray(s, float) - 1.0)


WALL_DASH = 0.16
N_DASH = 21
WALL_GAP = (GX1 - GX0 - N_DASH * WALL_DASH) / (N_DASH - 1)
WALL_PER = WALL_DASH + WALL_GAP
WALL_CRAWL = 0.45                              # dash crawl speed (units / s) once the wall is up
WALL_SW, WALL_SW_HB = 4.0, 4.0                 # stroke 4 -> 8 on each heartbeat dub
WALL_LAB_GAP = 0.16                            # wall -> wall-label cap bottom
HEAD_PUSH = 0.06                               # the head strains right (+x) on each dub

C_CY, C_AM, C_INK, C_DIM = RGB_CYAN, RGB_AMBER, RGB_INK, RGB_DIM

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
T_GLITCH1 = 0.30                               # HUD cracks out 0.00-0.30
T_Q_IN = 0.05                                  # 'So… infinite / speed?' POP (story frame 3)
T_WIPE0, T_WIPE1 = 0.30, 0.75                  # bar wipe (ease_out_cubic)
T_CAP_IN = 0.35                                # 'TIME TO FINISH'
T_N_IN = 0.45                                  # 'GPUs: 1' POP
T_SHIM = 0.75                                  # shimmer
T_PUSH0, T_PUSH1 = 1.00, 1.50                  # lean-in 1.00 -> 1.04
PUSH_K = 0.04
T_NO = 1.50                                    # 'No.' HIT: split, HUD back
T_GAP = 0.12
T_LABELS = 1.60
T_CRACKS = (1.95, 2.35)
T_SLIDE0, T_SLIDE1 = 2.45, 2.70                # SETUP -> GRAPH pose
T_NO_GONE = 2.60
T_AXES0, T_AXES1 = 2.60, 3.00
T_AXLAB = 2.70                                 # 'SPEEDUP' / 'GPUs →'
T_HEAD = 2.60
T_DBL = (2.70, 3.10, 3.50, 3.90)               # N 1 -> 2 -> 4 -> 8 -> 16
DBL_DUR = 0.30
T_SOME_IN, T_SOME_GONE = 2.60, 4.45
T_RAMP0, T_RAMP1 = 4.30, 5.90
T_DIM0, T_DIM1 = 4.45, 4.60
T_CAP_OUT0, T_CAP_OUT1 = 4.30, 4.45            # caption steps aside while N spins wide ...
T_CAP_BACK = 5.95                              # ... and returns beside 'GPUs: ∞'
T_WALL0, T_WALL1 = 4.90, 5.15
T_HIT = 5.90                                   # 'GPUs: ∞', '10×'
T_TAKE = 5.95                                  # '10% serial / = max 10×.'
T_COL, T_SET, T_EXIT = cut_times(8.0)          # 7.433 / 7.833 / 7.95: block -> E08 token
# Heartbeat (5.90 -> T_COL): LUB on the serial block (scale 1.25 + AMBER ring), a light pulse runs
# up the curve into the wall, DUB at the wall (head flare + ring, wall flash, the meter pulses).
# The serial part is what caps the speedup, and the picture says so twice a second.
HB_PER = (T_COL - T_HIT) / 3                   # 0.511 s: lubs at 5.90 (the hit), 6.41, 6.92; the 4th is the morph
T_LUBS = tuple(T_HIT + k * HB_PER for k in range(3))
HB_AMP = 0.25                                  # block scale 1.0 -> 1.25 per lub
LUB_DUR = 0.22
HB_TRAVEL = 0.24                               # pulse run time, block beat -> wall
DUB_DUR = 0.24
T_DUBS = tuple(t + HB_TRAVEL for t in T_LUBS[1:])   # 6.65, 7.16 (over by 7.40 < T_COL)
T_CLEAR0, T_CLEAR1 = T_COL, T_COL + 0.25
MORPH_FILL0 = 0.95                             # the block's fill, held high while it flies to AC
T_TEXT_OUT = 7.85

LOG16 = math.log10(16)

# serial-block crack (failed split): story frames from the crack time
CRACK_DX = np.array([0.0, 0.034, -0.026, 0.036, -0.030, 0.022, 0.0])
CRACK_SEP = 0.10                               # each half flies 0.10 out ...
CRACK_SHEAR = 0.04                             # ... and shears 0.04 (left up, right down)
CRACK_HOLD = 6                                 # frames 0-5 split (0-1 INK flash), 6 snaps back
CRACK_SHAKE = (0.10, -0.09, 0.07, -0.05, 0.03, -0.015)   # x offsets on the snap-back frames 6-11
CRACK_ALPHA = (0.85, 0.65, 0.45, 0.30, 0.18, 0.08)       # residual crack line on frames 6-11
THUNK = (1.15, 0.15)                           # snap-back scale 1.15 -> 1.0 (ease_out_back), s

_CIRCLE = Circle(radius=1.0).points.copy()


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def P(x, y):
    return np.array([x, y, 0.0])


def corner(a, b, c) -> np.ndarray:
    """Quarter-arc cubic from a to c with the (sharp) corner at b."""
    a, b, c = (np.asarray(p, float) for p in (a, b, c))
    k = 0.5522847498307936
    return np.array([a, a + k * (b - a), c + k * (b - c), c])


def rr_box(x0, y0, x1, y1, r) -> np.ndarray:
    """Rounded rectangle by its corners (manim's RoundedRectangle point layout, r clamped)."""
    if x1 - x0 < 1e-5 or y1 - y0 < 1e-5:
        return EMPTY.copy()
    return rrect_c(((x0 + x1) / 2, (y0 + y1) / 2), x1 - x0, y1 - y0, r)


def bar_pose(now: float) -> tuple[float, float]:
    """(y0, y1) of the job bar / serial block / rows: SETUP pose -> GRAPH pose over 2.45-2.70."""
    e = ease_in_out_quart(ramp(now, T_SLIDE0, T_SLIDE1))
    yc, h = lerp(BAR_YC0, BAR_YC1, e), lerp(BAR_H0, BAR_H1, e)
    return yc - h / 2, yc + h / 2


def push_k(now: float) -> float:
    """Lean-in camera scale (about the bar centre) before 'No.'."""
    if now >= T_NO - EPS:
        return 1.0
    return 1.0 + PUSH_K * ease_in_out_sine(ramp(now, T_PUSH0, T_PUSH1))


def crack_pts(y0, y1) -> np.ndarray:
    ys = np.linspace(y1, y0, len(CRACK_DX))
    return polyline([P(SER_CX + dx, y) for dx, y in zip(CRACK_DX, ys)])


def half_pts(side: int, y0: float, y1: float) -> np.ndarray:
    """Left (-1) / right (+1) half of the serial block, split along the zigzag crack."""
    x0, x1, r = BAR_X0, SER_X1, SER_R
    ys = np.linspace(y1, y0, len(CRACK_DX))
    zz = [P(SER_CX + dx, y) for dx, y in zip(CRACK_DX, ys)]       # top -> bottom
    out = []
    if side < 0:
        out += [seg(zz[0], P(x0 + r, y1)), corner(P(x0 + r, y1), P(x0, y1), P(x0, y1 - r)),
                seg(P(x0, y1 - r), P(x0, y0 + r)), corner(P(x0, y0 + r), P(x0, y0), P(x0 + r, y0)),
                seg(P(x0 + r, y0), zz[-1])]
        out += [seg(a, b) for a, b in zip(zz[::-1][:-1], zz[::-1][1:])]
    else:
        out += [seg(a, b) for a, b in zip(zz[:-1], zz[1:])]
        out += [seg(zz[-1], P(x1 - r, y0)), corner(P(x1 - r, y0), P(x1, y0), P(x1, y0 + r)),
                seg(P(x1, y0 + r), P(x1, y1 - r)), corner(P(x1, y1 - r), P(x1, y1), P(x1 - r, y1)),
                seg(P(x1 - r, y1), zz[0])]
    return np.concatenate(out)


def crack_state(f: int):
    """(phase, df) of the failed-split animation at story frame f: ('split', 0..5),
    ('snap', 0..5) or (None, None)."""
    for t in T_CRACKS:
        df = f - story_frame(t)
        if 0 <= df < CRACK_HOLD:
            return "split", df
        if CRACK_HOLD <= df < CRACK_HOLD + len(CRACK_SHAKE):
            return "snap", df - CRACK_HOLD
    return None, None


def thunk_scale(now: float) -> float:
    s = 1.0
    for t in T_CRACKS:
        s *= slam_scale(now - (t + CRACK_HOLD / 60), *THUNK)
    return s


def log_n_ramp(now: float) -> float:
    """log10 N during the ramp: 16 -> 1,000,000 (ease_in_out_cubic, 4.30-5.90)."""
    return LOG16 + (LOG_MAX - LOG16) * ease_in_out_cubic(ramp(now, T_RAMP0, T_RAMP1))


def n_ramp(now: float) -> int:
    """The integer GPU count of the ramp: the readout, the meter and the sliver all use it."""
    return int(round(10 ** log_n_ramp(now)))


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
        return n_ramp(now)
    return 2 ** (dbl_index(now) + 1)


def n_str(n) -> str:
    return "GPUs: ∞" if n == math.inf else f"GPUs: {n:,}"


def lub(now: float) -> float:
    """Serial-block LUB envelope (0..1, fast attack) on each beat from 5.90; 0 from T_COL."""
    return max(bump((now - t) / LUB_DUR) for t in T_LUBS)


def dub(now: float) -> float:
    """Wall DUB envelope (0..1) where each heartbeat pulse lands; 0 from 7.40."""
    return max(bump((now - t) / DUB_DUR) for t in T_DUBS)


def hb_pulse(now: float):
    """Where the heartbeat pulse is along the curve (0 = origin .. 1 = the head at the wall;
    ease_in_quad: it accelerates into the wall), or None."""
    for t in T_LUBS[1:]:
        p = (now - t) / HB_TRAVEL
        if -EPS <= p < 1.0:
            return ease_in_quad(max(p, 0.0))
    return None


def row_r(h: float, w: float) -> float:
    return min(SER_R, 0.3 * h, 0.3 * w)


def row_rect(n: int, k: int, y0: float, y1: float):
    """Row k (top = 0) of the n stacked parallel rows at rest: (x0, y0, x1, y1, r)."""
    h, w = (y1 - y0 - ROW_GAP * (n - 1)) / n, PAR_W / n
    top = y1 - k * (h + ROW_GAP)
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


def caption(text: str, opacity: float = 0.70) -> MarkupText:
    """hud_label-style caption: Inter Display BOLD 34 (Inter's tighter cut, so 'TIME TO FINISH'
    shares its row with 'GPUs: 16'), INK, tracked."""
    t = MarkupText(f'<span letter_spacing="150">{text}</span>', font=FONT_DISPLAY, font_size=34,
                   color=INK, weight=BOLD)
    t.set_fill(INK, opacity=opacity)
    return t


class B08Amdahl(BeatScene):
    def beat(self):
        # ---------------- entry: E07 (HUD only) ----------------
        st = enter(self)
        hud = self.hud = st.hud

        # ---------------- world (pooled VMobjects, drawn by the driver) ----------------
        self.axes = VGroup(vm(), vm())
        for m in self.axes:
            m.cap_style = CapStyleType.ROUND
        self.wall = vm()
        self.area = VGroup(*[vm() for _ in range(AREA_N)])
        for m in self.area:
            m.sheen_direction = np.array([0.0, 1.0, 0.0])      # gradient stop 0 at the bottom
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
        self.graph = VGroup(self.area, self.axes, self.wall, self.curve, self.head)
        self.barg = VGroup(self.bar, self.rows, self.finish, self.halves, self.crack, self.serial)

        # ---------------- stage text (TextSlots, drawn by the driver; built at the GRAPH pose) ----
        lab_a = Text("10% can't split", font=FONT_BODY, font_size=34, color=AMBER, weight=MEDIUM)
        lab_b = Text("90% can split", font=FONT_BODY, font_size=34, color=CYAN, weight=MEDIUM)
        for t, edge in ((lab_a, "left"), (lab_b, "right")):
            b, top = cap_box(t)
            dx = (BAR_X0 - t.get_left()[0]) if edge == "left" else (BAR_X1 - t.get_right()[0])
            t.shift([dx, BAR_Y0F - LAB_BELOW - top, 0])
            assert_safe(t)
        assert lab_b.get_left()[0] - lab_a.get_right()[0] > 0.3
        cap_t = caption("TIME TO FINISH")
        b, top = cap_box(cap_t)
        cap_t.shift([BAR_X0 - cap_t.get_left()[0], BAR_Y1F + ROW_ABOVE - b, 0])
        assert_safe(cap_t)
        self.row_base = BAR_Y1F + ROW_ABOVE
        wl = Text("THE WALL: 10×", font=FONT_DISPLAY, font_size=44, color=AMBER, weight=BOLD)
        b, top = cap_box(wl)
        wl.shift([GX1 - wl.get_right()[0], WALL_Y + WALL_LAB_GAP - b, 0])
        assert_safe(wl)
        assert lab_a.get_bottom()[1] - wl.get_top()[1] > 0.30, "wall label crowds the bar labels"
        ax_y = caption("SPEEDUP", 0.60).rotate(PI / 2)
        ax_y.move_to([GX0 - 0.12 - ax_y.width / 2, (GY0 + WALL_Y) / 2, 0])
        ax_x = caption("GPUs →", 0.60)
        b, top = cap_box(ax_x)
        ax_x.shift([GX1 - 0.04 - ax_x.get_right()[0], GY0 + 0.17 - b, 0])
        for t in (ax_y, ax_x):
            assert_safe(t)
        self.t_lab_a, self.t_lab_b, self.t_wall, self.t_cap = lab_a, lab_b, wl, cap_t
        self.t_ax_y, self.t_ax_x = ax_y, ax_x
        self.s_lab_a, self.s_lab_b, self.s_wall, self.s_cap = TextSlot(), TextSlot(), TextSlot(), TextSlot()
        self.s_ax_y, self.s_ax_x = TextSlot(), TextSlot()
        self.s_n, self.s_nc, self.s_nm = TextSlot(), TextSlot(), TextSlot()
        self._ncache: dict = {}
        self.stage_txt = VGroup(self.s_ax_y, self.s_ax_x, self.s_lab_a, self.s_lab_b, self.s_wall,
                                self.s_cap, self.s_nc, self.s_nm, self.s_n)
        for s in ("GPUs: 1", "GPUs: 16", "GPUs: ∞"):           # widest steady states vs the caption
            assert self._ntmpl(s).get_left()[0] - cap_t.get_right()[0] > 0.28, s

        # ---------------- cards + kicker (frame-exact schedule) ----------------
        tr = TextTrack(self)
        q = card([("So… infinite", "H84", INK), ("speed?", "H84", INK)])
        for ln in q:                               # POPs on story frame 3, under the HUD crack-out
            tr.on(ln, T_Q_IN, T_NO, enter="pop", exit="cut")
        no = card([("No.", "H150", AMBER)])[0]
        # The H150 'No.' sits ~0.1 under the kicker, so a centre SLAM (1.3x) and CARD_OUT's
        # 0.15 upward drift both pushed it into 'THE CATCH · AMDAHL'S LAW': SLAM about the top
        # edge (grows downward) and fade out in place.
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
        for ln in (*q, *some, *take):              # the stage never reaches into a card
            assert ln.get_bottom()[1] - self._ntmpl("GPUs: 1,000,000").get_top()[1] > 0.15

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
            s = amdahl(n_ramp(now))                # the same integer N the readout shows
            return speedup_str(s), (AMBER if s >= 9.5 else INK)
        hud.drive(T_RAMP0, meter)
        hud.slam(T_HIT, "10×", AMBER)
        hud.label(T_HIT, "MAX SPEEDUP · 10% SERIAL", AMBER, 0.85)
        for t in T_DUBS:                           # the meter beats as each pulse hits the wall
            hud.pulse(t, 1.10, 0.25)

        # ---------------- effects ----------------
        head_wall = P(GX1, WALL_Y)
        rings = [ShockRing(self, T_HIT, head_wall, 0.2, 5.0, 12.0, 0.9, 0.55, AMBER)]
        rings += [ShockRing(self, t, P(SER_CX, BAR_YC1), 0.5, 2.4, 8.0, 0.9, 0.38, AMBER)
                  for t in T_LUBS[1:]]                                  # lub: from the block
        rings += [ShockRing(self, t, head_wall, 0.15, 1.4, 6.0, 0.9, 0.30, AMBER)
                  for t in T_DUBS]                                      # dub: at the wall

        # ---------------- z-order: dust, rings, graph, bar, stage text, cards, HUD, flash -------
        add_behind(self, *rings, self.graph, self.barg, self.stage_txt)
        self.add(tr)
        self.remove(hud)
        self.add(hud)
        self.add(RadialFlash(self, T_HIT, head_wall, peak=0.40, color=AMBER))
        self.collapse = None
        self._draw(0.0)
        drive(self, self._draw)

        # ---------------- audio ----------------
        self.cue("glitch", dt=0.0, gain=0.8, dur=0.30, note="the counter cracks out of the silence")
        self.cue("whoosh", dt=0.52, dur=0.45, direction="up", note="bar wipe")
        self.cue("silence", dt=1.00, dur=0.50, note="duck the bed under 'So… infinite speed?'")
        self.cue("hit", dt=T_NO, gain=1.0, note="'No.'")
        self.cue("sub_drop", dt=T_NO, dur=1.6, note="dark")
        for t in T_CRACKS:
            self.cue("glitch", dt=t, gain=0.5, dur=0.10, note="failed split: the block cracks")
            self.cue("hit", dt=round(t + CRACK_HOLD / 60, 4), gain=0.3, note="snap-back thunk")
        self.cue("whoosh", dt=round((T_SLIDE0 + T_SLIDE1) / 2, 4), dur=0.40, gain=0.6, direction="up",
                 note="bar group whips up, graph draws in")
        for t, fr in zip(T_DBL, (2400, 3000, 3600, 4400)):
            self.cue("tick", dt=t, gain=0.9, freq=fr, pan=0, note="doubling")
        self.cue("ticks_burst", dt=T_RAMP0, dur=0.72, rate0=30, rate1=4,
                 note="speedup ticks slowing, silent at ~5.02 (the stall)")
        self.cue("riser", dt=T_RAMP0, dur=1.60, note="into the wall")
        self.cue("hit", dt=T_WALL1, gain=0.6, note="THE WALL label")
        self.cue("hit", dt=T_HIT, gain=1.0, note="'10×' and '∞'")
        self.cue("sub_drop", dt=T_HIT, dur=2.0)
        self.cue("pad_swell", dt=T_TAKE, dur=1.6, chord=1, note="dark chord under the takeaway")
        for t, d in zip(T_LUBS[1:], T_DUBS):
            self.cue("hit", dt=round(t, 4), gain=0.35, note="heartbeat lub: the block + ring")
            self.cue("hit", dt=round(d, 4), gain=0.22, note="heartbeat dub: the pulse hits the wall")
        self.cue("whoosh", dt=round((T_COL + T_SET) / 2, 4), dur=0.45, direction="down",
                 note="the serial block becomes the token")
        self.cue("tick", dt=round(T_SET, 4), gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_COL)
        self._draw(T_COL)                          # the block at rest (lub / dub exactly 0)
        self.collapse = COLLAPSE(self.serial, None, "plain", AMBER)
        self.add(self.collapse.glyph)
        hold_floor(self, T_EXIT)
        exit_to(self)                              # canonical E08, static 7.95-8.00

    # ======================================================================
    # Driver
    # ======================================================================
    def _draw(self, now: float) -> None:
        f = story_frame(now)
        sh = shake(now - T_NO, 0.06, 0.15, seed=81) + shake(now - T_HIT, 0.06, 0.20, seed=82)
        clear = 1.0 - smooth(ramp(now, T_CLEAR0, T_CLEAR1))
        y0, y1 = bar_pose(now)
        self._draw_bar(now, f, sh, clear, y0, y1)
        self._draw_graph(now, f, sh, clear)
        self._draw_labels(now, f, sh, clear, y0, y1)
        self._draw_readout(now, f, sh, clear, y1)

    # ---------------- job bar / serial block / parallel rows ----------------
    def _draw_bar(self, now, f, sh, clear, y0, y1) -> None:
        yc, h = (y0 + y1) / 2, y1 - y0
        if now < T_NO - EPS:
            for m in (self.serial, *self.halves, self.crack, self.finish, *self.rows):
                m.points = EMPTY.copy()
            w = (BAR_X1 - BAR_X0) * ease_out_cubic(ramp(now, T_WIPE0, T_WIPE1))
            if w < 1e-4:
                self.bar.points = EMPTY.copy()
                return
            k, c = push_k(now), P(BAR_CX, yc)
            x0, x1 = BAR_X0, BAR_X0 + w
            self.bar.points = (rr_box(x0, y0, x1, y1, BAR_R) - c) * k + c
            # fill gradient across x: wipe head (INK-lifted leading edge) + travelling shimmer
            K = 40
            x = x0 + np.linspace(0.0, 1.0, K) * w
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

        # ---- serial block (from T_COL the Collapse owns it)
        phase, df = crack_state(f)
        dx = CRACK_SHAKE[df] if phase == "snap" else 0.0
        off = sh + np.array([dx, 0.0, 0.0])
        if self.collapse is not None:
            # cubic, not the default expo: the block travels ~3.4 units to AC and expo teleported it
            self.collapse.render(now, T_COL, T_SET, T_EXIT, ease=ease_in_out_cubic)
            if now < T_SET - EPS:              # keep the amber punch in flight; land on E08's 0.32
                e = ease_in_out_cubic(ramp(now, T_COL, T_SET))
                fr = self.serial.fill_rgbas.copy()
                fr[:, 3] = lerp(MORPH_FILL0, TOKEN_FILL, e ** 2.5)
                self.serial.fill_rgbas = fr
            for m in (*self.halves, self.crack):
                m.points = EMPTY.copy()
        elif phase == "split":
            self.serial.points = EMPTY.copy()
            fl = 0.70 if df <= 1 else 0.20                       # 2-frame INK flash, then hot amber
            for m, sgn in ((self.halves[0], -1), (self.halves[1], 1)):
                m.points = half_pts(sgn, y0, y1) + off + np.array([sgn * CRACK_SEP, -sgn * CRACK_SHEAR, 0.0])
                set_style(m, [*lerp(C_AM, C_INK, fl), 0.95], [*C_INK, 1.0], 2.0)
            self.crack.points = crack_pts(y0, y1) + off
            set_style(self.crack, None, [*C_INK, 1.0], 5.0 if df <= 1 else 3.5)
        else:
            hb = min(1.0, lub(now) + 0.3 * dub(now))
            flash = max(decay(now - T_NO, 0.22), decay(now - T_HIT, 0.30))
            s_fill = [*lerp(C_AM, C_INK, max(0.65 * flash, 0.15 * hb)), MORPH_FILL0]
            sc = (1.0 + HB_AMP * hb) * thunk_scale(now)
            self.serial.points = rrect_c(P(SER_CX, yc) + off, SER_W * sc, h * sc, SER_R * sc)
            set_style(self.serial, s_fill, [*C_INK, 1.0], 2.0)
            for m in self.halves:
                m.points = EMPTY.copy()
            if phase == "snap":
                self.crack.points = crack_pts(y0, y1) + off
                set_style(self.crack, None, [*C_INK, CRACK_ALPHA[df]], 2.5)
            else:
                self.crack.points = EMPTY.copy()

        # ---- parallel rows
        rects, gl = self._rows(now, y0, y1)
        flick = 1.0
        if T_HIT - EPS <= now < T_COL - EPS:
            flick = (0.8 + 0.2 * jitter(f // 3, 1.0, seed=58)) / 0.8           # 0.6 .. 1.0 (/0.8)
        rgb = lerp(C_CY, C_INK, 0.45 * gl)
        op = min(1.0, (0.8 + 0.2 * gl) * flick) * clear
        for i, m in enumerate(self.rows):
            if i < len(rects):
                x0, ry0, x1, ry1, r = rects[i]
                m.points = rr_box(x0, ry0, x1, ry1, r) + sh
                set_style(m, [*rgb, op])
            else:
                m.points = EMPTY.copy()
        # ---- finish line at the parallel part's right edge
        xf = max(r[2] for r in rects) + FIN_W * 0.005
        self.finish.points = seg(P(xf, y0 - FIN_EXT), P(xf, y1 + FIN_EXT)) + sh
        set_style(self.finish, None, [*C_INK, clear], FIN_W)

    def _rows(self, now, y0, y1):
        """Parallel-part rectangles [(x0, y0, x1, y1, r)] and the doubling glow (0..1)."""
        if now < T_DBL[0] - EPS:
            x0 = lerp(SER_X1, PAR_X0, ease_out_cubic(ramp(now, T_NO, T_NO + T_GAP)))
            return [np.array([x0, y0, BAR_X1, y1, SER_R])], 0.0
        if now < T_RAMP0 - EPS:
            k = dbl_index(now)
            n0 = 2 ** k
            p = ease_out_back(ramp(now, T_DBL[k], T_DBL[k] + DBL_DUR))
            gl = 1.0 - ease_out_cubic(ramp(now, T_DBL[k], T_DBL[k] + 0.28))
            out = [lerp(row_rect(n0, c // 2, y0, y1), row_rect(2 * n0, c, y0, y1), p) for c in range(2 * n0)]
            for r in out:                       # overshoot safety (heights/widths stay > 0)
                r[3] = max(r[3], r[1] + 1e-3)
                r[2] = max(r[2], r[0] + 1e-3)
            return out, gl
        n = n_ramp(now) if now < T_HIT - EPS else math.inf
        w = max(PAR_W / n, SLIVER_MIN)
        m = ease_in_out_sine(clip01(math.log2(n) - 4.0)) if n < math.inf else 1.0
        if m < 1.0:
            g = ROW_GAP * (1 - m)
            h = (y1 - y0 - 15 * g) / 16
            r = row_r((y1 - y0 - ROW_GAP * 15) / 16, PAR_W / 16) * (1 - m)
            return [np.array([PAR_X0, y1 - k * (h + g) - h, PAR_X0 + w, y1 - k * (h + g), r])
                    for k in range(16)], 0.0
        return [np.array([PAR_X0, y0, PAR_X0 + w, y1, min(SER_R, 0.3 * w)])], 0.0

    # ---------------- graph: axes, area, wall, curve, head ----------------
    def _draw_graph(self, now, f, sh, clear) -> None:
        dd = dub(now)
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
                set_style(m, None, [*C_DIM, clear], AX_W)
        # wall: draws left -> right, then its dashes crawl right; pulses with the heartbeat
        pw = ease_out_cubic(ramp(now, T_WALL0, T_WALL1))
        if pw <= 0 or clear <= 0:
            self.wall.points = EMPTY.copy()
        else:
            xd = GX0 + (GX1 - GX0) * pw
            o = (WALL_CRAWL * max(0.0, now - T_WALL1)) % WALL_PER
            parts = []
            for k in range(-1, N_DASH + 1):
                a, b = GX0 + o + k * WALL_PER, GX0 + o + k * WALL_PER + WALL_DASH
                a, b = max(a, GX0), min(b, xd, GX1)
                if b - a > 1e-3:
                    parts.append(seg(P(a, WALL_Y), P(b, WALL_Y)))
            self.wall.points = np.concatenate(parts) + sh if parts else EMPTY.copy()
            if flash_frames(now, T_HIT, 2):
                set_style(self.wall, None, [*C_INK, clear], WALL_SW + WALL_SW_HB)
            else:
                set_style(self.wall, None, [*lerp(C_AM, C_INK, 0.35 * dd), (0.85 + 0.15 * dd) * clear],
                          WALL_SW + WALL_SW_HB * dd)
        # curve + the area under it
        L = log_n_curve(now)
        if now < T_HEAD - EPS or clear <= 0:
            for m in (self.curve, self.head, *self.area):
                m.points = EMPTY.copy()
            return
        if L > 1e-4:
            n = int(np.clip(L / LOG_MAX * 199 + 2, 2, 200))
            u = np.linspace(0.0, L, n)
            pts = np.stack([gx(u), gy(amdahl(10 ** u)), np.zeros(n)], axis=1) + sh
            self.curve.points = polyline(pts)
            K = 64
            uk = np.linspace(0.0, L, K)
            rgb = curve_rgb(amdahl(10 ** uk))
            pp = hb_pulse(now)
            glint = (lambda uu: np.zeros_like(uu)) if pp is None else (
                lambda uu: np.exp(-((uu - pp * L) / 0.45) ** 2) * min(1.0, 4.0 * pp + 0.25))
            rgb = lerp(rgb, C_INK[None, :], 0.85 * glint(uk)[:, None])
            set_style(self.curve, None, np.column_stack([rgb, np.full(K, clear)]), CURVE_W)
            # the glow under the curve: one strip per 1/40 of the curve, fading to 0 at the axis
            # (a flat fill read as an olive slab under the wall)
            ue = np.linspace(0.0, L, AREA_N + 1)
            us = ue[:-1, None] + (ue[1:] - ue[:-1])[:, None] * np.linspace(0.0, 1.0, 4)[None, :]
            tx, ty = gx(us) + sh[0], gy(amdahl(10 ** us)) + sh[1]
            srgb = curve_rgb(amdahl(10 ** ((ue[:-1] + ue[1:]) / 2)))
            ak = (1.0 + 0.8 * dd + 1.6 * glint((ue[:-1] + ue[1:]) / 2)) * clear
            yb = GY0 + sh[1]
            for i, m in enumerate(self.area):
                xs = np.concatenate([[tx[i, 0]], tx[i], [tx[i, -1], tx[i, 0]]])
                ys = np.concatenate([[yb], ty[i], [yb, yb]])
                m.points = polyline(np.column_stack([xs, ys, np.zeros(len(xs))]))
                m.fill_rgbas = np.array([[*srgb[i], min(1.0, a * ak[i])] for a in AREA_OP])
                m.stroke_rgbas = np.zeros((1, 4))
                m.stroke_width = 0
        else:
            self.curve.points = EMPTY.copy()
            for m in self.area:
                m.points = EMPTY.copy()
        hs = ease_out_back(ramp(now, T_HEAD, T_HEAD + 0.20))
        hr = (HEAD_R + 0.06 * dd) * hs
        if hr <= 1e-4:
            self.head.points = EMPTY.copy()
        else:
            c = P(float(gx(L)) + HEAD_PUSH * dd, float(gy(amdahl(10 ** L)))) + sh
            self.head.points = _CIRCLE * hr + c
            set_style(self.head, [*C_INK, clear])

    # ---------------- bar labels, caption, wall label, axis labels ----------------
    def _draw_labels(self, now, f, sh, clear, y0, y1) -> None:
        lab_dy = np.array([0.0, y0 - BAR_Y0F, 0.0])
        if now < T_LABELS - EPS:
            self.s_lab_a.clear_slot()
            self.s_lab_b.clear_slot()
        else:
            dim = 1.0 - 0.45 * ramp(now, T_DIM0, T_DIM1) if now < T_HIT - EPS else 1.0
            a = dim * clear
            sc = pop_scale(now - T_LABELS)
            sh_a, tint_a = sh + lab_dy, None
            phase, df = crack_state(f)             # '10% can't split' flinches with the block
            if phase == "split" and df <= 1:
                tint_a = INK
            elif phase == "snap":
                sh_a = sh_a + np.array([CRACK_SHAKE[df], 0.0, 0.0])
            # POP about the inner edges so the gap between the labels never closes
            self.s_lab_a.display(self.t_lab_a, a, sc, self.t_lab_a.get_right() + lab_dy, sh_a, tint_a)
            self.s_lab_b.display(self.t_lab_b, a, sc, self.t_lab_b.get_left() + lab_dy, sh + lab_dy)
        # 'TIME TO FINISH' (rides the bar; the lean-in camera too)
        ca = ease_out_cubic(ramp(now, T_CAP_IN, T_CAP_IN + 0.25))
        if now < T_HIT - EPS:
            ca *= 1.0 - smooth(ramp(now, T_CAP_OUT0, T_CAP_OUT1))
        else:
            ca = ease_out_cubic(ramp(now, T_CAP_BACK, T_CAP_BACK + 0.25))
        ca *= clear
        if ca <= 0:
            self.s_cap.clear_slot()
        else:
            k = push_k(now)
            row = self.t_cap.get_center() + np.array([0.0, y1 - BAR_Y1F, 0.0])
            c = P(BAR_CX, (y0 + y1) / 2)
            shift = np.array([0.0, y1 - BAR_Y1F - 0.12 * (1 - ca), 0.0]) + (row - c) * (k - 1) + sh
            self.s_cap.display(self.t_cap, ca, k, self.t_cap.get_center(), shift)
        # axis titles
        aa = ease_out_cubic(ramp(now, T_AXLAB, T_AXLAB + 0.25)) * clear
        for slot, t in ((self.s_ax_y, self.t_ax_y), (self.s_ax_x, self.t_ax_x)):
            if aa <= 0:
                slot.clear_slot()
            else:
                slot.display(t, aa, 1.0, None, sh)
        # THE WALL: 10×
        if now < T_WALL1 - EPS:
            self.s_wall.clear_slot()
        else:
            ab = np.array([self.t_wall.get_right()[0], self.t_wall.get_center()[1], 0.0])
            self.s_wall.display(self.t_wall, clear, slam_scale(now - T_WALL1, s0=1.2), ab, sh)

    # ---------------- N readout (right-aligned on the caption row) ----------------
    def _ntmpl(self, s: str) -> Text:
        t = self._ncache.get(s)
        if t is None:
            t = mono(s, 44, CYAN, BOLD)
            b, top = cap_box(t)
            if s.endswith("∞"):                   # mono '∞' is x-height small: give it digit size
                g = t.family_members_with_points()[-1]
                g.scale(1.55).move_to([g.get_center()[0] + 0.04, (b + top) / 2, 0])
            t.shift([BAR_X1 - t.get_right()[0], self.row_base - b, 0])
            assert t.width <= 5.45, s
            assert_safe(t)
            t._glyphs = t.family_members_with_points()
            t._anchor = np.array([BAR_X1, (b + top) / 2 + self.row_base - b, 0.0])
            self._ncache[s] = t
        return t

    def _draw_readout(self, now, f, sh, clear, y1) -> None:
        n = n_display(now)
        if n is None or clear <= 0:
            for sl in (self.s_n, self.s_nc, self.s_nm):
                sl.clear_slot()
            return
        tm = self._ntmpl(n_str(n))
        ab = tm._anchor                            # right edge, cap-band centre
        # pops on every change of N (before the ramp), the lean-in camera, rides the bar
        pop = 1.0
        for t in (T_N_IN, *T_DBL):
            pop *= pop_scale(now - t)
        pop *= slam_scale(now - T_HIT, s0=1.15)
        k = push_k(now)
        c = P(BAR_CX, BAR_YC0)
        shift = np.array([0.0, y1 - BAR_Y1F, 0.0]) + (ab + np.array([0.0, y1 - BAR_Y1F, 0.0]) - c) * (k - 1) + sh
        # glitch-snap to '∞' at 5.90: decaying x-jitter + CYAN/MAGENTA split copies
        df = f - story_frame(T_HIT)
        if 0 <= df <= 6:
            shift = shift + np.array([jitter(df, 0.07 * (1 - df / 7), seed=59), 0.0, 0.0])
        self.s_n.display(tm, clear, pop * k, ab, shift)
        if 0 <= df <= 4:
            self.s_nc.display(tm, 0.55 * clear, pop * k, ab, shift + np.array([-0.06, 0, 0]), CYAN)
            self.s_nm.display(tm, 0.55 * clear, pop * k, ab, shift + np.array([0.06, 0, 0]), MAGENTA)
        else:
            self.s_nc.clear_slot()
            self.s_nm.clear_slot()

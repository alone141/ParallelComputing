"""b08 - THE TWIST: FINAL BOSS, Amdahl's law (12.5 s, film 39.0-51.5; script v3 'twist', calm motion).

Frame 0 is E06: the cluster token + the HUD '1,689,600,000+' / CORES WORKING AT ONCE, exactly
as b06 leaves them (b06's riser crests on the seam). FADE (0.00-0.80, replaces v2's record
scratch): fade_away fades the token and the whole counter to black TOGETHER at a constant value
(ease_in_out_sine, opacity only), so the count is never seen dropping, and swell_cut() lays a soft
swell over the seam that decays with it. 'So… infinite / speed?' POPs on story frame 1 and holds
still. 0.80-1.40 one big CYAN job bar, captioned 'TIME TO FINISH', wipes in at frame centre and
'GPUs: 1' fades in; 1.40-1.85 a hush.
1.85: the kicker FINAL BOSS: AMDAHL'S LAW cuts in with the AMBER 'No.' SLAM and its sub-line
'Some work can't be split.'; the bar splits (its left 10% becomes the AMBER serial block). 1.95:
bar labels ('10% can't split' / '90% can split') and the same Hud fades back in as a separately
labelled SPEEDUP meter ('1.0×' / SPEEDUP; it only ever climbs). The block breathes while the card
is read; 3.05-3.65 ONE slow split attempt: a thin INK crack draws through the block, its halves
ease apart and back together (low thunk). 3.65-4.45 the bar group glides to the top (0.80 s) and
the speedup graph ('SPEEDUP' over 'GPUs →', log axis) draws in under it.
4.55: hard swap to 'Million GPUs? / Only the 90% speeds up.' on the empty graph: read 1.0 s, then
four doublings (5.55 / 5.90 / 6.25 / 6.60) take N to 2, 4, 8, 16 (1.8×, 3.0×, 4.7×, 6.4×) while
the CYAN 90% splits into ever-thinner rows and the AMBER block never shrinks. 6.75-7.30 the GPU
count climbs smoothly to 1,000,000: the speedup turns AMBER and settles at 9.9× (floor-truncated,
never 10.0×) as the curve flattens onto THE WALL (drawn 7.00-7.30, label fades in 7.30); still
from 7.45.
7.60: 'GPUs: ∞' cross-fades in with an AMBER burst + shock ring at the wall, the HUD SLAMs '10×' /
'MAX SPEEDUP · 10% SERIAL' and the takeaway POPs ('10% serial / = max 10×.'). The axis titles and
bar labels fade out (7.60-7.90) and THE WALL steps back to 0.60. Light flows up the curve into the
wall (7.75-8.55) while the stage leans in slowly, and a STEADY heartbeat on the bed's 0.5 s grid
(lubs 8.50 ... 11.00) swells the AMBER block with a ring on every lub while a pulse runs up the
curve to a DUB at the wall. 9.70, between lubs: hard swap to 'Forever. / ∞ GPUs won't change it.'.
The next lub (11.50) launches the block into the bright AMBER serial token (0.80 s, E08:
token_fill(AMBER) 0.58, #FFD27A rim), settle 12.30, static E08 from 12.45 (the kicker and the
card hard-cut there). b09 opens it back into the lone hook worker.

Everything that moves is a pure function of self.now (one driver), so 30-fps previews and
the 60-fps final show the same story frames.
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# World layout
# --------------------------------------------------------------------------
# The bar group has two poses: SETUP (0.80-3.65: big, at frame centre, so the question, 'No.'
# and the failed split play on a balanced frame) and GRAPH (from 4.45: compact at the top, the
# speedup graph below). It glides between them over 3.65-4.45 (MOVE_TIME on CALM_EASE).
BAR_X0, BAR_X1 = -3.70, 3.00                   # job bar
BAR_CX = (BAR_X0 + BAR_X1) / 2
BAR_R = 0.08
BAR_YC0, BAR_H0 = 0.45, 1.40                   # SETUP pose: centre y, height
BAR_YC1, BAR_H1 = 2.17, 0.80                   # GRAPH pose -> bar y[1.77, 2.57] (v3: 0.05 lower, so the
                                               # readout row clears the 'No.' card's sub-line)
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
ROW_ABOVE = 0.20                               # bar top -> caption / N readout baseline
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
# Timeline (beat-local seconds; story frames on the 60-fps grid). Script v3 'twist'
# (reel/design/script_v3.json), calm motion: card lands on a calm stage -> read -> the motion it
# describes -> payoff hold; the moves are smooth (fade 0.80, wipe 0.60, glide 0.80, launch 0.80 on
# ease_in_out_sine); flashes, rings and slams stay. Card windows are the reading floor (0.3 s/word
# + 0.8 s, counted from all lines at full opacity (POP / SLAM: at once) to the exit).
#   0.00-0.80   FADE: fade_away(E06) (token + 1,689,600,000+ HUD, together, constant value);
#               swell_cut() carries b06's riser over the seam and decays with it
#   0.017       'So… infinite / speed?' POPs and holds still                (-> 1.85 = 1.83 s >= 1.70)
#   0.80-1.40   bar wipe (0.60 s), caption 'TIME TO FINISH', 'GPUs: 1' fades in 1.10-1.40
#   1.40-1.85   hush
#   1.85        'No. / Some work can't be split.' SLAM + kicker + split  (-> 4.55 = 2.70 s >= 2.60)
#   1.95        bar labels + the SPEEDUP meter fades in ('1.0×', 1.95-2.25)
#   3.05-3.65   ONE slow split attempt: an INK crack draws, the halves ease apart and back (thunk)
#   3.65-4.45   the bar group glides to the top (0.80 s), the graph draws in under it
#   4.55        hard swap -> 'Million GPUs? / Only the 90% speeds up.'  (-> 7.60 = 3.05 s >= 2.90)
#   5.55-6.60   four doublings (read the card 1.0 s first): 1.8× 3.0× 4.7× 6.4×
#   6.75-7.30   the count climbs to 1,000,000: AMBER, 9.9×; the wall draws 7.00-7.30, its label
#               fades in 7.30-7.45; still from 7.45
#   7.60        HIT: 'GPUs: ∞' cross-fades in, '10×' + 'MAX SPEEDUP · 10% SERIAL' SLAM, the
#               takeaway '10% serial / = max 10×.' POPs                    (-> 9.70 = 2.10 s >= 2.00)
#   7.75-8.55   light flows up the curve into the wall
#   8.50 ...    STEADY heartbeat on the bed's 0.5 s grid: lubs 8.50 9.00 9.50 10.00 10.50 11.00
#   9.70        hard swap -> 'Forever. / ∞ GPUs won't change it.'        (-> 12.45 = 2.75 s >= 2.60)
#   11.50       the next lub launches the block into the E08 token (0.80 s); settle 12.30
#   12.45       kicker + card hard-cut (exit_to); static E08 12.45-12.50 (3 frames)
# --------------------------------------------------------------------------
BEAT_SLOT = 12.5                               # timeline.py slot (asserted in beat())


def F(n: int) -> float:
    """Story frame n -> beat-local seconds."""
    return n / 60


ZERO = np.zeros(3)                             # (no camera shake in v3: the stage offset is always 0)
T_FADE1 = FADE_AWAY_TIME                       # 0.80: the E06 token + counter have faded out
T_Q_IN = F(1)                                  # 'So… infinite / speed?' POPs on story frame 1
T_WIPE0, T_WIPE1 = F(48), F(84)                # bar wipe 0.80-1.40 (CALM_EASE)
T_CAP_IN = T_WIPE0 + 0.05                      # 'TIME TO FINISH'
T_N_IN, N_FADE = F(66), 0.30                   # 'GPUs: 1' fades in 1.10-1.40
T_SHIM = T_WIPE1                               # shimmer (bar, then the CYAN 90% until the 1st doubling)
SHIM_AMP = 0.15
T_SIL0 = T_WIPE1                               # 1.40-1.85 hush before the answer
T_NO = F(111)                                  # 1.85 'No.' SLAM: split, kicker
T_GAP = 0.12
T_LABELS = F(117)                              # 1.95 bar labels + the SPEEDUP meter
T_METER1 = T_LABELS + 0.30                     # the meter has faded in by 2.25
# ONE slow split attempt (the card is read 1.20 s first): the crack draws, the halves ease apart
# and back together, a low thunk (no glitch, no shake)
T_CRACK = F(183)                               # 3.05
CRACK_DRAW = 0.15                              # the INK crack draws top -> bottom 3.05-3.20
T_APART0, T_APART1 = T_CRACK + 0.10, T_CRACK + 0.35   # 3.15-3.40 the halves ease apart ...
T_REJOIN = T_CRACK + 0.60                      # ... and back together by 3.65 (thunk)
CRACK_FADE = 0.25                              # the crack line fades after the rejoin
T_SLIDE0 = T_REJOIN                            # 3.65 SETUP -> GRAPH pose ...
T_SLIDE1 = T_SLIDE0 + MOVE_TIME                # ... 4.45 (0.80 s glide, CALM_EASE)
T_AXES0, T_AXES1 = T_SLIDE0 + 0.30, T_SLIDE1
T_AXLAB = T_SLIDE0 + 0.50                      # 'SPEEDUP' / 'GPUs →'
T_HEAD = T_AXES0
T_ASK = F(273)                                 # 4.55 hard swap -> 'Million GPUs?' (on the empty graph)
T_DBL = (F(333), F(354), F(375), F(396))       # 5.55 / 5.90 / 6.25 / 6.60: N 1 -> 16 (read the card 1.0 s)
DBL_DUR = 0.30
T_RAMP0 = F(405)                               # 6.75 the count climbs (the 4th doubling is still settling)
T_RAMP1 = F(438)                               # 7.30 N = 1,000,000
T_DIM0, T_DIM1 = T_RAMP0 + 0.15, T_RAMP0 + 0.30
T_CAP_OUT0, T_CAP_OUT1 = T_RAMP0, T_RAMP0 + 0.30   # caption steps aside while N spins wide (and stays
                                               # out: the takeaway frame has no room for it)
T_WALL0, T_WALL1 = F(420), F(438)              # 7.00-7.30 the wall draws; 'THE WALL: 10×' fades in ...
WALL_LAB_FADE = 0.15                           # ... 7.30-7.45 (still from 7.45)
T_HIT = F(456)                                 # 7.60 'GPUs: ∞', '10×', the takeaway
T_TAKE = T_HIT                                 # '10% serial / = max 10×.' POPs on the hit
N_XFADE = 0.30                                 # 'GPUs: 1,000,000' -> 'GPUs: ∞' cross-fade
T_FOREVER = F(582)                             # 9.70 hard swap -> 'Forever.' (between lubs)
T_DECL0, T_DECL1 = T_TAKE, T_TAKE + 0.30       # declutter: axis titles + bar labels fade out
WALL_LAB_TAKE = 0.60                           # 'THE WALL: 10×' steps back under the takeaway (3rd '10×')
T_COL, T_SET, T_EXIT = cut_times(BEAT_SLOT)    # 11.50 / 12.30 / 12.45: block -> E08 token
# Heartbeat: LUB on the serial block (scale 1.25 + AMBER ring), a light pulse runs up the curve
# into the wall, DUB at the wall (head flare + ring, wall flash, the meter pulses). The serial part
# is what caps the speedup, and the picture says so with every beat. The hit swells the block
# once; then the heart beats STEADILY on the bed's 0.5 s grid (8.50 ... 11.00) and the next beat
# (T_COL = 11.50) launches the block into the token.
HB_PERIOD = 0.50
T_LUBS = (T_HIT,) + tuple(8.50 + HB_PERIOD * k for k in range(6))   # 7.60 | 8.50 ... 11.00
HB_INT = (HB_PERIOD,) * 6
HB_AMP = 0.25                                  # block scale 1.0 -> 1.25 per lub
LUB_DUR = 0.22
HB_TRAVELS = tuple(0.45 * i for i in HB_INT)   # pulse run time, block beat -> wall
T_DUBS = tuple(t + d for t, d in zip(T_LUBS[1:], HB_TRAVELS))
DUB_DURS = tuple(0.55 * i for i in HB_INT)     # each dub ends exactly on the next lub (the last on T_COL)
assert abs(T_LUBS[-1] + HB_PERIOD - T_COL) < 1e-9
T_CLEAR0, T_CLEAR1 = T_COL, T_COL + 0.40
MORPH_FILL0 = 0.95                             # the block's fill, held high while it flies to AC
# Never static after the hit: the stage (graph, bar, stage text; never the cards or the HUD) leans
# in 1.0 -> 1.05 about AC (where the token will land) over T_HIT -> T_CAM1 (ease_in_out_sine), and
# light flows up the curve into the wall (7.75-8.55).
CAM_K, T_CAM1 = 0.05, T_CLEAR1 + 0.30
T_FLOW0, T_FLOW1 = 7.75, 8.55
FLOW_N, FLOW_PER, FLOW_W, FLOW_A = 3, 1.35, 0.40, 0.38   # glints, s per run, width (log10 N), strength
# Ambient life while 'No. / Some work…' and 'Million GPUs?' are read (with the CYAN 90%'s travelling
# shimmer, see shimmer_amp): the serial block breathes (a slow INK lift of its fill) and, on the
# graph, the curve head breathes with it.
BREATH_PER, BREATH_INK = 0.80, 0.16
HEAD_BREATH = 0.30                             # head radius +30% at the top of a breath (before the doublings)

LOG16 = math.log10(16)

# serial-block crack (the failed split)
CRACK_DX = np.array([0.0, 0.034, -0.026, 0.036, -0.030, 0.022, 0.0])
CRACK_SEP = 0.10                               # each half eases 0.10 out ...
CRACK_SHEAR = 0.04                             # ... and shears 0.04 (left up, right down)

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
    """(y0, y1) of the job bar / serial block / rows: SETUP pose -> GRAPH pose over T_SLIDE0-T_SLIDE1."""
    e = CALM_EASE(ramp(now, T_SLIDE0, T_SLIDE1))
    yc, h = lerp(BAR_YC0, BAR_YC1, e), lerp(BAR_H0, BAR_H1, e)
    return yc - h / 2, yc + h / 2


def crack_pts(y0, y1, c: float = 1.0) -> np.ndarray:
    """The zigzag crack through the serial block, drawn top -> bottom up to fraction c."""
    ys = np.linspace(y1, y0, len(CRACK_DX))
    pts = [P(SER_CX + dx, y) for dx, y in zip(CRACK_DX, ys)]
    if c >= 1.0:
        return polyline(pts)
    u = c * (len(pts) - 1)
    k = int(u)
    out = pts[:k + 1] + ([lerp(pts[k], pts[k + 1], u - k)] if u - k > 1e-6 else [])
    return polyline(out) if len(out) >= 2 else EMPTY.copy()


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


def split_sep(now: float) -> float:
    """The failed split: how far apart the two halves are (0 -> 1 -> 0, CALM_EASE both ways)."""
    return CALM_EASE(ramp(now, T_APART0, T_APART1)) - CALM_EASE(ramp(now, T_APART1, T_REJOIN))


def crack_draw(now: float) -> float:
    """How much of the crack has drawn on (0 -> 1 over CRACK_DRAW from T_CRACK)."""
    return ease_out_cubic(ramp(now, T_CRACK, T_CRACK + CRACK_DRAW))


def crack_alpha(now: float) -> float:
    """Crack line opacity: 1 from T_CRACK, fading out after the rejoin."""
    if now < T_CRACK - EPS:
        return 0.0
    return 1.0 - ease_in_out_sine(ramp(now, T_REJOIN, T_REJOIN + CRACK_FADE))


def thunk_scale(now: float) -> float:
    """The low thunk as the halves meet again: a soft 1.05 -> 1.0 pop (no overshoot)."""
    return pop_scale(now - T_REJOIN, 1.05, 0.20)


def log_n_ramp(now: float) -> float:
    """log10 N during the climb: 16 -> 1,000,000 (CALM_EASE, T_RAMP0-T_RAMP1)."""
    return LOG16 + (LOG_MAX - LOG16) * CALM_EASE(ramp(now, T_RAMP0, T_RAMP1))


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


def log_n_dbl(now: float) -> float:
    """log10 N of the doublings (eased 0.30 s per step; LOG16 once the 4th has settled)."""
    k = dbl_index(now)
    if k < 0:
        return 0.0
    return math.log10(2) * (k + ease_out_cubic(ramp(now, T_DBL[k], T_DBL[k] + DBL_DUR)))


def dbl_settling(now: float) -> bool:
    """True while the 4th doubling (16 rows) is still easing in after the race has started."""
    return T_RAMP0 - EPS <= now < T_DBL[-1] + DBL_DUR - EPS


def log_n_curve(now: float) -> float:
    """log10 N the curve (and its head) has reached. The race starts while the 4th doubling is
    still settling: its progress is added on top (log_n_ramp is LOG16 at T_RAMP0), so the curve
    never jumps and equals log_n_ramp once the doubling has settled."""
    if now >= T_HIT - EPS:
        return LOG_MAX
    if now >= T_RAMP0 - EPS:
        return log_n_ramp(now) + (log_n_dbl(now) - LOG16 if dbl_settling(now) else 0.0)
    return log_n_dbl(now)


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


def breath(now: float) -> float:
    """Idle glow of the serial block (0..1) under 'No. / Some work can't be split.' and 'Million
    GPUs?': eases in after the slam, out over the climb's first 0.30 s (0 before T_BREATH and from
    T_RAMP0 + 0.30)."""
    env = smooth(ramp(now, T_BREATH, T_BREATH + 0.30)) * (1.0 - smooth(ramp(now, T_RAMP0, T_RAMP0 + 0.30)))
    if env <= 0.0:
        return 0.0
    return env * (0.5 - 0.5 * math.cos(TAU * (now - T_BREATH) / BREATH_PER))


T_BREATH = T_NO + 0.25                         # the block starts to breathe once 'No.' has landed


def head_breath(now: float) -> float:
    """The curve head breathes with the block on the empty graph ('A million / GPUs?' is read),
    eased out into the first doubling (0 from T_DBL[0])."""
    return breath(now) * (1.0 - smooth(ramp(now, T_DBL[0] - 0.20, T_DBL[0])))


def shimmer_amp(now: float) -> float:
    """Travelling shimmer on the CYAN bar from T_SHIM, kept on the 90% part after the split
    (the part that CAN split stays alive while 'No.', 'Some work' and 'A million' are read),
    faded out into the first doubling (the split rows are flat)."""
    return SHIM_AMP * ramp(now, T_SHIM, T_SHIM + 0.20) * (1.0 - smooth(ramp(now, T_DBL[0] - 0.20, T_DBL[0])))


def lub(now: float) -> float:
    """Serial-block LUB envelope (0..1, fast attack) on each beat from T_HIT; 0 from T_COL."""
    return max(bump((now - t) / LUB_DUR) for t in T_LUBS)


def dub(now: float) -> float:
    """Wall DUB envelope (0..1) where each heartbeat pulse lands; 0 from T_COL."""
    return max(bump((now - t) / d) for t, d in zip(T_DUBS, DUB_DURS))


def hb_pulse(now: float):
    """Where the heartbeat pulse is along the curve (0 = origin .. 1 = the head at the wall;
    ease_in_quad: it accelerates into the wall), or None."""
    for t, d in zip(T_LUBS[1:], HB_TRAVELS):
        p = (now - t) / d
        if -EPS <= p < 1.0:
            return ease_in_quad(max(p, 0.0))
    return None


def stage_cam(now: float) -> tuple[float, np.ndarray]:
    """(k, b) of the post-hit stage lean-in p -> k*p + b (identity before T_HIT)."""
    e = ease_in_out_sine(ramp(now, T_HIT, T_CAM1))
    k = 1.0 + CAM_K * e
    return k, AC * (1.0 - k)


def flow(now: float, u, L: float):
    """Light flowing up the curve into the wall after the hit (T_FLOW0 -> T_FLOW1): FLOW_N soft
    glints at constant speed in log10 N (fast up the slope, crawling along the wall), tapered to 0
    at both ends so the wrap-around never pops. u: positions (log10 N) -> 0..FLOW_A."""
    a = ease_out_cubic(ramp(now, T_FLOW0, T_FLOW0 + 0.25)) * (1.0 - ease_in_out_sine(ramp(now, T_FLOW1 - 0.25, T_FLOW1)))
    u = np.asarray(u, float)
    if a <= 0 or L <= 0:
        return np.zeros_like(u)
    out = np.zeros_like(u)
    for j in range(FLOW_N):
        q = ((now - T_FLOW0) / FLOW_PER + j / FLOW_N) % 1.0
        out = np.maximum(out, math.sin(math.pi * q) * np.exp(-((u - q * L) / FLOW_W) ** 2))
    return FLOW_A * a * out


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
        # ---------------- entry: E06 (cluster token + HUD 1,689,600,000+ / CORES WORKING AT ONCE) ----
        st = enter(self)
        hud = self.hud = st.hud
        # FADE (script v3, replaces the record-scratch tear): the token and the counter fade to black
        # TOGETHER at a constant value (0.00-0.80, ease_in_out_sine, opacity only), so the count is
        # never seen dropping. Frame 0 stays the exact E06; the token's FadeAway takes its place in
        # the scene. From 0.80 the Hud draws nothing until the SPEEDUP meter fades in at 1.95.
        self.fade = fade_away(self, st, 0.0, FADE_AWAY_TIME)

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
        self.rows[0].sheen_direction = np.array([1.0, 0.0, 0.0])   # the 90% part keeps the bar's shimmer
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
        self.s_n, self.s_nc = TextSlot(), TextSlot()      # the readout (+ the old value in the ∞ cross-fade)
        self._ncache: dict = {}
        self.stage_txt = VGroup(self.s_ax_y, self.s_ax_x, self.s_lab_a, self.s_lab_b, self.s_wall,
                                self.s_cap, self.s_nc, self.s_n)
        for s in ("GPUs: 1", "GPUs: 16", "GPUs: ∞"):           # widest steady states vs the caption
            assert self._ntmpl(s).get_left()[0] - cap_t.get_right()[0] > 0.28, s

        # ---------------- cards + kicker (frame-exact schedule; every card hard-swaps) ----------
        assert beat_duration(type(self).__name__) == BEAT_SLOT, "timeline.py slot changed: re-time b08"
        tr = TextTrack(self)
        # the opening question POPs on story frame 1 and holds still (no jitter); no kicker yet
        q = self.t_q = card([("So… infinite", "H84", INK), ("speed?", "H84", INK)])
        tr.on(q, T_Q_IN, T_NO, enter="pop", exit="cut")
        no_c = card([("No.", "H150", AMBER), ("Some work can't be split.", "S56", INK, {"can't": AMBER})],
                    leading=0.72)
        no = no_c[0]
        # The H150 'No.' sits ~0.1 under the kicker, so a centre SLAM (1.3x) would push it into
        # the kicker: SLAM about the top edge (grows downward); its sub-line POPs with it.
        tr.on(no, T_NO, T_ASK, enter="slam", exit="cut", about=no.get_top())
        tr.on(no_c[1], T_NO, T_ASK, enter="pop", exit="cut")
        ask = card([("Million GPUs?", "H96", INK, {"GPUs?": CYAN}),
                    ("Only the 90% speeds up.", "S56", INK, {"90%": CYAN})])
        take = card([("10% serial", "H84", INK, {"10% serial": AMBER}),
                     ("= max 10×.", "H84", INK, {"10×": AMBER})])
        forever = card([("Forever.", "H96", AMBER), ("∞ GPUs won't change it.", "S56", INK)])
        # (whole cards POP on their swap, so each reading floor starts at once)
        for c, t0, t1 in ((ask, T_ASK, T_TAKE), (take, T_TAKE, T_FOREVER), (forever, T_FOREVER, T_EXIT)):
            for ln in c:
                tr.on(ln, t0, t1, enter="pop", exit="cut")
        # The kicker cuts in on 'No.' and hard-cuts with the card at T_EXIT (exit_to swaps in E08).
        tr.on(kick("FINAL BOSS: AMDAHL'S LAW", AMBER, tracking=80), T_NO, T_EXIT, enter="cut", exit="cut")
        top_n = self._ntmpl("GPUs: 1,000,000").get_top()[1]
        for ln in (*q, *no_c, *ask, *take, *forever):     # the stage never reaches into a card
            assert ln.get_bottom()[1] - top_n > 0.15, ln.original_text
        # reading floors (0.3 s / word + 0.8 s; a bare '=' is not a word)
        for c, t0, t1 in ((q, T_Q_IN, T_NO), (no_c, T_NO, T_ASK), (ask, T_ASK, T_TAKE),
                          (take, T_TAKE, T_FOREVER), (forever, T_FOREVER, T_EXIT)):
            words = sum(len([w for w in ln.original_text.split() if w not in ("=",)]) for ln in c)
            assert t1 - t0 >= 0.3 * words + 0.8 - 1e-6, (c[0].original_text, t1 - t0, 0.3 * words + 0.8)

        # ---------------- HUD: the same Hud comes back as the SPEEDUP meter (after the fade) ----
        hud.cut(T_LABELS, "1.0×", INK)
        hud.label(T_LABELS, "SPEEDUP", INK, 0.70)
        hud.fade_to(T_LABELS, T_METER1, 0.0, 1.0, ease=CALM_EASE)   # a new meter: fades in, then climbs
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
        hud.pulse(T_COL, 1.06, 0.25)               # ... and once more on the launch beat
        assert hud.number_at(T_FADE1 - 0.02)[0] == "1,689,600,000+" and hud._env_at(T_FADE1)[0] == 0.0
        assert hud._env_at(T_LABELS - 0.02)[0] == 0.0 and hud.number_at(T_LABELS)[0] == "1.0×"
        assert hud._env_at(T_METER1)[0] == 1.0

        # ---------------- effects (centres follow the stage lean-in at each ring's start) -------
        def cam_pt(t, p):
            k, b = stage_cam(t)
            return p * k + b
        head_wall = P(GX1, WALL_Y)
        blk = P(SER_CX, BAR_YC1)
        rings = [ShockRing(self, T_HIT, head_wall, 0.2, 5.0, 12.0, 0.9, 0.55, AMBER)]
        rings += [ShockRing(self, t, cam_pt(t, blk), 0.5, 2.4, 8.0, 0.9, 0.38, AMBER)
                  for t in (*T_LUBS[1:], T_COL)]                        # lub: from the block (last = launch)
        rings += [ShockRing(self, t, cam_pt(t, head_wall), 0.15, 1.4, 6.0, 0.9, 0.30, AMBER)
                  for t in T_DUBS]                                      # dub: at the wall

        # ---------------- z-order: dust, rings, graph, bar, stage text, fading token, cards, HUD, flash
        add_behind(self, *rings, self.graph, self.barg, self.stage_txt)
        self.add(tr)
        self.remove(hud)
        self.add(hud)
        self.add(RadialFlash(self, T_HIT, head_wall, peak=0.40, color=AMBER))
        self.collapse = None
        self._draw(0.0)
        drive(self, self._draw)

        # ---------------- audio ----------------
        swell_cut(self)                            # soft swell over the seam, decays with the fade
        self.cue("whoosh", dt=round((T_WIPE0 + T_WIPE1) / 2, 4), dur=round(T_WIPE1 - T_WIPE0, 4), gain=0.6,
                 direction="up", note="bar wipe")
        self.cue("silence", dt=round(T_SIL0, 4), dur=round(T_NO - T_SIL0, 4),
                 note="hush: the pause before the answer")
        self.cue("hit", dt=round(T_NO, 4), gain=1.0, note="'No.' (+ kicker, split)")
        self.cue("sub_drop", dt=round(T_NO, 4), dur=1.6, note="dark")
        self.cue("tick", dt=round(T_CRACK, 4), gain=0.35, freq=5200, pan=0, note="failed split: a crack draws")
        self.cue("hit", dt=round(T_REJOIN, 4), gain=0.3, note="the halves meet again: low thunk")
        self.cue("whoosh", dt=round((T_SLIDE0 + T_SLIDE1) / 2, 4), dur=MOVE_TIME, gain=0.5, direction="up",
                 note="bar group glides up, graph draws in")
        self.cue("tick", dt=round(T_ASK, 4), gain=0.6, note="'Million GPUs?'")
        for t, fr in zip(T_DBL, (2400, 3000, 3600, 4400)):
            self.cue("tick", dt=round(t, 4), gain=0.9, freq=fr, pan=0, note="doubling (pitch rising)")
        self.cue("ticks_burst", dt=round(T_RAMP0, 4), dur=round(T_RAMP1 - T_RAMP0, 4), rate0=12, rate1=24,
                 gain=0.7, note="the GPU count climbs")
        self.cue("riser", dt=round(T_RAMP0, 4), dur=round(T_WALL1 - T_RAMP0, 4), gain=0.6,
                 note="the speedup climbs, thins out into the wall")
        self.cue("hit", dt=round(T_WALL1, 4), gain=0.4, note="THE WALL label fades in")
        self.cue("silence", dt=round(T_WALL1 + WALL_LAB_FADE, 4), dur=round(T_HIT - T_WALL1 - WALL_LAB_FADE, 4),
                 depth=0.03, note="near-silence: the stall")
        self.cue("hit", dt=round(T_HIT, 4), gain=1.0, note="'10×', '∞' and the takeaway")
        self.cue("sub_drop", dt=round(T_HIT, 4), dur=2.0)
        self.cue("pad_swell", dt=round(T_TAKE, 4), dur=round(T_COL - T_TAKE, 4), chord=1,
                 note="dark chord under the takeaway, into the launch")
        for t, d in zip(T_LUBS[1:], T_DUBS):
            self.cue("hit", dt=round(t, 4), gain=0.35, note="heartbeat lub: the block + ring (steady, 0.5 s grid)")
            self.cue("hit", dt=round(d, 4), gain=0.22, note="heartbeat dub: the pulse hits the wall")
        self.cue("hit", dt=round(T_COL, 4), gain=0.35, note="final lub: ring, the block launches into the token")
        self.cue("whoosh", dt=round((T_COL + T_SET) / 2, 4), dur=COLLAPSE_TIME, direction="down",
                 note="the serial block becomes the token")
        self.cue("tick", dt=round(T_SET, 4), gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_COL)
        self._draw(T_COL)                          # the block at rest (lub / dub exactly 0)
        self.collapse = COLLAPSE(self.serial, None, "plain", AMBER)
        self.add(self.collapse.glyph)
        hold_floor(self, T_EXIT)
        exit_to(self)                              # canonical E08, static 12.45-12.50

    # ======================================================================
    # Driver
    # ======================================================================
    def _draw(self, now: float) -> None:
        f = story_frame(now)
        sh = ZERO                                  # no camera shake (script v3 calm motion)
        clear = 1.0 - smooth(ramp(now, T_CLEAR0, T_CLEAR1))
        y0, y1 = bar_pose(now)
        self._k, self._b = stage_cam(now)
        self._draw_bar(now, f, sh, clear, y0, y1)
        self._draw_graph(now, f, sh, clear)
        self._draw_labels(now, f, sh, clear, y0, y1)
        self._draw_readout(now, f, sh, clear, y1)
        self._apply_cam()

    # ---------------- stage camera (post-hit lean-in) ----------------
    def _apply_cam(self) -> None:
        """Lean-in on the stage VMobjects, applied on top of this frame's geometry (every one of
        them is redrawn from scratch each frame, so nothing compounds). The serial block is left
        alone once the Collapse owns it (it captured the leaned-in block at T_COL)."""
        k, b = self._k, self._b
        if k == 1.0:
            return
        mobs = [*self.graph.family_members_with_points(), self.bar, *self.rows, self.finish,
                *self.halves, self.crack]
        if self.collapse is None:
            mobs.append(self.serial)
        for m in mobs:
            if len(m.points):
                m.points = m.points * k + b

    def _show(self, slot: TextSlot, tmpl, alpha, scale=1.0, about=None, shift=None, tint=None) -> None:
        """slot.display(...) seen through the stage camera: p -> k*(about + s*(p - about) + shift) + b."""
        k, b = self._k, self._b
        if k != 1.0:
            ab = np.asarray(about if about is not None else getattr(tmpl, "_about", tmpl.get_center()), float)
            sh = np.zeros(3) if shift is None else np.asarray(shift, float)
            shift, scale, about = k * (ab + sh) + b - ab, scale * k, ab
        slot.display(tmpl, alpha, scale, about, shift, tint)

    # ---------------- job bar / serial block / parallel rows ----------------
    def _draw_bar(self, now, f, sh, clear, y0, y1) -> None:
        yc, h = (y0 + y1) / 2, y1 - y0
        if now < T_NO - EPS:
            for m in (self.serial, *self.halves, self.crack, self.finish, *self.rows):
                m.points = EMPTY.copy()
            w = (BAR_X1 - BAR_X0) * CALM_EASE(ramp(now, T_WIPE0, T_WIPE1))
            if w < 1e-4:
                self.bar.points = EMPTY.copy()
                return
            x0, x1 = BAR_X0, BAR_X0 + w
            self.bar.points = rr_box(x0, y0, x1, y1, BAR_R)
            # fill gradient across x: wipe head (INK-lifted leading edge) + travelling shimmer
            K = 40
            x = x0 + np.linspace(0.0, 1.0, K) * w
            op = np.full(K, 0.75)
            amp = shimmer_amp(now)
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
        off = sh
        sep = split_sep(now)
        if self.collapse is not None:
            # the launch: the Collapse's own curve (COLLAPSE_EASE, 0.80 s) carries the block to AC
            self.collapse.render(now, T_COL, T_SET, T_EXIT)
            if now < T_SET - EPS:              # keep the amber punch in flight; land on E08's 0.58
                e = self.collapse.e_at(now, T_COL, T_SET)
                fr = self.serial.fill_rgbas.copy()
                fr[:, 3] = lerp(MORPH_FILL0, token_fill(AMBER), e ** 2.5)
                self.serial.fill_rgbas = fr
            for m in (*self.halves, self.crack):
                m.points = EMPTY.copy()
        else:
            hb = min(1.0, lub(now) + 0.3 * dub(now))
            flash = max(decay(now - T_NO, 0.22), decay(now - T_HIT, 0.30))
            s_fill = [*lerp(C_AM, C_INK, max(0.65 * flash, 0.15 * hb, BREATH_INK * breath(now))), MORPH_FILL0]
            ca = crack_alpha(now)
            if sep > 1e-4:                     # the failed split: the halves ease apart and back
                self.serial.points = EMPTY.copy()
                for m, sgn in ((self.halves[0], -1), (self.halves[1], 1)):
                    m.points = half_pts(sgn, y0, y1) + off + np.array([sgn * CRACK_SEP * sep,
                                                                       -sgn * CRACK_SHEAR * sep, 0.0])
                    set_style(m, s_fill, [*C_INK, 1.0], 2.0)
                self.crack.points = crack_pts(y0, y1) + off
                set_style(self.crack, None, [*C_INK, ca], 3.0)
            else:
                sc = (1.0 + HB_AMP * hb) * thunk_scale(now)
                self.serial.points = rrect_c(P(SER_CX, yc) + off, SER_W * sc, h * sc, SER_R * sc)
                set_style(self.serial, s_fill, [*C_INK, 1.0], 2.0)
                for m in self.halves:
                    m.points = EMPTY.copy()
                cd = crack_draw(now)
                if ca > 0.0 and cd > 0.0:      # the thin INK crack draws on, then fades after the rejoin
                    self.crack.points = crack_pts(y0, y1, cd) + off
                    set_style(self.crack, None, [*C_INK, ca], 3.0)
                else:
                    self.crack.points = EMPTY.copy()

        # ---- parallel rows
        rects, gl = self._rows(now, y0, y1)
        rgb = lerp(C_CY, C_INK, 0.45 * gl)
        op = min(1.0, 0.8 + 0.2 * gl) * clear
        amp = shimmer_amp(now) if len(rects) == 1 else 0.0
        for i, m in enumerate(self.rows):
            if i < len(rects):
                x0, ry0, x1, ry1, r = rects[i]
                m.points = rr_box(x0, ry0, x1, ry1, r) + sh
                if i == 0 and amp > 0:          # the 90% part keeps the bar's travelling shimmer
                    x = np.linspace(x0, x1, 40)
                    opk = np.clip(op + amp * np.sin(TAU * (1.5 * now - x / 3.0)), 0.0, 1.0)
                    m.fill_rgbas = np.column_stack([np.tile(rgb, (len(x), 1)), opk])
                    m.stroke_rgbas = np.zeros((1, 4))
                else:
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
        if now < T_RAMP0 - EPS or dbl_settling(now):
            k = dbl_index(now)
            n0 = 2 ** k
            p = ease_out_cubic(ramp(now, T_DBL[k], T_DBL[k] + DBL_DUR))
            gl = 1.0 - ease_out_cubic(ramp(now, T_DBL[k], T_DBL[k] + 0.28))
            out = [lerp(row_rect(n0, c // 2, y0, y1), row_rect(2 * n0, c, y0, y1), p) for c in range(2 * n0)]
            # the race starts while the 16 rows are still settling: they thin with N already
            ws = 16.0 / n_ramp(now) if now >= T_RAMP0 - EPS else 1.0
            for r in out:                       # overshoot safety (heights/widths stay > 0)
                r[3] = max(r[3], r[1] + 1e-3)
                if ws != 1.0:
                    r[2] = r[0] + max((r[2] - r[0]) * ws, SLIVER_MIN)
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
            beat = (lambda uu: np.zeros_like(uu)) if pp is None else (
                lambda uu: np.exp(-((uu - pp * L) / 0.45) ** 2) * min(1.0, 4.0 * pp + 0.25))
            glint = lambda uu: np.maximum(beat(uu), flow(now, uu, L))     # noqa: E731
            # the area under the curve lifts with the beat at full strength, with the flow at
            # half (its 40 strips read as blocky columns at full strength)
            area_gl = lambda uu: np.maximum(beat(uu), 0.5 * flow(now, uu, L))  # noqa: E731
            rgb = lerp(rgb, C_INK[None, :], 0.85 * glint(uk)[:, None])
            set_style(self.curve, None, np.column_stack([rgb, np.full(K, clear)]), CURVE_W)
            # the glow under the curve: one strip per 1/40 of the curve, fading to 0 at the axis
            # (a flat fill read as an olive slab under the wall)
            ue = np.linspace(0.0, L, AREA_N + 1)
            us = ue[:-1, None] + (ue[1:] - ue[:-1])[:, None] * np.linspace(0.0, 1.0, 4)[None, :]
            tx, ty = gx(us) + sh[0], gy(amdahl(10 ** us)) + sh[1]
            srgb = curve_rgb(amdahl(10 ** ((ue[:-1] + ue[1:]) / 2)))
            ak = (1.0 + 0.8 * dd + 1.6 * area_gl((ue[:-1] + ue[1:]) / 2)) * clear
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
        hs = ease_out_cubic(ramp(now, T_HEAD, T_HEAD + 0.30))
        hr = (HEAD_R + 0.06 * dd) * hs * (1.0 + HEAD_BREATH * head_breath(now))
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
            # dim to 0.55 for the ramp, then leave as the takeaway lands (it says '10% serial')
            a = (1.0 - 0.45 * ramp(now, T_DIM0, T_DIM1)) * (1.0 - smooth(ramp(now, T_DECL0, T_DECL1))) * clear
            sc = pop_scale(now - T_LABELS)
            # POP about the inner edges so the gap between the labels never closes
            self._show(self.s_lab_a, self.t_lab_a, a, sc, self.t_lab_a.get_right() + lab_dy, sh + lab_dy)
            self._show(self.s_lab_b, self.t_lab_b, a, sc, self.t_lab_b.get_left() + lab_dy, sh + lab_dy)
        # 'TIME TO FINISH' (rides the bar; the lean-in camera too). Out for good as the ramp starts: the
        # takeaway frame keeps only 'GPUs: ∞' on that row.
        ca = ease_out_cubic(ramp(now, T_CAP_IN, T_CAP_IN + 0.25))
        ca *= (1.0 - smooth(ramp(now, T_CAP_OUT0, T_CAP_OUT1))) * clear
        if ca <= 0:
            self.s_cap.clear_slot()
        else:
            shift = np.array([0.0, y1 - BAR_Y1F - 0.12 * (1 - ca), 0.0]) + sh
            self._show(self.s_cap, self.t_cap, ca, 1.0, self.t_cap.get_center(), shift)
        # axis titles (taught from T_AXLAB; gone under the takeaway)
        aa = ease_out_cubic(ramp(now, T_AXLAB, T_AXLAB + 0.25)) * (1.0 - smooth(ramp(now, T_DECL0, T_DECL1))) * clear
        for slot, t in ((self.s_ax_y, self.t_ax_y), (self.s_ax_x, self.t_ax_x)):
            if aa <= 0:
                slot.clear_slot()
            else:
                self._show(slot, t, aa, 1.0, None, sh)
        # THE WALL: 10× fades in (steps back to WALL_LAB_TAKE under the takeaway, which says
        # '= max 10×.')
        if now < T_WALL1 - EPS:
            self.s_wall.clear_slot()
        else:
            ab = np.array([self.t_wall.get_right()[0], self.t_wall.get_center()[1], 0.0])
            wa = (CALM_EASE(ramp(now, T_WALL1, T_WALL1 + WALL_LAB_FADE))
                  * lerp(1.0, WALL_LAB_TAKE, smooth(ramp(now, T_DECL0, T_DECL1))) * clear)
            self._show(self.s_wall, self.t_wall, wa, 1.0, ab, sh)

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
            for sl in (self.s_n, self.s_nc):
                sl.clear_slot()
            return
        tm = self._ntmpl(n_str(n))
        ab = tm._anchor                            # right edge, cap-band centre
        # fades in, pops on every doubling (before the climb), rides the bar
        a = CALM_EASE(ramp(now, T_N_IN, T_N_IN + N_FADE)) * clear
        pop = 1.0
        for t in T_DBL:
            pop *= pop_scale(now - t)
        shift = np.array([0.0, y1 - BAR_Y1F, 0.0]) + sh
        # 'GPUs: ∞' cross-fades in over 'GPUs: 1,000,000' at T_HIT (no glitch-snap)
        q = CALM_EASE(ramp(now, T_HIT, T_HIT + N_XFADE)) if now >= T_HIT - EPS else 1.0
        self._show(self.s_n, tm, a * q, pop, ab, shift)
        if q < 1.0:
            old = self._ntmpl(n_str(n_ramp(T_RAMP1)))
            self._show(self.s_nc, old, a * (1.0 - q), pop, old._anchor, shift)
        else:
            self.s_nc.clear_slot()

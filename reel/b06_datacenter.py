"""b06 - LEVEL 3/3 - DATA CENTER (12.5 s, script v2 'datacenter'): many boxes with no shared
memory pass messages; then the hall, 100,000+ GPUs training one AI.

Entry E04 (the gpu token + HUD '16,896' / CORES WORKING AT ONCE) is box (c3, r3) of one pod
(8 racks x 8 boxes, 5.36 units wide). The camera pulls back (EXPAND 0.00-0.50; the hero outline
INK-flashes on story frames 1-2, its 8 x 8 gpu glyph fades out by e 0.35). 'Many boxes. / No
shared memory.' POPs on the cut (headline story frame 1, sub-line frame 2) over a calm, DIM pod:
only the hero is lit (it keeps computing on its own clock, a lone worker among dark boxes), the
other 63 boxes, the rack lines and the ToR/spine switches are dim and the network breathes
faintly. Once the card has been read (1.50) the pod powers on rack by rack
in a CYAN wave (one rising tick per rack), and every box computes on its own data (out-of-sync
CYAN flashes); nothing travels between boxes yet.

'So they pass / messages.' hard-swaps in at 2.60 over a calm stage (the boxes hum, the network
brightens faintly) and is read for 1.0 s. Then SYNC: 16 MAGENTA messages hop like letters, up
the sender's rack line into its top-of-rack switch, over an uplink into a spine, down into a
DIFFERENT rack's ToR and down to a box there (each switch flashes as a message passes, the
receiver flashes INK); at 4.50 every box applies the same update (LIME flash + pop). A faster
second cycle runs 4.65-5.25.

'Go bigger.' hard-swaps in at 5.40; the pod holds calm under a slow push-in (1.0 -> 1.025) for
the 1.0 s read. At 6.40 the second pull-back shrinks the pod into the 16 squares of a hall that
overfills its window on every side (a representative 66 x 56 squares seen through a soft
6.9 x 5.5 window; 2 x 2 boxes land on one square), while the HUD log-rolls up from 16,896,
accelerating toward 1,689,600,000. At 7.300 everything stops dead: 80 ms freeze + true silence
(5 story frames; 'Go bigger.' hard-cuts out, the HUD is frozen mid-roll at ~260,000,000), then
THE BIGGEST DROP of the film at 7.383 (film 33.88 s): the HUD SLAMs 1,689,600,000+ with the
LIME line '100,000+ GPUs × 16,896' in the label slot (7.383-9.183), the card '100,000+' SLAMs
in (INK on its first two story frames, then CYAN; start scale capped to the safe zone) and
'GPUs. One AI.' POPs 2 story frames later. The punch lives on the stage: the hall flashes
INK -> CYAN, a radial INK burst, a CYAN shock ring and a 0.16 impact shake, and messages race
along the trunks. 8.40-9.60 runs one all-reduce: a MAGENTA gather into the hall centre, the
reduced result glows LIME there (cover frame ~9.0), then a LIME broadcast sweeps back out and
the network grid overlay (trunk lines + 48 junction switches) draws on behind its front. Then
the SCREENSHOT HOLD under a slow push-in (1.0 -> 1.012).

'More GPUs…' hard-swaps in at 10.30 over a riser that climbs into the seam and never resolves
(riser_to_cut: it stops dead on the cut, b08's record scratch cuts it). The camera eases back
over the humming hall (1.012 -> 0.90: fresh squares stream in through every edge of the window),
the hall's energy and the message traffic build with the riser; at 11.30 a last gather pulls
into the centre while an outline draws on around the hall window (11.45-11.75), which COLLAPSEs
(cut_times(12.5): 11.933 / 12.333 / 12.45) into handoff_token('cluster') (E06) with the HUD
riding the cut at 1,689,600,000+ / CORES WORKING AT ONCE. The kicker and the card hold through
the collapse and HARD-CUT at the exit (12.45, never a fade); only the 3 static E06 frames are
text-less.

Everything that moves is a pure function of self.now (one driver). The hall and the junction
switches are motif.Fields with fixed quantisation (temporally stable colours); off-window
squares are skipped. No dead holds: the stage (never text or HUD) always has life (network
breathing, box hum, push-ins, hall hum, trunk traffic).
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# Stage A: one pod (8 racks x 8 boxes)
# --------------------------------------------------------------------------
SRV_U = 0.46
SRV_PX, SRV_PY = 0.70, 0.56                               # rack pitch (x), box pitch in a rack (y)
RACK_X = X_C + (np.arange(8) - 3.5) * SRV_PX             # rack c = 0..7: x[-2.80, 2.10]
ROW_Y = 1.93 - np.arange(8) * SRV_PY                     # row r = 0..7 (top = 0): y[-1.99, 1.93]
SRV_C = np.array([[RACK_X[k % 8], ROW_Y[k // 8], 0.0] for k in range(64)])
HERO = 3 * 8 + 3                                         # (c3, r3) at (-0.70, 0.25)
SLOT = SRV_C[HERO].copy()
RACK_BOT = ROW_Y[7] - SRV_U / 2                          # -2.22
TOR_W, TOR_H = 0.54, 0.18
TOR_Y = ROW_Y[0] + SRV_U / 2 + 0.10 + TOR_H / 2          # 2.35
RACK_TOP = TOR_Y - TOR_H / 2                             # 2.26 (rack lines end in the ToR)
TOR_TOP = TOR_Y + TOR_H / 2                              # 2.44
SPINE_W, SPINE_H = 1.00, 0.24
SPINE_Y = 3.00                                           # spines y[2.88, 3.12]
SPINE_BOT = SPINE_Y - SPINE_H / 2
SPINE_X = X_C + (np.arange(4) - 1.5) * 1.40
PIVOT_A = np.array([X_C, (ROW_Y[0] + ROW_Y[7]) / 2, 0.0])   # box-block centre (zoom-out 2 pivot)
# box styles: the 63 others start DIM (they power on at 1.50); the hero is lit from the cut
DIM_FILL, DIM_STROKE = 0.08, 0.45                        # fill opacity, stroke opacity
ON_FILL = 0.18                                           # powered fill (stroke opacity 1.0)
SW_DIM = 0.45                                            # ToR / spine switches before the power wave
TOR_FILL, SPINE_FILL = 0.30, 0.35                        # their powered fill opacity

# --------------------------------------------------------------------------
# Stage B: the hall (66 x 56 squares, representative; seen through a 6.9 x 5.5 window)
# --------------------------------------------------------------------------
HNX, HNY = 66, 56
HP = 0.12                                                # hall pitch
_I, _J = np.meshgrid(np.arange(HNX), np.arange(HNY), indexing="ij")   # i = column, j = row
HI, HJ = _I.ravel(), _J.ravel()
HCX, HCY = (HNX - 1) / 2, (HNY - 1) / 2                  # 32.5, 27.5: the hall centre is AC
HALL_C = np.stack([X_C + (HI - HCX) * HP, AC[1] + (HCY - HJ) * HP, np.zeros(HNX * HNY)], axis=1)
HALL_D = np.hypot(HALL_C[:, 0] - AC[0], HALL_C[:, 1] - AC[1])
POD = (np.abs(HI - HCX) < 2) & (np.abs(HJ - HCY) < 2)    # where the pod lands (4 x 4 squares)
HALL_SQ = 0.7 * HP
HALL_UNIT = RoundedRectangle(width=1.0, height=1.0, corner_radius=0.012 / 0.07).points.copy()
_hrng = np.random.default_rng(61)
HALL_VAR = 0.80 + 0.20 * _hrng.random(HNX * HNY)         # per-square brightness texture
HUM_F = 0.7 + 0.9 * _hrng.random(HNX * HNY)              # per-square hum frequency (Hz) ...
HUM_PH = TAU * _hrng.random(HNX * HNY)                   # ... and phase (the hall hums, never static)
HUM_AMP = 0.12
HALL_HALF_X = HCX * HP + HALL_SQ / 2                     # 3.942
HALL_HALF_Y = HCY * HP + HALL_SQ / 2                     # 3.342
# The hall overfills the stage on every side: it is seen through a window (soft 0.16 edges, 0.2
# clear of the HUD digits) that the collapse outline frames exactly.
WIN_X0, WIN_X1 = AC[0] - 3.45, AC[0] + 3.45              # -3.80 / 3.10 (inside STAGE x)
WIN_LO, WIN_HI = -2.30, 3.20                             # centred on AC
WIN_SOFT = 0.16
OUT_W, OUT_H = WIN_X1 - WIN_X0, WIN_HI - WIN_LO          # outline 6.9 x 5.5 centred on AC
ZX = SRV_PX / (HP / 2)                                   # 11.667: a rack pitch -> half a hall pitch
ZY = SRV_PY / (HP / 2)                                   # 9.333 (2 x 2 boxes land on one square)
TRUNK_OX = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) * 4 * HP   # +-0.48, 1.44, 2.40, 3.36: in the gutters
TRUNK_OY = np.array([-5, -3, -1, 1, 3, 5]) * 4 * HP
TRUNK_X = X_C + TRUNK_OX
TRUNK_Y = AC[1] - TRUNK_OY
NODE_C = np.array([[x, y, 0.0] for x in TRUNK_X for y in TRUNK_Y])   # 48 junction switches
NODE_D = np.hypot(NODE_C[:, 0] - AC[0], NODE_C[:, 1] - AC[1])
NODE_SQ = 0.10
NODE_UNIT = RoundedRectangle(width=1.0, height=1.0, corner_radius=0.25).points.copy()
POD_OP = 0.60                                            # the pod's 16 squares right after the zoom
TRUNK_DIM, TRUNK_HI = 0.16, 0.55                         # trunk opacity: after the drop / overlay drawn
R_PULSE = 4.6                                            # gather / broadcast radius (window corner 4.41)

# --------------------------------------------------------------------------
# Timeline (beat-local seconds), script v2. The 12.5 s slot is filled by SEQUENCING (card lands
# -> read on a calm stage -> the motion it describes -> payoff hold), never by slowing: the
# EXPAND (0.50), power wave (0.57), SYNC cycles (0.90 / 0.60), zoom-out (0.90), roll, freeze
# (80 ms), slam, pulses (0.40 / 0.60 / 0.45), outline (0.30) and collapse keep snappy durations;
# the time sits BETWEEN them. Card windows (fully visible -> cut; floor 0.3 s/word + 0.8 s):
#   0.00-0.50  EXPAND; 'Many boxes. / No shared memory.' POPs on story frames 1-2
#              (fully visible 0.033 -> 2.60 = 2.57 s >= 2.30); calm dim pod, read 1.47 s
#   1.50-2.07  power-on wave rack by rack; async CYAN compute flashes from each box's power-on
#   2.60       'So they pass / messages.' hard swap (2.60 -> 5.40 = 2.80 s >= 2.00); calm hum 1.0 s
#   3.60-4.50  SYNC 1: compute flash, 16 messages hop, LIME update 4.50
#   4.65-5.25  SYNC 2 (faster hops), LIME update 5.25
#   5.40       'Go bigger.' hard swap (5.40 -> 7.30 = 1.90 s >= 1.40); push-in 1.0 -> 1.025
#   6.40-7.30  zoom-out 2 (pod -> hall) + HUD log-roll from 16,896 (cut at the freeze)
#   7.30-7.383 80 ms freeze + true silence; 'Go bigger.' cuts out
#   7.383      DROP: '100,000+' SLAM + 'GPUs. One AI.' POP 7.417 (-> 10.30 = 2.88 s >= 2.00);
#              1,689,600,000+ SLAM + LIME '100,000+ GPUs × 16,896' (hold 1.8: 7.383 -> 9.183)
#   8.40-9.60  all-reduce: gather 8.40-8.80, LIME result 8.80-9.00, broadcast 9.00-9.60 with the
#              network grid overlay drawing on behind its front
#   9.60-10.30 SCREENSHOT HOLD (hall push-in 1.0 -> 1.012, hum, trunk traffic)
#   10.30      'More GPUs…' hard swap (10.30 -> 12.45 = 2.15 s >= 1.40); riser to the seam;
#              camera eases back 1.012 -> 0.90, energy and traffic build
#   11.30-11.75 the last gather; outline draws on 11.45-11.75
#   11.933-12.45 collapse + settle under the held card; 12.45-12.50 static E06 (3 frames)
# --------------------------------------------------------------------------
BEAT_SLOT = 12.5                                         # timeline.py slot (asserted in beat())
T_EXP = EXPAND_TIME                                      # 0.50
T_C1_SUB = 2 / 60                                        # 'No shared memory.' POPs 1 story frame after the headline
T_POWER = 1.50                                           # the pod powers on, rack by rack
T_CARD2 = 2.60                                           # 'So they pass / messages.' hard swap
# SYNC cycles: (compute flash c0, send offset, hop-duration factor, LIME update)
SYNC = ((3.60, 0.12, 1.00, 4.50), (4.65, 0.03, 0.72, 5.25))
T_CARD3 = 5.40                                           # 'Go bigger.' hard swap
T_Z0 = 6.40                                              # zoom-out 2 starts (0.90 s, ends ON the freeze)
T_FREEZE = 438 / 60                                      # 7.300: 80 ms freeze + silence (5 story frames)
T_DROP = 443 / 60                                        # 7.383: the biggest drop of the film
T_Z1 = T_FREEZE
T_ROLL0 = T_Z0                                           # HUD roll 16,896 -> 1,689,600,000 (cut at the freeze)
T_C4_SUB = T_DROP + 2 / 60                               # 'GPUs. One AI.' POPs after the 2 INK frames
LINE_HOLD = 1.80                                         # LIME line 7.383 -> 9.183 (floor 1.5)
T_CARD5 = 10.30                                          # 'More GPUs…' hard swap + the riser
T_COL, T_SET, T_EXIT = cut_times(BEAT_SLOT)              # 11.933 / 12.333 / 12.45
PULSES = (("gather", 8.40, 8.80), ("bcast", 9.00, 9.60), ("gather", 11.30, 11.75))
T_HOT = (8.78, 9.00)                                     # the reduced result glows LIME in the centre
T_OUT0, T_OUT1 = 11.45, 11.75                            # outline draws on around the hall window
T_TEXT_OUT = T_EXIT                                      # kicker + card hold to the exit, then hard-cut
# ambient camera life (stage only, never text or HUD): p -> piv + k (p - piv)
LIFE_A, LIFE_PIV_A = 1.025, np.array([X_C, 0.95, 0.0])   # pod push-in under 'Go bigger.'
IDLE_AMP, IDLE_PER = 0.16, 1.6                           # pod network breathing: glow level, period
IDLE_CALM = 0.24                                         # ... brighter under 'So they pass messages.'
LIFE_B, LIFE_END = 1.012, 0.90                           # hall: push-in to the screenshot hold, ease back
HUM_BOX = 0.40                                           # async compute amplitude while the boxes 'hum'
MORE_GAIN = 0.12                                         # hall brightness build under 'More GPUs…'
MSG_V0, MSG_V1 = 3.6, 7.0                                # trunk message speed: at the drop / at the collapse
SLAM_BOX = Box(SAFE_LEFT + 0.05, SAFE_RIGHT - 0.05, SAFE_BOTTOM, SAFE_TOP)   # '100,000+' slam stays in here
SLAM_INK = 2 / 60                                        # the slammed card is INK for 2 story frames
DROP_SHAKE = (0.16, 0.30)                                # hall impact shake: amplitude, duration

# --------------------------------------------------------------------------
# Per-box schedule (k = r * 8 + c): power-on wave rack by rack (bottom to top inside a rack),
# then each box computes on its own clock (out-of-sync CYAN flashes).
# --------------------------------------------------------------------------
BOX_R, BOX_COL = np.arange(64) // 8, np.arange(64) % 8
T_ON = T_POWER + 0.07 * BOX_COL + 0.012 * (7 - BOX_R)
T_ON[HERO] = T_EXP                                        # the hero is lit from the cut: it computes alone
_brng = np.random.default_rng(616)
ASYNC_P = 0.55 + 0.40 * _brng.random(64)                  # compute period per box
ASYNC_PH = 0.08 + _brng.random(64) * ASYNC_P              # first flash after its power-on
ASYNC_W = 0.20                                            # flash length
POWER_FL = 0.30                                           # power-on flash decay
NOT_HERO = np.arange(64) != HERO
T_TOR_ON = T_POWER + 0.07 * np.arange(8) + 0.096           # a rack's ToR powers when the wave tops it
T_SPINE_ON = T_POWER + 0.07 * 7 + 0.14                     # then the spine tier (2.13)

# --------------------------------------------------------------------------
# Stage-A messages: 16 per SYNC cycle, each hops box -> ToR -> spine -> ToR' -> box' in a
# different rack. Phases: move, dwell (ToR), move, dwell (spine), move, dwell (ToR'), move.
# --------------------------------------------------------------------------
N_HOP = 16
HOP = np.array([0.12, 0.035, 0.10, 0.035, 0.10, 0.035, 0.12])
HOP_EDGE = np.concatenate([[0.0], np.cumsum(HOP)])       # phase starts (8 values; last = arrival)
HOP_MOVE = np.array([1, 0, 1, 0, 1, 0, 1], bool)
TRAIL = 0.30                                             # message trail length (units)


def _hops():
    rng = np.random.default_rng(606)
    t0, kk, src, dst, sp, paths = [], [], [], [], [], []
    for c0, off, fac, _ in SYNC:
        rows = [rng.permutation(8)[:2] for _ in range(8)]     # 2 senders per rack, distinct rows
        for m in range(N_HOP):
            cs, rs = m % 8, int(rows[m % 8][m // 8])
            cd = int((cs + rng.integers(1, 8)) % 8)            # always a different rack
            rd = int(rng.integers(0, 8))
            s = int(rng.integers(0, 4))
            ts = c0 + off + 0.10 * rng.random()
            xs, xd, xp = RACK_X[cs], RACK_X[cd], SPINE_X[s]
            paths.append([(xs, ROW_Y[rs]), (xs, TOR_Y), (xs, TOR_TOP), (xp, SPINE_BOT), (xp, SPINE_Y),
                          (xp, SPINE_BOT), (xd, TOR_TOP), (xd, TOR_Y), (xd, ROW_Y[rd])])
            t0.append(ts)
            kk.append(fac)
            src.append((cs, rs))
            dst.append((cd, rd))
            sp.append(s)
    P = np.array(paths, float)                                   # (n, 9, 2)
    seg = np.hypot(*(P[:, 1:] - P[:, :-1]).transpose(2, 0, 1))   # (n, 8)
    cum = np.concatenate([np.zeros((len(P), 1)), np.cumsum(seg, axis=1)], axis=1)   # (n, 9)
    # arc length at each phase boundary: A ends at point 1, B at 4, C at 7, D at 8
    s_lo = cum[:, [0, 1, 1, 4, 4, 7, 7]]
    s_hi = cum[:, [1, 1, 4, 4, 7, 7, 8]]
    return (np.array(t0), np.array(kk), np.array(src), np.array(dst), np.array(sp), P, cum,
            s_lo, s_hi)


HOP_T0, HOP_K, HOP_SRC, HOP_DST, HOP_SP, HOP_P, HOP_CUM, HOP_SLO, HOP_SHI = _hops()
N_HOPS = len(HOP_T0)
# every message lands >= 0.08 s before its cycle's LIME update
assert np.all(HOP_T0 + HOP_EDGE[-1] * HOP_K < np.repeat([c[3] for c in SYNC], N_HOP) - 0.08)

# --------------------------------------------------------------------------
# Hall messages on the trunks: 120 seeded random walks on the inner 6 x 6 trunk lattice
# --------------------------------------------------------------------------
N_MSG = 120
EDGE = 0.96
MSG_TAIL = 0.17
_mrng = np.random.default_rng(66)


def _walks(steps: int = 30) -> np.ndarray:
    out = np.zeros((N_MSG, steps + 1, 2), int)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for n in range(N_MSG):
        a, b = int(_mrng.integers(0, 6)), int(_mrng.integers(0, 6))
        out[n, 0] = (a, b)
        prev = None
        for s in range(steps):
            opts = [(a + dx, b + dy) for dx, dy in dirs if 0 <= a + dx < 6 and 0 <= b + dy < 6]
            if prev is not None and len(opts) > 1:
                opts = [o for o in opts if o != prev]
            nxt = opts[int(_mrng.integers(0, len(opts)))]
            prev = (a, b)
            a, b = nxt
            out[n, s + 1] = (a, b)
    return out


WALK = _walks()                                          # (120, 31, 2) lattice indices
WALK_XY = np.stack([TRUNK_X[1:7][WALK[..., 0]], TRUNK_Y[WALK[..., 1]]], axis=-1)
MSG_PHASE = _mrng.random(N_MSG) * EDGE                    # start partway along the first edge


def msg_dist(now: float) -> float:
    """Distance every trunk message has travelled since the drop: MSG_V0, then accelerating
    linearly to MSG_V1 over 'More GPUs…' (T_CARD5 -> T_COL), constant after."""
    ta, tb = T_CARD5, T_COL
    if now <= ta:
        extra = 0.0
    elif now < tb:
        extra = (now - ta) ** 2 / (2 * (tb - ta))
    else:
        extra = (tb - ta) / 2 + (now - tb)
    return MSG_V0 * (now - T_DROP) + (MSG_V1 - MSG_V0) * extra


assert (msg_dist(BEAT_SLOT) + EDGE) / EDGE < WALK.shape[1] - 1          # the walks never run out


def stage_mask(y):
    """Soft stage window for stage A: 1 for y in [-2.37, 3.25], 0 below -2.62 / above 3.55, so
    camera moves never drag geometry under the text zone or the HUD. Identity at rest."""
    y = np.asarray(y, float)
    return np.clip((y + 2.62) / 0.25, 0, 1) * np.clip((3.55 - y) / 0.30, 0, 1)


def hall_mask(x, y):
    """The hall window (camera space, before the collapse): full inside the 6.9 x 5.5 window
    shrunk by the soft edge, 0 outside it."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (np.clip((y - WIN_LO) / WIN_SOFT, 0, 1) * np.clip((WIN_HI - y) / WIN_SOFT, 0, 1)
            * np.clip((x - WIN_X0) / WIN_SOFT, 0, 1) * np.clip((WIN_X1 - x) / WIN_SOFT, 0, 1))


def edge_mask(bot: float, top: float) -> float:
    """Stage window for a whole member: fades as its bottom edge goes under -2.45 (HUD number
    top is -2.52) or its top edge over 3.40 (cards end at 3.48)."""
    return clip01((bot + 2.62) / 0.17) * clip01((3.58 - top) / 0.18)


def win(d: float, a: float, b: float, tail: float) -> float:
    """1 on [a, b), then easing out over `tail`; 0 before a."""
    if d < a - EPS:
        return 0.0
    if d < b - EPS:
        return 1.0
    return decay(d - b, tail)


def async_amp(now: float) -> float:
    """Amplitude of the out-of-sync compute flashes: full while every box computes alone
    (power-on -> 'So they pass messages.'), a hum under the card's read, off through the two
    SYNC cycles (bulk synchronous), a hum again under 'Go bigger.'."""
    c0, tl = SYNC[0][0], SYNC[1][3]
    if now < T_CARD2:
        return 1.0
    if now < c0 - 0.15:
        return lerp(1.0, HUM_BOX, ease_in_out_sine(ramp(now, T_CARD2, T_CARD2 + 0.30)))
    if now < c0:
        return HUM_BOX * (1.0 - ramp(now, c0 - 0.15, c0))
    if now < tl + 0.10:
        return 0.0
    return HUM_BOX * ramp(now, tl + 0.10, tl + 0.40)


def idle_amp(now: float) -> float:
    """Pod network breathing level: faint, a little brighter under 'So they pass messages.'."""
    c0 = SYNC[0][0]
    up = ramp(now, T_CARD2, T_CARD2 + 0.30) * (1.0 - ramp(now, c0, c0 + 0.20))
    return lerp(IDLE_AMP, IDLE_CALM, ease_in_out_sine(up))


class B06DataCenter(BeatScene):
    def beat(self):
        # ---------------- entry: E04 (gpu token + HUD '16,896' / CORES WORKING AT ONCE) ----------------
        st = enter(self)
        hud = st.hud

        # ---------------- stage A at rest (the DIM pod; only the hero is lit) ----------------
        self.racks = VGroup(*[Line([x, RACK_TOP, 0], [x, RACK_BOT, 0], stroke_color=MAGENTA,
                                   stroke_width=1.5, stroke_opacity=0.35) for x in RACK_X])
        self.uplinks = VGroup(*[Line([RACK_X[c], TOR_TOP, 0], [SPINE_X[s], SPINE_BOT, 0], stroke_color=MAGENTA,
                                     stroke_width=1.2, stroke_opacity=0.30)
                                for c in range(8) for s in range(4)])
        self.tors = VGroup(*[RoundedRectangle(width=TOR_W, height=TOR_H, corner_radius=0.05, stroke_color=MAGENTA,
                                              stroke_width=2, stroke_opacity=SW_DIM, fill_color=MAGENTA,
                                              fill_opacity=TOR_FILL * SW_DIM).move_to([x, TOR_Y, 0])
                             for x in RACK_X])
        self.spines = VGroup(*[RoundedRectangle(width=SPINE_W, height=SPINE_H, corner_radius=0.07,
                                                stroke_color=MAGENTA, stroke_width=2.5, stroke_opacity=SW_DIM,
                                                fill_color=MAGENTA, fill_opacity=SPINE_FILL * SW_DIM)
                               .move_to([x, SPINE_Y, 0]) for x in SPINE_X])
        srv = []
        for k in range(64):
            if k == HERO:
                # the gpu glyph's 64 squares pair one-to-one with the E04 token's: frame 0 is exact
                self.hero = unit_with_glyph("gpu", SRV_U, SRV_C[k], CYAN, fill=ON_FILL, stroke=2.5)
                srv.append(self.hero)
            else:
                srv.append(unit(SRV_U, CYAN, DIM_FILL, 2.5).set_stroke(opacity=DIM_STROKE).move_to(SRV_C[k]))
        self.servers = VGroup(*srv)
        self.srv_units = [s.unit if k == HERO else s for k, s in enumerate(srv)]
        self.world = VGroup(self.racks, self.uplinks, self.tors, self.spines, self.servers)
        rig = EXPAND(self.world, SLOT, SRV_U, 0.0, hero=self.hero, token="gpu",
                     fades={self.hero.glyph: (0.0, 0.35, 1.0, 0.0)})   # captures rest, draws e = 0
        self.rig = rig
        self.ix_srv = [rig.index[id(u)] for u in self.srv_units]
        self.ix_glyph = rig.indices_of(self.hero.glyph)
        self.ix_rack = rig.indices_of(self.racks)
        self.ix_up = rig.indices_of(self.uplinks)
        self.ix_tor = rig.indices_of(self.tors)
        self.ix_spine = rig.indices_of(self.spines)
        self.ix_hero = set(rig.indices_of(self.hero))
        self.ix_rack_set = set(self.ix_rack)
        self.ix_line = set(self.ix_rack) | set(self.ix_up)
        # rest centre of every member (zoom-out 2 moves boxes by their centres, so they stay square)
        self.rest_c = [0.5 * (r.p.min(axis=0) + r.p.max(axis=0)) for r in rig.rest]
        self.stage_a_on = True

        # stage-A messages: trail, head, hot core (one multi-subpath stroke VMobject each)
        self.m_trail = streak_mob(MAGENTA, 8)
        self.m_head = streak_mob(MAGENTA, 18)
        self.m_core = streak_mob(INK, 7)

        # ---------------- stage B: hall, trunks (dim + the drawn-on overlay), switches, messages ----------------
        self.hall = Field(HALL_UNIT, pool=256, qc=16, qa=24, fixed=True)
        self.nodes = Field(NODE_UNIT, pool=24, qc=16, qa=16, fixed=True)
        self.trunk_rest = ([((x, AC[1] - HALL_HALF_Y), (x, AC[1] + HALL_HALF_Y)) for x in TRUNK_X]
                           + [((X_C - HALL_HALF_X, y), (X_C + HALL_HALF_X, y)) for y in TRUNK_Y])
        self.trunk_vert = [True] * len(TRUNK_X) + [False] * len(TRUNK_Y)
        self.trunk_off = list(TRUNK_OX) + list(TRUNK_OY)            # |offset| from AC across the line
        self.trunks = VGroup(*[Line(LEFT, RIGHT) for _ in self.trunk_rest])
        self.trunks_hi = VGroup(*[Line(LEFT, RIGHT) for _ in self.trunk_rest])
        for ln in (*self.trunks, *self.trunks_hi):
            ln.set_stroke(MAGENTA, width=1.5, opacity=0)
        self.msgs = streak_mob(MAGENTA, 6)
        self.ring = ShockRing(self, T_DROP, AC, 0.3, 7.0, 14.0, 0.9, 0.55, CYAN, ease_out_cubic)
        # the drop's punch lives on the stage (the card slam is capped to the safe zone)
        self.flash = RadialFlash(self, T_DROP, AC, peak=0.55, radius=6.5, decay=0.12, tint=0.04)
        # the all-reduce's reduced result lands in the hall centre (LIME), before the broadcast
        self.result = RadialFlash(self, T_HOT[0], AC, peak=0.42, radius=1.9, decay=0.30, color=LIME)

        # ---------------- outline + COLLAPSE (captured now from the full outline) ----------------
        self.outline_full = RoundedRectangle(width=OUT_W, height=OUT_H, corner_radius=0.3, stroke_color=CYAN,
                                             stroke_width=3, fill_color=CYAN, fill_opacity=0).move_to(AC)
        self.outline = self.outline_full.copy()
        self.collapse = COLLAPSE(self.outline, None, "cluster")
        self.outline.set_stroke(opacity=0)

        # ---------------- text ----------------
        assert beat_duration(type(self).__name__) == BEAT_SLOT, "timeline.py slot changed: re-time b06"
        track = TextTrack(self)
        # the kicker arrives on the cut (frame 0 stays the exact E04) and holds through the
        # collapse; every card swap and the final exit are hard cuts (never a crossfade)
        track.on(kick("LEVEL 3/3 · DATA CENTER"), 0.0, T_TEXT_OUT, exit="cut")
        c1 = card([("Many boxes.", "H96", INK), ("No shared memory.", "S56", INK)])
        c2 = card([("So they pass", "H84"), ("messages.", "H84", MAGENTA)])
        c3 = card([("Go bigger.", "H96", INK)])
        c4 = card([("100,000+", "H96", CYAN), ("GPUs. One AI.", "S56", INK)])
        c5 = card([("More GPUs…", "H96", INK)])
        self.cards = (c1, c2, c3, c4, c5)
        # the switch tier keeps 0.3 of air under the pod's cards, also at the top of the push-in
        spine_top = LIFE_PIV_A[1] + (SPINE_Y + SPINE_H / 2 - LIFE_PIV_A[1]) * LIFE_A
        assert spine_top < min(c.get_bottom()[1] for c in (c1, c2, c3)) - 0.3, spine_top
        assert LIFE_PIV_A[1] + (RACK_BOT - LIFE_PIV_A[1]) * LIFE_A > -2.37      # racks stay in the stage window
        assert WIN_HI < min(c.get_bottom()[1] for c in (c4, c5)) - 0.3         # the hall window clears the cards
        # the hall overfills its window on every side, also at the end of the ease-back
        assert HALL_HALF_X * LIFE_END > WIN_X1 - AC[0] + 0.05 and HALL_HALF_Y * LIFE_END > WIN_HI - AC[1] + 0.05
        # card 1 POPs on the cut (headline story frame 1, sub-line frame 2): its reading floor starts at once
        track.on(c1[0], 1 / 60, T_CARD2, enter="pop", exit="cut")
        track.on(c1[1], T_C1_SUB, T_CARD2, enter="pop", exit="cut")
        track.on(c2[0], T_CARD2, T_CARD3, enter="pop", exit="cut")
        track.on(c2[1], T_CARD2, T_CARD3, enter="pop", exit="cut")
        track.on(c3[0], T_CARD3, T_FREEZE, enter="pop", exit="cut")        # the freeze blanks it
        # '100,000+' SLAMs (grows down from its top edge, clear of the kicker) with the start
        # scale capped to the safe zone; an INK twin carries the first 2 story frames (hit flash)
        about = np.array([X_C, CARD_TOP, 0.0])
        s_slam = fit_scale(c4[0], 1.3, about, SLAM_BOX)
        c4_ink = c4[0].copy().set_color(INK)
        track.on(c4_ink, T_DROP, T_DROP + SLAM_INK, enter="slam", exit="cut", about=about, scale=s_slam)
        track.on(c4[0], T_DROP, T_CARD5, enter="slam", exit="cut", about=about, scale=s_slam,
                 alpha=lambda now: 0.0 if now < T_DROP + SLAM_INK - EPS else 1.0)
        track.on(c4[1], T_C4_SUB, T_CARD5, enter="pop", exit="cut")
        lo = (c4[0].get_left() - about) * s_slam + about
        hi = (c4[0].get_right() - about) * s_slam + about
        assert SAFE_LEFT <= lo[0] and hi[0] <= SAFE_RIGHT, (lo, hi)
        track.on(c5[0], T_CARD5, T_TEXT_OUT, enter="pop", exit="cut")

        # ---------------- HUD ----------------
        # 16,896 holds while the pod visibly computes (undercount, never overclaim); the roll
        # starts with the zoom-out, aims at 1,689,600,000 by the hit but is CUT at the freeze
        # (~260,000,000): the destination first appears ON the drop frame, a 6x jump with the hit.
        hud.roll(T_ROLL0, T_DROP, 16896, 1689600000, ease=ease_in_quad)
        self.v_freeze = hud.number_at(T_FREEZE - 1e-4)[0]
        hud.cut(T_FREEZE, self.v_freeze)
        hud.slam(T_DROP, "1,689,600,000+")                  # start scale auto-capped (1.058, safe zone)
        hud.mult(T_DROP, COUNTER_LINES["B06DataCenter"], hold=LINE_HOLD)
        # the plural label (CORES WORKING AT ONCE) is back by 9.30 and rides the cut into E06

        # ---------------- z-order: dust, ring, hall, network, world, fx, collapse, text, HUD ----------------
        self.remove(st.token)                 # the hero box draws the token now
        add_behind(self, self.hall, self.trunks, self.trunks_hi, self.nodes, self.world, self.m_trail,
                   self.m_head, self.m_core, self.msgs, self.ring, self.flash, self.result, self.outline,
                   self.collapse.glyph)
        entry_ring(self)
        self.add(track)
        self.remove(hud)
        self.add(hud)
        self._draw(0.0)
        drive(self, self._draw)

        # ---------------- run: each cue sits next to the hold it lands in ----------------
        def cue_at(t, kind, **kw):
            self.cue(kind, dt=round(t - self.t, 4), **kw)

        # 0.00 the match-cut: pull-back into the dim pod, 'Many boxes.' read on a calm stage
        cue_at(0.0, "hit", gain=0.8, note="match-cut and ring")
        cue_at(0.25, "whoosh", dur=0.50, direction="up", note="pod pull-back (peaks mid-way: 0.00-0.50)")
        self.hold_until(T_POWER)
        # 1.50 the pod powers on rack by rack: one rising tick per rack
        for c in range(8):
            cue_at(T_POWER + 0.07 * c, "tick", gain=round(0.35 + 0.06 * c, 3), freq=2400 + 300 * c,
                   pan=round(-0.6 + 0.17 * c, 3), note="rack powers on")
        self.hold_until(SYNC[0][0])
        # 3.60 / 4.65 SYNC: messages up to the spines, down into other racks, the same update
        for c0, off, fac, tl in SYNC:
            up0, up1 = c0 + off, c0 + off + 0.10 + HOP_EDGE[3] * fac
            cue_at(up0, "ticks_burst", dur=round(up1 - up0, 4), rate0=18, rate1=30,
                   note="messages up to the spines")
            cue_at(up1, "ticks_burst", dur=round((HOP_EDGE[-1] - HOP_EDGE[3]) * fac, 4), rate0=30, rate1=18,
                   note="messages down to other racks")
            cue_at(tl, "pop", note="same update everywhere")
        self.hold_until(T_CARD3)
        # 5.40 'Go bigger.'
        cue_at(T_CARD3, "tick", gain=0.6, note="card swap")
        self.hold_until(T_Z0)
        # 6.40 zoom-out 2 + the accelerating odometer, a steep riser into the gap
        cue_at(T_Z0 + 0.45, "whoosh", dur=0.90, direction="up", note="hall zoom-out (6.40-7.30)")
        cue_at(T_ROLL0, "ticks_burst", dur=round(T_FREEZE - T_ROLL0, 4), rate0=6, rate1=40,
               note="odometer accelerating")
        cue_at(6.80, "riser", dur=round(T_FREEZE - 6.80, 4), note="steep, ends at the gap")
        self.hold_until(T_FREEZE)
        # 7.300 the 80 ms freeze: true silence
        cue_at(T_FREEZE, "silence", dur=round(T_DROP - T_FREEZE, 4), note="the gap before the drop")
        self.hold_until(T_DROP)
        # stage A has cross-faded into the hall: drop it (and its messages) from the scene
        self.stage_a_on = False
        self.remove(self.world, self.m_trail, self.m_head, self.m_core)
        # 7.383 THE DROP: top-tier hit (hit + sub_drop -> climax), euphoric lift
        cue_at(T_DROP, "hit", gain=1.0, note="the biggest drop of the film")
        cue_at(T_DROP, "sub_drop", dur=1.4)
        cue_at(T_DROP, "pad_swell", dur=1.6, note="euphoric lift")
        cue_at(PULSES[0][1], "tick", gain=0.5, note="all-reduce: gather")
        cue_at(PULSES[1][1], "tick", gain=0.5, note="all-reduce: broadcast")
        self.hold_until(T_CARD5)
        # 10.30 'More GPUs…': a riser that peaks and stops dead ON the seam (never resolves;
        # b08's record scratch cuts it), the last gather, the collapse (no resolving hit)
        riser_to_cut(self)
        cue_at(PULSES[2][1], "tick", gain=0.45, note="last gather")
        cue_at((T_COL + T_SET) / 2, "whoosh", gain=0.5, dur=0.40, direction="up", note="collapse")
        hold_floor(self, T_EXIT)
        exit_to(self)                         # canonical E06, static to 12.50
        print(f"[b06] frozen roll value {self.v_freeze}; hall buckets used (max): {self.hall.max_used}, "
              f"overflow frames: {self.hall.overflow}; switch buckets (max): {self.nodes.max_used}, "
              f"overflow: {self.nodes.overflow}")

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.stage_a_on:
            self._draw_stage_a(now)
        self._draw_hall(now)
        self._draw_outline(now)

    # ---------------- camera life (holds are never dead) ----------------
    def _life(self, now: float) -> tuple[float, np.ndarray]:
        """(k, piv): the stage's slow ambient camera p -> piv + k (p - piv), applied on top of the
        pod / hall cameras (never text or HUD). Identity until 'Go bigger.' (frame 0 = E04), the
        pod push-in under 'Go bigger.' is handed back during the zoom-out (the same k on the pod
        and the hall keeps the 2 x 2 groups on their squares), identity through the freeze; the
        hall's push-in (screenshot hold) and ease-back ('More GPUs…') fold into the collapse
        (which scales the contents about AC from wherever they are)."""
        if now < T_CARD3 - EPS:
            return 1.0, LIFE_PIV_A
        if now < T_Z0 - EPS:
            return lerp(1.0, LIFE_A, ease_in_out_sine(ramp(now, T_CARD3, T_Z0))), LIFE_PIV_A
        if now < T_DROP - EPS:
            return lerp(LIFE_A, 1.0, self._zoom2(now)), LIFE_PIV_A
        if now < T_CARD5 - EPS:
            return lerp(1.0, LIFE_B, ease_in_out_sine(ramp(now, T_DROP + 0.25, T_CARD5))), AC
        return lerp(LIFE_B, LIFE_END, ease_in_out_sine(ramp(now, T_CARD5, T_COL))), AC

    @staticmethod
    def _lf(P: np.ndarray, life) -> np.ndarray:
        k, piv = life
        return P if k == 1.0 else piv + (P - piv) * k

    # ---------------- stage A ----------------
    def _zoom2(self, now: float) -> float:
        if now >= T_FREEZE - EPS:
            return 1.0
        return ease_in_out_cubic(ramp(now, T_Z0, T_Z1))

    def _hop_s(self, t: float):
        """Arc length of every stage-A message along its path at time t, and its phase
        (-1 = not sent yet, 7 = arrived). Cycle 2's hops run HOP_K x the nominal durations."""
        d = (t - HOP_T0) / HOP_K
        ph = np.searchsorted(HOP_EDGE, d, side="right") - 1          # -1 .. 7
        p = np.clip(ph, 0, 6)
        frac = np.clip((d - HOP_EDGE[p]) / HOP[p], 0, 1)
        e = np.where(HOP_MOVE[p], ease_in_out_sine_arr(frac), 1.0)
        r = np.arange(N_HOPS)
        s = HOP_SLO[r, p] + (HOP_SHI[r, p] - HOP_SLO[r, p]) * e
        s = np.where(ph < 0, 0.0, np.where(ph > 6, HOP_CUM[:, -1], s))
        return s, ph

    @staticmethod
    def _hop_pos(s: np.ndarray) -> np.ndarray:
        """(n, 3) points at arc length s along each message's 9-point path."""
        k = np.clip((HOP_CUM <= s[:, None] + 1e-12).sum(axis=1) - 1, 0, 7)
        r = np.arange(N_HOPS)
        L = np.maximum(HOP_CUM[r, k + 1] - HOP_CUM[r, k], 1e-9)
        f = np.clip((s - HOP_CUM[r, k]) / L, 0, 1)[:, None]
        p = HOP_P[r, k] * (1 - f) + HOP_P[r, k + 1] * f
        return np.concatenate([p, np.zeros((N_HOPS, 1))], axis=1)

    def _draw_stage_a(self, now: float) -> None:
        rig = self.rig
        if now < T_EXP - EPS:
            EXPAND(self.world, SLOT, SRV_U, ease_out_expo(now / T_EXP),
                   ink=1.0 if flash_frames(now, 1 / 60, 2) else 0.0)
            # stage window on everything but the hero (which is the token at e = 0): rack lines
            # are clipped, every other member fades as an edge leaves the window
            for i, r in enumerate(rig.rest):
                if i in self.ix_hero:
                    continue
                m = r.m
                if i in self.ix_rack_set:
                    m.points[:, 1] = np.clip(m.points[:, 1], -2.40, 3.30)
                    continue
                k = edge_mask(m.points[:, 1].min(), m.points[:, 1].max())
                if k < 1.0:
                    m.fill_rgbas = m.fill_rgbas * [1, 1, 1, k]
                    m.stroke_rgbas = m.stroke_rgbas * [1, 1, 1, k]
            for mob in (self.m_trail, self.m_head, self.m_core):
                mob.points = EMPTY.copy()
            return

        # zoom-out 2: per-axis camera (x: ZX, y: ZY) so the 2 x 2 box groups land exactly on
        # the pod's hall squares at e2 = 1; boxes move by their centres (they stay square),
        # lines map affinely. Identity at e2 = 0.
        e2 = self._zoom2(now)
        zx, zy = ZX ** (-e2), ZY ** (-e2)
        piv = lerp(PIVOT_A, AC, e2)
        zv = np.array([zx, zy, 1.0])
        life = self._life(now)
        for i, r in enumerate(rig.rest):
            m = r.m
            if i in self.ix_line:
                m.points = self._lf(piv + (r.p - PIVOT_A) * zv, life)
            else:
                c = self.rest_c[i]
                m.points = self._lf((r.p - c) * zx + piv + (c - PIVOT_A) * zv, life)
            m.stroke_width = r.sw * zx
        a_net = 1.0 - ramp(e2, 0.25, 0.65)
        a_srv = 1.0 - ramp(e2, 0.75, 1.0)
        boost = ramp(e2, 0.30, 0.75)                     # boxes turn solid as they shrink

        # cycle state: compute flash, update flash
        cyan_f = lime_f = 0.0
        for c0, _, _, tl in SYNC:
            cyan_f = max(cyan_f, decay(now - c0, 0.25))
            lime_f = max(lime_f, decay(now - tl, 0.32))
        # per-box power-on (rack by rack) and out-of-sync compute flashes
        pw = np.where(NOT_HERO, (now >= T_ON - EPS).astype(float), 1.0)
        dt_on = now - T_ON
        pf = np.where(NOT_HERO & (dt_on >= -EPS), 1.0 - ease_out_cubic_arr(np.clip(dt_on / POWER_FL, 0, 1)), 0.0)
        dd = now - T_ON - ASYNC_PH
        ph = np.mod(np.maximum(dd, 0.0), ASYNC_P)
        af = np.where(dd >= 0, (1.0 - np.clip(ph / ASYNC_W, 0, 1)) ** 2, 0.0) * async_amp(now)
        # per-message glows: rack lines, uplinks, ToRs, spines, receivers
        rack_g, tor_f, up_g, spine_f, recv = np.zeros(8), np.zeros(8), np.zeros(32), np.zeros(4), np.zeros(64)
        for n in range(N_HOPS):
            d = (now - HOP_T0[n]) / HOP_K[n]
            if d < -EPS or d > HOP_EDGE[-1] + 0.30:
                continue
            (cs, rs), (cd, rd), s = HOP_SRC[n], HOP_DST[n], HOP_SP[n]
            E = HOP_EDGE
            rack_g[cs] = max(rack_g[cs], win(d, E[0], E[1], 0.10))
            tor_f[cs] = max(tor_f[cs], win(d, E[1], E[2], 0.16))
            up_g[cs * 4 + s] = max(up_g[cs * 4 + s], win(d, E[2], E[3], 0.12))
            spine_f[s] = max(spine_f[s], win(d, E[3], E[4], 0.16))
            up_g[cd * 4 + s] = max(up_g[cd * 4 + s], win(d, E[4], E[5], 0.12))
            tor_f[cd] = max(tor_f[cd], win(d, E[5], E[6], 0.16))
            rack_g[cd] = max(rack_g[cd], win(d, E[6], E[7], 0.10))
            if d >= E[7] - EPS:
                recv[rd * 8 + cd] = max(recv[rd * 8 + cd], decay(d - E[7], 0.24))
        # idle network breathing (holds are never dead): a faint glow travels across the racks'
        # lines, ToRs and uplinks; ramped in after the EXPAND, under every message glow
        br = idle_amp(now) * ramp(now, T_EXP, T_EXP + 0.40) * (0.5 + 0.5 * np.sin(TAU * now / IDLE_PER
                                                                                   - 0.8 * np.arange(8)))
        rack_g, tor_f = np.maximum(rack_g, br), np.maximum(tor_f, br)
        up_g = np.maximum(up_g, np.repeat(br, 4))

        def put(i, fill_rgb, fill_a, stroke_rgb, stroke_a):
            m = rig.rest[i].m
            yc = 0.5 * (m.points[:, 1].min() + m.points[:, 1].max())
            k = float(stage_mask(yc))
            m.fill_rgbas = np.array([[*fill_rgb, fill_a * k]])
            m.stroke_rgbas = np.array([[*stroke_rgb, stroke_a * k]])

        # boxes: dim -> powered (rack wave), CYAN compute flashes (own clock, then the SYNC
        # compute), INK on receiving a message, LIME update flash, solid as they shrink,
        # cross-fade out into the hall
        base_rgb = mix(RGB_CYAN, RGB_LIME, lime_f)[0]
        for k, i in enumerate(self.ix_srv):
            rv = recv[k]
            b_f = lerp(DIM_FILL, ON_FILL, pw[k])
            comp = max(cyan_f, af[k])
            f_rgb = mix(base_rgb, RGB_INK, max(0.7 * rv, 0.45 * pf[k]))[0]
            f_a = max(lerp(b_f, 0.85, comp), lerp(b_f, 0.95, pf[k]), lerp(b_f, 0.85, lime_f),
                      lerp(b_f, 0.95, rv), lerp(b_f, POD_OP, boost)) * a_srv
            s_rgb = mix(base_rgb, RGB_INK, max(0.6 * comp, rv, 0.6 * pf[k]))[0]
            s_a = max(lerp(DIM_STROKE, 1.0, pw[k]), comp, lime_f) * a_srv * (1 - 0.8 * boost)
            put(i, f_rgb, f_a, s_rgb, s_a)
        for i in self.ix_glyph:
            put(i, RGB_CYAN, 0.0, RGB_CYAN, 0.0)
        for c, i in enumerate(self.ix_rack):
            put(i, RGB_MAG, 0.0, mix(RGB_MAG, RGB_INK, 0.35 * rack_g[c])[0], lerp(0.35, 0.95, rack_g[c]) * a_net)
            self.rig.rest[i].m.stroke_width = lerp(1.5, 3.0, rack_g[c]) * zx
        for j, i in enumerate(self.ix_up):
            put(i, RGB_MAG, 0.0, mix(RGB_MAG, RGB_INK, 0.35 * up_g[j])[0], lerp(0.30, 1.0, up_g[j]) * a_net)
            self.rig.rest[i].m.stroke_width = lerp(1.2, 3.0, up_g[j]) * zx
        # switches: dim until the power wave reaches them (a ToR when its rack is lit, then the
        # spine tier), each with a short INK power flash
        tor_lv = np.where(now >= T_TOR_ON - EPS, 1.0, SW_DIM)
        tor_pf = np.array([decay(now - t, 0.25) for t in T_TOR_ON])
        sp_lv = 1.0 if now >= T_SPINE_ON - EPS else SW_DIM
        sp_pf = decay(now - T_SPINE_ON, 0.25)
        for c, i in enumerate(self.ix_tor):
            f = max(tor_f[c], tor_pf[c])
            put(i, mix(RGB_MAG, RGB_INK, 0.5 * f)[0], lerp(TOR_FILL * tor_lv[c], 1.0, f) * a_net,
                mix(RGB_MAG, RGB_INK, f)[0], max(tor_lv[c], f) * a_net)
        for s, i in enumerate(self.ix_spine):
            f = max(spine_f[s], sp_pf)
            put(i, mix(RGB_MAG, RGB_INK, 0.5 * f)[0], lerp(SPINE_FILL * sp_lv, 1.0, f) * a_net,
                mix(RGB_MAG, RGB_INK, f)[0], max(sp_lv, f) * a_net)

        # messages in flight (dot + trail + hot core), on the pod camera
        s, ph = self._hop_s(now)
        live = (ph >= 0) & (ph <= 6)
        if np.any(live) and a_net > 0:
            s_prev, _ = self._hop_s(now - 0.035)
            s_tail = np.maximum(s_prev, s - TRAIL)
            H = self._hop_pos(s)[live]
            T = self._hop_pos(s_tail)[live]
            H = self._lf(piv + (H - PIVOT_A) * zv, life)
            T = self._lf(piv + (T - PIVOT_A) * zv, life)
            self.m_trail.points = seg_points(T, H)
            self.m_head.points = seg_points(H, H)
            self.m_core.points = seg_points(H, H)
            self.m_trail.set_stroke(MAGENTA, width=8 * zx, opacity=0.6 * a_net)
            self.m_head.set_stroke(MAGENTA, width=18 * zx, opacity=a_net)
            self.m_core.set_stroke(INK, width=7 * zx, opacity=0.9 * a_net)
        else:
            for mob in (self.m_trail, self.m_head, self.m_core):
                mob.points = EMPTY.copy()

    # ---------------- stage B ----------------
    def _hall_cam(self, now: float):
        """(k, piv, shake, e2): camera-space point = piv + k*(q - AC) + shake for hall-world q."""
        e2 = self._zoom2(now)
        k = ZX ** (1 - e2)
        piv = lerp(PIVOT_A, AC, e2)
        sh = shake(now - T_DROP, *DROP_SHAKE, seed=6)
        return k, piv, sh, e2

    def _collapse_state(self, now: float):
        """(sc, a_col): the contents' scale about AC and opacity during the COLLAPSE."""
        if now < T_COL - EPS:
            return 1.0, 1.0
        ec = 1.0 if now >= T_SET - EPS else ease_in_out_expo(ramp(now, T_COL, T_SET))
        return self.collapse.s1 ** ec, 1.0 - ramp(ec, *self.collapse.contents_fade)

    def _clear_hall(self) -> None:
        self.hall.clear()
        self.nodes.clear()
        for ln in (*self.trunks, *self.trunks_hi):
            ln.set_stroke(opacity=0)
        self.msgs.points = EMPTY.copy()

    def _bcast_rho(self, now: float) -> float:
        """Radius of the broadcast front (the network grid overlay is drawn inside it)."""
        _, t0, t1 = PULSES[1]
        return R_PULSE * ramp(now, t0, t1)

    def _draw_hall(self, now: float) -> None:
        sc, a_col = self._collapse_state(now)
        if now < T_Z0 - EPS or a_col <= 0:
            self._clear_hall()
            return
        k, piv, sh, e2 = self._hall_cam(now)
        life = self._life(now)                    # ambient camera life, inside the (fixed) hall window
        kl = life[0]

        def cam(P):                               # hall world -> camera space
            return self._lf(piv + k * (P - AC) + sh, life)

        def col(P):                               # camera space -> screen (collapse about AC)
            return P if sc == 1.0 else (P - AC) * sc + AC

        C0 = cam(HALL_C)
        n = len(HALL_C)

        # ---- colour + opacity per square
        if now < T_DROP - EPS:
            a = np.where(POD, POD_OP * ramp(e2, 0.75, 1.0), 0.25 * ramp(e2, 0.20, 0.80))
            rgb = np.tile(RGB_CYAN, (n, 1))
            a = a * HALL_VAR
        else:
            d = now - T_DROP
            ink = np.full(n, 1.0 - ease_out_cubic(ramp(d, 0.0, 0.20)))
            if d < 0.20:
                b0 = lerp(1.0, 0.9, ease_out_cubic(d / 0.20))
            else:
                b0 = lerp(0.9, 0.55, ease_in_out_sine(ramp(now, T_DROP + 0.20, T_DROP + 0.55)))
            base = np.full(n, b0)
            mag, lime = np.zeros(n), np.zeros(n)
            for kind, t0, t1 in PULSES:
                if now < t0 - EPS:
                    continue
                if kind == "gather":
                    if now < t1 + 0.12:
                        rho = R_PULSE * (1.0 - ease_in_quad(ramp(now, t0, t1)))
                        band = np.clip(1.0 - np.abs(HALL_D - rho) / 0.30, 0, 1) * (1 - ramp(now, t1, t1 + 0.12))
                        mag = np.maximum(mag, band)
                else:
                    td = t0 + (t1 - t0) * HALL_D / R_PULSE           # the front passes a square at td
                    passed = now >= td - EPS
                    fl = np.where(passed, 1.0 - ease_out_quad_arr(np.clip((now - td) / 0.16, 0, 1)), 0.0)
                    lime = np.maximum(lime, fl)
                    base = np.where(passed, np.maximum(base, 0.64), base)   # the update is applied
            # the reduced result glows LIME in the centre between the gather and the broadcast
            hot = ramp(now, T_HOT[0], T_HOT[0] + 0.04) * (1.0 - ramp(now, T_HOT[1], T_HOT[1] + 0.15))
            if hot > 0:
                lime = np.maximum(lime, hot * np.clip((0.95 - HALL_D) / 0.45, 0, 1))
            # the hall hums (never static) and builds under 'More GPUs…'
            hum = 1.0 + HUM_AMP * ramp(now, T_DROP + 0.40, T_DROP + 0.90) * np.sin(TAU * HUM_F * now + HUM_PH)
            base = base + MORE_GAIN * ease_in_sine(ramp(now, T_CARD5, T_COL))
            rgb = mix(mix(mix(RGB_CYAN, RGB_MAG, mag), RGB_LIME, lime), RGB_INK, ink)
            a = np.maximum(base * hum, np.maximum(mag * 0.95, np.maximum(lime * 0.95, ink))) * HALL_VAR
            a = np.maximum(a, ink)                       # flashes read at full strength
        a = np.clip(a, 0, 1) * hall_mask(C0[:, 0], C0[:, 1]) * a_col
        self.hall.draw(col(C0), HALL_SQ * k * kl * sc, np.concatenate([rgb, a[:, None]], axis=1))

        # ---- trunks: dim from the drop; the overlay draws on behind the broadcast front
        if now < T_DROP - EPS:
            for ln in (*self.trunks, *self.trunks_hi):
                ln.set_stroke(opacity=0)
            self.nodes.clear()
            self.msgs.points = EMPTY.copy()
            return
        ta = lerp(0.9, TRUNK_DIM, ease_out_cubic(ramp(now - T_DROP, 0.0, 0.40))) * a_col
        rho = self._bcast_rho(now)
        w = 1.5 * min(k * kl * sc, 1.6)
        for j, ((p0, p1), vert, off) in enumerate(zip(self.trunk_rest, self.trunk_vert, self.trunk_off)):
            self._put_trunk(self.trunks[j], p0, p1, vert, cam, col, ta, w)
            if rho > abs(off):                     # the part of this line inside the front
                h = math.sqrt(rho * rho - off * off)
                if vert:
                    q0 = (p0[0], max(p0[1], AC[1] - h))
                    q1 = (p0[0], min(p1[1], AC[1] + h))
                else:
                    q0 = (max(p0[0], AC[0] - h), p0[1])
                    q1 = (min(p1[0], AC[0] + h), p0[1])
                self._put_trunk(self.trunks_hi[j], q0, q1, vert, cam, col, TRUNK_HI * a_col, 1.25 * w)
            else:
                self.trunks_hi[j].set_stroke(opacity=0)

        # ---- the junction switches pop on as the broadcast front passes them
        _, b0, b1 = PULSES[1]
        tn = b0 + (b1 - b0) * NODE_D / R_PULSE
        dn = now - tn
        on = dn >= -EPS
        if np.any(on):
            fl = np.where(on, 1.0 - ease_out_cubic_arr(np.clip(dn / 0.18, 0, 1)), 0.0)
            N0 = cam(NODE_C)
            na = np.where(on, 0.90, 0.0) * hall_mask(N0[:, 0], N0[:, 1]) * a_col
            nrgb = mix(RGB_MAG, RGB_INK, 0.25 + 0.75 * fl)
            sz = NODE_SQ * k * kl * sc * (1.0 + 0.7 * fl)
            self.nodes.draw(col(N0), sz, np.concatenate([nrgb, na[:, None]], axis=1))
        else:
            self.nodes.clear()

        # ---- messages race along the trunks from the drop, faster and faster under 'More GPUs…'
        env = ramp(now, T_DROP, T_DROP + 0.10) * a_col
        if env > 0:
            v = MSG_V0 + (MSG_V1 - MSG_V0) * ramp(now, T_CARD5, T_COL)
            s = msg_dist(now) + MSG_PHASE
            head = self._walk_pos(s)
            tail = self._walk_pos(np.maximum(s - MSG_TAIL * v / MSG_V0, MSG_PHASE))
            H = cam(np.concatenate([head, np.zeros((N_MSG, 1))], axis=1))
            T = cam(np.concatenate([tail, np.zeros((N_MSG, 1))], axis=1))
            self.msgs.points = seg_points(col(T), col(H))
            self.msgs.set_stroke(MAGENTA, width=6 * k * kl * sc, opacity=0.95 * env)
        else:
            self.msgs.points = EMPTY.copy()

    @staticmethod
    def _put_trunk(ln, p0, p1, vert, cam, col, op, w) -> None:
        """One trunk line from hall-world p0 -> p1, clipped to the hall window in camera space
        (and faded by the window across it), then collapsed."""
        q0 = cam(np.array([p0[0], p0[1], 0.0]))
        q1 = cam(np.array([p1[0], p1[1], 0.0]))
        if vert:
            lo, hi = max(min(q0[1], q1[1]), WIN_LO + 0.08), min(max(q0[1], q1[1]), WIN_HI - 0.08)
            mk = float(hall_mask(q0[0], AC[1]))
            a, b = np.array([q0[0], lo, 0.0]), np.array([q0[0], hi, 0.0])
        else:
            lo, hi = max(min(q0[0], q1[0]), WIN_X0 + 0.08), min(max(q0[0], q1[0]), WIN_X1 - 0.08)
            mk = float(hall_mask(AC[0], q0[1]))
            a, b = np.array([lo, q0[1], 0.0]), np.array([hi, q0[1], 0.0])
        if hi <= lo + 1e-4 or op * mk <= 0:
            ln.set_stroke(opacity=0)
            return
        ln.put_start_and_end_on(col(a), col(b))
        ln.set_stroke(MAGENTA, width=w, opacity=op * mk)

    @staticmethod
    def _walk_pos(s: np.ndarray) -> np.ndarray:
        e = np.floor(s / EDGE).astype(int)
        e = np.clip(e, 0, WALK_XY.shape[1] - 2)
        f = np.clip(s / EDGE - e, 0, 1)[:, None]
        r = np.arange(N_MSG)
        return WALK_XY[r, e] * (1 - f) + WALK_XY[r, e + 1] * f

    # ---------------- outline + collapse ----------------
    def _draw_outline(self, now: float) -> None:
        if now < T_OUT0 - EPS:
            self.outline.set_stroke(opacity=0)
            return
        if now < T_COL - EPS:
            a = ease_out_cubic(ramp(now, T_OUT0, T_OUT1))
            self.outline.pointwise_become_partial(self.outline_full, 0, max(a, 1e-3))
            self.outline.set_stroke(CYAN, width=3, opacity=1)
            self.outline.set_fill(CYAN, opacity=0)
            return
        self.collapse.render(now, T_COL, T_SET, T_EXIT)

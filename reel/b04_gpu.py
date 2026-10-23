"""b04 - LEVEL 2/3 - ONE GPU: thousands of tiny cores, groups of 32 obey one command (SIMT).

Script v3 'gpu' segment (12.0 s, reel/design/script_v3.json), calm motion: every idea lands on a
calm stage and is read before its animation fires; the moves are smooth 0.80 s ease_in_out_sine
(flip, die-open, push-in, pull-back, collapse); flashes, light waves, rings and slams stay.

  0.00  E03 (cpu token + HUD '16' / CORES WORKING AT ONCE). Card-flip 0.00-0.80 (replacement,
        not containment, no glitch): the CPU token turns over and lands as the GPU token (8x8
        pixel glyph). '16? Cute. / A GPU has thousands of / tiny cores.' POPs on story frame 1.
        The token rests 0.80-1.05; its 64 pixels idle-twinkle.
  1.05  The token's base opens into the H100 die (0.80 s); 3,072 tiny cells (1 cell = 1 CUDA
        core, 8x4 blocks = groups of 32) appear in a soft radial wave and settle to a dim hum.
        1.50-2.00 the HUD winds up (log-roll 16 -> 16,896, ticks + riser).
  2.00  SLAM 16,896 + LIME 'H100: 16,896 CUDA CORES' (2.00-3.80). LIGHT WAVE: every cell
        flashes INK, then CYAN, racing out from the centre behind a shock ring; 2.30-2.90 ONE
        diagonal dispatch wave of 32-cell groups, then a busy shimmer.
  3.80  Hard swap (on the frame the LIME line leaves): 'Groups of 32. / All 32 obey one command.'
        4.10-4.90 smooth push-in 4.5x on ONE group of 32, outlined as a LIME block; its bus
        charges and all 32 brighten together; lockstep steps 5.20 / 5.55 / 5.90 (an INK beam,
        all 32 recolour on the SAME frame: CYAN, LIME, then each its own value colour); payoff
        holds to 6.25, smooth pull-back to the whole die 6.25-7.05.
  7.15  Hard swap: 'Same math, / every number.' on the humming die; 8.15-8.85 six waves of 16
        groups, scattered (never a scanline), develop a shaded ball; 8.90 a burst of light from
        the ball's centre, pop, shock ring.
  9.25  Hard swap: 'For AI? / One GPU isn't enough.' over the finished ball under a slow stage
        push-in (1.0 -> 1.02); the die keeps its idle hum (no repaint).
 11.00  COLLAPSE (cut_times(12.0): 0.80 s sine) into handoff_token('gpu'), settle 11.80-11.95
        (no overshoot); kicker + card hard-cut at 11.95 with the canonical swap to E04 (gpu
        token + HUD '16,896'), static to 12.00.

Everything that moves is a pure function of self.now (one driver updater; the cell field is one
vectorised state function drawn through motif.Field, one VMobject per quantised colour); the
scene only holds until the structural switch points (collapse capture, canonical exit).
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# World layout (storyboard b04 visuals)
# --------------------------------------------------------------------------
NCOL, NROW = 64, 48
N_PIX = NCOL * NROW                                   # 3,072 cells (1 cell = 1 CUDA core)
PIX = 0.075                                           # pixel side
PIX_CORNER = 0.015
_I = np.tile(np.arange(NCOL), NROW)                   # column i of pixel k = 64*j + i
_J = np.repeat(np.arange(NROW), NCOL)                 # row j (0 = top)
PX = X_C + (_I - 31.5) * 0.10 + 0.03 * (_I // 8 - 3.5)
PY = AC[1] + (23.5 - _J) * 0.10 - 0.03 * (_J // 4 - 5.5)
PC = np.stack([PX, PY, np.zeros(N_PIX)], axis=1)
WARP_C, WARP_R = _I // 8, _J // 4                     # 8 x 12 = 96 warps of 32 (on screen: "groups of 32")
N_WARP = 96
WARP_ID = WARP_R * 8 + WARP_C
FOCUS = (WARP_C == 3) & (WARP_R == 5)                 # cols 24-31 x rows 20-23
FOCUS_W = 5 * 8 + 3
DIST = np.hypot(PX - AC[0], PY - AC[1])
DMAX = float(DIST.max())

DIE_W, DIE_H, DIE_R = 7.0, 5.4, 0.28
OUT_C = np.array([-0.765, 0.665, 0.0])                # focus-warp outline
OUT_W, OUT_H, OUT_R = 0.86, 0.46, 0.05

# --------------------------------------------------------------------------
# Timeline (beat-local seconds; script v3 'gpu' segment, 12.0 s slot, calm motion)
# --------------------------------------------------------------------------
# Cards (all hard swaps, every line POPs at full opacity; visible / reading floor 0.3 s/word + 0.8):
#   g1 '16? Cute. / A GPU has thousands of / tiny cores.'  0.017 -> 3.80  (3.783 / 3.5)
#   g2 'Groups of 32. / All 32 obey one command.'         3.80  -> 7.15  (3.35 / 3.2)
#   g3 'Same math, / every number.'                       7.15  -> 9.25  (2.10 / 2.0)
#   g4 'For AI? / One GPU isn't enough.'                  9.25  -> 11.95 (2.70 / 2.6) hard-cut by E04
#   LIME 'H100: 16,896 CUDA CORES'                        2.00  -> 3.80  (1.80; g2 lands as it leaves)
T_FLIP_END = FLIP_TIME                                # 0.80 card-flip (CALM_EASE, no glitch)
TW_T0, TW_T1 = 0.80, 1.00                             # GPU token idle twinkle fades in (the token rests)
T_DIE0, T_DIE1 = 1.05, 1.85                           # token base -> die (0.80 s, CALM_EASE)
T_GLYPH_OUT = 1.45
T_REV0, REV_SPREAD = 1.15, 0.60                       # soft radial reveal of the 3,072 cells (1.15 -> 1.87)
T_APPEAR = T_REV0 + REV_SPREAD * DIST / DMAX
T_ROLL0, T_LAND = 1.50, 2.00                          # HUD windup (log-roll), 16,896 slams
LW_DUR = 0.25                                         # 2.00 LIGHT WAVE: front reaches the corners
LW_R0 = 0.80                                          # ... already this wide on the slam frame
MULT_HOLD = 1.80                                      # 'H100: 16,896 CUDA CORES' 2.00 -> 3.80
T_C2 = T_LAND + MULT_HOLD                             # 3.80 'Groups of 32.' (on the frame the LIME line leaves)
T_OUTLINE = 3.95                                      # LIME outline marks the target group
Z_IN0 = T_C2 + 0.30                                   # 4.10 camera push-in on the focus group ...
Z_IN1 = Z_IN0 + MOVE_TIME                             # ... 4.90 (0.80 s, CALM_EASE)
STEPS = (5.20, 5.55, 5.90)                            # lockstep instructions (all 32, one frame)
Z_OUT0 = 6.25                                         # pull-back to the whole die ...
Z_OUT1 = Z_OUT0 + MOVE_TIME                           # ... 7.05 (0.80 s, CALM_EASE)
ZK = 4.5                                              # 8x4 group -> ~3.9 x 2.1 units
T_C3 = 7.15                                           # 'Same math, / every number.'
T_WAVE0, WAVE_DT = 8.15, 0.10                         # six waves of 16 groups (8.15 -> 8.85)
T_HIT = 8.90
T_C4 = 9.25                                           # 'For AI? / One GPU isn't enough.'
T_COL, T_SET, T_EXIT = cut_times(12.0)                # 11.00 / 11.80 / 11.95 (0.80 s sine, 0.15 s settle)
T_TEXT_OUT = T_EXIT                                   # kicker + card hold to the exit, then hard-cut
# 2.30-2.90 DISPATCH: after the light wave, ONE diagonal wave of groups sweeps the die
# (top-left -> bottom-right). A group fires on ONE frame (all 32 cells together).
T_DISP = (2.30,)
DISP_CROSS = 0.60                                     # the front crosses the die in this long
DISP_DECAY = 0.38                                     # fired group: bright CYAN -> its idle-busy level
DISP_HOT = 0.55                                       # INK lift on the 2 fire frames
# 9.15 -> 11.00: slow stage push-in (die + cells, never text/HUD) under 'For AI?', so the
# finished image is never a dead hold (the die keeps its idle hum); folded into the collapse.
T_DRIFT0, T_DRIFT1 = T_HIT + 0.25, T_COL
DRIFT_S1 = 1.02
# 4.90-5.20 the zoomed group waits for its instruction (while 'Groups of 32.' is read): the dim
# instruction bus charges up and all 32 cells brighten together toward the first issue
BUS_CHARGE = 0.32                                     # bus alpha 0.28 -> 0.60
READY_CHARGE = 0.13                                   # ready tint 0.15 -> 0.28 toward CYAN


def charge_at(now: float) -> float:
    """0 -> 1 (ease_in_sine) over the zoomed wait Z_IN1 -> STEPS[0]; 0 from the first issue on
    (its 2-frame INK flash hides the drop)."""
    if now >= STEPS[0] - EPS:
        return 0.0
    return ease_in_sine(ramp(now, Z_IN1, STEPS[0]))


# --------------------------------------------------------------------------
# Colours / shading (precomputed; per frame only the light direction changes)
# --------------------------------------------------------------------------
C_DIM = RGB_DIMMER
C_CY = RGB_CYAN
C_INK = RGB_INK
C_LIME = RGB_LIME
BASE_OP = 0.90
IMG_OP = 0.95
IDLE_TINT = 0.05                                      # pre-wave cells: DIMMER lifted 5% toward CYAN ...
HUM_AMP, HUM_HZ = 0.22, 3.0                           # ... plus a slow per-group breath (the dim idle hum)
BUSY0, BUSY1 = 0.18, 0.36                             # idle-busy: CYAN opacity range per warp (low, so the sweeps read)
BUSY_HZ = 8.0
DIM_K = 0.35                                          # other warps while one is in focus ...
DIM_BACK = 0.80                                       # ... and how far they come back on the pull-back (the
#   die hums at nearly full level again while 'Same math, / every number.' is read)
READY_TINT = 0.15                                     # the focus warp waiting for its instruction
STEP_RGB = (C_CY, C_LIME, None)                       # None = each thread its own pixel colour

_U = (_I - 31.5) / 21.0
_V = (23.5 - _J) / 21.0
INSIDE = _U ** 2 + _V ** 2 <= 1.0
NRM = np.stack([_U, _V, np.sqrt(np.clip(1.0 - _U ** 2 - _V ** 2, 0.0, None))], axis=1)
OUTSIDE_RGB = lerp(C_DIM[None, :], C_CY[None, :], (0.18 * np.clip((-0.2 - _V) / 1.2, 0, 1))[:, None])
LX0, LX1 = -1.6, 2.0                                  # light x range: the hit frame, then stepped per frame
SPEC_GAIN = 1.6


def shade(lx: float) -> np.ndarray:
    """(N_PIX, 3) RGB of the shaded sphere for a light at (lx, 0.7, 0.9) (storyboard formula,
    widened sweep and a hotter specular)."""
    L = np.array([lx, 0.7, 0.9])
    L /= np.linalg.norm(L)
    d = np.maximum(0.0, NRM @ L)
    H = L + np.array([0.0, 0.0, 1.0])
    H /= np.linalg.norm(H)
    sp = np.maximum(0.0, NRM @ H) ** 24
    inside = lerp(lerp(C_DIM[None, :], C_CY[None, :], (d ** 0.9)[:, None]), C_INK[None, :],
                  np.minimum(1.0, SPEC_GAIN * sp)[:, None])
    return np.where(INSIDE[:, None], inside, OUTSIDE_RGB)


SHADE0 = shade(LX0)                                   # the finished ball (no repaint in v3)

# warp-wave fronts: each warp's position along the sweep direction (0 = top-left, 1 = bottom-right)
_WX = np.array([PX[WARP_ID == w].mean() for w in range(N_WARP)])
_WY = np.array([PY[WARP_ID == w].mean() for w in range(N_WARP)])
_proj = _WX * 0.866 - _WY * 0.5
SWEEP_P = ((_proj - _proj.min()) / (_proj.max() - _proj.min()))[WARP_ID]   # per pixel
DISP_F = np.rint((np.array(T_DISP)[:, None] + DISP_CROSS * SWEEP_P[None, :]) * 60).astype(int)

# 8.90 hit flash: a burst of light from the centre of the ball, not a solid INK slab. Pixels go
# HIT_K toward HIT_RGB: 0.9 toward INK at the centre, ~0.4 at the ball's rim, ~0.05 in the
# corners, and the burst cools from INK to CYAN toward its edge.
HIT_K = 0.90 * np.exp(-(DIST / 2.4) ** 2)
HIT_RGB = lerp(C_CY[None, :], C_INK[None, :], np.exp(-(DIST / 2.0) ** 2)[:, None])


# 2.00 light wave: per-pixel arrival (linear in distance from AC beyond LW_R0), 60 fps grid
LW_T = T_LAND + LW_DUR * np.maximum(DIST - LW_R0, 0.0) / (DMAX - LW_R0)
LW_F = np.rint(LW_T * 60).astype(int)

# idle-busy: every group re-rolls its CYAN level at ~8 Hz, each on its own phase (128 steps:
# the 12.0 s slot needs ~97)
_brng = np.random.default_rng(404)
BUSY_R = _brng.random((N_WARP, 128))
BUSY_PH = _brng.random(N_WARP)
# idle hum (1.15-2.00, before the light wave): every group of 32 breathes smoothly between random
# levels at HUM_HZ, each on its own phase, so the calm die is never a dead hold
_hrng = np.random.default_rng(1104)
HUM_R = _hrng.random((N_WARP, 16))
HUM_PH = _hrng.random(N_WARP)


def hum_level(now: float) -> np.ndarray:
    """(N_WARP,) 0..1 idle-hum level of each group at now (smoothstep between random levels)."""
    x = now * HUM_HZ + HUM_PH
    b = np.clip(np.floor(x).astype(int), 0, HUM_R.shape[1] - 2)
    q = np.clip(x - b, 0.0, 1.0)
    q = q * q * (3.0 - 2.0 * q)
    w = np.arange(N_WARP)
    return lerp(HUM_R[w, b], HUM_R[w, b + 1], q)


# GPU token idle twinkle (0.80-1.45, the token rests under '16? Cute.'): the 64 pixels of the
# token glyph re-roll their brightness at TW_HZ, each on its own phase (rest opacity +- TW_AMP / 2)
TW_HZ, TW_AMP = 6.0, 0.50
_trng = np.random.default_rng(64)
TW_R = _trng.random((64, 16))
TW_PH = _trng.random(64)


def twinkle(now: float, fo: np.ndarray) -> np.ndarray:
    """(64,) glyph fill opacities at now for rest opacities fo (== fo before TW_T0)."""
    k = ease_in_out_sine(ramp(now, TW_T0, TW_T1))
    if k <= 0:
        return fo
    b = np.clip(np.floor(now * TW_HZ + TW_PH).astype(int), 0, TW_R.shape[1] - 1)
    return np.clip(fo + k * TW_AMP * (TW_R[np.arange(64), b] - 0.5), 0.15, 1.0)


def busy_level(now: float) -> np.ndarray:
    """(N_WARP,) CYAN opacity of each warp's idle-busy flicker at now."""
    b = np.clip(np.floor(now * BUSY_HZ + BUSY_PH).astype(int), 0, BUSY_R.shape[1] - 1)
    return BUSY0 + (BUSY1 - BUSY0) * BUSY_R[np.arange(N_WARP), b]


# waves of warps: stratified scatter. The die is cut into 16 regions of 2x3 warps and each
# region's 6 warps get a random order 0..5, so every wave lights exactly one warp in every
# region: 16 warps spread over the whole die per wave (the sphere develops everywhere at once).
_wrng = np.random.default_rng(7)
WAVE_OF_WARP = np.zeros(N_WARP, int)
for _rr in range(4):
    for _rc in range(4):
        _ws = [r * 8 + c for r in range(3 * _rr, 3 * _rr + 3) for c in range(2 * _rc, 2 * _rc + 2)]
        WAVE_OF_WARP[_ws] = _wrng.permutation(6)
WAVE_T = T_WAVE0 + WAVE_DT * WAVE_OF_WARP[WARP_ID]   # per pixel
WAVE_F = np.rint(WAVE_T * 60).astype(int)
IN_WAVES = ~FOCUS                                     # the focus group already computed its cells

# zoom window: zoomed pixels stay inside the die (inset), fading out near its edge
WIN = (AC[0] - DIE_W / 2 + 0.08, AC[0] + DIE_W / 2 - 0.08, AC[1] - DIE_H / 2 + 0.08, AC[1] + DIE_H / 2 - 0.08)
WIN_FADE = 0.30

_RR = RoundedRectangle(width=1.0, height=1.0, corner_radius=PIX_CORNER / PIX).points.copy()


def zoom_e(now: float) -> float:
    """Camera push-in progress (0 = rest, 1 = focus warp at 4.5x on AC)."""
    if now < Z_IN0 - EPS:
        return 0.0
    if now < Z_OUT0 - EPS:
        return CALM_EASE(ramp(now, Z_IN0, Z_IN1))
    return 1.0 - CALM_EASE(ramp(now, Z_OUT0, Z_OUT1))


def cam_at(now: float) -> tuple[float, np.ndarray, float]:
    """(k, c, e): p -> c + (p - OUT_C) * k (log-space zoom about the focus warp, which
    slides to AC)."""
    e = zoom_e(now)
    return ZK ** e, lerp(OUT_C, AC, e), e


def flash_decay(now: float, t: float, dur: float) -> float:
    """1 on the 2 story frames from t, then 1 -> 0 (ease_out_cubic) over dur; 0 before t."""
    if now < t - EPS:
        return 0.0
    if flash_frames(now, t, 2):
        return 1.0
    return 1.0 - ease_out_cubic(ramp(now, t + 2 / 60, t + 2 / 60 + dur))


def warp_fires(now: float, f: int, F: np.ndarray, decay: float) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel, for warp fires at story frames F (sweeps, N_PIX): (hot, glow). hot = on one of the
    2 fire frames of a sweep; glow = 1 on them, then 1 -> 0 (ease_out_cubic) over decay; the
    strongest sweep wins. Whole groups share F, so all 32 cells of a group change on one frame."""
    hot = np.zeros(N_PIX, bool)
    glow = np.zeros(N_PIX)
    for Fk in F:
        d = f - Fk
        if d.max() < 0:
            continue
        hot |= (d >= 0) & (d < 2)
        g = 1.0 - ease_out_cubic_arr(ramp_arr(now, (Fk + 2) / 60, (Fk + 2) / 60 + decay))
        glow = np.maximum(glow, np.where(d >= 0, g, 0.0))
    return hot, glow


def field_pop(now: float) -> float:
    if now < T_HIT - EPS:
        return 1.0
    p = ramp(now, T_HIT, T_HIT + 0.25)
    return 1.0 if p >= 1.0 else lerp(1.08, 1.0, ease_out_cubic(p))


def pixel_state(now: float):
    """(centers (N,3), sizes (N,), rgba (N,4)) of the 3,072 cells at now (pure)."""
    f = story_frame(now)
    rgb = np.tile(C_DIM, (N_PIX, 1))
    alpha = np.full(N_PIX, BASE_OP)
    size = np.ones(N_PIX)

    # radial reveal 1.15-1.87 (scale 0 -> 1 in 0.12 s) with a CYAN wavefront, settling to the
    # dim idle hum (every group of 32 breathes on its own)
    a = ramp_arr(now, T_APPEAR, T_APPEAR + 0.12)
    size *= ease_out_cubic_arr(a)
    alpha = np.where(a > 0, alpha, 0.0)
    tint = np.where(now >= T_APPEAR, 1.0 - ease_out_cubic_arr(ramp_arr(now, T_APPEAR, T_APPEAR + 0.40)), 0.0)
    idle = IDLE_TINT + HUM_AMP * hum_level(now)[WARP_ID] if now < T_LAND + LW_DUR + 0.1 else IDLE_TINT
    tint = np.maximum(tint * 0.75, idle * (a > 0))
    rgb = lerp(rgb, C_CY[None, :], tint[:, None])

    # 2.00 LIGHT WAVE: INK (2 frames) -> CYAN (0.10 s) -> idle-busy (0.40 s), front from AC
    if now >= T_LAND - EPS:
        busy = busy_level(now)[WARP_ID]
        d = now - LW_T
        hit = f >= LW_F
        q1 = ramp_arr(d, 2 / 60, 2 / 60 + 0.10)
        q2 = ease_in_out_sine_arr(ramp_arr(d, 2 / 60 + 0.10, 2 / 60 + 0.50))
        c_rgb = lerp(C_INK[None, :], C_CY[None, :], q1[:, None])
        c_a = lerp(lerp(1.0, 0.95, q1), busy, q2)
        ink = hit & (f - LW_F < 2)
        rgb = np.where(hit[:, None], c_rgb, rgb)
        alpha = np.where(hit, c_a, alpha)
        rgb[ink] = C_INK
        alpha[ink] = 1.0

    # 2.30-2.90 DISPATCH: one wave of groups sweeps the die (fired warp: INK-hot, then bright CYAN
    # easing back to its idle-busy level)
    if T_DISP[0] - EPS <= now < Z_OUT0:
        hot, glow = warp_fires(now, f, DISP_F, DISP_DECAY)
        if glow.any():
            rgb = lerp(rgb, C_CY[None, :], glow[:, None])
            alpha = lerp(alpha, 1.0, glow)
            rgb[hot] = lerp(C_CY, C_INK, DISP_HOT)

    # warp focus: the others dim; the focus warp goes 'ready', then 3 lockstep steps
    dimk = ease_out_cubic(ramp(now, Z_IN0, Z_IN1)) * (1.0 - DIM_BACK * ease_in_out_sine(ramp(now, Z_OUT0, Z_OUT1)))
    if dimk > 0:
        alpha = np.where(FOCUS, alpha, alpha * lerp(1.0, DIM_K, dimk))
        rk = ease_out_cubic(ramp(now, Z_IN0, Z_IN0 + 0.25))
        ready = lerp(C_DIM, C_CY, READY_TINT + READY_CHARGE * charge_at(now))
        rgb[FOCUS] = lerp(rgb[FOCUS], ready[None, :], rk)
        alpha[FOCUS] = lerp(alpha[FOCUS], BASE_OP, rk)
    step = -1
    for k, s in enumerate(STEPS):
        if now >= s - EPS:
            step = k
    if step >= 0:
        ts = STEPS[step]
        target = SHADE0[FOCUS] if STEP_RGB[step] is None else STEP_RGB[step][None, :]
        if flash_frames(now, ts, 2):
            rgb[FOCUS] = C_INK
            alpha[FOCUS] = 1.0
        else:
            q = ease_out_cubic(ramp(now, ts + 2 / 60, ts + 2 / 60 + 0.12))
            rgb[FOCUS] = lerp(C_INK[None, :], target, q)
            alpha[FOCUS] = lerp(1.0, IMG_OP, q)
        pp = ramp(now, ts, ts + 0.15)
        if pp < 1.0:
            size[FOCUS] *= lerp(1.15, 1.0, ease_out_cubic(pp))

    # waves of groups 8.15-8.85 (scattered): INK for 2 frames, settles to the image in 0.15 s
    img = SHADE0
    if now >= T_WAVE0 - EPS:
        on = IN_WAVES & (f >= WAVE_F)
        ink = on & (f - WAVE_F < 2)
        sp = ease_out_cubic_arr(ramp_arr(now, WAVE_T + 2 / 60, WAVE_T + 2 / 60 + 0.15))
        sett = on & ~ink
        rgb[sett] = lerp(C_INK[None, :], img[sett], sp[sett][:, None])
        alpha[sett] = lerp(1.0, IMG_OP, sp[sett])
        rgb[ink] = C_INK
        alpha[ink] = 1.0
        if now >= T_HIT - EPS:                        # after the hit: the live image everywhere
            rgb[FOCUS] = img[FOCUS]
            alpha[FOCUS] = IMG_OP

    # 8.90 HIT: a 2-frame burst of light from the centre (HIT_K / HIT_RGB: INK at the core, CYAN
    # and faint toward the corners, so the image still shows through), then it fades back
    if now >= T_HIT - EPS:
        if flash_frames(now, T_HIT, 2):
            rgb = lerp(rgb, HIT_RGB, HIT_K[:, None])
            alpha = lerp(alpha, 1.0, HIT_K)
        else:
            lift = 0.70 * (1.0 - ease_out_cubic(ramp(now, T_HIT + 2 / 60, T_HIT + 0.28)))
            if lift > 0:
                rgb = lerp(rgb, HIT_RGB, (lift * HIT_K)[:, None])

    # camera: push-in on the focus warp (pixels outside the die window fade / are culled)
    k, c, e = cam_at(now)
    s = field_pop(now)
    if e > 0:
        centers = c + (PC - OUT_C) * k
        sz = PIX * size * k
        hs = sz / 2
        din = np.minimum(np.minimum(centers[:, 0] - WIN[0], WIN[1] - centers[:, 0]),
                         np.minimum(centers[:, 1] - WIN[2], WIN[3] - centers[:, 1])) - hs
        ce = clip01(e * 3.0)
        fade = np.where(din < 0, 0.0, lerp(1.0, np.clip(din / WIN_FADE, 0.0, 1.0), ce))
        alpha = alpha * fade
    elif s != 1.0:
        centers = (PC - AC) * s + AC
        sz = PIX * size * s
    else:
        centers = PC
        sz = PIX * size
    return centers, sz, np.column_stack([rgb, alpha])


class _Snap:
    """Rest geometry + style of one VMobject (for the card-flip stretch)."""
    __slots__ = ("m", "p", "fr", "sr", "sw")

    def __init__(self, m):
        self.m, self.p = m, m.points.copy()
        self.fr, self.sr, self.sw = m.fill_rgbas.copy(), m.stroke_rgbas.copy(), float(m.stroke_width)


class B04Gpu(BeatScene):
    def beat(self):
        # ---------------- entry: E03 (cpu token + HUD '16' / CORES WORKING AT ONCE) ----------------
        st = enter(self)
        hud = st.hud

        # ---------------- card-flip tokens (cpu -> gpu) ----------------
        self.tok_cpu = handoff_token("cpu")
        self.tok_gpu = handoff_token("gpu")
        self._snap_cpu = [_Snap(m) for m in self.tok_cpu.family_members_with_points()]
        self._snap_gpu_base = _Snap(self.tok_gpu.base)
        self._snap_gpu_glyph = [_Snap(m) for m in self.tok_gpu.glyph.family_members_with_points()]
        assert len(self._snap_gpu_glyph) == 64
        self._glyph_fo = np.array([s.fr[0, 3] for s in self._snap_gpu_glyph])

        # ---------------- world at rest ----------------
        self.die_rest = RoundedRectangle(width=DIE_W, height=DIE_H, corner_radius=DIE_R).move_to(AC).points.copy()
        self.die = RoundedRectangle(width=DIE_W, height=DIE_H, corner_radius=DIE_R, stroke_color=CYAN,
                                    stroke_width=3, fill_color=CYAN, fill_opacity=0.04).move_to(AC)
        assert len(self.die.points) == len(self.tok_gpu.base.points)
        self.field = Field(_RR, pool=360, qc=24, qa=32, fixed=True)   # the radial hit burst needs ~340
        self.outline = RoundedRectangle(width=OUT_W, height=OUT_H, corner_radius=OUT_R, stroke_color=LIME,
                                        stroke_width=3, fill_opacity=0).move_to(OUT_C)
        self.outline_rest = self.outline.points.copy()
        self.beam = Line(LEFT, RIGHT, stroke_color=INK, stroke_width=7)
        self.beam.cap_style = CapStyleType.ROUND
        # 3.45: the light wave races out behind a shock ring that rides its front (same speed)
        lw_r0, lw_r1 = LW_R0 + 0.12, DMAX + 0.45
        lw_dur = LW_DUR * (lw_r1 - lw_r0) / (DMAX - LW_R0)
        self.lw_ring = ShockRing(self, T_LAND, AC, r0=lw_r0, r1=lw_r1, w0=9.0, w1=2.0, op0=0.85,
                                 dur=lw_dur, color=INK, ease=linear)
        self.ring = ShockRing(self, T_HIT, AC, r0=0.5, r1=4.0, w0=8.0, op0=1.0, dur=0.50, color=INK)

        # ---------------- text: kicker + cards (hard swaps; the kicker and the last card hold
        # through the collapse and settle and hard-cut at T_EXIT with the canonical E04 swap) ----
        track = TextTrack(self)
        track.on(kick("LEVEL 2/3 · ONE GPU"), 0.0, T_TEXT_OUT, exit="cut")
        c1 = card([("16? Cute.", "H84", INK, {"16?": CYAN}),
                   ("A GPU has thousands of", "S56", INK),
                   ("tiny cores.", "S56", INK, {"tiny cores.": CYAN})])
        track.on(c1, 1 / 60, T_C2, enter="pop", exit="cut")         # POPs on story frame 1 (frame 0 = E03)
        c2 = card([("Groups of 32.", "H96", INK, {"32": CYAN}),
                   ("All 32 obey one command.", "S56", INK)])
        track.on(c2, T_C2, T_C3, enter="pop", exit="cut")
        c3 = card([("Same math,", "H84", INK),
                   ("every number.", "H84", INK, {"every number.": CYAN})])
        track.on(c3, T_C3, T_C4, enter="pop", exit="cut")
        c4 = card([("For AI?", "H96", INK, {"AI": CYAN}),
                   ("One GPU isn't enough.", "S56", INK)])
        track.on(c4, T_C4, T_TEXT_OUT, enter="pop", exit="cut")

        # ---------------- HUD: '16' holds, the windup roll runs into the slam ----------------
        hud.roll(T_ROLL0, T_LAND, 16, 16896, ease=ease_in_quad)
        hud.slam(T_LAND, "16,896")
        hud.mult(T_LAND, COUNTER_LINES["B04Gpu"], hold=MULT_HOLD)   # names the kind of core
        assert hud.number_at(T_EXIT - 1e-4)[0] == "16,896" and hud.label_at(T_EXIT - 1e-4)[0] == CALC_LABEL

        # ---- z-order: dust, entry ring, die, cells, outline, beam, rings, flip tokens, text, HUD ----
        self.remove(st.token)                          # the flip token now draws E03
        add_behind(self, self.die, self.field, self.outline, self.beam, self.lw_ring, self.ring,
                   self.tok_cpu, self.tok_gpu)
        entry_ring(self)
        self.add(track)
        self.remove(hud)
        self.add(hud)
        self.collapse = None
        self._draw(0.0)
        drive(self, self._draw)
        # the finished image holds under a slow push-in (stage only; added after the driver so it
        # sees this frame's drawing); released right before the collapse captures the die
        self.drift = drift(self, VGroup(self.die, self.field), T_DRIFT0, T_DRIFT1, s1=DRIFT_S1, dy=0.0,
                           ease=ease_in_out_sine)

        # ---------------- audio (each cue sits with the visual it scores) ----------------
        self.cue("whoosh", dt=round(T_FLIP_END / 2, 4), dur=T_FLIP_END, gain=0.6, direction="up",
                 note="smooth card-flip (no glitch)")
        self.cue("hit", dt=round(T_FLIP_END / 2, 4), gain=0.6, note="GPU turns face up")
        self.cue("whoosh", dt=round((T_DIE0 + T_DIE1) / 2, 4), dur=round(T_DIE1 - T_DIE0, 4), direction="up",
                 note="die opens")
        self.cue("shimmer", dt=T_DIE0, dur=0.80, gain=0.35, note="low shimmer: 3,072 cells appear")
        self.cue("ticks_burst", dt=T_ROLL0, dur=round(T_LAND - T_ROLL0 - 0.04, 4), rate0=8, rate1=60,
                 note="odometer windup (accelerating roll 16 -> 16,896)")
        self.cue("riser", dt=T_ROLL0, dur=round(T_LAND - T_ROLL0, 4), gain=0.7, note="windup into the slam")
        self.cue("hit", dt=T_LAND, note="16,896 lands (tier 1, with the sub drop)")
        self.cue("sub_drop", dt=T_LAND, dur=0.9)
        self.cue("shimmer", dt=T_LAND, dur=0.55, gain=0.6, note="light wave across the die")
        for t in T_DISP:
            self.cue("tick", dt=t, gain=0.45, freq=2600, note="dispatch wave of groups sweeps the die")
        self.cue("whoosh", dt=round((Z_IN0 + Z_IN1) / 2, 4), dur=MOVE_TIME, gain=0.7, direction="up",
                 note="smooth push-in on one group of 32")
        for s in STEPS:
            self.cue("tick", dt=s, gain=0.9, freq=3800, pan=0, note="lockstep step: all 32 cells")
        self.cue("whoosh", dt=round((Z_OUT0 + Z_OUT1) / 2, 4), dur=MOVE_TIME, gain=0.7, direction="down",
                 note="smooth pull back")
        self.cue("riser", dt=round(T_WAVE0 - 0.40, 4), dur=0.40, gain=0.8, note="short riser into the waves")
        self.cue("ticks_burst", dt=T_WAVE0, dur=0.55, rate0=10, rate1=10, note="one tick per wave of groups")
        self.cue("hit", dt=T_HIT, gain=0.9, note="image complete (tier 2)")
        self.cue("shimmer", dt=T_HIT, dur=0.90, note="burst of light from the ball")
        self.cue("whoosh", dt=round((T_COL + T_SET) / 2, 4), dur=COLLAPSE_TIME, direction="down",
                 note="collapse (soft whoosh down)")
        self.cue("tick", dt=round(T_SET, 4), gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_COL)
        self._draw(T_COL)                              # exact frozen state at the light of T_COL ...
        self.drift.render_at(T_COL)                    # ... in the drifted pose,
        self.drift.release()                           # which the collapse now starts from
        for m in (self.outline, self.beam):
            m.points = EMPTY.copy()
        self.collapse = COLLAPSE(self.die, self.field, "gpu")
        self.add(self.collapse.glyph)
        hold_floor(self, T_EXIT)
        exit_to(self)                                  # canonical E04, static to 12.00

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.collapse is not None:
            self.collapse.render(now, T_COL, T_SET, T_EXIT)
            return
        self._draw_flip(now)
        self._draw_die(now)
        self.field.draw(*pixel_state(now))
        self._draw_focus(now)

    # ---- 0.00-1.45: smooth card-flip (0.80 s, CALM_EASE: the token turns over about its vertical
    # axis, CPU face -> GPU face at the half turn, no flash / glitch), the token rests, then the
    # glyph zooms/fades as the die opens
    def _draw_flip(self, now: float) -> None:
        flash = False
        e = CALM_EASE(ramp(now, 0.0, T_FLIP_END))
        th = math.pi * e
        sx, sy = max(abs(math.cos(th)), 0.02), 1.0 + 0.06 * math.sin(th)
        cpu_vis, gpu_vis = e < 0.5, e >= 0.5
        if e <= 0.0 or e >= 1.0:
            sx, sy = 1.0, 1.0
        stretch = np.array([sx, sy, 1.0])
        exact = sx == 1.0 and sy == 1.0
        # cpu token
        for s in self._snap_cpu:
            self._apply_snap(s, cpu_vis, exact, stretch, flash)
        # gpu token base (hidden once the die takes over)
        self._apply_snap(self._snap_gpu_base, gpu_vis and now < T_DIE0 - EPS, exact, stretch, flash)
        # gpu glyph: stretch during the flip; idle twinkle through the calm hold (from TW_T0); zooms
        # with the opening die and fades 1.10 -> 1.29
        ga = 1.0 - ramp(now, T_DIE0, T_GLYPH_OUT)
        fo = twinkle(now, self._glyph_fo) if gpu_vis and exact else None
        if gpu_vis and now >= T_DIE0 - EPS:
            de = CALM_EASE(ramp(now, T_DIE0, T_DIE1))
            z = lerp(1.0, 2.4, de)
            for i, s in enumerate(self._snap_gpu_glyph):
                m = s.m
                if ga <= 0:
                    m.points = EMPTY.copy()
                    continue
                m.points = (s.p - AC) * z + AC
                v = s.fr.copy()
                v[:, 3] = fo[i] * ga
                m.fill_rgbas = v
        else:
            for i, s in enumerate(self._snap_gpu_glyph):
                self._apply_snap(s, gpu_vis, exact, stretch, flash)
                if fo is not None and fo[i] != s.fr[0, 3]:
                    v = s.fr.copy()
                    v[:, 3] = fo[i]
                    s.m.fill_rgbas = v

    @staticmethod
    def _apply_snap(s: _Snap, vis: bool, exact: bool, stretch, flash: bool) -> None:
        m = s.m
        if not vis:
            m.points = EMPTY.copy()
            return
        m.points = s.p.copy() if exact else (s.p - AC) * stretch + AC
        if flash:
            m.fill_rgbas = rgba(INK, 0.55 if s.fr[0, 3] > 0 else 0.0).reshape(1, 4)
            m.stroke_rgbas = rgba(INK, 1.0 if s.sr[0, 3] > 0 else 0.0).reshape(1, 4)
        else:
            m.fill_rgbas, m.stroke_rgbas = s.fr.copy(), s.sr.copy()
        m.stroke_width = s.sw

    # ---- die: token base -> 7.0 x 5.4 die (0.80 s, CALM_EASE); on the 8.90 hit an INK w8 stroke
    # (2 frames) + pop
    def _draw_die(self, now: float) -> None:
        d = self.die
        if now < T_DIE0 - EPS:
            d.points = EMPTY.copy()
            return
        e = CALM_EASE(ramp(now, T_DIE0, T_DIE1))
        tb = self._snap_gpu_base
        pts = lerp(tb.p, self.die_rest, e)
        s = field_pop(now)
        d.points = pts if s == 1.0 else (pts - AC) * s + AC
        hk = flash_decay(now, T_HIT, 0.40)
        lw = flash_decay(now, T_LAND, 0.30) * 0.6       # the light wave also charges the die edge
        k = max(hk, lw)
        stroke_c = interpolate_color(CYAN, INK, k) if k > 0 else CYAN
        d.stroke_rgbas = rgba(stroke_c, 1.0).reshape(1, 4)
        d.stroke_width = lerp(TOKEN_STROKE, 3.0, e) + 5.0 * hk + 2.0 * lw
        d.fill_rgbas = rgba(CYAN, lerp(TOKEN_FILL, 0.04, e) + 0.08 * hk).reshape(1, 4)

    # ---- focus group: LIME outline + the INK instruction beam (both ride the camera)
    def _draw_focus(self, now: float) -> None:
        o, bm = self.outline, self.beam
        k, c, e = cam_at(now)
        vis_in = ease_out_cubic(ramp(now, T_OUTLINE, T_OUTLINE + 0.30))
        vis_out = 1.0 - ease_in_out_sine(ramp(now, Z_OUT0, Z_OUT0 + 0.40))
        a = vis_in * vis_out
        if now < T_OUTLINE - EPS or a <= 0:
            o.points = EMPTY.copy()
            bm.points = EMPTY.copy()
            return
        fl = max(flash_decay(now, s, 0.22) for s in STEPS)
        sc = lerp(1.35, 1.0, vis_in) * (1.0 + 0.05 * fl)
        local = (self.outline_rest - OUT_C) * sc
        o.points = c + local * k
        o.stroke_rgbas = rgba(interpolate_color(LIME, INK, 0.6 * fl) if fl > 0 else LIME, a).reshape(1, 4)
        o.stroke_width = lerp(3.0, 5.0, e) + 4.0 * fl
        # the instruction beam: a dim bus over the warp while zoomed, full INK on every issue
        ba = clip01((e - 0.85) / 0.15) * vis_out
        if ba <= 0:
            bm.points = EMPTY.copy()
            return
        top = c[1] + (OUT_H / 2) * k + 0.22
        hw = (OUT_W / 2) * k + 0.06
        bm.points = seg([c[0] - hw, top, 0.0], [c[0] + hw, top, 0.0])
        # while the warp waits for its first instruction the dim bus charges up (no dead hold);
        # the first issue's full-INK flash hides the drop back to the resting level
        base = 0.28 + BUS_CHARGE * charge_at(now)
        bm.stroke_rgbas = rgba(INK, ba * lerp(base, 1.0, fl)).reshape(1, 4)
        bm.stroke_width = 7.0 + 5.0 * fl

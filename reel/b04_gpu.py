"""b04 - SCALE 3/6 - ONE GPU: warps of 32, every pixel in parallel (SIMT).

The CPU token (E03) card-flips (replacement, not containment) into a GPU token, whose
base opens into the H100 die (0.51, after a beat on the GPU token). 3,072 pixels (1 pixel =
1 thread, 8x4 blocks = warps of 32) appear in a radial wave while the HUD rolls 256 -> 16,896
('66x ONE CPU'). On the 1.55 slam the LIGHT WAVE races out from the centre (every pixel INK,
then CYAN); 1.85-2.67 three DISPATCH waves of warps sweep the die (diagonal fronts: each warp
fires on one frame, INK-hot, then bright CYAN easing back to a low idle-busy hum, every warp
flickering on its own at 8 Hz). 2.70-3.00 the camera pushes in 4.5x on one warp; while
'32 threads, 1 instruction.' is read its bus charges and its 32 threads brighten together, then run three
instructions in lockstep (3.60 / 3.90 / 4.20: an INK instruction beam fires, all 32 recolour
on the SAME frame: CYAN, LIME, then each thread its own pixel colour), then the camera pulls
back straight into six waves of 16 warps scattered over the whole die (stratified, never a
top-to-bottom scanline), which develop a shaded sphere; 5.20 hit: a burst of light from the
ball's centre (INK core, CYAN and faint toward the corners, never a solid slab), pop, shock
ring. 5.50-6.52 EVERY FRAME: four more frames are rendered (the light steps across the ball),
each by a wave of warps sweeping the die (every warp repaints its 32 pixels on one frame
with a CYAN-white write-flash), so the image visibly updates warp by warp, under a slow
1.0 -> 1.02 stage push-in that carries the finished image to the collapse. The die
collapses (cut_times) into handoff_token('gpu') (E04); kicker + headline hold to the exit
and hard-cut with the canonical swap.

Everything that moves is a pure function of self.now (one driver updater; the pixel
field is one vectorised state function drawn through motif.Field, one VMobject per
quantised colour); the scene only holds until the structural switch points (collapse
capture, canonical exit).
"""

from style import *
from motif import *

# --------------------------------------------------------------------------
# World layout (storyboard b04 visuals)
# --------------------------------------------------------------------------
NCOL, NROW = 64, 48
N_PIX = NCOL * NROW                                   # 3,072 threads
PIX = 0.075                                           # pixel side
PIX_CORNER = 0.015
_I = np.tile(np.arange(NCOL), NROW)                   # column i of pixel k = 64*j + i
_J = np.repeat(np.arange(NROW), NCOL)                 # row j (0 = top)
PX = X_C + (_I - 31.5) * 0.10 + 0.03 * (_I // 8 - 3.5)
PY = AC[1] + (23.5 - _J) * 0.10 - 0.03 * (_J // 4 - 5.5)
PC = np.stack([PX, PY, np.zeros(N_PIX)], axis=1)
WARP_C, WARP_R = _I // 8, _J // 4                     # 8 x 12 = 96 warps of 32
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
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
# Pacing (7.5 s slot): every idea is SEQUENCED - its card lands while the die is calm, the
# viewer gets a read before the motion it describes, then the payoff holds before the next idea.
# Only the gaps between motions grew; every motion keeps its old duration (flip 0.24, die 0.54,
# roll 0.90, light wave 0.25, push-in 0.30, steps 0.30 apart, pull-back 0.25, waves 0.50, ...).
#   0.00-0.24  card-flip on the cut; 'Now, a GPU.' POPs on frame 1, '... 16,896 math lanes' 0.05
#   0.51       (after a 0.27 s look at the GPU token) the die opens; 0.65 reveal + roll (read:
#              0.35 s after the sub-line is fully in, 0.60 s after it lands)
#   1.55       16,896 SLAMS + '66× ONE CPU' (1.6 s fully visible); light wave, 3 dispatch sweeps;
#              the payoff holds 1.05 s before the next idea (c1 hard-cuts at 2.60, push-in 2.70)
#   2.60       hard swap: 'Warps of 32.' + '32 threads, 1 instruction.' POP together (with the focus
#              outline 2.63, before the push-in); the steps wait 1.0 s (3.60). Reading: c1
#              0.017 -> 2.60 (2.58 s), c2 2.60 -> 5.20 (2.60 s); no fade/CARD_IN between the cards
#   5.20       hit: 'Every pixel, / in parallel.' POPs on the finished image; it re-renders 4 frames
#              (5.50-6.52) under a slow stage push-in, holds, collapses 6.933, text cut at 7.45
T_FLIP_MID = 0.12
T_FLIP_END = 0.24
FLASH_FRAMES = (7, 8)                                 # 2-frame INK flash at 0.12 (60 fps grid)
T_DIE0, T_DIE1 = 0.51, 1.05                           # token base -> die
T_GLYPH_OUT = 0.70
T_ROLL0, T_LAND = 0.65, 1.55                          # HUD roll (accelerating), 16,896 slams
T_APPEAR = T_ROLL0 + 0.55 * DIST / DMAX               # radial reveal from AC, with the roll
LW_DUR = 0.25                                         # 1.55 LIGHT WAVE: front reaches the corners
LW_R0 = 0.80                                          # ... already this wide on the slam frame
MULT_HOLD = 1.70                                      # '66× ONE CPU': fully visible 1.65 -> 3.25
T_C1_OUT = 2.60                                       # 'Now, a GPU.' card hard-cuts to 'Warps of 32.'
T_OUTLINE = 2.63                                      # focus outline telegraphs the target
Z_IN0, Z_IN1 = 2.70, 3.00                             # camera push-in on the focus warp
T_C2 = T_C1_OUT                                       # 'Warps of 32.' + '32 threads, 1 instruction.' POP
Z_OUT0, Z_OUT1 = 4.40, 4.65                           # pull-back, straight into the waves
ZK = 4.5                                              # 8x4 warp -> ~3.9 x 2.1 units
STEPS = (3.60, 3.90, 4.20)                            # lockstep instructions (all 32, one frame)
T_WAVE0, WAVE_DT = 4.65, 0.10
T_HIT = 5.20
T_COL, T_SET, T_EXIT = cut_times(7.5)                 # 6.933 / 7.333 / 7.45
T_TEXT_OUT = T_EXIT                                   # kicker + headline hold to the exit, then hard-cut
# 1.85-2.67 DISPATCH: after the light wave, waves of warps keep sweeping the die (diagonal
# fronts, top-left -> bottom-right). A warp fires on ONE frame (all 32 threads together).
T_DISP = (1.85, 2.11, 2.37)
DISP_CROSS = 0.30                                     # a front crosses the die in this long
DISP_DECAY = 0.38                                     # fired warp: bright CYAN -> its idle-busy level
DISP_HOT = 0.55                                       # INK lift on the 2 fire frames
# 5.50-6.52 FRAMES: 'every pixel, every frame'. The light steps to a new position per frame and
# each frame is re-rendered by a wave of warps sweeping the die (each warp repaints its 32 pixels
# on one frame with a short CYAN-white write-flash), so the image visibly updates warp by warp.
# The hit's burst + shock ring get 0.30 s on their own first.
T_FRAMES = (5.50, 5.76, 6.02, 6.28)
# 5.45 -> 6.933: slow stage push-in (die + pixels, never text/HUD) under 'Every pixel, in parallel.',
# so the finished image is never a dead hold; folded into the collapse at T_COL.
T_DRIFT0, T_DRIFT1 = T_HIT + 0.25, T_COL
DRIFT_S1 = 1.02
# 3.00-3.60 the zoomed warp waits for its instruction (while '32 threads, 1 instruction.' is read):
# the dim instruction bus charges up and all 32 threads brighten together toward the first issue
BUS_CHARGE = 0.32                                     # bus alpha 0.28 -> 0.60
READY_CHARGE = 0.13                                   # ready tint 0.15 -> 0.28 toward CYAN


def charge_at(now: float) -> float:
    """0 -> 1 (ease_in_sine) over the zoomed wait Z_IN1 -> STEPS[0]; 0 from the first issue on
    (its 2-frame INK flash hides the drop)."""
    if now >= STEPS[0] - EPS:
        return 0.0
    return ease_in_sine(ramp(now, Z_IN1, STEPS[0]))
FR_CROSS = 0.24
FR_HOT = 0.45                                         # write-flash: CYAN lifted this far toward INK
FR_GLOW, FR_DECAY = 0.55, 0.12                        # then a short CYAN afterglow

# --------------------------------------------------------------------------
# Colours / shading (precomputed; per frame only the light direction changes)
# --------------------------------------------------------------------------
C_DIM = RGB_DIMMER
C_CY = RGB_CYAN
C_INK = RGB_INK
C_LIME = RGB_LIME
BASE_OP = 0.90
IMG_OP = 0.95
IDLE_TINT = 0.07                                      # pre-wave threads: DIMMER lifted 7% toward CYAN
BUSY0, BUSY1 = 0.18, 0.36                             # idle-busy: CYAN opacity range per warp (low, so the sweeps read)
BUSY_HZ = 8.0
DIM_K = 0.35                                          # other warps while one is in focus ...
DIM_BACK = 0.55                                       # ... and how far they come back on the pull-back
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


SHADE0 = shade(LX0)
# frame k (k = 1..4) is lit from FR_LX[k]: the light steps LX0 -> LX1 across the ball
FR_LX = [lerp(LX0, LX1, (k + 1) / len(T_FRAMES)) for k in range(len(T_FRAMES))]
SHADES = np.stack([SHADE0, *[shade(lx) for lx in FR_LX]])      # (1 + frames, N_PIX, 3)

# warp-wave fronts: each warp's position along the sweep direction (0 = top-left, 1 = bottom-right)
_WX = np.array([PX[WARP_ID == w].mean() for w in range(N_WARP)])
_WY = np.array([PY[WARP_ID == w].mean() for w in range(N_WARP)])
_proj = _WX * 0.866 - _WY * 0.5
SWEEP_P = ((_proj - _proj.min()) / (_proj.max() - _proj.min()))[WARP_ID]   # per pixel
DISP_F = np.rint((np.array(T_DISP)[:, None] + DISP_CROSS * SWEEP_P[None, :]) * 60).astype(int)
FR_F = np.rint((np.array(T_FRAMES)[:, None] + FR_CROSS * SWEEP_P[None, :]) * 60).astype(int)

# 5.20 hit flash: a burst of light from the centre of the ball, not a solid INK slab. Pixels go
# HIT_K toward HIT_RGB: 0.9 toward INK at the centre, ~0.4 at the ball's rim, ~0.05 in the
# corners, and the burst cools from INK to CYAN toward its edge.
HIT_K = 0.90 * np.exp(-(DIST / 2.4) ** 2)
HIT_RGB = lerp(C_CY[None, :], C_INK[None, :], np.exp(-(DIST / 2.0) ** 2)[:, None])


# 1.55 light wave: per-pixel arrival (linear in distance from AC beyond LW_R0), 60 fps grid
LW_T = T_LAND + LW_DUR * np.maximum(DIST - LW_R0, 0.0) / (DMAX - LW_R0)
LW_F = np.rint(LW_T * 60).astype(int)

# idle-busy: every warp re-rolls its CYAN level at ~8 Hz, each on its own phase
_brng = np.random.default_rng(404)
BUSY_R = _brng.random((N_WARP, 64))
BUSY_PH = _brng.random(N_WARP)


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
IN_WAVES = ~FOCUS                                     # the focus warp already computed its pixels

# zoom window: zoomed pixels stay inside the die (inset), fading out near its edge
WIN = (AC[0] - DIE_W / 2 + 0.08, AC[0] + DIE_W / 2 - 0.08, AC[1] - DIE_H / 2 + 0.08, AC[1] + DIE_H / 2 - 0.08)
WIN_FADE = 0.30

_RR = RoundedRectangle(width=1.0, height=1.0, corner_radius=PIX_CORNER / PIX).points.copy()


def zoom_e(now: float) -> float:
    """Camera push-in progress (0 = rest, 1 = focus warp at 4.5x on AC)."""
    if now < Z_IN0 - EPS:
        return 0.0
    if now < Z_OUT0 - EPS:
        return ease_in_out_quart(ramp(now, Z_IN0, Z_IN1))
    return 1.0 - ease_in_out_quart(ramp(now, Z_OUT0, Z_OUT1))


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
    strongest sweep wins. Whole warps share F, so all 32 threads of a warp change on one frame."""
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


def image_at(f: int) -> np.ndarray:
    """(N_PIX, 3) image each pixel shows at story frame f: SHADE0 until its warp repaints it for
    frame k (then SHADES[k])."""
    k = (f >= FR_F).sum(axis=0)
    return SHADES[k, np.arange(N_PIX)]


def field_pop(now: float) -> float:
    if now < T_HIT - EPS:
        return 1.0
    p = ramp(now, T_HIT, T_HIT + 0.25)
    return 1.0 if p >= 1.0 else lerp(1.08, 1.0, ease_out_cubic(p))


def pixel_state(now: float):
    """(centers (N,3), sizes (N,), rgba (N,4)) of the 3,072 threads at now (pure)."""
    f = story_frame(now)
    rgb = np.tile(C_DIM, (N_PIX, 1))
    alpha = np.full(N_PIX, BASE_OP)
    size = np.ones(N_PIX)

    # radial reveal 0.65-1.32 (scale 0 -> 1 in 0.12 s) with a CYAN wavefront
    a = ramp_arr(now, T_APPEAR, T_APPEAR + 0.12)
    size *= ease_out_cubic_arr(a)
    alpha = np.where(a > 0, alpha, 0.0)
    tint = np.where(now >= T_APPEAR, 1.0 - ease_out_cubic_arr(ramp_arr(now, T_APPEAR, T_APPEAR + 0.40)), 0.0)
    tint = np.maximum(tint * 0.75, IDLE_TINT * (a > 0))
    rgb = lerp(rgb, C_CY[None, :], tint[:, None])

    # 1.55 LIGHT WAVE: INK (2 frames) -> CYAN (0.10 s) -> idle-busy (0.40 s), front from AC
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

    # 1.85-2.67 DISPATCH: waves of warps sweep the die (fired warp: INK-hot, then bright CYAN
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

    # waves of warps 4.65-5.15 (scattered): INK for 2 frames, settles to the image in 0.15 s
    img = image_at(f)
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

    # 5.20 HIT: a 2-frame burst of light from the centre (HIT_K / HIT_RGB: INK at the core, CYAN
    # and faint toward the corners, so the image still shows through), then it fades back
    if now >= T_HIT - EPS:
        if flash_frames(now, T_HIT, 2):
            rgb = lerp(rgb, HIT_RGB, HIT_K[:, None])
            alpha = lerp(alpha, 1.0, HIT_K)
        else:
            lift = 0.70 * (1.0 - ease_out_cubic(ramp(now, T_HIT + 2 / 60, T_HIT + 0.28)))
            if lift > 0:
                rgb = lerp(rgb, HIT_RGB, (lift * HIT_K)[:, None])

    # 5.50-6.52 FRAMES: each warp repaints its 32 pixels with the next frame (image_at) on one
    # frame (a 2-frame CYAN-white write-flash, then a short CYAN afterglow) as the front sweeps
    if now >= T_FRAMES[0] - EPS:
        hot, glow = warp_fires(now, f, FR_F, FR_DECAY)
        if glow.any():
            w = FR_GLOW * glow
            rgb = lerp(rgb, C_CY[None, :], w[:, None])
            alpha = lerp(alpha, 1.0, w)
            rgb[hot] = lerp(C_CY, C_INK, FR_HOT)
            alpha[hot] = 1.0

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
        # ---------------- entry: E03 (cpu token + HUD '256') ----------------
        st = enter(self)
        hud = st.hud

        # ---------------- card-flip tokens (cpu -> gpu) ----------------
        self.tok_cpu = handoff_token("cpu")
        self.tok_gpu = handoff_token("gpu")
        self._snap_cpu = [_Snap(m) for m in self.tok_cpu.family_members_with_points()]
        self._snap_gpu_base = _Snap(self.tok_gpu.base)
        self._snap_gpu_glyph = [_Snap(m) for m in self.tok_gpu.glyph.family_members_with_points()]

        # ---------------- world at rest ----------------
        self.die_rest = RoundedRectangle(width=DIE_W, height=DIE_H, corner_radius=DIE_R).move_to(AC).points.copy()
        self.die = RoundedRectangle(width=DIE_W, height=DIE_H, corner_radius=DIE_R, stroke_color=CYAN,
                                    stroke_width=3, fill_color=CYAN, fill_opacity=0.04).move_to(AC)
        assert len(self.die.points) == len(self.tok_gpu.base.points)
        self.field = Field(_RR, pool=360, qc=24, qa=32, fixed=True)   # the radial hit burst needs ~340
        self.outline = RoundedRectangle(width=OUT_W, height=OUT_H, corner_radius=OUT_R, stroke_color=INK,
                                        stroke_width=3, fill_opacity=0).move_to(OUT_C)
        self.outline_rest = self.outline.points.copy()
        self.beam = Line(LEFT, RIGHT, stroke_color=INK, stroke_width=7)
        self.beam.cap_style = CapStyleType.ROUND
        self.ring = ShockRing(self, T_HIT, AC, r0=0.5, r1=4.0, w0=8.0, op0=1.0, dur=0.50, color=INK)

        # ---------------- text (kicker + headline in on the cut; they hold through the collapse and
        # settle and hard-cut at T_EXIT with the canonical E04 swap, so no frame near a cut is bare) ----
        track = TextTrack(self)
        track.on(kick("SCALE 3/6 · ONE GPU"), 0.0, T_TEXT_OUT, exit="cut")
        c1 = card([("Now, a GPU.", "H96", INK),
                   ("H100: 16,896 math lanes", "S56", INK, {"16,896 math lanes": CYAN})])
        track.on(c1[0], 1 / 60, T_C1_OUT, enter="pop", exit="cut")   # POPs on story frame 1 (frame 0
        track.on(c1[1], 0.05, T_C1_OUT, exit="cut")                  # stays E03); hard swap at 2.60
        c2 = card([("Warps of 32.", "H96", INK, {"32": CYAN}),
                   ("32 threads, 1 instruction.", "S56", INK)])
        track.on(c2[0], T_C2, T_HIT, enter="pop", exit="cut")        # both lines POP on the swap;
        track.on(c2[1], T_C2, T_HIT, enter="pop", exit="cut")        # hard swap on the 5.20 hit
        c3 = card([("Every pixel,", "H84", INK), ("in parallel.", "H84", CYAN)])
        track.on(c3, T_HIT, T_TEXT_OUT, enter="pop", exit="cut")

        # ---------------- HUD: the roll accelerates into the slam (with its ticks) ----------------
        hud.roll(T_ROLL0, T_LAND, 256, 16896, ease=ease_in_quad)
        hud.slam(T_LAND, "16,896")
        hud.mult(T_LAND, "66× ONE CPU", hold=MULT_HOLD)  # word line: what the jump means

        # ---------------- z-order: dust, ring, die, pixels, outline, beam, hit ring, flip tokens, text, HUD ----
        self.remove(st.token)                          # the flip token now draws E03
        add_behind(self, self.die, self.field, self.outline, self.beam, self.ring, self.tok_cpu, self.tok_gpu)
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
        self.cue("glitch", dur=0.12, note="card-flip")
        self.cue("hit", dt=T_FLIP_MID, gain=0.8, note="GPU swapped in")
        self.cue("ticks_burst", dt=T_ROLL0, dur=0.86, rate0=8, rate1=60, note="odometer (accelerating roll)")
        self.cue("whoosh", dt=round(T_DIE0 + 0.27, 4), dur=0.55, direction="up", note="die opens")
        self.cue("hit", dt=T_LAND, gain=0.8, note="16,896 lands")
        self.cue("shimmer", dt=T_LAND, dur=0.55, gain=0.6, note="light wave across the die")
        for t in T_DISP:
            self.cue("tick", dt=t, gain=0.45, freq=2600, note="dispatch wave of warps sweeps the die")
        self.cue("whoosh", dt=round(Z_IN0 - 0.05, 4), dur=0.40, direction="up", note="push-in on one warp")
        for s in STEPS:
            self.cue("tick", dt=s, gain=0.9, freq=3800, pan=0, note="lockstep step: all 32 threads")
        self.cue("riser", dt=round(Z_OUT0 - 0.15, 4), dur=0.40, note="into the waves")
        self.cue("whoosh", dt=round(Z_OUT0 - 0.03, 4), dur=0.30, direction="down", note="pull back")
        self.cue("ticks_burst", dt=T_WAVE0, dur=0.55, rate0=10, rate1=10, note="one tick per warp wave")
        self.cue("hit", dt=T_HIT, note="image complete")
        self.cue("sub_drop", dt=T_HIT, dur=0.9)
        self.cue("shimmer", dt=T_HIT, dur=1.40, note="rides the re-rendered frames")
        for t in T_FRAMES:
            self.cue("tick", dt=t, gain=0.55, freq=3200, note="next frame: a wave of warps repaints the die")
        self.cue("whoosh", dt=round(T_COL - 0.10, 4), dur=0.55, direction="down", note="collapse")
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
        exit_to(self)                                  # canonical E04, static to 7.50

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.collapse is not None:
            self.collapse.render(now, T_COL, T_SET, T_EXIT)
            return
        self._draw_flip(now)
        self._draw_die(now)
        self.field.draw(*pixel_state(now))
        self._draw_focus(now)

    # ---- 0.00-0.70: card-flip, then the token glyph zooms/fades as the die opens
    def _draw_flip(self, now: float) -> None:
        f = story_frame(now)
        flash = f in FLASH_FRAMES
        if now < T_FLIP_MID - EPS:
            e = ease_in_cubic(clip01(now / T_FLIP_MID))
            sx, sy = 1.0 - 0.98 * e, 1.0 + 0.06 * e
            cpu_vis, gpu_vis = True, False
        else:
            e = ease_out_cubic(clip01((now - T_FLIP_MID) / (T_FLIP_END - T_FLIP_MID)))
            sx, sy = 0.02 + 0.98 * e, 1.06 - 0.06 * e
            if e >= 1.0:
                sx, sy = 1.0, 1.0
            cpu_vis, gpu_vis = False, True
        stretch = np.array([sx, sy, 1.0])
        exact = sx == 1.0 and sy == 1.0
        # cpu token
        for s in self._snap_cpu:
            self._apply_snap(s, cpu_vis, exact, stretch, flash)
        # gpu token base (hidden once the die takes over)
        self._apply_snap(self._snap_gpu_base, gpu_vis and now < T_DIE0 - EPS, exact, stretch, flash)
        # gpu glyph: stretch during the flip; zooms with the opening die and fades 0.51 -> 0.70
        ga = 1.0 - ramp(now, T_DIE0, T_GLYPH_OUT)
        if gpu_vis and now >= T_DIE0 - EPS:
            de = ease_out_expo(ramp(now, T_DIE0, T_DIE1))
            z = lerp(1.0, 2.4, de)
            for s in self._snap_gpu_glyph:
                m = s.m
                if ga <= 0:
                    m.points = EMPTY.copy()
                    continue
                m.points = (s.p - AC) * z + AC
                v = s.fr.copy()
                v[:, 3] *= ga
                m.fill_rgbas = v
        else:
            for s in self._snap_gpu_glyph:
                self._apply_snap(s, gpu_vis, exact, stretch, flash)

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

    # ---- die: token base -> 7.0 x 5.4 die; on the 5.20 hit an INK w8 stroke (2 frames) + pop
    def _draw_die(self, now: float) -> None:
        d = self.die
        if now < T_DIE0 - EPS:
            d.points = EMPTY.copy()
            return
        e = ease_out_expo(ramp(now, T_DIE0, T_DIE1))
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

    # ---- focus warp: INK outline + the instruction beam (both ride the camera)
    def _draw_focus(self, now: float) -> None:
        o, bm = self.outline, self.beam
        k, c, e = cam_at(now)
        vis_in = ease_out_expo(ramp(now, T_OUTLINE, T_OUTLINE + 0.22))
        vis_out = 1.0 - ease_in_out_sine(ramp(now, Z_OUT0, Z_OUT0 + 0.20))
        a = vis_in * vis_out
        if now < T_OUTLINE - EPS or a <= 0:
            o.points = EMPTY.copy()
            bm.points = EMPTY.copy()
            return
        fl = max(flash_decay(now, s, 0.22) for s in STEPS)
        sc = lerp(1.35, 1.0, vis_in) * (1.0 + 0.05 * fl)
        local = (self.outline_rest - OUT_C) * sc
        o.points = c + local * k
        o.stroke_rgbas = rgba(INK, a).reshape(1, 4)
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

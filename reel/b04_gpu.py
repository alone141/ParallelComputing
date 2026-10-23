"""b04 - SCALE 3/6 - ONE GPU: warps of 32, every pixel in parallel (SIMT).

The CPU token (E03) card-flips (replacement, not containment) into a GPU token, whose
base opens into the H100 die. 3,072 pixels (1 pixel = 1 thread, 8x4 blocks = warps of
32) appear in a radial wave while the HUD log-rolls 256 -> 16,896 (x66). One warp is
singled out and runs three instructions in lockstep (all 32 pixels change on the same
frame); then six waves of 16 warps shade a sphere top to bottom; on the 4.55 hit the
light sweeps across the ball (all 3,072 colours recomputed every frame). The die then
collapses into handoff_token('gpu') (E04).

Everything that moves is a pure function of self.now (one driver updater, the pixel
field is one vectorised redraw); the scene only holds until the structural switch
points (collapse capture, canonical exit).
"""

from style import *
from motif import *
import math

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
WARP_ID = WARP_R * 8 + WARP_C
BAND = _J // 8                                        # wave k = block rows 2k, 2k+1
FOCUS = (WARP_C == 3) & (WARP_R == 5)                 # cols 24-31 x rows 20-23
DIST = np.hypot(PX - AC[0], PY - AC[1])
T_APPEAR = 0.40 + 0.55 * DIST / DIST.max()            # radial reveal from AC

DIE_W, DIE_H, DIE_R = 7.0, 5.4, 0.28
OUT_C = np.array([-0.765, 0.665, 0.0])                # focus-warp outline
OUT_W, OUT_H, OUT_R = 0.86, 0.46, 0.05

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
T_FLIP_MID = 0.12
T_FLIP_END = 0.24
FLASH_FRAMES = (7, 8)                                 # 2-frame INK flash at 0.12 (60 fps grid)
T_DIE0, T_DIE1 = 0.26, 0.80                           # token base -> die
T_GLYPH_OUT = 0.45
T_ROLL0, T_LAND = 0.40, 1.30                          # HUD log-roll, 16,896 lands
T_RIPPLE1 = 1.85
TWINKLE = (1.48, 2.58)                                # idle warps blink (as whole warps)
T_FOCUS, T_FOCUS_END = 2.70, 3.90
STEPS = (2.95, 3.30, 3.65)                            # lockstep instructions
T_WAVE0, WAVE_DT = 4.00, 0.10
T_HIT = 4.55
T_SWEEP1 = 5.50
T_COLLAPSE = 5.50
T_SETTLE = 6.15
T_SETTLED = 6.32
T_EXIT = 380 / 60                                     # 6.333: canonical E04 (on the 30 & 60 fps grids)

THETA0, THETA1 = 2.6, 0.5

# --------------------------------------------------------------------------
# Colours / shading (precomputed; per frame only the light direction changes)
# --------------------------------------------------------------------------
C_DIM = np.array(DIMMER.to_rgb())
C_CY = np.array(CYAN.to_rgb())
C_INK = np.array(INK.to_rgb())
BASE_OP = 0.90
IMG_OP = 0.95
IDLE_TINT = 0.07                                      # idle threads: DIMMER lifted 7% toward CYAN (reads on a phone)

_U = (_I - 31.5) / 21.0
_V = (23.5 - _J) / 21.0
INSIDE = _U ** 2 + _V ** 2 <= 1.0
NRM = np.stack([_U, _V, np.sqrt(np.clip(1.0 - _U ** 2 - _V ** 2, 0.0, None))], axis=1)
OUTSIDE_RGB = lerp(C_DIM[None, :], C_CY[None, :], (0.18 * np.clip((-0.2 - _V) / 1.2, 0, 1))[:, None])


def shade(theta: float) -> np.ndarray:
    """(N_PIX, 3) RGB of the shaded sphere for light angle theta (storyboard formula)."""
    L = np.array([math.cos(theta), 0.7, 0.9])
    L /= np.linalg.norm(L)
    d = np.maximum(0.0, NRM @ L)
    H = L + np.array([0.0, 0.0, 1.0])
    H /= np.linalg.norm(H)
    sp = np.maximum(0.0, NRM @ H) ** 24
    inside = lerp(lerp(C_DIM[None, :], C_CY[None, :], (d ** 0.9)[:, None]), C_INK[None, :],
                  np.minimum(1.0, 1.2 * sp)[:, None])
    return np.where(INSIDE[:, None], inside, OUTSIDE_RGB)


SHADE0 = shade(THETA0)


def theta_at(now: float) -> float:
    if now < T_HIT:
        return THETA0
    return lerp(THETA0, THETA1, ease_in_out_sine(ramp(now, T_HIT, T_SWEEP1)))


# idle twinkle: whole warps blink faint CYAN (seeded schedule)
_trng = np.random.default_rng(404)
TW_N = 34
TW_T = np.sort(_trng.uniform(*TWINKLE, TW_N))
TW_W = _trng.permutation(96)[:TW_N]
TW_DUR = 0.34

_RR = RoundedRectangle(width=1.0, height=1.0, corner_radius=PIX_CORNER / PIX).points.copy()


def ease_out_cubic_arr(x):
    return 1.0 - (1.0 - x) ** 3


def hold_floor(scene, t: float) -> None:
    """Hold until the last frame at or before t (exact at 60 fps; 30-fps previews round down)."""
    n = int(math.floor(t * config.frame_rate + 1e-6)) - scene._frames
    if n > 0:
        scene.wait((n - 0.5) / config.frame_rate)


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
        self.pixels = VGroup(*[VMobject(stroke_width=0, fill_color=DIMMER, fill_opacity=0) for _ in range(N_PIX)])
        for k, sq in enumerate(self.pixels):
            sq.points = _RR * PIX + PC[k]
        self.outline = RoundedRectangle(width=OUT_W, height=OUT_H, corner_radius=OUT_R, stroke_color=INK,
                                        stroke_width=3, fill_opacity=0).move_to(OUT_C)
        self.outline_rest = self.outline.points.copy()
        self._pix_rgba = np.zeros((N_PIX, 4))
        self._pix_pts = np.zeros((N_PIX,) + _RR.shape)

        # ---------------- text ----------------
        track = TextTrack(self)
        track.on(kick("SCALE 3/6 · ONE GPU"), 0.20, 5.95)
        c1 = card([("Now, a GPU.", "H96", INK),
                   ("H100: 16,896 math lanes", "S56", INK, {"16,896 math lanes": CYAN})])
        track.on(c1[0], 0.25, 2.45)                    # gone by 2.65, when 'Warps of 32.' enters
        track.on(c1[1], 0.40, 2.45)
        c2 = card([("Warps of 32.", "H96", INK, {"32": CYAN}), ("One instruction.", "S56", INK)])
        track.on(c2[0], 2.65, T_HIT, exit="cut")        # hard swap on the 4.55 hit
        track.on(c2[1], 2.80, T_HIT, exit="cut")
        c3 = card([("Every pixel,", "H84", INK), ("in parallel.", "H84", CYAN)])
        track.on(c3, T_HIT, 5.95, enter="pop")

        # ---------------- HUD ----------------
        hud.roll(T_ROLL0, T_LAND, 256, 16896)
        hud.slam(T_LAND, "16,896", mult="×66")

        # ---------------- z-order: dust, ring, die, pixels, outline, flip tokens, text, HUD ----
        self.remove(st.token)                          # the flip token now draws E03
        add_behind(self, self.die, self.pixels, self.outline, self.tok_cpu, self.tok_gpu)
        entry_ring(self)
        self.add(track)
        self.remove(hud)
        self.add(hud)
        self.collapse = None
        self._draw(0.0)
        driver = Mobject()
        driver.add_updater(lambda m, dt: self._draw(self.now))
        self.add(driver)

        # ---------------- audio ----------------
        self.cue("glitch", dur=0.12, note="card-flip")
        self.cue("hit", dt=T_FLIP_MID, gain=0.8, note="GPU swapped in")
        self.cue("ticks_burst", dt=T_ROLL0, dur=0.90, rate0=8, rate1=60, note="odometer")
        self.cue("whoosh", dt=0.53, dur=0.55, direction="up", note="die opens")
        self.cue("hit", dt=T_LAND, gain=0.8, note="16,896 lands")
        for s in STEPS:
            self.cue("tick", dt=s, gain=0.6, freq=3800, pan=0, note="lockstep step")
        self.cue("riser", dt=3.60, dur=0.40, note="into the waves")
        self.cue("ticks_burst", dt=T_WAVE0, dur=0.55, rate0=10, rate1=10, note="one tick per warp wave")
        self.cue("hit", dt=T_HIT, note="image complete")
        self.cue("sub_drop", dt=T_HIT, dur=0.9)
        self.cue("shimmer", dt=T_HIT, dur=0.95, note="rides the light sweep")
        self.cue("whoosh", dt=5.83, dur=0.65, direction="down", note="collapse")
        self.cue("tick", dt=T_SETTLE, gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_COLLAPSE)
        self._draw(T_COLLAPSE)                         # exact frozen state at theta(5.50)
        self.outline.points = np.zeros((0, 3))
        self.collapse = COLLAPSE(self.die, self.pixels, "gpu")
        self.add(self.collapse.glyph)
        hold_floor(self, T_EXIT)
        exit_to(self)                                  # canonical E04, static to 6.50

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.collapse is not None:
            if now < T_SETTLE - EPS:
                self.collapse.at(ease_in_out_expo(ramp(now, T_COLLAPSE, T_SETTLE)))
            else:
                self.collapse.settle(ramp(now, T_SETTLE, T_SETTLED))
            return
        self._draw_flip(now)
        self._draw_die(now)
        self._draw_pixels(now)
        self._draw_outline(now)

    # ---- 0.00-0.45: card-flip, then the token glyph zooms/fades as the die opens
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
        # gpu glyph: stretch during the flip; zooms with the opening die and fades 0.26 -> 0.45
        ga = 1.0 - ramp(now, T_DIE0, T_GLYPH_OUT)
        if gpu_vis and now >= T_DIE0 - EPS:
            de = ease_out_expo(ramp(now, T_DIE0, T_DIE1))
            z = lerp(1.0, 2.4, de)
            for s in self._snap_gpu_glyph:
                m = s.m
                if ga <= 0:
                    m.points = np.zeros((0, 3))
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
            m.points = np.zeros((0, 3))
            return
        m.points = s.p.copy() if exact else (s.p - AC) * stretch + AC
        if flash:
            m.fill_rgbas = rgba(INK, 0.55 if s.fr[0, 3] > 0 else 0.0).reshape(1, 4)
            m.stroke_rgbas = rgba(INK, 1.0 if s.sr[0, 3] > 0 else 0.0).reshape(1, 4)
        else:
            m.fill_rgbas, m.stroke_rgbas = s.fr.copy(), s.sr.copy()
        m.stroke_width = s.sw

    # ---- die: token base -> 7.0 x 5.4 die; INK stroke flash + pop on the 4.55 hit
    def _draw_die(self, now: float) -> None:
        d = self.die
        if now < T_DIE0 - EPS:
            d.points = np.zeros((0, 3))
            return
        e = ease_out_expo(ramp(now, T_DIE0, T_DIE1))
        tb = self._snap_gpu_base
        pts = lerp(tb.p, self.die_rest, e)
        s = self._field_pop(now)
        d.points = pts if s == 1.0 else (pts - AC) * s + AC
        hk = 1.0 - ease_out_cubic(ramp(now, T_HIT, T_HIT + 0.40)) if now >= T_HIT - EPS else 0.0
        stroke_c = interpolate_color(CYAN, INK, hk) if hk > 0 else CYAN
        d.stroke_rgbas = rgba(stroke_c, 1.0).reshape(1, 4)
        d.stroke_width = lerp(lerp(TOKEN_STROKE, 3.0, e), 5.0, hk)
        d.fill_rgbas = rgba(CYAN, lerp(TOKEN_FILL, 0.04, e) + 0.05 * hk).reshape(1, 4)

    @staticmethod
    def _field_pop(now: float) -> float:
        if now < T_HIT - EPS:
            return 1.0
        p = ramp(now, T_HIT, T_HIT + 0.25)
        return 1.0 if p >= 1.0 else lerp(1.04, 1.0, ease_out_cubic(p))

    # ---- the 3,072-pixel field: one vectorised redraw
    def _draw_pixels(self, now: float) -> None:
        f = story_frame(now)
        rgb = np.tile(C_DIM, (N_PIX, 1))
        alpha = np.full(N_PIX, BASE_OP)
        size = np.ones(N_PIX)

        # radial reveal 0.40-1.10 (scale 0 -> 1 in 0.12 s) with a CYAN wavefront
        a = ramp_arr(now, T_APPEAR, T_APPEAR + 0.12)
        size *= ease_out_cubic_arr(a)
        alpha = np.where(a > 0, alpha, 0.0)
        tint = np.where(now >= T_APPEAR, 1.0 - ease_out_cubic_arr(ramp_arr(now, T_APPEAR, T_APPEAR + 0.40)), 0.0)
        tint = np.maximum(tint * 0.75, IDLE_TINT * (a > 0))
        # 16,896 lands (1.30): one ripple of light across the field
        if T_LAND - EPS <= now < T_RIPPLE1:
            p = ramp(now, T_LAND, T_RIPPLE1)
            rho = (DIST.max() + 0.4) * ease_out_cubic(p)
            tint = np.maximum(tint, 0.55 * (1 - p) * np.exp(-((DIST - rho) / 0.32) ** 2))
        # idle: whole warps blink (lockstep units), 1.48-2.84
        if TWINKLE[0] - EPS <= now < TWINKLE[1] + TW_DUR:
            d = now - TW_T
            env = np.where((d >= 0) & (d < TW_DUR),
                           ramp_arr(d, 0.0, 0.05) * (1 - ease_out_cubic_arr(ramp_arr(d, 0.05, TW_DUR))), 0.0)
            wt = np.zeros(96)
            np.maximum.at(wt, TW_W, 0.42 * env * (1 - ramp(now, 2.60, T_FOCUS + 0.02)))
            tint = np.maximum(tint, wt[WARP_ID])
        rgb = lerp(rgb, C_CY[None, :], tint[:, None])

        # warp focus 2.70-3.90: the others drop to 0.4, the focus warp runs 3 lockstep steps
        dim = ease_out_cubic(ramp(now, T_FOCUS, T_FOCUS + 0.18)) * (1 - ease_in_out_sine(ramp(now, T_FOCUS_END, T_FOCUS_END + 0.15)))
        if dim > 0:
            alpha = np.where(FOCUS, alpha, lerp(BASE_OP, 0.40, dim))
        step = -1
        for k, s in enumerate(STEPS):
            if now >= s - EPS:
                step = k
        if step >= 0:
            if step == 0:
                rgb[FOCUS] = C_INK
                alpha[FOCUS] = 1.0
            elif step == 1:
                rgb[FOCUS] = C_CY
                alpha[FOCUS] = IMG_OP
            else:
                rgb[FOCUS] = SHADE0[FOCUS]
                alpha[FOCUS] = IMG_OP
            pp = ramp(now, STEPS[step], STEPS[step] + 0.15)
            if pp < 1.0:
                size[FOCUS] *= lerp(1.15, 1.0, ease_out_cubic(pp))

        # waves of warps 4.00-4.50: band k flashes INK for 2 frames, settles to the image in 0.15 s
        img = SHADE0 if now < T_HIT else shade(theta_at(now))
        tw = T_WAVE0 + WAVE_DT * BAND
        fw = np.rint(tw * 60).astype(int)
        on = now >= tw - EPS
        if np.any(on):
            ink = on & (f - fw < 2)
            sp = ease_out_cubic_arr(ramp_arr(now, tw + 2 / 60, tw + 2 / 60 + 0.15))
            sett = on & ~ink
            rgb[sett] = lerp(C_INK[None, :], img[sett], sp[sett][:, None])
            alpha[sett] = lerp(1.0, IMG_OP, sp[sett])
            rgb[ink] = C_INK
            alpha[ink] = 1.0

        # 4.55 hit: the whole field pops 1.04 -> 1 and lifts toward INK
        s = self._field_pop(now)
        if now >= T_HIT - EPS:
            lift = 0.16 * (1 - ease_out_cubic(ramp(now, T_HIT, T_HIT + 0.28)))
            if lift > 0:
                rgb = lerp(rgb, C_INK[None, :], lift)
        centers = PC if s == 1.0 else (PC - AC) * s + AC
        sz = PIX * size * s

        pts = self._pix_pts
        np.multiply(_RR[None, :, :], sz[:, None, None], out=pts)
        pts += centers[:, None, :]
        rg = self._pix_rgba
        rg[:, :3] = rgb
        rg[:, 3] = alpha
        for k, sq in enumerate(self.pixels.submobjects):
            sq.points = pts[k]
            sq.fill_rgbas = rg[k:k + 1]

    # ---- focus-warp outline
    def _draw_outline(self, now: float) -> None:
        o = self.outline
        if now < T_FOCUS - EPS or now >= T_FOCUS_END + 0.15 - EPS:
            o.points = np.zeros((0, 3))
            return
        ein = ease_out_expo(ramp(now, T_FOCUS, T_FOCUS + 0.22))
        aout = 1 - ease_in_out_sine(ramp(now, T_FOCUS_END, T_FOCUS_END + 0.15))
        sc = lerp(1.35, 1.0, ein)
        fl = 0.0
        for s in STEPS:
            if now >= s - EPS:
                fl = 1 - ease_out_cubic(ramp(now, s, s + 0.22))
        sc *= 1.0 + 0.06 * fl
        o.points = self.outline_rest.copy() if sc == 1.0 else (self.outline_rest - OUT_C) * sc + OUT_C
        o.stroke_rgbas = rgba(INK, ein * aout).reshape(1, 4)
        o.stroke_width = 3.0 + 3.0 * fl

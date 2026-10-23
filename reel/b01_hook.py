"""b01 HOOK: one at a time vs all at once (4.5 s, film 0.0-4.5).

Frame 0 is handoff_state('LOOP') (== b09's last frame): the x1.4 amber worker with its soft
halo is mid-crawl on idx 32 (10-cell bright trail, #2E3445 idle field), under the
'PARALLEL COMPUTING' kicker, which is part of the LOOP frame (b09 POPs it in at 2.40): on
screen from frame 0 with no fade-in, it stays to the cut. The worker crawls on the integer
frame clock to idx 49 while the camera creeps in about it (1.0 -> 1.035, ease_in_quad) and,
with the audio riser (0.45), the idle field charges (outlines DIM -> CYAN 0.45). Both stop
dead on the 8-frame micro-freeze (headline jitter, silence); at 1.00 s every other cell
lights up in a radial INK->CYAN wave (white radial burst + CYAN shock ring from the worker)
while the lone worker stays AMBER, drawn on the grid's top layer with its halo (above the
1.3x flashing neighbours); the halo fades out during the push-in, before the token lands.
The camera then never sits still: a slow linear push-in about the worker (1.035 -> 1.06,
1.30-3.40) carries straight into the ease_in_out_expo push-in (3.40-4.33), with a second,
softer diagonal light pulse on the open-loop card (2.35). The worker morphs into the cyan
E01 token and settles (cut_times(4.5): land 4.333, settle 0.12 s, static E01 f267-f269).
The kicker + open-loop card stay fully visible until the token is static (hard cut at 4.45),
so the cut never shows an empty, text-less frame before b02's frame 0.

Everything per frame is a pure function of self.now (one vectorised grid updater + the
motif text tracks), so 30 fps previews and 60 fps finals agree.
"""

from style import *
from motif import *

# ---------------------------------------------------------------- timing (local s)
F = 1 / 60
W_IDX = 49                      # where the crawl stops (worker cell)
W_PT = hook_center(W_IDX)       # (-0.52, 2.32)
T_FREEZE0 = 52 * F              # f52..f59 micro-freeze (headline jitter)
T_HIT = 1.00                    # f60 the parallel wave
WAVE_SPAN = 0.35                # flash front reaches the farthest cell 0.35 s after the hit
FLASH_SETTLE = 0.25             # INK flash -> LIT
T_HOOK_OUT = 2.15               # hook cards leave (gone by 2.35, no overlap with the next card)
T_CARDS = 2.35                  # open-loop card in (+ the second light pulse)
T_RO_OUT = 2.85                 # '320 WORKERS · 1 STEP' out
CREEP_S = 1.035                 # camera creeps in about the worker during the crawl (1.0 -> 1.035,
                                # ease_in_quad from f0, stops dead on the micro-freeze, holds)
T_CHARGE0 = 0.45                # visual riser (with the audio riser): the idle field charges,
CHARGE_STROKE = rgba(CYAN, 0.45)   # outline only: DIM 0.35 -> CYAN 0.45 by the freeze (the fill
                                # stays idle grey: the cyan fill is the hit's reveal)
T_DRIFT0, T_DRIFT1 = 1.30, 3.40 # slow push-in about the worker (linear) ...
DRIFT_S = 1.06                  # ... CREEP_S -> 1.06, handing its speed to the push-in
T_PUSH0 = T_DRIFT1              # push-in on the worker (ease_in_out_expo)
_, T_PUSH1, T_SETTLE1 = cut_times(4.5)   # 4.3333 (f260) lands, 4.45 (f267) settled
T_FADE0, T_FADE1 = 3.55, 4.20   # non-worker cells fade out
T_MORPH0 = 3.70                 # worker -> token style
T_TXT_OUT = T_SETTLE1           # kicker + open-loop card: full until the token is static, then cut

PULSE_SPAN = 0.26               # second light pulse: front crosses the grid diagonal in 0.26 s ...
PULSE_DUR = 0.16                # ... each cell 0 -> peak -> 0 over 0.16 s (whole sweep ~0.42 s)
PULSE_OP = 0.97                 # LIT opacity at the pulse peak (shimmer 0.62 +- 0.25)

SHAKE_SEED = 1
JITTER_SEED = 4

# ---------------------------------------------------------------- precomputed grid data
_NOT_W = np.arange(HOOK_N) != W_IDX
_DIST = np.linalg.norm(HOOK_CENTERS - W_PT, axis=1)
_T_FLASH = T_HIT + WAVE_SPAN * _DIST / _DIST[_NOT_W].max()     # per-cell flash time
_PHASE = 1.2 * (HOOK_CENTERS[:, 0] - HOOK_CENTERS[:, 1])        # diagonal sheen phase
_DIAG = HOOK_CENTERS[:, 0] - HOOK_CENTERS[:, 1]                 # top-left -> bottom-right
_T_PULSE = T_CARDS + PULSE_SPAN * (_DIAG - _DIAG.min()) / (_DIAG.max() - _DIAG.min())
_BASE49 = HookGrid.crawl_arrays(W_IDX)
_W_SW = float(_BASE49["sw"][W_IDX])                             # the worker's own stroke width
_INK1 = rgba(INK, 1.0)
_CYAN_RGB = rgba(CYAN)[:3]
_LIT_RGB_HOT = lerp(_CYAN_RGB, rgba(INK)[:3], 0.30)             # pulse peak: cyan lifted toward INK
_FLASH_SW = 1.5
_V_DRIFT = (DRIFT_S - CREEP_S) / (T_DRIFT1 - T_DRIFT0)          # drift speed carried into the push
_IDLE49 = np.arange(HOOK_N) > W_IDX                             # cells still IDLE at the hit


_HOT = 0.42                     # worker morph passes through hot INK (no muddy olive midpoint)


def _morph_fill(mo: float) -> np.ndarray:
    """Worker fill AMBER op1 -> (hot, INK-ish) -> CYAN op TOKEN_FILL (the E01 token fill)."""
    hot = rgba(INK, 1.0)
    if mo <= _HOT:
        return lerp(HOOK_WORKER_FILL, lerp(HOOK_WORKER_FILL, hot, 0.75), mo / _HOT)
    return lerp(lerp(HOOK_WORKER_FILL, hot, 0.75), rgba(CYAN, TOKEN_FILL), (mo - _HOT) / (1 - _HOT))


def _cam_scale(now: float) -> float:
    """Camera scale about the worker before the push-in: exactly 1.0 on frame 0, creeps in to
    CREEP_S (ease_in_quad) by the micro-freeze, holds through the hit, then linear to 1.06."""
    if now < T_FREEZE0 - EPS:
        return 1.0 + (CREEP_S - 1.0) * ease_in_quad(ramp(now, 0.0, T_FREEZE0))
    return CREEP_S + (DRIFT_S - CREEP_S) * ramp(now, T_DRIFT0, T_DRIFT1)


def _charge(now: float) -> float:
    """Visual riser 0..1: the idle field charges T_CHARGE0 -> freeze (ease_in_quad), then holds."""
    return ease_in_quad(ramp(now, T_CHARGE0, T_FREEZE0))


def _charge_idle(stroke: np.ndarray, sel: np.ndarray, c: float) -> None:
    """Charge the outlines of the IDLE cells `sel` by c (in place)."""
    if c > 0.0 and np.any(sel):
        stroke[sel] = lerp(HOOK_IDLE_STROKE, CHARGE_STROKE, c)


def _pulse(now: float) -> np.ndarray:
    """Second light pulse (2.35): per-cell 0 -> 1 -> 0 bump as the diagonal front passes."""
    p = (now - _T_PULSE) / PULSE_DUR
    out = np.zeros(HOOK_N)
    on = (p > 0.0) & (p < 1.0)
    if np.any(on):
        q = p[on]
        out[on] = np.where(q < 0.35, 1.0 - (1.0 - q / 0.35) ** 2,
                           1.0 - ease_in_out_sine_arr((q - 0.35) / 0.65))
    return out


def finish(scene) -> None:
    """Fill the rest of the slot frame-exactly (1 frame -> wait(1/fps), no sub-frame wait)."""
    n = int(round(beat_duration(type(scene).__name__) * config.frame_rate)) - scene._frames
    if n == 1:
        scene.wait(1 / config.frame_rate)
    elif n > 1:
        scene.hold_until(beat_duration(type(scene).__name__))


class B01Hook(BeatScene):
    def beat(self):
        st = enter(self)                         # LOOP: grid(32) + card + readout, exact
        grid = st.grid
        self.remove(st.card, st.readout, st.kicker)   # re-drawn below by the frame-exact tracks

        # ---------------------------------------------------------------- grid
        def render_grid(m, dt):
            now = self.now
            if now < T_HIT - EPS:                # crawl, then the micro-freeze (idx 49)
                w = b01_worker_idx(sframe(self))
                s, c = _cam_scale(now), _charge(now)
                if s == 1.0 and c == 0.0:        # frame 0: exactly the LOOP frame
                    if m.worker_idx != w:
                        m.set_crawl(w)
                    return
                m.worker_idx = None              # creep + charge: the crawl styles, re-posed
                a = HookGrid.crawl_arrays(w)
                _charge_idle(a["stroke"], np.arange(HOOK_N) > w, c)
                not_w = np.arange(HOOK_N) != w
                a["sw"][not_w] *= s              # (the worker keeps its own stroke, as below)
                a["size"] = a["size"] * s
                m.draw(centers=W_PT + s * (HOOK_CENTERS - W_PT), **a)
                return
            m.worker_idx = None
            fill = _BASE49["fill"].copy()
            stroke = _BASE49["stroke"].copy()
            sw = _BASE49["sw"].copy()
            size = _BASE49["size"].copy()
            # the charged idle field holds until the wave front reaches each cell
            _charge_idle(stroke, _IDLE49 & (now - _T_FLASH < -EPS), _charge(now))

            # radial flash -> LIT (diagonal shimmer); the worker stays AMBER
            lit_op = 0.62 + 0.25 * np.sin(TAU * now / 0.9 - _PHASE)
            since = now - _T_FLASH
            hit = (since >= -EPS) & _NOT_W
            if np.any(hit):
                nh = int(hit.sum())
                e = ease_out_expo_arr(np.clip(since[hit] / FLASH_SETTLE, 0.0, 1.0))[:, None]
                lit_rgb = np.tile(_CYAN_RGB, (nh, 1))
                lit_st = np.tile(HOOK_LIT_STROKE, (nh, 1))
                lit_sz = np.ones(nh)
                pu = _pulse(now)[hit]
                if np.any(pu > 0):               # the second, softer diagonal light pulse
                    lit_op[hit] = lerp(lit_op[hit], PULSE_OP, pu)
                    lit_rgb = lerp(lit_rgb, _LIT_RGB_HOT, pu[:, None])
                    lit_st[:, 3] = lerp(lit_st[:, 3], 0.95, pu)
                    lit_sz = 1.0 + 0.07 * pu
                lit_fill = np.column_stack([lit_rgb, lit_op[hit]])
                fill[hit] = lerp(_INK1, lit_fill, e)
                stroke[hit] = lerp(_INK1, lit_st, e)
                sw[hit] = _FLASH_SW
                size[hit] = lerp(1.3, lit_sz, e[:, 0])
            jolt = shake(now - T_HIT, 0.06, 0.20, SHAKE_SEED)
            corner = None
            halo = 1.0

            if now <= T_PUSH0 + EPS:             # slow push-in about the worker (camera)
                s = _cam_scale(now)
                centers = W_PT + s * (HOOK_CENTERS - W_PT) + jolt
                if s != 1.0:
                    size = size * s              # (the worker keeps its own stroke, as in the push)
                    sw[_NOT_W] *= s
            else:                                # push-in on the worker, from the drift's pose
                e = ease_in_out_expo(ramp(now, T_PUSH0, T_PUSH1))
                base = DRIFT_S + _V_DRIFT * (now - T_PUSH0)      # the drift keeps its speed ...
                s = base ** (1.0 - e) * HOOK_PUSH_SCALE ** e      # ... and hands over to the push
                centers = lerp(W_PT, AC, e) + s * (HOOK_CENTERS - W_PT)
                fade = 1.0 - ease_in_out_sine(ramp(now, T_FADE0, T_FADE1))
                fill[_NOT_W, 3] *= fade
                stroke[_NOT_W, 3] *= fade
                sw[_NOT_W] *= s                  # camera: strokes grow with the push
                size[_NOT_W] *= s
                mo = ease_in_out_cubic(ramp(now, T_MORPH0, T_PUSH1))
                fill[W_IDX] = _morph_fill(mo)
                stroke[W_IDX] = lerp(HOOK_WORKER_STROKE, rgba(CYAN, 1.0), mo)
                sw[W_IDX] = lerp(_W_SW, TOKEN_STROKE, mo)
                k = s * lerp(HOOK_WORKER_SCALE, 1.0, e)
                if now >= T_PUSH1 - EPS:         # SETTLE 0.94 -> 1.0 about AC
                    p = ramp(now, T_PUSH1, T_SETTLE1)
                    k *= lerp(SETTLE_FROM, 1.0, ease_out_back(p)) / SETTLE_FROM
                size[W_IDX] = k
                corner = {W_IDX: lerp(HOOK_CORNER_RATIO, TOKEN_CORNER, mo)}
                halo = 1.0 - ease_in_out_sine(ramp(now, T_PUSH0, T_MORPH0 + 0.25))
            m.draw(centers=centers, size=size, fill=fill, stroke=stroke, sw=sw, corner=corner,
                   hero=W_IDX, halo=halo)

        grid.add_updater(render_grid)

        # ---------------------------------------------------------------- effects
        # CYAN shock ring (w12, r 0.2 -> 7.0) + a white RadialFlash from the worker instead of the
        # old full-frame INK tint (which bloomed into a flat grey wash over the cyan wave).
        ring = ShockRing(self, T_HIT, W_PT, 0.2, 7.0, 12.0, 1.0, 0.5, color=CYAN)
        self.add(ring)
        self.add(RadialFlash(self, T_HIT, W_PT, peak=0.45, decay=0.08))   # under the text

        # ---------------------------------------------------------------- text
        tr = TextTrack(self)
        self.add(tr)
        head = st.headline                       # 'One at a time.' exactly as in LOOP

        def head_alpha(now):
            return 1.0 if now < T_HIT - EPS else 0.40

        def head_shift(now):
            f = story_frame(now)
            if 52 <= f <= 59:
                return np.array([jitter(f, 0.03, JITTER_SEED), 0.0, 0.0])
            return ORIGIN

        tr.on(head, 0.0, T_HOOK_OUT, enter="cut", exit="out", alpha=head_alpha, shift=head_shift)
        # 'All at once.' shares line 1's H96 scale (a matched pair, card()'s baseline pitch): it
        # sits 0.75 above the grid and its SLAM can run at full strength inside the safe zone.
        pair = card([("One at a time.", "H96", INK), ("All at once.", "H96", CYAN)])
        allc = pair[1]
        tr.on(allc, T_HIT, T_HOOK_OUT, enter="slam", exit="out",
              scale=fit_scale(allc, 1.3, box=SAFE_BOX), cap=False)
        # The topic is part of the LOOP frame (b09 POPs it in at 2.40), so it is on screen from
        # frame 0 with no fade-in, stays when the open-loop card arrives and cuts with the token.
        tr.on(st.kicker, 0.0, T_TXT_OUT, enter="cut", exit="cut")
        ol = card([("It scales…", "H84", INK), ("until it doesn't.", "H84", AMBER)])
        tr.on(ol[0], T_CARDS, T_TXT_OUT, exit="cut")
        tr.on(ol[1], T_CARDS, T_TXT_OUT, exit="cut")

        par = "320 WORKERS · 1 STEP"

        def ro_str(now):
            if now < T_HIT - EPS:
                return step_str(b01_worker_idx(sframe(self)))
            if now < T_RO_OUT + 0.20 - EPS:
                return par
            return None

        def ro_out(now):
            return smooth(clip01((now - T_RO_OUT) / 0.20)) if now >= T_RO_OUT - EPS else 0.0

        ro = LiveText(ro_str, lambda s: step_readout_tmpl(s, CYAN if s == par else AMBER),
                      scene=self,
                      alpha=lambda now: 1.0 - ro_out(now),
                      shift=lambda now: DOWN * 0.12 * ro_out(now),   # away from the growing grid
                      scale=lambda now: pop_scale(now - T_HIT))
        self.add(ro)

        # ---------------------------------------------------------------- audio
        for k in range(17):                      # one tick per hop: f2, f5, ... f50
            self.cue("tick", dt=(2 + 3 * k) * F, gain=0.6, freq=3800, pan=0)
        self.cue("riser", dt=T_CHARGE0, dur=0.42, note="visual riser: the idle field charges")
        self.cue("silence", dt=T_FREEZE0, dur=0.133)
        self.cue("hit", dt=T_HIT, gain=1.0)
        self.cue("sub_drop", dt=T_HIT, dur=1.4)
        self.cue("shimmer", dt=1.05, dur=1.3)
        self.cue("tick", dt=T_CARDS, gain=0.5)
        self.cue("shimmer", dt=T_CARDS, gain=0.45, dur=0.5, note="second light pulse")
        self.cue("pad_swell", dt=2.90, dur=1.4)
        self.cue("whoosh", dt=(T_PUSH0 + T_PUSH1) / 2, dur=0.9, direction="up")
        self.cue("tick", dt=T_PUSH1, gain=0.5)

        # ---------------------------------------------------------------- run
        self.hold_until(T_SETTLE1)
        exit_to(self)                            # canonical E01 for the last 3 frames
        finish(self)

"""b01 HOOK: one at a time vs all at once (5.0 s, film 0.0-5.0). Script v3, segment 'hook'.

Frame 0 is handoff_state('LOOP') (== b09's last frame): the x1.4 amber worker with its soft
halo is mid-crawl on idx 32 (10-cell bright trail, #2E3445 idle field), under the
'PARALLEL COMPUTING' kicker, which is part of the LOOP frame (b09 POPs it in at 2.40): on
screen from frame 0 with no fade-in, it stays to the cut. The worker crawls on the integer
frame clock to idx 49 while the camera creeps in about it (1.0 -> 1.035, ease_in_quad) and,
with the audio riser (0.45), the idle field charges (outlines DIM -> CYAN 0.45). Both stop
dead on the 8-frame micro-freeze (the headline holds still, silence); at 1.00 s every other cell
lights up in a radial INK->CYAN wave (white radial burst + CYAN shock ring from the worker)
while the lone worker stays AMBER, drawn on the grid's top layer with its halo (above the
1.3x flashing neighbours); the halo fades out during the push-in, before the token lands.
The first second (riser 0.45, freeze f52-f59, hit 1.00) is LOCKED.
Read-holds: 'All at once.' + '320 WORKERS · 1 STEP' hold over the shimmering field until 2.70
(f162: exactly the 1.70 s floor of 'All at once.'; the readout fades out 3.25-3.55), then the hook
pair hard-swaps to the open-loop card 'It scales… / then stalls.' (AMBER 2nd line, POP, f162)
on the calm field, read for 1.20 s before the push-in fires (visible 2.70 -> 4.95 = 2.25 s,
floor 2.0). The swap is the film's card-swap grammar: no 0.20 s CARD_OUT + 0.25 s CARD_IN gap
eats the reading windows. The camera then never sits still: a slow linear push-in about the
worker (1.035 -> 1.06, 1.30-3.90) carries straight into the ease_in_out_sine push-in
(3.90-4.83, 56 frames), with a second, softer diagonal light pulse on the open-loop card
(2.70). The worker morphs into the plain cyan E01 token (no HUD: b03 opens it as core T6 and
its counter first appears there) and settles (cut_times_v2(5.0): land 4.833, settle 0.12 s on
settle_scale(), no overshoot, static E01 f297-f299). Script v3 calm motion: no impact shake on
the hit, no headline jitter in the freeze. The kicker + open-loop card stay fully visible until the token is
static (hard cut at 4.95), so the cut never shows an empty, text-less frame before b03's
frame 0.

Everything per frame is a pure function of self.now (one vectorised grid updater + the
motif text tracks), so 30 fps previews and 60 fps finals agree.
"""

from style import *
from motif import *

# ---------------------------------------------------------------- timing (local s)
F = 1 / 60
W_IDX = 49                      # where the crawl stops (worker cell)
W_PT = hook_center(W_IDX)       # (-0.52, 2.32)
T_FREEZE0 = 52 * F              # f52..f59 micro-freeze (worker + camera stop; the headline holds still)
T_HIT = 1.00                    # f60 the parallel wave
WAVE_SPAN = 0.35                # flash front reaches the farthest cell 0.35 s after the hit
FLASH_SETTLE = 0.25             # INK flash -> LIT
# Read-holds (5.0 s slot; the first second above is untouched): the payoff ('All at once.' +
# '320 WORKERS · 1 STEP' over the shimmering cyan field) holds 1.70 s after the hit, then the
# hook pair HARD-SWAPS to the open-loop card 'It scales… / then stalls.' (POP), read on the
# calm field for 1.20 s before the push-in fires. The text window is 1.00 (the sacred hit) ->
# 4.95 (exit) = 3.95 s for floors of 1.70 ('All at once.', 3 words) + 2.00 (open-loop card,
# 4 words); the swap at f162 (script v2) gives 'All at once.' exactly its 1.70 s and the card
# 2.25 s (counted from full opacity: SLAM / POP are opaque at once).
T_SWAP = 162 * F                # 2.70 hook pair cut -> open-loop card POP
T_CARDS = T_SWAP                # open-loop card in (+ the second light pulse)
T_RO_OUT = T_CARDS + 0.55       # 3.25 '320 WORKERS · 1 STEP' fades out (0.30 s: gone 3.55, before the push-in)
RO_FADE = 0.30
CREEP_S = 1.035                 # camera creeps in about the worker during the crawl (1.0 -> 1.035,
                                # ease_in_quad from f0, stops dead on the micro-freeze, holds)
T_CHARGE0 = 0.45                # visual riser (with the audio riser): the idle field charges,
CHARGE_STROKE = rgba(CYAN, 0.45)   # outline only: DIM 0.35 -> CYAN 0.45 by the freeze (the fill
                                # stays idle grey: the cyan fill is the hit's reveal)
SLOT = beat_duration("B01Hook") # 5.0
_, T_PUSH1, T_SETTLE1 = cut_times_v2(SLOT)  # 4.8333 (f290) lands, 4.95 (f297) settled (b01's own window)
PUSH_DUR = 56 * F               # push-in on the worker (CALM_EASE = ease_in_out_sine), 0.933 s
T_PUSH0 = T_PUSH1 - PUSH_DUR    # 3.90 (f234)
T_DRIFT0, T_DRIFT1 = 1.30, T_PUSH0   # slow push-in about the worker (linear) under both holds ...
DRIFT_S = 1.06                  # ... CREEP_S -> 1.06, handing its speed to the push-in
T_FADE0, T_FADE1 = T_PUSH0 + 0.15, T_PUSH0 + 0.80   # 4.05-4.70 non-worker cells fade out
T_MORPH0 = T_PUSH0 + 0.30       # 4.20 worker -> token style
T_TXT_OUT = T_SETTLE1           # kicker + open-loop card: full until the token is static, then cut

PULSE_SPAN = 0.26               # second light pulse: front crosses the grid diagonal in 0.26 s ...
PULSE_DUR = 0.16                # ... each cell 0 -> peak -> 0 over 0.16 s (whole sweep ~0.42 s)
PULSE_OP = 0.97                 # LIT opacity at the pulse peak (shimmer 0.62 +- 0.25)


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
    """Second light pulse (T_CARDS): per-cell 0 -> 1 -> 0 bump as the diagonal front passes."""
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
            corner = None
            halo = 1.0

            if now <= T_PUSH0 + EPS:             # slow push-in about the worker (camera)
                s = _cam_scale(now)
                centers = W_PT + s * (HOOK_CENTERS - W_PT)
                if s != 1.0:
                    size = size * s              # (the worker keeps its own stroke, as in the push)
                    sw[_NOT_W] *= s
            else:                                # push-in on the worker, from the drift's pose
                e = CALM_EASE(ramp(now, T_PUSH0, T_PUSH1))
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
                if now >= T_PUSH1 - EPS:         # SETTLE 0.94 -> 1.0 about AC (no overshoot)
                    k *= settle_scale(ramp(now, T_PUSH1, T_SETTLE1))
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

        tr.on(head, 0.0, T_SWAP, enter="cut", exit="cut", alpha=head_alpha)   # holds still (no jitter)
        # 'All at once.' shares line 1's H96 scale (a matched pair, card()'s baseline pitch): it
        # sits 0.75 above the grid and its SLAM can run at full strength inside the safe zone.
        pair = card([("One at a time.", "H96", INK), ("All at once.", "H96", CYAN)])
        allc = pair[1]
        tr.on(allc, T_HIT, T_SWAP, enter="slam", exit="cut",
              scale=fit_scale(allc, 1.3, box=SAFE_BOX), cap=False)
        # The topic is part of the LOOP frame (b09 POPs it in at 2.40), so it is on screen from
        # frame 0 with no fade-in, stays when the open-loop card arrives and cuts with the token.
        tr.on(st.kicker, 0.0, T_TXT_OUT, enter="cut", exit="cut")
        ol = card([("It scales…", "H84", INK), ("then stalls.", "H84", AMBER)])
        # BG knockout (as b09's CTA): during the push-in (~4.30-4.60) the enlarged, fading top row
        # of cells passes under 'then stalls.'; the knockout keeps the glyph edges clean there.
        # It overlaps nothing before the push-in, and the card is cut at 4.95 (before E01).
        ol.set_stroke(BG, width=8, opacity=0.9, background=True)
        tr.on(ol[0], T_CARDS, T_TXT_OUT, enter="pop", exit="cut")   # both lines POP on the swap
        tr.on(ol[1], T_CARDS, T_TXT_OUT, enter="pop", exit="cut")

        par = "320 WORKERS · 1 STEP"

        def ro_str(now):
            if now < T_HIT - EPS:
                return step_str(b01_worker_idx(sframe(self)))
            if now < T_RO_OUT + RO_FADE - EPS:
                return par
            return None

        def ro_out(now):                         # 0 -> 1 over the 0.30 s fade (fade_at clock)
            return 1.0 - fade_at(now, T_RO_OUT, T_RO_OUT + RO_FADE)

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
        self.cue("shimmer", dt=1.05, dur=round(T_CARDS - 1.05, 4), note="the lit field, hit -> open-loop card")
        self.cue("tick", dt=T_CARDS, gain=0.5)
        self.cue("shimmer", dt=T_CARDS, gain=0.45, dur=0.5, note="second light pulse")
        self.cue("pad_swell", dt=T_PUSH0 - 0.50, dur=1.4, note="swells into the push-in landing")
        self.cue("whoosh", dt=(T_PUSH0 + T_PUSH1) / 2, dur=0.9, direction="up")
        self.cue("tick", dt=T_PUSH1, gain=0.5)

        # ---------------------------------------------------------------- run
        self.hold_until(T_SETTLE1)
        exit_to(self)                            # canonical E01 for the last 3 frames
        finish(self)

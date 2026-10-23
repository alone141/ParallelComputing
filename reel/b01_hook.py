"""b01 HOOK: one at a time vs all at once (4.5 s, film 0.0-4.5).

Frame 0 is handoff_state('LOOP') (== b09's last frame): the amber worker is mid-crawl on
idx 32. It crawls on the integer frame clock to idx 49, freezes for 8 frames (headline
jitter, silence), then at 1.00 s every other cell lights up in a radial INK->CYAN wave
while the lone worker stays AMBER. The open-loop card follows, then the camera pushes into
the worker, which morphs into the cyan E01 token and settles.

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
T_CARDS = 2.35                  # kicker + open loop in
SLAM_S = 1.15                   # 'All at once.' is 6.9 wide: 1.3x would leave the frame
T_RO_OUT = 2.85                 # '320 WORKERS · 1 STEP' out
T_PUSH0 = 3.40                  # push-in on the worker
T_PUSH1 = 257 * F               # 4.2833: push-in lands (frame-aligned, see SETTLE)
T_FADE0, T_FADE1 = 3.55, 4.20   # non-worker cells fade out
T_MORPH0 = 3.70                 # worker -> token style
T_SETTLE1 = 267 * F             # 4.45: settle done; f267-f269 = static E01 (3 frames)
T_TXT_OUT = 4.20                # kicker + open-loop card out (gone by 4.40)

SHAKE_SEED = 1
JITTER_SEED = 4

# ---------------------------------------------------------------- precomputed grid data
_NOT_W = np.arange(HOOK_N) != W_IDX
_DIST = np.linalg.norm(HOOK_CENTERS - W_PT, axis=1)
_T_FLASH = T_HIT + WAVE_SPAN * _DIST / _DIST[_NOT_W].max()     # per-cell flash time
_PHASE = 1.2 * (HOOK_CENTERS[:, 0] - HOOK_CENTERS[:, 1])        # diagonal sheen phase
_BASE49 = HookGrid.crawl_arrays(W_IDX)
_INK1 = rgba(INK, 1.0)
_CYAN_RGB = rgba(CYAN)[:3]
_FLASH_SW = 1.5


_HOT = 0.42                     # worker morph passes through hot INK (no muddy olive midpoint)


def _morph_fill(mo: float) -> np.ndarray:
    """Worker fill AMBER op1 -> (hot, INK-ish) -> CYAN op 0.18 (the E01 token fill)."""
    hot = rgba(INK, 1.0)
    if mo <= _HOT:
        return lerp(HOOK_WORKER_FILL, lerp(HOOK_WORKER_FILL, hot, 0.75), mo / _HOT)
    return lerp(lerp(HOOK_WORKER_FILL, hot, 0.75), rgba(CYAN, TOKEN_FILL), (mo - _HOT) / (1 - _HOT))


def finish(scene) -> None:
    """Fill the rest of the slot frame-exactly (1 frame -> wait(1/fps), no sub-frame wait)."""
    n = int(round(beat_duration(type(scene).__name__) * config.frame_rate)) - scene._frames
    if n == 1:
        scene.wait(1 / config.frame_rate)
    elif n > 1:
        scene.hold_until(beat_duration(type(scene).__name__))


def _eox(p):
    """Vectorised ease_out_expo (exactly 1 at p >= 1)."""
    p = np.asarray(p, float)
    return np.where(p >= 1.0, 1.0, 1.0 - np.power(2.0, -10.0 * p))


class B01Hook(BeatScene):
    def beat(self):
        st = enter(self)                         # LOOP: grid(32) + card + readout, exact
        grid = st.grid
        self.remove(st.card, st.readout)         # re-drawn below by the frame-exact tracks

        # ---------------------------------------------------------------- grid
        def render_grid(m, dt):
            now = self.now
            if now < T_HIT - EPS:                # crawl, then the micro-freeze (idx 49)
                w = b01_worker_idx(sframe(self))
                if m.worker_idx != w:
                    m.set_crawl(w)
                return
            m.worker_idx = None
            fill = _BASE49["fill"].copy()
            stroke = _BASE49["stroke"].copy()
            sw = _BASE49["sw"].copy()
            size = _BASE49["size"].copy()

            # radial flash -> LIT (diagonal shimmer); the worker stays AMBER
            lit_op = 0.62 + 0.25 * np.sin(TAU * now / 0.9 - _PHASE)
            since = now - _T_FLASH
            hit = (since >= -EPS) & _NOT_W
            if np.any(hit):
                e = _eox(np.clip(since[hit] / FLASH_SETTLE, 0.0, 1.0))[:, None]
                lit_fill = np.column_stack([np.tile(_CYAN_RGB, (int(hit.sum()), 1)), lit_op[hit]])
                fill[hit] = lerp(_INK1, lit_fill, e)
                stroke[hit] = lerp(_INK1, HOOK_LIT_STROKE, e)
                sw[hit] = _FLASH_SW
                size[hit] = lerp(1.3, 1.0, e[:, 0])
            centers = HOOK_CENTERS + shake(now - T_HIT, 0.06, 0.20, SHAKE_SEED)
            corner = None

            if now > T_PUSH0 + EPS:              # push-in on the worker
                e = ease_in_out_expo(ramp(now, T_PUSH0, T_PUSH1))
                s = HOOK_PUSH_SCALE ** e
                centers = lerp(W_PT, AC, e) + s * (HOOK_CENTERS - W_PT)
                fade = 1.0 - ease_in_out_sine(ramp(now, T_FADE0, T_FADE1))
                fill[_NOT_W, 3] *= fade
                stroke[_NOT_W, 3] *= fade
                sw[_NOT_W] *= s                  # camera: strokes grow with the push
                size[_NOT_W] *= s
                mo = ease_in_out_cubic(ramp(now, T_MORPH0, T_PUSH1))
                fill[W_IDX] = _morph_fill(mo)
                stroke[W_IDX] = lerp(HOOK_WORKER_STROKE, rgba(CYAN, 1.0), mo)
                sw[W_IDX] = lerp(2.0, TOKEN_STROKE, mo)
                k = s * lerp(HOOK_WORKER_SCALE, 1.0, e)
                if now >= T_PUSH1 - EPS:         # SETTLE 0.94 -> 1.0 about AC
                    p = ramp(now, T_PUSH1, T_SETTLE1)
                    k *= lerp(SETTLE_FROM, 1.0, ease_out_back(p)) / SETTLE_FROM
                size[W_IDX] = k
                corner = {W_IDX: lerp(HOOK_CORNER_RATIO, TOKEN_CORNER, mo)}
            m.draw(centers=centers, size=size, fill=fill, stroke=stroke, sw=sw, corner=corner)

        grid.add_updater(render_grid)

        # ---------------------------------------------------------------- effects
        ring = ShockRing(self, T_HIT, W_PT, 0.2, 6.5, 6.0, 1.0, 0.5)
        self.add(ring)

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
        allc = card([("All at once.", "H96", CYAN)], top=below(head))[0]

        def hook_out(now):
            return smooth(clip01((now - T_HOOK_OUT) / 0.20)) if now >= T_HOOK_OUT - EPS else 0.0

        def all_scale(now):
            d = now - T_HIT
            return lerp(SLAM_S, 1.0, ease_out_back(d / 0.20)) if -EPS <= d < 0.20 - EPS else 1.0

        all_live = LiveText(lambda now: "A" if T_HIT - EPS <= now < T_HOOK_OUT + 0.20 - EPS else None,
                            lambda s: allc, scene=self, about=allc.get_center(),
                            alpha=lambda now: 1.0 - hook_out(now),
                            shift=lambda now: UP * 0.15 * hook_out(now), scale=all_scale)
        tr.on(kick("PARALLEL COMPUTING", CYAN), T_CARDS, T_TXT_OUT)
        ol = card([("It scales…", "H84", INK), ("until it doesn't.", "H84", AMBER)])
        tr.on(ol[0], T_CARDS, T_TXT_OUT)
        tr.on(ol[1], T_CARDS, T_TXT_OUT)

        par = "320 WORKERS · 1 STEP"

        def ro_str(now):
            if now < T_HIT - EPS:
                return step_str(b01_worker_idx(sframe(self)))
            if now < T_RO_OUT + 0.20 - EPS:
                return par
            return None

        def ro_out(now):
            return smooth(clip01((now - T_RO_OUT) / 0.20)) if now >= T_RO_OUT - EPS else 0.0

        self.add(all_live)
        ro = LiveText(ro_str, lambda s: step_readout_tmpl(s, CYAN if s == par else AMBER),
                      scene=self,
                      alpha=lambda now: 1.0 - ro_out(now),
                      shift=lambda now: UP * 0.15 * ro_out(now),
                      scale=lambda now: pop_scale(now - T_HIT))
        self.add(ro)

        flash = FrameFlash(self, T_HIT, 0.22, 0.12)
        self.add(flash)

        # ---------------------------------------------------------------- audio
        for k in range(17):                      # one tick per hop: f2, f5, ... f50
            self.cue("tick", dt=(2 + 3 * k) * F, gain=0.6, freq=3800, pan=0)
        self.cue("riser", dt=0.45, dur=0.42)
        self.cue("silence", dt=T_FREEZE0, dur=0.133)
        self.cue("hit", dt=T_HIT, gain=1.0)
        self.cue("sub_drop", dt=T_HIT, dur=1.4)
        self.cue("shimmer", dt=1.05, dur=1.3)
        self.cue("tick", dt=T_CARDS, gain=0.5)
        self.cue("pad_swell", dt=2.90, dur=1.4)
        self.cue("whoosh", dt=(T_PUSH0 + T_PUSH1) / 2, dur=0.9, direction="up")
        self.cue("tick", dt=T_PUSH1, gain=0.5)

        # ---------------------------------------------------------------- run
        self.hold_until(T_SETTLE1)
        exit_to(self)                            # canonical E01 for the last 3 frames
        finish(self)

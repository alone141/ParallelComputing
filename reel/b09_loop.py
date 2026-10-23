"""b09 LOOP: why buy 100,000 GPUs? Back to one worker (3.5 s, film 51.5-55.0). Script v3,
segment 'ending' (calm motion: 0.90 s ease_in_out_sine pull-back ending on f54, the 10× meter
fades out over 0.30 s).

Frame 0 is E08 (the twist -> ending boundary): the bright serial token (AMBER fill 0.58,
crisp #FFD27A rim) + the '10×' HUD. On frame 1 the comment question 'Why buy / 100,000 GPUs?'
('100,000' CYAN) POPs in and the camera is already pulling back
(ease_in_out_sine 0.00-0.90, at rest from f54): the token was cell 0 of the hook grid all along. Drawn on the grid's top layer, it
fills in to the lone solid-amber worker within 0.6 s (rim -> INK, x1.4, its halo growing in
with the morph), while the HUD fades out over 0.30 s. The question
is fully visible f1 -> 2.40 = 2.38 s (floor 0.3 s x 4 words + 0.8 = 2.0). 'Wrong
answers only ↓' (the caption's CTA, BG knockout over the still-settling field) joins at 0.43,
once the question has been read alone for 0.4 s, so the CTA gets its full reading window
(fully in 0.68 -> 2.40 = 1.72 s >= 0.3 s x 3 words + 0.8). Count-in: the settled worker pulses twice (f84, f100; halo swells, the cell
blinks hot, a low tick, the CTA arrow nudges down), and the crawl's first hop (f116) lands on
the same 16-frame interval. From f113 the worker crawls on the same integer frame clock as
b01 (hop every 3 frames, LOCKED); at 2.40 the question cuts and 'One at a time.' + the
'PARALLEL COMPUTING' kicker + the readout POP in (no kicker before). The worker lands on idx 32
on the very last frame (f209), which is exactly b01's frame 0 (handoff_state('LOOP')).
"""

from style import *
from motif import *

# ---------------------------------------------------------------- timing (local s)
F = 1 / 60
T_HUD_OUT = 0.30                # the 10× meter fades out 0.00-0.30 (opacity only, fade_at clock)
T_Q_IN = F                      # 'Why buy / 100,000 GPUs?' POPs on story frame 1
T_CB_IN = 26 * F                # 0.4333 'Wrong answers only ↓' CARD_IN (BG knockout over the settling grid);
                                # fully in 0.683 -> T_CUT 2.40 = 1.717 s (reading floor 1.70)
T_ZOOM0 = 0.0                   # smooth pull-back (CALM_EASE = ease_in_out_sine) from the cut ...
T_ZOOM1 = 54 * F                # ... lands on f54 (0.90), before the locked f84 count-in
T_FILL1 = 0.60                  # token fill 0.58 -> 1.0 (the worker's solid amber) by 0.6 s
T_MORPH1 = 0.90                 # rim #FFD27A w6 -> INK w2, corner, own scale 1.0 -> 1.4, halo 0 -> 1
GRID_A0, GRID_A1 = 0.06, 0.32   # grid opacity ramp over zoom progress e (token alone for ~2 frames)
T_CUT = 2.40                    # f144 hard cut: question out, 'One at a time.' + readout POP in
F_LAST = 209                    # last frame == b01 frame 0
CTA = "Wrong answers only ↓"

# Count-in: once the pull-back has settled, the lone worker pulses twice (halo swells, the cell
# blinks toward INK) on f84 and f100, and the crawl's first hop (f116) lands on the same
# 16-frame interval: 'ready, ready, go'. The CTA's arrow nudges down on the same beats.
COUNT_IN = (84 * F, 100 * F)    # 1.40, 1.6667
PULSE_UP, PULSE_DOWN = 2 * F, 9 * F   # attack / release: each pulse is back to rest 11 frames later (<= f111)
PULSE_SIZE = 0.16               # worker scale x(1 + 0.16)
PULSE_HALO = 0.45               # halo size x(1 + 0.45) on top of that
PULSE_HOT = 0.35                # worker fill -> INK at the peak
ARROW_NUDGE = 0.10              # CTA arrow drop at the peak


def finish(scene) -> None:
    """Fill the rest of the slot frame-exactly (1 frame -> wait(1/fps), no sub-frame wait)."""
    n = int(round(beat_duration(type(scene).__name__) * config.frame_rate)) - scene._frames
    if n == 1:
        scene.wait(1 / config.frame_rate)
    elif n > 1:
        scene.hold_until(beat_duration(type(scene).__name__))


_SLOT = hook_center(0)          # (-3.58, 3.00)
_NOT0 = np.arange(HOOK_N) != 0
_BASE0 = HookGrid.crawl_arrays(0)
_TOKEN_FILL, _TOKEN_STROKE = token_rgbas(AMBER)   # E08's serial token (fill 0.58, #FFD27A rim)
_W_SW = float(_BASE0["sw"][0])  # the worker's own stroke width (hook_state)
_HOT_FILL = rgba(INK, 1.0)


def count_in(now: float) -> float:
    """0..1 count-in pulse envelope: fast rise (ease_out_quad), soft release (ease_in_out_sine).
    Exactly 0 outside the two pulses, so the crawl from f113 starts from the exact rest state."""
    out = 0.0
    for t0 in COUNT_IN:
        d = now - t0
        if -EPS < d < PULSE_UP:
            out = max(out, ease_out_quad(clip01(d / PULSE_UP)))
        elif PULSE_UP <= d < PULSE_UP + PULSE_DOWN - EPS:
            out = max(out, 1.0 - ease_in_out_sine((d - PULSE_UP) / PULSE_DOWN))
    return out


class B09Loop(BeatScene):
    def beat(self):
        st = enter(self)                         # E08: amber token + '10×' HUD (exact)
        hud = st.hud
        hud.fade_out(0.0, T_HUD_OUT)

        # ---------------------------------------------------------------- grid
        grid = HookGrid()

        def render_grid(m, dt):
            f = sframe(self)
            if f >= 113:                         # crawl: idx 0 on f113 ... idx 32 on f209
                w = b09_worker_idx(f)
                if m.worker_idx != w:
                    m.set_crawl(w)
                return
            now = self.now
            m.worker_idx = None
            e = CALM_EASE(ramp(now, T_ZOOM0, T_ZOOM1))
            k, b = expand_cam(_SLOT, HOOK_CELL, e)
            mf = ease_out_cubic(ramp(now, 0.0, T_FILL1))
            mo = ease_out_cubic(ramp(now, 0.0, T_MORPH1))
            fill = _BASE0["fill"].copy()
            stroke = _BASE0["stroke"].copy()
            sw = _BASE0["sw"].copy()
            size = _BASE0["size"].copy()
            a = ramp(e, GRID_A0, GRID_A1)        # the field fades in as the camera pulls back
            fill[_NOT0, 3] *= a
            stroke[_NOT0, 3] *= a
            sw[_NOT0] *= k                       # camera: strokes scale with the zoom
            fill[0] = lerp(_TOKEN_FILL, HOOK_WORKER_FILL, mf)
            stroke[0] = lerp(_TOKEN_STROKE, HOOK_WORKER_STROKE, mo)
            sw[0] = lerp(TOKEN_STROKE, _W_SW, mo)
            size[0] = lerp(1.0, HOOK_WORKER_SCALE, mo)
            p = count_in(now)                    # 'ready, ready': the worker pulses before the crawl
            if p > 0.0:
                fill[0] = lerp(fill[0], _HOT_FILL, PULSE_HOT * p)
                size[0] *= 1.0 + PULSE_SIZE * p
            size = size * k
            m.draw(centers=HOOK_CENTERS * k + b, size=size, fill=fill, stroke=stroke, sw=sw,
                   corner={0: lerp(TOKEN_CORNER, HOOK_CORNER_RATIO, mo)}, hero=0, halo=mo,
                   halo_scale=size[0] / HOOK_WORKER_SCALE * (1.0 + PULSE_HALO * p))

        render_grid(grid, 0)                     # frame 0: cell 0 IS the E08 token
        grid.add_updater(render_grid)
        add_behind(self, grid)
        self.remove(st.token)

        # ---------------------------------------------------------------- text
        tr = TextTrack(self)
        self.add(tr)
        q = card([("Why buy", "H84", INK), ("100,000 GPUs?", "H84", INK, {"100,000": CYAN})])
        tr.on(q[0], T_Q_IN, T_CUT, enter="pop", exit="cut")
        tr.on(q[1], T_Q_IN, T_CUT, enter="pop", exit="cut")
        cb = card([(CTA, "S56", INK)], top=STEP_TOP)[0]
        cb.set_fill(opacity=0.92)
        cb.set_stroke(BG, width=8, opacity=0.9, background=True)   # knockout over grid cells
        tr.on(VGroup(*cb[:-1]), T_CB_IN, T_CUT, exit="cut")
        tr.on(cb[-1], T_CB_IN, T_CUT, exit="cut",              # the '↓' nudges on the count-in
              shift=lambda now: DOWN * ARROW_NUDGE * count_in(now))
        head = loop_headline()
        tr.on(head, T_CUT, None, enter="pop", exit=None)
        tr.on(loop_kicker(), T_CUT, None, enter="pop", exit=None)
        ro = LiveText(lambda now: step_str(b09_worker_idx(sframe(self))) if now >= T_CUT - EPS else None,
                      lambda s: step_readout_tmpl(s, AMBER), scene=self,
                      scale=lambda now: pop_scale(now - T_CUT))
        self.add(ro)

        # ---------------------------------------------------------------- audio
        self.cue("pop", dt=0.0, gain=0.5)        # the cut + the question POP
        self.cue("whoosh", dt=0.45, dur=0.80, direction="down")   # the smooth 0.90 s pull-back
        self.cue("tick", dt=T_CB_IN, gain=0.4)   # CTA in
        for t in COUNT_IN:                       # count-in: two low ticks, then the crawl's 3800 Hz
            self.cue("tick", dt=t, gain=0.45, freq=2400, pan=0, note="count-in pulse")
        for k in range(1, 33):                   # one tick per hop: f116, f119, ... f209 (idx 1..32)
            self.cue("tick", dt=(113 + 3 * k) * F, gain=0.6, freq=3800, pan=0)
        self.cue("pop", dt=T_CUT, gain=0.5)

        # ---------------------------------------------------------------- run
        self.hold_until(F_LAST * F)
        exit_to(self)                            # f209 == LOOP == b01 frame 0 (the hop onto 32)
        finish(self)

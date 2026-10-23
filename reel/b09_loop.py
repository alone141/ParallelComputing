"""b09 LOOP: so why build 100,000 GPUs? Back to one worker (3.5 s, film 44.0-47.5).

Frame 0 is E08 (the amber serial token + the '10×' HUD). On frame 1 the comment question
POPs in and the camera is already pulling back (ease_out_quart from 0.00: the grid rushes in
at once and decelerates all the way into the crawl): the token was cell 0 of the hook grid
all along, and it fills in to the lone amber worker within 0.6 s. The HUD falls away.
'Wrong answers only ↓' (the caption's CTA, BG knockout) joins at 1.00 as the grid settles
into its own footprint; the 32 crawl ticks start on the first hop (f116). From f113 the
worker crawls on the same integer frame clock as b01 (hop every 3 frames) and lands on idx 32
on the very last frame (f209), which is exactly b01's frame 0 (handoff_state('LOOP')).
"""

from style import *
from motif import *

# ---------------------------------------------------------------- timing (local s)
F = 1 / 60
T_HUD_OUT = 0.28                # HUD fades out + drops 0.2 (ease_out_quad: clear before the grid is dense)
T_Q_IN = F                      # 'So why build / 100,000 GPUs?' POPs on story frame 1
T_CB_IN = 1.00                  # 'Wrong answers only ↓' (grid inside its footprint by ~1.1; BG knockout)
T_ZOOM0 = 0.0                   # reverse zoom (ease_out_quart) from the cut ...
T_ZOOM1 = 113 * F               # ... lands exactly on f113 (1.8833)
T_FILL1 = 0.60                  # token fill 0.32 -> 1.0 (the worker's solid amber) by 0.6 s
T_MORPH1 = 0.90                 # stroke AMBER w6 -> INK w2, corner, own scale 1.0 -> 1.15
GRID_A0, GRID_A1 = 0.02, 0.25   # grid opacity ramp over zoom progress e
T_CUT = 2.40                    # f144 hard cut: question out, 'One at a time.' + readout POP in
F_LAST = 209                    # last frame == b01 frame 0
CTA = "Wrong answers only ↓"


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
_TOKEN_FILL = rgba(AMBER, TOKEN_FILL)
_TOKEN_STROKE = rgba(AMBER, 1.0)
_W_SW = float(_BASE0["sw"][0])  # the worker's own stroke width (hook_state)


class B09Loop(BeatScene):
    def beat(self):
        st = enter(self)                         # E08: amber token + '10×' HUD (exact)
        hud = st.hud
        hud.fade_to(0.0, T_HUD_OUT, 1.0, 0.0, ORIGIN, DOWN * 0.2, ease_out_quad)

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
            e = ease_out_quart(ramp(now, T_ZOOM0, T_ZOOM1))
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
            m.draw(centers=HOOK_CENTERS * k + b, size=size * k, fill=fill, stroke=stroke, sw=sw,
                   corner={0: lerp(TOKEN_CORNER, HOOK_CORNER_RATIO, mo)})

        render_grid(grid, 0)                     # frame 0: cell 0 IS the E08 token
        grid.add_updater(render_grid)
        add_behind(self, grid)
        self.remove(st.token)

        # ---------------------------------------------------------------- text
        tr = TextTrack(self)
        self.add(tr)
        q = card([("So why build", "H84", INK), ("100,000 GPUs?", "H84", INK, {"100,000": CYAN})])
        tr.on(q[0], T_Q_IN, T_CUT, enter="pop", exit="cut")
        tr.on(q[1], T_Q_IN, T_CUT, enter="pop", exit="cut")
        cb = card([(CTA, "S56", INK)], top=STEP_TOP)[0]
        cb.set_fill(opacity=0.92)
        cb.set_stroke(BG, width=8, opacity=0.9, background=True)   # knockout over grid cells
        tr.on(cb, T_CB_IN, T_CUT, exit="cut")
        head = card([("One at a time.", "H96", INK)])[0]
        tr.on(head, T_CUT, None, enter="pop", exit=None)
        ro = LiveText(lambda now: step_str(b09_worker_idx(sframe(self))) if now >= T_CUT - EPS else None,
                      lambda s: step_readout_tmpl(s, AMBER), scene=self,
                      scale=lambda now: pop_scale(now - T_CUT))
        self.add(ro)

        # ---------------------------------------------------------------- audio
        self.cue("pop", dt=0.0, gain=0.5)        # the cut + the question POP
        self.cue("whoosh", dt=0.35, dur=0.70, direction="down")   # the fast half of the pull-back
        self.cue("tick", dt=T_CB_IN, gain=0.4)   # CTA in
        for k in range(1, 33):                   # one tick per hop: f116, f119, ... f209 (idx 1..32)
            self.cue("tick", dt=(113 + 3 * k) * F, gain=0.6, freq=3800, pan=0)
        self.cue("pop", dt=T_CUT, gain=0.5)

        # ---------------------------------------------------------------- run
        self.hold_until(F_LAST * F)
        exit_to(self)                            # f209 == LOOP == b01 frame 0 (the hop onto 32)
        finish(self)

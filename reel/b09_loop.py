"""b09 LOOP: so why build 100,000 GPUs? Back to one worker (3.5 s, film 44.0-47.5).

Frame 0 is E08 (the amber serial token + the '10×' HUD). The HUD falls away, the comment
card asks the question, and the camera pulls back from the token: it was cell 0 of the
hook grid all along, and it turns into the lone amber worker. From f113 the worker crawls
on the same integer frame clock as b01 (hop every 3 frames) and lands on idx 32 on the very
last frame (f209), which is exactly b01's frame 0 (handoff_state('LOOP')), mid-crawl.
"""

from style import *
from motif import *

# ---------------------------------------------------------------- timing (local s)
F = 1 / 60
T_HUD_OUT = 0.35                # HUD fades out + drops 0.2 (ease_in_cubic)
T_Q_IN = 0.05                   # 'So why build / 100,000 GPUs?'
T_CB_IN = 0.60                  # 'Comment below.'
T_ZOOM0 = 0.20                  # reverse zoom (ease_in_out_quart) ...
T_ZOOM1 = 113 * F               # ... lands exactly on f113 (1.8833)
T_MORPH0, T_MORPH1 = 0.25, 1.45 # token -> worker style (visible while the camera is still close)
T_CUT = 2.40                    # f144 hard cut: question out, 'One at a time.' + readout POP in
F_LAST = 209                    # last frame == b01 frame 0

def ease_in_out_quart(t: float) -> float:
    """Zoom curve: confident like expo, but with a visible lead-in and a landing that ends
    right before the crawl (expo over 1.68 s would hold ~0.5 s static on both ends)."""
    return 8 * t ** 4 if t < 0.5 else 1 - (-2 * t + 2) ** 4 / 2


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


class B09Loop(BeatScene):
    def beat(self):
        st = enter(self)                         # E08: amber token + '10×' HUD (exact)
        hud = st.hud
        hud.fade_to(0.0, T_HUD_OUT, 1.0, 0.0, ORIGIN, DOWN * 0.2, ease_in_cubic)

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
            e = ease_in_out_quart(ramp(now, T_ZOOM0, T_ZOOM1))
            k, b = expand_cam(_SLOT, HOOK_CELL, e)
            mo = ease_in_out_sine(ramp(now, T_MORPH0, T_MORPH1))
            fill = _BASE0["fill"].copy()
            stroke = _BASE0["stroke"].copy()
            sw = _BASE0["sw"].copy()
            size = _BASE0["size"].copy()
            a = ramp(e, 0.10, 0.50)              # the field fades in while the camera pulls back
            fill[_NOT0, 3] *= a
            stroke[_NOT0, 3] *= a
            sw[_NOT0] *= k                       # camera: strokes scale with the zoom
            fill[0] = lerp(_TOKEN_FILL, HOOK_WORKER_FILL, mo)
            stroke[0] = lerp(_TOKEN_STROKE, HOOK_WORKER_STROKE, mo)
            sw[0] = lerp(TOKEN_STROKE, 2.0, mo)
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
        tr.on(q[0], T_Q_IN, T_CUT, exit="cut")
        tr.on(q[1], T_Q_IN, T_CUT, exit="cut")
        cb = card([("Comment below.", "S56", INK)], top=STEP_TOP)[0]
        cb.set_fill(opacity=0.85)
        tr.on(cb, T_CB_IN, T_CUT, exit="cut")
        head = card([("One at a time.", "H96", INK)])[0]
        tr.on(head, T_CUT, None, enter="pop", exit=None)
        ro = LiveText(lambda now: step_str(b09_worker_idx(sframe(self))) if now >= T_CUT - EPS else None,
                      lambda s: step_readout_tmpl(s, AMBER), scene=self,
                      scale=lambda now: pop_scale(now - T_CUT))
        self.add(ro)

        # ---------------------------------------------------------------- audio
        self.cue("pop", dt=0.0, gain=0.5)
        self.cue("whoosh", dt=(T_ZOOM0 + T_ZOOM1) / 2, dur=1.68, direction="down")
        for k in range(33):                      # one tick per hop: f113, f116, ... f209
            self.cue("tick", dt=(113 + 3 * k) * F, gain=0.6, freq=3800, pan=0)
        self.cue("pop", dt=T_CUT, gain=0.5)

        # ---------------------------------------------------------------- run
        self.hold_until(F_LAST * F)
        exit_to(self)                            # f209 == LOOP == b01 frame 0 (the hop onto 32)
        finish(self)

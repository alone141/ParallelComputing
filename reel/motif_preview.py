"""Visual checks for motif.py (NOT part of the timeline).

Stills (PNG lands in reel/build/media/images/motif_preview/):
  PYTHONPATH=reel /home/user/.venvs/manim/bin/manim render -s -r 540,960 --disable_caching \
      --media_dir reel/build/media reel/motif_preview.py StateE02

  StateE01 .. StateE08, StateLOOP   one handoff state each (exactly what beats hand over)
  Tokens                            the 6 handoff_token glyphs, enlarged + at true size
  LayoutB01/B02/B06/B08             representative text + HUD frames, with layout guides
Set MOTIF_GUIDES=1 to overlay the safe zone / STAGE / TTZ / AC guides on the State stills.

Motion test (video or PNG sequence): MotifMotionTest walks E02 -> EXPAND -> HUD slam/roll/
multiplier -> COLLAPSE('cpu') -> SETTLE -> exit_to('E03'); its first frame must equal
StateE02 and its last frame StateE03.
"""

from __future__ import annotations

import os

from style import *
from motif import *

GUIDES = os.environ.get("MOTIF_GUIDES", "0") == "1"


def guides() -> VGroup:
    g = VGroup()
    g.add(Rectangle(width=SAFE_RIGHT - SAFE_LEFT, height=SAFE_TOP - SAFE_BOTTOM,
                    stroke_color=RED_HOT, stroke_width=1.5, stroke_opacity=0.6)
          .move_to(SAFE_CENTER))
    g.add(Rectangle(width=STAGE.w, height=STAGE.h, stroke_color=DIM, stroke_width=1.5,
                    stroke_opacity=0.7).move_to(STAGE.center))
    for y, c in ((TTZ_TOP, LIME), (TTZ_BOTTOM, LIME), (KICKER_Y, CYAN), (HUD_NUM_TOP, AMBER),
                 (HUD_LABEL_Y, AMBER), (STEP_TOP, MAGENTA)):
        g.add(DashedLine([SAFE_LEFT, y, 0], [SAFE_RIGHT, y, 0], stroke_color=c, stroke_width=1,
                         stroke_opacity=0.55, dash_length=0.08))
    g.add(DashedLine([X_C, SAFE_BOTTOM, 0], [X_C, SAFE_TOP, 0], stroke_color=DIM, stroke_width=1,
                     stroke_opacity=0.5, dash_length=0.08))
    g.add(Line(AC + LEFT * 0.15, AC + RIGHT * 0.15, stroke_color=RED_HOT, stroke_width=1.5),
          Line(AC + DOWN * 0.15, AC + UP * 0.15, stroke_color=RED_HOT, stroke_width=1.5))
    return g


class _StateStill(Scene):
    state = "E01"

    def construct(self):
        if GUIDES:
            self.add(guides())
        self.add(handoff_state(self.state))


class StateE01(_StateStill):
    state = "E01"


class StateE02(_StateStill):
    state = "E02"


class StateE03(_StateStill):
    state = "E03"


class StateE04(_StateStill):
    state = "E04"


class StateE05(_StateStill):
    state = "E05"


class StateE06(_StateStill):
    state = "E06"


class StateE07(_StateStill):
    state = "E07"


class StateE08(_StateStill):
    state = "E08"


class StateLOOP(_StateStill):
    state = "LOOP"


class Tokens(Scene):
    """Top: the six glyphs at 2.4x (strokes unscaled) with their kind; bottom row: true size."""

    def construct(self):
        kinds = ["plain", "core", "cpu", "gpu", "node", "cluster"]
        for i, k in enumerate(kinds):
            col, row = i % 2, i // 2
            c = np.array([-2.15 + 4.3 * col, 5.3 - 3.35 * row, 0])
            t = handoff_token(k).scale(2.0, about_point=AC).shift(c - AC)
            lab = mono(k, 34, DIM).next_to(t, DOWN, buff=0.14)
            self.add(t, lab)
        self.add(handoff_token("plain", AMBER).shift(np.array([-3.2, -4.6, 0]) - AC),
                 mono("plain, AMBER (E08)", 34, DIM).move_to([0.6, -4.6, 0]))
        for i, k in enumerate(kinds):
            t = handoff_token(k).shift(np.array([-3.5 + 1.4 * i, -6.5, 0]) - AC)
            self.add(t)


class LayoutB01(Scene):
    """b01 at ~1.2 s: dimmed 'One at a time.' + SLAMmed 'All at once.', lit grid, readout."""

    def construct(self):
        self.add(guides())
        g = hook_state(49)
        arr = g.crawl_arrays(49)
        lit = np.arange(HOOK_N) != 49
        arr["fill"][lit] = rgba(CYAN, 0.75)
        arr["stroke"][lit] = HOOK_LIT_STROKE
        arr["sw"][lit] = 1.5
        g.draw(**arr)
        one = card([("One at a time.", "H96", INK)])
        one.set_opacity(0.40)
        allc = card([("All at once.", "H96", CYAN)], top=below(one[0]))
        self.add(g, one, allc, step_readout("320 WORKERS · 1 STEP", CYAN))


class LayoutB02(Scene):
    """b02 at ~1.6 s: kicker, 2-style card, HUD '16' with the ×16 multiplier."""

    def construct(self):
        self.add(guides())
        self.add(kick("SCALE 1/6 · ONE CORE"))
        self.add(card([("1 instruction.", "H96", INK, {"1": CYAN}), ("16 numbers at once.", "S56", CYAN)]))
        self.add(handoff_token("core"))
        hud = Hud("16", "CALCULATION AT ONCE")
        hud.mult(0.0, "×16")
        hud.render_at(0.5)
        self.add(hud)


class LayoutB06(Scene):
    """b06 at ~3.6 s: '100,000+' climax card, HUD 1,689,600,000+ and ×12,500."""

    def construct(self):
        self.add(guides())
        self.add(kick("SCALE 5/6 · DATA CENTER"))
        self.add(card([("100,000+", "H96", CYAN), ("GPUs, training one AI.", "S56", INK)]))
        self.add(handoff_token("cluster"))
        hud = Hud("1,689,600,000+", CALC_LABEL)
        hud.mult(0.0, "×12,500")
        hud.render_at(0.5)
        self.add(hud)


class LayoutB08(Scene):
    """b08 at ~6.5 s: the screenshot frame (catch kicker at tracking 80, takeaway card, HUD 10×)."""

    def construct(self):
        self.add(guides())
        self.add(kick("THE CATCH · AMDAHL'S LAW", AMBER, tracking=80))
        self.add(card([("10% serial", "H84", INK, {"10% serial": AMBER}),
                       ("= max 10×.", "H84", INK, {"10×": AMBER})]))
        self.add(handoff_token("plain", AMBER))
        self.add(Hud("10×", "MAX SPEEDUP · 10% SERIAL", AMBER, AMBER, 0.85))


class MotifMotionTest(BeatScene):
    """A mini b03 built only from motif helpers (not in the timeline; no dust so frames diff
    exactly against the State stills)."""

    with_dust = False

    def construct(self):  # no timeline padding, no cue file
        self.beat()

    def beat(self):
        st = enter(self, "E02")
        hud = st.hud
        # world at rest: 16 cores with their lane glyphs + labels + memory bar
        xs, ys = (-2.15, -0.95, 0.25, 1.45), (2.75, 1.55, 0.35, -0.85)
        cores, labels = VGroup(), VGroup()
        for i in range(16):
            c = np.array([xs[i % 4], ys[i // 4], 0])
            cores.add(unit_with_glyph("core", 0.95, c, glyph_fill=0.35))
            labels.add(mono(f"T{i + 1}", 34, INK).set_opacity(0.9).move_to(c + UP * 0.21))
        bar = RoundedRectangle(width=6.4, height=0.5, corner_radius=0.12, stroke_color=MAGENTA,
                               stroke_width=3, fill_color=MAGENTA, fill_opacity=0.10).move_to([X_C, -1.95, 0])
        bar_lab = kick("SHARED MEMORY", MAGENTA).move_to(bar)
        world = VGroup(cores, labels, bar, bar_lab)
        hero = cores[5]
        track = TextTrack(self)
        self.add(track)
        track.on(kick("SCALE 2/6 · ONE CPU"), 0.20, 3.00)
        head = card([("16 cores.", "H96", INK, {"16": CYAN}), ("× 16 lanes each", "S56", CYAN)])
        track.on(head[0], 0.30, 3.00)
        track.on(head[1], 1.50, 3.00, enter="cut")
        hud.slam(1.50, "256", mult="×16")
        hud.pulse(2.60, 1.06)
        exp = Expand(world, hero.unit.get_center(), 0.95, run_time=0.70, hero=hero, token="core",
                     fades={labels: (0.3, 0.8)})
        add_behind(self, world)      # content behind token + HUD
        entry_ring(self)             # ring behind content, from frame 1
        self.remove(st.token)
        self.cue("hit")
        self.play(exp)
        self.hold_until(3.20)
        pkg = RoundedRectangle(width=6.9, height=5.6, corner_radius=0.35, stroke_color=CYAN,
                               stroke_width=3).move_to([X_C, 0.50, 0])
        self.add(pkg)
        col = COLLAPSE(pkg, world, "cpu")
        self.play(col.anim(0.45))
        self.play(col.settle_anim())
        exit_to(self, "E03")
        self.wait(0.2)


class LoopSeamTest(BeatScene):
    """The b09 ending built dynamically (frame-clock crawl, POP headline, live readout) must end
    pixel-identical to StateLOOP. Capture at 60 fps: story frame = round(now*60) + 174, so the
    last of its 36 frames is b09's f209 (idx 32). No dust, no timeline slot."""

    with_dust = False

    def construct(self):
        self.beat()

    def beat(self):
        off = 174
        sf = lambda now: story_frame(now) + off
        grid = HookGrid().set_crawl(b09_worker_idx(sf(0.0)))
        grid.add_updater(lambda m, dt: m.set_crawl(b09_worker_idx(sf(self.now)))
                         if m.worker_idx != b09_worker_idx(sf(self.now)) else None)
        self.add(grid)
        track = TextTrack(self)
        self.add(track)
        head = card([("One at a time.", "H96", INK)])[0]
        track.on(head, 0.0, None, enter="pop", exit=None)
        track.on(loop_kicker(), 0.0, None, enter="pop", exit=None)  # LOOP now carries the kicker
        ro = LiveText(lambda now: step_str(b09_worker_idx(sf(now))),
                      lambda s: step_readout_tmpl(s, AMBER), scene=self,
                      scale=lambda now: pop_scale(now))
        self.add(ro)
        self.wait(0.6 - 0.5 / 60)

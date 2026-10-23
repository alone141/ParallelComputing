"""Visual checks for motif.py (NOT part of the timeline).

Stills (PNG lands in reel/build/media/images/motif_preview/):
  PYTHONPATH=reel /home/user/.venvs/manim/bin/manim render -s -r 540,960 --disable_caching \
      --media_dir reel/build/media reel/motif_preview.py StateE03

  StateE01 StateE03 StateE04 StateE06 StateE08 StateLOOP
                         one handoff state each (exactly what the script v2 beats hand over:
                         b01 -E01-> b03 -E03-> b04 -E04-> b06 -E06-> b08 -E08-> b09 -LOOP-> b01)
  Tokens                 the handoff_token glyphs the film uses (plain, cpu, gpu, cluster, AMBER
                         plain), enlarged + at true size
  LayoutB01/B03/B04/B06/B08   representative text + HUD frames of the new script, with guides
  TearContact            b08's tear-away of E06 (HUD + cluster token) at 7 times 0.00-0.30 s
Set MOTIF_GUIDES=1 to overlay the safe zone / STAGE / TTZ / AC guides on the State stills.

Motion tests (video or PNG sequence; no dust, so frames diff exactly against the State stills):
  MotifMotionTest  E01 -> EXPAND (plain token = core T6) -> HUD POP '1' (singular label) -> SLAM
                   '16' + plural label + LIME line -> COLLAPSE('cpu') -> SETTLE -> exit_to('E03');
                   its first frame must equal StateE01 and its last frame StateE03.
  TearMotionTest   E06 -> tear_away(0.00-0.30) -> black; first frame == StateE06, BG from 0.30.
  LoopSeamTest     the b09 ending built dynamically must end pixel-identical to StateLOOP.
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
    """b01 -> b03: the plain CYAN token, no HUD."""
    state = "E01"


class StateE03(_StateStill):
    """b03 -> b04: cpu token + '16' / CORES WORKING AT ONCE."""
    state = "E03"


class StateE04(_StateStill):
    """b04 -> b06: gpu token + '16,896' / CORES WORKING AT ONCE."""
    state = "E04"


class StateE06(_StateStill):
    """b06 -> b08: cluster token + '1,689,600,000+' / CORES WORKING AT ONCE."""
    state = "E06"


class StateE08(_StateStill):
    """b08 -> b09: the AMBER serial token + '10×' / MAX SPEEDUP · 10% SERIAL."""
    state = "E08"


class StateLOOP(_StateStill):
    """b09 -> b01: hook grid (worker on 32) + kicker + 'One at a time.' + STEP 032 / 320."""
    state = "LOOP"


class Tokens(Scene):
    """Top: the film's glyphs at 2x (strokes unscaled) with their handoff; bottom row: true size."""

    def construct(self):
        kinds = [("plain", CYAN, "plain (E01)"), ("cpu", CYAN, "cpu (E03)"),
                 ("gpu", CYAN, "gpu (E04)"), ("cluster", CYAN, "cluster (E06)")]
        for i, (k, col, name) in enumerate(kinds):
            c = np.array([-2.15 + 4.3 * (i % 2), 5.2 - 3.75 * (i // 2), 0])
            t = handoff_token(k, col).scale(1.5, about_point=AC).shift(c - AC)
            lab = mono(name, 34, DIM).next_to(t, DOWN, buff=0.14)
            self.add(t, lab)
        self.add(handoff_token("plain", AMBER).shift(np.array([-3.0, -2.2, 0]) - AC),
                 mono("plain, AMBER (E08)", 34, DIM).move_to([0.9, -2.2, 0]))
        self.add(mono("true size", 30, DIM).move_to([0.0, -3.55, 0]))
        for i, (k, col, _) in enumerate(kinds):
            t = handoff_token(k, col).shift(np.array([-3.0 + 2.0 * i, -4.8, 0]) - AC)
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


class LayoutB03(Scene):
    """b03 at ~0.5 s: LEVEL 1/3 kicker, 'That worker? / One CPU core.', the counter's first
    appearance with the SINGULAR label."""

    def construct(self):
        self.add(guides())
        self.add(kick("LEVEL 1/3 · ONE CPU"))
        self.add(card([("That worker?", "H84", INK), ("One CPU core.", "H84", INK, {"CPU core.": CYAN})]))
        self.add(handoff_token("plain"))
        self.add(core_hud("1"))


class LayoutB03Power(Scene):
    """b03 at ~4.0 s: '16 cores. / 16 different jobs.' with '16' and its LIME line."""

    def construct(self):
        self.add(guides())
        self.add(kick("LEVEL 1/3 · ONE CPU"))
        self.add(card([("16 cores.", "H96", INK, {"16": CYAN}), ("16 different jobs.", "S56", INK)]))
        self.add(handoff_token("cpu"))
        hud = Hud("16", core_label(16))
        hud.mult(0.0, COUNTER_LINES["B03Cpu"], hold=1.8)
        hud.render_at(0.5)
        self.add(hud)


class LayoutB04(Scene):
    """b04 at ~3.6 s: 'Thousands of / tiny cores.', HUD 16,896 + 'H100: 16,896 CUDA CORES'."""

    def construct(self):
        self.add(guides())
        self.add(kick("LEVEL 2/3 · ONE GPU"))
        self.add(card([("Thousands of", "H84", INK), ("tiny cores.", "H84", INK, {"tiny cores.": CYAN})]))
        self.add(handoff_token("gpu"))
        hud = Hud("16,896", CALC_LABEL)
        hud.mult(0.0, COUNTER_LINES["B04Gpu"], hold=1.8)
        hud.render_at(0.5)
        self.add(hud)


class LayoutB06(Scene):
    """b06 at ~7.6 s: the biggest drop: '100,000+ / GPUs. One AI.', HUD 1,689,600,000+ and the
    '100,000+ GPUs × 16,896' line."""

    def construct(self):
        self.add(guides())
        self.add(kick("LEVEL 3/3 · DATA CENTER"))
        self.add(card([("100,000+", "H96", CYAN), ("GPUs. One AI.", "S56", INK)]))
        self.add(handoff_token("cluster"))
        hud = Hud("1,689,600,000+", CALC_LABEL)
        hud.mult(0.0, COUNTER_LINES["B06DataCenter"], hold=1.8)
        hud.render_at(0.5)
        self.add(hud)


class LayoutB08(Scene):
    """b08 at ~9.5 s: the screenshot frame (FINAL BOSS kicker at tracking 80, takeaway, HUD 10×)."""

    def construct(self):
        self.add(guides())
        self.add(kick("FINAL BOSS: AMDAHL'S LAW", AMBER, tracking=80))
        self.add(card([("10% serial", "H84", INK, {"10% serial": AMBER}),
                       ("= max 10×.", "H84", INK, {"10×": AMBER})]))
        self.add(handoff_token("plain", AMBER))
        self.add(Hud("10×", "MAX SPEEDUP · 10% SERIAL", AMBER, AMBER, 0.85))


class TearContact(Scene):
    """b08's opening tear-away of E06 at 0.000 / 0.050 / 0.100 / 0.150 / 0.200 / 0.267 / 0.300 s
    (story frames 0, 3, 6, 9, 12, 16, 18), each frame shrunk to a third, left to right, top to
    bottom. 0.000 must be the exact E06; 0.300 must be empty."""

    TIMES = (0.0, 0.05, 0.10, 0.15, 0.20, 16 / 60, 0.30)

    def construct(self):
        k = 0.30
        for i, t in enumerate(self.TIMES):
            st = handoff_state("E06")
            st.hud.tear(0.0, TEAR_TIME)
            tear = Tear(st.token, 0.0, TEAR_TIME)
            st.hud.render_at(t)
            tear.render_at(t)
            frame = VGroup(tear, st.hud).copy()
            box = Rectangle(width=9.0, height=16.0, stroke_color=DIM, stroke_width=1.5)
            cell = VGroup(box, frame)
            cell.scale(k, about_point=ORIGIN)
            c = np.array([-3.0 + 3.0 * (i % 3), 5.45 - 5.3 * (i // 3), 0])
            cell.shift(c)
            lab = mono(f"{t:.3f}s  f{story_frame(t)}", 26, DIM).next_to(box, DOWN, buff=0.05)
            self.add(cell, lab)


class MotifMotionTest(BeatScene):
    """A mini b03 built only from motif helpers (not in the timeline; no dust so frames diff
    exactly against the State stills): E01 -> core T6 -> '1' (singular) -> '16' -> E03."""

    with_dust = False

    def construct(self):  # no timeline padding, no cue file
        self.beat()

    def beat(self):
        st = enter(self, "E01")
        assert st.hud is None
        hud = core_hud("1", scene=self, pop_at=0.10)
        # world at rest: 16 PLAIN cores (no SIMD lane glyphs) + labels + memory bar
        xs, ys = (-2.15, -0.95, 0.25, 1.45), (2.75, 1.55, 0.35, -0.85)
        cores, labels = VGroup(), VGroup()
        for i in range(16):
            c = np.array([xs[i % 4], ys[i // 4], 0])
            cores.add(unit_with_glyph("plain", 0.95, c))
            labels.add(mono(f"T{i + 1}", 34, INK).set_opacity(0.9).move_to(c + UP * 0.21))
        bar = RoundedRectangle(width=6.4, height=0.5, corner_radius=0.12, stroke_color=MAGENTA,
                               stroke_width=3, fill_color=MAGENTA, fill_opacity=0.10).move_to([X_C, -1.95, 0])
        bar_lab = kick("SHARED MEMORY", MAGENTA).move_to(bar)
        world = VGroup(cores, labels, bar, bar_lab)
        hero = cores[5]
        track = TextTrack(self)
        self.add(track)
        track.on(kick("LEVEL 1/3 · ONE CPU"), 0.0, 3.00, exit="cut")
        head = card([("16 cores.", "H96", INK, {"16": CYAN}), ("16 different jobs.", "S56", INK)])
        track.on(head[0], 0.30, 3.00, exit="cut")
        track.on(head[1], 0.30, 3.00, enter="cut", exit="cut")
        hud.slam(1.50, "16")
        hud.label(1.50, core_label(16))
        hud.mult(1.50, COUNTER_LINES["B03Cpu"], hold=1.0)
        exp = Expand(world, hero.unit.get_center(), 0.95, run_time=0.70, hero=hero, token="plain",
                     fades={labels: (0.3, 0.8)})
        add_behind(self, world)      # content behind token + HUD
        entry_ring(self)             # ring behind content, from frame 1
        self.remove(st.token)
        self.add(hud)
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


class TearMotionTest(BeatScene):
    """b08's opening on E06: tear_away + scratch_cut, then black. Frame 0 == StateE06; every
    frame from 0.30 is BG. No dust, no timeline slot."""

    with_dust = False

    def construct(self):
        self.beat()

    def beat(self):
        st = enter(self, "E06")
        tear_away(self, st, 0.0, TEAR_TIME)
        scratch_cut(self)
        self.wait(0.5)


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

"""Visual checks for motif.py (NOT part of the timeline). Script v3: calm motion.

Stills (PNG lands in reel/build/media/images/motif_preview/):
  PYTHONPATH=reel /home/user/.venvs/manim/bin/manim render -s -r 540,960 --disable_caching \
      --media_dir reel/build/media reel/motif_preview.py StateE03

  StateE01 StateE03 StateE04 StateE06 StateE08 StateLOOP
                         one handoff state each (exactly what the beats hand over:
                         b01 -E01-> b03 -E03-> b04 -E04-> b06 -E06-> b08 -E08-> b09 -LOOP-> b01;
                         unchanged by script v3, so the seams stay bit-exact)
  Tokens                 the handoff_token glyphs the film uses (plain, cpu, gpu, cluster, AMBER
                         plain), enlarged + at true size
  LayoutB01/B03/B03Power/B04/B06/B08
                         representative text + HUD frames of script v3 (v3 card texts), with guides
  FadeContact            b08's fade-away of E06 (HUD + cluster token, together, constant value) at
                         0.00 / 0.10 / ... / 0.80 s: 0.00 must be the exact E06, 0.80 empty
  CutCurves              the cut rhythm, v3 (colour) vs v2 (grey): COLLAPSE progress, SETTLE scale
                         (no overshoot), EXPAND progress, and the E06 fade vs the v2 tear's flicker
Set MOTIF_GUIDES=1 to overlay the safe zone / STAGE / TTZ / AC guides on the State stills.

Motion tests (PNG sequence: add --format png; no dust, so frames diff exactly against the stills):
  MotifMotionTest  E01 -> Expand (0.90 s sine: plain token = core T6) -> HUD fades in '1' (singular
                   label, core_hud(fade_in=)) -> SLAM '16' + plural label + LIME line ->
                   COLLAPSE('cpu') 0.80 s sine -> SETTLE 0.15 s cubic -> exit_to('E03') on the
                   cut_times() grid; its first frame must equal StateE01 and its last StateE03.
  FadeMotionTest   E06 -> fade_away(0.00-0.80) + swell_cut -> black; first frame == StateE06,
                   BG from 0.80, and the HUD number never changes while it is seen.
  LoopSeamTest     the b09 ending built dynamically must end pixel-identical to StateLOOP.

Self-test (pure, no render): PYTHONPATH=reel /home/user/.venvs/manim/bin/python reel/motif_preview.py
  asserts the v3 cut rhythm (cut_times, eases, no settle overshoot), the fade clock, CALM_MOTION
  (shake / jitter / glitch neutral) and the audio helpers' cues.
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
    """b01 at ~1.2 s (LOCKED text): dimmed 'One at a time.' + SLAMmed 'All at once.', lit grid,
    readout."""

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
    """b03 at ~1.0 s (script v3 + user decision): LEVEL 1/3 kicker, 'That worker? / It's one CPU
    core.', the counter's first appearance (faded in 0.30-0.60) with the SINGULAR label."""

    def construct(self):
        self.add(guides())
        self.add(kick("LEVEL 1/3 · ONE CPU"))
        self.add(card([("That worker?", "H96", INK), ("It's one CPU core.", "S56", INK, {"CPU core.": CYAN})]))
        self.add(handoff_token("plain"))
        self.add(core_hud("1"))


class LayoutB03Power(Scene):
    """b03 at ~4.5 s: '16 cores. / Each runs its own task.' with '16' and its LIME line."""

    def construct(self):
        self.add(guides())
        self.add(kick("LEVEL 1/3 · ONE CPU"))
        self.add(card([("16 cores.", "H96", INK, {"16": CYAN}), ("Each runs its own task.", "S56", INK)]))
        self.add(handoff_token("cpu"))
        hud = Hud("16", core_label(16))
        hud.mult(0.0, COUNTER_LINES["B03Cpu"], hold=1.8)
        hud.render_at(0.5)
        self.add(hud)


class LayoutB04(Scene):
    """b04 at ~2.5 s: '16? Cute. / A GPU has thousands of / tiny cores.', HUD 16,896 + 'H100:
    16,896 CUDA CORES'."""

    def construct(self):
        self.add(guides())
        self.add(kick("LEVEL 2/3 · ONE GPU"))
        self.add(card([("16? Cute.", "H84", INK, {"16?": CYAN}), ("A GPU has thousands of", "S56", INK),
                       ("tiny cores.", "S56", INK, {"tiny cores.": CYAN})]))
        self.add(handoff_token("gpu"))
        hud = Hud("16,896", CALC_LABEL)
        hud.mult(0.0, COUNTER_LINES["B04Gpu"], hold=1.8)
        hud.render_at(0.5)
        self.add(hud)


class LayoutB06(Scene):
    """b06 at ~8.6 s: the biggest drop: '100,000+ / GPUs train one AI. / Each works on a slice.',
    HUD 1,689,600,000+ and the '100,000+ GPUs × 16,896' line."""

    def construct(self):
        self.add(guides())
        self.add(kick("LEVEL 3/3 · DATA CENTER"))
        self.add(card([("100,000+", "H84", CYAN), ("GPUs train one AI.", "S56", INK),
                       ("Each works on a slice.", "S56", INK)]))
        self.add(handoff_token("cluster"))
        hud = Hud("1,689,600,000+", CALC_LABEL)
        hud.mult(0.0, COUNTER_LINES["B06DataCenter"], hold=1.8)
        hud.render_at(0.5)
        self.add(hud)


class LayoutB08(Scene):
    """b08 at ~8.0 s: the screenshot frame (FINAL BOSS kicker at tracking 80, takeaway, HUD 10×)."""

    def construct(self):
        self.add(guides())
        self.add(kick("FINAL BOSS: AMDAHL'S LAW", AMBER, tracking=80))
        self.add(card([("10% serial", "H84", INK, {"10% serial": AMBER}),
                       ("= max 10×.", "H84", INK, {"10×": AMBER})]))
        self.add(handoff_token("plain", AMBER))
        self.add(Hud("10×", "MAX SPEEDUP · 10% SERIAL", AMBER, AMBER, 0.85))


class FadeContact(Scene):
    """b08's opening fade-away of E06 (fade_away: the cluster token and the HUD together, opacity
    only, constant value) at 0.00, 0.10, ..., 0.80 s, each frame shrunk to 0.30, left to right, top
    to bottom. 0.00 must be the exact E06; 0.80 must be empty; the number never changes."""

    TIMES = tuple(k / 10 for k in range(9))

    def construct(self):
        k = 0.30
        for i, t in enumerate(self.TIMES):
            st = handoff_state("E06")
            st.hud.fade_out(0.0, FADE_AWAY_TIME)
            fade = FadeAway(st.token, 0.0, FADE_AWAY_TIME)
            st.hud.render_at(t)
            fade.render_at(t)
            assert st.hud.number_at(t)[0] == "1,689,600,000+"
            frame = VGroup(fade, st.hud).copy()
            box = Rectangle(width=9.0, height=16.0, stroke_color=DIM, stroke_width=1.5)
            cell = VGroup(box, frame)
            cell.scale(k, about_point=ORIGIN)
            c = np.array([-3.0 + 3.0 * (i % 3), 5.45 - 5.3 * (i // 3), 0])
            cell.shift(c)
            lab = mono(f"{t:.2f}s  a={fade_at(t):.2f}", 26, DIM).next_to(box, DOWN, buff=0.05)
            self.add(cell, lab)


class CutCurves(Scene):
    """The cut rhythm, script v3 (colour) vs v2 (grey): 1 COLLAPSE progress
    and 2 SETTLE scale against time before the exit frame (both end on the same exit, the boundary
    frame never moves), 3 EXPAND progress from the cut, 4 the E06 fade (fade_at) vs the v2 tear's
    alpha flicker (tear_at)."""

    X0, X1, H = -3.55, 3.15, 2.30

    def panel(self, yc, title, t0, t1, y0, y1, curves, ticks, marks=()):
        X0, X1, H = self.X0, self.X1, self.H
        g = VGroup()
        box = Rectangle(width=X1 - X0, height=H, stroke_color=DIM, stroke_width=1.5).move_to(
            [(X0 + X1) / 2, yc, 0])
        g.add(box)
        g.add(mono(title, 22, INK).next_to(box, UP, buff=0.10).align_to(box, LEFT))

        def P(t, v):
            return np.array([X0 + (t - t0) / (t1 - t0) * (X1 - X0), yc - H / 2 + (v - y0) / (y1 - y0) * H, 0])

        for v, lab in marks:                       # horizontal reference lines
            g.add(DashedLine(P(t0, v), P(t1, v), stroke_color=DIM, stroke_width=1, dash_length=0.06))
            g.add(mono(lab, 20, DIM).next_to(P(t0, v), LEFT, buff=0.06))
        for t in ticks:                            # time ticks
            g.add(Line(P(t, y0), P(t, y0) + UP * 0.08, stroke_color=DIM, stroke_width=1.5))
            g.add(mono(f"{t:+.2f}" if t0 < 0 else f"{t:.1f}", 18, DIM).next_to(P(t, y0), DOWN, buff=0.04))
        for fn, color, width in curves:
            ts = np.linspace(t0, t1, 600)
            pts = [P(t, fn(t)) for t in ts]
            g.add(VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(pts))
        self.add(g)

    def construct(self):
        # 1 collapse progress e, time relative to the exit frame (t_exit = 0)
        v3c = lambda t: COLLAPSE_EASE(ramp(t, -SETTLE_TIME - COLLAPSE_TIME, -SETTLE_TIME))
        v2c = lambda t: COLLAPSE_EASE_V2(ramp(t, -SETTLE_TIME_V2 - COLLAPSE_TIME_V2, -SETTLE_TIME_V2))
        self.panel(6.05, "COLLAPSE e  v3 0.80s sine | v2 0.40s expo", -1.05, 0.05, -0.05, 1.05,
                   [(v2c, DIM, 3), (v3c, CYAN, 4)], (-1.0, -0.5, 0.0))

        # 2 settle: token scale relative to its full size (0.94 -> 1.0), t_exit = 0
        def sc(t, dur, ease):
            if t < -dur:
                return SETTLE_FROM
            return SETTLE_FROM * settle_scale(ramp(t, -dur, 0.0), ease)
        v3s = lambda t: sc(t, SETTLE_TIME, SETTLE_EASE)
        v2s = lambda t: sc(t, SETTLE_TIME_V2, SETTLE_EASE_V2)
        self.panel(2.20, "SETTLE  v3 0.15s cubic | v2 0.12s back", -0.25, 0.05, 0.93, 1.015,
                   [(v2s, DIM, 3), (v3s, LIME, 4)], (-0.2, -0.1, 0.0),
                   marks=((1.0, "1.00"), (SETTLE_FROM, "0.94")))

        # 3 expand progress e from the cut (t = 0)
        v3e = lambda t: expand_e(t)
        v2e = lambda t: expand_e(t, dur=EXPAND_TIME_V2, ease=EXPAND_EASE_V2)
        self.panel(-1.65, "EXPAND e  v3 0.90s sine | v2 0.50s expo", -0.05, 1.05, -0.05, 1.05,
                   [(v2e, DIM, 3), (v3e, MAGENTA, 4)], (0.0, 0.5, 1.0))

        # 4 E06 opening: v3 fade (token + HUD alpha) vs v2 tear alpha (30 Hz flicker)
        v3f = lambda t: fade_at(t, 0.0, FADE_AWAY_TIME)
        v2f = lambda t: tear_at(t, 0.0, TEAR_TIME)[0]
        self.panel(-5.50, "E06 alpha  v3 fade 0.80s | v2 tear 0.30s", -0.05, 1.05, -0.05, 1.05,
                   [(v2f, DIM, 3), (v3f, AMBER, 4)], (0.0, 0.5, 1.0))


class MotifMotionTest(BeatScene):
    """A mini b03 built only from motif helpers, on the script v3 defaults (not in the timeline;
    no dust so frames diff exactly against the State stills): E01 -> core T6 (Expand 0.90 s sine)
    -> '1' fades in (singular) -> SLAM '16' -> COLLAPSE 0.80 s sine + SETTLE 0.15 s cubic -> E03,
    on cut_times(T_END). Kicker + card hold through the collapse and hard-cut at the exit."""

    with_dust = False
    T_END = 3.30                                 # cut_times(3.30) -> (2.30, 3.10, 3.25)

    def construct(self):  # no timeline padding, no cue file
        self.beat()

    def beat(self):
        t_col, t_set, t_exit = cut_times(self.T_END)
        st = enter(self, "E01")
        assert st.hud is None
        hud = core_hud("1", scene=self, fade_in=(0.30, 0.60))
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
        track.on(kick("LEVEL 1/3 · ONE CPU"), 0.0, t_exit, exit="cut")
        head = card([("16 cores.", "H96", INK, {"16": CYAN}), ("Each runs its own task.", "S56", INK)])
        track.on(head[0], 1.00, t_exit, enter="pop", exit="cut")
        track.on(head[1], 1.00, t_exit, enter="pop", exit="cut")
        hud.slam(1.50, "16")
        hud.label(1.50, core_label(16))
        hud.mult(1.50, COUNTER_LINES["B03Cpu"], hold=0.70)
        exp = Expand(world, hero.unit.get_center(), 0.95, hero=hero, token="plain",
                     fades={labels: (0.3, 0.8)})   # EXPAND_TIME 0.90, EXPAND_EASE
        add_behind(self, world)      # content behind token + HUD
        entry_ring(self)             # ring behind content, from frame 1
        self.remove(st.token)
        self.add(hud)
        self.cue("hit")
        self.play(exp)
        self.hold_until(t_col)
        pkg = RoundedRectangle(width=6.9, height=5.6, corner_radius=0.35, stroke_color=CYAN,
                               stroke_width=3).move_to([X_C, 0.50, 0])
        add_behind(self, pkg)
        col = COLLAPSE(pkg, world, "cpu")
        self.add(col.glyph)
        drive(self, lambda now: col.render(now, t_col, t_set, t_exit))
        hold_floor(self, t_exit)
        exit_to(self, "E03")
        self.wait(0.2)


class FadeMotionTest(BeatScene):
    """b08's opening on E06 (script v3): fade_away (token + HUD, together, 0.00-0.80, constant
    value) + swell_cut, then black. Frame 0 == StateE06; every frame from 0.80 is BG. No dust, no
    timeline slot."""

    with_dust = False

    def construct(self):
        self.beat()

    def beat(self):
        st = enter(self, "E06")
        fade_away(self, st, 0.0, FADE_AWAY_TIME)
        swell_cut(self)
        assert [c["type"] for c in self._cues] == ["whoosh"], self._cues
        self.wait(1.0)


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


# --------------------------------------------------------------------------
# pure self-test (no render)
# --------------------------------------------------------------------------
def _selftest() -> None:
    near = lambda a, b: all(abs(x - y) < 1e-9 for x, y in zip(a, b))
    # 1. the v3 cut rhythm (script v3: 0.80 s collapse sine, 0.15 s settle cubic, 0.90 s expand sine)
    assert (COLLAPSE_TIME, SETTLE_TIME, EXPAND_TIME) == (0.80, 0.15, 0.90)
    assert (COLLAPSE_EASE, SETTLE_EASE, EXPAND_EASE) == (ease_in_out_sine, ease_out_cubic, ease_in_out_sine)
    assert near(cut_times(9.5), (8.50, 9.30, 9.45)), cut_times(9.5)
    assert near(cut_times(12.0), (11.00, 11.80, 11.95)), cut_times(12.0)
    assert near(cut_times(12.5), (11.50, 12.30, 12.45)), cut_times(12.5)
    assert near(cut_times_v2(5.0), (266 / 60, 290 / 60, 4.95)), cut_times_v2(5.0)   # b01's push window
    for T in (5.0, 9.5, 12.0, 12.5):                     # the exit frame never moves
        assert cut_times(T)[2] == cut_times_v2(T)[2]
    # settle: monotonic, never above the token (no overshoot), exact ends; v2 overshot
    ss = [settle_scale(i / 600) for i in range(601)]
    assert ss[0] == 1.0 and ss[-1] == 1.0 / SETTLE_FROM
    assert all(b >= a for a, b in zip(ss, ss[1:])) and max(ss) <= 1.0 / SETTLE_FROM
    assert max(settle_scale(i / 600, SETTLE_EASE_V2) for i in range(601)) > 1.0 / SETTLE_FROM
    # expand / collapse progress: exact 0 and 1 at the ends, monotonic
    assert expand_e(0.0) == 0.0 and expand_e(EXPAND_TIME) == 1.0 and expand_e(5.0) == 1.0
    ee = [expand_e(i / 60) for i in range(55)]
    assert all(b >= a for a, b in zip(ee, ee[1:]))
    col = COLLAPSE(token(2.0), None, "cpu")
    assert col.ease is COLLAPSE_EASE and col.settle_ease is SETTLE_EASE
    assert col.e_at(8.50, 8.50, 9.30) == 0.0 and col.e_at(9.30, 8.50, 9.30) == 1.0
    # 2. the fade-away clock + Hud.fade_out: exact ends, monotonic, HUD and token together, the
    #    number never changes
    assert fade_at(0.0) == 1.0 and fade_at(FADE_AWAY_TIME) == 0.0 and fade_at(2.0) == 0.0
    st = handoff_state("E06")
    st.hud.fade_out(0.0, FADE_AWAY_TIME)
    fa = FadeAway(st.token, 0.0, FADE_AWAY_TIME)
    prev = 2.0
    for f in range(0, 61):
        t = f / 60
        env = st.hud._env_at(t)[0]
        assert abs(env - fade_at(t)) < 1e-12 and abs(fa.alpha_at(t) - fade_at(t)) < 1e-12, t
        assert env <= prev
        prev = env
        assert st.hud.number_at(t)[0] == "1,689,600,000+" and st.hud.label_at(t)[0] == CALC_LABEL
    # 3. CALM_MOTION: shakes, jitter and glitches are neutral; calm=False restores v2
    assert CALM_MOTION
    assert not np.any(shake(0.05, 0.16, 0.30, 6)) and np.any(shake(0.05, 0.16, 0.30, 6, calm=False))
    assert jitter(53, 0.03, 4) == 0.0 and jitter(53, 0.03, 4, calm=False) != 0.0
    assert glitch_at(0.10, 0.0, 0.30, 0.09, 8, (6, 14)) == (0.0, None, False)
    assert glitch_at(0.10, 0.0, 0.30, 0.09, 8, (6, 14), calm=False) != (0.0, None, False)
    h = Hud("16", CALC_LABEL)
    h.glitch(0.0, 0.5, split_frames=(6,))
    assert h._glitch_at(0.10) == (0.0, None, False)
    # 4. core_hud(fade_in=): hidden until t0, opacity only, exactly on from t1
    h = core_hud("1", fade_in=(0.30, 0.60))
    assert h._env_at(0.0)[0] == 0.0 and h._env_at(0.29)[0] == 0.0 and h._env_at(0.60)[0] == 1.0
    assert 0.0 < h._env_at(0.45)[0] < 1.0 and h._scale_at(0.45) == 1.0
    # 5. TextTrack 'fade': opacity only
    tr = TextTrack()
    m = tr.on(kick("SPEEDUP"), 1.0, 2.0, enter="fade", exit="fade")
    it = tr.items[-1]
    a, s, sh = TextTrack._state(it, 1.15)
    assert 0 < a < 1 and s == 1.0 and sh is None
    assert TextTrack._state(it, 1.30)[0] == 1.0 and TextTrack._state(it, 2.30)[0] == 0.0
    # 6. audio: swell_cut = one soft whoosh down, peak on the seam; scratch_cut is its alias;
    #    riser_to_cut crests on the seam at 0.6
    class _Cues:
        t = 0.0

        def __init__(self):
            self.cues = []

        def cue(self, kind, dt=0.0, gain=1.0, note="", **kw):
            self.cues.append(dict(type=kind, dt=dt, gain=gain, **kw))

    class B08Amdahl(_Cues):
        pass

    class B06DataCenter(_Cues):
        pass

    s8 = B08Amdahl()
    swell_cut(s8)
    scratch_cut(s8)
    for c in s8.cues:
        assert (c["type"], c["direction"], c["dur"], c["dt"]) == ("whoosh", "down", 1.6, 0.0), c
        assert c["gain"] == SWELL_GAIN
    s6 = B06DataCenter()
    s6.t = 11.5
    assert riser_to_cut(s6) == 1.0 and s6.cues[0]["gain"] == RISER_CREST_GAIN
    # 7. the boundary states are untouched (cheap structural check; check_handoffs does pixels)
    assert handoff_state("E06").hud.number_at(0.0)[0] == "1,689,600,000+"
    print("motif_preview self-test: OK")


if __name__ == "__main__":
    _selftest()

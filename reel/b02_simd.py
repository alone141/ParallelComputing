"""b02 - SCALE 1/6 - ONE CORE: one instruction, 16 numbers (SIMD).

The incoming plain token opens into a CPU core. Sixteen lanes hold the digits of pi
(A, cyan) and e (B, ink). A scalar add fires on lane 0 only, then ONE instruction
(vaddps zmm0, zmm1, zmm2) drops all 16 B bars onto their A bars on the same frame:
the light wave, HUD 1 -> 16. Five metronomic lockstep pulses, 'That's SIMD.', then the
core collapses into handoff_token('core') (E02).

Everything that moves is a pure function of self.now (one driver updater); the scene
only holds until the structural switch points (collapse capture, canonical exit).
"""

from style import *
from motif import *

# --------------------------------------------------------------------------
# World layout (storyboard b02 visuals)
# --------------------------------------------------------------------------
N_LANES = 16
LANE_X = np.array([X_C + (i - 7.5) * 0.38 for i in range(N_LANES)])
A_DIG = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]   # pi
B_DIG = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5]   # e
BASE_Y = -1.55
HANG_Y = 2.10                       # top edge of the hanging B bars / top of the guides
BAR_W = 0.30
BAR_R = 0.05
UNIT_H = 0.16
A_H = np.array([UNIT_H * a for a in A_DIG])
B_H = np.array([max(UNIT_H * b, 0.03) for b in B_DIG])      # b_13 = 0 -> 0.03 sliver
BEAM_Y = 2.62
BEAM_X0, BEAM_X1 = -3.45, 2.75
FRAME_W, FRAME_H, FRAME_R = 6.8, 5.5, 0.35
INSTR = "vaddps zmm0,zmm1,zmm2"
INSTR_Y = 2.90

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
T_OPEN = 0.45                       # token -> core frame
T_BUILD = 0.15                      # base line / guides / A bars start
A_GROW = 0.20                       # A bar grow (ease_out_back)
GUIDE_GROW = 0.26
PAIR_STAGGER = 0.024                # = 0.012 s per lane, centre outwards (pairs land together)
B_IN0, B_IN_STEP, B_IN_DUR = 0.40, 0.012, 0.22
T_SCALAR = 0.70                     # scalar fire on lane 0
T_SCALAR_LAND = 0.85
T_LIFT0, T_LIFT1 = 1.00, 1.15
T_SIMD = 1.30                       # instruction issued
T_LAND = 1.50                       # all 16 lanes land on this frame
T_LIME = 1.65
PULSES = (1.95, 2.40, 2.85, 3.30, 3.75)
PULSE_DUR = 0.18
T_CARD1_OUT = 2.85
T_TEXT_OUT = 4.05                   # instruction + digits fade 4.05 -> 4.15
T_COLLAPSE = 4.15
T_SETTLE = 4.78                     # collapse 4.15 -> 4.78 (0.63 s), settle 4.78 -> 4.95
T_EXIT = 4.95                       # canonical E02 from here: 3 static 60-fps frames

LANE_RANK = np.array([int(abs(i - 7.5) - 0.5) for i in range(N_LANES)])   # 0 = centre pair
T_A = T_BUILD + PAIR_STAGGER * LANE_RANK
T_B = B_IN0 + B_IN_STEP * np.arange(N_LANES)

# styles (RGBA rows)
A_FILL = rgba(CYAN, 0.95)
A_STROKE = rgba(INK, 0.5)
B_FILL = rgba(INK, 0.85)
B_STROKE = rgba(INK, 0.0)
LIME_FILL = rgba(LIME, 0.9)
INK_FILL = rgba(INK, 1.0)
GUIDE_OP = 0.35

_K = 0.5522847498307936             # cubic quarter-circle handle (same as manim's arcs)


def rrect_pts(cx: float, y0: float, w: float, h: float, r: float) -> np.ndarray:
    """(32, 3) points of a rounded rectangle with bottom edge at y0, same point order as
    manim's RoundedRectangle (4 cubic arcs + 4 lines). The corner shrinks for tiny h."""
    hw, hh = w / 2, h / 2
    r = max(0.0, min(r, hw, hh))
    k = _K * r
    cy = y0 + hh
    P = np.array([
        # arc top-right
        (hw, hh - r), (hw, hh - r + k), (hw - r + k, hh), (hw - r, hh),
        # top edge
        (hw - r, hh), (hw - r - (w - 2 * r) / 3, hh), (-hw + r + (w - 2 * r) / 3, hh), (-hw + r, hh),
        # arc top-left
        (-hw + r, hh), (-hw + r - k, hh), (-hw, hh - r + k), (-hw, hh - r),
        # left edge
        (-hw, hh - r), (-hw, hh - r - (h - 2 * r) / 3), (-hw, -hh + r + (h - 2 * r) / 3), (-hw, -hh + r),
        # arc bottom-left
        (-hw, -hh + r), (-hw, -hh + r - k), (-hw + r - k, -hh), (-hw + r, -hh),
        # bottom edge
        (-hw + r, -hh), (-hw + r + (w - 2 * r) / 3, -hh), (hw - r - (w - 2 * r) / 3, -hh), (hw - r, -hh),
        # arc bottom-right
        (hw - r, -hh), (hw - r + k, -hh), (hw, -hh + r - k), (hw, -hh + r),
        # right edge
        (hw, -hh + r), (hw, -hh + r + (h - 2 * r) / 3), (hw, hh - r - (h - 2 * r) / 3), (hw, hh - r),
    ])
    out = np.zeros((32, 3))
    out[:, 0] = P[:, 0] + cx
    out[:, 1] = P[:, 1] + cy
    return out


def line_pts(p0, p1) -> np.ndarray:
    """(4, 3) points of a straight cubic segment p0 -> p1 (what manim's Line holds)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return np.array([p0, p0 + (p1 - p0) / 3, p0 + 2 * (p1 - p0) / 3, p1])


def bump(p: float) -> float:
    """Punchy 0 -> 1 -> 0 envelope on p in [0, 1]: fast attack, smooth release."""
    if p <= 0 or p >= 1:
        return 0.0
    if p < 0.3:
        return ease_out_quad(p / 0.3)
    return 1.0 - ease_in_out_sine((p - 0.3) / 0.7)


def frames_on(now: float, f0: int, n: int) -> bool:
    """True on story frames f0 .. f0+n-1 (60-fps grid; previews see every other one)."""
    f = story_frame(now)
    return f0 <= f < f0 + n


def set_style(m: VMobject, fill=None, stroke=None, width=None) -> None:
    if fill is not None:
        m.fill_rgbas = np.asarray(fill, float).reshape(1, 4).copy()
    if stroke is not None:
        m.stroke_rgbas = np.asarray(stroke, float).reshape(1, 4).copy()
    if width is not None:
        m.stroke_width = float(width)


def with_alpha(row: np.ndarray, a: float) -> np.ndarray:
    r = row.copy()
    r[3] *= a
    return r


def hold_floor(scene, t: float) -> None:
    """hold until the last frame at or before t (60 fps: exactly t; 30-fps previews round
    down, so the static exit hold is never a single too-short wait)."""
    n = int(np.floor(t * config.frame_rate + 1e-6)) - scene._frames
    if n > 0:
        scene.wait((n - 0.5) / config.frame_rate)


class B02Simd(BeatScene):
    def beat(self):
        # ---------------- entry: E01 (plain token at AC) ----------------
        st = enter(self)
        tok = st.token
        frame = tok.base                       # the token base IS the core frame (exact at t=0)
        self._frame = frame
        self._f_tok = (frame.points.copy(), frame.fill_rgbas.copy(), frame.stroke_rgbas.copy(),
                       float(frame.stroke_width))
        rest = RoundedRectangle(width=FRAME_W, height=FRAME_H, corner_radius=FRAME_R,
                                stroke_color=CYAN, stroke_width=3, fill_color=CYAN,
                                fill_opacity=0.04).move_to(AC)
        assert len(rest.points) == len(frame.points)
        self._f_rest = (rest.points.copy(), rest.fill_rgbas.copy(), rest.stroke_rgbas.copy(), 3.0)

        # ---------------- world ----------------
        self.baseline = Line([BEAM_X0, BASE_Y, 0], [BEAM_X1, BASE_Y, 0], stroke_color=DIM, stroke_width=2)
        self.guides = VGroup(*[Line([x, BASE_Y, 0], [x, HANG_Y, 0], stroke_color=DIM, stroke_width=1.5,
                                    stroke_opacity=GUIDE_OP) for x in LANE_X])
        self.abars = VGroup(*[RoundedRectangle(width=BAR_W, height=0.2, corner_radius=BAR_R,
                                               stroke_color=INK, stroke_width=1, stroke_opacity=0.5,
                                               fill_color=CYAN, fill_opacity=0.95) for _ in LANE_X])
        self.bbars = VGroup(*[RoundedRectangle(width=BAR_W, height=0.2, corner_radius=BAR_R,
                                               stroke_color=INK, stroke_width=1, stroke_opacity=0.0,
                                               fill_color=INK, fill_opacity=0.85) for _ in LANE_X])
        self.beam = Line([BEAM_X0, BEAM_Y, 0], [BEAM_X1, BEAM_Y, 0], stroke_color=INK, stroke_width=5,
                         stroke_opacity=0)
        self.beam_seg = Line([LANE_X[0] - 0.20, BEAM_Y, 0], [LANE_X[0] + 0.20, BEAM_Y, 0],
                             stroke_color=INK, stroke_width=5, stroke_opacity=0)
        self.content = VGroup(self.guides, self.baseline, self.abars, self.bbars, self.beam_seg, self.beam)

        # ---------------- text ----------------
        track = TextTrack(self)
        track.on(kick("SCALE 1/6 · ONE CORE"), 0.10, 4.10)
        c1 = card([("1 instruction.", "H96", INK, {"1": CYAN}), ("16 numbers at once.", "S56", CYAN)])
        # out at 2.85 (on the 3rd lockstep pulse) so the stack is gone when 'That's SIMD.' lands
        track.on(c1[0], 0.30, T_CARD1_OUT)
        track.on(c1[1], T_SIMD, T_CARD1_OUT)
        c2 = card([("That's SIMD.", "H96", INK, {"SIMD.": CYAN})])
        track.on(c2[0], 3.05, 4.15)
        # objdump's Intel-syntax spelling (no spaces after commas): 5.91 wide at mono 34, so the
        # 1.08 POP stays inside the 6.8-wide core frame (the spaced form is 6.47 -> 6.99 popped)
        ins = mono(INSTR, 34, INK).move_to([X_C, INSTR_Y, 0])
        assert_safe(ins)
        assert ins.get_left()[0] > X_C - FRAME_W / 2 and ins.get_right()[0] < X_C + FRAME_W / 2
        track.on(ins, T_SIMD, T_TEXT_OUT, enter="pop", exit_dur=0.10)
        for i, x in enumerate(LANE_X):
            da = mono(str(A_DIG[i]), 34, CYAN).move_to([x, -1.88, 0])
            db = mono(str(B_DIG[i]), 34, INK).set_opacity(0.85).move_to([x, 2.32, 0])
            assert_safe(da, db)
            track.on(da, float(T_A[i]) + 0.03, T_TEXT_OUT, enter="pop", exit_dur=0.10)
            track.on(db, float(T_B[i]) + 0.03, T_TEXT_OUT, enter="pop", exit_dur=0.10)

        # ---------------- HUD (not part of E01: fades up 0.10 -> 0.40) ----------------
        hud = Hud("1", "CALCULATION AT ONCE", scene=self)
        hud.fade_to(0.10, 0.40, 0.0, 1.0, DOWN * 0.2, ORIGIN, ease=ease_out_expo)
        hud.pulse(T_SCALAR_LAND, 1.10)                # the scalar result lands (tick is on the issue)
        hud.slam(T_LAND, "16", mult="×16")
        hud.label(T_LAND + 0.12, CALC_LABEL)          # swapped while the ×16 fully covers it
        for tp in PULSES:
            hud.pulse(tp, 1.06, PULSE_DUR)

        # ---------------- z-order: dust, ring, frame(token), content, text, HUD ----------------
        self.add(self.content, track, hud)
        entry_ring(self)
        self.collapse = None
        self._draw(0.0)
        driver = Mobject()
        driver.add_updater(lambda m, dt: self._draw(self.now))
        self.add(driver)

        # ---------------- audio ----------------
        self.cue("hit", gain=0.8, note="match-cut")
        self.cue("whoosh", dt=0.22, dur=0.45, direction="up", note="the token opens")
        self.cue("ticks_burst", dt=0.40, dur=0.30, rate0=40, rate1=60, note="bars landing")
        self.cue("tick", dt=T_SCALAR, gain=0.7, note="single scalar add")
        self.cue("riser", dt=1.00, dur=0.50, note="into the SIMD landing")
        self.cue("tick", dt=T_SIMD, freq=6000, gain=0.8, note="instruction issued")
        self.cue("hit", dt=T_LAND, note="all 16 lanes land on one frame")
        self.cue("sub_drop", dt=T_LAND, dur=0.6)
        for tp in PULSES:
            self.cue("tick", dt=tp, freq=3800, pan=0, gain=0.5, note="lockstep pulse")
        self.cue("tick", dt=3.05, gain=0.4, note="'That's SIMD.' card")
        self.cue("whoosh", dt=4.47, dur=0.65, direction="down", note="collapse")
        self.cue("tick", dt=T_SETTLE, gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_COLLAPSE)
        # capture the (static) landed state and collapse it into the core token
        self.collapse = COLLAPSE(frame, self.content, "core")
        self.add(self.collapse.glyph)
        hold_floor(self, T_EXIT)
        exit_to(self)                               # canonical E02, static to 5.00

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.collapse is not None:
            if now < T_SETTLE - EPS:
                self.collapse.at(ease_in_out_expo(ramp(now, T_COLLAPSE, T_SETTLE)))
            else:
                self.collapse.settle(ramp(now, T_SETTLE, T_EXIT))
            return
        self._draw_frame(now)
        self._draw_lanes(now)
        self._draw_beam(now)

    def _draw_frame(self, now: float) -> None:
        e = ease_out_expo(ramp(now, 0.0, T_OPEN))
        p0, f0, s0, w0 = self._f_tok
        p1, f1, s1, w1 = self._f_rest
        m = self._frame
        m.points = lerp(p0, p1, e)
        m.fill_rgbas = lerp(f0, f1, e)
        m.stroke_rgbas = lerp(s0, s1, e)
        m.stroke_width = lerp(w0, w1, e)
        # light wave: the core outline flashes INK with the stacks, then cools back to CYAN
        if now >= T_LAND - EPS:
            k = 1.0 if frames_on(now, story_frame(T_LAND), 2) else 1 - ease_out_cubic(
                ramp(now, T_LAND + 2 / 60, T_LAND + 0.30))
            if k > 0:
                m.stroke_rgbas = lerp(s1, rgba(INK, 1.0).reshape(1, 4), 0.8 * k)
                m.stroke_width = lerp(w1, 5.0, k)

    def _draw_lanes(self, now: float) -> None:
        # base line grows from the centre outwards
        eb = ease_out_expo(ramp(now, T_BUILD, T_BUILD + 0.32))
        hx = lerp(0.0, (BEAM_X1 - BEAM_X0) / 2, eb)
        cxl = (BEAM_X0 + BEAM_X1) / 2
        self.baseline.points = line_pts([cxl - max(hx, 1e-4), BASE_Y, 0], [cxl + max(hx, 1e-4), BASE_Y, 0])
        self.baseline.set_stroke(opacity=1.0 if hx > 1e-3 else 0.0)

        # global lane state
        land_simd = ease_in_cubic(ramp(now, T_SIMD, T_LAND))
        flash_ink = frames_on(now, story_frame(T_LAND), 2)
        to_lime = ease_out_cubic(ramp(now, T_LAND + 2 / 60, T_LIME))
        sy, env = 1.0, 0.0
        for tp in PULSES:
            if tp - EPS <= now < tp + PULSE_DUR:
                env = bump((now - tp) / PULSE_DUR)
                sy = 1.0 + 0.06 * env
        # guides glow while the instruction travels down the lanes (1.30 -> 1.50) and on landing
        g_boost = 0.0
        if T_SIMD <= now < T_LAND + 0.25:
            g_boost = ease_out_cubic(ramp(now, T_SIMD, T_LAND)) * (1 - ease_out_cubic(ramp(now, T_LAND, T_LAND + 0.25)))

        g_scalar = 0.0
        if T_SCALAR <= now < T_SCALAR_LAND + 0.25:
            g_scalar = (ease_out_cubic(ramp(now, T_SCALAR, T_SCALAR_LAND))
                        * (1 - ease_out_cubic(ramp(now, T_SCALAR_LAND, T_SCALAR_LAND + 0.25))))

        for i in range(N_LANES):
            # ---- guide
            g = self.guides[i]
            gb = max(g_boost, g_scalar) if i == 0 else g_boost
            eg = ease_out_expo(ramp(now, T_A[i], T_A[i] + GUIDE_GROW))
            if eg <= 1e-4:
                g.set_stroke(opacity=0)
                g.points = line_pts([LANE_X[i], BASE_Y, 0], [LANE_X[i], BASE_Y + 1e-4, 0])
            else:
                g.points = line_pts([LANE_X[i], BASE_Y, 0], [LANE_X[i], lerp(BASE_Y, HANG_Y, eg), 0])
                g.set_stroke(DIM if gb <= 0 else interpolate_color(DIM, INK, 0.6 * gb),
                             opacity=GUIDE_OP + 0.45 * gb)

            # ---- A bar (grows from 0, ease_out_back)
            pa = ramp(now, T_A[i], T_A[i] + A_GROW)
            ga = ease_out_back(pa) if pa < 1 else 1.0
            ha = A_H[i] * ga * sy
            a = self.abars[i]
            # ---- B bar position: hanging (+ drop-in) <-> landed
            pb = ramp(now, T_B[i], T_B[i] + B_IN_DUR)
            b_alpha = ease_out_cubic(pb)
            drop_in = 0.6 * (1 - (ease_out_back(pb) if pb < 1 else 1.0))
            d = land_simd
            if i == 0:
                if now < T_LIFT0:
                    d = max(d, ease_in_cubic(ramp(now, T_SCALAR, T_SCALAR_LAND)))
                elif now < T_LIFT1:
                    d = max(d, 1 - ease_in_out_cubic(ramp(now, T_LIFT0, T_LIFT1)))
            hb = B_H[i] * (sy if d >= 1 else 1.0)
            hang_bottom = HANG_Y - B_H[i] + drop_in
            land_bottom = BASE_Y + ha
            yb = lerp(hang_bottom, land_bottom, d)
            b = self.bbars[i]

            # ---- colours
            fa, fb = A_FILL, B_FILL
            mix = 0.0
            if i == 0 and T_SCALAR_LAND - EPS <= now < T_LIFT1:
                mix = 1.0 if now < T_LIFT0 else 1 - ease_in_out_cubic(ramp(now, T_LIFT0, T_LIFT1))
                lime0 = rgba(LIME, 1.0) if frames_on(now, story_frame(T_SCALAR_LAND), 2) else LIME_FILL
                fa, fb = lerp(A_FILL, lime0, mix), lerp(B_FILL, lime0, mix)
            if now >= T_LAND - EPS:
                if flash_ink:
                    fa = fb = INK_FILL
                else:
                    fa = fb = lerp(INK_FILL, LIME_FILL, to_lime)
            sa = lerp(A_STROKE, rgba(INK, 1.0), env)
            sb = lerp(B_STROKE, rgba(INK, 1.0), env) if now >= T_LAND else B_STROKE
            sw = 1.0 + 1.5 * env

            if ha < 1e-3:
                a.points = rrect_pts(LANE_X[i], BASE_Y, BAR_W, 1e-3, BAR_R)
                set_style(a, with_alpha(fa, 0), with_alpha(sa, 0), sw)
            else:
                a.points = rrect_pts(LANE_X[i], BASE_Y, BAR_W, ha, BAR_R)
                set_style(a, fa, sa, sw)
            b.points = rrect_pts(LANE_X[i], yb, BAR_W, hb, BAR_R)
            set_style(b, with_alpha(fb, b_alpha), with_alpha(sb, b_alpha), sw)

    def _draw_beam(self, now: float) -> None:
        # scalar: a 0.40-wide segment over lane 0, 2 frames on, quick decay
        f_s = story_frame(T_SCALAR)
        if frames_on(now, f_s, 2):
            seg = 1.0
        else:
            seg = 1.0 - ease_out_cubic(ramp(now, (f_s + 2) / 60, (f_s + 2) / 60 + 0.08)) if now >= T_SCALAR else 0.0
        self.beam_seg.set_stroke(opacity=seg)
        # SIMD: full beam 2 frames at 1.0, then fades over 0.2 s; lockstep pulses flash it to 0.6
        f_b = story_frame(T_SIMD)
        op = 0.0
        if frames_on(now, f_b, 2):
            op = 1.0
        elif now >= T_SIMD:
            t2 = (f_b + 2) / 60
            op = 1.0 - ease_out_cubic(ramp(now, t2, t2 + 0.20))
        for tp in PULSES:
            if tp - EPS <= now < tp + PULSE_DUR:
                op = max(op, 0.6 * (1 - ease_out_cubic(ramp(now, tp + 1 / 60, tp + PULSE_DUR))))
        self.beam.set_stroke(opacity=op)

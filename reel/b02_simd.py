"""b02 - SCALE 1/6 - ONE CORE: one instruction, 16 numbers (SIMD).

The incoming plain token opens into a CPU core. Sixteen lanes hold the digits of pi
(A, cyan bars) and e (B, ink bars hanging above), with a '+' between the two operand
rows. '1 instruction.' is read through the build; then SCALAR (0.87): the other lanes dim and
three lanes are added one at a time (each lane outlined, its digits popped, one tick each, a
lime result; HUD '1' pulses). '16 numbers at once.' lands on the three results (1.31) and is
read before the deliberate RESET (1.79 -> 1.97, one eased move on the riser): the three
results lift back to their hang together, drain back to INK/CYAN and dim with the other 13
lanes, so the register is back at its start line exactly when ONE instruction
(vaddps zmm0,zmm1,zmm2) issues at 1.97, relights all 16 lanes (the reset's apex flows into the
drop) and drops every B bar onto its A bar on the same frame, 2.17: the light wave (16 lane
outlines + stacks flash INK, then LIME, '+' becomes '='), HUD 1 -> 16 + LIME x16 (1.7 s).
The payoff HOLDS 1.02 s (the stacks breathe, the stage drifts). LOCKSTEP: five more literal
issues of the same instruction on the next 16 digits of pi and e, 0.45 s apart (digits
hard-swap, bars re-stage and re-drop, all 16 land on one frame); 'That's SIMD.' pops on the
2nd landing (3.82) and holds; then the core collapses (cut_times) into handoff_token('core')
(E02). The kicker and 'That's SIMD.' stay fully visible through the collapse and settle and
are hard-cut by the canonical exit at 5.95 (like b01), so no frame before the cut is
text-less except the contract's static E02 hold.

Everything that moves is a pure function of self.now (one driver updater + the stage drift);
the scene only holds until the structural switch points (collapse capture, canonical exit).
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# World layout (storyboard b02 visuals; top band loosened per the review: the
# instruction, beam and e digits each moved down so nothing touches the frame edge)
# --------------------------------------------------------------------------
N_LANES = 16
LANE_X = np.array([X_C + (i - 7.5) * 0.38 for i in range(N_LANES)])
# 16 digits per register load: chunk 0 = the scalar/SIMD demo (the storyboard's a_i, b_i),
# chunks 1..5 = the lockstep issues (same instruction, next data)
PI_DIGITS = "3141592653589793" "2384626433832795" "0288419716939937" "5105820974944592" "3078164062862089" "9862803482534211"
E_DIGITS = "2718281828459045" "2353602874713526" "6249775724709369" "9959574966967627" "7240766303535475" "9457138217852516"
A_CH = np.array([[int(c) for c in PI_DIGITS[16 * k:16 * k + 16]] for k in range(len(PI_DIGITS) // 16)])
B_CH = np.array([[int(c) for c in E_DIGITS[16 * k:16 * k + 16]] for k in range(len(E_DIGITS) // 16)])
BASE_Y = -1.55
HANG_Y = 1.95                       # top edge of the hanging B bars / top of the guides
BAR_W = 0.30
BAR_R = 0.05
UNIT_H = 0.16
A_H = A_CH * UNIT_H                                        # (chunks, 16)
B_H = np.maximum(B_CH * UNIT_H, 0.03)                      # a 0 is a 0.03 sliver
A_DIG_Y = -1.88
B_DIG_Y = 2.17
BEAM_Y = 2.50
BEAM_X0, BEAM_X1 = -3.45, 2.75
FRAME_W, FRAME_H, FRAME_R = 6.8, 5.5, 0.35
INSTR = "vaddps zmm0,zmm1,zmm2"
INSTR_Y = 2.82
# lane outline: the whole lane (e digit, bars, pi digit), 0.04 gaps between neighbours
BOX_W, BOX_Y0, BOX_Y1, BOX_R = 0.34, -2.15, 2.43, 0.08
# operator sign in the left margin, between the two operand groups: '+' while the B bars
# are apart from the A bars, '=' (LIME) once they have landed as results
SIGN_X, SIGN_Y = -3.56, 0.22
SIGN_POP = 1.12

# --------------------------------------------------------------------------
# Timeline (beat-local seconds). The 6.0 s slot is filled by SEQUENCING, not by slowing: every
# motion keeps its snappy duration (scalar step 0.14, drop 0.10, reset 0.18, issue -> land
# 0.20, re-drop 0.18, pulse pitch 0.45); the extra second sits BETWEEN the ideas:
#   0.00-0.81  open + build; '1 instruction.' pops on story frame 1 and is read through the build
#   0.81-0.87  calm                      -> the headline has been up 0.66 s when the scalar demo starts
#   0.87-1.41  SCALAR: three adds, one tick each, HUD '1' pulses
#   1.31-1.56  '16 numbers at once.' lands on the scalar results (read 0.61 s before the landing)
#   1.79-1.97  RESET -> 1.97 issue -> 2.17 all 16 land: light wave, HUD 16 + LIME x16 (1.7 s)
#   2.17-3.19  PAYOFF HOLD (1.02 s): the lime stacks breathe, the stage drifts
#   3.19-5.17  LOCKSTEP: five issues 0.45 s apart; 'That's SIMD.' pops on the 2nd landing (3.82)
#   5.43-5.95  collapse + settle, 5.95-6.00 static E02
# --------------------------------------------------------------------------
T_OPEN = 0.45                       # token -> core frame
T_BUILD = 0.15                      # base line / guides / A bars start
A_GROW = 0.20                       # A bar grow (ease_out_back)
GUIDE_GROW = 0.26
PAIR_STAGGER = 0.024                # = 0.012 s per lane, centre outwards (pairs land together)
B_IN0, B_IN_STEP, B_IN_DUR = 0.40, 0.012, 0.22
T_SIGN = 0.55                       # '+' pops in
T_DIM0, T_DIM1 = 0.87, 0.97         # scalar spotlight: lanes not being worked on dim to DIM_K
DIM_K = 0.30
SCALAR_T = (0.88, 1.02, 1.16)       # one scalar add per tick on lanes 0, 1, 2
SCALAR_STEP = 0.14
SCALAR_DROP = 0.10
N_SCALAR = len(SCALAR_T)
T_SUB = SCALAR_T[-1] + SCALAR_DROP + 0.05   # 1.31: '16 numbers at once.' fades in on the 3 results
T_LAND = 130 / 60                   # 2.1667: all 16 lanes land on this frame
T_SIMD = T_LAND - 0.20              # 1.9667: instruction issued: every lane relights
# RESET: the scalar results lift back to their hang (and dim with the rest) in ONE eased move
# that ends on the issue frame, so the register is at its start line when the instruction fires
T_REW0, T_REW1 = T_SIMD - 0.18, T_SIMD
T_REW_DIM = T_SIMD - 0.10           # the lifted lanes dim back over the reset's last 0.10 s
T_LIME = T_LAND + 0.15              # the 16 stacks are LIME
PAY_HOLD = 1.02                     # the payoff (16 lime stacks, HUD 16 + x16) holds before the lockstep
PULSE_STEP = 0.45
PULSES = tuple(T_LAND + PAY_HOLD + PULSE_STEP * k for k in range(5))  # 3.19 ... 4.99: same instruction, next data (chunks 1..5)
PULSE_DROP = 0.18                   # re-drop: every lane lands at tp + 0.18 (one frame)
PULSE_LIFT = 0.30
T_CARD2 = PULSES[1] + PULSE_DROP                # 3.817: 'That's SIMD.' pops on the 2nd lockstep landing
T_CARD1_OUT = T_CARD2 - 0.20                    # 3.617: card 1 is gone exactly when it pops
MULT_HOLD = 1.70                    # the LIME x16 pill: settled (after its 0.2 s slam) for 1.5 s
T_COL, T_SET, T_EXIT = cut_times(6.0)      # 5.433 / 5.833 / 5.95
T_TEXT_OUT = T_EXIT                        # kicker + headline hold to the exit, then hard-cut
# ambient life under the holds: the stage (frame + lanes) drifts in 1.0 -> 1.015 (frame top 3.24,
# inside STAGE) and the landed lime stacks breathe in lockstep (alpha x 1 -> 0.85 -> 1, 1.2 s)
DRIFT_S1 = 1.015
BREATH_P, BREATH_A = 1.20, 0.15

assert SCALAR_T[-1] + SCALAR_DROP + 2 / 60 + 0.06 < T_REW0     # the 3rd result turns LIME first
assert T_SUB + 0.25 + 0.15 <= T_REW0                             # the sub-line is read before the reset
assert PULSES[-1] + PULSE_DROP + 0.25 < T_COL                   # the last landing settles before the collapse
LANE_RANK = np.array([int(abs(i - 7.5) - 0.5) for i in range(N_LANES)])   # 0 = centre pair
T_A = T_BUILD + PAIR_STAGGER * LANE_RANK
T_B = B_IN0 + B_IN_STEP * np.arange(N_LANES)
assert len(A_CH) == len(PULSES) + 1 == len(B_CH)
assert float(np.max(A_H + B_H)) + PULSE_LIFT < HANG_Y - BASE_Y      # re-staged bars stay under the hang line

# styles (RGBA rows)
A_FILL = rgba(CYAN, 0.95)
A_STROKE = rgba(INK, 0.5)
B_FILL = rgba(INK, 0.85)
B_STROKE = rgba(INK, 0.0)
LIME_FILL = rgba(LIME, 0.9)
INK_FILL = rgba(INK, 1.0)
GUIDE_OP = 0.35


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


def chunk_at(now: float) -> int:
    """Index of the register load on screen (0 until the first lockstep issue)."""
    k = 0
    for j, tp in enumerate(PULSES):
        if now >= tp - EPS:
            k = j + 1
    return k


def lane_dim(i: int, now: float) -> float:
    """Opacity factor of lane i: the scalar spotlight dims every lane not (yet) worked on; the
    scalar lanes dim back with the reset, so all 16 relight together on the issue."""
    if now < T_DIM0 - EPS or now >= T_SIMD - EPS:
        return 1.0
    if i < N_SCALAR and now >= SCALAR_T[i] - EPS:       # (bright while they lift, dim at the top)
        return lerp(1.0, DIM_K, ease_in_out_cubic(ramp(now, T_REW_DIM, T_REW1)))
    return lerp(1.0, DIM_K, ease_out_cubic(ramp(now, T_DIM0, T_DIM1)))


def rewind_at(now: float) -> float:
    """0..1 progress of the scalar reset (ease_in_out_cubic, T_REW0 -> T_REW1 = T_SIMD)."""
    return ease_in_out_cubic(ramp(now, T_REW0, T_REW1))


def spot_at(i: int, now: float) -> float:
    """0..1: lane i is the scalar lane being worked on (outline / INK guide column)."""
    if i >= N_SCALAR:
        return 0.0
    t0 = SCALAR_T[i]
    t1 = t0 + SCALAR_STEP
    if now < t0 - EPS or now >= t1 + 0.06 - EPS:
        return 0.0
    return 1.0 if now < t1 - EPS else 1.0 - ease_out_cubic(ramp(now, t1, t1 + 0.06))


def breath(now: float) -> float:
    """Alpha factor of the landed lime stacks from T_LIME on: one slow, shared (lockstep) breath,
    1 -> 1 - BREATH_A -> 1 every BREATH_P seconds, so the payoff hold is never a dead frame."""
    if now < T_LIME:
        return 1.0
    return 1.0 - BREATH_A * (0.5 - 0.5 * math.cos(2 * math.pi * (now - T_LIME) / BREATH_P))


def landed_color(now: float, t_land: float) -> np.ndarray:
    """INK on the landing frame + the next (2 story frames), then INK -> LIME by +0.15 s; the
    16-lane results (not the scalar ones) breathe."""
    if flash_frames(now, t_land, 2):
        return INK_FILL
    lime = LIME_FILL if t_land < T_LAND - EPS else with_alpha(LIME_FILL, breath(now))
    return lerp(INK_FILL, lime, ease_out_cubic(ramp(now, t_land + 2 / 60, t_land + 0.15)))


def flash_decay(now: float, t: float, dur: float) -> float:
    """1 on the 2 story frames from t, then 1 -> 0 (ease_out_cubic) over dur; 0 before t."""
    if now < t - EPS:
        return 0.0
    if flash_frames(now, t, 2):
        return 1.0
    return 1.0 - ease_out_cubic(ramp(now, t + 2 / 60, t + 2 / 60 + dur))


def landings(now: float) -> list[float]:
    """Every all-lane landing time <= now (the SIMD landing + the lockstep re-drops)."""
    return [t for t in (T_LAND, *[tp + PULSE_DROP for tp in PULSES]) if t <= now + EPS]


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
        self.boxes = VGroup(*[RoundedRectangle(width=BOX_W, height=BOX_Y1 - BOX_Y0, corner_radius=BOX_R,
                                               stroke_color=INK, stroke_width=3, stroke_opacity=0.0,
                                               fill_opacity=0).move_to([x, (BOX_Y0 + BOX_Y1) / 2, 0])
                              for x in LANE_X])
        self.beam = Line([BEAM_X0, BEAM_Y, 0], [BEAM_X1, BEAM_Y, 0], stroke_color=INK, stroke_width=5,
                         stroke_opacity=0)
        self.beam_seg = Line([-0.20, BEAM_Y, 0], [0.20, BEAM_Y, 0], stroke_color=INK, stroke_width=5,
                             stroke_opacity=0)
        self._seg0 = self.beam_seg.points.copy()

        # ---------------- in-world text (drawn per frame through TextSlots, so the digits can
        # dim, pop and hard-swap, and everything collapses into the token with the lanes) ----
        self._dig_a = [mono(str(d), 34, CYAN).move_to(ORIGIN) for d in range(10)]
        self._dig_b = [mono(str(d), 34, INK).set_opacity(0.85).move_to(ORIGIN) for d in range(10)]
        for x in (LANE_X[0], LANE_X[-1]):      # extremes, at the largest (1.3x scalar) pop
            for y in (A_DIG_Y, B_DIG_Y):
                probe = mono("8", 34, INK).move_to([x, y, 0]).scale(1.3)
                assert_safe(probe)
                assert BOX_Y0 < probe.get_bottom()[1] and probe.get_top()[1] < BOX_Y1
        self._ins = mono(INSTR, 34, INK).move_to([X_C, INSTR_Y, 0])
        # objdump's Intel spelling (no spaces after commas): 5.90 wide at mono 34, so the
        # 1.08 POP stays inside the 6.8-wide core frame
        assert_safe(self._ins)
        assert self._ins.width * 1.08 < FRAME_W - 0.3
        assert self._ins.get_bottom()[1] > BEAM_Y + 0.05 and self._ins.get_top()[1] * 1.0 < AC[1] + FRAME_H / 2 - 0.1
        self._plus = mono("+", 42, INK).move_to([SIGN_X, SIGN_Y, 0])
        self._eq = mono("=", 42, LIME).move_to([SIGN_X, SIGN_Y, 0])
        for m in (self._plus, self._eq):
            assert_safe(m)
            big = m.copy().scale(SIGN_POP)
            assert big.get_left()[0] > AC[0] - FRAME_W / 2 + 0.03 and big.get_right()[0] < LANE_X[0] - BOX_W / 2
        self.dig_a = VGroup(*[TextSlot() for _ in range(N_LANES)])
        self.dig_b = VGroup(*[TextSlot() for _ in range(N_LANES)])
        self.ins_slot, self.sign_slot = TextSlot(), TextSlot()

        self.content = VGroup(self.guides, self.baseline, self.abars, self.bbars, self.boxes,
                              self.beam_seg, self.beam, self.dig_a, self.dig_b, self.ins_slot,
                              self.sign_slot)

        # ---------------- text ----------------
        track = TextTrack(self)
        # kicker + headline arrive on the cut (frame 0 stays the exact E01 token: 'in' at 0.0
        # is invisible on frame 0, the POP starts on story frame 1); both stay FULLY visible
        # through the collapse and settle and hard-cut at T_EXIT (the canonical E02 swap), like
        # b01, so the screen never sits text-less before the cut
        track.on(kick("SCALE 1/6 · ONE CORE"), 0.0, T_TEXT_OUT, exit="cut")
        c1 = card([("1 instruction.", "H96", INK, {"1": CYAN}), ("16 numbers at once.", "S56", CYAN)])
        track.on(c1[0], 1 / 60, T_CARD1_OUT, enter="pop")
        # the sub-line lands on the three scalar results, BEFORE the issue it describes
        track.on(c1[1], T_SUB, T_CARD1_OUT)
        c2 = card([("That's SIMD.", "H96", INK, {"SIMD.": CYAN})])
        track.on(c2[0], T_CARD2, T_TEXT_OUT, enter="pop", exit="cut")   # on the 2nd lockstep landing

        # ---------------- HUD (not part of E01: fades up 0.10 -> 0.40) ----------------
        hud = Hud("1", "CALCULATION AT ONCE", scene=self)
        hud.fade_to(0.10, 0.40, 0.0, 1.0, DOWN * 0.2, ORIGIN, ease=ease_out_expo)
        for ts in SCALAR_T:
            hud.pulse(ts + SCALAR_DROP, 1.25)          # each scalar result: still '1' at once
        hud.slam(T_LAND, "16")
        hud.mult(T_LAND, "×16", hold=MULT_HOLD)        # the LIME pill holds through the payoff
        hud.label(T_LAND, CALC_LABEL)                  # plural on the hit frame itself
        for tp in PULSES:
            hud.pulse(tp + PULSE_DROP, 1.06, 0.18)

        # ---------------- z-order: dust, ring, frame(token), content, text, HUD ----------------
        self.add(self.content, track, hud)
        entry_ring(self)
        self.collapse = None
        self._draw(0.0)
        drive(self, self._draw)
        # camera life: the stage (core frame + everything in it; never text or HUD) drifts in
        # from the end of the open to the collapse (ease_in_out_sine: it leaves the open and
        # meets the collapse at rest); added after the driver so it composes on this frame's
        # drawing, released right before COLLAPSE captures the drifted pose
        self.drift = drift(self, VGroup(frame, self.content), T_OPEN, T_COL, s1=DRIFT_S1, dy=0.0,
                           ease=ease_in_out_sine)

        # ---------------- audio ----------------
        self.cue("hit", gain=0.8, note="match-cut")
        self.cue("whoosh", dt=0.22, dur=0.45, direction="up", note="the token opens")
        self.cue("ticks_burst", dt=0.40, dur=0.30, rate0=40, rate1=60, note="bars landing")
        for k, ts in enumerate(SCALAR_T):
            self.cue("tick", dt=ts, gain=0.7, freq=(3200, 3600, 4000)[k], pan=round(-0.55 + 0.1 * k, 2),
                     note=f"scalar add {k + 1} (lane {k})")
        self.cue("riser", dt=round(T_LAND - 0.50, 4), dur=0.50,
                 note=f"into the SIMD landing (scores the scalar reset {T_REW0:.2f}-{T_REW1:.2f})")
        self.cue("tick", dt=T_SIMD, freq=6000, gain=0.8, note="instruction issued")
        self.cue("hit", dt=T_LAND, note="all 16 lanes land on one frame")
        self.cue("sub_drop", dt=T_LAND, dur=0.6)
        for tp in PULSES:
            self.cue("tick", dt=tp, freq=6000, pan=0, gain=0.35, note="same instruction re-issued")
            land = round(tp + PULSE_DROP, 4)
            self.cue("tick", dt=land, freq=3800, pan=0, gain=0.5,
                     note="lockstep landing (all 16 lanes)" + (" + 'That's SIMD.' pops" if land == round(T_CARD2, 4) else ""))
        self.cue("whoosh", dt=round((T_COL + T_SET) / 2, 4), dur=0.45, direction="down", note="collapse")
        self.cue("tick", dt=round(T_SET, 4), gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_COL)
        # capture the (static, lime, drifted) state and collapse it -- digits and instruction
        # included -- into the core token
        self.drift.release()
        self.collapse = COLLAPSE(frame, self.content, "core")
        self.add(self.collapse.glyph)
        hold_floor(self, T_EXIT)
        exit_to(self)                               # canonical E02, static to 6.00

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.collapse is not None:
            self.collapse.render(now, T_COL, T_SET, T_EXIT)
            return
        self._draw_frame(now)
        self._draw_lanes(now)
        self._draw_beam(now)
        self._draw_text(now)

    def _draw_frame(self, now: float) -> None:
        e = ease_out_expo(ramp(now, 0.0, T_OPEN))
        p0, f0, s0, w0 = self._f_tok
        p1, f1, s1, w1 = self._f_rest
        m = self._frame
        m.points = lerp(p0, p1, e)
        m.fill_rgbas = lerp(f0, f1, e)
        m.stroke_rgbas = lerp(s0, s1, e)
        m.stroke_width = lerp(w0, w1, e)
        # the cut charge: the opening outline flashes INK on story frames 1-2 (never frame 0)
        if flash_frames(now, 1 / 60, 2):
            ink_stroke(m, 1.0)
        # light wave: the core outline flashes INK with the stacks, then cools back to CYAN
        # (full on the SIMD landing, softer on each lockstep landing)
        k = 0.0
        for t in landings(now):
            k = max(k, (1.0 if t == T_LAND else 0.45) * flash_decay(now, t, 0.28))
        if k > 0:
            m.stroke_rgbas = lerp(s1, rgba(INK, 1.0).reshape(1, 4), 0.8 * k)
            m.stroke_width = lerp(w1, 5.0, k)

    def _draw_lanes(self, now: float) -> None:
        # base line grows from the centre outwards
        eb = ease_out_expo(ramp(now, T_BUILD, T_BUILD + 0.32))
        hx = lerp(0.0, (BEAM_X1 - BEAM_X0) / 2, eb)
        cxl = (BEAM_X0 + BEAM_X1) / 2
        self.baseline.points = seg([cxl - max(hx, 1e-4), BASE_Y, 0], [cxl + max(hx, 1e-4), BASE_Y, 0])
        self.baseline.set_stroke(opacity=1.0 if hx > 1e-3 else 0.0)

        k = chunk_at(now)
        # guides glow while an instruction travels down the lanes (the SIMD issue, each re-issue)
        g_boost = 0.0
        if T_SIMD <= now < T_LAND + 0.25:
            g_boost = ease_out_cubic(ramp(now, T_SIMD, T_LAND)) * (1 - ease_out_cubic(ramp(now, T_LAND, T_LAND + 0.25)))
        for tp in PULSES:
            if tp - EPS <= now < tp + PULSE_DROP + 0.20:
                g_boost = max(g_boost, 0.7 * ease_out_cubic(ramp(now, tp, tp + PULSE_DROP))
                              * (1 - ease_out_cubic(ramp(now, tp + PULSE_DROP, tp + PULSE_DROP + 0.20))))
        # lane outlines: all 16 on the SIMD landing frame (full), each lockstep landing (softer)
        box_all = 0.0
        for t in landings(now):
            box_all = max(box_all, (1.0 if t == T_LAND else 0.5) * flash_decay(now, t, 0.30 if t == T_LAND else 0.20))

        for i in range(N_LANES):
            dim = lane_dim(i, now)
            spot = spot_at(i, now)
            # ---- guide (+ the INK column of the scalar lane being worked on)
            g = self.guides[i]
            eg = ease_out_expo(ramp(now, T_A[i], T_A[i] + GUIDE_GROW))
            if eg <= 1e-4:
                g.set_stroke(opacity=0)
                g.points = seg([LANE_X[i], BASE_Y, 0], [LANE_X[i], BASE_Y + 1e-4, 0])
            else:
                g.points = seg([LANE_X[i], BASE_Y, 0], [LANE_X[i], lerp(BASE_Y, HANG_Y, eg), 0])
                gb = max(g_boost, spot)
                g.set_stroke(DIM if gb <= 0 else interpolate_color(DIM, INK, max(0.6 * g_boost, spot)),
                             width=lerp(1.5, 4.0, spot),
                             opacity=(GUIDE_OP + 0.45 * g_boost + (0.9 - GUIDE_OP) * spot) * dim)
            # ---- lane outline
            bop = max(spot * 0.9, box_all)
            self.boxes[i].set_stroke(INK, width=3, opacity=bop)

            # ---- bars
            fa, fb = A_FILL, B_FILL
            if k == 0:
                pa = ramp(now, T_A[i], T_A[i] + A_GROW)
                ha = A_H[0, i] * (ease_out_back(pa) if pa < 1 else 1.0)
                hb = B_H[0, i]
                pb = ramp(now, T_B[i], T_B[i] + B_IN_DUR)
                b_alpha = ease_out_cubic(pb)
                drop_in = 0.6 * (1 - (ease_out_back(pb) if pb < 1 else 1.0))
                d = ease_in_cubic(ramp(now, T_SIMD, T_LAND))
                t_land = T_LAND
                if i < N_SCALAR and now < T_SIMD:
                    ts = SCALAR_T[i]
                    t_land = ts + SCALAR_DROP
                    if now < T_REW0:
                        d = ease_in_cubic(ramp(now, ts, t_land))
                    else:                               # reset: the result lifts back to its hang
                        d = 1.0 - rewind_at(now)
                    if now >= t_land - EPS:
                        res = landed_color(now, t_land)
                        w = 1.0 - rewind_at(now)
                        fa, fb = lerp(A_FILL, res, w), lerp(B_FILL, res, w)
                elif now >= T_LAND - EPS:
                    fa = fb = landed_color(now, T_LAND)
                yb = lerp(HANG_Y - hb + drop_in, BASE_Y + ha, d)
            else:
                tp = PULSES[k - 1]
                ha, hb, b_alpha = A_H[k, i], B_H[k, i], 1.0
                e = ease_in_cubic(ramp(now, tp, tp + PULSE_DROP))
                yb = BASE_Y + ha + PULSE_LIFT * (1.0 - e)
                if now >= tp + PULSE_DROP - EPS:
                    fa = fb = landed_color(now, tp + PULSE_DROP)
            sa = lerp(A_STROKE, rgba(INK, 1.0), spot)
            sw = 1.0 + 1.0 * spot

            a, b = self.abars[i], self.bbars[i]
            if ha < 1e-3:
                a.points = rrect_pts(LANE_X[i], BASE_Y, BAR_W, 1e-3, BAR_R)
                set_style(a, with_alpha(fa, 0), with_alpha(sa, 0), sw)
            else:
                a.points = rrect_pts(LANE_X[i], BASE_Y, BAR_W, ha, BAR_R)
                set_style(a, with_alpha(fa, dim), with_alpha(sa, dim), sw)
            b.points = rrect_pts(LANE_X[i], yb, BAR_W, hb, BAR_R)
            set_style(b, with_alpha(fb, b_alpha * dim), with_alpha(B_STROKE, 0), 1.0)

    def _draw_beam(self, now: float) -> None:
        # scalar: a 0.40-wide segment over the lane being worked on, 2 frames on, quick decay
        seg_op, seg_x = 0.0, LANE_X[0]
        for i, ts in enumerate(SCALAR_T):
            if now >= ts - EPS:
                seg_x = LANE_X[i]
                seg_op = 1.0 if flash_frames(now, ts, 2) else 1.0 - ease_out_cubic(
                    ramp(now, ts + 2 / 60, ts + 2 / 60 + 0.08))
        self.beam_seg.points = self._seg0 + np.array([seg_x, 0.0, 0.0])
        self.beam_seg.set_stroke(opacity=seg_op)
        # SIMD issue + every lockstep re-issue: full beam 2 frames at 1.0, then fades over 0.2 s
        op = 0.0
        for t in (T_SIMD, *PULSES):
            op = max(op, flash_decay(now, t, 0.20))
        self.beam.set_stroke(opacity=op)

    def _draw_text(self, now: float) -> None:
        k = chunk_at(now)
        swap = pop_scale(now - PULSES[k - 1], 1.12, 0.20) if k > 0 else 1.0
        for i in range(N_LANES):
            dim = lane_dim(i, now)
            pop_s = 1.0
            if i < N_SCALAR:
                pop_s = pop_scale(now - SCALAR_T[i], 1.3, 0.20)
            for row, slot, tmpls, t_in, y, dig in ((0, self.dig_a, self._dig_a, T_A[i] + 0.03, A_DIG_Y, A_CH[k, i]),
                                                   (1, self.dig_b, self._dig_b, T_B[i] + 0.03, B_DIG_Y, B_CH[k, i])):
                if now < t_in - EPS:
                    slot[i].clear_slot()
                    continue
                s = pop_scale(now - t_in, 1.08, 0.20) * pop_s * swap
                slot[i].display(tmpls[dig], dim, s, None, np.array([LANE_X[i], y, 0.0]))
        # the instruction: POP on the issue, a small re-pop on every re-issue (same instruction)
        if now < T_SIMD - EPS:
            self.ins_slot.clear_slot()
        else:
            s = pop_scale(now - T_SIMD, 1.08, 0.20)
            for tp in PULSES:
                s *= pop_scale(now - tp, 1.05, 0.18)
            self.ins_slot.display(self._ins, 1.0, s)
        # operator: '+' while the operands are apart, '=' once the results have landed
        if now < T_SIGN - EPS:
            self.sign_slot.clear_slot()
            return
        events = [(T_SIGN, "+")] + [(T_LAND, "=")]
        for tp in PULSES:
            events += [(tp, "+"), (tp + PULSE_DROP, "=")]
        t_ev, sym = max((ev for ev in events if ev[0] <= now + EPS), key=lambda ev: ev[0])
        self.sign_slot.display(self._plus if sym == "+" else self._eq, 1.0, pop_scale(now - t_ev, SIGN_POP, 0.20))

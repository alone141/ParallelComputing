"""b05 - SCALE 4/6 - ONE SERVER: 8 GPUs in one box, wired by NVLink through the NVSwitch.

The GPU token (E04) is one of the 8 GPUs of an HGX H100 node: the camera pulls back
(EXPAND, 0.50 s, INK charge on the hero outline) while '8 GPUs. / Wired together by NVLink.'
is read. Three all-reduce cycles (0.80, 1.85, 3.10): every GPU computes its own slice (its
texture rewrites itself to a different partial result; the first compute is the 135,168 slam,
and its partial results sit for 0.10 s before they are sent), then every GPU sends a packet
GPU-to-GPU THROUGH the NVSwitch to its partner on the other side
(~2.75 units: along its NVLink, across the switch, along the partner's link; full duplex,
left->right just above the link, right->left just below, so each pair passes inside the
switch). Packets carry long, bright comet trails with a hot INK core; they are MAGENTA (data
in flight) until they enter the switch, flash white-hot in it as the switch body lights up
(its label turns INK so it stays readable) and leave it LIME (the reduced result); on
arrival every GPU flashes LIME and now shows the SAME texture (every GPU holds the full
answer). Cycle 2 runs under a slow 1.0 -> 1.04 stage push-in. The first card hard-swaps to
'Share results.' (POP) on the second arrival (2.65), read 0.67 s before the third exchange
leaves (3.32). The third arrival
(3.90) is the light wave: GPUs, links and switch flash INK, then the GPUs go LIME (results)
while the switch and links glow MAGENTA (data movement), and a MAGENTA shock ring leaves the
switch; the GPUs keep a breathing LIME glow (every GPU holds the shared result; a faint sheen
sweeps all 8 textures in sync) for a 1.03 s hold. The chassis then collapses (cut_times) into
handoff_token('node') (E05); kicker + headline hold to the exit and hard-cut with the
canonical swap.

Everything that moves is a pure function of self.now (one driver updater + the drift);
the scene only holds until the structural switch points (collapse capture, canonical exit).
"""

from style import *
from motif import *

# --------------------------------------------------------------------------
# World layout (storyboard b05 visuals)
# --------------------------------------------------------------------------
GPU_XS = (-2.35, 1.65)
GPU_YS = (2.40, 1.05, -0.30, -1.65)
GPU_U = 1.05
N_GPU = 8
GPU_C = np.array([[GPU_XS[g % 2], GPU_YS[g // 2], 0.0] for g in range(N_GPU)])   # g = 2*row + col
SIDE = np.array([-1.0 if g % 2 == 0 else 1.0 for g in range(N_GPU)])            # -1 left, +1 right
HERO = 2                                                                       # (-2.35, 1.05)
INNER_X = np.where(SIDE < 0, GPU_XS[0] + GPU_U / 2, GPU_XS[1] - GPU_U / 2)     # -1.825 / 1.125
SW_C = np.array([X_C, 0.375, 0.0])
SW_W, SW_H, SW_R = 0.9, 4.8, 0.2
SW_EDGE_X = np.where(SIDE < 0, SW_C[0] - SW_W / 2, SW_C[0] + SW_W / 2)        # -0.80 / 0.10
CH_W, CH_H, CH_R = 6.4, 5.5, 0.30
LINK_W, LINK_OP = 4.0, 0.80
SW_X0, SW_X1 = SW_C[0] - SW_W / 2, SW_C[0] + SW_W / 2                       # -0.80 / 0.10
# all-reduce traffic: every GPU sends a packet GPU-to-GPU THROUGH the NVSwitch to its partner on
# the other side (same row). Full duplex: left->right rides just above the link, right->left
# just below, so the two packets of a pair pass each other inside the switch.
PARTNER = np.arange(N_GPU) ^ 1
DIRN = -SIDE                                                                 # +1 = moving right
PKT = 0.22
LANE = 0.10
LANE_Y = GPU_C[:, 1] + DIRN * LANE
X_FROM = INNER_X + DIRN * PKT / 2                                            # leaves its GPU's edge ...
X_TO = INNER_X[PARTNER] - DIRN * PKT / 2                                     # ... lands on the partner's
TRAIL = 1.50                                                                 # long, bright comet trail
TRAIL_A = (0.0, 0.30, 0.65, 1.0)                                             # trail alpha stops, tail -> head
CORE_A = (0.0, 0.0, 0.45, 0.90)                                              # its hot INK core (front half)
PKT_CORE = 0.30                                                              # packet fill lifted toward INK

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
# Pacing (5.5 s slot): each idea is SEQUENCED - card first, a read, the motion, then a held payoff.
# Only the gaps between motions grew; every motion keeps its old duration (EXPAND 0.50, texture
# write 0.25, packet flight 0.58, LIME flash 0.30, light wave 0.55, collapse 0.40).
#   0.00-0.50  EXPAND; '8 GPUs.' POPs on frame 1, 'Wired together by NVLink.' 0.12 (fully in 0.37)
#   0.80       cycle 1 compute = 135,168 SLAM + '8 GPUs × 16,896' (1.6 s fully visible); the
#              first packets leave at 1.12, 0.75 s after the sub-line is fully in
#   1.85       cycle 2 (under the push-in); on its arrival (2.65) c1 hard-swaps to 'Share results.'
#              (POP): c1 0.017 -> 2.65 = 2.63 s for its 6 words (>= 2.60), no fade/CARD_IN between
#   3.10       cycle 3 compute; its packets leave 3.32 (0.67 s after 'Share results.' lands)
#   3.90       third arrival = the light wave; held LIME result 3.90 -> 4.933 (1.03 s), collapse
C_COMP = (0.80, 1.85, 3.10)          # compute: every GPU writes its own partial result (texture)
C_SEND = (0.90, 1.85, 3.10)          # exchange origin c0: +T_DEP leave, +T_SW switch, +T_ARR land
T_DEP, T_ARR = 0.22, 0.80            # packet flight inside a cycle (c0 + ...): inside the switch +0.45..0.57
T_SW = 0.45                          # the switch flashes as the packets enter it
MULT_HOLD = 1.70                     # '8 GPUs × 16,896': fully visible 0.90 -> 2.50
T_C2 = round(C_SEND[1] + T_ARR, 4)   # 2.65: 'Share results.' POPs with the second arrival ...
T_C1_OUT = T_C2                      # ... as the '8 GPUs.' card hard-cuts (the card-swap grammar)
T_WAVE = round(C_SEND[2] + T_ARR, 4) # 3.90: third arrival = light wave
T_DRIFT0, T_DRIFT1 = C_COMP[1], C_COMP[1] + 0.80   # cycle 2: slow stage push-in 1.0 -> 1.04
T_COL, T_SET, T_EXIT = cut_times(5.5)               # 4.933 / 5.333 / 5.45
T_TEXT_OUT = T_EXIT                                 # kicker + headline hold to the exit, then hard-cut
# the held result: after the light wave every GPU keeps a LIME glow (it holds the shared answer)
# that breathes until the collapse fades it, so the 1 s hold on the synchronized wave is alive
HELD, HELD_AMP, HELD_T = 0.55, 0.10, 0.90           # LIME level, breathing amplitude, period (s)
# ... and a faint diagonal sheen sweeps all 8 shared-result textures IN SYNC (same phase on every GPU)
SHEEN, SHEEN_T, SHEEN_IN = 0.35, 0.75, (0.30, 0.60)  # peak lift, sweep period, fade-in after T_WAVE

# --------------------------------------------------------------------------
# GPU textures: partial results differ per GPU, the all-reduced result is shared
# --------------------------------------------------------------------------
_GPU_P0 = (0.25 + 0.65 * np.random.default_rng(5).random((8, 8))).reshape(64)   # token glyph pattern
_ROW, _COL = np.divmod(np.arange(64), 8)
PHASE = (_ROW + _COL) / 14.0         # diagonal write order inside a texture
_prng = np.random.default_rng(55)
PARTIAL = 0.25 + 0.65 * _prng.random((len(C_COMP), N_GPU, 64))
COMBINED = 0.25 + 0.65 * _prng.random((len(C_COMP), 64))
COMBINED[-1] = _GPU_P0               # the last all-reduce returns the token's own glyph

C_CY = RGB_CYAN
C_LIME = RGB_LIME
C_INK = RGB_INK
C_MAG = RGB_MAGENTA

_RECT = Rectangle(width=1.0, height=1.0).points.copy()


class B05Node(BeatScene):
    def beat(self):
        # ---------------- entry: E04 (gpu token + HUD '16,896') ----------------
        st = enter(self)
        hud = st.hud

        # ---------------- world at rest ----------------
        self.chassis = RoundedRectangle(width=CH_W, height=CH_H, corner_radius=CH_R, stroke_color=CYAN,
                                        stroke_width=2.5, stroke_opacity=0.8, fill_color=CYAN,
                                        fill_opacity=0.03).move_to(AC)
        self.gpus = VGroup(*[unit_with_glyph("gpu", GPU_U, GPU_C[g]) for g in range(N_GPU)])
        self.switch = RoundedRectangle(width=SW_W, height=SW_H, corner_radius=SW_R, stroke_color=MAGENTA,
                                       stroke_width=3, fill_color=MAGENTA, fill_opacity=0.12).move_to(SW_C)
        lab = mono("NVSWITCH", 34, MAGENTA, BOLD).rotate(PI / 2).move_to(SW_C)
        assert lab.height < SW_H - 0.4 and lab.width < SW_W - 0.2
        assert_safe(lab)
        self.sw_label = lab
        self._lab_glyphs = lab.family_members_with_points()
        self.links = VGroup(*[Line([INNER_X[g], GPU_C[g, 1], 0], [SW_EDGE_X[g], GPU_C[g, 1], 0],
                                   stroke_color=MAGENTA, stroke_width=LINK_W, stroke_opacity=LINK_OP)
                              for g in range(N_GPU)])
        self.world = VGroup(self.chassis, self.links, self.switch, self.sw_label, self.gpus)
        bottom = VGroup(self.gpus[6], self.gpus[7])
        # late fades keep the chassis / switch / bottom GPUs off the HUD number while the
        # camera is still pulling back (at e=0 they would sit over '16,896')
        self._exp_kw = dict(hero=self.gpus[HERO], token="gpu",
                            fades={self.chassis: (0.75, 0.97), self.switch: (0.6, 0.9), self.links: (0.5, 0.9),
                                   self.sw_label: (0.93, 1.0), bottom: (0.7, 0.95)})
        EXPAND(self.world, GPU_C[HERO], GPU_U, 0.0, **self._exp_kw)

        # ---------------- pooled packets + gradient trails ----------------
        self.packets = VGroup(*[VMobject(stroke_width=0, fill_color=MAGENTA, fill_opacity=0)
                                for _ in range(N_GPU)])
        self.trails = VGroup(*[VMobject(stroke_width=0, fill_opacity=0) for _ in range(N_GPU)])
        self.cores = VGroup(*[VMobject(stroke_width=0, fill_opacity=0) for _ in range(N_GPU)])
        for m in [*self.packets, *self.trails, *self.cores]:
            m.points = EMPTY.copy()
        self.fx = VGroup(self.trails, self.cores, self.packets)

        # ---------------- text (kicker + headline in on the cut; they hold through the collapse and
        # settle and hard-cut at T_EXIT with the canonical E05 swap, so no frame near a cut is bare) ----
        track = TextTrack(self)
        track.on(kick("SCALE 4/6 · ONE SERVER"), 0.0, T_TEXT_OUT, exit="cut")
        c1 = card([("8 GPUs.", "H96", INK, {"8": CYAN}), ("Wired together by NVLink.", "S56", MAGENTA)])
        track.on(c1[0], 1 / 60, T_C1_OUT, enter="pop", exit="cut")   # POPs on story frame 1 (frame 0
        track.on(c1[1], 0.12, T_C1_OUT, exit="cut")                  # stays E04); hard swap at 2.65
        c2 = card([("Share results.", "H96", INK, {"results.": LIME})])
        track.on(c2, T_C2, T_TEXT_OUT, enter="pop", exit="cut")

        # ---------------- HUD: the jump explains its own multiplication ----------------
        hud.slam(C_COMP[0], "135,168")
        hud.mult(C_COMP[0], "8 GPUs × 16,896", hold=MULT_HOLD)

        # ---------------- z-order: dust, ring, world, packets, shock ring, text, HUD ----------------
        self.remove(st.token)                         # the hero GPU now draws the token
        add_behind(self, self.world, self.fx)
        entry_ring(self)
        self.add(track)
        self.remove(hud)
        self.add(hud)
        self.collapse = None
        self._draw(0.0)
        drive(self, self._draw)
        # cycle 2 runs under a slow push-in (stage only; added after the driver so it sees
        # this frame's drawing); released right before the collapse captures the world
        self.drift = drift(self, VGroup(self.world, self.fx), T_DRIFT0, T_DRIFT1, s1=1.04, dy=0.04,
                           ease=ease_in_out_sine)
        k, b = self.drift.pose(T_WAVE)
        self.ring = ShockRing(self, T_WAVE, SW_C * k + b, r0=0.3, r1=3.5, w0=10.0, op0=1.0, dur=0.50,
                              color=MAGENTA)
        self.add(self.ring)
        self.remove(track, hud)
        self.add(track, hud)

        # ---------------- audio ----------------
        self.cue("hit", gain=0.8, note="match-cut and ring")
        self.cue("whoosh", dt=0.05, dur=0.50, direction="up", note="GPUs fly in")
        self.cue("hit", dt=C_COMP[0], gain=0.6, note="x8 compute")
        for c0 in C_SEND:
            self.cue("whoosh", dt=round(c0 + T_DEP, 4), dur=0.55, direction="up",
                     note="packets fly GPU -> NVSwitch -> partner GPU")
            self.cue("tick", dt=round(c0 + T_SW, 4), note="packets cross the switch (flash)")
        for c0 in C_SEND[:2]:
            self.cue("pop", dt=round(c0 + T_ARR, 4), note="results land: every GPU flashes LIME")
        self.cue("riser", dt=round(T_WAVE - 1.2, 4), dur=1.2, note="into the third cycle")
        self.cue("hit", dt=T_WAVE, gain=0.8, note="all 8 flash together")
        self.cue("shimmer", dt=T_WAVE, dur=1.0, note="rides the held LIME result")
        self.cue("whoosh", dt=round(T_COL - 0.10, 4), dur=0.55, direction="down", note="collapse")
        self.cue("tick", dt=round(T_SET, 4), gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_COL)
        for m in [*self.packets, *self.trails, *self.cores]:
            m.points = EMPTY.copy()
        self.drift.release()                          # the collapse starts from the drifted pose
        self.collapse = COLLAPSE(self.chassis, VGroup(self.links, self.switch, self.sw_label, self.gpus), "node")
        self.add(self.collapse.glyph)
        hold_floor(self, T_EXIT)
        exit_to(self)                                 # canonical E05, static to 5.50

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.collapse is not None:
            self.collapse.render(now, T_COL, T_SET, T_EXIT)
            if now < T_SET - EPS:                     # the light wave keeps decaying as it fades
                e = ease_in_out_expo(ramp(now, T_COL, T_SET))
                self._style(now, 1.0 - ramp(e, 0.45, 0.80), self.collapse.s1 ** e)
            return
        if now < C_COMP[0] - EPS:
            ink = 1.0 if flash_frames(now, 1 / 60, 2) else 0.0
            EXPAND(self.world, GPU_C[HERO], GPU_U, ease_out_expo(ramp(now, 0.0, EXPAND_TIME)), ink=ink)
            return
        self._style(now, 1.0)
        self._draw_packets(now)

    # ---- styles of GPUs, textures, switch, label and links (alpha a multiplies everything;
    # sw_k = the collapse's stroke scale for the link widths this sets)
    def _style(self, now: float, a: float, sw_k: float = 1.0) -> None:
        f = story_frame(now)
        ink = 0 <= f - story_frame(T_WAVE) < 2
        wave = held = 0.0
        if now >= T_WAVE - EPS and not ink:
            wave = 1.0 - ease_in_out_sine(ramp(now, T_WAVE + 2 / 60, T_WAVE + 0.55))
            held = held_glow(now)                     # takes over as the wave decays below it
        lime = max(flash_env(now - (c0 + T_ARR), 0.05, 0.30) for c0 in C_SEND[:2])
        lime = max(lime, wave, held)
        comp = 0.0
        for c0 in C_COMP:
            if now >= c0 - EPS:
                comp = max(comp, 1.0 - ease_out_quad(ramp(now, c0, c0 + 0.30)))
        swf = max(flash_env(now - (c0 + T_SW), 0.06, 0.30) for c0 in C_SEND)   # while the packets cross it
        transit = max(ramp(now, c0 + 0.20, c0 + 0.30) * (1 - ramp(now, c0 + 0.78, c0 + 0.90)) for c0 in C_SEND)

        # ---- GPU units
        if ink:
            u_fill, u_stroke = rgba(INK, 0.85 * a), rgba(INK, a)
        else:
            u_fill = np.array([*lerp(C_CY, C_LIME, lime), (lerp(0.15, 0.60, lime) + 0.08 * comp) * a])
            u_stroke = np.array([*lerp(C_CY, C_LIME, lime), a])
        uf, us = u_fill.reshape(1, 4), u_stroke.reshape(1, 4)

        # ---- textures: partial results written diagonally at the compute time, shared result
        # on arrival (exchange origin + T_ARR)
        tex_rgb = lerp(C_CY, C_LIME, 0.85 * lime)
        sheen, sheen_pos = held_sheen(now)
        for g in range(N_GPU):
            pat = _GPU_P0.copy()
            band = np.zeros(64)
            for cyc, (c0, s0) in enumerate(zip(C_COMP, C_SEND)):
                if now >= c0 - EPS:
                    pos = ease_out_cubic(ramp(now, c0, c0 + 0.25)) * 1.2
                    pat = np.where(PHASE <= pos - 0.1, PARTIAL[cyc, g], pat)
                    if now < c0 + 0.32:
                        band = np.maximum(band, 0.55 * np.exp(-((PHASE - (pos - 0.1)) / 0.10) ** 2)
                                          * (1 - ramp(now, c0 + 0.22, c0 + 0.32)))
                if now >= s0 + T_ARR - EPS:
                    pat = COMBINED[cyc]
            if sheen > 0:
                band = np.maximum(band, sheen * np.exp(-((PHASE - sheen_pos) / 0.12) ** 2))
            op = np.clip(pat * (1.0 + 0.6 * comp) + band, 0.0, 1.0)
            if ink:
                rows = np.tile(np.array([*C_INK, 1.0]), (64, 1))
            else:
                op = np.maximum(op, lerp(op, 0.95, wave))
                rows = np.column_stack([np.tile(tex_rgb, (64, 1)), op])
            rows[:, 3] *= a
            gp = self.gpus[g]
            gp.unit.fill_rgbas = uf.copy()
            gp.unit.stroke_rgbas = us.copy()
            for k, sq in enumerate(gp.glyph.submobjects):
                sq.fill_rgbas = rows[k:k + 1].copy()

        # ---- switch: MAGENTA throughout (it moves data): fill 0.12 -> 0.6 on each arrival and
        # on the light wave, INK stroke on arrivals. The label turns INK whenever the body is lit,
        # so it never sits MAGENTA-on-MAGENTA.
        glow = max(swf, wave)
        if ink:
            s_fill, s_stroke = rgba(INK, 0.85 * a), rgba(INK, a)
            l_rgba = rgba(MAGENTA, a)
        else:
            s_fill = np.array([*C_MAG, lerp(0.12, 0.6, glow) * a])
            s_stroke = np.array([*lerp(C_MAG, C_INK, swf), a])
            l_rgba = np.array([*lerp(C_MAG, C_INK, ease_out_cubic(clip01(1.6 * glow))), a])
        self.switch.fill_rgbas = s_fill.reshape(1, 4)
        self.switch.stroke_rgbas = s_stroke.reshape(1, 4)
        for gl in self._lab_glyphs:
            gl.fill_rgbas = l_rgba.reshape(1, 4).copy()

        # ---- NVLinks (MAGENTA): brighten while packets ride them and on the light wave
        if ink:
            l_stroke = rgba(INK, a)
        else:
            l_stroke = np.array([*C_MAG, lerp(LINK_OP + (1.0 - LINK_OP) * transit, 1.0, wave) * a])
        for ln in self.links:
            ln.stroke_rgbas = l_stroke.reshape(1, 4).copy()
            ln.stroke_width = (LINK_W + 2.0 * wave) * sw_k

    # ---- all-reduce packets (pure function of now): GPU -> NVSwitch -> partner GPU
    def _draw_packets(self, now: float) -> None:
        for g in range(N_GPU):
            pk, tr, co = self.packets[g], self.trails[g], self.cores[g]
            q = None
            for c0 in C_SEND:
                if c0 + T_DEP - EPS <= now < c0 + T_ARR - EPS:
                    q = ramp(now, c0 + T_DEP, c0 + T_ARR)
            if q is None:
                for m in (pk, tr, co):
                    m.points = EMPTY.copy()
                continue
            dirn, y, x0 = DIRN[g], LANE_Y[g], X_FROM[g]
            x = lerp(x0, X_TO[g], ease_in_out_sine(q))
            al = ramp(q, 0.0, 0.06) * (1 - ramp(q, 0.94, 1.0))
            sc = lerp(0.6, 1.0, ease_out_cubic(ramp(q, 0.0, 0.20)))
            col = pkt_rgb(x, dirn)
            pk.points = _RECT * np.array([PKT * sc, PKT * sc, 1.0]) + np.array([x, y, 0.0])
            pk.fill_rgbas = np.array([[*lerp(col, C_INK, PKT_CORE), al]])
            # comet trail behind the packet (never longer than the path travelled), coloured by
            # where each part of it is: MAGENTA before the switch, LIME after it
            near = x - dirn * PKT * sc / 2
            far = near - dirn * min(TRAIL, max(0.0, dirn * (near - x0)))
            if abs(near - far) < 1e-3:
                tr.points = EMPTY.copy()
                co.points = EMPTY.copy()
                continue
            us = np.linspace(0, 1, len(TRAIL_A))
            for m, hk, stops, tint in ((tr, 1.0, TRAIL_A, 0.0), (co, 0.34, CORE_A, 0.75)):
                h = PKT * sc * hk
                m.points = poly_pts([[far, y - h / 8, 0], [near, y - h / 2, 0],
                                     [near, y + h / 2, 0], [far, y + h / 8, 0]])
                m.sheen_direction = np.array([dirn, 0.0, 0.0])
                m.fill_rgbas = np.array([[*lerp(pkt_rgb(lerp(far, near, u), dirn), C_INK, tint), a * al]
                                         for u, a in zip(us, stops)])


def held_glow(now: float) -> float:
    """LIME level every GPU keeps after the light wave (it holds the shared result): HELD,
    breathing +/- HELD_AMP over HELD_T (starts at HELD on the wave frame, so max() with the
    decaying wave hands over without a jump)."""
    return HELD + HELD_AMP * math.sin(2 * PI * (now - T_WAVE) / HELD_T)


def held_sheen(now: float) -> tuple[float, float]:
    """(lift, diagonal position) of the synchronized sheen over the held result: fades in over
    SHEEN_IN after the light wave, then a band sweeps every texture (PHASE 0 -> 1) once per
    SHEEN_T; it wraps far outside the texture (-0.35 / 1.35), so the loop has no visible jump."""
    t0 = T_WAVE + SHEEN_IN[0]
    if now < t0 - EPS:
        return 0.0, 0.0
    k = SHEEN * ramp(now, t0, T_WAVE + SHEEN_IN[1])
    return k, ((now - t0) / SHEEN_T) % 1.0 * 1.7 - 0.35


def pkt_rgb(x: float, dirn: float) -> np.ndarray:
    """Packet colour at x: MAGENTA (data in flight) until it enters the NVSwitch, white-hot in
    the middle of the switch (where it is combined), LIME (the reduced result) once it leaves."""
    u = clip01((x - SW_X0) / (SW_X1 - SW_X0))
    if dirn < 0:
        u = 1.0 - u
    if u < 0.5:
        return lerp(C_MAG, C_INK, 0.8 * ease_in_out_sine(2 * u))
    return lerp(C_INK, C_LIME, lerp(0.2, 1.0, ease_in_out_sine(2 * u - 1)))

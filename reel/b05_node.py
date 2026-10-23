"""b05 - SCALE 4/6 - ONE SERVER: 8 GPUs in one box, wired by NVLink through the NVSwitch.

The GPU token (E04) is one of the 8 GPUs of an HGX H100 node: the camera pulls back
(EXPAND, 0.50 s, INK charge on the hero outline). Three all-reduce cycles: every GPU
computes its own slice (its texture rewrites itself to a different partial result),
MAGENTA packets (with long trails) converge on the NVSwitch and stop at its edge, the
switch body flashes (its label turns INK so it stays readable), LIME packets carry the
combined result back and every GPU flashes LIME and now shows the SAME texture (every GPU
holds the full answer). Cycle 2 runs under a slow 1.0 -> 1.04 stage push-in. The third
arrival (3.45) is the light wave: GPUs, links and switch flash INK, then the GPUs go
LIME (results) while the switch and links glow MAGENTA (data movement), and a MAGENTA
shock ring leaves the switch. The chassis then collapses (cut_times) into
handoff_token('node') (E05).

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
PKT = 0.22
TRAIL = 0.50

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
CYCLES = (0.75, 1.70, 2.65)          # c0: compute; +0.25..0.50 out; +0.50 switch; +0.55..0.80 back; +0.80 LIME
T_WAVE = 3.45                        # third arrival = light wave
T_DRIFT0, T_DRIFT1 = CYCLES[1], CYCLES[1] + 0.80   # cycle 2: slow stage push-in 1.0 -> 1.04
T_COL, T_SET, T_EXIT = cut_times(4.5)               # 3.933 / 4.333 / 4.45
T_TEXT_OUT = T_EXIT - 0.20                          # kicker + headline leave during the settle

# --------------------------------------------------------------------------
# GPU textures: partial results differ per GPU, the all-reduced result is shared
# --------------------------------------------------------------------------
_GPU_P0 = (0.25 + 0.65 * np.random.default_rng(5).random((8, 8))).reshape(64)   # token glyph pattern
_ROW, _COL = np.divmod(np.arange(64), 8)
PHASE = (_ROW + _COL) / 14.0         # diagonal write order inside a texture
_prng = np.random.default_rng(55)
PARTIAL = 0.25 + 0.65 * _prng.random((len(CYCLES), N_GPU, 64))
COMBINED = 0.25 + 0.65 * _prng.random((len(CYCLES), 64))
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
        self.packets = VGroup(*[VMobject(stroke_width=0, fill_color=MAGENTA if k < N_GPU else LIME,
                                         fill_opacity=0) for k in range(2 * N_GPU)])
        self.trails = VGroup(*[VMobject(stroke_width=0, fill_opacity=0) for _ in range(2 * N_GPU)])
        for m in [*self.packets, *self.trails]:
            m.points = EMPTY.copy()
        self.fx = VGroup(self.trails, self.packets)

        # ---------------- text (kicker + headline in on the cut, out during the settle) ----------------
        track = TextTrack(self)
        track.on(kick("SCALE 4/6 · ONE SERVER"), 0.0, T_TEXT_OUT)
        c1 = card([("8 GPUs.", "H96", INK, {"8": CYAN}), ("Wired together by NVLink.", "S56", MAGENTA)])
        track.on(c1[0], 0.05, 2.45, enter="pop")      # gone by 2.65, when 'Share results.' enters
        track.on(c1[1], 0.25, 2.45)
        c2 = card([("Share results.", "H96", INK, {"results.": LIME})])
        track.on(c2, 2.65, T_TEXT_OUT)

        # ---------------- HUD: the jump explains its own multiplication ----------------
        hud.slam(CYCLES[0], "135,168")
        hud.mult(CYCLES[0], "8 GPUs × 16,896")

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
        self.cue("hit", dt=CYCLES[0], gain=0.6, note="x8 compute")
        for c0 in CYCLES:
            self.cue("whoosh", dt=round(c0 + 0.375, 4), dur=0.25, direction="up", note="packets converge")
            self.cue("tick", dt=round(c0 + 0.50, 4), note="switch flash")
        for c0 in CYCLES[:2]:
            self.cue("pop", dt=round(c0 + 0.80, 4), note="results return")
        self.cue("riser", dt=2.25, dur=1.2, note="into the third cycle")
        self.cue("hit", dt=T_WAVE, gain=0.8, note="all 8 flash together")
        self.cue("shimmer", dt=T_WAVE, dur=0.8)
        self.cue("whoosh", dt=round(T_COL - 0.10, 4), dur=0.55, direction="down", note="collapse")
        self.cue("tick", dt=round(T_SET, 4), gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_COL)
        for m in [*self.packets, *self.trails]:
            m.points = EMPTY.copy()
        self.drift.release()                          # the collapse starts from the drifted pose
        self.collapse = COLLAPSE(self.chassis, VGroup(self.links, self.switch, self.sw_label, self.gpus), "node")
        self.add(self.collapse.glyph)
        hold_floor(self, T_EXIT)
        exit_to(self)                                 # canonical E05, static to 4.50

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.collapse is not None:
            self.collapse.render(now, T_COL, T_SET, T_EXIT)
            if now < T_SET - EPS:                     # the light wave keeps decaying as it fades
                e = ease_in_out_expo(ramp(now, T_COL, T_SET))
                self._style(now, 1.0 - ramp(e, 0.45, 0.80), self.collapse.s1 ** e)
            return
        if now < CYCLES[0] - EPS:
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
        wave = 0.0
        if now >= T_WAVE - EPS and not ink:
            wave = 1.0 - ease_in_out_sine(ramp(now, T_WAVE + 2 / 60, T_WAVE + 0.55))
        lime = max(flash_env(now - (c0 + 0.80), 0.05, 0.30) for c0 in CYCLES[:2])
        lime = max(lime, wave)
        comp = 0.0
        for c0 in CYCLES:
            if now >= c0 - EPS:
                comp = max(comp, 1.0 - ease_out_quad(ramp(now, c0, c0 + 0.30)))
        swf = max(flash_env(now - (c0 + 0.50), 0.04, 0.25) for c0 in CYCLES)
        transit = max(ramp(now, c0 + 0.20, c0 + 0.30) * (1 - ramp(now, c0 + 0.78, c0 + 0.90)) for c0 in CYCLES)

        # ---- GPU units
        if ink:
            u_fill, u_stroke = rgba(INK, 0.85 * a), rgba(INK, a)
        else:
            u_fill = np.array([*lerp(C_CY, C_LIME, lime), (lerp(0.15, 0.60, lime) + 0.08 * comp) * a])
            u_stroke = np.array([*lerp(C_CY, C_LIME, lime), a])
        uf, us = u_fill.reshape(1, 4), u_stroke.reshape(1, 4)

        # ---- textures: partial results written diagonally at c0, shared result at c0 + 0.80
        tex_rgb = lerp(C_CY, C_LIME, 0.85 * lime)
        for g in range(N_GPU):
            pat = _GPU_P0.copy()
            band = np.zeros(64)
            for cyc, c0 in enumerate(CYCLES):
                if now >= c0 - EPS:
                    pos = ease_out_cubic(ramp(now, c0, c0 + 0.25)) * 1.2
                    pat = np.where(PHASE <= pos - 0.1, PARTIAL[cyc, g], pat)
                    if now < c0 + 0.32:
                        band = np.maximum(band, 0.55 * np.exp(-((PHASE - (pos - 0.1)) / 0.10) ** 2)
                                          * (1 - ramp(now, c0 + 0.22, c0 + 0.32)))
                if now >= c0 + 0.80 - EPS:
                    pat = COMBINED[cyc]
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

    # ---- all-reduce packets (pure function of now): GPU edge <-> switch edge
    def _draw_packets(self, now: float) -> None:
        for g in range(N_GPU):
            y = GPU_C[g, 1]
            d = 1.0 if SW_EDGE_X[g] > INNER_X[g] else -1.0          # toward the switch
            near_gpu, near_sw = INNER_X[g] + d * PKT / 2, SW_EDGE_X[g] - d * PKT / 2
            for k, (leg, x0, x1, edge0, color) in enumerate((("out", near_gpu, near_sw, INNER_X[g], C_MAG),
                                                              ("back", near_sw, near_gpu, SW_EDGE_X[g], C_LIME))):
                pk = self.packets[g + k * N_GPU]
                tr = self.trails[g + k * N_GPU]
                q, live = None, False
                for c0 in CYCLES:
                    ta = c0 + (0.25 if leg == "out" else 0.55)
                    if ta - EPS <= now < ta + 0.25 - EPS:
                        q, live = ramp(now, ta, ta + 0.25), True
                if not live:
                    pk.points = EMPTY.copy()
                    tr.points = EMPTY.copy()
                    continue
                dirn = 1.0 if x1 > x0 else -1.0
                x = lerp(x0, x1, ease_in_out_cubic(q))
                al = ramp(q, 0.0, 0.10) * (1 - ramp(q, 0.90, 1.0))
                sc = lerp(0.6, 1.0, ease_out_cubic(ramp(q, 0.0, 0.25)))
                pk.points = _RECT * np.array([PKT * sc, PKT * sc, 1.0]) + np.array([x, y, 0.0])
                pk.fill_rgbas = np.array([[*color, al]])
                # fading trail behind the packet (never longer than the path travelled)
                near = x - dirn * PKT * sc / 2
                far = near - dirn * min(TRAIL, max(0.0, dirn * (near - edge0)))
                if abs(near - far) < 1e-3:
                    tr.points = EMPTY.copy()
                    continue
                h = PKT * sc * 0.72
                tr.points = poly_pts([[far, y - h / 5, 0], [near, y - h / 2, 0],
                                      [near, y + h / 2, 0], [far, y + h / 5, 0]])
                tr.sheen_direction = np.array([dirn, 0.0, 0.0])
                tr.fill_rgbas = np.array([[*color, 0.0], [*color, 0.85 * al]])

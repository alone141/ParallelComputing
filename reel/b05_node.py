"""b05 - SCALE 4/6 - ONE SERVER: 8 GPUs in one box, NVLink through the NVSwitch.

The GPU token (E04) is one of the 8 GPUs of an HGX H100 node: the camera pulls back
(EXPAND). Three all-reduce cycles: every GPU computes its own slice (its texture
rewrites itself to a different partial result), MAGENTA packets converge on the
NVSwitch, the switch flashes, LIME packets carry the combined result back and every
GPU flashes LIME and now shows the SAME texture (every GPU holds the full answer).
The third arrival (3.45) is the light wave: GPUs, links and switch flash INK, then
LIME. The chassis then collapses into handoff_token('node') (E05).

Everything that moves is a pure function of self.now (one driver updater); the scene
only holds until the structural switch points (collapse capture, canonical exit).
"""

from style import *
from motif import *
import math

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
PKT = 0.14
TRAIL = 0.30

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
T_EXPAND = 0.70
CYCLES = (0.75, 1.70, 2.65)          # c0: compute; +0.25..0.50 out; +0.50 switch; +0.55..0.80 back; +0.80 LIME
T_WAVE = 3.45                        # third arrival = light wave
T_COLLAPSE = 3.65
T_SETTLE = 4.28
T_EXIT = 4.45                        # canonical E05: 3 static 60-fps frames

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

C_CY = np.array(CYAN.to_rgb())
C_LIME = np.array(LIME.to_rgb())
C_INK = np.array(INK.to_rgb())
C_MAG = np.array(MAGENTA.to_rgb())

_RECT = Rectangle(width=1.0, height=1.0).points.copy()


def flash_env(d: float, rise: float, dur: float) -> float:
    """0 -> 1 over `rise`, back to 0 at `dur` (ease_in_out_sine), 0 outside."""
    if d < -EPS or d >= dur - EPS:
        return 0.0
    return ramp(d, 0.0, rise) * (1.0 - ease_in_out_sine(ramp(d, rise, dur)))


def poly_pts(corners) -> np.ndarray:
    """Closed polygon as straight cubic segments (manim point layout)."""
    c = [np.asarray(p, float) for p in corners]
    out = []
    for a, b in zip(c, c[1:] + c[:1]):
        out.extend([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b])
    return np.array(out)


def hold_floor(scene, t: float) -> None:
    """Hold until the last frame at or before t (exact at 60 fps; 30-fps previews round down)."""
    n = int(math.floor(t * config.frame_rate + 1e-6)) - scene._frames
    if n > 0:
        scene.wait((n - 0.5) / config.frame_rate)


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
        self.sw_label = lab
        self.links = VGroup(*[Line([INNER_X[g], GPU_C[g, 1], 0], [SW_EDGE_X[g], GPU_C[g, 1], 0],
                                   stroke_color=MAGENTA, stroke_width=3, stroke_opacity=0.55)
                              for g in range(N_GPU)])
        self.world = VGroup(self.chassis, self.links, self.switch, self.sw_label, self.gpus)
        bottom = VGroup(self.gpus[6], self.gpus[7])
        # late fades keep the chassis / switch / bottom GPUs off the HUD number while the
        # camera is still pulling back (at e=0 they would sit over '16,896')
        EXPAND(self.world, GPU_C[HERO], GPU_U, 0.0, hero=self.gpus[HERO], token="gpu",
               fades={self.chassis: (0.75, 0.97), self.switch: (0.6, 0.9), self.links: (0.5, 0.9),
                      self.sw_label: (0.93, 1.0), bottom: (0.7, 0.95)})

        # ---------------- pooled packets + gradient trails ----------------
        self.packets = VGroup(*[VMobject(stroke_width=0, fill_color=MAGENTA if k < N_GPU else LIME,
                                         fill_opacity=0) for k in range(2 * N_GPU)])
        self.trails = VGroup(*[VMobject(stroke_width=0, fill_opacity=0) for _ in range(2 * N_GPU)])
        for m in [*self.packets, *self.trails]:
            m.points = np.zeros((0, 3))
        self.fx = VGroup(self.trails, self.packets)

        # ---------------- text ----------------
        track = TextTrack(self)
        track.on(kick("SCALE 4/6 · ONE SERVER"), 0.20, 4.0)
        c1 = card([("8 GPUs.", "H96", INK, {"8": CYAN}), ("NVLink: 900 GB/s per GPU", "S56", MAGENTA)])
        track.on(c1[0], 0.25, 2.45)                   # gone by 2.65, when 'Share results.' enters
        track.on(c1[1], 0.40, 2.45)
        c2 = card([("Share results.", "H96", INK, {"results.": LIME})])
        track.on(c2, 2.65, 4.0)

        # ---------------- HUD ----------------
        hud.slam(CYCLES[0], "135,168", mult="×8")

        # ---------------- z-order: dust, ring, world, packets, text, HUD ----------------
        self.remove(st.token)                         # the hero GPU now draws the token
        add_behind(self, self.world, self.fx)
        entry_ring(self)
        self.add(track)
        self.remove(hud)
        self.add(hud)
        self.collapse = None
        self._draw(0.0)
        driver = Mobject()
        driver.add_updater(lambda m, dt: self._draw(self.now))
        self.add(driver)

        # ---------------- audio ----------------
        self.cue("hit", gain=0.8, note="match-cut and ring")
        self.cue("whoosh", dt=0.35, dur=0.7, direction="up", note="GPUs fly in")
        self.cue("hit", dt=CYCLES[0], gain=0.6, note="x8 compute")
        for c0 in CYCLES:
            self.cue("whoosh", dt=round(c0 + 0.375, 4), dur=0.25, direction="up", note="packets converge")
            self.cue("tick", dt=round(c0 + 0.50, 4), note="switch flash")
        for c0 in CYCLES[:2]:
            self.cue("pop", dt=round(c0 + 0.80, 4), note="results return")
        self.cue("riser", dt=2.25, dur=1.2, note="into the third cycle")
        self.cue("hit", dt=T_WAVE, gain=0.8, note="all 8 flash together")
        self.cue("shimmer", dt=T_WAVE, dur=0.8)
        self.cue("whoosh", dt=3.97, dur=0.63, direction="down", note="collapse")
        self.cue("tick", dt=T_SETTLE, gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_COLLAPSE)
        for m in [*self.packets, *self.trails]:
            m.points = np.zeros((0, 3))
        self.collapse = COLLAPSE(self.chassis, VGroup(self.links, self.switch, self.sw_label, self.gpus), "node")
        self.add(self.collapse.glyph)
        hold_floor(self, T_EXIT)
        exit_to(self)                                 # canonical E05, static to 4.50

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.collapse is not None:
            if now < T_SETTLE - EPS:
                e = ease_in_out_expo(ramp(now, T_COLLAPSE, T_SETTLE))
                self.collapse.at(e)
                self._style(now, 1.0 - ramp(e, 0.45, 0.80))   # the light wave keeps decaying
            else:
                self.collapse.settle(ramp(now, T_SETTLE, T_EXIT))
            return
        if now < CYCLES[0] - EPS:
            EXPAND(self.world, GPU_C[HERO], GPU_U, ease_out_expo(ramp(now, 0.0, T_EXPAND)))
            return
        self._style(now, 1.0)
        self._draw_packets(now)

    # ---- styles of GPUs, textures, switch and links (alpha a multiplies everything)
    def _style(self, now: float, a: float) -> None:
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

        # ---- switch: fill 0.12 -> 0.6 -> 0.12, INK stroke on each arrival. After the light
        # wave the switch and links stay LIME while their glow decays (no MAGENTA<->LIME
        # hue lerp: its midpoint is a muddy grey); they fade out in the collapse.
        lit = now >= T_WAVE - EPS
        if ink:
            s_fill, s_stroke = rgba(INK, 0.85 * a), rgba(INK, a)
        else:
            col = C_LIME if lit else C_MAG
            s_fill = np.array([*col, lerp(0.12, 0.6, max(swf, wave)) * a])
            s_stroke = np.array([*lerp(col, C_INK, swf), a])
        self.switch.fill_rgbas = s_fill.reshape(1, 4)
        self.switch.stroke_rgbas = s_stroke.reshape(1, 4)

        # ---- NVLinks: brighten while packets ride them
        if ink:
            l_stroke = rgba(INK, a)
        else:
            col = C_LIME if lit else C_MAG
            l_stroke = np.array([*col, lerp(0.55 + 0.40 * transit, 0.95, wave) * a])
        for ln in self.links:
            ln.stroke_rgbas = l_stroke.reshape(1, 4).copy()

    # ---- all-reduce packets (pure function of now)
    def _draw_packets(self, now: float) -> None:
        for g in range(N_GPU):
            y = GPU_C[g, 1]
            for k, (t0, x0, x1, color) in enumerate((("out", INNER_X[g], X_C, C_MAG),
                                                      ("back", X_C, INNER_X[g], C_LIME))):
                pk = self.packets[g + k * N_GPU]
                tr = self.trails[g + k * N_GPU]
                q, live = None, False
                for c0 in CYCLES:
                    ta = c0 + (0.25 if t0 == "out" else 0.55)
                    if ta - EPS <= now < ta + 0.25 - EPS:
                        q, live = ramp(now, ta, ta + 0.25), True
                if not live:
                    pk.points = np.zeros((0, 3))
                    tr.points = np.zeros((0, 3))
                    continue
                d = 1.0 if x1 > x0 else -1.0
                x = lerp(x0, x1, ease_in_out_cubic(q))
                al = ramp(q, 0.0, 0.12) * (1 - ramp(q, 0.88, 1.0))
                sc = lerp(0.6, 1.0, ease_out_cubic(ramp(q, 0.0, 0.25)))
                pk.points = _RECT * np.array([PKT * sc, PKT * sc, 1.0]) + np.array([x, y, 0.0])
                pk.fill_rgbas = np.array([[*color, al]])
                # fading trail behind the packet (never longer than the path travelled)
                near = x - d * PKT * sc / 2
                far = near - d * min(TRAIL, abs(near - x0))
                if abs(near - far) < 1e-3:
                    tr.points = np.zeros((0, 3))
                    continue
                h = PKT * sc * 0.72
                tr.points = poly_pts([[far, y - h / 4, 0], [near, y - h / 2, 0],
                                      [near, y + h / 2, 0], [far, y + h / 4, 0]])
                tr.sheen_direction = np.array([d, 0.0, 0.0])
                tr.fill_rgbas = np.array([[*color, 0.0], [*color, 0.8 * al]])

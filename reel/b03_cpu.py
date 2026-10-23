"""b03 - SCALE 2/6 - ONE CPU: 16 cores, 16 separate threads, one shared memory.

The core token (E02) is core T6 of a 16-core CPU: the camera pulls back (EXPAND). Each
core runs its own thread on its own clock: fetch (MAGENTA packet from shared memory),
compute (its 16 lanes light), write back (LIME packet + flash on the memory bar), idle;
visibly out of phase, irregular ticks. At 2.20 the power-on light wave lights all 256
lanes (HUD 16 -> 256, x16) and traffic doubles. At 3.80 traffic drains, the CPU package
outline draws on and the chip collapses into handoff_token('cpu') (E03).

Everything that moves is a pure function of self.now (one driver updater).
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# World layout (storyboard b03 visuals)
# --------------------------------------------------------------------------
CORE_XS = (-2.15, -0.95, 0.25, 1.45)
CORE_YS = (2.75, 1.55, 0.35, -0.85)
CORE_U = 0.95
N_CORES = 16
CORE_C = np.array([[CORE_XS[i % 4], CORE_YS[i // 4], 0.0] for i in range(N_CORES)])
COL_X = np.array([CORE_XS[i % 4] for i in range(N_CORES)])
HERO = 5                                    # T6 at (-0.95, 1.55)
MEM_C = np.array([X_C, -1.95, 0.0])
MEM_TOP = -1.70
PKT = 0.12

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
T_EXPAND = 0.70
T_THREADS = 0.75
T_POWER = 2.20
T_DRAIN = 3.80                              # no new fetches; in-flight work fades by 3.92
T_DRAINED = 235 / 60                        # 3.917: on the 60-fps grid (the collapse capture frame)
T_PKG = 3.80                                # package outline Create 3.80 -> 3.92
T_COLLAPSE = T_DRAINED
T_SETTLE = 4.31                             # collapse 3.92 -> 4.31, settle 4.31 -> 4.45
T_EXIT = 4.45                               # canonical E03: 3 static 60-fps frames

# per-core thread clocks (storyboard: P = 0.85 + 0.35u, o = 0.7v, u, v from default_rng(3))
_rng = np.random.default_rng(3)
U = _rng.random(N_CORES)
V = _rng.random(N_CORES)
PER = 0.85 + 0.35 * U
OFF = 0.70 * V
C22 = (T_POWER - T_THREADS - OFF) / PER     # cycles completed at the power-on (rate doubles after)
MAX_CYC = 24
_drng = np.random.default_rng(31)
D_FETCH = _drng.uniform(-0.35, 0.35, (N_CORES, MAX_CYC))   # bar point a fetch leaves from
D_WRITE = _drng.uniform(-0.35, 0.35, (N_CORES, MAX_CYC))   # bar point a result lands on

GLYPH_IDLE = 0.35
GLYPH_ON = 0.90
FILL_REST = 0.15
FILL_HOT = 0.45


def cycles_at(t: float) -> np.ndarray:
    """Cumulative thread cycles per core at time t (rate doubles after the power-on);
    negative = not started yet."""
    tau = t - T_THREADS - OFF
    if t < T_POWER:
        return tau / PER
    return C22 + 2.0 * (t - T_POWER) / PER


def time_of(i: int, c: float) -> float:
    """Inverse of cycles_at for core i."""
    if c <= C22[i]:
        return T_THREADS + OFF[i] + c * PER[i]
    return T_POWER + (c - C22[i]) * PER[i] / 2.0


def compute_starts() -> list[tuple[float, int]]:
    """(time, core) of every COMPUTE start before the drain, sorted."""
    out = []
    for i in range(N_CORES):
        for n in range(MAX_CYC):
            if time_of(i, n) >= T_DRAIN - EPS:
                break
            t = time_of(i, n + 0.30)
            if t < T_DRAIN - EPS:
                out.append((t, i))
    return sorted(out)


def thread_label(i: int) -> MarkupText:
    """'T{i+1}' in JetBrains Mono BOLD 34, INK op 0.9; tracking -700 (1/1024 pt) so 'T16'
    is 0.74 wide instead of 0.80 and keeps clear of the 0.95 core's walls."""
    m = MarkupText(f'<span letter_spacing="-700">T{i + 1}</span>', font=FONT_MONO, font_size=34,
                   color=INK, weight=BOLD)
    return m.set_opacity(0.9)


def square_pts(side: float) -> np.ndarray:
    return Square(side_length=side).points.copy()


def ellipse_pts() -> np.ndarray:
    return Ellipse(width=0.5, height=0.2).points.copy()


def hold_floor(scene, t: float) -> None:
    """hold until the last frame at or before t (60 fps: exactly t; 30-fps previews round
    down, so the static exit hold is never a single too-short wait)."""
    n = int(math.floor(t * config.frame_rate + 1e-6)) - scene._frames
    if n > 0:
        scene.wait((n - 0.5) / config.frame_rate)


def tail_pts(pos, prev, w: float) -> np.ndarray:
    """Triangle (as closed corner path) from the packet's back edge to `prev`."""
    d = pos - prev
    L = float(np.hypot(d[0], d[1]))
    if L < 1e-4:
        return None
    nrm = np.array([-d[1], d[0], 0.0]) / L * (w / 2)
    return [pos + nrm, pos - nrm, prev, pos + nrm]


class B03Cpu(BeatScene):
    def beat(self):
        # ---------------- entry: E02 (core token + HUD '16') ----------------
        st = enter(self)
        hud = st.hud

        # ---------------- world at rest ----------------
        self.cores = VGroup()
        self.labels = VGroup()
        for i in range(N_CORES):
            self.cores.add(unit_with_glyph("core", CORE_U, CORE_C[i], glyph_fill=GLYPH_IDLE))
            lab = thread_label(i).move_to(CORE_C[i] + UP * 0.21)
            assert lab.width < CORE_U - 0.18
            self.labels.add(lab)
        self.mem = RoundedRectangle(width=6.4, height=0.50, corner_radius=0.12, stroke_color=MAGENTA,
                                    stroke_width=3, fill_color=MAGENTA, fill_opacity=0.10).move_to(MEM_C)
        mem_lab = kick("SHARED MEMORY", MAGENTA)
        b, t = cap_box(mem_lab)
        mem_lab.shift([X_C - mem_lab.get_center()[0], MEM_C[1] - (b + t) / 2, 0])
        assert_safe(mem_lab)
        self.mem_lab = mem_lab
        self.stubs = VGroup(*[Line([x, -1.325, 0], [x, MEM_TOP, 0], stroke_color=DIM, stroke_width=2)
                              for x in CORE_XS])
        self.world = VGroup(self.stubs, self.mem, self.mem_lab, self.cores, self.labels)
        hero = self.cores[HERO]
        # the memory bar/stubs/label come in late (e 0.80 -> 0.97, i.e. by ~0.35 s): earlier they
        # would sweep across the HUD number while the camera is still pulling back
        EXPAND(self.world, CORE_C[HERO], CORE_U, 0.0, hero=hero, token="core",
               fades={self.labels: (0.3, 0.8), self.mem: (0.8, 0.97), self.mem_lab: (0.8, 0.97),
                      self.stubs: (0.8, 0.97)})          # captures the rest layout, draws e = 0

        # rest copies for the thread updater (taken at rest, before any camera move)
        rig = self.world._motif_expand
        self._core_rest = []
        for i, core in enumerate(self.cores):
            mems = [core.unit] + list(core.glyph.submobjects) + list(self.labels[i].family_members_with_points())
            self._core_rest.append([(m, rig.rest[rig.index[id(m)]].p.copy()) for m in mems])
        self._lab_rest = [(m, rig.rest[rig.index[id(m)]].fr.copy())
                          for m in self.labels.family_members_with_points() + self.mem_lab.family_members_with_points()]

        # ---------------- pooled packets + write flashes (above the cores) ----------------
        self._sq = square_pts(PKT)
        self._el = ellipse_pts()
        self.packets = VGroup(*[Square(side_length=PKT, stroke_width=0, fill_color=MAGENTA if k < 16 else LIME,
                                       fill_opacity=0) for k in range(32)])
        self.flashes = VGroup(*[Ellipse(width=0.5, height=0.2, stroke_width=0, fill_color=LIME, fill_opacity=0)
                                for _ in range(N_CORES)])
        # comet tails: one triangle per packet, pointing back along its path
        self.tails = VGroup(*[VMobject(stroke_width=0, fill_color=MAGENTA if k < 16 else LIME, fill_opacity=0)
                              .set_points_as_corners([ORIGIN, RIGHT * 0.01, UP * 0.01, ORIGIN]) for k in range(32)])
        self.fx = VGroup(self.flashes, self.tails, self.packets)

        # ---------------- package outline (drawn on at 3.80) ----------------
        self.pkg_full = RoundedRectangle(width=6.9, height=5.6, corner_radius=0.35, stroke_color=CYAN,
                                         stroke_width=3, fill_color=CYAN, fill_opacity=0).move_to([X_C, 0.50, 0])
        self.pkg = self.pkg_full.copy().set_stroke(opacity=0)

        # ---------------- text ----------------
        track = TextTrack(self)
        track.on(kick("SCALE 2/6 · ONE CPU"), 0.20, T_DRAIN)
        c_a = card([("16 cores.", "H96", INK, {"16": CYAN}), ("16 separate threads.", "S56", INK)])
        c_b = card([("16 cores.", "H96", INK, {"16": CYAN}), ("× 16 lanes each", "S56", CYAN)])
        track.on(c_a[0], 0.30, T_DRAIN)
        # 'separate threads' lifts off 2.15 -> 2.20 so the power-on hard cut lands on a clean line
        track.on(c_a[1], 0.40, 2.15, exit_dur=0.05)
        track.on(c_b[1], T_POWER, T_DRAIN, enter="pop")

        # ---------------- HUD ----------------
        hud.slam(T_POWER, "256", mult="×16")

        # ---------------- z-order: dust, ring, world, fx, package, text, HUD ----------------
        self.remove(st.token)                  # the hero core now draws the token
        add_behind(self, self.world, self.fx, self.pkg)
        entry_ring(self)
        self.add(track)
        self.remove(hud)
        self.add(hud)                          # HUD stays on top
        self.collapse = None
        self._draw(0.0)
        driver = Mobject()
        driver.add_updater(lambda m, dt: self._draw(self.now))
        self.add(driver)

        # ---------------- audio ----------------
        self.cue("hit", gain=0.8, note="match-cut and ring")
        self.cue("whoosh", dt=0.35, dur=0.7, direction="up", note="camera pull-back")
        trng = np.random.default_rng(17)
        for t, i in compute_starts():
            pan = float(np.clip((COL_X[i] - X_C) / 1.8 * 0.55 + trng.uniform(-0.15, 0.15), -0.8, 0.8))
            self.cue("tick", dt=round(t, 4), gain=0.3, freq=int(trng.uniform(2600, 5400)), pan=round(pan, 3),
                     note=f"T{i + 1} compute")
        self.cue("riser", dt=1.70, dur=0.50, note="into the power-on")
        self.cue("hit", dt=T_POWER, note="256 lanes light")
        self.cue("sub_drop", dt=T_POWER, dur=0.6)
        self.cue("shimmer", dt=T_POWER, dur=1.0)
        self.cue("whoosh", dt=4.12, dur=0.45, direction="down", note="collapse")
        self.cue("tick", dt=T_SETTLE, gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_COLLAPSE)
        self.collapse = COLLAPSE(self.pkg, self.world, "cpu")
        self.add(self.collapse.glyph)
        hold_floor(self, T_EXIT)
        exit_to(self)                          # canonical E03, static to 4.50

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.collapse is not None:
            if now < T_SETTLE - EPS:
                self.collapse.at(ease_in_out_expo(ramp(now, T_COLLAPSE, T_SETTLE)))
            else:
                self.collapse.settle(ramp(now, T_SETTLE, T_EXIT))
            return
        if now < T_THREADS - EPS:
            EXPAND(self.world, CORE_C[HERO], CORE_U, ease_out_expo(ramp(now, 0.0, T_EXPAND)))
        else:
            self._draw_threads(now)
        self._draw_package(now)

    def _draw_package(self, now: float) -> None:
        a = ease_out_cubic(ramp(now, T_PKG, T_COLLAPSE))
        if a <= 0:
            self.pkg.set_stroke(opacity=0)
        elif a >= 1:
            self.pkg.points = self.pkg_full.points.copy()
            self.pkg.set_stroke(CYAN, width=3, opacity=1)
        else:
            self.pkg.pointwise_become_partial(self.pkg_full, 0, a)
            self.pkg.set_stroke(CYAN, width=3, opacity=1)

    def _draw_threads(self, now: float) -> None:
        drain = 1.0 - ramp(now, T_DRAIN, T_DRAINED)          # in-flight work fades 3.80 -> 3.92
        c = cycles_at(now)
        powered = now >= T_POWER - EPS
        ink_flash = powered and story_frame(now) - story_frame(T_POWER) < 2
        # lane bars at rest: idle .35 before the power-on; after: CYAN 1.0 settling to .9
        if powered:
            g_rest = lerp(1.0, GLYPH_ON, ease_out_cubic(ramp(now, T_POWER + 2 / 60, T_POWER + 0.35)))
        else:
            g_rest = GLYPH_IDLE
        pop = pop_scale(now - T_POWER, 1.06, 0.20) if powered else 1.0
        # light wave: every core flashes (fill .45, INK stroke) on the hit, cooling over 0.3 s
        pw = 0.0
        if powered:
            pw = 1.0 if ink_flash else 1.0 - ease_out_cubic(ramp(now, T_POWER + 2 / 60, T_POWER + 0.32))

        for i in range(N_CORES):
            ci = c[i]
            n = int(math.floor(ci)) if ci >= 0 else -1
            s = ci - n if n >= 0 else -1.0
            alive = n >= 0 and n < MAX_CYC and time_of(i, n) < T_DRAIN - EPS
            # ---- compute glow (s in [0.30, 0.50))
            hot = 0.0
            if alive and 0.30 <= s < 0.50:
                sc = (s - 0.30) / 0.20
                hot = 1.0 if sc < 0.6 else 1.0 - ease_in_out_sine((sc - 0.6) / 0.4)
            hot *= drain
            fill_k = 0.0
            if alive and 0.30 <= s < 0.50:
                fill_k = (1.0 - ease_out_quad((s - 0.30) / 0.20)) * drain
            core = self.cores[i]
            fk = max(fill_k, pw)
            core.unit.fill_rgbas = rgba(CYAN, lerp(FILL_REST, FILL_HOT, fk)).reshape(1, 4)
            core.unit.stroke_rgbas = rgba(interpolate_color(CYAN, INK, max(0.5 * fill_k, 0.85 * pw)), 1.0).reshape(1, 4)
            if ink_flash:
                grow = rgba(INK, 1.0)
            else:
                grow = rgba(CYAN, lerp(g_rest, 1.0, hot))
            for bar in core.glyph.submobjects:
                bar.fill_rgbas = grow.reshape(1, 4).copy()
            # ---- power-on pop (whole core + label, about its centre)
            ctr = CORE_C[i]
            for m, p in self._core_rest[i]:
                m.points = p.copy() if pop == 1.0 else (p - ctr) * pop + ctr

            # ---- packets: fetch (s in [0, .30)), write-back (s in [.50, .80))
            fetch, write = self.packets[i], self.packets[16 + i]
            ftail, wtail = self.tails[i], self.tails[16 + i]
            fa = wa = fta = wta = 0.0
            if alive and s < 0.30:
                q = s / 0.30
                start = np.array([COL_X[i] + D_FETCH[i, n], MEM_TOP, 0.0])
                pos = lerp(start, ctr, ease_in_out_cubic(q))
                fetch.points = self._sq * lerp(0.6, 1.0, ramp(q, 0, 0.15)) + pos
                fa = ramp(q, 0.0, 0.10) * (1 - ramp(q, 0.88, 1.0))
                tp = tail_pts(pos, lerp(start, ctr, ease_in_out_cubic(max(q - 0.16, 0.0))), PKT * 0.8)
                if tp is not None:
                    ftail.set_points_as_corners(tp)
                    fta = 0.35 * fa
            if alive and 0.50 <= s < 0.80:
                q = (s - 0.50) / 0.30
                end = np.array([COL_X[i] + D_WRITE[i, n], MEM_TOP, 0.0])
                pos = lerp(ctr, end, ease_in_out_cubic(q))
                write.points = self._sq * lerp(1.0, 0.7, ramp(q, 0.85, 1.0)) + pos
                wa = ramp(q, 0.0, 0.12) * (1 - ramp(q, 0.92, 1.0))
                tp = tail_pts(pos, lerp(ctr, end, ease_in_out_cubic(max(q - 0.16, 0.0))), PKT * 0.8)
                if tp is not None:
                    wtail.set_points_as_corners(tp)
                    wta = 0.35 * wa
            fetch.fill_rgbas = rgba(MAGENTA, fa * drain).reshape(1, 4)
            write.fill_rgbas = rgba(LIME, wa * drain).reshape(1, 4)
            ftail.fill_rgbas = rgba(MAGENTA, fta * drain).reshape(1, 4)
            wtail.fill_rgbas = rgba(LIME, wta * drain).reshape(1, 4)

            # ---- write flash on the memory bar (0.25 s after each arrival)
            fl = self.flashes[i]
            fop = 0.0
            if n >= 0:
                na = n if s >= 0.80 else n - 1
                if 0 <= na < MAX_CYC and time_of(i, na) < T_DRAIN - EPS:
                    ta = time_of(i, na + 0.80)
                    d = now - ta
                    if 0 <= d < 0.25:
                        k = d / 0.25
                        fop = 0.6 * (1 - ease_out_quad(k)) * drain
                        sc = lerp(0.6, 1.25, ease_out_cubic(k))
                        fl.points = self._el * np.array([sc, sc, 1.0]) + np.array(
                            [COL_X[i] + D_WRITE[i, na], MEM_TOP, 0.0])
            fl.fill_rgbas = rgba(LIME, fop).reshape(1, 4)

        # ---- T labels + SHARED MEMORY fade out with the drain
        la = ease_in_out_sine(drain) if now >= T_DRAIN else 1.0
        if la < 1.0 or getattr(self, "_lab_a", 1.0) != la:
            for m, fr in self._lab_rest:
                v = fr.copy()
                v[:, 3] *= la
                m.fill_rgbas = v
            self._lab_a = la

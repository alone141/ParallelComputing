"""b07 - SCALE 6/6 - THE PLANET: volunteer home PCs over the internet (Folding@home, 2020).

The cluster token (E06) shrinks into the hub (project server) at the centre of a spinning
globe that settles in from 3x (the widest zoom-out). Volunteer PCs pop up across the
front side; from 1.00 a work unit leaves the hub every 0.10 s (MAGENTA arc + packet),
the device computes (INK flash + ring) and a LIME result rides back. Job 0 is slowed so
its 'work unit' / 'result' labels can be read. 1.00: the HUD swaps units like a slot
machine ('VOLUNTEER GPUs · 2020', counting 0 -> 350,000+); 2.40 SLAM + a light wave
across every visible device. 4.45 glitch-out, 4.75 hard cut to BG + HUD (E07).

Everything that moves is a pure function of self.now (one driver updater). The 900 globe
points and 300 devices are drawn as a handful of multi-subpath VMobjects (one per
quantised colour).
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# World layout (storyboard b07 visuals)
# --------------------------------------------------------------------------
G = AC.copy()                                  # globe centre
R = 2.75
BETA = 0.35                                    # tilt
SPIN = 0.30                                    # rad/s
N_PTS = 900
_k = np.arange(N_PTS)
PT_PHI = np.arcsin(-1 + 2 * (_k + 0.5) / N_PTS)
PT_LAM = 2.39996 * _k
PT_R = 0.022

N_DEV = 300
_drng = np.random.default_rng(7)
_sinmax = math.sin(math.radians(65))
DEV_PHI = np.arcsin(_drng.uniform(-_sinmax, _sinmax, N_DEV))
DEV_LAM = _drng.uniform(0, TAU, N_DEV)
DEV_PHI[0], DEV_LAM[0] = math.radians(35), 0.52            # job 0's device: (1.30, 1.41) at t = 1.0
DEV_SQ = 0.05
HUB_SIZE = 0.40

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
T_FORM = 0.80
T_POP0, T_POP1 = 0.80, 2.40
POP_DUR = 0.20
T_JOBS0, T_JOBS1 = 1.00, 4.30
JOB_DT = 0.10
T_SWAP = 1.00
T_COUNT0, T_COUNT1 = 1.25, 2.40
T_WAVE = 2.40
T_CHIP0 = 2.20
T_PUSH0 = 3.50
T_GLITCH0, T_GLITCH1 = 4.45, 4.75
T_TEXT_CUT = 4.50
COUNT_TO = 350000

# device pop starts: device 0 first, the rest staggered so every pop ends by 2.40
_prng = np.random.default_rng(71)
POP_T = T_POP0 + (T_POP1 - POP_DUR - T_POP0) * _prng.random(N_DEV)
POP_T[0] = T_POP0


def sphere_xyz(phi, lam, t):
    """Screen offsets (X, Y) from G and depth Z in [-1, 1] at time t (spin + tilt)."""
    lp = lam + SPIN * t
    cphi = np.cos(phi)
    X = R * cphi * np.sin(lp)
    Y = R * (np.sin(phi) * math.cos(BETA) - cphi * np.cos(lp) * math.sin(BETA))
    Z = np.sin(phi) * math.sin(BETA) + cphi * np.cos(lp) * math.cos(BETA)
    return X, Y, Z


def dev_pos(i, t):
    X, Y, Z = sphere_xyz(DEV_PHI[i], DEV_LAM[i], t)
    return np.array([G[0] + X, G[1] + Y, 0.0]), Z


# --------------------------------------------------------------------------
# Jobs: (t_send, device, t_arrive, t_compute_end, t_back_end)
# --------------------------------------------------------------------------
def _schedule():
    rng = np.random.default_rng(17)
    jobs = [(1.00, 0, 1.90, 2.20, 3.00)]       # job 0: slowed so its labels can be read
    recent = [0]
    n = int(round((T_JOBS1 - T_JOBS0) / JOB_DT)) + 1
    for j in range(1, n):
        t = T_JOBS0 + j * JOB_DT
        _, _, Z = sphere_xyz(DEV_PHI, DEV_LAM, t)
        _, _, Z1 = sphere_xyz(DEV_PHI, DEV_LAM, t + 1.1)
        ok = (Z > 0.35) & (Z1 > 0.15) & (POP_T + POP_DUR <= t)
        ok[recent[-10:]] = False
        cand = np.nonzero(ok)[0]
        d = int(cand[rng.integers(0, len(cand))])
        recent.append(d)
        jobs.append((t, d, t + 0.40, t + 0.70, t + 1.10))
    return jobs


JOBS = _schedule()
ARC_FADE = 0.15

# arc template: ArcBetweenPoints(0 -> 1, angle 0.5) as a circle (complex plane)
ARC_ANGLE = 0.5
_ARC_R = 0.5 / math.sin(ARC_ANGLE / 2)
_ARC_C = complex(0.5, 0.5 / math.tan(ARC_ANGLE / 2))
_ARC_A0 = math.atan2(-_ARC_C.imag, -0.5)


def arc_template_point(s):
    """Point on the template arc (0 -> 1) at fraction s (vectorised), as complex."""
    a = _ARC_A0 + ARC_ANGLE * np.asarray(s, float)
    return _ARC_C + _ARC_R * np.exp(1j * a)


def arc_template_pts(s0: float, s1: float, n: int = 4) -> np.ndarray:
    """Cubic bezier points (complex, (4n,)) of the template arc between fractions s0 < s1."""
    a0 = _ARC_A0 + ARC_ANGLE * s0
    a1 = _ARC_A0 + ARC_ANGLE * s1
    da = (a1 - a0) / n
    h = 4 / 3 * math.tan(da / 4) * _ARC_R
    out = []
    for k in range(n):
        u, v = a0 + k * da, a0 + (k + 1) * da
        p0 = _ARC_C + _ARC_R * complex(math.cos(u), math.sin(u))
        p3 = _ARC_C + _ARC_R * complex(math.cos(v), math.sin(v))
        p1 = p0 + h * complex(-math.sin(u), math.cos(u))
        p2 = p3 - h * complex(-math.sin(v), math.cos(v))
        out += [p0, p1, p2, p3]
    return np.array(out)


def to_screen(A: np.ndarray, B: np.ndarray, w) -> np.ndarray:
    """Map template complex points w onto the chord A -> B: (n,3) screen points."""
    za, zb = complex(A[0], A[1]), complex(B[0], B[1])
    z = za + (zb - za) * np.asarray(w)
    return np.stack([z.real, z.imag, np.zeros(z.shape)], axis=-1)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def hold_floor(scene, t: float) -> None:
    """Hold until the last frame at or before t (exact at 60 fps, rounds down at 30)."""
    n = int(math.floor(t * config.frame_rate + 1e-6)) - scene._frames
    if n > 0:
        scene.wait((n - 0.5) / config.frame_rate)


def seg_points(a: np.ndarray, b: np.ndarray, min_len: float = 0.004) -> np.ndarray:
    """(n,3),(n,3) -> cubic points of n separate straight subpaths a->b (kept open, so round
    caps draw a dot even when a == b)."""
    d = b - a
    L = np.hypot(d[:, 0], d[:, 1])
    short = L < min_len
    if np.any(short):
        b = b.copy()
        b[short] = a[short] + np.array([min_len, 0.0, 0.0])
        d = b - a
    return np.stack([a, a + d / 3, a + 2 * d / 3, b], axis=1).reshape(-1, 3)


def streak_mob(color, width: float) -> VMobject:
    m = VMobject(stroke_color=color, stroke_width=width, fill_opacity=0)
    m.cap_style = CapStyleType.ROUND
    m.points = np.zeros((0, 3))
    return m


EMPTY = np.zeros((0, 3))


class Field(VGroup):
    """N same-shape cells drawn as a few multi-subpath VMobjects, one per quantised RGBA
    (a pool; unused members get no points). Off-frame and transparent cells are skipped."""

    def __init__(self, unit_pts: np.ndarray, pool: int = 96):
        super().__init__(*[VMobject(stroke_width=0, fill_opacity=0) for _ in range(pool)])
        for m in self.submobjects:
            m.points = EMPTY.copy()
        self.unit = np.asarray(unit_pts, float)
        self.max_used = 0

    def clear(self) -> None:
        for m in self.submobjects:
            if len(m.points):
                m.points = EMPTY.copy()

    def draw(self, centers: np.ndarray, sizes, rgba: np.ndarray, qc: int = 24, qa: int = 32) -> None:
        sizes = np.broadcast_to(np.asarray(sizes, float), (len(centers),))
        half = sizes / 2
        vis = ((rgba[:, 3] > 0.004) & (sizes > 1e-4) & (np.abs(centers[:, 0]) < FRAME_W / 2 + half)
               & (np.abs(centers[:, 1]) < FRAME_H / 2 + half))
        idx = np.nonzero(vis)[0]
        used = 0
        pool = len(self.submobjects)
        if len(idx):
            col = rgba[idx]
            while True:
                Q = np.rint(col * [qc, qc, qc, qa]).astype(np.int64)
                keys = ((Q[:, 0] * (qc + 1) + Q[:, 1]) * (qc + 1) + Q[:, 2]) * (qa + 1) + Q[:, 3]
                uniq, inv = np.unique(keys, return_inverse=True)
                if len(uniq) <= pool:
                    break
                qc, qa = max(qc * 2 // 3, 2), max(qa * 2 // 3, 2)
            order = np.argsort(inv, kind="stable")
            counts = np.bincount(inv, minlength=len(uniq))
            starts = np.concatenate([[0], np.cumsum(counts)])
            pts = self.unit[None] * sizes[idx][order, None, None] + centers[idx][order, None, :]
            csum = np.add.reduceat(col[order], starts[:-1], axis=0) / counts[:, None]
            for b in range(len(uniq)):
                m = self.submobjects[b]
                m.points = pts[starts[b]:starts[b + 1]].reshape(-1, 3)
                m.fill_rgbas = csum[b:b + 1].copy()
            used = len(uniq)
        for m in self.submobjects[used:]:
            if len(m.points):
                m.points = EMPTY.copy()
        self.max_used = max(self.max_used, used)


RGB_CYAN = np.array(CYAN.to_rgb())
RGB_MAG = np.array(MAGENTA.to_rgb())
RGB_INK = np.array(INK.to_rgb())
RGB_LIME = np.array(LIME.to_rgb())
RGB_DIM = np.array(DIM.to_rgb())


def pop_curve(p):
    """Device pop: 0 -> 1.3 -> 1 (ease_out_back, overshoot tuned to peak at ~1.3)."""
    p = np.clip(p, 0, 1)
    s = 3.5
    q = p - 1
    return q * q * ((s + 1) * q + s) + 1


def glitch_step(now: float) -> int:
    """Glitch clock: one step per 1/30 s (two 60-fps story frames), so 30-fps previews and
    the 60-fps final both alternate."""
    return story_frame(now - T_GLITCH0) // 2


class B07Planet(BeatScene):
    def beat(self):
        # ---------------- entry: E06 (cluster token + HUD '1,689,600,000+') ----------------
        st = enter(self)
        hud = st.hud
        self.hub = st.token                    # the data center becomes the hub, in place
        self.hub_mems = self.hub.family_members_with_points()
        self.hub_rest = [(m.points.copy(), float(m.stroke_width)) for m in self.hub_mems]
        self.hub_base_rest = (self.hub.base.fill_rgbas.copy(), self.hub.base.stroke_rgbas.copy())

        # ---------------- globe ----------------
        self.outline = Circle(radius=R, stroke_color=CYAN, stroke_width=2, stroke_opacity=0.4,
                              fill_color=CYAN, fill_opacity=0.03).move_to(G)
        self.outline_rest = self.outline.points.copy()
        self.points = Field(Circle(radius=0.5).points.copy(), pool=96)
        self.devices = Field(Square(side_length=1.0).points.copy(), pool=96)

        # ---------------- job pools ----------------
        self.arcs = VGroup(*[VMobject(stroke_color=MAGENTA, stroke_width=1.5, fill_opacity=0) for _ in JOBS])
        for a in self.arcs:
            a.points = EMPTY.copy()
        self.rings = VGroup(*[Circle(radius=1.0, stroke_color=INK, stroke_width=2, fill_opacity=0) for _ in JOBS])
        self.ring_unit = Circle(radius=1.0).points.copy()
        for r in self.rings:
            r.set_stroke(opacity=0)
        self.pk_out = streak_mob(MAGENTA, 9)
        self.pk_back = streak_mob(LIME, 9)

        # ---------------- glitch ghosts (RGB split copies; empty except on split frames) ----------------
        self.stage = [self.outline, self.points, self.arcs, self.devices, self.rings, self.pk_out,
                      self.pk_back, self.hub]
        n_ghost = sum(len(m.family_members_with_points()) if not isinstance(m, Field) else len(m)
                      for m in self.stage) + 8
        self.ghost_c = VGroup(*[VMobject(fill_opacity=0, stroke_width=0) for _ in range(n_ghost)])
        self.ghost_m = VGroup(*[VMobject(fill_opacity=0, stroke_width=0) for _ in range(n_ghost)])
        for g in (*self.ghost_c, *self.ghost_m):
            g.points = EMPTY.copy()

        # ---------------- text ----------------
        track = TextTrack(self)
        gl = lambda now: self._glitch_dx(now) * RIGHT        # text jitters with the stage, then cuts
        track.on(kick("SCALE 6/6 · THE PLANET"), 0.20, T_TEXT_CUT, exit="cut", shift=gl)
        c1 = card([("Home PCs.", "H96", INK), ("Over the internet.", "S56", MAGENTA)])
        c2 = card([("In 2020,", "H96", INK), ("home PCs hit 1 exaflop first.", "S56", INK, {"1 exaflop": CYAN})])
        track.on(c1[0], 0.25, 2.10)
        track.on(c1[1], 0.40, 2.10)
        track.on(c2[0], 2.15, T_TEXT_CUT, exit="cut", shift=gl)
        track.on(c2[1], 2.30, T_TEXT_CUT, exit="cut", shift=gl)
        # job-0 labels ride the packet: right edge at packet.x - 0.15, centre y = packet.y + 0.32
        self.lab_wu = Text("work unit", font=FONT_BODY, font_size=34, color=MAGENTA, weight=SEMIBOLD)
        self.lab_res = Text("result", font=FONT_BODY, font_size=34, color=LIME, weight=SEMIBOLD)
        for lab in (self.lab_wu, self.lab_res):
            lab.move_to(ORIGIN, aligned_edge=RIGHT).shift(-lab.get_center()[1] * UP)
        j0 = JOBS[0]
        track.on(self.lab_wu, j0[0], j0[2], shift=lambda now: self._label_anchor(now, out=True))
        track.on(self.lab_res, j0[3], j0[4], shift=lambda now: self._label_anchor(now, out=False))
        for t in np.arange(j0[0], j0[4] + 1e-9, 1 / 60):           # labels stay inside the safe zone
            for lab, out in ((self.lab_wu, True), (self.lab_res, False)):
                c = self._label_anchor(t, out)
                assert SAFE_LEFT <= c[0] - lab.width and c[0] <= SAFE_RIGHT
        chip_fg = chip("FOLDING@HOME", CYAN, 34)
        chip_bg = RoundedRectangle(width=chip_fg[0].width, height=chip_fg[0].height,
                                   corner_radius=chip_fg[0].height / 2, stroke_width=0,
                                   fill_color=BG, fill_opacity=0.85)
        self.chip = VGroup(chip_bg, chip_fg).move_to([X_C, -1.95, 0])
        assert_safe(self.chip)
        track.on(self.chip, T_CHIP0, T_TEXT_CUT, exit="cut", shift=gl)

        # ---------------- HUD: slot-machine unit swap, count, SLAM ----------------
        hud.swap(T_SWAP, "0", dur=0.25, dist=0.35)
        hud.label(T_SWAP, "VOLUNTEER GPUs · 2020", CYAN, 0.90, fade=0.25)
        hud.roll(T_COUNT0, T_COUNT1, 0, COUNT_TO, ease=ease_out_cubic, log=False)
        hud.slam(T_WAVE, "350,000+")

        # ---------------- z-order: dust, ring, globe, arcs, devices, rings, packets, hub, text, HUD ----------------
        self.remove(self.hub)
        add_behind(self, self.ghost_c, self.ghost_m, self.outline, self.points, self.arcs, self.devices,
                   self.rings, self.pk_out, self.pk_back, self.hub)
        entry_ring(self)
        self.add(track)
        self.remove(hud)
        self.add(hud)
        self._draw(0.0)
        driver = Mobject()
        driver.add_updater(lambda m, dt: self._draw(self.now))
        self.add(driver)

        # ---------------- audio ----------------
        self.cue("hit", gain=0.8, note="match-cut")
        self.cue("whoosh", dt=0.40, dur=0.8, direction="up", note="the widest zoom-out")
        self.cue("glitch", dt=T_SWAP, dur=0.15, note="HUD unit swap")
        self.cue("ticks_burst", dt=T_COUNT0, dur=1.15, rate0=40, rate1=8, note="odometer decelerating")
        trng = np.random.default_rng(23)
        for t_send, d, t_arr, t_cmp, t_back in JOBS:
            for t in (t_send, t_back):
                if t <= T_JOBS1 + EPS:
                    self.cue("tick", dt=round(t, 4), gain=0.15, pan=round(float(trng.uniform(-0.6, 0.6)), 3),
                             note="work unit sent" if t == t_send else "result received")
        self.cue("hit", dt=T_WAVE, gain=0.8, note="350,000+ lands, devices flash")
        self.cue("shimmer", dt=T_WAVE, dur=1.0)
        self.cue("riser", dt=T_PUSH0, dur=0.95, note="feels like it should peak higher")
        self.cue("glitch", dt=T_GLITCH0, dur=0.30, note="harsh digital tear")
        self.cue("silence", dt=T_GLITCH1, dur=0.30, note="pattern interrupt into b08")

        # ---------------- run ----------------
        self.hold_until(T_GLITCH1)
        exit_to(self)                          # 4.75 HARD CUT: BG + dust + HUD only (E07)
        print(f"[b07] buckets used (max): points {self.points.max_used}, devices {self.devices.max_used}")

    # ------------------------------------------------------------------
    def _glitch_dx(self, now: float) -> float:
        if not (T_GLITCH0 - EPS <= now < T_GLITCH1 - EPS):
            return 0.0
        k = glitch_step(now)
        return 0.08 * (1 if k % 2 == 0 else -1) * (0.6 + 0.4 * abs(jitter(k, 1.0, seed=71)))

    def _cam(self, now: float):
        """Globe camera: scale about G (FORM 3 -> 1, ease_out_expo; slight push-in under the
        riser) and the glitch x-offset."""
        s = lerp(3.0, 1.0, ease_out_expo(clip01(now / T_FORM)))
        s *= 1.0 + 0.035 * ease_in_quad(ramp(now, T_PUSH0, T_GLITCH0))
        return s, np.array([self._glitch_dx(now), 0.0, 0.0])

    def _to_screen(self, X, Y, s, off):
        return np.stack([G[0] + s * X + off[0], G[1] + s * Y + off[1], np.zeros_like(X)], axis=1)

    def _label_anchor(self, now: float, out: bool) -> np.ndarray:
        """Where job 0's label sits: right edge at packet.x - 0.15, centre y at packet.y + 0.32."""
        p = self._job_packet(JOBS[0], now, out)
        if p is None:
            t_send, d, t_arr, t_cmp, t_back = JOBS[0]
            p = self._job_packet(JOBS[0], t_arr - 1e-6 if out else t_cmp + 1e-6, out)
        return np.array([p[0] - 0.15, p[1] + 0.32, 0.0])

    def _job_packet(self, job, now: float, out: bool):
        """Head position of job's packet at now (no camera offsets), or None if not riding."""
        t_send, d, t_arr, t_cmp, t_back = job
        s, off = self._cam(now)
        if out and t_send - EPS <= now < t_arr:
            q = ease_in_out_sine(clip01((now - t_send) / (t_arr - t_send)))
        elif not out and t_cmp - EPS <= now < t_back:
            q = 1 - ease_in_out_sine(clip01((now - t_cmp) / (t_back - t_cmp)))
        else:
            return None
        hub = G + off
        X, Y, _ = sphere_xyz(DEV_PHI[d], DEV_LAM[d], now)
        dev = np.array([G[0] + s * X, G[1] + s * Y, 0.0]) + off
        return to_screen(hub, dev, arc_template_point(np.array([q])))[0]

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        s, off = self._cam(now)
        form = ramp(now, 0.0, 0.5 * T_FORM)            # globe fades in over the first 50%
        k = glitch_step(now) if now >= T_GLITCH0 - EPS else -1
        drop = 0.3 if (k >= 0 and jitter(k, 0.5, seed=72) + 0.5 < 0.30) else 1.0

        # ---- globe outline
        self.outline.points = (self.outline_rest - G) * s + G + off
        self.outline.set_stroke(CYAN, width=2, opacity=0.4 * form * drop)
        self.outline.set_fill(CYAN, opacity=0.03 * form * drop)

        # ---- 900 points
        X, Y, Z = sphere_xyz(PT_PHI, PT_LAM, now)
        C = self._to_screen(X, Y, s, off)
        z01 = np.clip(Z, 0, 1)
        rgb = RGB_DIM[None] * (1 - z01[:, None]) + RGB_INK[None] * z01[:, None]
        a = np.where(Z > -0.05, 0.12 + 0.7 * z01, 0.0) * form * drop
        self.points.draw(C, 2 * PT_R * s, np.concatenate([rgb, a[:, None]], axis=1))

        # ---- devices
        X, Y, Z = sphere_xyz(DEV_PHI, DEV_LAM, now)
        D = self._to_screen(X, Y, s, off)
        limb = np.clip(Z / 0.08, 0, 1)
        pop = np.where(now >= POP_T - EPS, pop_curve((now - POP_T) / POP_DUR), 0.0)
        # light wave: delayed 0.30 * d / R from the hub
        dist = np.hypot(D[:, 0] - G[0] - off[0], D[:, 1] - G[1] - off[1])
        t_w = T_WAVE + 0.30 * dist / R
        waved = now >= t_w - EPS
        w_fl = np.where(waved, 1.0 - np.clip((now - t_w) / 0.28, 0, 1), 0.0) ** 1.5
        base_a = np.where(waved, 1.0, 0.25 + 0.75 * np.clip(Z, 0, 1))
        # job compute flash on arrival
        j_fl = np.zeros(N_DEV)
        for t_send, d, t_arr, t_cmp, t_back in JOBS:
            if t_arr - EPS <= now < t_arr + 0.30:
                j_fl[d] = max(j_fl[d], 1.0 - (now - t_arr) / 0.30)
        ink = np.maximum(w_fl, j_fl)
        rgb = RGB_CYAN[None] * (1 - ink[:, None]) + RGB_INK[None] * ink[:, None]
        a = np.maximum(base_a, ink) * limb * (Z > 0) * drop
        size = DEV_SQ * s * pop * (1 + 0.5 * w_fl + 0.4 * j_fl)
        self.devices.draw(D, size, np.concatenate([rgb, a[:, None]], axis=1))

        # ---- jobs: arcs, packets, rings
        hub = G + off
        outs, backs = [], []
        hub_pulse = 0.0
        for j, (t_send, d, t_arr, t_cmp, t_back) in enumerate(JOBS):
            arc, ring = self.arcs[j], self.rings[j]
            dev = D[d]
            # arc: draws on with the outbound packet, stays, fades 0.15 s after the result lands
            if t_send - EPS <= now < t_back + ARC_FADE and (k < 0 or now < T_GLITCH1):
                if now < t_arr:
                    q = ease_in_out_sine(clip01((now - t_send) / (t_arr - t_send)))
                else:
                    q = 1.0
                fa = 1.0 - clip01((now - t_back) / ARC_FADE) if now >= t_back else 1.0
                if q > 1e-3:
                    arc.points = to_screen(hub, dev, arc_template_pts(0.0, q))
                    arc.set_stroke(MAGENTA, width=1.5, opacity=0.4 * fa * drop)
                else:
                    arc.points = EMPTY.copy()
            else:
                arc.points = EMPTY.copy()
            # packets (streak back along the arc)
            for out, lst in ((True, outs), (False, backs)):
                if out and t_send - EPS <= now < t_arr:
                    q = ease_in_out_sine(clip01((now - t_send) / (t_arr - t_send)))
                    qt = ease_in_out_sine(clip01((now - 0.035 - t_send) / (t_arr - t_send)))
                elif not out and t_cmp - EPS <= now < t_back:
                    q = 1 - ease_in_out_sine(clip01((now - t_cmp) / (t_back - t_cmp)))
                    qt = 1 - ease_in_out_sine(clip01((now - 0.035 - t_cmp) / (t_back - t_cmp)))
                else:
                    continue
                p = to_screen(hub, dev, arc_template_point(np.array([qt, q])))
                lst.append(p)
            # hub pulse when a result lands
            if t_back - EPS <= now < t_back + 0.20:
                hub_pulse = max(hub_pulse, 1.0 - (now - t_back) / 0.20)
            # compute ring at the device
            if t_arr - EPS <= now < t_arr + 0.30:
                p = (now - t_arr) / 0.30
                r = lerp(0.05, 0.18, ease_out_cubic(p)) * s
                ring.points = self.ring_unit * r + dev
                ring.set_stroke(INK, width=2, opacity=0.9 * (1 - p) * drop)
            else:
                ring.set_stroke(opacity=0)
        for mob, lst in ((self.pk_out, outs), (self.pk_back, backs)):
            if lst:
                P = np.stack(lst)
                mob.points = seg_points(P[:, 0], P[:, 1])
                mob.set_stroke(opacity=drop)
            else:
                mob.points = EMPTY.copy()

        # ---- hub: the cluster token shrinks 1.2 -> 0.40 (ease_out_expo); stroke pulses on results
        hs = lerp(1.0, HUB_SIZE / TOKEN_SIZE, ease_out_expo(clip01(now / T_FORM)))
        for m, (p, sw) in zip(self.hub_mems, self.hub_rest):
            m.points = (p - AC) * hs + AC + off
        b = self.hub.base
        fr, sr = self.hub_base_rest
        b.stroke_width = lerp(TOKEN_STROKE, 2.5, (1.0 - hs) / (1.0 - HUB_SIZE / TOKEN_SIZE)) + 2.0 * hub_pulse
        b.stroke_rgbas = np.array([[*(RGB_CYAN * (1 - hub_pulse) + RGB_INK * hub_pulse), sr[0, 3] * drop]])
        b.fill_rgbas = np.array([[*RGB_CYAN, lerp(fr[0, 3], 0.45, hub_pulse) * drop]])
        if drop != 1.0 or getattr(self, "_hub_drop", 1.0) != drop:
            for m in self.hub.glyph.family_members_with_points():
                m.fill_rgbas = m.fill_rgbas * 0 + rgba(CYAN, 0.65 * drop)
            self._hub_drop = drop

        # ---- glitch ghosts (CYAN / MAGENTA copies at -/+ 0.05)
        self._draw_ghosts(now, k)

    def _draw_ghosts(self, now: float, k: int) -> None:
        split = k >= 0 and now < T_GLITCH1 - EPS and (jitter(k, 0.5, seed=73) + 0.5) < 0.75
        if not split:
            if getattr(self, "_ghosts_on", True):
                for g in (*self.ghost_c, *self.ghost_m):
                    if len(g.points):
                        g.points = EMPTY.copy()
                self._ghosts_on = False
            return
        self._ghosts_on = True
        srcs = []
        for m in self.stage:
            srcs += [x for x in m.family_members_with_points() if len(x.points)]
        for pool, dx, tint in ((self.ghost_c, -0.05, RGB_CYAN), (self.ghost_m, 0.05, RGB_MAG)):
            for g, src in zip(pool, srcs):
                g.points = src.points + np.array([dx, 0.0, 0.0])
                fr = src.fill_rgbas.copy()
                sr = src.stroke_rgbas.copy()
                fr[:, :3] = tint
                sr[:, :3] = tint
                fr[:, 3] *= 0.5
                sr[:, 3] *= 0.5
                g.fill_rgbas, g.stroke_rgbas = fr, sr
                g.stroke_width = src.stroke_width
                g.cap_style = src.cap_style
            for g in pool[len(srcs):]:
                if len(g.points):
                    g.points = EMPTY.copy()

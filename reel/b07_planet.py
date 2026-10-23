"""b07 - SCALE 6/6 - THE PLANET: volunteer home PCs over the internet (Folding@home, 2020).

The cluster token (E06) shrinks into the hub (project server) at the centre of a spinning
globe (R 2.85, bright points, an atmospheric CYAN rim) that settles in from 3x (the widest
zoom-out). 'Home PCs. / 350,000+ GPUs, online.' POPs on the cut, so the planet's number sits
in the SAME unit as b06's '100,000+ GPUs' headline. Volunteer PCs pop up across the front side;
from 1.00 a work unit leaves the hub every 0.10 s (MAGENTA arc + packet), the device computes
(INK flash + ring) and a LIME result rides back. Job 0 is slowed and drawn bolder so its
'work unit' / 'result' labels (Inter Bold 44) tell the request/response story.

HUD: the big counter does not count here: from 1.00 '1,689,600,000+ / CALCULATIONS AT ONCE'
dims to 0.5 (frozen, never falls to 0, never counts up to a smaller number). 2.40 SLAM + a light
wave (CYAN wavefront ring + point band, devices flash INK as it passes, rim flash) across every
visible device: the HUD relabels to the number the card already showed, '350,000+ /
VOLUNTEER GPUs · 2020' (E07), and the jobs speed up (every 0.04 s) so the planet buzzes.
1.95 hard swap to '2020: they beat / the top 500 combined.'; 2.60 the kicker becomes the
source, 'FOLDING@HOME'. 4.45 glitch-out (x-jolts, RGB-split ghosts, opacity drops, slice
displacement + tear bars), 4.75 hard cut to BG + HUD (E07).

Everything that moves is a pure function of self.now (one driver). The 900 globe points and
300 devices are motif.Fields (fixed quantisation: temporally stable colours).
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# World layout
# --------------------------------------------------------------------------
G = AC.copy()                                  # globe centre
R = 2.85                                       # y[-2.40, 3.30]: clear of the HUD digits and the cards
BETA = 0.35                                    # tilt
SPIN = 0.30                                    # rad/s
N_PTS = 900
_k = np.arange(N_PTS)
PT_PHI = np.arcsin(-1 + 2 * (_k + 0.5) / N_PTS)
PT_LAM = 2.39996 * _k
PT_R = 0.03
PT_BASE = 0.25                                 # point opacity: PT_BASE + 0.65 z
RIM = ((0.04, 0.25), (0.08, 0.12), (0.14, 0.06))   # atmospheric rim: (R + dr, CYAN opacity)

N_DEV = 300
_drng = np.random.default_rng(7)
_sinmax = math.sin(math.radians(65))
DEV_PHI = np.arcsin(_drng.uniform(-_sinmax, _sinmax, N_DEV))
DEV_LAM = _drng.uniform(0, TAU, N_DEV)
DEV_PHI[0], DEV_LAM[0] = math.radians(35), 0.52            # job 0's device: upper right at t = 1.0
DEV_SQ = 0.085
HUB_SIZE = 0.40

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
T_FORM = 0.80
T_POP0, T_POP1 = 0.80, 2.40
POP_DUR = 0.20
T_JOBS0, T_JOBS1 = 1.00, 4.30
JOB_DT0, JOB_DT1 = 0.10, 0.04                  # before / after the light wave (the planet buzzes)
T_DIM0, T_DIM1 = 1.00, 1.30                    # the frozen HUD dims to 0.5: 'not counting here'
HUD_DIM = 0.5
T_CARD2 = 1.95                                 # hard card swap
T_WAVE = 2.40
T_SRC = 2.60                                   # the kicker becomes the source tag
T_PUSH0 = 3.50
T_GLITCH0, T_GLITCH1 = 4.45, 4.75
T_TEXT_CUT = 4.50
NEW_NUM, NEW_LAB = "350,000+", "VOLUNTEER GPUs · 2020"
WAVE_DUR = 0.30                                # light wave reaches the rim (d = R) after 0.30 s
N_TEAR = 9                                     # glitch-out tear bars

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


# --------------------------------------------------------------------------
# Jobs: (t_send, device, t_arrive, t_compute_end, t_back_end)
# --------------------------------------------------------------------------
def _schedule():
    rng = np.random.default_rng(17)
    jobs = [(1.00, 0, 1.90, 2.20, 3.00)]       # job 0: slowed so its labels can be read
    recent = [0]
    ts = [T_JOBS0 + JOB_DT0 * i for i in range(1, 100) if T_JOBS0 + JOB_DT0 * i < T_WAVE - 1e-6]
    ts += [T_WAVE + JOB_DT1 * i for i in range(0, 1000) if T_WAVE + JOB_DT1 * i <= T_JOBS1 + 1e-6]
    for t in ts:
        t = round(float(t), 4)
        _, _, Z = sphere_xyz(DEV_PHI, DEV_LAM, t)
        _, _, Z1 = sphere_xyz(DEV_PHI, DEV_LAM, t + 1.1)
        ok = (Z > 0.35) & (Z1 > 0.15) & (POP_T + POP_DUR <= t)
        ok[recent[-30:]] = False               # never two jobs on one device at once
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
        self.outline = Circle(radius=R, stroke_color=CYAN, stroke_width=3, stroke_opacity=0.8,
                              fill_color=CYAN, fill_opacity=0.04).move_to(G)
        self.outline_rest = self.outline.points.copy()
        self.rim = VGroup(*[Circle(radius=R + dr, stroke_color=CYAN, stroke_width=4, stroke_opacity=op,
                                   fill_opacity=0).move_to(G) for dr, op in RIM])
        self.rim_rest = [c.points.copy() for c in self.rim]
        self.points = Field(Circle(radius=0.5).points.copy(), pool=128, fixed=True)
        self.devices = Field(Square(side_length=1.0).points.copy(), pool=128, fixed=True)

        # ---------------- job pools ----------------
        self.arcs = VGroup(*[VMobject(stroke_color=MAGENTA, stroke_width=2.5, fill_opacity=0) for _ in JOBS])
        for a in self.arcs:
            a.points = EMPTY.copy()
        self.rings = VGroup(*[Circle(radius=1.0, stroke_color=INK, stroke_width=2.5, fill_opacity=0) for _ in JOBS])
        self.ring_unit = Circle(radius=1.0).points.copy()
        for r in self.rings:
            r.set_stroke(opacity=0)
        self.pk_out = streak_mob(MAGENTA, 14)
        self.pk_back = streak_mob(LIME, 14)
        self.pk0 = streak_mob(MAGENTA, 22)       # job 0's packet (the labelled one)
        self.pk0_core = streak_mob(INK, 8)

        # ---------------- light wave front (2.40): radius R*(t-2.40)/0.30, the device-flash delay ----------------
        self.wave = ShockRing(self, T_WAVE, G, 0.02, R, 8.0, 0.9, WAVE_DUR, CYAN, ease=linear,
                              w1=2.0, op1=0.35)

        # ---------------- glitch-out tear bars (empty outside 4.45-4.75) ----------------
        self.sq_unit = Square(side_length=1.0).points.copy()
        self.tears = VGroup(*[VMobject(stroke_width=0, fill_color=CYAN, fill_opacity=0) for _ in range(N_TEAR)])
        for t in self.tears:
            t.points = EMPTY.copy()

        # ---------------- glitch ghosts (RGB split copies; empty except on split frames) ----------------
        self.stage = [self.rim, self.outline, self.points, self.arcs, self.devices, self.rings, self.pk_out,
                      self.pk_back, self.pk0, self.pk0_core, self.hub]
        n_ghost = sum(len(m.family_members_with_points()) if not isinstance(m, Field) else len(m)
                      for m in self.stage) + 8
        self.ghost_c = VGroup(*[VMobject(fill_opacity=0, stroke_width=0) for _ in range(n_ghost)])
        self.ghost_m = VGroup(*[VMobject(fill_opacity=0, stroke_width=0) for _ in range(n_ghost)])
        for g in (*self.ghost_c, *self.ghost_m):
            g.points = EMPTY.copy()

        # ---------------- text ----------------
        track = TextTrack(self)
        gl = lambda now: self._glitch_dx(now) * RIGHT        # text jitters with the stage, then cuts
        # kicker + headline arrive on the cut (frame 0 stays the exact E06); card swaps are hard
        # (cut + POP on one frame, never a crossfade)
        track.on(kick("SCALE 6/6 · THE PLANET"), 0.0, T_SRC, exit="cut")
        track.on(kick("FOLDING@HOME", CYAN), T_SRC, T_TEXT_CUT, enter="pop", exit="cut", shift=gl)
        c1 = card([("Home PCs.", "H96", INK), ("350,000+ GPUs, online.", "S56", INK, {"350,000+ GPUs": CYAN})])
        c2 = card([("2020: they beat", "H96", INK), ("the top 500 combined.", "S56", INK,
                                                     {"top 500 combined.": CYAN})])
        track.on(c1[0], 1 / 60, T_CARD2, enter="pop", exit="cut")
        track.on(c1[1], 0.12, T_CARD2, exit="cut")
        track.on(c2[0], T_CARD2, T_TEXT_CUT, enter="pop", exit="cut", shift=gl)
        track.on(c2[1], T_CARD2, T_TEXT_CUT, enter="pop", exit="cut", shift=gl)
        assert G[1] + R < min(c1.get_bottom()[1], c2.get_bottom()[1]) - 0.3
        # job-0 labels ride the packet: right edge at packet.x - 0.15, centre y = packet.y + 0.36
        self.lab_wu = Text("work unit", font=FONT_BODY, font_size=44, color=MAGENTA, weight=BOLD)
        self.lab_res = Text("result", font=FONT_BODY, font_size=44, color=LIME, weight=BOLD)
        for lab in (self.lab_wu, self.lab_res):
            lab.move_to(ORIGIN, aligned_edge=RIGHT).shift(-lab.get_center()[1] * UP)
            # BG knockout halo: other jobs' packets and arcs cross the labels' path
            lab.set_stroke(BG, width=9, opacity=0.92, background=True)
        j0 = JOBS[0]
        track.on(self.lab_wu, j0[0], j0[2], shift=lambda now: self._label_anchor(now, out=True))
        track.on(self.lab_res, j0[3], j0[4], shift=lambda now: self._label_anchor(now, out=False))
        for t in np.arange(j0[0], j0[4] + 1e-9, 1 / 60):           # labels stay inside the safe zone
            for lab, out in ((self.lab_wu, True), (self.lab_res, False)):
                c = self._label_anchor(t, out)
                assert SAFE_LEFT <= c[0] - lab.width and c[0] <= SAFE_RIGHT
                assert c[1] + lab.height / 2 < TTZ_BOTTOM and c[1] - lab.height / 2 > -2.40

        # ---------------- HUD: frozen + dimmed, then relabelled on the light wave ----------------
        # (the storyboard's fallback: no fall to '0', no count up to a smaller number; the exit
        # state E07 fixes the HUD at '350,000+ / VOLUNTEER GPUs · 2020', which the card has
        # already shown as GPUs, the unit of b06's '100,000+ GPUs')
        hud.fade_to(T_DIM0, T_DIM1, 1.0, HUD_DIM)
        hud.fade_to(T_WAVE, T_WAVE, 1.0, 1.0)
        hud.slam(T_WAVE, NEW_NUM)                 # start scale capped (1.15); the guard hides the label
        hud.label(T_WAVE, NEW_LAB, CYAN, 0.90)    # ... so the unit flips while hidden, then fades in

        # ---------------- z-order: dust, ring, globe, arcs, devices, rings, packets, hub, text, HUD ----------------
        self.remove(self.hub)
        add_behind(self, self.ghost_c, self.ghost_m, self.rim, self.outline, self.points, self.wave, self.arcs,
                   self.devices, self.rings, self.pk_out, self.pk_back, self.pk0, self.pk0_core, self.hub,
                   self.tears)
        entry_ring(self)
        self.add(track)
        self.remove(hud)
        self.add(hud)
        self._draw(0.0)
        drive(self, self._draw)

        # ---------------- audio ----------------
        self.cue("hit", gain=0.8, note="match-cut")
        self.cue("whoosh", dt=0.40, dur=0.8, direction="up", note="the widest zoom-out")
        trng = np.random.default_rng(23)
        for j, (t_send, d, t_arr, t_cmp, t_back) in enumerate(JOBS):
            fast = t_send >= T_WAVE - EPS
            if fast and j % 2:                 # after the wave: every other job, quieter (a buzz)
                continue
            for t in (t_send, t_back):
                if t <= T_JOBS1 + EPS:
                    self.cue("tick", dt=round(t, 4), gain=0.10 if fast else 0.15,
                             pan=round(float(trng.uniform(-0.6, 0.6)), 3),
                             note="work unit sent" if t == t_send else "result received")
        self.cue("hit", dt=T_WAVE, gain=0.8, note="the light wave; HUD relabels to volunteer GPUs")
        self.cue("shimmer", dt=T_WAVE, dur=1.0)
        self.cue("riser", dt=T_PUSH0, dur=0.95, note="feels like it should peak higher")
        self.cue("glitch", dt=T_GLITCH0, dur=0.30, note="harsh digital tear")
        self.cue("silence", dt=T_GLITCH1, dur=0.30, note="pattern interrupt into b08")

        # ---------------- run ----------------
        self.hold_until(T_GLITCH1)
        exit_to(self)                          # 4.75 HARD CUT: BG + dust + HUD only (E07)
        print(f"[b07] jobs {len(JOBS)}; buckets used (max): points {self.points.max_used}, devices "
              f"{self.devices.max_used}; overflow frames: {self.points.overflow + self.devices.overflow}")

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
        """Where job 0's label sits: right edge at packet.x - 0.15, centre y at packet.y + 0.36."""
        p = self._job_packet(JOBS[0], now, out)
        if p is None:
            t_send, d, t_arr, t_cmp, t_back = JOBS[0]
            p = self._job_packet(JOBS[0], t_arr - 1e-6 if out else t_cmp + 1e-6, out)
        return np.array([p[0] - 0.15, p[1] + 0.36, 0.0])

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

        # ---- globe outline + atmospheric rim (the rim flashes when the light wave reaches it)
        rim = 0.0
        if now >= T_WAVE + WAVE_DUR - EPS:
            rim = 1.0 - ease_out_cubic(ramp(now, T_WAVE + WAVE_DUR, T_WAVE + WAVE_DUR + 0.35))
        self.outline.points = (self.outline_rest - G) * s + G + off
        self.outline.set_stroke(CYAN, width=3 + 2.5 * rim, opacity=min(1.0, 0.8 + 0.2 * rim) * form * drop)
        self.outline.set_fill(CYAN, opacity=(0.04 + 0.05 * rim) * form * drop)
        for c, p, (dr, op) in zip(self.rim, self.rim_rest, RIM):
            c.points = (p - G) * s + G + off
            c.set_stroke(CYAN, width=4 + 4 * rim, opacity=min(1.0, op * (1 + 1.5 * rim)) * form * drop)

        # ---- 900 points (+ a bright band riding the light-wave front)
        X, Y, Z = sphere_xyz(PT_PHI, PT_LAM, now)
        C = self._to_screen(X, Y, s, off)
        z01 = np.clip(Z, 0, 1)
        rgb = RGB_DIM[None] * (1 - z01[:, None]) + RGB_INK[None] * z01[:, None]
        a = np.where(Z > -0.05, PT_BASE + 0.65 * z01, 0.0) * form * drop
        if T_WAVE - EPS <= now < T_WAVE + WAVE_DUR + 0.12:
            rho = R * (now - T_WAVE) / WAVE_DUR
            band = np.clip(1.0 - np.abs(np.hypot(X, Y) * s - rho) / 0.40, 0, 1) * (Z > -0.05)
            band *= 1.0 - ramp(now, T_WAVE + WAVE_DUR, T_WAVE + WAVE_DUR + 0.12)
            rgb = rgb * (1 - band[:, None]) + RGB_CYAN[None] * band[:, None]
            a = np.minimum(1.0, a + 0.75 * band)
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
        base_a = np.where(waved, 1.0, 0.35 + 0.65 * np.clip(Z, 0, 1))
        # job compute flash on arrival
        j_fl = np.zeros(N_DEV)
        for t_send, d, t_arr, t_cmp, t_back in JOBS:
            if t_arr - EPS <= now < t_arr + 0.30:
                j_fl[d] = max(j_fl[d], 1.0 - (now - t_arr) / 0.30)
        ink = np.maximum(w_fl, j_fl)
        rgb = RGB_CYAN[None] * (1 - ink[:, None]) + RGB_INK[None] * ink[:, None]
        a = np.maximum(base_a, ink) * limb * (Z > 0) * drop
        size = DEV_SQ * s * pop * (1 + 1.0 * w_fl + 0.4 * j_fl)
        self.devices.draw(D, size, np.concatenate([rgb, a[:, None]], axis=1))

        # ---- jobs: arcs, packets, rings
        hub = G + off
        outs, backs, p0 = [], [], []
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
                    if j == 0:
                        arc.set_stroke(MAGENTA, width=3.5, opacity=0.95 * fa * drop)
                    else:
                        arc.set_stroke(MAGENTA, width=2.5, opacity=0.7 * fa * drop)
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
                if j == 0:
                    p0.append((p, out))
                else:
                    lst.append(p)
            # hub pulse when a result lands
            if t_back - EPS <= now < t_back + 0.20:
                hub_pulse = max(hub_pulse, 1.0 - (now - t_back) / 0.20)
            # compute ring at the device
            if t_arr - EPS <= now < t_arr + 0.30:
                p = (now - t_arr) / 0.30
                big = 1.6 if j == 0 else 1.0
                r = lerp(0.07, 0.24, ease_out_cubic(p)) * s * big
                ring.points = self.ring_unit * r + dev
                ring.set_stroke(INK, width=2.5 * big, opacity=0.9 * (1 - p) * drop)
            else:
                ring.set_stroke(opacity=0)
        for mob, lst in ((self.pk_out, outs), (self.pk_back, backs)):
            if lst:
                P = np.stack(lst)
                mob.points = seg_points(P[:, 0], P[:, 1])
                mob.set_stroke(opacity=drop)
            else:
                mob.points = EMPTY.copy()
        if p0:
            (P, out), = p0
            self.pk0.points = seg_points(P[None, 0], P[None, 1])
            self.pk0.set_stroke(MAGENTA if out else LIME, width=22, opacity=drop)
            self.pk0_core.points = seg_points(P[None, 1], P[None, 1])
            self.pk0_core.set_stroke(INK, width=8, opacity=0.9 * drop)
        else:
            self.pk0.points = EMPTY.copy()
            self.pk0_core.points = EMPTY.copy()

        # ---- hub: the cluster token shrinks 1.7 -> 0.40 (ease_out_expo); stroke pulses on results
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
                m.fill_rgbas = m.fill_rgbas * 0 + rgba(CYAN, TOKEN_GLYPH_FILL * drop)
            self._hub_drop = drop

        # ---- glitch: slice displacement, ghosts (CYAN / MAGENTA copies at -/+ 0.05), tear bars
        if k >= 0 and now < T_GLITCH1 - EPS:
            self._slice_tear(k)
        self._draw_ghosts(now, k)
        self._draw_tears(now, k)

    def _slice_tear(self, k: int) -> None:
        """Harsh digital tear: 2-3 seeded horizontal bands per glitch step; every stage point
        inside a band is displaced sideways (the globe, arcs, packets and hub tear apart)."""
        rng = np.random.default_rng((92, k))
        bands = [(rng.uniform(G[1] - R, G[1] + R), rng.uniform(0.07, 0.30),
                  (1 if rng.random() < 0.5 else -1) * rng.uniform(0.15, 0.45)) for _ in range(3)]
        bands = bands[:2 + int(rng.random() < 0.5)]
        mobs = [*self.rim, self.outline, *self.points, *self.arcs, *self.devices, *self.rings, self.pk_out,
                self.pk_back, self.pk0, self.pk0_core, *self.hub.family_members_with_points()]
        for m in mobs:
            p = m.points
            if not len(p):
                continue
            dx = np.zeros(len(p))
            for yc, hh, d in bands:
                dx += np.where(np.abs(p[:, 1] - yc) < hh, d, 0.0)
            if np.any(dx):
                p = p.copy()
                p[:, 0] += dx
                m.points = p

    def _draw_tears(self, now: float, k: int) -> None:
        """Glitch-out 'digital tear': thin horizontal CYAN / MAGENTA / INK slivers across the
        stage, re-drawn every glitch step (1/30 s), seeded by the step."""
        if k < 0 or now >= T_GLITCH1 - EPS:
            if getattr(self, "_tears_on", True):
                for t in self.tears:
                    t.points = EMPTY.copy()
                self._tears_on = False
            return
        self._tears_on = True
        rng = np.random.default_rng((91, k))
        cols = (RGB_CYAN, RGB_MAG, RGB_INK)
        for i, t in enumerate(self.tears):
            on, y, u_h, u_x, u_w, ci, op = (rng.random(), rng.uniform(G[1] - R + 0.15, G[1] + R - 0.15),
                                            rng.random(), rng.random(), rng.random(),
                                            int(rng.integers(0, 3)), rng.uniform(0.35, 0.75))
            if i < 3:      # drop-out band: a slice of the stage goes missing
                h, x0, w, col, op, p_on = lerp(0.06, 0.26, u_h), lerp(-3.9, -1.2, u_x), lerp(2.5, 6.5, u_w), RGB_BG, 0.94, 0.70
            else:          # bright sliver
                h, x0, w, col, p_on = lerp(0.02, 0.10, u_h), lerp(-3.75, 1.4, u_x), lerp(0.7, 3.6, u_w), cols[ci], 0.65
            if on > p_on:
                t.points = EMPTY.copy()
                continue
            x1 = min(x0 + w, 3.20)
            t.points = self.sq_unit * np.array([x1 - x0, h, 1.0]) + np.array([(x0 + x1) / 2, y, 0.0])
            t.fill_rgbas = np.array([[*col, op]])

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

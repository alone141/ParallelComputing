"""b03 - SCALE 2/6 - ONE CPU: 16 cores, each running its own thread, one shared memory.

The core token (E02) is core T6 of a 16-core CPU: the camera pull-back (EXPAND, with the
cut's INK charge) ends at 0.50 and the stage keeps easing back under a slow drift. '16 cores.
/ Each runs its own thread.' is read on the calm grid; at 0.80 each core starts its own thread
on its own clock, visibly out of phase (the opposite of b02's lockstep): FETCH (a MAGENTA
packet with a comet trail leaves the shared-memory bar, which flashes where it leaves),
COMPUTE (INK outline kick + pop, the core's progress fill rises at the core's own rate while
its 16 lanes light), DONE (the full core flashes LIME), WRITE BACK (a LIME packet returns; the
bar flashes on arrival). At 2.77 the thread line has lifted off and '× 16 lanes each' pops; it
is read before the power-on at 3.20: every core flashes INK -> CYAN, all 256 lanes light, a
CYAN shock ring runs out from the grid centre and the cores pop as it passes, the memory bar
flashes, HUD 16 -> 256 + LIME x16; traffic doubles and the payoff holds 1.6 s.
Hierarchy in the dense powered stretch (headline > grid action > labels): the T labels step
back to ~0.5 opacity once they have done their job, the progress level's INK edge fades out
under each label (no strike-through) and the LIME done flash is softer (0.60) so the lit
lanes still read. At 4.80 traffic drains, the CPU package outline draws on and the chip
collapses (cut_times) into handoff_token('cpu') (E03). Kicker + headline stay fully visible
through the collapse and settle and are hard-cut by the canonical exit at 5.45 (like b01).

Everything that moves is a pure function of self.now (one driver updater + the stage drift).
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# World layout: a 5.6-wide grid of 1.10 cores (the storyboard's 0.95 / 1.20 pitch used
# half the frame width). Vertical budget: grid top 3.31 (under the text zone), memory bar
# bottom -2.07, i.e. 0.45 clear of the HUD number's top (-2.52).
# --------------------------------------------------------------------------
CORE_XS = (-2.60, -1.10, 0.40, 1.90)
CORE_YS = (2.76, 1.52, 0.28, -0.96)
CORE_U = 1.10
N_CORES = 16
CORE_C = np.array([[CORE_XS[i % 4], CORE_YS[i // 4], 0.0] for i in range(N_CORES)])
COL_X = np.array([CORE_XS[i % 4] for i in range(N_CORES)])
HERO = 5                                    # T6 at (-1.10, 1.52)
GRID_C = np.array([X_C, (CORE_YS[0] + CORE_YS[-1]) / 2, 0.0])
CORE_BOTTOM = CORE_YS[-1] - CORE_U / 2      # -1.51
MEM_W, MEM_H = 6.6, 0.44
MEM_C = np.array([X_C, -1.85, 0.0])
MEM_TOP = MEM_C[1] + MEM_H / 2              # -1.63
PKT = 0.22                                  # packet (rounded square)
PKT_R = 0.05
TRAIL = 0.40                                # comet trail length
# Traffic rides BUS LANES in the gutters between the core columns (on black, not over the
# cores): a fetch climbs the gutter left of its core and turns in, a result leaves to the right
# and drops down that gutter. Every gutter carries MAGENTA up and LIME down.
GUT = 0.75                                  # gutter centre offset from a column centre
BUS_X = tuple(sorted({round(x + d, 6) for x in CORE_XS for d in (-GUT, GUT)}))   # 5 lanes
PORT_IN = 0.22                              # a packet ends / starts this far inside its core
LAB_DY = 0.235                              # T label centre above the core centre
FILL_IN = 0.07                              # progress fill inset from the core edge
FILL_W = CORE_U - 2 * FILL_IN
FILL_R = TOKEN_CORNER * CORE_U - FILL_IN

# --------------------------------------------------------------------------
# Timeline (beat-local seconds). The 5.5 s slot is filled by SEQUENCING, not by slowing: the
# pull-back, thread phases, power-on wave, drain and collapse keep their durations; the extra
# second sits BETWEEN the ideas:
#   0.00-0.50  pull-back; '16 cores.' pops on story frame 1, 'Each runs its own thread.' 0.10-0.35
#   0.50-0.80  calm grid (stage drift)  -> the thread line has been read 0.45 s when the threads start
#   0.80-3.20  16 out-of-phase threads (every core busy by 1.30)
#   2.72-2.77  the thread line lifts off; '× 16 lanes each' pops at 2.77 (read 0.43 s first)
#   3.20       POWER-ON: 256 lanes light, ring, HUD 16 -> 256 + LIME x16 (1.6 s)
#   3.20-4.80  PAYOFF HOLD (1.6 s) under doubled traffic
#   4.80-4.93  drain + package outline; 4.93-5.45 collapse + settle; 5.45-5.50 static E03
# --------------------------------------------------------------------------
T_EXPAND = EXPAND_TIME                      # 0.50 pull-back
T_THREADS = 0.80                            # thread clocks start (offsets 0..0.49 on top)
T_LANES = 166 / 60                          # 2.767: '× 16 lanes each' pops (the thread line lifts off 2.717 -> 2.767)
T_POWER = 3.20                              # the power-on: '× 16 lanes each' has been read 0.43 s
T_COL, T_SET, T_EXIT = cut_times(5.5)       # 4.933 / 5.333 / 5.45
T_DRAIN = T_COL - 8 / 60                    # 4.80: no new fetches; in-flight work fades by T_COL
T_PKG = T_DRAIN                             # package outline Create 4.80 -> T_COL
MULT_HOLD = T_DRAIN - T_POWER               # 1.60: the LIME x16 pill holds the whole payoff, leaves with the drain
T_TEXT_OUT = T_EXIT                         # kicker + headline hold to the exit, then hard-cut
# camera life: after the pull-back the stage keeps easing back, 1.0 -> 0.98 about AC by the
# collapse (ease_in_out_sine: it leaves the EXPAND and meets the COLLAPSE at rest)
DRIFT_S1 = 0.98

# thread cycle, as fractions of a period: fetch | compute | write back | idle
S_FETCH, S_DONE, S_WB = 0.30, 0.68, 0.96
DONE_FLASH = 0.16                           # LIME completion flash (s)
WRITE_FLASH = 0.25                          # arrival flash on the bar (s)
LEAVE_FLASH = 0.18                          # departure flash on the bar (s)

# per-core thread clocks (storyboard: P = 0.85 + 0.35u from default_rng(3); offsets 0.5v so
# every core is busy within half a second of the thread start)
_rng = np.random.default_rng(3)
U = _rng.random(N_CORES)
V = _rng.random(N_CORES)
PER = 0.85 + 0.35 * U
OFF = 0.50 * V
C_POW = (T_POWER - T_THREADS - OFF) / PER   # cycles completed at the power-on (rate doubles after)
MAX_CYC = 24

GLYPH_IDLE = 0.35
GLYPH_ON = 0.90
FILL_REST, FILL_REST_ON = 0.15, 0.25        # unit fill before / after the power-on
PROG_A = 0.55                               # progress fill opacity (CYAN)
# hierarchy after the power-on (headline > grid action > labels): the T labels have said
# 'each runs its own thread' and step back so the doubled traffic + '× 16 lanes each' lead
LAB_POWERED = 0.55                          # T label opacity factor once powered (0.9 -> ~0.5)
LAB_RECEDE = 0.40                           # ... eased in over this long from the power-on
DONE_A, DONE_A_ON = 0.85, 0.60              # LIME completion flash peak before / after the power-on
                                            # (traffic doubles: softer, so the lit lanes still read)
MEM_FILL = 0.10

# power-on light wave: a CYAN ring from the grid centre; each core pops as it passes
RING = dict(r0=0.5, r1=4.5, w0=10.0, op0=0.9, dur=0.45)


def ring_time(d: float) -> float:
    """Seconds after the power-on at which the ring's radius reaches d (ease_out_cubic)."""
    p = clip01((d - RING["r0"]) / (RING["r1"] - RING["r0"]))
    return RING["dur"] * (1.0 - (1.0 - p) ** (1.0 / 3.0))


POP_DELAY = np.array([ring_time(float(np.hypot(*(CORE_C[i, :2] - GRID_C[:2])))) for i in range(N_CORES)])


def cycles_at(t: float) -> np.ndarray:
    """Cumulative thread cycles per core at time t (rate doubles after the power-on);
    negative = not started yet."""
    if t < T_POWER:
        return (t - T_THREADS - OFF) / PER
    return C_POW + 2.0 * (t - T_POWER) / PER


def time_of(i: int, c: float) -> float:
    """Inverse of cycles_at for core i."""
    if c <= C_POW[i]:
        return T_THREADS + OFF[i] + c * PER[i]
    return T_POWER + (c - C_POW[i]) * PER[i] / 2.0


def alive(i: int, n: int) -> bool:
    """Cycle n of core i exists (started before the drain)."""
    return 0 <= n < MAX_CYC and time_of(i, n) < T_DRAIN - EPS


def last_event(i: int, ci: float, s_ev: float):
    """(cycle, time) of the latest phase-s_ev event of core i at or before cycle count ci."""
    n = int(math.floor(ci - s_ev + 1e-9))
    if not alive(i, n):
        return None
    return n, time_of(i, n + s_ev)


def compute_starts() -> list[tuple[float, int]]:
    """(time, core) of every visible COMPUTE start (before the drain), sorted."""
    out = []
    for i in range(N_CORES):
        for n in range(MAX_CYC):
            if not alive(i, n):
                break
            t = time_of(i, n + S_FETCH)
            if t < T_DRAIN - EPS:                 # (later starts are drawn already drained)
                out.append((t, i))
    return sorted(out)


def thread_label(i: int) -> MarkupText:
    """'T{i+1}' in JetBrains Mono BOLD 38, INK op 0.9; tracking -700 (1/1024 pt) so 'T16'
    stays clear of the 1.10 core's rounded corners."""
    m = MarkupText(f'<span letter_spacing="-700">T{i + 1}</span>', font=FONT_MONO, font_size=38,
                   color=INK, weight=BOLD)
    return m.set_opacity(0.9)


def fill_pts(cx: float, y0: float, w: float, h: float, r: float, top: float) -> np.ndarray:
    """Progress fill: the bottom part (height h) of a w x top rounded rect whose bottom edge is
    y0; the top corners round in only as the level reaches the rect's own top corners."""
    hw = w / 2
    rb = min(r, hw, h / 2 if h < 2 * r else r)
    rt = r * clip01((h - (top - r)) / r) if h > top - r else 0.0
    rt = min(rt, max(h - rb, 0.0))
    k_b, k_t = 0.5522847498307936 * rb, 0.5522847498307936 * rt
    y1 = y0 + h
    x0, x1 = cx - hw, cx + hw
    P = [
        (x1, y1 - rt), (x1, y1 - rt + k_t), (x1 - rt + k_t, y1), (x1 - rt, y1),        # arc TR
        (x1 - rt, y1), (x1 - rt, y1), (x0 + rt, y1), (x0 + rt, y1),                    # top
        (x0 + rt, y1), (x0 + rt - k_t, y1), (x0, y1 - rt + k_t), (x0, y1 - rt),        # arc TL
        (x0, y1 - rt), (x0, y1 - rt), (x0, y0 + rb), (x0, y0 + rb),                    # left
        (x0, y0 + rb), (x0, y0 + rb - k_b), (x0 + rb - k_b, y0), (x0 + rb, y0),        # arc BL
        (x0 + rb, y0), (x0 + rb, y0), (x1 - rb, y0), (x1 - rb, y0),                    # bottom
        (x1 - rb, y0), (x1 - rb + k_b, y0), (x1, y0 + rb - k_b), (x1, y0 + rb),        # arc BR
        (x1, y0 + rb), (x1, y0 + rb), (x1, y1 - rt), (x1, y1 - rt),                    # right
    ]
    out = np.zeros((32, 3))
    out[:, :2] = P
    return out


class Route:
    """A packet's path: bar port -> up the gutter -> into the core (fetch), or the reverse
    (write back), parametrised by arc length u in [0, L]."""

    def __init__(self, pts):
        self.p = np.array([[x, y, 0.0] for x, y in pts], float)
        seglen = np.hypot(*np.diff(self.p[:, :2], axis=0).T)
        self.cum = np.concatenate([[0.0], np.cumsum(seglen)])
        self.L = float(self.cum[-1])

    def at(self, u: float) -> np.ndarray:
        u = min(max(u, 0.0), self.L)
        j = min(int(np.searchsorted(self.cum, u, side="right")) - 1, len(self.p) - 2)
        t = (u - self.cum[j]) / max(self.cum[j + 1] - self.cum[j], 1e-9)
        return lerp(self.p[j], self.p[j + 1], t)

    def poly(self, u0: float, u1: float) -> np.ndarray:
        """Cubic points of the stretch [u0, u1] (keeps the corner)."""
        inner = [self.p[j] for j in range(1, len(self.p) - 1) if u0 < self.cum[j] < u1]
        return polyline([self.at(u0), *inner, self.at(u1)])


def fetch_route(i: int) -> Route:
    cx, cy = CORE_C[i, 0], CORE_C[i, 1]
    return Route([(cx - GUT, MEM_TOP), (cx - GUT, cy), (cx - CORE_U / 2 + PORT_IN, cy)])


def write_route(i: int) -> Route:
    cx, cy = CORE_C[i, 0], CORE_C[i, 1]
    return Route([(cx + CORE_U / 2 - PORT_IN, cy), (cx + GUT, cy), (cx + GUT, MEM_TOP)])


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
            lab = thread_label(i).move_to(CORE_C[i] + UP * LAB_DY)
            assert_safe(lab)
            assert lab.width < CORE_U - 0.20
            assert lab.get_bottom()[1] > self.cores[i].glyph.get_top()[1] + 0.03
            self.labels.add(lab)
        # the progress level's INK edge fades out just under the T label (through the label it
        # read as a strike-through); measured at rest, relative to the core centre
        self._edge_y1 = min(lab.get_bottom()[1] - CORE_C[i, 1] for i, lab in enumerate(self.labels)) - 0.03
        self._edge_y0 = self._edge_y1 - 0.10
        # progress fills (empty at rest: not part of the EXPAND capture), under the cores so the
        # unit stroke, lanes and label stay on top
        self.pfills = VGroup(*[VMobject(stroke_width=0, fill_color=CYAN, fill_opacity=0) for _ in range(N_CORES)])
        self.plines = VGroup(*[VMobject(stroke_color=INK, stroke_width=3, stroke_opacity=0, fill_opacity=0)
                               for _ in range(N_CORES)])        # the progress level (a bright edge)
        for m in [*self.pfills, *self.plines]:
            m.points = EMPTY.copy()
        self.mem = RoundedRectangle(width=MEM_W, height=MEM_H, corner_radius=0.12, stroke_color=MAGENTA,
                                    stroke_width=3, fill_color=MAGENTA, fill_opacity=MEM_FILL).move_to(MEM_C)
        mem_lab = kick("SHARED MEMORY", MAGENTA)
        b, t = cap_box(mem_lab)
        mem_lab.shift([X_C - mem_lab.get_center()[0], MEM_C[1] - (b + t) / 2, 0])
        assert_safe(mem_lab)
        assert mem_lab.get_top()[1] < MEM_TOP and mem_lab.get_bottom()[1] > MEM_C[1] - MEM_H / 2
        assert self.mem.get_bottom()[1] - hud_number_tmpl("256").get_top()[1] >= 0.44
        self.mem_lab = mem_lab
        # bus lanes: from the bar up each gutter to the top row's centre line
        self.stubs = VGroup(*[Line([x, MEM_TOP, 0], [x, CORE_YS[0], 0], stroke_color=DIM, stroke_width=2,
                                   stroke_opacity=0.8) for x in BUS_X])
        assert min(BUS_X) > STAGE.x0 and max(BUS_X) < STAGE.x1
        assert all(self.mem.get_left()[0] < x < self.mem.get_right()[0] for x in BUS_X)
        self.world = VGroup(self.stubs, self.mem, self.mem_lab, self.pfills, self.plines, self.cores, self.labels)
        assert self.cores.get_top()[1] < TTZ_BOTTOM
        hero = self.cores[HERO]
        # the memory bar/stubs/label come in late (e 0.80 -> 0.97): earlier they would sweep
        # across the HUD number while the camera is still pulling back
        rig = EXPAND(self.world, CORE_C[HERO], CORE_U, 0.0, hero=hero, token="core",
                     fades={self.labels: (0.3, 0.8), self.mem: (0.8, 0.97), self.mem_lab: (0.8, 0.97),
                            self.stubs: (0.8, 0.97)})          # captures the rest layout, draws e = 0

        # rest copies for the thread updater (taken at rest, before any camera move)
        self._core_rest = []
        for i, core in enumerate(self.cores):
            mems = [core.unit] + list(core.glyph.submobjects) + list(self.labels[i].family_members_with_points())
            self._core_rest.append([(m, rig.rest[rig.index[id(m)]].p.copy()) for m in mems])
        self._lab_rest = [(m, rig.rest[rig.index[id(m)]].fr.copy(), True)
                          for m in self.labels.family_members_with_points()]
        self._lab_rest += [(m, rig.rest[rig.index[id(m)]].fr.copy(), False)
                           for m in self.mem_lab.family_members_with_points()]
        self._unit_sw = rig.rest[rig.index[id(self.cores[0].unit)]].sw

        # ---------------- pooled packets, trails, bar flashes (above the cores) ----------------
        self._pkt = rrect_c(ORIGIN, PKT, PKT, PKT_R)
        self._el = Ellipse(width=1.0, height=1.0).points.copy()
        self.packets = VGroup(*[VMobject(stroke_width=0, fill_color=MAGENTA if k < 16 else LIME, fill_opacity=0)
                                for k in range(32)])
        self.trails = VGroup(*[streak_mob(MAGENTA if k < 32 else LIME, 8) for k in range(64)])   # 2 layers / packet
        self._routes = {"f": [fetch_route(i) for i in range(N_CORES)], "w": [write_route(i) for i in range(N_CORES)]}
        self.wflash = VGroup(*[VMobject(stroke_width=0, fill_color=LIME, fill_opacity=0) for _ in range(N_CORES)])
        self.fflash = VGroup(*[VMobject(stroke_width=0, fill_color=MAGENTA, fill_opacity=0) for _ in range(N_CORES)])
        for m in [*self.packets, *self.trails, *self.wflash, *self.fflash]:
            m.points = EMPTY.copy()
        self.fx = VGroup(self.fflash, self.wflash, self.trails, self.packets)

        # ---------------- package outline (drawn on at 3.80) ----------------
        self.pkg_full = RoundedRectangle(width=6.9, height=5.56, corner_radius=0.35, stroke_color=CYAN,
                                         stroke_width=3, fill_color=CYAN, fill_opacity=0).move_to([X_C, 0.62, 0])
        self.pkg = self.pkg_full.copy().set_stroke(opacity=0)

        # ---------------- text ----------------
        track = TextTrack(self)
        # kicker + headline arrive on the cut (frame 0 stays the exact E02: 'in' at 0.0 is
        # invisible on frame 0, the POP starts on story frame 1) and stay FULLY visible through
        # the collapse and settle, hard-cut at T_EXIT by the canonical E03 swap (like b01)
        track.on(kick("SCALE 2/6 · ONE CPU"), 0.0, T_TEXT_OUT, exit="cut")
        c_a = card([("16 cores.", "H96", INK, {"16": CYAN}), ("Each runs its own thread.", "S56", INK)])
        c_b = card([("16 cores.", "H96", INK, {"16": CYAN}), ("× 16 lanes each", "S56", CYAN)])
        track.on(c_a[0], 1 / 60, T_TEXT_OUT, enter="pop", exit="cut")
        # the thread line lifts off T_LANES - 0.05 -> T_LANES so '× 16 lanes each' pops on a
        # clean line, and is read BEFORE the power-on it describes
        track.on(c_a[1], 0.10, T_LANES - 0.05, exit_dur=0.05)
        track.on(c_b[1], T_LANES, T_TEXT_OUT, enter="pop", exit="cut")

        # ---------------- HUD ----------------
        hud.slam(T_POWER, "256")
        hud.mult(T_POWER, "×16", hold=MULT_HOLD)       # the LIME pill holds through the payoff

        # ---------------- z-order: dust, rings, world, fx, package, text, HUD ----------------
        self.remove(st.token)                  # the hero core now draws the token
        add_behind(self, self.world, self.fx, self.pkg)
        # the stage drift (world + traffic + package; never text or HUD), added after the driver
        # below; the power-on ring is placed with the drift's pose at T_POWER
        self.drift = drift(self, VGroup(self.world, self.fx, self.pkg), T_EXPAND, T_COL, s1=DRIFT_S1,
                           dy=0.0, ease=ease_in_out_sine, add=False)
        k, b = self.drift.pose(T_POWER)
        ring = ShockRing(self, T_POWER, GRID_C * k + b, RING["r0"] * k, RING["r1"] * k, RING["w0"],
                         RING["op0"], RING["dur"], CYAN)
        add_behind(self, ring)                 # the power-on ring runs behind the cores
        entry_ring(self)
        self.add(track)
        self.remove(hud)
        self.add(hud)                          # HUD stays on top
        self.collapse = None
        self._draw(0.0)
        drive(self, self._draw)
        self.add(self.drift)                   # after the driver: composes on this frame's drawing

        # ---------------- audio ----------------
        self.cue("hit", gain=0.8, note="match-cut and ring")
        self.cue("whoosh", dt=0.20, dur=0.50, direction="up", note="camera pull-back")
        trng = np.random.default_rng(17)
        for t, i in compute_starts():
            pan = float(np.clip((COL_X[i] - X_C) / 2.25 * 0.55 + trng.uniform(-0.15, 0.15), -0.8, 0.8))
            self.cue("tick", dt=round(t, 4), gain=0.3, freq=int(trng.uniform(2600, 5400)), pan=round(pan, 3),
                     note=f"T{i + 1} compute")
        self.cue("tick", dt=T_LANES, gain=0.4, freq=4200, pan=0, note="'× 16 lanes each' pops")
        self.cue("riser", dt=round(T_POWER - 0.50, 4), dur=0.50, note="into the power-on")
        self.cue("hit", dt=T_POWER, note="256 lanes light")
        self.cue("sub_drop", dt=T_POWER, dur=0.6)
        self.cue("shimmer", dt=T_POWER, dur=1.0)
        self.cue("whoosh", dt=round((T_COL + T_SET) / 2, 4), dur=0.45, direction="down", note="collapse")
        self.cue("tick", dt=round(T_SET, 4), gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_COL)
        # draw the fully drained state first, so no packet / trail / flash hangs over the collapse,
        # put the drift camera back on top of it, and let the collapse start from that pose
        self._draw_threads(T_COL)
        self._draw_package(T_COL)
        self.drift.render_at(T_COL)
        self.drift.release()
        self.collapse = COLLAPSE(self.pkg, self.world, "cpu")
        self.add(self.collapse.glyph)
        hold_floor(self, T_EXIT)
        exit_to(self)                          # canonical E03, static to 5.50

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.collapse is not None:
            self.collapse.render(now, T_COL, T_SET, T_EXIT)
            return
        if now < T_EXPAND + 0.1:               # (from T_EXPAND on: e = 1 exactly, i.e. the rest layout)
            EXPAND(self.world, CORE_C[HERO], CORE_U, ease_out_expo(ramp(now, 0.0, T_EXPAND)),
                   ink=1.0 if flash_frames(now, 1 / 60, 2) else 0.0)
        if now >= T_THREADS - EPS:
            self._draw_threads(now)
        self._draw_package(now)

    def _draw_package(self, now: float) -> None:
        a = ease_out_cubic(ramp(now, T_PKG, T_COL))
        if a <= 0:
            self.pkg.set_stroke(opacity=0)
        elif a >= 1:
            self.pkg.points = self.pkg_full.points.copy()
            self.pkg.set_stroke(CYAN, width=3, opacity=1)
        else:
            self.pkg.pointwise_become_partial(self.pkg_full, 0, a)
            self.pkg.set_stroke(CYAN, width=3, opacity=1)

    def _draw_threads(self, now: float) -> None:
        drain = 1.0 - ramp(now, T_DRAIN, T_COL)               # in-flight work fades 4.80 -> 4.93
        c = cycles_at(now)
        powered = now >= T_POWER - EPS
        ink_flash = powered and flash_frames(now, T_POWER, 2)
        # power-on light wave (pw: 1 on the hit's 2 frames, then cools over 0.35 s)
        pw = flash_decay = 0.0
        if powered:
            pw = 1.0 if ink_flash else 1.0 - ease_out_cubic(ramp(now, T_POWER + 2 / 60, T_POWER + 0.35))
            flash_decay = pw
        # lanes at rest: idle 0.35 before the power-on; after: CYAN 1.0 settling to 0.9
        g_rest = (lerp(1.0, GLYPH_ON, ease_out_cubic(ramp(now, T_POWER + 2 / 60, T_POWER + 0.35)))
                  if powered else GLYPH_IDLE)
        fill_rest = FILL_REST_ON if powered else FILL_REST
        mem_busy = 0.0

        for i in range(N_CORES):
            ci = c[i]
            n = int(math.floor(ci)) if ci >= 0 else -1
            s = ci - n if n >= 0 else -1.0
            live = alive(i, n)
            ctr = CORE_C[i]
            core = self.cores[i]

            # ---- compute: INK outline kick + pop at the start, progress fill at the core's own rate
            comp = live and S_FETCH <= s < S_DONE
            kick_ink = 0.0
            pop = 1.0
            ev = last_event(i, ci, S_FETCH)
            if ev is not None:
                t_s = ev[1]
                kick_ink = 1.0 if flash_frames(now, t_s, 2) else 0.0
                pop = pop_scale(now - t_s, 1.08, 0.20)
            if powered:
                pop *= pop_scale(now - T_POWER - POP_DELAY[i], 1.12, 0.25)
            # ---- unit: fill + stroke (power-on: INK 0.9 -> CYAN 0.6 -> rest 0.25)
            if ink_flash:
                ufill = rgba(INK, 0.9)
            elif powered and pw > 0:
                ufill = rgba(CYAN, lerp(fill_rest, 0.6, pw))
            else:
                ufill = rgba(CYAN, fill_rest)
            core.unit.fill_rgbas = ufill.reshape(1, 4)
            core.unit.stroke_rgbas = rgba(interpolate_color(CYAN, INK, max(kick_ink * drain, 0.85 * pw)),
                                          1.0).reshape(1, 4)
            core.unit.stroke_width = self._unit_sw * (1.0 + 0.6 * max(kick_ink * drain, pw))
            # ---- lanes (all 16 at once: SIMD inside every core)
            if ink_flash:
                lane = rgba(INK, 1.0)
            elif comp:
                lane = rgba(interpolate_color(CYAN, INK, 0.45 if powered else 0.0),
                            lerp(g_rest, 1.0, drain))
            else:
                lane = rgba(CYAN, g_rest)
            for bar in core.glyph.submobjects:
                bar.fill_rgbas = lane.reshape(1, 4).copy()
            # ---- pop (whole core + label, about its centre)
            for m, p in self._core_rest[i]:
                m.points = p.copy() if pop == 1.0 else (p - ctr) * pop + ctr
            # ---- progress fill / LIME completion flash
            pf = self.pfills[i]
            h_full = FILL_W
            y0 = ctr[1] - h_full / 2
            prog, pcol, pa = 0.0, CYAN, 0.0
            if comp:
                prog = (s - S_FETCH) / (S_DONE - S_FETCH)
                pa = PROG_A * drain
            else:
                ev = last_event(i, ci, S_DONE)
                if ev is not None and 0.0 <= now - ev[1] < DONE_FLASH:
                    prog, pcol = 1.0, LIME
                    pa = ((DONE_A_ON if ev[1] >= T_POWER else DONE_A)
                          * (1.0 - ease_in_quad((now - ev[1]) / DONE_FLASH)) * drain)
            pl = self.plines[i]
            if pa > 0.004 and prog > 0.004:
                pts = fill_pts(ctr[0], y0, FILL_W, h_full * prog, FILL_R, h_full)
                pf.points = (pts - ctr) * pop + ctr if pop != 1.0 else pts
                pf.fill_rgbas = rgba(pcol, pa).reshape(1, 4)
                ly = y0 + h_full * prog
                ek = 1.0 - ramp(ly - ctr[1], self._edge_y0, self._edge_y1)
                if comp and prog < 0.97 and ek > 0.0:   # the rising level: a bright INK edge
                    dy = max(FILL_R - min(h_full * prog, h_full * (1.0 - prog)), 0.0)   # inside a corner arc
                    inset = 0.02 + FILL_R - math.sqrt(max(FILL_R ** 2 - dy ** 2, 0.0))
                    a_ = np.array([ctr[0] - FILL_W / 2 + inset, ly, 0.0])
                    b_ = np.array([ctr[0] + FILL_W / 2 - inset, ly, 0.0])
                    pl.points = (seg(a_, b_) - ctr) * pop + ctr
                    pl.stroke_rgbas = rgba(INK, 0.85 * drain * ek).reshape(1, 4)
                elif len(pl.points):
                    pl.points = EMPTY.copy()
            else:
                for m in (pf, pl):
                    if len(m.points):
                        m.points = EMPTY.copy()

            # ---- packets: fetch (s in [0, S_FETCH)), write-back (s in [S_DONE, S_WB))
            for kind, s0, s1, slot, col in (("f", 0.0, S_FETCH, i, MAGENTA), ("w", S_DONE, S_WB, 16 + i, LIME)):
                pk = self.packets[slot]
                t1, t2 = self.trails[2 * slot], self.trails[2 * slot + 1]
                if live and s0 <= s < s1:
                    q = (s - s0) / (s1 - s0)
                    R = self._routes[kind][i]
                    u = ease_in_out_cubic(q) * R.L
                    pos = R.at(u)
                    sc = lerp(0.6, 1.0, ramp(q, 0, 0.12)) if kind == "f" else lerp(1.0, 0.7, ramp(q, 0.88, 1.0))
                    al = (ramp(q, 0.0, 0.08) * (1 - ramp(q, 0.90, 1.0)) if kind == "f"
                          else ramp(q, 0.0, 0.10) * (1 - ramp(q, 0.94, 1.0))) * drain
                    if kind == "f" and q > 0.80:      # shrinks into the core as it is absorbed
                        sc *= lerp(1.0, 0.6, ramp(q, 0.80, 1.0))
                    pk.points = self._pkt * sc + pos
                    pk.fill_rgbas = rgba(col, al).reshape(1, 4)
                    for tm, ln, w, ta in ((t1, TRAIL, 9.0, 0.35), (t2, TRAIL * 0.5, 15.0, 0.35)):
                        u0 = max(u - ln, 0.0)
                        if u - u0 > 0.02:
                            tm.points = R.poly(u0, u)
                            tm.set_stroke(col, width=w * sc, opacity=ta * al)
                        elif len(tm.points):
                            tm.points = EMPTY.copy()
                else:
                    for m in (pk, t1, t2):
                        if len(m.points):
                            m.points = EMPTY.copy()

            # ---- bar flashes: LIME where a result lands, MAGENTA where a fetch leaves
            for fl, s_ev, dur, w_h, op, port in ((self.wflash[i], S_WB, WRITE_FLASH, (0.9, 0.3), 0.9, CORE_C[i, 0] + GUT),
                                                 (self.fflash[i], 0.0, LEAVE_FLASH, (0.6, 0.22), 0.7, CORE_C[i, 0] - GUT)):
                ev = last_event(i, ci, s_ev)
                if ev is not None and 0.0 <= now - ev[1] < dur:
                    k = (now - ev[1]) / dur
                    env = (1.0 - ease_out_quad(k)) * drain
                    sc = lerp(0.6, 1.25, ease_out_cubic(k))
                    fl.points = self._el * np.array([w_h[0] * sc, w_h[1] * sc, 1.0]) + np.array([port, MEM_TOP, 0.0])
                    fl.fill_rgbas = fl.fill_rgbas.copy()
                    fl.fill_rgbas[:, 3] = op * env
                    mem_busy += env
                elif len(fl.points):
                    fl.points = EMPTY.copy()

        # ---- the shared-memory bar: busy with traffic, flashes MAGENTA on the power-on
        mf = MEM_FILL + 0.07 * min(mem_busy, 3.0) * drain
        if powered:
            mf = max(mf, lerp(MEM_FILL, 0.6, flash_decay))
        self.mem.fill_rgbas = rgba(MAGENTA, mf).reshape(1, 4)
        self.mem.stroke_rgbas = rgba(interpolate_color(MAGENTA, INK, 0.7 * (1.0 if ink_flash else 0.0)), 1.0).reshape(1, 4)

        # ---- T labels step back after the power-on; T labels + SHARED MEMORY fade out with the drain
        la = ease_in_out_sine(drain) if now >= T_DRAIN else 1.0
        lt = la * lerp(1.0, LAB_POWERED, ease_out_cubic(ramp(now, T_POWER, T_POWER + LAB_RECEDE)))
        if getattr(self, "_lab_a", (1.0, 1.0)) != (lt, la):
            for m, fr, is_t in self._lab_rest:
                v = fr.copy()
                v[:, 3] *= lt if is_t else la
                m.fill_rgbas = v
            self._lab_a = (lt, la)

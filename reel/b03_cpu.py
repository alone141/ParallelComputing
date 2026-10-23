"""b03 - LEVEL 1/3 - ONE CPU (script v2 segment 'cpu', 9.5 s): that worker is one CPU core; 16
cores run 16 different jobs and share one memory.

Entry E01 (the hook's plain CYAN token, no HUD): the token is core T6 (HERO = 5) of a 4x4 grid
of 16 plain cores (no SIMD lane glyphs). EXPAND pulls back 0.00-0.50 (with the cut's INK charge
and the entry ring); 'That worker? / One CPU core.' POPs on story frame 1 and the counter makes
its first appearance (core_hud: '1' / CORE WORKING AT ONCE, POP at 0.10). While it is read the
other 15 cores ripple in DIM outward from T6 and the SHARED MEMORY bar settles in dim below.

  1.00-2.60  T6 runs ONE job alone, twice (and keeps humming after): FETCH (a MAGENTA packet
             with a comet trail leaves the bar, which flashes where it leaves), COMPUTE (INK
             outline kick + pop, the progress fill rises), DONE (LIME flash), WRITE BACK (a LIME
             packet returns; the bar flashes where it lands). The other 15 stay dark.
  2.70       hard swap: '16 cores. / 16 different jobs.' on a calm stage; T1-T16 fade in dim.
  3.70       POWER-ON: every core flashes INK -> CYAN behind a CYAN shock ring from the grid
             centre, each core pops as the ring passes and then starts its OWN job on its own
             clock (own period, own job lengths, own phase). HUD SLAM 1 -> 16 (plural label) +
             LIME '16-CORE RYZEN 9 9950X' 3.70-5.50. Payoff hold 1.8 s.
  5.50       hard swap: 'They share / one memory.'; traffic eases to a low hum, the bar brightens,
             the T labels step back to 0.5. Calm read 1.0 s.
  6.50       traffic doubles: packets stream both ways, jobs of different lengths finish at their
             own moments (out-of-sync LIME flashes), under the slow stage drift.
  7.90       hard swap: 'Need more?'. 8.10: no new jobs (every job in flight finishes and writes
             back by 8.70), the CPU package outline draws on, labels leave. COLLAPSE 8.933 ->
             handoff_token('cpu'), settle 9.333, canonical E03 (token + HUD '16' / CORES WORKING
             AT ONCE) static from 9.45; kicker + card hard-cut there.

Every job is a pure function of time: each core has a work clock (T6: real time from 1.00; all
cores after the power-on: the shared load clock W(t), whose rate eases 1 -> 0.45 -> 2.0) and
precomputed job boundaries on it; one driver updater draws everything from self.now, the stage
drift composes on top.
"""

from style import *
from motif import *
import math

SLOT = 9.5
assert beat_duration("B03Cpu") == SLOT, "timeline.py slot changed: re-time b03"

# --------------------------------------------------------------------------
# World layout: a 5.6-wide grid of 1.10 cores. Vertical budget: grid top 3.31 (under the text
# zone), memory bar bottom -2.07, i.e. 0.45 clear of the HUD number's top (-2.52).
# --------------------------------------------------------------------------
CORE_XS = (-2.60, -1.10, 0.40, 1.90)
CORE_YS = (2.76, 1.52, 0.28, -0.96)
CORE_U = 1.10
N_CORES = 16
CORE_C = np.array([[CORE_XS[i % 4], CORE_YS[i // 4], 0.0] for i in range(N_CORES)])
COL_X = np.array([CORE_XS[i % 4] for i in range(N_CORES)])
HERO = 5                                    # T6 at (-1.10, 1.52): the hook's worker
GRID_C = np.array([X_C, (CORE_YS[0] + CORE_YS[-1]) / 2, 0.0])
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
FILL_IN = 0.07                              # progress fill inset from the core edge
FILL_W = CORE_U - 2 * FILL_IN
FILL_R = TOKEN_CORNER * CORE_U - FILL_IN

# --------------------------------------------------------------------------
# Timeline (beat-local seconds; script v2 'cpu'). Motion keeps its snappy durations; the slot
# is filled by calm reads BETWEEN the moves.
# --------------------------------------------------------------------------
T_EXPAND = EXPAND_TIME                      # 0.50 pull-back
T_HUD = 0.10                                # the counter's first appearance ('1', singular)
T_SOLO = 1.00                               # T6's first job ...
SOLO_PER = 0.80                             # ... one at a time: 1.00-1.80, 1.80-2.60, 2.60-3.40, 3.40-
T_CORES = 2.70                              # '16 cores. / 16 different jobs.' (read 1.0 s)
T_POWER = 3.70                              # power-on: SLAM 1 -> 16 + LIME line
T_SHARE = 5.50                              # 'They share / one memory.' (read 1.0 s)
T_BUSY = 6.50                               # traffic doubles
T_MORE = 7.90                               # 'Need more?'
T_DRAIN = 8.10                              # no new jobs; package outline draws on
T_CLEAR = 8.70                              # every job started before the drain has written back
T_COL, T_SET, T_EXIT = cut_times(SLOT)      # 8.933 / 9.333 / 9.45
T_PKG0, T_PKG1 = T_DRAIN, T_DRAIN + 0.45    # package outline Create
LINE_HOLD = T_SHARE - T_POWER               # 1.80: the LIME line holds the whole payoff
T_RISE = T_POWER - 0.50                     # riser 3.20 -> 3.70
DRIFT_S1 = 0.975                            # stage eases back 1.0 -> 0.975 about AC by the collapse

# the dim stage while T6 works alone: the other 15 ripple in outward from T6, then the bar
REV0, REV_K, REV_DUR = 0.12, 0.42, 0.30     # core k: from REV0 + REV_K * d_k (d in 0..1) over REV_DUR
MEM_REV = (0.55, 0.90)                      # bar + bus lanes + SHARED MEMORY settle in (dim)

# job cycle, as fractions of a job: fetch | compute | write back | idle
S_FETCH, S_DONE, S_WB = 0.30, 0.68, 0.96
PHASES = (0.0, S_FETCH, S_DONE, S_WB, 1.0)
DONE_FLASH = 0.16                           # LIME completion flash (s)
WRITE_FLASH = 0.25                          # arrival flash on the bar (s)
LEAVE_FLASH = 0.18                          # departure flash on the bar (s)

# the shared load clock after the power-on: rate (jobs per base period) eases between levels
RATE_KEYS = ((T_POWER, 1.0), (T_SHARE, 1.0), (T_SHARE + 0.30, 0.45),     # payoff -> low hum
             (T_BUSY, 0.45), (T_BUSY + 0.25, 2.0), (SLOT + 1.0, 2.0))     # -> traffic doubles
_TG = np.arange(T_POWER, SLOT + 1.0 + 1e-9, 1 / 1200)
_RG = np.interp(_TG, [k[0] for k in RATE_KEYS], [k[1] for k in RATE_KEYS])
_WG = np.concatenate([[0.0], np.cumsum((_RG[1:] + _RG[:-1]) / 2 * np.diff(_TG))])
SOLO_W = T_POWER - T_SOLO                   # 2.70: T6's own clock at the power-on (rate 1 both sides)


def work(t: float) -> float:
    """Shared load clock W(t): 0 until the power-on, then the integral of the rate."""
    return 0.0 if t <= T_POWER else float(np.interp(t, _TG, _WG))


def work_inv(w: float) -> float:
    return float(np.interp(w, _WG, _TG))


def clock(i: int, t: float) -> float:
    """Work clock of core i (T6 runs on real time from T_SOLO until it joins W at the power-on)."""
    if i == HERO:
        return t - T_SOLO if t < T_POWER else SOLO_W + work(t)
    return work(t)


def clock_inv(i: int, w: float) -> float:
    if i == HERO:
        return T_SOLO + w if w < SOLO_W else work_inv(w - SOLO_W)
    return work_inv(w)


# power-on light wave: a CYAN ring from the grid centre; each core pops as it passes
RING = dict(r0=0.5, r1=4.5, w0=10.0, op0=0.9, dur=0.45)


def ring_time(d: float) -> float:
    """Seconds after the power-on at which the ring's radius reaches d (ease_out_cubic)."""
    p = clip01((d - RING["r0"]) / (RING["r1"] - RING["r0"]))
    return RING["dur"] * (1.0 - (1.0 - p) ** (1.0 / 3.0))


POP_DELAY = np.array([ring_time(float(np.hypot(*(CORE_C[i, :2] - GRID_C[:2])))) for i in range(N_CORES)])
_DH = np.hypot(*(CORE_C[:, :2] - CORE_C[HERO, :2]).T)
REV_D = _DH / _DH.max()                     # ripple order of the dim reveal (T6 = 0)

# per-core jobs: base period P = 0.85 + 0.35u (default_rng(3), as before), every job its own
# length (x 0.70-1.35), the first job starts when the ring has passed + 0..0.55 (own phase)
MAX_CYC = 24
_rng = np.random.default_rng(3)
U = _rng.random(N_CORES)
V = _rng.random(N_CORES)
PER = 0.85 + 0.35 * U
JOB = 0.70 + 0.65 * _rng.random((N_CORES, MAX_CYC))
BOUND = np.zeros((N_CORES, MAX_CYC + 1))    # job boundaries on each core's work clock
for _i in range(N_CORES):
    if _i == HERO:
        _L = np.concatenate([np.full(4, SOLO_PER), PER[_i] * JOB[_i, 4:]])
        _b0 = 0.0
    else:
        _L = PER[_i] * JOB[_i]
        _b0 = POP_DELAY[_i] + 0.55 * V[_i]
    BOUND[_i] = _b0 + np.concatenate([[0.0], np.cumsum(_L)])
# EVT[i, n, k]: local time of phase PHASES[k] of job n on core i
EVT = np.array([[[clock_inv(i, BOUND[i, n] + p * (BOUND[i, n + 1] - BOUND[i, n])) for p in PHASES]
                 for n in range(MAX_CYC)] for i in range(N_CORES)])
# a job runs if it starts before the drain and writes back before T_CLEAR (a prefix per core)
_ok = (EVT[:, :, 0] < T_DRAIN - EPS) & (EVT[:, :, 3] < T_CLEAR - EPS)
N_ALIVE = np.array([int(np.argmin(r)) if not r.all() else MAX_CYC for r in _ok])
assert all(_ok[i, :N_ALIVE[i]].all() and not _ok[i, N_ALIVE[i]:].any() for i in range(N_CORES))
assert N_ALIVE.max() < MAX_CYC
assert EVT[HERO, 0, 0] == T_SOLO and abs(EVT[HERO, 1, 0] - (T_SOLO + SOLO_PER)) < 1e-9

# looks
DIM_FILL, DIM_STROKE = 0.10, 0.85           # unpowered cores: DIM outline, faint DIM fill
FILL_REST, FILL_REST_ON = 0.15, 0.25        # CYAN unit fill: T6 before / every core after the power-on
PROG_A = 0.55                               # progress fill opacity (CYAN)
DONE_A, DONE_A_ON = 0.85, 0.60              # LIME completion flash peak before / after the power-on
LAB_DIM, LAB_ON, LAB_BACK = 0.40, 0.90, 0.50   # T label opacity: '16 cores.' / powered / 'They share'
LAB_STAG = 0.025                            # T1 -> T16 fade-in stagger at 2.70
MEM_FILL, MEM_FILL_SHARE = 0.10, 0.24       # bar fill at rest / once 'They share one memory.'
MEM_DIM = 0.50                              # bar + label opacity factor before the power-on
LANE_A = 0.8                                # bus lane stub opacity


def label_alpha(i: int, now: float) -> float:
    """T label opacity factor of core i (0 until '16 cores.')."""
    a = LAB_DIM * ease_out_cubic(ramp(now, T_CORES + LAB_STAG * i, T_CORES + LAB_STAG * i + 0.25))
    tp = T_POWER + POP_DELAY[i]
    a = lerp(a, LAB_ON, ease_out_cubic(ramp(now, tp, tp + 0.15)))
    a = lerp(a, LAB_BACK, ease_out_cubic(ramp(now, T_SHARE, T_SHARE + 0.40)))
    return a * (1.0 - ease_in_out_sine(ramp(now, T_DRAIN, T_DRAIN + 0.50)))


def thread_label(i: int) -> MarkupText:
    """'T{i+1}' in JetBrains Mono BOLD 38, INK; tracking -700 (1/1024 pt) so 'T16' stays clear
    of the 1.10 core's rounded corners. Opacity is driven per frame (label_alpha)."""
    return MarkupText(f'<span letter_spacing="-700">T{i + 1}</span>', font=FONT_MONO, font_size=38,
                      color=INK, weight=BOLD)


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


def job_events(k: int, t0: float, t1: float) -> list[tuple[float, int]]:
    """(time, core) of phase PHASES[k] of every running job with t0 <= time < t1, sorted."""
    out = [(float(EVT[i, n, k]), i) for i in range(N_CORES) for n in range(N_ALIVE[i])
           if t0 - EPS <= EVT[i, n, k] < t1 - EPS]
    return sorted(out)


def col_pan(i: int, rng) -> float:
    return float(np.clip((COL_X[i] - X_C) / 2.25 * 0.55 + rng.uniform(-0.15, 0.15), -0.8, 0.8))


class B03Cpu(BeatScene):
    def beat(self):
        # ---------------- entry: E01 (the plain CYAN token, no HUD) ----------------
        st = enter(self)
        assert st.hud is None

        # ---------------- world at rest ----------------
        self.cores = VGroup()
        self.labels = VGroup()
        for i in range(N_CORES):
            if i == HERO:       # the token itself: CYAN unit, paired 1:1 with handoff_token('plain')
                core = unit_with_glyph("plain", CORE_U, CORE_C[i])
            else:               # unpowered until the power-on
                core = unit_with_glyph("plain", CORE_U, CORE_C[i], DIM, fill=DIM_FILL)
                core.unit.set_stroke(DIM, opacity=DIM_STROKE)
            self.cores.add(core)
            lab = thread_label(i).move_to(CORE_C[i])
            assert_safe(lab)
            assert lab.width < CORE_U - 0.20
            self.labels.add(lab)
        assert len(self.cores[HERO].family_members_with_points()) == 1     # plain: no lane glyph
        # the progress level's INK edge hides while it passes behind the T label (it read as a
        # strike-through), measured at rest relative to the core centre
        lb = min(lab.get_bottom()[1] - CORE_C[i, 1] for i, lab in enumerate(self.labels))
        lt = max(lab.get_top()[1] - CORE_C[i, 1] for i, lab in enumerate(self.labels))
        self._edge_band = (lb - 0.10, lb - 0.03, lt + 0.03, lt + 0.10)
        # progress fills (empty at rest: not part of the EXPAND capture), under the cores so the
        # unit stroke and label stay on top
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
        assert self.mem.get_bottom()[1] - hud_number_tmpl("16").get_top()[1] >= 0.44
        self.mem_lab = mem_lab
        # bus lanes: from the bar up each gutter to the top row's centre line
        self.stubs = VGroup(*[Line([x, MEM_TOP, 0], [x, CORE_YS[0], 0], stroke_color=DIM, stroke_width=2,
                                   stroke_opacity=LANE_A) for x in BUS_X])
        assert min(BUS_X) > STAGE.x0 and max(BUS_X) < STAGE.x1
        assert all(self.mem.get_left()[0] < x < self.mem.get_right()[0] for x in BUS_X)
        self.world = VGroup(self.stubs, self.mem, self.mem_lab, self.pfills, self.plines, self.cores, self.labels)
        assert self.cores.get_top()[1] < TTZ_BOTTOM
        # EXPAND: the hero is the token at e = 0; labels, bar, lanes and the 15 dim cores get
        # their opacity from the beat's own clock (the ripple reveal), so EXPAND draws them at 0
        hidden = (0.0, 1.0, 0.0, 0.0)
        rig = EXPAND(self.world, CORE_C[HERO], CORE_U, 0.0, hero=self.cores[HERO], token="plain",
                     fades={self.labels: hidden, self.mem: hidden, self.mem_lab: hidden, self.stubs: hidden})

        # rest copies (taken at rest, before any camera move)
        R = lambda m: rig.rest[rig.index[id(m)]]
        self._core_rest = []
        for i, core in enumerate(self.cores):
            mems = [core.unit] + list(self.labels[i].family_members_with_points())
            self._core_rest.append([(m, R(m).p.copy()) for m in mems])
        self._unit_rest = [(R(c.unit).fr.copy(), R(c.unit).sr.copy()) for c in self.cores]
        self._unit_sw = R(self.cores[0].unit).sw
        self._lab_rest = [[(m, R(m).fr.copy()) for m in lab.family_members_with_points()] for lab in self.labels]
        self._memlab_rest = [(m, R(m).fr.copy()) for m in self.mem_lab.family_members_with_points()]
        self._stub_rest = [(m, R(m).sr.copy()) for m in self.stubs]
        self._alpha_key: dict = {}

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

        # ---------------- package outline (drawn on at the drain) ----------------
        self.pkg_full = RoundedRectangle(width=6.9, height=5.56, corner_radius=0.35, stroke_color=CYAN,
                                         stroke_width=3, fill_color=CYAN, fill_opacity=0).move_to([X_C, 0.62, 0])
        self.pkg = self.pkg_full.copy().set_stroke(opacity=0)

        # ---------------- text: hard swaps, every card POPs on and hard-cuts off ----------------
        track = TextTrack(self)
        # kicker on the cut ('in' at 0.0 is invisible on frame 0), hard-cut by the E03 swap
        track.on(kick("LEVEL 1/3 · ONE CPU"), 0.0, T_EXIT, exit="cut")
        cards = (
            (card([("That worker?", "H84", INK), ("One CPU core.", "H84", INK, {"CPU core.": CYAN})]),
             1 / 60, T_CORES),                                  # POP on story frame 1 (frame 0 = E01)
            (card([("16 cores.", "H96", INK, {"16": CYAN}), ("16 different jobs.", "S56", INK)]),
             T_CORES, T_SHARE),
            (card([("They share", "H84", INK), ("one memory.", "H84", INK, {"memory.": MAGENTA})]),
             T_SHARE, T_MORE),
            (card([("Need more?", "H96", INK)]), T_MORE, T_EXIT),
        )
        for c, t_in, t_out in cards:
            for line in c:
                track.on(line, t_in, t_out, enter="pop", exit="cut")

        # ---------------- HUD: the counter's first appearance ----------------
        hud = core_hud("1", scene=self, pop_at=T_HUD)       # zero pixels until 0.10, then POP
        hud.slam(T_POWER, "16")
        hud.label(T_POWER, core_label(16))                   # plural from here (under the line)
        hud.mult(T_POWER, COUNTER_LINES["B03Cpu"], hold=LINE_HOLD)
        assert hud.number_at(T_EXIT)[0] == "16" and hud.label_at(T_EXIT)[0] == CORES_LABEL

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
        self.add(hud)                          # HUD on top
        self.collapse = None
        self._draw(0.0)
        drive(self, self._draw)
        self.add(self.drift)                   # after the driver: composes on this frame's drawing

        # ---------------- run (audio cues sit next to the moment they belong to) ----------------
        trng = np.random.default_rng(17)
        self.cue("hit", gain=0.8, note="match-cut: the worker is core T6")
        self.cue("whoosh", dt=0.20, dur=0.50, direction="up", note="camera pull-back")

        self.hold_until(T_SOLO)                # T6 alone: a soft tick on each fetch and each done
        for k, f, what in ((0, 3000, "fetch"), (2, 4400, "done")):
            for t, i in job_events(k, T_SOLO, T_POWER):
                self.cue("tick", dt=round(t - self.t, 4), gain=0.30, freq=f, pan=round(col_pan(i, trng), 3),
                         note=f"T6 {what}")

        self.hold_until(T_RISE)
        self.cue("riser", dur=round(T_POWER - T_RISE, 4), note="into the power-on")

        self.hold_until(T_POWER)
        self.cue("hit", note="power-on: 1 -> 16")
        self.cue("sub_drop", dur=0.6)
        self.cue("shimmer", dur=1.2)
        for t, i in job_events(1, T_POWER, T_SHARE):           # scattered, panned, out of phase
            self.cue("tick", dt=round(t - self.t, 4), gain=0.28, freq=int(trng.uniform(2600, 5400)),
                     pan=round(col_pan(i, trng), 3), note=f"T{i + 1} compute")

        self.hold_until(T_SHARE)
        self.cue("tick", gain=0.5, freq=4200, pan=0, note="'They share / one memory.'")
        for t, i in job_events(1, T_SHARE, T_BUSY):            # the low hum
            self.cue("tick", dt=round(t - self.t, 4), gain=0.16, freq=int(trng.uniform(2600, 4200)),
                     pan=round(col_pan(i, trng), 3), note=f"T{i + 1} compute (hum)")

        self.hold_until(T_BUSY)                # traffic doubles: busy panned blips + arrival ticks
        for t, i in job_events(1, T_BUSY, T_CLEAR):
            self.cue("tick", dt=round(t - self.t, 4), gain=0.24, freq=int(trng.uniform(1700, 2600)),
                     pan=round(col_pan(i, trng), 3), note=f"T{i + 1} blip")
        for t, i in job_events(3, T_BUSY, T_CLEAR):
            self.cue("tick", dt=round(t - self.t, 4), gain=0.14, freq=int(trng.uniform(5200, 6400)),
                     pan=round(col_pan(i, trng), 3), note=f"T{i + 1} result lands")

        self.hold_until(T_DRAIN)
        self.cue("riser", dur=round(T_COL - T_DRAIN, 4), note="drain + package into the collapse")

        self.hold_until(T_COL)
        # draw the fully drained state first, so no packet / trail / flash hangs over the collapse,
        # put the drift camera back on top of it, and let the collapse start from that pose
        self._draw_world(T_COL)
        self._draw_package(T_COL)
        self.drift.render_at(T_COL)
        self.drift.release()
        self.collapse = COLLAPSE(self.pkg, self.world, "cpu")
        self.add(self.collapse.glyph)
        self.cue("whoosh", dt=round((T_COL + T_SET) / 2 - self.t, 4), dur=0.45, direction="down",
                 note="collapse")
        self.cue("tick", dt=round(T_SET - self.t, 4), gain=0.5, note="settle")
        hold_floor(self, T_EXIT)
        exit_to(self)                          # canonical E03, static to 9.50

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.collapse is not None:
            self.collapse.render(now, T_COL, T_SET, T_EXIT)
            return
        if now < T_EXPAND + 0.1:               # (from T_EXPAND on: e = 1 exactly, i.e. the rest layout)
            EXPAND(self.world, CORE_C[HERO], CORE_U, ease_out_expo(ramp(now, 0.0, T_EXPAND)),
                   ink=1.0 if flash_frames(now, 1 / 60, 2) else 0.0)
            self._alpha_key.clear()            # EXPAND rewrote every style
        self._draw_world(now)
        self._draw_package(now)

    def _draw_package(self, now: float) -> None:
        a = ease_out_cubic(ramp(now, T_PKG0, T_PKG1))
        if a <= 0:
            self.pkg.set_stroke(opacity=0)
        elif a >= 1:
            self.pkg.points = self.pkg_full.points.copy()
            self.pkg.set_stroke(CYAN, width=3, opacity=1)
        else:
            self.pkg.pointwise_become_partial(self.pkg_full, 0, a)
            self.pkg.set_stroke(CYAN, width=3, opacity=1)

    def _set_alpha_cached(self, key, rest: list, a: float, attr: str = "fill_rgbas") -> None:
        """rest: [(member, rest rgbas)]; writes rgbas * a only when a changed."""
        a = round(float(a), 9)
        if self._alpha_key.get(key) == a:
            return
        self._alpha_key[key] = a
        for m, r in rest:
            v = r.copy()
            v[:, 3] *= a
            setattr(m, attr, v)

    def _draw_world(self, now: float) -> None:
        # ---- the dim stage: the 15 cores ripple in outward from T6, then bar + lanes + label
        a_mem = ease_out_cubic(ramp(now, *MEM_REV))
        self._set_alpha_cached("stubs", self._stub_rest, a_mem, "stroke_rgbas")
        drain_lab = 1.0 - ease_in_out_sine(ramp(now, T_DRAIN, T_DRAIN + 0.50))
        powered = now >= T_POWER - EPS
        share = ease_out_cubic(ramp(now, T_SHARE, T_SHARE + 0.35))
        ml = a_mem * (lerp(0.85, 1.0, share) if powered else 0.60) * drain_lab
        self._set_alpha_cached("memlab", self._memlab_rest, ml)
        for i in range(N_CORES):
            self._set_alpha_cached(("lab", i), self._lab_rest[i], label_alpha(i, now))
        if now < T_EXPAND - EPS:               # EXPAND owns points + the hero; set the reveal alphas
            for i in range(N_CORES):
                if i != HERO:
                    self._dim_style(i, now)
            self._draw_mem(now, a_mem, 0.0, 0.0, False, share)
            return
        self._draw_jobs(now, a_mem, powered, share)

    def _dim_style(self, i: int, now: float) -> None:
        """An unpowered core: its DIM rest style x the ripple reveal (T6 outward, 0.12-0.84)."""
        a = ease_out_cubic(ramp(now, REV0 + REV_K * REV_D[i], REV0 + REV_K * REV_D[i] + REV_DUR))
        fr, sr = self._unit_rest[i]
        u = self.cores[i].unit
        if a >= 1.0:
            u.fill_rgbas, u.stroke_rgbas = fr.copy(), sr.copy()
        else:
            u.fill_rgbas = np.array([[*fr[0, :3], fr[0, 3] * a]])
            u.stroke_rgbas = np.array([[*sr[0, :3], sr[0, 3] * a]])

    def _draw_mem(self, now, a_mem, mem_busy, flash_decay, ink_flash, share) -> None:
        dimk = 1.0 if now >= T_POWER - EPS else MEM_DIM
        base = lerp(MEM_FILL, MEM_FILL_SHARE, share)
        mf = base * dimk * a_mem + 0.07 * min(mem_busy, 3.0) + 0.22 * decay(now - T_SHARE, 0.45)
        if flash_decay > 0:                    # the power-on flash
            mf = max(mf, lerp(base, 0.6, flash_decay))
        self.mem.fill_rgbas = rgba(MAGENTA, mf).reshape(1, 4)
        ink = 0.7 if ink_flash else 0.25 * share
        self.mem.stroke_rgbas = rgba(interpolate_color(MAGENTA, INK, ink), dimk * a_mem).reshape(1, 4)

    def _draw_jobs(self, now: float, a_mem: float, powered: bool, share: float) -> None:
        drain = 1.0 - ramp(now, T_COL - 0.20, T_COL - 1 / 60)   # safety net: nothing hangs over the collapse
        ink_flash = powered and flash_frames(now, T_POWER, 2)
        # power-on light wave (pw: 1 on the hit's 2 frames, then cools over 0.35 s)
        pw = 0.0
        if powered:
            pw = 1.0 if ink_flash else 1.0 - ease_out_cubic(ramp(now, T_POWER + 2 / 60, T_POWER + 0.35))
        w_all = work(now)
        lab_vis = ramp(now, T_CORES, T_CORES + 0.10)            # the edge hides behind labels once they exist
        e0, e1, e2, e3 = self._edge_band
        mem_busy = 0.0

        for i in range(N_CORES):
            ctr = CORE_C[i]
            core = self.cores[i]
            on = powered or i == HERO
            # ---- this core's job state
            w = (now - T_SOLO if now < T_POWER else SOLO_W + w_all) if i == HERO else w_all
            B = BOUND[i]
            n = int(np.searchsorted(B, w, side="right")) - 1
            if 0 <= n < MAX_CYC:
                s = (w - B[n]) / (B[n + 1] - B[n])
                live = n < N_ALIVE[i]
            else:
                n, s, live = -1, -1.0, False
            comp = live and S_FETCH <= s < S_DONE

            def last(k):
                """(job, time) of the latest phase-k event at or before now, or None."""
                m = min(int(np.searchsorted(EVT[i, :, k], now + EPS, side="right")) - 1, N_ALIVE[i] - 1)
                return (m, EVT[i, m, k]) if m >= 0 else None

            # ---- compute start: INK outline kick + pop; the power-on pop as the ring passes
            kick_ink, pop = 0.0, 1.0
            ev = last(1)
            if ev is not None:
                kick_ink = 1.0 if flash_frames(now, ev[1], 2) else 0.0
                pop = pop_scale(now - ev[1], 1.08, 0.20)
            if powered:
                pop *= pop_scale(now - T_POWER - POP_DELAY[i], 1.12, 0.25)
            # ---- unit style
            u = core.unit
            if not on:                          # unpowered: the DIM rest style (+ the reveal)
                self._dim_style(i, now)
                u.stroke_width = self._unit_sw
            else:
                fill_rest = FILL_REST_ON if powered else FILL_REST
                if ink_flash:
                    ufill = rgba(INK, 0.9)
                elif pw > 0:
                    ufill = rgba(CYAN, lerp(fill_rest, 0.6, pw))
                else:
                    ufill = rgba(CYAN, fill_rest)
                u.fill_rgbas = ufill.reshape(1, 4)
                kk = max(kick_ink * drain, 0.85 * pw)
                u.stroke_rgbas = rgba(interpolate_color(CYAN, INK, kk), 1.0).reshape(1, 4)
                u.stroke_width = self._unit_sw * (1.0 + 0.6 * max(kick_ink * drain, pw))
            # ---- pop (whole core + label, about its centre)
            for m, p in self._core_rest[i]:
                m.points = p.copy() if pop == 1.0 else (p - ctr) * pop + ctr
            # ---- progress fill / LIME completion flash
            pf, pl = self.pfills[i], self.plines[i]
            y0 = ctr[1] - FILL_W / 2
            prog, pcol, pa = 0.0, CYAN, 0.0
            if comp:
                prog = (s - S_FETCH) / (S_DONE - S_FETCH)
                pa = PROG_A * drain
            else:
                ev = last(2)
                if ev is not None and 0.0 <= now - ev[1] < DONE_FLASH:
                    prog, pcol = 1.0, LIME
                    pa = ((DONE_A_ON if ev[1] >= T_POWER else DONE_A)
                          * (1.0 - ease_in_quad((now - ev[1]) / DONE_FLASH)) * drain)
            if pa > 0.004 and prog > 0.004:
                pts = fill_pts(ctr[0], y0, FILL_W, FILL_W * prog, FILL_R, FILL_W)
                pf.points = (pts - ctr) * pop + ctr if pop != 1.0 else pts
                pf.fill_rgbas = rgba(pcol, pa).reshape(1, 4)
                ly = y0 + FILL_W * prog
                dl = ly - ctr[1]
                ek = 1.0 - lab_vis * ramp(dl, e0, e1) * (1.0 - ramp(dl, e2, e3))
                if comp and prog < 0.97 and ek > 0.0:   # the rising level: a bright INK edge
                    dy = max(FILL_R - min(FILL_W * prog, FILL_W * (1.0 - prog)), 0.0)   # inside a corner arc
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
                    Rt = self._routes[kind][i]
                    uu = ease_in_out_cubic(q) * Rt.L
                    pos = Rt.at(uu)
                    sc = lerp(0.6, 1.0, ramp(q, 0, 0.12)) if kind == "f" else lerp(1.0, 0.7, ramp(q, 0.88, 1.0))
                    al = (ramp(q, 0.0, 0.08) * (1 - ramp(q, 0.90, 1.0)) if kind == "f"
                          else ramp(q, 0.0, 0.10) * (1 - ramp(q, 0.94, 1.0))) * drain
                    if kind == "f" and q > 0.80:      # shrinks into the core as it is absorbed
                        sc *= lerp(1.0, 0.6, ramp(q, 0.80, 1.0))
                    pk.points = self._pkt * sc + pos
                    pk.fill_rgbas = rgba(col, al).reshape(1, 4)
                    for tm, ln, wd, ta in ((t1, TRAIL, 9.0, 0.35), (t2, TRAIL * 0.5, 15.0, 0.35)):
                        u0 = max(uu - ln, 0.0)
                        if uu - u0 > 0.02:
                            tm.points = Rt.poly(u0, uu)
                            tm.set_stroke(col, width=wd * sc, opacity=ta * al)
                        elif len(tm.points):
                            tm.points = EMPTY.copy()
                else:
                    for m in (pk, t1, t2):
                        if len(m.points):
                            m.points = EMPTY.copy()

            # ---- bar flashes: LIME where a result lands, MAGENTA where a fetch leaves
            for fl, k, dur, w_h, op, port in ((self.wflash[i], 3, WRITE_FLASH, (0.9, 0.3), 0.9, ctr[0] + GUT),
                                              (self.fflash[i], 0, LEAVE_FLASH, (0.6, 0.22), 0.7, ctr[0] - GUT)):
                ev = last(k)
                if ev is not None and 0.0 <= now - ev[1] < dur:
                    kq = (now - ev[1]) / dur
                    env = (1.0 - ease_out_quad(kq)) * drain
                    sc = lerp(0.6, 1.25, ease_out_cubic(kq))
                    fl.points = self._el * np.array([w_h[0] * sc, w_h[1] * sc, 1.0]) + np.array([port, MEM_TOP, 0.0])
                    fl.fill_rgbas = fl.fill_rgbas.copy()
                    fl.fill_rgbas[:, 3] = op * env
                    mem_busy += env
                elif len(fl.points):
                    fl.points = EMPTY.copy()

        # ---- the shared-memory bar: dim until the power-on (flash), busy with traffic,
        # brightens on 'They share / one memory.'
        self._draw_mem(now, a_mem, mem_busy * drain, pw, ink_flash, share)

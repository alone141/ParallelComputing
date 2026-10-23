"""b06 - SCALE 5/6 - DATA CENTER: no shared memory, the servers pass messages.

The node token (E05) is server (c3, r3) of one pod (8 racks x 8 servers): the camera
pulls back (EXPAND). Two all-reduce SYNC cycles run over the pod's network: every server
computes on its own data (CYAN flash), MAGENTA gradient messages climb the racks to the
top-of-rack switches and on to the spines, LIME updates come back down and every server
applies the same update (LIME flash).

At 2.45 a second pull-back shrinks the pod into 16 squares of a 56 x 56 hall (3,136
squares = 100,352 GPUs) while the HUD log-rolls and accelerates. 80 ms freeze + silence,
then the biggest drop of the film at 3.40: HUD SLAMs 1,689,600,000+, the hall flashes, a
shock ring + impact shake; messages race along the trunks and two gather/broadcast
all-reduce pulses sweep the hall. 5.05 an outline draws on, the hall COLLAPSEs into
handoff_token('cluster') and settles (E06).

Everything that moves is a pure function of self.now (one driver updater). The hall is
drawn as a handful of multi-subpath VMobjects (one per quantised colour), off-frame
squares are skipped.
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# Stage A: one pod (8 racks x 8 servers), storyboard b06 visuals
# --------------------------------------------------------------------------
SRV_U = 0.36
SRV_PITCH = 0.56
RACK_X = X_C + (np.arange(8) - 3.5) * SRV_PITCH          # rack c = 0..7
ROW_Y = (3.5 - np.arange(8)) * SRV_PITCH                 # row r = 0..7 (top = 0)
SRV_C = np.array([[RACK_X[k % 8], ROW_Y[k // 8], 0.0] for k in range(64)])
HERO = 3 * 8 + 3                                         # (c3, r3) at (-0.63, 0.28)
SLOT = SRV_C[HERO].copy()
RACK_TOP, RACK_BOT = 2.42, -2.14
TOR_Y = 2.50
UP_Y0, UP_Y1 = 2.58, 2.95
SPINE_X = np.array([-2.03, -0.91, 0.21, 1.33])
SPINE_Y = 3.05
PIVOT_A = np.array([X_C, 0.0, 0.0])                      # stage-A centre (zoom-out 2 pivot)

# --------------------------------------------------------------------------
# Stage B: the hall (56 x 56 squares = 3,136; 1 square = 4 servers = 32 GPUs)
# --------------------------------------------------------------------------
HN = 56
_I, _J = np.meshgrid(np.arange(HN), np.arange(HN), indexing="ij")   # i = column, j = row
HI, HJ = _I.ravel(), _J.ravel()
HALL_C = np.stack([X_C + (HI - 27.5) * 0.10, 0.45 + (27.5 - HJ) * 0.10, np.zeros(HN * HN)], axis=1)
HALL_D = np.hypot(HALL_C[:, 0] - AC[0], HALL_C[:, 1] - AC[1])
POD = (HI >= 26) & (HI <= 29) & (HJ >= 26) & (HJ <= 29)            # where the pod lands
HALL_SQ = 0.07
HALL_UNIT = RoundedRectangle(width=1.0, height=1.0, corner_radius=0.012 / 0.07).points.copy()
HALL_VAR = 0.80 + 0.20 * np.random.default_rng(61).random(HN * HN)  # per-square brightness texture
HALL_X0, HALL_X1 = X_C - 2.785, X_C + 2.785
HALL_Y0, HALL_Y1 = 0.45 - 2.785, 0.45 + 2.785
TRUNK_X = np.array([-2.35, -1.55, -0.75, 0.05, 0.85, 1.65])
TRUNK_Y = np.array([2.45, 1.65, 0.85, 0.05, -0.75, -1.55])
ZOOM2 = 11.2
POD_OP = 0.60                                            # the pod's 16 squares right after the zoom

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
T_EXP = 0.80
SYNC = (0.85, 1.65)
T_Z0, T_Z1 = 2.45, 3.35
T_ROLL0, T_ROLL1 = 2.60, 3.32
T_FREEZE = 3.32
T_DROP = 3.40
T_MSG_END = 5.05
PULSES = (("gather", 3.60, 3.90), ("bcast", 3.95, 4.25), ("gather", 4.35, 4.65), ("bcast", 4.70, 5.00))
T_OUT0, T_OUT1 = 5.05, 5.15                               # outline draws on
T_COL0, T_COL1 = 5.15, 5.78                               # COLLAPSE
T_SET1 = 5.95                                             # SETTLE end; static to 6.00
T_TEXT_OUT = 5.25

# messages on the trunks: 120 seeded random walks on the 6 x 6 trunk lattice, 3 u/s
N_MSG = 120
MSG_SPEED = 3.0
EDGE = 0.80
_mrng = np.random.default_rng(66)


def _walks() -> np.ndarray:
    steps = 12
    out = np.zeros((N_MSG, steps + 1, 2), int)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for n in range(N_MSG):
        a, b = int(_mrng.integers(0, 6)), int(_mrng.integers(0, 6))
        out[n, 0] = (a, b)
        prev = None
        for s in range(steps):
            opts = [(a + dx, b + dy) for dx, dy in dirs if 0 <= a + dx < 6 and 0 <= b + dy < 6]
            if prev is not None and len(opts) > 1:
                opts = [o for o in opts if o != prev]
            nxt = opts[int(_mrng.integers(0, len(opts)))]
            prev = (a, b)
            a, b = nxt
            out[n, s + 1] = (a, b)
    return out


WALK = _walks()                                          # (120, 13, 2) lattice indices
WALK_XY = np.stack([TRUNK_X[WALK[..., 0]], TRUNK_Y[WALK[..., 1]]], axis=-1)
MSG_PHASE = _mrng.random(N_MSG) * EDGE                    # start partway along the first edge
MSG_TAIL = 0.14


def stage_mask(y):
    """Soft stage window: 1 for y in [-2.37, 3.25], 0 below -2.62 / above 3.55, so camera
    moves never drag geometry under the text zone or the HUD. Identity at rest."""
    y = np.asarray(y, float)
    return np.clip((y + 2.62) / 0.25, 0, 1) * np.clip((3.55 - y) / 0.30, 0, 1)


def edge_mask(bot: float, top: float) -> float:
    """Stage window for a whole member: fades as its bottom edge goes under -2.45 (HUD number
    top is -2.52) or its top edge over 3.40 (cards end at 3.63)."""
    return clip01((bot + 2.62) / 0.17) * clip01((3.58 - top) / 0.18)


def hold_floor(scene, t: float) -> None:
    """Hold until the last frame at or before t (exact at 60 fps, rounds down at 30)."""
    n = int(math.floor(t * config.frame_rate + 1e-6)) - scene._frames
    if n > 0:
        scene.wait((n - 0.5) / config.frame_rate)


def seg_points(a: np.ndarray, b: np.ndarray, min_len: float = 0.004) -> np.ndarray:
    """(n,3),(n,3) -> cubic points of n separate straight subpaths a->b. A minimum length
    keeps each subpath open, so round caps draw a dot even when a == b."""
    d = b - a
    L = np.hypot(d[:, 0], d[:, 1])
    short = L < min_len
    if np.any(short):
        b = b.copy()
        b[short] = a[short] + np.array([min_len, 0.0, 0.0])
        d = b - a
    return np.stack([a, a + d / 3, a + 2 * d / 3, b], axis=1).reshape(-1, 3)


PKT_LEN = 0.20                                           # packet streak length (units)


def streak(P0: np.ndarray, P1: np.ndarray, q_tail, q_head):
    """(tails, heads) of packets on segments P0->P1 at progress q_tail <= q_head."""
    qt = np.asarray(q_tail, float)[:, None]
    qh = np.asarray(q_head, float)[:, None]
    return P0 + (P1 - P0) * qt, P0 + (P1 - P0) * qh


def streak_mob(color, width: float) -> VMobject:
    m = VMobject(stroke_color=color, stroke_width=width, fill_opacity=0)
    m.cap_style = CapStyleType.ROUND
    m.points = np.zeros((0, 3))
    return m


class Field(VGroup):
    """N same-shape cells drawn as a few multi-subpath VMobjects, one per quantised RGBA
    (a pool; unused members get no points). Off-frame and transparent cells are skipped.
    ~6x cheaper in Cairo than one VMobject per cell."""

    def __init__(self, unit_pts: np.ndarray, pool: int = 160):
        super().__init__(*[VMobject(stroke_width=0, fill_opacity=0) for _ in range(pool)])
        for m in self.submobjects:
            m.points = np.zeros((0, 3))
        self.unit = np.asarray(unit_pts, float)
        self.max_used = 0

    def clear(self) -> None:
        for m in self.submobjects:
            if len(m.points):
                m.points = np.zeros((0, 3))

    def draw(self, centers: np.ndarray, sizes, rgba: np.ndarray, qc: int = 28, qa: int = 40) -> None:
        sizes = np.broadcast_to(np.asarray(sizes, float), (len(centers),))
        half = sizes / 2
        vis = ((rgba[:, 3] > 0.004) & (np.abs(centers[:, 0]) < FRAME_W / 2 + half)
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
                m.points = np.zeros((0, 3))
        self.max_used = max(self.max_used, used)


def mix(c0, c1, t):
    """Vectorised colour lerp: c0, c1 (3,) or (n,3) rgb; t scalar or (n,) -> (n,3)."""
    t = np.atleast_1d(np.asarray(t, float))[:, None]
    c0 = np.atleast_2d(np.asarray(c0, float))
    c1 = np.atleast_2d(np.asarray(c1, float))
    return c0 * (1 - t) + c1 * t


RGB_CYAN = np.array(CYAN.to_rgb())
RGB_MAG = np.array(MAGENTA.to_rgb())
RGB_INK = np.array(INK.to_rgb())
RGB_LIME = np.array(LIME.to_rgb())


def ease_out_quad_arr(x):
    return 1 - (1 - x) * (1 - x)


def decay(t_since: float, dur: float) -> float:
    """1 at the hit, easing out to 0 over dur (0 before the hit)."""
    if t_since < -EPS:
        return 0.0
    return 1.0 - ease_out_cubic(clip01(t_since / dur))


class B06DataCenter(BeatScene):
    def beat(self):
        # ---------------- entry: E05 (node token + HUD '135,168') ----------------
        st = enter(self)
        hud = st.hud

        # ---------------- stage A at rest ----------------
        self.racks = VGroup(*[Line([x, RACK_TOP, 0], [x, RACK_BOT, 0], stroke_color=MAGENTA,
                                   stroke_width=1.5, stroke_opacity=0.35) for x in RACK_X])
        self.uplinks = VGroup(*[Line([RACK_X[c], UP_Y0, 0], [SPINE_X[s], UP_Y1, 0], stroke_color=MAGENTA,
                                     stroke_width=1.2, stroke_opacity=0.30)
                                for c in range(8) for s in range(4)])
        self.tors = VGroup(*[RoundedRectangle(width=0.44, height=0.16, corner_radius=0.05, stroke_color=MAGENTA,
                                              stroke_width=2, fill_color=MAGENTA, fill_opacity=0.30)
                             .move_to([x, TOR_Y, 0]) for x in RACK_X])
        self.spines = VGroup(*[RoundedRectangle(width=0.80, height=0.20, corner_radius=0.06, stroke_color=MAGENTA,
                                                stroke_width=2.5, fill_color=MAGENTA, fill_opacity=0.35)
                               .move_to([x, SPINE_Y, 0]) for x in SPINE_X])
        srv = []
        for k in range(64):
            if k == HERO:
                self.hero = unit_with_glyph("node", SRV_U, SRV_C[k], CYAN, fill=0.18, stroke=2)
                srv.append(self.hero)
            else:
                srv.append(unit(SRV_U, CYAN, 0.18, 2).move_to(SRV_C[k]))
        self.servers = VGroup(*srv)
        self.srv_units = [s.unit if k == HERO else s for k, s in enumerate(srv)]
        self.world = VGroup(self.racks, self.uplinks, self.tors, self.spines, self.servers)
        EXPAND(self.world, SLOT, SRV_U, 0.0, hero=self.hero, token="node",
               fades={self.hero.glyph: (0.0, 0.35, 1.0, 0.0)})        # captures rest, draws e = 0
        self.rig = self.world._motif_expand
        rig = self.rig
        self.ix_srv = [rig.index[id(u)] for u in self.srv_units]
        self.ix_glyph = rig.indices_of(self.hero.glyph)
        self.ix_rack = rig.indices_of(self.racks)
        self.ix_up = rig.indices_of(self.uplinks)
        self.ix_tor = rig.indices_of(self.tors)
        self.ix_spine = rig.indices_of(self.spines)
        self.ix_net = self.ix_rack + self.ix_up + self.ix_tor + self.ix_spine
        self.ix_hero = set(rig.indices_of(self.hero))
        self.ix_rack_set = set(self.ix_rack)
        self.stage_a_on = True

        # stage-A packets (one stroke VMobject per colour; round caps = dots / streaks)
        self.dots_m = streak_mob(MAGENTA, 8)
        self.dots_l = streak_mob(LIME, 8)

        # ---------------- stage B: hall, trunks, messages ----------------
        self.hall = Field(HALL_UNIT, pool=200)
        self.trunks = VGroup(*[Line([x, HALL_Y0, 0], [x, HALL_Y1, 0]) for x in TRUNK_X],
                             *[Line([HALL_X0, y, 0], [HALL_X1, y, 0]) for y in TRUNK_Y])
        for ln in self.trunks:
            ln.set_stroke(MAGENTA, width=1.5, opacity=0)
        self.trunk_rest = [(ln.get_start().copy(), ln.get_end().copy()) for ln in self.trunks]
        self.msgs = streak_mob(MAGENTA, 5)
        self.ring = ShockRing(self, T_DROP, AC, 0.3, 5.0, 8.0, 0.9, 0.55, INK, ease_out_cubic)

        # ---------------- outline + COLLAPSE (captured now from the full outline) ----------------
        self.outline_full = RoundedRectangle(width=5.8, height=5.8, corner_radius=0.3, stroke_color=CYAN,
                                             stroke_width=3, fill_color=CYAN, fill_opacity=0).move_to(AC)
        self.outline = self.outline_full.copy()
        self.collapse = COLLAPSE(self.outline, None, "cluster")
        self.col_s1 = COLLAPSE_SIZE / 5.8
        self.outline.set_stroke(opacity=0)

        # ---------------- text ----------------
        track = TextTrack(self)
        track.on(kick("SCALE 5/6 · DATA CENTER"), 0.20, T_TEXT_OUT)
        c1 = card([("No shared", "H84"), ("memory.", "H84")])
        c2 = card([("They pass", "H84"), ("messages.", "H84", MAGENTA)])
        c3 = card([("100,000+", "H96", CYAN), ("GPUs, training one AI.", "S56", INK)])
        self.cards = (c1, c2, c3)
        track.on(c1[0], 0.25, 1.65)
        track.on(c1[1], 0.25, 1.65)
        track.on(c2[0], 1.70, 3.10)
        track.on(c2[1], 1.70, 3.10)
        track.on(c3[0], T_DROP, T_TEXT_OUT, enter="slam", about=[X_C, CARD_TOP, 0])   # grows down, clear of the kicker
        track.on(c3[1], 3.50, T_TEXT_OUT)

        # ---------------- HUD ----------------
        hud.roll(T_ROLL0, T_ROLL1, 135168, 1689600000, ease=ease_in_quad)
        # 1.18 (not 1.3): the 6.9-wide number would overshoot the 9-unit frame at 1.3
        hud.slam(T_DROP, "1,689,600,000+", mult="×12,500", scale=1.18)

        # ---------------- z-order: dust, ring, world, hall, fx, collapse, text, HUD ----------------
        self.remove(st.token)                 # the hero server draws the token now
        add_behind(self, self.hall, self.trunks, self.world, self.dots_m, self.dots_l, self.msgs,
                   self.ring, self.outline, self.collapse.glyph)
        entry_ring(self)
        self.add(track)
        self.remove(hud)
        self.add(hud)
        self._draw(0.0)
        driver = Mobject()
        driver.add_updater(lambda m, dt: self._draw(self.now))
        self.add(driver)

        # ---------------- audio ----------------
        self.cue("hit", gain=0.8, note="match-cut and ring")
        self.cue("whoosh", dt=0.40, dur=0.8, direction="up", note="rack zoom-out")
        for c0 in SYNC:
            self.cue("ticks_burst", dt=c0 + 0.10, dur=0.35, rate0=18, rate1=30, note="messages going up")
            self.cue("ticks_burst", dt=c0 + 0.45, dur=0.33, rate0=30, rate1=18, note="updates coming down")
            self.cue("pop", dt=c0 + 0.78, note="same update everywhere")
        self.cue("ticks_burst", dt=T_ROLL0, dur=0.72, rate0=6, rate1=40, note="odometer accelerating")
        self.cue("riser", dt=2.70, dur=0.62, note="steep, ends at the gap")
        self.cue("whoosh", dt=2.90, dur=0.9, direction="up", note="hall zoom-out")
        self.cue("silence", dt=T_FREEZE, dur=0.08, note="the gap before the drop")
        self.cue("hit", dt=T_DROP, gain=1.0, note="the biggest drop of the film")
        self.cue("sub_drop", dt=T_DROP, dur=1.4)
        self.cue("pad_swell", dt=T_DROP, dur=1.6, note="euphoric lift")
        self.cue("shimmer", dt=3.60, dur=1.3, note="rides the all-reduce pulses")
        for _, t0, _ in PULSES:
            self.cue("tick", dt=t0, gain=0.5, note="gather/broadcast")
        self.cue("whoosh", dt=5.47, dur=0.62, direction="down", note="collapse")
        self.cue("tick", dt=T_COL1, gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_DROP)
        # stage A has cross-faded into the hall: drop it (and its packets) from the scene
        self.stage_a_on = False
        self.remove(self.world, self.dots_m, self.dots_l)
        hold_floor(self, T_SET1)
        exit_to(self)                         # canonical E06, static to 6.00
        print(f"[b06] hall buckets used (max): {self.hall.max_used}")

    # ------------------------------------------------------------------
    def _draw(self, now: float) -> None:
        if self.stage_a_on:
            self._draw_stage_a(now)
        self._draw_hall(now)
        self._draw_outline(now)

    # ---------------- stage A ----------------
    def _zoom2(self, now: float) -> float:
        if now >= T_FREEZE - EPS:
            return 1.0
        return ease_in_out_cubic(ramp(now, T_Z0, T_Z1))

    def _draw_stage_a(self, now: float) -> None:
        rig = self.rig
        if now < T_EXP - EPS:
            EXPAND(self.world, SLOT, SRV_U, ease_out_expo(now / T_EXP))
            # stage window on everything but the hero (which is the token at e = 0): rack lines
            # are clipped, every other member fades as an edge leaves the window
            hero_ix = self.ix_hero
            for i, r in enumerate(rig.rest):
                if i in hero_ix:
                    continue
                m = r.m
                if i in self.ix_rack_set:
                    m.points[:, 1] = np.clip(m.points[:, 1], -2.40, 3.30)
                    continue
                k = edge_mask(m.points[:, 1].min(), m.points[:, 1].max())
                if k < 1.0:
                    m.fill_rgbas = m.fill_rgbas * [1, 1, 1, k]
                    m.stroke_rgbas = m.stroke_rgbas * [1, 1, 1, k]
            self.dots_m.points = np.zeros((0, 3))
            self.dots_l.points = np.zeros((0, 3))
            return

        e2 = self._zoom2(now)
        z = ZOOM2 ** (-e2)
        b = np.array([X_C, 0.45 * e2, 0.0]) - z * PIVOT_A
        a_net = 1.0 - ramp(e2, 0.25, 0.65)
        a_srv = 1.0 - ramp(e2, 0.75, 1.0)
        boost = ramp(e2, 0.30, 0.75)                     # servers turn solid as they shrink

        # cycle state
        cyan_f = lime_f = tor_f = spine_f = rack_glow = up_glow = 0.0
        for c0 in SYNC:
            cyan_f = max(cyan_f, decay(now - c0, 0.25))
            lime_f = max(lime_f, decay(now - (c0 + 0.78), 0.32))
            tor_f = max(tor_f, decay(now - (c0 + 0.30), 0.22), 0.6 * decay(now - (c0 + 0.60), 0.20))
            spine_f = max(spine_f, decay(now - (c0 + 0.45), 0.28))
            rack_glow = max(rack_glow, ramp(now, c0 + 0.08, c0 + 0.14) * (1 - ramp(now, c0 + 0.28, c0 + 0.40)),
                            ramp(now, c0 + 0.58, c0 + 0.64) * (1 - ramp(now, c0 + 0.76, c0 + 0.90)))
            up_glow = max(up_glow, ramp(now, c0 + 0.28, c0 + 0.33) * (1 - ramp(now, c0 + 0.58, c0 + 0.66)))

        for i, r in enumerate(rig.rest):
            m = r.m
            m.points = r.p * z + b
            m.stroke_width = r.sw * z

        def setc(ix, fill_rgb, fill_a, stroke_rgb, stroke_a):
            for i in ix:
                m = rig.rest[i].m
                yc = 0.5 * (m.points[:, 1].min() + m.points[:, 1].max())
                k = float(stage_mask(yc))
                m.fill_rgbas = np.array([[*fill_rgb, fill_a * k]])
                m.stroke_rgbas = np.array([[*stroke_rgb, stroke_a * k]])

        # servers: CYAN compute flash, LIME update flash, solid as they shrink, cross-fade out
        f_rgb = mix(RGB_CYAN, RGB_LIME, lime_f)[0]
        f_a = max(lerp(0.18, 1.0, cyan_f), lerp(0.18, 0.85, lime_f), lerp(0.18, POD_OP, boost)) * a_srv
        s_rgb = mix(mix(RGB_CYAN, RGB_LIME, lime_f)[0], RGB_INK, 0.6 * cyan_f)[0]
        setc(self.ix_srv, f_rgb, f_a, s_rgb, a_srv * (1 - 0.8 * boost))
        setc(self.ix_glyph, RGB_CYAN, 0.0, RGB_CYAN, 0.0)
        setc(self.ix_rack, RGB_MAG, 0.0, RGB_MAG, lerp(0.35, 0.85, rack_glow) * a_net)
        setc(self.ix_up, RGB_MAG, 0.0, RGB_MAG, lerp(0.30, 0.75, up_glow) * a_net)
        setc(self.ix_tor, mix(RGB_MAG, RGB_INK, 0.5 * tor_f)[0], lerp(0.30, 1.0, tor_f) * a_net,
             mix(RGB_MAG, RGB_INK, tor_f)[0], a_net)
        setc(self.ix_spine, mix(RGB_MAG, RGB_INK, 0.5 * spine_f)[0], lerp(0.35, 1.0, spine_f) * a_net,
             mix(RGB_MAG, RGB_INK, spine_f)[0], a_net)

        # packets (only during the SYNC cycles, before the zoom-out)
        up_pts, dn_pts = [], []
        for c0 in SYNC:
            up_pts += self._packets(now, c0, up=True)
            dn_pts += self._packets(now, c0, up=False)
        for mob, segs in ((self.dots_m, up_pts), (self.dots_l, dn_pts)):
            if segs:
                A = np.concatenate([s[0] for s in segs]) * z + b
                B = np.concatenate([s[1] for s in segs]) * z + b
                mob.points = seg_points(A, B)
                mob.stroke_width = 8 * z
                mob.set_stroke(opacity=a_net)
            else:
                mob.points = np.zeros((0, 3))

    @staticmethod
    def _packets(now: float, c0: float, up: bool):
        """[(tails (n,3), heads (n,3))] of the packets in flight for cycle c0 (world coords).
        Rack legs run as a train at one constant speed: going up, all leave their servers at
        c0+0.10 and are swallowed by the ToR one after another (last at c0+0.30); coming down
        is the exact time-reverse, so all 64 land on their servers together at c0+0.78."""
        out = []
        top = np.stack([SRV_C[:, 0], np.full(64, RACK_TOP), np.zeros(64)], axis=1)
        dist = RACK_TOP - SRV_C[:, 1]
        v = dist.max() / 0.20
        if up and c0 + 0.10 - EPS <= now < c0 + 0.30 + PKT_LEN / v:
            head = np.clip((now - c0 - 0.10) * v / dist, 0, 1)
            tail = np.clip((now - c0 - 0.10) * v / dist - PKT_LEN / dist, 0, 1)
            live = tail < 1
            out.append(streak(SRV_C[live], top[live], tail[live], head[live]))
        if not up and c0 + 0.60 - EPS <= now < c0 + 0.78 - EPS:
            head = 1 - np.clip((c0 + 0.78 - now) * v / dist, 0, 1)      # 0 = ToR, 1 = server
            tail = np.clip(head - PKT_LEN / dist, 0, 1)
            live = head > 0
            if np.any(live):
                out.append(streak(top[live], SRV_C[live], tail[live], head[live]))
        tor = np.array([[RACK_X[c], UP_Y0, 0.0] for c in range(8) for s in range(4)])
        spn = np.array([[SPINE_X[s], UP_Y1, 0.0] for c in range(8) for s in range(4)])
        L = np.hypot(*(spn - tor)[:, :2].T)
        for t0, t1, P0, P1, go in ((c0 + 0.30, c0 + 0.45, tor, spn, up), (c0 + 0.45, c0 + 0.60, spn, tor, not up)):
            if go and t0 - EPS <= now < t1 - EPS:
                q = ease_in_out_sine(clip01((now - t0) / (t1 - t0)))
                qd = q * L
                out.append(streak(P0, P1, np.clip(qd - PKT_LEN * 0.8, 0, None) / L, np.full(32, q)))
        return out

    # ---------------- stage B ----------------
    def _hall_cam(self, now: float):
        """(k, b, shake) with screen = k*q + b for hall-world points q."""
        e2 = self._zoom2(now)
        k = ZOOM2 ** (1 - e2)
        b = np.array([X_C, 0.45 * e2, 0.0]) - k * AC
        sh = shake(now - T_DROP, 0.06, 0.25, seed=6)
        return k, b + sh, e2

    def _draw_hall(self, now: float) -> None:
        if now < T_Z0 - EPS:
            self.hall.clear()
            for ln in self.trunks:
                ln.set_stroke(opacity=0)
            self.msgs.points = np.zeros((0, 3))
            return
        k, b, e2 = self._hall_cam(now)
        # collapse: contents scale about AC (C = AC) by s = (1.128/5.8)^e and fade over e in [.45, .8]
        ec = 0.0
        if now >= T_COL0 - EPS:
            ec = 1.0 if now >= T_COL1 - EPS else ease_in_out_expo(ramp(now, T_COL0, T_COL1))
        sc = self.col_s1 ** ec
        a_col = 1.0 - ramp(ec, 0.45, 0.80)
        if a_col <= 0:
            self.hall.clear()
            for ln in self.trunks:
                ln.set_stroke(opacity=0)
            self.msgs.points = np.zeros((0, 3))
            return
        kk = k * sc
        bb = (b - AC) * sc + AC if ec > 0 else b
        C = HALL_C * kk + bb
        n = len(HALL_C)

        # ---- colour + opacity per square
        if now < T_DROP - EPS:
            a = np.where(POD, POD_OP * ramp(e2, 0.75, 1.0), 0.25 * ramp(e2, 0.20, 0.80))
            rgb = np.tile(RGB_CYAN, (n, 1))
            a = a * HALL_VAR
        else:
            d = now - T_DROP
            ink = np.full(n, 1.0 - ease_out_cubic(ramp(d, 0.0, 0.20)))
            if d < 0.20:
                base = np.full(n, lerp(1.0, 0.9, ease_out_cubic(d / 0.20)))
            else:
                base = np.full(n, lerp(0.9, 0.55, ease_in_out_sine(ramp(now, 3.60, 3.95))))
            mag = np.zeros(n)
            for kind, t0, t1 in PULSES:
                if now < t0 - EPS:
                    continue
                if kind == "gather":
                    if now < t1 + 0.12:
                        rho = 4.0 * (1.0 - ease_in_quad(ramp(now, t0, t1)))
                        band = np.clip(1.0 - np.abs(HALL_D - rho) / 0.25, 0, 1) * (1 - ramp(now, t1, t1 + 0.12))
                        mag = np.maximum(mag, band)
                else:
                    td = t0 + (t1 - t0) * HALL_D / 4.0          # front passes square at td
                    passed = now >= td - EPS
                    fl = np.where(passed, 1.0 - ease_out_quad_arr(np.clip((now - td) / 0.085, 0, 1)), 0.0)
                    ink = np.maximum(ink, fl)
                    base = np.where(passed, np.maximum(base, 0.8), base)
            rgb = mix(mix(RGB_CYAN, RGB_MAG, mag), RGB_INK, ink)
            a = np.maximum(base, np.maximum(mag * 0.95, ink)) * HALL_VAR
            a = np.maximum(a, ink)                       # flashes read at full strength
        a = a * stage_mask(C[:, 1]) * a_col
        self.hall.draw(C, HALL_SQ * kk, np.concatenate([rgb, a[:, None]], axis=1))

        # ---- trunks
        if now < T_DROP - EPS:
            ta = 0.30 * ramp(e2, 0.6, 1.0)
        else:
            d = now - T_DROP
            ta = lerp(0.9, 0.5, ease_out_cubic(ramp(d, 0.0, 0.40)))
        ta *= a_col
        for ln, (p0, p1) in zip(self.trunks, self.trunk_rest):
            q0, q1 = p0 * kk + bb, p1 * kk + bb
            if abs(q0[0] - q1[0]) < 1e-9:                 # vertical: clip to the stage window
                lo, hi = max(min(q0[1], q1[1]), -2.50), min(max(q0[1], q1[1]), 3.40)
                if hi <= lo or ta <= 0:
                    ln.set_stroke(opacity=0)
                    continue
                ln.put_start_and_end_on(np.array([q0[0], lo, 0]), np.array([q0[0], hi, 0]))
                ln.set_stroke(MAGENTA, width=1.5 * min(kk, 1.6), opacity=ta)
            else:
                mk = float(stage_mask(q0[1]))
                lo, hi = max(min(q0[0], q1[0]), -4.7), min(max(q0[0], q1[0]), 4.7)
                if hi <= lo or ta * mk <= 0:
                    ln.set_stroke(opacity=0)
                    continue
                ln.put_start_and_end_on(np.array([lo, q0[1], 0]), np.array([hi, q0[1], 0]))
                ln.set_stroke(MAGENTA, width=1.5 * min(kk, 1.6), opacity=ta * mk)

        # ---- messages along the trunks (3.40 - 5.05)
        if T_DROP - EPS <= now < T_MSG_END:
            env = ramp(now, T_DROP, T_DROP + 0.10) * (1 - ramp(now, 4.90, T_MSG_END))
            s = (now - T_DROP) * MSG_SPEED + MSG_PHASE
            head = self._walk_pos(s)
            tail = self._walk_pos(np.maximum(s - MSG_TAIL, MSG_PHASE))
            H = np.concatenate([head, np.zeros((N_MSG, 1))], axis=1) * kk + bb
            T = np.concatenate([tail, np.zeros((N_MSG, 1))], axis=1) * kk + bb
            self.msgs.points = seg_points(T, H)
            self.msgs.set_stroke(MAGENTA, width=5 * kk, opacity=0.95 * env * a_col)
        else:
            self.msgs.points = np.zeros((0, 3))

    @staticmethod
    def _walk_pos(s: np.ndarray) -> np.ndarray:
        e = np.floor(s / EDGE).astype(int)
        e = np.clip(e, 0, WALK_XY.shape[1] - 2)
        f = np.clip(s / EDGE - e, 0, 1)[:, None]
        r = np.arange(N_MSG)
        return WALK_XY[r, e] * (1 - f) + WALK_XY[r, e + 1] * f

    # ---------------- outline + collapse ----------------
    def _draw_outline(self, now: float) -> None:
        if now < T_OUT0 - EPS:
            self.outline.set_stroke(opacity=0)
            return
        if now < T_COL0 - EPS:
            a = ease_out_cubic(ramp(now, T_OUT0, T_OUT1))
            self.outline.pointwise_become_partial(self.outline_full, 0, max(a, 1e-3))
            self.outline.set_stroke(CYAN, width=3, opacity=1)
            self.outline.set_fill(CYAN, opacity=0)
            return
        if now < T_COL1 - EPS:
            self.collapse.at(ease_in_out_expo(ramp(now, T_COL0, T_COL1)))
        else:
            self.collapse.settle(ramp(now, T_COL1, T_SET1))

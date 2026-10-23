"""b06 - SCALE 5/6 - DATA CENTER: separate boxes, no shared memory: the servers pass messages.

The node token (E05) is server (c3, r3) of one pod (8 racks x 8 servers, 5.36 units wide):
the camera pulls back (EXPAND, hero outline INK-flashes on story frames 1-2). 'Many boxes. /
No shared memory.' POPs on the cut. Two SYNC cycles run over the pod's network: every server
computes on its own data (CYAN flash), then 16 MAGENTA messages (dots with trails) hop like
letters: up the sender's rack line into its top-of-rack switch, over an uplink into a spine,
down another uplink into a DIFFERENT rack's ToR and down to a server there (each switch
flashes as a message passes, the receiver flashes INK); then every server applies the same
update (LIME flash). 'They pass / messages.' hard-swaps in on the second cycle.

At 2.45 a second pull-back shrinks the pod into 16 squares of a 56 x 56 hall (3,136 squares =
100,352 GPUs; pitch 0.12, the hall overfills the stage window top and bottom) while the HUD
log-rolls and accelerates toward 1,689,600,000 but is CUT at ~281,000,000 at 3.32: 80 ms freeze
+ silence (the headline cuts out too), then the biggest drop of the film at 3.40: HUD SLAMs
1,689,600,000+ with the word equation '100,000+ GPUs × 16,896' in the label slot and the card
'100,000+' SLAMs in (INK on its first two story frames, then CYAN; start scale capped so it
never leaves the safe zone x[-4.0, 3.3]: ~1.17, not 1.3). The punch comes from the stage: the
hall flashes INK, a radial INK burst (0.55 peak + a faint frame tint), a CYAN shock ring (w14,
r 0.3 -> 7.0) and a 0.16 impact shake; messages race along the trunks and gather/broadcast
all-reduce pulses sweep the hall (the last gather pulls into the centre). 5.20 an outline draws
on around the hall window, which COLLAPSEs (cut_times) into handoff_token('cluster') (E06). The
kicker and the card hold, untouched, through the collapse, the landing and the settle, then
HARD-CUT at the exit (T_EXIT 5.95, no fade; only the 3 static E06 frames are text-less, as in
b02-b05): b07's text POPs on its own cut.

Everything that moves is a pure function of self.now (one driver). The hall is a motif.Field
with fixed quantisation (temporally stable colours); off-frame squares are skipped.
"""

from style import *
from motif import *
import math

# --------------------------------------------------------------------------
# Stage A: one pod (8 racks x 8 servers)
# --------------------------------------------------------------------------
SRV_U = 0.46
SRV_PX, SRV_PY = 0.70, 0.56                               # rack pitch (x), server pitch in a rack (y)
RACK_X = X_C + (np.arange(8) - 3.5) * SRV_PX             # rack c = 0..7: x[-2.80, 2.10]
ROW_Y = 1.93 - np.arange(8) * SRV_PY                     # row r = 0..7 (top = 0): y[-1.99, 1.93]
SRV_C = np.array([[RACK_X[k % 8], ROW_Y[k // 8], 0.0] for k in range(64)])
HERO = 3 * 8 + 3                                         # (c3, r3) at (-0.70, 0.25)
SLOT = SRV_C[HERO].copy()
RACK_BOT = ROW_Y[7] - SRV_U / 2                          # -2.22
TOR_W, TOR_H = 0.54, 0.18
TOR_Y = ROW_Y[0] + SRV_U / 2 + 0.10 + TOR_H / 2          # 2.35
RACK_TOP = TOR_Y - TOR_H / 2                             # 2.26 (rack lines end in the ToR)
TOR_TOP = TOR_Y + TOR_H / 2                              # 2.44
SPINE_W, SPINE_H = 1.00, 0.24
SPINE_Y = 3.00                                           # spines y[2.88, 3.12]
SPINE_BOT = SPINE_Y - SPINE_H / 2
SPINE_X = X_C + (np.arange(4) - 1.5) * 1.40
PIVOT_A = np.array([X_C, (ROW_Y[0] + ROW_Y[7]) / 2, 0.0])   # server-block centre (zoom-out 2 pivot)

# --------------------------------------------------------------------------
# Stage B: the hall (56 x 56 squares = 3,136; 1 square = 4 servers = 32 GPUs)
# --------------------------------------------------------------------------
HN = 56
HP = 0.12                                                # hall pitch: x[-3.74, 3.03]
_I, _J = np.meshgrid(np.arange(HN), np.arange(HN), indexing="ij")   # i = column, j = row
HI, HJ = _I.ravel(), _J.ravel()
HALL_C = np.stack([X_C + (HI - 27.5) * HP, AC[1] + (27.5 - HJ) * HP, np.zeros(HN * HN)], axis=1)
HALL_D = np.hypot(HALL_C[:, 0] - AC[0], HALL_C[:, 1] - AC[1])
POD = (HI >= 26) & (HI <= 29) & (HJ >= 26) & (HJ <= 29)            # where the pod lands
HALL_SQ = 0.7 * HP
HALL_UNIT = RoundedRectangle(width=1.0, height=1.0, corner_radius=0.012 / 0.07).points.copy()
HALL_VAR = 0.80 + 0.20 * np.random.default_rng(61).random(HN * HN)  # per-square brightness texture
HALL_HALF = 27.5 * HP + HALL_SQ / 2
# The hall overfills the stage vertically: it is seen through a window (soft 0.16 edges) that
# the collapse outline frames exactly.
WIN_LO, WIN_HI, WIN_SOFT = -2.30, 3.20, 0.16             # centred on AC; 0.2 clear of the HUD digits
OUT_W, OUT_H = 6.90, WIN_HI - WIN_LO                     # outline 6.9 x 5.5 centred on AC
ZX = SRV_PX / (HP / 2)                                   # 11.667: a rack pitch -> half a hall pitch
ZY = SRV_PY / (HP / 2)                                   # 9.333 (2 x 2 servers land on one square)
TRUNK_OFF = np.array([-5, -3, -1, 1, 3, 5]) * 4 * HP    # +-0.48, 1.44, 2.40: in the gutters
TRUNK_X = X_C + TRUNK_OFF
TRUNK_Y = AC[1] - TRUNK_OFF
POD_OP = 0.60                                            # the pod's 16 squares right after the zoom

# --------------------------------------------------------------------------
# Timeline (beat-local seconds)
# --------------------------------------------------------------------------
T_EXP = 0.80
SYNC = (0.85, 1.65)
T_CARD2 = 1.75                                           # 'They pass / messages.' hard swap
T_Z0, T_Z1 = 2.45, 3.35
T_ROLL0 = 2.60
T_FREEZE = 3.32
T_DROP = 3.40
PULSES = (("gather", 3.60, 3.90), ("bcast", 3.95, 4.25), ("gather", 4.35, 4.65), ("bcast", 4.70, 5.00),
          ("gather", 5.05, 5.35))
T_MSG_END = 5.35
T_OUT0, T_OUT1 = 5.20, 5.40                              # outline draws on
T_COL, T_SET, T_EXIT = cut_times(6.0)                    # 5.433 / 5.833 / 5.95
T_TEXT_OUT = T_EXIT                                      # 5.95: kicker + card hold to the exit, then hard-cut
SLAM_BOX = Box(SAFE_LEFT + 0.05, SAFE_RIGHT - 0.05, SAFE_BOTTOM, SAFE_TOP)   # '100,000+' slam stays in here
SLAM_INK = 2 / 60                                        # the slammed card is INK for 2 story frames
DROP_SHAKE = (0.16, 0.30)                                # hall impact shake: amplitude, duration

# --------------------------------------------------------------------------
# Stage-A messages: 16 per SYNC cycle, each hops server -> ToR -> spine -> ToR' -> server' in a
# different rack. Phases: move, dwell (ToR), move, dwell (spine), move, dwell (ToR'), move.
# --------------------------------------------------------------------------
N_HOP = 16
HOP = np.array([0.12, 0.035, 0.10, 0.035, 0.10, 0.035, 0.12])
HOP_EDGE = np.concatenate([[0.0], np.cumsum(HOP)])       # phase starts (8 values; last = arrival)
HOP_MOVE = np.array([1, 0, 1, 0, 1, 0, 1], bool)
TRAIL = 0.30                                             # message trail length (units)


def _hops():
    rng = np.random.default_rng(606)
    t0, src, dst, sp, paths = [], [], [], [], []
    for c0 in SYNC:
        rows = [rng.permutation(8)[:2] for _ in range(8)]     # 2 senders per rack, distinct rows
        for m in range(N_HOP):
            cs, rs = m % 8, int(rows[m % 8][m // 8])
            cd = int((cs + rng.integers(1, 8)) % 8)            # always a different rack
            rd = int(rng.integers(0, 8))
            s = int(rng.integers(0, 4))
            ts = c0 + 0.05 + 0.10 * rng.random()
            xs, xd, xp = RACK_X[cs], RACK_X[cd], SPINE_X[s]
            paths.append([(xs, ROW_Y[rs]), (xs, TOR_Y), (xs, TOR_TOP), (xp, SPINE_BOT), (xp, SPINE_Y),
                          (xp, SPINE_BOT), (xd, TOR_TOP), (xd, TOR_Y), (xd, ROW_Y[rd])])
            t0.append(ts)
            src.append((cs, rs))
            dst.append((cd, rd))
            sp.append(s)
    P = np.array(paths, float)                                   # (n, 9, 2)
    seg = np.hypot(*(P[:, 1:] - P[:, :-1]).transpose(2, 0, 1))   # (n, 8)
    cum = np.concatenate([np.zeros((len(P), 1)), np.cumsum(seg, axis=1)], axis=1)   # (n, 9)
    # arc length at each phase boundary: A ends at point 1, B at 4, C at 7, D at 8
    s_lo = cum[:, [0, 1, 1, 4, 4, 7, 7]]
    s_hi = cum[:, [1, 1, 4, 4, 7, 7, 8]]
    return np.array(t0), np.array(src), np.array(dst), np.array(sp), P, cum, s_lo, s_hi


HOP_T0, HOP_SRC, HOP_DST, HOP_SP, HOP_P, HOP_CUM, HOP_SLO, HOP_SHI = _hops()
N_HOPS = len(HOP_T0)
assert np.all(HOP_T0 + HOP_EDGE[-1] < np.repeat(SYNC, N_HOP) + 0.74)      # all land before the LIME flash

# --------------------------------------------------------------------------
# Hall messages on the trunks: 120 seeded random walks on the 6 x 6 trunk lattice
# --------------------------------------------------------------------------
N_MSG = 120
MSG_SPEED = 3.6
EDGE = 0.96
_mrng = np.random.default_rng(66)


def _walks() -> np.ndarray:
    steps = 14
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


WALK = _walks()                                          # (120, 15, 2) lattice indices
WALK_XY = np.stack([TRUNK_X[WALK[..., 0]], TRUNK_Y[WALK[..., 1]]], axis=-1)
MSG_PHASE = _mrng.random(N_MSG) * EDGE                    # start partway along the first edge
MSG_TAIL = 0.17


def stage_mask(y):
    """Soft stage window for stage A: 1 for y in [-2.37, 3.25], 0 below -2.62 / above 3.55, so
    camera moves never drag geometry under the text zone or the HUD. Identity at rest."""
    y = np.asarray(y, float)
    return np.clip((y + 2.62) / 0.25, 0, 1) * np.clip((3.55 - y) / 0.30, 0, 1)


def hall_mask(y):
    """The hall window (camera space, before the collapse): full inside [WIN_LO+soft, WIN_HI-soft]."""
    y = np.asarray(y, float)
    return np.clip((y - WIN_LO) / WIN_SOFT, 0, 1) * np.clip((WIN_HI - y) / WIN_SOFT, 0, 1)


def edge_mask(bot: float, top: float) -> float:
    """Stage window for a whole member: fades as its bottom edge goes under -2.45 (HUD number
    top is -2.52) or its top edge over 3.40 (cards end at 3.48)."""
    return clip01((bot + 2.62) / 0.17) * clip01((3.58 - top) / 0.18)


def win(d: float, a: float, b: float, tail: float) -> float:
    """1 on [a, b), then easing out over `tail`; 0 before a."""
    if d < a - EPS:
        return 0.0
    if d < b - EPS:
        return 1.0
    return decay(d - b, tail)


class B06DataCenter(BeatScene):
    def beat(self):
        # ---------------- entry: E05 (node token + HUD '135,168') ----------------
        st = enter(self)
        hud = st.hud

        # ---------------- stage A at rest ----------------
        self.racks = VGroup(*[Line([x, RACK_TOP, 0], [x, RACK_BOT, 0], stroke_color=MAGENTA,
                                   stroke_width=1.5, stroke_opacity=0.35) for x in RACK_X])
        self.uplinks = VGroup(*[Line([RACK_X[c], TOR_TOP, 0], [SPINE_X[s], SPINE_BOT, 0], stroke_color=MAGENTA,
                                     stroke_width=1.2, stroke_opacity=0.30)
                                for c in range(8) for s in range(4)])
        self.tors = VGroup(*[RoundedRectangle(width=TOR_W, height=TOR_H, corner_radius=0.05, stroke_color=MAGENTA,
                                              stroke_width=2, fill_color=MAGENTA, fill_opacity=0.30)
                             .move_to([x, TOR_Y, 0]) for x in RACK_X])
        self.spines = VGroup(*[RoundedRectangle(width=SPINE_W, height=SPINE_H, corner_radius=0.07,
                                                stroke_color=MAGENTA, stroke_width=2.5, fill_color=MAGENTA,
                                                fill_opacity=0.35).move_to([x, SPINE_Y, 0]) for x in SPINE_X])
        srv = []
        for k in range(64):
            if k == HERO:
                self.hero = unit_with_glyph("node", SRV_U, SRV_C[k], CYAN, fill=0.18, stroke=2.5)
                srv.append(self.hero)
            else:
                srv.append(unit(SRV_U, CYAN, 0.18, 2.5).move_to(SRV_C[k]))
        self.servers = VGroup(*srv)
        self.srv_units = [s.unit if k == HERO else s for k, s in enumerate(srv)]
        self.world = VGroup(self.racks, self.uplinks, self.tors, self.spines, self.servers)
        rig = EXPAND(self.world, SLOT, SRV_U, 0.0, hero=self.hero, token="node",
                     fades={self.hero.glyph: (0.0, 0.35, 1.0, 0.0)})   # captures rest, draws e = 0
        self.rig = rig
        self.ix_srv = [rig.index[id(u)] for u in self.srv_units]
        self.ix_glyph = rig.indices_of(self.hero.glyph)
        self.ix_rack = rig.indices_of(self.racks)
        self.ix_up = rig.indices_of(self.uplinks)
        self.ix_tor = rig.indices_of(self.tors)
        self.ix_spine = rig.indices_of(self.spines)
        self.ix_hero = set(rig.indices_of(self.hero))
        self.ix_rack_set = set(self.ix_rack)
        self.ix_line = set(self.ix_rack) | set(self.ix_up)
        # rest centre of every member (zoom-out 2 moves boxes by their centres, so they stay square)
        self.rest_c = [0.5 * (r.p.min(axis=0) + r.p.max(axis=0)) for r in rig.rest]
        self.stage_a_on = True

        # stage-A messages: trail, head, hot core (one multi-subpath stroke VMobject each)
        self.m_trail = streak_mob(MAGENTA, 8)
        self.m_head = streak_mob(MAGENTA, 18)
        self.m_core = streak_mob(INK, 7)

        # ---------------- stage B: hall, trunks, messages ----------------
        self.hall = Field(HALL_UNIT, pool=256, qc=16, qa=24, fixed=True)
        self.trunks = VGroup(*[Line([x, AC[1] - HALL_HALF, 0], [x, AC[1] + HALL_HALF, 0]) for x in TRUNK_X],
                             *[Line([X_C - HALL_HALF, y, 0], [X_C + HALL_HALF, y, 0]) for y in TRUNK_Y])
        for ln in self.trunks:
            ln.set_stroke(MAGENTA, width=1.5, opacity=0)
        self.trunk_rest = [(ln.get_start().copy(), ln.get_end().copy()) for ln in self.trunks]
        self.msgs = streak_mob(MAGENTA, 6)
        self.ring = ShockRing(self, T_DROP, AC, 0.3, 7.0, 14.0, 0.9, 0.55, CYAN, ease_out_cubic)
        # the drop's punch lives on the stage (the card slam is capped to the safe zone)
        self.flash = RadialFlash(self, T_DROP, AC, peak=0.55, radius=6.5, decay=0.12, tint=0.04)

        # ---------------- outline + COLLAPSE (captured now from the full outline) ----------------
        self.outline_full = RoundedRectangle(width=OUT_W, height=OUT_H, corner_radius=0.3, stroke_color=CYAN,
                                             stroke_width=3, fill_color=CYAN, fill_opacity=0).move_to(AC)
        self.outline = self.outline_full.copy()
        self.collapse = COLLAPSE(self.outline, None, "cluster")
        self.outline.set_stroke(opacity=0)

        # ---------------- text ----------------
        track = TextTrack(self)
        # the kicker and headline arrive on the cut (frame 0 stays the exact E05) and hold
        # through the collapse; every swap and the final exit are hard cuts (never a crossfade)
        track.on(kick("SCALE 5/6 · DATA CENTER"), 0.0, T_TEXT_OUT, exit="cut")
        c1 = card([("Many boxes.", "H96", INK), ("No shared memory.", "S56", INK)])
        c2 = card([("They pass", "H84"), ("messages.", "H84", MAGENTA)])
        c3 = card([("100,000+", "H96", CYAN), ("GPUs, training one AI.", "S56", INK)])
        self.cards = (c1, c2, c3)
        assert SPINE_Y + SPINE_H / 2 < min(c1.get_bottom()[1], c2.get_bottom()[1]) - 0.3   # switch tier has air
        track.on(c1[0], 1 / 60, T_CARD2, enter="pop", exit="cut")
        track.on(c1[1], 0.12, T_CARD2, exit="cut")
        track.on(c2[0], T_CARD2, T_FREEZE, enter="pop", exit="cut")     # the freeze blanks the headline
        track.on(c2[1], T_CARD2, T_FREEZE, enter="pop", exit="cut")
        # '100,000+' SLAMs (grows down from its top edge, clear of the kicker) with the start
        # scale capped to the safe zone; an INK twin carries the first 2 story frames (hit flash)
        about = np.array([X_C, CARD_TOP, 0.0])
        s_slam = fit_scale(c3[0], 1.3, about, SLAM_BOX)
        c3_ink = c3[0].copy().set_color(INK)
        track.on(c3_ink, T_DROP, T_DROP + SLAM_INK, enter="slam", exit="cut", about=about, scale=s_slam)
        track.on(c3[0], T_DROP, T_TEXT_OUT, enter="slam", exit="cut", about=about, scale=s_slam,
                 alpha=lambda now: 0.0 if now < T_DROP + SLAM_INK - EPS else 1.0)
        track.on(c3[1], 3.50, T_TEXT_OUT, exit="cut")
        lo = (c3[0].get_left() - about) * s_slam + about
        hi = (c3[0].get_right() - about) * s_slam + about
        assert SAFE_LEFT <= lo[0] and hi[0] <= SAFE_RIGHT, (lo, hi)

        # ---------------- HUD ----------------
        # The roll aims at 1,689,600,000 by the hit but is CUT at the freeze (~281,000,000): the
        # destination number first appears ON the drop frame, a 6x jump with the hit.
        hud.roll(T_ROLL0, T_DROP, 135168, 1689600000, ease=ease_in_quad)
        self.v_freeze = hud.number_at(T_FREEZE - 1e-4)[0]
        hud.cut(T_FREEZE, self.v_freeze)
        hud.slam(T_DROP, "1,689,600,000+")                  # start scale capped (1.058, safe zone)
        hud.mult(T_DROP, "100,000+ GPUs × 16,896")         # word equation in the label slot, 1.4 s

        # ---------------- z-order: dust, ring, hall, world, fx, collapse, text, HUD ----------------
        self.remove(st.token)                 # the hero server draws the token now
        add_behind(self, self.hall, self.trunks, self.world, self.m_trail, self.m_head, self.m_core,
                   self.msgs, self.ring, self.flash, self.outline, self.collapse.glyph)
        entry_ring(self)
        self.add(track)
        self.remove(hud)
        self.add(hud)
        self._draw(0.0)
        drive(self, self._draw)

        # ---------------- audio ----------------
        self.cue("hit", gain=0.8, note="match-cut and ring")
        self.cue("whoosh", dt=0.40, dur=0.8, direction="up", note="rack zoom-out")
        for c0 in SYNC:
            self.cue("ticks_burst", dt=c0 + 0.08, dur=0.32, rate0=18, rate1=30, note="messages up to the spines")
            self.cue("ticks_burst", dt=c0 + 0.40, dur=0.32, rate0=30, rate1=18, note="messages down to other racks")
            self.cue("pop", dt=c0 + 0.78, note="same update everywhere")
        self.cue("ticks_burst", dt=T_ROLL0, dur=round(T_FREEZE - T_ROLL0, 4), rate0=6, rate1=40,
                 note="odometer accelerating")
        self.cue("riser", dt=2.70, dur=0.62, note="steep, ends at the gap")
        self.cue("whoosh", dt=2.90, dur=0.9, direction="up", note="hall zoom-out")
        self.cue("silence", dt=T_FREEZE, dur=0.08, note="the gap before the drop")
        self.cue("hit", dt=T_DROP, gain=1.0, note="the biggest drop of the film")
        self.cue("sub_drop", dt=T_DROP, dur=1.4)
        self.cue("pad_swell", dt=T_DROP, dur=1.6, note="euphoric lift")
        self.cue("shimmer", dt=3.60, dur=1.6, note="rides the all-reduce pulses")
        for _, t0, _ in PULSES:
            self.cue("tick", dt=t0, gain=0.5, note="gather/broadcast")
        self.cue("whoosh", dt=round((T_COL + T_SET) / 2, 4), dur=0.45, direction="down", note="collapse")
        self.cue("tick", dt=round(T_SET, 4), gain=0.5, note="settle")

        # ---------------- run ----------------
        self.hold_until(T_DROP)
        # stage A has cross-faded into the hall: drop it (and its messages) from the scene
        self.stage_a_on = False
        self.remove(self.world, self.m_trail, self.m_head, self.m_core)
        hold_floor(self, T_EXIT)
        exit_to(self)                         # canonical E06, static to 6.00
        print(f"[b06] frozen roll value {self.v_freeze}; hall buckets used (max): {self.hall.max_used}, "
              f"overflow frames: {self.hall.overflow}")

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

    def _hop_s(self, t: float):
        """Arc length of every stage-A message along its path at time t, and its phase
        (-1 = not sent yet, 7 = arrived)."""
        d = t - HOP_T0
        ph = np.searchsorted(HOP_EDGE, d, side="right") - 1          # -1 .. 7
        p = np.clip(ph, 0, 6)
        frac = np.clip((d - HOP_EDGE[p]) / HOP[p], 0, 1)
        e = np.where(HOP_MOVE[p], ease_in_out_sine_arr(frac), 1.0)
        r = np.arange(N_HOPS)
        s = HOP_SLO[r, p] + (HOP_SHI[r, p] - HOP_SLO[r, p]) * e
        s = np.where(ph < 0, 0.0, np.where(ph > 6, HOP_CUM[:, -1], s))
        return s, ph

    @staticmethod
    def _hop_pos(s: np.ndarray) -> np.ndarray:
        """(n, 3) points at arc length s along each message's 9-point path."""
        k = np.clip((HOP_CUM <= s[:, None] + 1e-12).sum(axis=1) - 1, 0, 7)
        r = np.arange(N_HOPS)
        L = np.maximum(HOP_CUM[r, k + 1] - HOP_CUM[r, k], 1e-9)
        f = np.clip((s - HOP_CUM[r, k]) / L, 0, 1)[:, None]
        p = HOP_P[r, k] * (1 - f) + HOP_P[r, k + 1] * f
        return np.concatenate([p, np.zeros((N_HOPS, 1))], axis=1)

    def _draw_stage_a(self, now: float) -> None:
        rig = self.rig
        if now < T_EXP - EPS:
            EXPAND(self.world, SLOT, SRV_U, ease_out_expo(now / T_EXP),
                   ink=1.0 if flash_frames(now, 1 / 60, 2) else 0.0)
            # stage window on everything but the hero (which is the token at e = 0): rack lines
            # are clipped, every other member fades as an edge leaves the window
            for i, r in enumerate(rig.rest):
                if i in self.ix_hero:
                    continue
                m = r.m
                if i in self.ix_rack_set:
                    m.points[:, 1] = np.clip(m.points[:, 1], -2.40, 3.30)
                    continue
                k = edge_mask(m.points[:, 1].min(), m.points[:, 1].max())
                if k < 1.0:
                    m.fill_rgbas = m.fill_rgbas * [1, 1, 1, k]
                    m.stroke_rgbas = m.stroke_rgbas * [1, 1, 1, k]
            for mob in (self.m_trail, self.m_head, self.m_core):
                mob.points = EMPTY.copy()
            return

        # zoom-out 2: per-axis camera (x: ZX, y: ZY) so the 2 x 2 server groups land exactly on
        # the pod's hall squares at e2 = 1; boxes move by their centres (they stay square),
        # lines map affinely. Identity at e2 = 0.
        e2 = self._zoom2(now)
        zx, zy = ZX ** (-e2), ZY ** (-e2)
        piv = lerp(PIVOT_A, AC, e2)
        zv = np.array([zx, zy, 1.0])
        for i, r in enumerate(rig.rest):
            m = r.m
            if i in self.ix_line:
                m.points = piv + (r.p - PIVOT_A) * zv
            else:
                c = self.rest_c[i]
                m.points = (r.p - c) * zx + piv + (c - PIVOT_A) * zv
            m.stroke_width = r.sw * zx
        a_net = 1.0 - ramp(e2, 0.25, 0.65)
        a_srv = 1.0 - ramp(e2, 0.75, 1.0)
        boost = ramp(e2, 0.30, 0.75)                     # servers turn solid as they shrink

        # cycle state: compute flash, update flash
        cyan_f = lime_f = 0.0
        for c0 in SYNC:
            cyan_f = max(cyan_f, decay(now - c0, 0.25))
            lime_f = max(lime_f, decay(now - (c0 + 0.78), 0.32))
        # per-message glows: rack lines, uplinks, ToRs, spines, receivers
        rack_g, tor_f, up_g, spine_f, recv = np.zeros(8), np.zeros(8), np.zeros(32), np.zeros(4), np.zeros(64)
        for n in range(N_HOPS):
            d = now - HOP_T0[n]
            if d < -EPS or d > HOP_EDGE[-1] + 0.30:
                continue
            (cs, rs), (cd, rd), s = HOP_SRC[n], HOP_DST[n], HOP_SP[n]
            E = HOP_EDGE
            rack_g[cs] = max(rack_g[cs], win(d, E[0], E[1], 0.10))
            tor_f[cs] = max(tor_f[cs], win(d, E[1], E[2], 0.16))
            up_g[cs * 4 + s] = max(up_g[cs * 4 + s], win(d, E[2], E[3], 0.12))
            spine_f[s] = max(spine_f[s], win(d, E[3], E[4], 0.16))
            up_g[cd * 4 + s] = max(up_g[cd * 4 + s], win(d, E[4], E[5], 0.12))
            tor_f[cd] = max(tor_f[cd], win(d, E[5], E[6], 0.16))
            rack_g[cd] = max(rack_g[cd], win(d, E[6], E[7], 0.10))
            if d >= E[7] - EPS:
                recv[rd * 8 + cd] = max(recv[rd * 8 + cd], decay(d - E[7], 0.24))

        def put(i, fill_rgb, fill_a, stroke_rgb, stroke_a):
            m = rig.rest[i].m
            yc = 0.5 * (m.points[:, 1].min() + m.points[:, 1].max())
            k = float(stage_mask(yc))
            m.fill_rgbas = np.array([[*fill_rgb, fill_a * k]])
            m.stroke_rgbas = np.array([[*stroke_rgb, stroke_a * k]])

        # servers: CYAN compute flash, INK on receiving a message, LIME update flash, solid as
        # they shrink, cross-fade out into the hall
        base_rgb = mix(RGB_CYAN, RGB_LIME, lime_f)[0]
        for k, i in enumerate(self.ix_srv):
            rv = recv[k]
            f_rgb = mix(base_rgb, RGB_INK, 0.7 * rv)[0]
            f_a = max(lerp(0.18, 1.0, cyan_f), lerp(0.18, 0.85, lime_f), lerp(0.18, 0.95, rv),
                      lerp(0.18, POD_OP, boost)) * a_srv
            s_rgb = mix(base_rgb, RGB_INK, max(0.6 * cyan_f, rv))[0]
            put(i, f_rgb, f_a, s_rgb, a_srv * (1 - 0.8 * boost))
        for i in self.ix_glyph:
            put(i, RGB_CYAN, 0.0, RGB_CYAN, 0.0)
        for c, i in enumerate(self.ix_rack):
            put(i, RGB_MAG, 0.0, mix(RGB_MAG, RGB_INK, 0.35 * rack_g[c])[0], lerp(0.35, 0.95, rack_g[c]) * a_net)
            self.rig.rest[i].m.stroke_width = lerp(1.5, 3.0, rack_g[c]) * zx
        for j, i in enumerate(self.ix_up):
            put(i, RGB_MAG, 0.0, mix(RGB_MAG, RGB_INK, 0.35 * up_g[j])[0], lerp(0.30, 1.0, up_g[j]) * a_net)
            self.rig.rest[i].m.stroke_width = lerp(1.2, 3.0, up_g[j]) * zx
        for c, i in enumerate(self.ix_tor):
            f = tor_f[c]
            put(i, mix(RGB_MAG, RGB_INK, 0.5 * f)[0], lerp(0.30, 1.0, f) * a_net, mix(RGB_MAG, RGB_INK, f)[0], a_net)
        for s, i in enumerate(self.ix_spine):
            f = spine_f[s]
            put(i, mix(RGB_MAG, RGB_INK, 0.5 * f)[0], lerp(0.35, 1.0, f) * a_net, mix(RGB_MAG, RGB_INK, f)[0], a_net)

        # messages in flight (dot + trail + hot core); identity camera while they run
        s, ph = self._hop_s(now)
        live = (ph >= 0) & (ph <= 6)
        if np.any(live) and a_net > 0:
            s_prev, _ = self._hop_s(now - 0.035)
            s_tail = np.maximum(s_prev, s - TRAIL)
            H = self._hop_pos(s)[live]
            T = self._hop_pos(s_tail)[live]
            H = piv + (H - PIVOT_A) * zv
            T = piv + (T - PIVOT_A) * zv
            self.m_trail.points = seg_points(T, H)
            self.m_head.points = seg_points(H, H)
            self.m_core.points = seg_points(H, H)
            self.m_trail.set_stroke(MAGENTA, width=8 * zx, opacity=0.6 * a_net)
            self.m_head.set_stroke(MAGENTA, width=18 * zx, opacity=a_net)
            self.m_core.set_stroke(INK, width=7 * zx, opacity=0.9 * a_net)
        else:
            for mob in (self.m_trail, self.m_head, self.m_core):
                mob.points = EMPTY.copy()

    # ---------------- stage B ----------------
    def _hall_cam(self, now: float):
        """(k, piv, shake, e2): camera-space point = piv + k*(q - AC) + shake for hall-world q."""
        e2 = self._zoom2(now)
        k = ZX ** (1 - e2)
        piv = lerp(PIVOT_A, AC, e2)
        sh = shake(now - T_DROP, *DROP_SHAKE, seed=6)
        return k, piv, sh, e2

    def _collapse_state(self, now: float):
        """(sc, a_col): the contents' scale about AC and opacity during the COLLAPSE."""
        if now < T_COL - EPS:
            return 1.0, 1.0
        ec = 1.0 if now >= T_SET - EPS else ease_in_out_expo(ramp(now, T_COL, T_SET))
        return self.collapse.s1 ** ec, 1.0 - ramp(ec, *self.collapse.contents_fade)

    def _draw_hall(self, now: float) -> None:
        sc, a_col = self._collapse_state(now)
        if now < T_Z0 - EPS or a_col <= 0:
            self.hall.clear()
            for ln in self.trunks:
                ln.set_stroke(opacity=0)
            self.msgs.points = EMPTY.copy()
            return
        k, piv, sh, e2 = self._hall_cam(now)

        def col(P):                               # camera space -> screen (collapse about AC)
            return P if sc == 1.0 else (P - AC) * sc + AC

        C0 = piv + k * (HALL_C - AC) + sh
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
                        rho = 4.6 * (1.0 - ease_in_quad(ramp(now, t0, t1)))
                        band = np.clip(1.0 - np.abs(HALL_D - rho) / 0.30, 0, 1) * (1 - ramp(now, t1, t1 + 0.12))
                        mag = np.maximum(mag, band)
                else:
                    td = t0 + (t1 - t0) * HALL_D / 4.6          # front passes square at td
                    passed = now >= td - EPS
                    fl = np.where(passed, 1.0 - ease_out_quad_arr(np.clip((now - td) / 0.085, 0, 1)), 0.0)
                    ink = np.maximum(ink, fl)
                    base = np.where(passed, np.maximum(base, 0.8), base)
            rgb = mix(mix(RGB_CYAN, RGB_MAG, mag), RGB_INK, ink)
            a = np.maximum(base, np.maximum(mag * 0.95, ink)) * HALL_VAR
            a = np.maximum(a, ink)                       # flashes read at full strength
        a = a * hall_mask(C0[:, 1]) * a_col
        self.hall.draw(col(C0), HALL_SQ * k * sc, np.concatenate([rgb, a[:, None]], axis=1))

        # ---- trunks (clipped to the hall window in camera space, then collapsed)
        if now < T_DROP - EPS:
            ta = 0.30 * ramp(e2, 0.6, 1.0)
        else:
            ta = lerp(0.9, 0.5, ease_out_cubic(ramp(now - T_DROP, 0.0, 0.40)))
        ta *= a_col
        w = 1.5 * min(k * sc, 1.6)
        for ln, (p0, p1) in zip(self.trunks, self.trunk_rest):
            q0, q1 = piv + k * (p0 - AC) + sh, piv + k * (p1 - AC) + sh
            if abs(q0[0] - q1[0]) < 1e-9:                 # vertical: clip to the window
                lo, hi = max(min(q0[1], q1[1]), WIN_LO + 0.08), min(max(q0[1], q1[1]), WIN_HI - 0.08)
                if hi <= lo or ta <= 0:
                    ln.set_stroke(opacity=0)
                    continue
                ln.put_start_and_end_on(col(np.array([q0[0], lo, 0])), col(np.array([q0[0], hi, 0])))
                ln.set_stroke(MAGENTA, width=w, opacity=ta)
            else:
                mk = float(hall_mask(q0[1]))
                lo, hi = max(min(q0[0], q1[0]), -4.7), min(max(q0[0], q1[0]), 4.7)
                if hi <= lo or ta * mk <= 0:
                    ln.set_stroke(opacity=0)
                    continue
                ln.put_start_and_end_on(col(np.array([lo, q0[1], 0])), col(np.array([hi, q0[1], 0])))
                ln.set_stroke(MAGENTA, width=w, opacity=ta * mk)

        # ---- messages along the trunks (3.40 - 5.35)
        if T_DROP - EPS <= now < T_MSG_END:
            env = ramp(now, T_DROP, T_DROP + 0.10) * (1 - ramp(now, T_MSG_END - 0.15, T_MSG_END))
            s = (now - T_DROP) * MSG_SPEED + MSG_PHASE
            head = self._walk_pos(s)
            tail = self._walk_pos(np.maximum(s - MSG_TAIL, MSG_PHASE))
            H = piv + k * (np.concatenate([head, np.zeros((N_MSG, 1))], axis=1) - AC) + sh
            T = piv + k * (np.concatenate([tail, np.zeros((N_MSG, 1))], axis=1) - AC) + sh
            self.msgs.points = seg_points(col(T), col(H))
            self.msgs.set_stroke(MAGENTA, width=6 * k * sc, opacity=0.95 * env * a_col)
        else:
            self.msgs.points = EMPTY.copy()

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
        if now < T_COL - EPS:
            a = ease_out_cubic(ramp(now, T_OUT0, T_OUT1))
            self.outline.pointwise_become_partial(self.outline_full, 0, max(a, 1e-3))
            self.outline.set_stroke(CYAN, width=3, opacity=1)
            self.outline.set_fill(CYAN, opacity=0)
            return
        self.collapse.render(now, T_COL, T_SET, T_EXIT)

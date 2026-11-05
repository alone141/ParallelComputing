"""The song data model and everything the engine derives from it.

A song module (songs/<slug>.py) builds a `Song` and exports it as SONG. The engine never reads
anything song-specific from anywhere else. All positions are in STEPS: the song chooses what a
step is (a sixteenth, an eighth, a 32nd) through `steps_per_beat`.

Film step k maps to a record step through `segments` (the kept stretches of the record, in order;
anything not in a segment is a cut). Each segment may carry its own bpm (a tempo map); film time
is the cumulative sum of step lengths, and the frame count follows from the total.
"""
from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from math import gcd

FPS = 60
W, H = 1080, 1920

# Built-in tooth types. A song may add its own or override these through Song.tooth_types:
#   add      height of the tooth above the pitch circle (world units; a tall tooth is 18)
#   width    tangential width as a fraction of the standard tooth (1.0)
#   bright   0..1: how strongly the tooth is painted in the gear's colour (0 = bare steel, a rest)
#   subhits  when the tooth strikes within its step, as fractions (() = it never strikes: a rest)
#   tips     how many tips it is drawn with (a compound tooth: a double, a herta)
#   ext      how far it runs on into the next step (a held note)
#   shade    an explicit RGB to paint it instead of the gear colour (optional)
TOOTH_TYPES = {
    "tall":   dict(add=18.0, width=1.0, bright=1.0, subhits=(0.0,), tips=1, ext=0.0),
    "medium": dict(add=11.0, width=1.0, bright=1.0, subhits=(0.0,), tips=1, ext=0.0),
    "stub":   dict(add=6.0,  width=1.0, bright=0.0, subhits=(),     tips=1, ext=0.0),
    "long":   dict(add=18.0, width=1.0, bright=1.0, subhits=(0.0,), tips=1, ext=0.6),
    "double": dict(add=18.0, width=1.0, bright=1.0, subhits=(0.0, 0.5), tips=2, ext=0.0),       # two 32nds in a 16th
    "herta":  dict(add=18.0, width=1.0, bright=1.0, subhits=(0.0, 0.25, 0.5), tips=3, ext=0.0),  # 32nd 32nd 16th in an eighth
}
TOOTH_DEFAULTS = dict(add=18.0, width=1.0, bright=1.0, subhits=(0.0,), tips=1, ext=0.0, shade=None)
# Sounds a gear can make: the picture's flash style per kind.
HIT_KINDS = ("click", "pulse", "kick", "snare", "note", "chug", "ride", "bell", "crash")


@dataclass(frozen=True)
class Segment:
    """A kept stretch of the record: `steps` record steps from `record_start`, at `bpm` (None = the
    song's nominal bpm). Consecutive segments are the film, in order; gaps between them are cuts."""
    record_start: int
    steps: int
    bpm: float | None = None
    label: str = ""              # shown instead of LAP n/N while the cycle readout does not apply
    level_jump: bool = False     # for the audio fit: the segment begins with a band entry or a cut
    search_s: float | None = None   # for the audio fit: widen the search around the record's own clock


@dataclass(frozen=True)
class Section:
    """A section of the record, for bar numbering and the README: `bars` bars of `meter`
    (steps per bar may vary within a section: give `bar_steps` as a list in that case)."""
    name: str
    record_start: int
    bar_steps: tuple            # steps per bar, in order, e.g. (16,) * 10 + (20,)


@dataclass(frozen=True)
class Gear:
    key: str
    name: str                   # stack label (short, CAPS)
    short: str                  # label on or beside the gear
    colour: tuple
    teeth: int                  # tooth positions = its cycle in steps
    strikes: dict               # {position: tooth type}; positions not listed are stubs
    side: int                   # +1 above the rack, -1 below
    x: float                    # resting position along the rack, world units (a multiple of PITCH)
    ranges: tuple               # ((k_in, k_out), ...) engaged while k_in <= k < k_out
    sound: str = "click"        # HIT_KINDS; "ride" means medium teeth ping and tall teeth ring the bell
    first_hit: str | None = None   # a special kind for the very first strike of a range, e.g. "crash"
    colour2: tuple | None = None   # alternate colour for positions in `colour2_positions`
    colour2_positions: frozenset = frozenset()
    white_position: int | None = None   # a position painted white (the cycle start on the rim gear)
    stack: str | None = None    # which stack row it lights (defaults to its own key)
    lead: float = 0.0           # seconds: starts its glide earlier and hangs longer (stagger)
    label_dx: float = 0.0       # screen px: its beside-label steps aside
    label_dy: float = 0.0       # screen px (positive = away from the rack)
    origin: int = 0             # the film step at which strike position 0 is on the contact line. The rack
                                # drives every gear with one phase (tooth index k is on the contact at step
                                # k), so a gear whose count does not divide its entry step names its origin:
                                # a 13-tooth chorus gear entering at 240 has origin=240.
    phase: int = 0              # deprecated alias: position `phase` is on the contact at step 0, i.e.
                                # origin = -phase mod teeth. Use `origin`.
    rocker: str | None = None   # the Rocker this gear belongs to: it then keeps its own phase (it advances
                                # only while its bar is on) and is drawn on the rocker's beam

    def __post_init__(self):
        if self.phase and not self.origin:
            object.__setattr__(self, "origin", (-self.phase) % self.teeth)

    def tooth_type(self, pos: int) -> str:
        return self.strikes.get((pos - self.origin) % self.teeth, "stub")

    def position(self, tooth: int) -> int:
        """The strike position (0..teeth-1) that tooth index `tooth` carries."""
        return (tooth - self.origin) % self.teeth

    def engaged(self, k: float) -> bool:
        return any(a <= k < b for a, b in self.ranges)

    def range_at(self, k: float):
        for a, b in self.ranges:
            if a <= k < b:
                return a, b
        return None


@dataclass(frozen=True)
class Rocker:
    """A seesaw: two gears of one instrument that take turns meshing with the rack, bar by bar. For a
    song that alternates bar lengths (5/8 + 7/8) the bar lengths become the gears: `members` maps a
    bar length in steps to the gear that serves bars of that length. A member's tooth count must be a
    multiple of its bar length (a 14-tooth gear serves alternating 7/8 bars half a turn at a time), it
    advances only while its bar is on, and its tooth 0 is on the contact at every one of its bar lines.
    The swap is a quick eased tip of the beam that lands the incoming gear's tooth on the bar line.
    Members are ordinary Gears with `rocker=key`, the rocker's `ranges` and `side`, and their own x
    (multiples of the pitch, either side of the pivot); the rocker glides in and out as one piece."""
    key: str
    name: str                   # stack row label (the members light it)
    short: str                  # label under (or over) the assembly
    colour: tuple
    members: dict               # {bar_steps: gear_key}
    ranges: tuple               # film step ranges in which the rocker is on the rack (each starts on a bar line)
    side: int = -1
    lead: float = 0.0           # seconds: glide stagger, as for a gear
    swap_s: float = 0.1         # the seesaw tip, seconds, ending exactly on the bar line
    lift: float = 46.0          # world units the resting gear is raised off the rack
    label_dx: float = 0.0
    label_dy: float = 0.0

    def engaged(self, k: float) -> bool:
        return any(a <= k < b for a, b in self.ranges)

    def range_at(self, k: float):
        for a, b in self.ranges:
            if a <= k < b:
                return a, b
        return None


@dataclass(frozen=True)
class Moment:
    """A caption. It is shown from `step` until the next moment's step. kind: "hook" (up from frame
    0, fades out only), "big" (appears at once), "home" (gold), "caption" (quick fade in)."""
    step: int
    title: str
    subtitle: str
    kind: str = "caption"
    colour: tuple | str | None = None    # RGB, or a gear key whose colour tints the title


@dataclass(frozen=True)
class Counter:
    """The group counter: the current bar's steps as dots, filled when one of `gears` strikes."""
    gears: tuple
    bars: tuple                 # steps per bar, repeating from `k_from`
    k_from: int
    k_to: int


@dataclass(frozen=True)
class SlipSpec:
    """The drift readout, derived from two gears: the cycle gear (period = its teeth) and a beat
    period in steps. Slip per lap = cycle mod beat; laps to home = beat / gcd. `beat_gears` are the
    gears that carry the slip bracket (the first one engaged); the readout fades in when the first
    of them enters, and HOME flashes where the slip returns to 0 while one of them is engaged."""
    cycle_gear: str
    beat_steps: int
    beat_gears: tuple
    k_from: int = 0                     # film steps where this spec applies (a song may give a list of
    k_to: int | None = None             # specs, one per stretch with its own cycle gear and beat)


@dataclass(frozen=True)
class CameraMove:
    """Ease (smoothstep) to a frame over [step_from, step_to] and hold it. Either fit `keys` into
    the safe box (with `zoom_cap`) or give cam_x / zoom explicitly."""
    step_from: float
    step_to: float
    keys: tuple = ()
    zoom_cap: float = 0.95
    cam_x: float | None = None
    zoom: float | None = None


@dataclass(frozen=True)
class Anchor:
    """A distinctive instrument for aligning the recording: its gears, and a frequency band."""
    name: str
    band_hz: tuple
    gears: tuple
    tooth_types: tuple = ()     # if given, only hits of these tooth types count (the melody notes, not the pedal)
    on_grid: bool = True        # only strikes on the step itself count (a "late" tooth striking off the grid is no anchor)


@dataclass
class Song:
    slug: str
    title: str
    artist: str
    album: str
    year: int
    bpm: float
    steps_per_beat: int
    segments: list
    gears: list
    moments: list
    card_lines: list = field(default_factory=list)       # closing card, after the title lines
    credit_lines: list = field(default_factory=list)     # disclaimer / credit, small
    first_click_offset_s: float = 0.0
    sections: list = field(default_factory=list)
    counters: list = field(default_factory=list)
    slip: SlipSpec | list | None = None  # one SlipSpec, or a list of them (each with k_from / k_to)
    cycle_steps: int | None = None       # the riff cycle, for LAP n / N and the rack's lap marks
    cycle_end: int | None = None         # film step after which LAP gives way to the segment label
    rack_marks: tuple | None = None      # film steps that get a notched rack tooth (None = every cycle_steps)
    stack: list = field(default_factory=list)            # stack row order (gear keys)
    camera: list = field(default_factory=list)           # CameraMove list; empty = automatic
    opening_zoom: float = 0.9
    anchors: list = field(default_factory=list)
    end_fade_s: float = 1.2
    cut_note: str | None = None          # shown for a few seconds after the first cut
    first_step_guess_s: float = 0.5      # for the owner's audio tools: where the first step may be
    crossfade_s: float = 0.04
    gain_db: float = -2.4
    tooth_types: dict = field(default_factory=dict)      # the song's own tooth types (merged over TOOTH_TYPES)
    lap_word: str = "LAP"                # the readout's word for a cycle ("LAP 3 / 8", or "BAR 7 / 24")
    rockers: list = field(default_factory=list)          # Rocker list (members are gears with rocker=key)
    bar_lengths: tuple = ()              # steps per bar over the whole film, from step 0 (rockers need it)
    step_times: list | None = None       # film seconds of every step 0..total_steps (a beat-tracked record:
                                         # when given it replaces the segments' bpm as the picture's clock)
    lap_label_pos: tuple | None = None   # screen (x, y) for the LAP / segment label when the counter row is
                                         # too wide for it (16-dot bars push it into the apps' button column)
    # environment overrides (named generically): GEARS_BPM replaces the nominal bpm (segments that
    # name their own bpm keep it), GEARS_OFFSET replaces first_click_offset_s.

    # ---- derived timeline -----------------------------------------------------------------------
    def __post_init__(self):
        self.bpm = float(os.environ.get("GEARS_BPM", self.bpm))
        self.first_click_offset_s = float(os.environ.get("GEARS_OFFSET", self.first_click_offset_s))
        self.by_key = {g.key: g for g in self.gears}
        self.keys = tuple(g.key for g in self.gears)
        self.slips = list(self.slip) if isinstance(self.slip, (list, tuple)) else ([self.slip] if self.slip else [])
        if isinstance(self.slip, (list, tuple)):
            self.slip = self.slips[0] if self.slips else None
        self.types = {name: {**TOOTH_DEFAULTS, **spec} for name, spec in TOOTH_TYPES.items()}
        for name, spec in self.tooth_types.items():
            self.types[name] = {**TOOTH_DEFAULTS, **spec}
        for name, spec in self.types.items():
            if spec["bright"] <= 0 and spec["subhits"]:
                raise ValueError(f"tooth type {name!r}: a striking tooth must be painted (bright > 0)")
        if not self.stack:
            self.stack = [g.key for g in self.gears if g.stack is None]
        # segments -> film step boundaries and seconds
        self.seg_film_start, self.seg_time_start, self.seg_step_s = [], [], []
        k, t = 0, self.first_click_offset_s
        for s in self.segments:
            bpm = s.bpm if s.bpm else self.bpm
            step_s = 60.0 / (bpm * self.steps_per_beat)
            self.seg_film_start.append(k)
            self.seg_time_start.append(t)
            self.seg_step_s.append(step_s)
            k += s.steps
            t += s.steps * step_s
        self.total_steps = k
        self.total_s = t
        if self.step_times is not None:
            st = [float(v) for v in self.step_times]
            if len(st) != self.total_steps + 1:
                raise ValueError(f"step_times has {len(st)} entries; the film has {self.total_steps} steps (needs total_steps + 1)")
            self.step_times = st
            self.seg_time_start = [st[a] for a in self.seg_film_start]
            self.seg_step_s = [(st[a + sg.steps] - st[a]) / sg.steps for a, sg in zip(self.seg_film_start, self.segments)]
            self.total_s = st[-1]
        self.n_frames = int(round(self.total_s * FPS))
        self.nominal_step_s = 60.0 / (self.bpm * self.steps_per_beat)
        # rockers: which member serves which bar, and each member's own phase (steps engaged so far)
        self.by_rocker = {r.key: r for r in self.rockers}
        self.bar_starts = []
        pos = 0
        for n in self.bar_lengths:
            self.bar_starts.append((pos, n))
            pos += n
        self.member_bars = {}            # gear key -> [(k_start, k_end, Te0)]
        self.member_of = {}              # gear key -> rocker key
        for r in self.rockers:
            counts = {key: 0 for key in r.members.values()}
            for key in r.members.values():
                self.member_bars[key] = []
                self.member_of[key] = r.key
            for start, n in self.bar_starts:
                if not r.engaged(start):
                    continue
                key = r.members.get(n)
                if key is None:
                    continue
                self.member_bars[key].append((start, start + n, counts[key]))
                counts[key] += n

    def segment_index(self, k: float) -> int:
        i = 0
        for j, start in enumerate(self.seg_film_start):
            if k >= start:
                i = j
        return i

    def step_time(self, k: float) -> float:
        """Film seconds of (fractional) film step k, through the tempo map (or the step time list)."""
        if self.step_times is not None:
            st = self.step_times
            if k <= 0:
                return st[0] + k * (st[1] - st[0])
            if k >= self.total_steps:
                return st[-1] + (k - self.total_steps) * (st[-1] - st[-2])
            i = int(k)
            return st[i] + (k - i) * (st[i + 1] - st[i])
        if k <= 0:
            return self.first_click_offset_s + k * self.seg_step_s[0]
        i = self.segment_index(min(k, self.total_steps - 1e-9))
        return self.seg_time_start[i] + (k - self.seg_film_start[i]) * self.seg_step_s[i]

    def time_step(self, t: float) -> float:
        """Fractional film step at film second t (inverse of step_time)."""
        if self.step_times is not None:
            st = self.step_times
            if t <= st[0]:
                return (t - st[0]) / (st[1] - st[0])
            if t >= st[-1]:
                return self.total_steps + (t - st[-1]) / (st[-1] - st[-2])
            import bisect
            i = bisect.bisect_right(st, t) - 1
            return i + (t - st[i]) / (st[i + 1] - st[i])
        if t <= self.first_click_offset_s:
            return (t - self.first_click_offset_s) / self.seg_step_s[0]
        i = 0
        for j, ts in enumerate(self.seg_time_start):
            if t >= ts:
                i = j
        return self.seg_film_start[i] + (t - self.seg_time_start[i]) / self.seg_step_s[i]

    def k_at(self, frame: int) -> float:
        return self.time_step(frame / FPS)

    def step_s_at(self, k: float) -> float:
        if self.step_times is not None:
            i = int(max(0.0, min(k, self.total_steps - 1e-9)))
            return self.step_times[i + 1] - self.step_times[i]
        return self.seg_step_s[self.segment_index(max(0.0, min(k, self.total_steps - 1e-9)))]

    def record_step(self, k: int) -> int:
        i = self.segment_index(k)
        return self.segments[i].record_start + (k - self.seg_film_start[i])

    def record_bar(self, k: int):
        """(section name, 1-based bar number within the record) for film step k, from `sections`."""
        r = self.record_step(k)
        bar = 0
        for sec in self.sections:
            pos = sec.record_start
            for n, bs in enumerate(sec.bar_steps):
                bar += 1
                if pos <= r < pos + bs:
                    return sec.name, bar
                pos += bs
        return None, None

    # ---- hits ------------------------------------------------------------------------------------
    def member_tooth(self, key: str, k: int):
        """A rocker member's own tooth index on the contact at film step k, or None when its bar is off."""
        for a, b, te0 in self.member_bars.get(key, ()):
            if a <= k < b:
                return te0 + (k - a)
        return None

    def member_Te(self, key: str, k: float) -> float:
        """A rocker member's rotation (its own phase) at fractional film step k: it advances only while
        its bar is on and stays put while it rests."""
        te = 0.0
        for a, b, te0 in self.member_bars.get(key, ()):
            if k < a:
                break
            te = te0 + min(k, b) - a
        return te

    def member_meshed(self, key: str, k: float) -> bool:
        return any(a <= k < b for a, b, _ in self.member_bars.get(key, ()))

    def tooth_at(self, key: str, k: int) -> str:
        """The tooth type gear `key` presents to the rack on film step k (a rocker member off its bar: a stub)."""
        g = self.by_key[key]
        if key in self.member_of:
            mt = self.member_tooth(key, k)
            return "stub" if mt is None else g.tooth_type(mt)
        return g.tooth_type(k)

    def hits_at(self, key: str, k: int) -> list:
        """Hit kinds gear `key` makes on film step k if engaged (the caller checks engagement)."""
        g = self.by_key[key]
        tt = self.tooth_at(key, k)
        if not self.types[tt]["subhits"]:
            return []
        rng = g.range_at(k)
        if g.first_hit and rng and k == rng[0]:
            return [g.first_hit]
        if g.sound == "ride":
            return ["bell"] if self.types[tt]["add"] >= 15 else ["ride"]
        return [g.sound]

    def subhits(self, key: str, k: int):
        return self.types[self.tooth_at(key, k)]["subhits"]

    def tip_add(self, g) -> float:
        """The gear's tallest tooth (for its tip radius): the layout and labels use it."""
        return max([self.types[t]["add"] for t in g.strikes.values()] + [TOOTH_TYPES["tall"]["add"]])

    # ---- slip / home --------------------------------------------------------------------------------
    def slip_for(self, k: float | None = None):
        """The SlipSpec that applies at film step k (the first one when k is None)."""
        if not self.slips:
            return None
        if k is None:
            return self.slips[0]
        for sp in self.slips:
            if sp.k_from <= k < (sp.k_to if sp.k_to is not None else self.total_steps):
                return sp
        return None

    def slip_params(self, k: float | None = None):
        sp = self.slip_for(k)
        if sp is None:
            return None
        cyc = self.by_key[sp.cycle_gear].teeth
        per_lap = cyc % sp.beat_steps
        laps = sp.beat_steps // gcd(cyc, sp.beat_steps)
        k_ref = max(sp.k_from, min(a for key in sp.beat_gears for a, b in self.by_key[key].ranges))
        return cyc, per_lap, laps, k_ref

    def slip_at(self, k: float) -> int:
        p = self.slip_params(k)
        if p is None:
            return 0
        cyc, per_lap, laps, k_ref = p
        return (int((k - k_ref) // cyc) * per_lap) % laps if laps else 0

    @property
    def homes(self) -> tuple:
        """Film steps where the slip returns to 0 while the spec's beat gear is engaged or just ending
        (or exactly where a ranged spec ends: the home its stretch was heading for)."""
        out = []
        for sp in self.slips:
            cyc = self.by_key[sp.cycle_gear].teeth
            per_lap = cyc % sp.beat_steps
            laps = sp.beat_steps // gcd(cyc, sp.beat_steps)
            k_ref = max(sp.k_from, min(a for key in sp.beat_gears for a, b in self.by_key[key].ranges))
            if per_lap == 0:                 # no drift: nothing to come home from
                continue
            k = k_ref + cyc * laps
            while k < self.total_steps and (sp.k_to is None or k <= sp.k_to):
                if any(self.by_key[key].engaged(k - 1e-9) for key in sp.beat_gears[:1]) or k == sp.k_to:
                    out.append(k)
                k += cyc * laps
        return tuple(out)

    # ---- captions ------------------------------------------------------------------------------------
    def moment_index(self, k: float) -> int:
        idx = 0
        for i, m in enumerate(self.moments):
            if k >= m.step:
                idx = i
        return idx

    def moment_window(self, idx: int):
        a = self.moments[idx].step
        b = self.moments[idx + 1].step if idx + 1 < len(self.moments) else self.total_steps
        return a, b

    def counter_at(self, k: int):
        """(bar length, slot index within the bar, [(filled?, colour)] per slot) or None."""
        for c in self.counters:
            if c.k_from <= k < c.k_to:
                rel = k - c.k_from
                cyc = sum(c.bars)
                pos = rel % cyc
                start = 0
                for b in c.bars:
                    if start <= pos < start + b:
                        slots = []
                        for q in range(b):
                            step = k - (pos - start) + q
                            colour = None
                            for gk in c.gears:
                                spec = self.types[self.tooth_at(gk, step)]
                                if spec["subhits"]:
                                    colour = tuple(spec["shade"]) if spec["shade"] else self.by_key[gk].colour
                                    break
                            slots.append(colour)
                        return b, pos - start, slots
                    start += b
        return None

    def lap_label(self, k: float) -> str:
        if self.cycle_steps and (self.cycle_end is None or k < self.cycle_end):
            n_laps = (self.cycle_end or self.total_steps) // self.cycle_steps
            return f"{self.lap_word} {min(int(k // self.cycle_steps), n_laps - 1) + 1} / {n_laps}"
        return self.segments[self.segment_index(k)].label

    def anchor_hit(self, a, key: str, k: int) -> bool:
        """Does gear `key` give anchor `a` a hit on film step k? Engaged, striking, of an allowed tooth
        type, and (unless on_grid is False) striking on the step itself, not late inside it."""
        g = self.by_key[key]
        if not (g.engaged(k) and self.hits_at(key, k)):
            return False
        if a.tooth_types and self.tooth_at(key, k) not in a.tooth_types:
            return False
        if a.on_grid and 0.0 not in self.subhits(key, k):
            return False
        return True

    # ---- record map ----------------------------------------------------------------------------------
    def record_map(self) -> dict:
        segs = []
        for i, s in enumerate(self.segments):
            d = dict(film_step=self.seg_film_start[i], record_step=s.record_start, steps=s.steps,
                     bpm=s.bpm if s.bpm else self.bpm, film_s=round(self.seg_time_start[i], 6),
                     step_s=round(self.seg_step_s[i], 9), label=s.label)
            if s.level_jump:
                d["level_jump"] = True
            if s.search_s is not None:
                d["search_s"] = s.search_s
            segs.append(d)
        anchors = []
        for a in self.anchors:
            rs = sorted({self.record_step(k) for key in a.gears for k in range(self.total_steps)
                         if self.anchor_hit(a, key, k)})
            anchors.append(dict(name=a.name, band_hz=list(a.band_hz), record_steps=rs))
        out = dict(song=self.slug, title=self.title, artist=self.artist, bpm=self.bpm,
                   steps_per_beat=self.steps_per_beat, first_click_offset_s=self.first_click_offset_s,
                   first_step_guess_s=self.first_step_guess_s, total_steps=self.total_steps,
                   total_s=round(self.total_s, 6), frames=self.n_frames, fps=FPS, segments=segs,
                   anchors=anchors, crossfade_s=self.crossfade_s, fade_out_s=self.end_fade_s, gain_db=self.gain_db)
        if self.step_times is not None:
            out["step_film_s"] = [round(v, 6) for v in self.step_times]   # the picture's clock, every step
        return out

    # ---- validation ----------------------------------------------------------------------------------
    def check(self) -> list:
        problems = []
        if self.steps_per_beat not in (1, 2, 3, 4, 6, 8, 12):
            problems.append(f"steps_per_beat {self.steps_per_beat} is unusual")
        if not self.segments:
            problems.append("no segments")
        for g in self.gears:
            for pos, tt in g.strikes.items():
                if not (0 <= pos < g.teeth):
                    problems.append(f"gear {g.key}: strike position {pos} outside 0..{g.teeth - 1}")
                if tt not in self.types:
                    problems.append(f"gear {g.key}: unknown tooth type {tt!r} (built-in: {', '.join(TOOTH_TYPES)}; song: {', '.join(self.tooth_types)})")
            if g.sound not in HIT_KINDS:
                problems.append(f"gear {g.key}: unknown sound {g.sound!r}")
            for a, b in g.ranges:
                if not (0 <= a < b <= self.total_steps):
                    problems.append(f"gear {g.key}: range {a}..{b} outside the film 0..{self.total_steps}")
                if (a - g.origin) % g.teeth != 0 and not g.rocker:
                    problems.append(f"gear {g.key}: enters at {a}, not a multiple of its {g.teeth} teeth from its origin {g.origin} (it must land on position 0)")
            if abs(g.x / 42.0 - round(g.x / 42.0)) > 1e-9:
                problems.append(f"gear {g.key}: x {g.x} is not a multiple of the pitch 42")
            if g.side not in (-1, 1):
                problems.append(f"gear {g.key}: side must be +1 or -1")
            if g.stack and g.stack not in self.by_key:
                problems.append(f"gear {g.key}: stack row {g.stack!r} is not a gear")
        # gears on the same side that are on screen together must not overlap
        m = 42.0 / math.pi
        for a in self.gears:
            for b in self.gears:
                if a.key >= b.key or a.side != b.side:
                    continue
                for ra in a.ranges:
                    for rb in b.ranges:
                        if ra[1] <= rb[0] or rb[1] <= ra[0]:
                            continue
                        d = abs(a.x - b.x)
                        need = a.teeth * m / 2 + b.teeth * m / 2 + 36 + 4
                        if d < need:
                            problems.append(f"gears {a.key} and {b.key} overlap on the same side ({d:.0f} < {need:.0f})")
        for mo in self.moments:
            if not (0 <= mo.step < self.total_steps):
                problems.append(f"moment {mo.title!r} at step {mo.step} outside the film")
            if mo.kind not in ("hook", "big", "home", "caption"):
                problems.append(f"moment {mo.title!r}: unknown kind {mo.kind!r}")
        if self.moments and self.moments[0].step != 0:
            problems.append("the first moment should be at step 0 (the hook)")
        for sp in self.slips:
            for key in (sp.cycle_gear, *sp.beat_gears):
                if key not in self.by_key:
                    problems.append(f"slip: {key!r} is not a gear")
        for a in self.anchors:
            for key in a.gears:
                if key not in self.by_key:
                    problems.append(f"anchor {a.name!r}: {key!r} is not a gear")
            for tt in a.tooth_types:
                if tt not in self.types:
                    problems.append(f"anchor {a.name!r}: unknown tooth type {tt!r}")
        for c in self.counters:
            for gk in c.gears:
                if gk not in self.by_key:
                    problems.append(f"counter: {gk!r} is not a gear")
        if self.step_times is not None:
            st = self.step_times
            if any(b <= a for a, b in zip(st, st[1:])):
                problems.append("step_times must increase strictly")
            if abs(st[0] - self.first_click_offset_s) > 1e-6:
                problems.append(f"step_times[0] = {st[0]} but first_click_offset_s = {self.first_click_offset_s}")
        if self.rockers and sum(self.bar_lengths) != self.total_steps:
            problems.append(f"bar_lengths sum to {sum(self.bar_lengths)}, the film has {self.total_steps} steps")
        starts = {a for a, n in self.bar_starts}
        for r in self.rockers:
            keys = list(r.members.values())
            for n, key in r.members.items():
                if key not in self.by_key:
                    problems.append(f"rocker {r.key}: member {key!r} is not a gear")
                    continue
                g = self.by_key[key]
                if g.rocker != r.key:
                    problems.append(f"rocker {r.key}: gear {key} does not name rocker={r.key!r}")
                if g.teeth % n != 0:
                    problems.append(f"rocker {r.key}: gear {key} has {g.teeth} teeth, not a multiple of its {n}-step bar")
                if g.side != r.side:
                    problems.append(f"rocker {r.key}: gear {key} is on side {g.side}, the rocker on {r.side}")
                if tuple(g.ranges) != tuple(r.ranges):
                    problems.append(f"rocker {r.key}: gear {key} ranges {g.ranges} differ from the rocker's {r.ranges}")
                if not self.member_bars.get(key):
                    problems.append(f"rocker {r.key}: gear {key} never gets a bar")
            for a, b in r.ranges:
                if a not in starts:
                    problems.append(f"rocker {r.key}: enters at {a}, which is not a bar line")
                if b not in starts and b != self.total_steps:
                    problems.append(f"rocker {r.key}: leaves at {b}, which is not a bar line")
                for start, n in self.bar_starts:
                    if a <= start < b and n not in r.members:
                        problems.append(f"rocker {r.key}: no member for a {n}-step bar at {start}")
                        break
        for g in self.gears:
            if g.rocker and g.rocker not in self.by_rocker:
                problems.append(f"gear {g.key}: rocker {g.rocker!r} does not exist")
        for key in self.stack:
            if key not in self.by_key:
                problems.append(f"stack: {key!r} is not a gear")
        if self.n_frames != round(self.total_s * FPS):
            problems.append("frame count")
        return problems


def load_song(path: str) -> Song:
    """Import songs/<slug>.py (by path) and return its SONG."""
    spec = importlib.util.spec_from_file_location("gears_song_" + os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    song = getattr(mod, "SONG")
    if not isinstance(song, Song):
        raise TypeError(f"{path} must export SONG = Song(...)")
    return song

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

    def tooth_type(self, pos: int) -> str:
        return self.strikes.get(pos % self.teeth, "stub")

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
    slip: SlipSpec | None = None
    cycle_steps: int | None = None       # the riff cycle, for LAP n / N and the rack's lap marks
    cycle_end: int | None = None         # film step after which LAP gives way to the segment label
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
    # environment overrides (named generically): GEARS_BPM replaces the nominal bpm (segments that
    # name their own bpm keep it), GEARS_OFFSET replaces first_click_offset_s.

    # ---- derived timeline -----------------------------------------------------------------------
    def __post_init__(self):
        self.bpm = float(os.environ.get("GEARS_BPM", self.bpm))
        self.first_click_offset_s = float(os.environ.get("GEARS_OFFSET", self.first_click_offset_s))
        self.by_key = {g.key: g for g in self.gears}
        self.keys = tuple(g.key for g in self.gears)
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
        self.n_frames = int(round(self.total_s * FPS))
        self.nominal_step_s = 60.0 / (self.bpm * self.steps_per_beat)

    def segment_index(self, k: float) -> int:
        i = 0
        for j, start in enumerate(self.seg_film_start):
            if k >= start:
                i = j
        return i

    def step_time(self, k: float) -> float:
        """Film seconds of (fractional) film step k, through the tempo map."""
        if k <= 0:
            return self.first_click_offset_s + k * self.seg_step_s[0]
        i = self.segment_index(min(k, self.total_steps - 1e-9))
        return self.seg_time_start[i] + (k - self.seg_film_start[i]) * self.seg_step_s[i]

    def time_step(self, t: float) -> float:
        """Fractional film step at film second t (inverse of step_time)."""
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
    def hits_at(self, key: str, k: int) -> list:
        """Hit kinds gear `key` makes on film step k if engaged (the caller checks engagement)."""
        g = self.by_key[key]
        tt = g.tooth_type(k)
        if not self.types[tt]["subhits"]:
            return []
        rng = g.range_at(k)
        if g.first_hit and rng and k == rng[0]:
            return [g.first_hit]
        if g.sound == "ride":
            return ["bell"] if self.types[tt]["add"] >= 15 else ["ride"]
        return [g.sound]

    def subhits(self, key: str, k: int):
        return self.types[self.by_key[key].tooth_type(k)]["subhits"]

    def tip_add(self, g) -> float:
        """The gear's tallest tooth (for its tip radius): the layout and labels use it."""
        return max([self.types[t]["add"] for t in g.strikes.values()] + [TOOTH_TYPES["tall"]["add"]])

    # ---- slip / home --------------------------------------------------------------------------------
    def slip_params(self):
        s = self.slip
        if s is None:
            return None
        cyc = self.by_key[s.cycle_gear].teeth
        per_lap = cyc % s.beat_steps
        laps = s.beat_steps // gcd(cyc, s.beat_steps)
        k_ref = min(a for key in s.beat_gears for a, b in self.by_key[key].ranges)
        return cyc, per_lap, laps, k_ref

    def slip_at(self, k: float) -> int:
        p = self.slip_params()
        if p is None:
            return 0
        cyc, per_lap, laps, k_ref = p
        return (int((k - k_ref) // cyc) * per_lap) % laps if laps else 0

    @property
    def homes(self) -> tuple:
        """Film steps where the slip returns to 0 while the song's beat gear is engaged or just ending."""
        p = self.slip_params()
        if p is None:
            return ()
        cyc, per_lap, laps, k_ref = p
        out = []
        if per_lap == 0:                 # no drift: nothing to come home from
            return ()
        k = k_ref + cyc * laps
        while k < self.total_steps:
            if any(self.by_key[key].engaged(k - 1e-9) for key in self.slip.beat_gears[:1]):
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
                                if self.types[self.by_key[gk].tooth_type(step)]["subhits"]:
                                    colour = self.by_key[gk].colour
                                    break
                            slots.append(colour)
                        return b, pos - start, slots
                    start += b
        return None

    def lap_label(self, k: float) -> str:
        if self.cycle_steps and (self.cycle_end is None or k < self.cycle_end):
            n_laps = (self.cycle_end or self.total_steps) // self.cycle_steps
            return f"LAP {min(int(k // self.cycle_steps), n_laps - 1) + 1} / {n_laps}"
        return self.segments[self.segment_index(k)].label

    # ---- record map ----------------------------------------------------------------------------------
    def record_map(self) -> dict:
        segs = []
        for i, s in enumerate(self.segments):
            segs.append(dict(film_step=self.seg_film_start[i], record_step=s.record_start, steps=s.steps,
                             bpm=s.bpm if s.bpm else self.bpm, film_s=round(self.seg_time_start[i], 6),
                             step_s=round(self.seg_step_s[i], 9), label=s.label))
        anchors = []
        for a in self.anchors:
            rs = sorted({self.record_step(k) for key in a.gears for k in range(self.total_steps)
                         if self.by_key[key].engaged(k) and self.hits_at(key, k)})
            anchors.append(dict(name=a.name, band_hz=list(a.band_hz), record_steps=rs))
        return dict(song=self.slug, title=self.title, artist=self.artist, bpm=self.bpm,
                    steps_per_beat=self.steps_per_beat, first_click_offset_s=self.first_click_offset_s,
                    first_step_guess_s=self.first_step_guess_s, total_steps=self.total_steps,
                    total_s=round(self.total_s, 6), frames=self.n_frames, fps=FPS, segments=segs,
                    anchors=anchors, crossfade_s=self.crossfade_s, fade_out_s=self.end_fade_s, gain_db=self.gain_db)

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
                if a % g.teeth != 0:
                    problems.append(f"gear {g.key}: enters at {a}, not a multiple of its {g.teeth} teeth (it must land on tooth 0)")
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
        if self.slip:
            for key in (self.slip.cycle_gear, *self.slip.beat_gears):
                if key not in self.by_key:
                    problems.append(f"slip: {key!r} is not a gear")
        for c in self.counters:
            for gk in c.gears:
                if gk not in self.by_key:
                    problems.append(f"counter: {gk!r} is not a gear")
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

"""Round 4: one gear per instrument. The picture follows Gojira's recording of "The Art of Dying" (The Way of All Flesh, 2008).

The film is silent; the owner muxes the original recording. Everything on screen is a function
of one sixteenth counter k on the record's own clock:

    film time (s) = FIRST_CLICK_OFFSET_S + k * 15 / BPM

k = 0 is the first rim click (bar 1). The record's bars 34-41 (the pulse alone, cycle suspended)
are cut: film sixteenth k is record sixteenth k for k < 540 and record sixteenth k + 128 after.

Two parameters the owner may change after measuring the recording: BPM (default 140.000) and
FIRST_CLICK_OFFSET_S (default 0.0). Override with the environment variables AOD_BPM and
AOD_OFFSET; the frame count follows from them.

    python3 rhythm.py      # prints the timeline and runs the self-check
"""
from __future__ import annotations

import os
from dataclasses import dataclass

BPM = float(os.environ.get("AOD_BPM", "140.000"))
FIRST_CLICK_OFFSET_S = float(os.environ.get("AOD_OFFSET", "0.0"))

# ---- the cycle --------------------------------------------------------------------------------
GROUPS = (5, 5, 5, 3, 3, 3, 5, 5, 5, 3, 3)
CYCLE = sum(GROUPS)                                         # 45
BEAT = 4
GROUP_STARTS = tuple(sum(GROUPS[:i]) for i in range(len(GROUPS)))
RIM_RESTS = frozenset(s + g - 1 for s, g in zip(GROUP_STARTS, GROUPS))
RIM_HITS = tuple(t for t in range(CYCLE) if t not in RIM_RESTS)
SNARES = frozenset(s + (3 if g == 5 else 2) for s, g in zip(GROUP_STARTS, GROUPS))
KICK_RESTS = frozenset(s + 4 for s, g in zip(GROUP_STARTS, GROUPS) if g == 5)
KICKS = frozenset(range(CYCLE)) - SNARES - KICK_RESTS

# ---- the film's sections, in sixteenths --------------------------------------------------------
K_PULSE = 180          # bar 12: the tribal pulse joins (lap 5)
K_BAND = 540           # bar 42 after the cut: kit, ride, bass, guitars; clicks and pulse stop (lap 13)
K_TURN = 720           # bar 53: the 6/4 turnaround, 24 sixteenths
K_END = 744
TURN_LEN = K_END - K_TURN                                   # 24
CUT_RECORD_SIXTEENTHS = 128                                 # bars 34-41 removed (8 bars of 4/4)
HOMES = (360, 540)     # the pulse enters on a beat at a cycle start, so it is home 4 laps later, and again
N_LAPS_CYCLE = K_TURN // CYCLE                              # 16 laps on the 45 cycle

# turnaround cell, 6 sixteenths, played four times: K K K S . S; bass C1 x3 then C2 x3
TURN_CELL = ("K", "K", "K", "S", ".", "S")
TURN_KICKS = frozenset(i for i, c in enumerate(TURN_CELL) if c == "K")
TURN_SNARES = frozenset(i for i, c in enumerate(TURN_CELL) if c == "S")

# ---- timing -----------------------------------------------------------------------------------
FPS = 60
SIXTEENTH = 15.0 / BPM
LAP = CYCLE * SIXTEENTH
TOTAL = FIRST_CLICK_OFFSET_S + K_END * SIXTEENTH
N_FRAMES = round(TOTAL * FPS)                               # 4783 at 140 bpm, offset 0
END_FADE_S = 1.2       # the owner fades the audio over the last 1.2 s; the picture closes with it


def k_at(frame: int) -> float:
    """Continuous sixteenth index at a frame (negative before the first click, if offset > 0)."""
    return (frame / FPS - FIRST_CLICK_OFFSET_S) / SIXTEENTH


def t_of(k: int) -> int:
    return k % CYCLE


def group_of(t: int):
    for i, (s, g) in enumerate(zip(GROUP_STARTS, GROUPS)):
        if s <= t < s + g:
            return i, s, g
    raise ValueError(t)


def record_sixteenth(k: int) -> int:
    return k if k < K_BAND else k + CUT_RECORD_SIXTEENTHS


def record_bar(k: int) -> int:
    """1-based bar of the record. The tab writes each 180-sixteenth stretch as ten bars of 4/4 and one
    of 5/4 (eleven bars); the cut bars 34-41 are plain 4/4 and are skipped."""
    block, within = divmod(k, 180)
    bar = block * 11 + (within // 16 + 1 if within < 160 else 11)
    return bar + 8 if k >= K_BAND else bar


# ---- the gears: one per instrument --------------------------------------------------------------
# A gear has `n` tooth positions; `tall` positions are hits (they reach the rack's floor and strike),
# every other position is a stub tooth (it still meshes, it does not strike). `medium` positions are
# the ride's ghosted pings. `wide` positions are the bass's held notes (a longer tooth). `x` is the
# gear's resting position along the rack in world units (multiples of the 42-unit pitch), `side` is
# above (+1) or below (-1) the rack. k_in / k_out: engaged while k_in <= k < k_out.
@dataclass(frozen=True)
class Gear:
    key: str
    name: str          # stack label
    short: str         # label on the gear
    n: int
    tall: frozenset
    colour: tuple
    k_in: int
    k_out: int
    x: float
    side: int
    medium: frozenset = frozenset()
    wide: frozenset = frozenset()
    colour2: tuple = None        # second colour for the turnaround's C1/C2 and C2/Eb2 teeth
    stack: str = None            # which stack row lights for it (turnaround gears share rows)


PITCH = 42.0
RACK_HALF = 20.0               # half-thickness of the rack; tall teeth (18) never cross its centre line
_12_BEAT = frozenset(j for j in range(12) if j % 4 == 0)
_12_OFF = frozenset(j for j in range(12) if j % 4 == 2)
_T6 = tuple(range(12))
GEARS = (
    Gear("rim",    "RIM",   "RIM CLICKS",  45, frozenset(RIM_HITS), (226, 206, 166), 0, K_BAND, 0 * PITCH, +1),
    Gear("pulse",  "PULSE", "PULSE",       12, _12_BEAT,            (255, 170, 60),  K_PULSE, K_BAND, 0 * PITCH, -1),
    Gear("snare",  "SNARE", "SNARE",       45, SNARES,              (255, 255, 255), K_BAND, K_TURN, 11 * PITCH, -1),
    Gear("kick",   "KICK",  "KICK",        45, KICKS,               (255, 92, 70),   K_BAND, K_TURN, 17 * PITCH, +1),
    Gear("bass",   "BASS",  "BASS",        45, frozenset(RIM_HITS), (182, 140, 255), K_BAND, K_TURN, 28 * PITCH, -1,
         wide=frozenset(t for t in RIM_HITS if (t + 1) % CYCLE in RIM_RESTS)),
    Gear("gtr",    "GTRS",  "GUITARS",     12, frozenset(range(12)), (125, 255, 154), K_BAND, K_TURN, 31 * PITCH, +1),
    Gear("ride",   "RIDE",  "RIDE",        12, _12_BEAT,            (108, 224, 255), K_BAND, K_END, 36 * PITCH, +1, medium=_12_OFF),
    Gear("tbass",  "BASS",  "BASS 6/4",    12, frozenset(range(12)), (182, 140, 255), K_TURN, K_END, 39 * PITCH, -1,
         colour2=(226, 206, 255), stack="bass"),
    Gear("tkick",  "KICK",  "KICK 6/4",    12, frozenset(j for j in _T6 if j % 6 in (0, 1, 2)), (255, 92, 70), K_TURN, K_END, 42 * PITCH, +1, stack="kick"),
    Gear("tgtr",   "GTRS",  "GUITARS 6/4", 12, frozenset(range(12)), (125, 255, 154), K_TURN, K_END, 45 * PITCH, -1,
         colour2=(206, 255, 214), stack="gtr"),
    Gear("tsnare", "SNARE", "SNARE 6/4",   12, frozenset(j for j in _T6 if j % 6 in (3, 5)), (255, 255, 255), K_TURN, K_END, 48 * PITCH, +1, stack="snare"),
)
BY_KEY = {g.key: g for g in GEARS}
KEYS = tuple(g.key for g in GEARS)
STACK_KEYS = ("rim", "pulse", "kick", "snare", "ride", "bass", "gtr")
HOME_WHITE_TOOTH = 0           # the rim gear's tooth 0 is painted white: the cycle start
SLIDE_S = 0.28                 # a gear slides in (lifted) over this long, then drops into mesh
DROP_S = 0.08


def tooth_kind(key: str, pos: int) -> str:
    """'tall' (a hit), 'medium' (ghost ping), 'wide' (held bass note) or 'stub' (a rest)."""
    g = BY_KEY[key]
    p = pos % g.n
    if p in g.wide:
        return "wide"
    if p in g.tall:
        return "tall"
    if p in g.medium:
        return "medium"
    return "stub"


def hits_at(key: str, k: int) -> list[str]:
    """What gear `key` strikes on sixteenth k if engaged (the caller checks engagement)."""
    g = BY_KEY[key]
    kind = tooth_kind(key, k)
    if kind == "stub":
        return []
    if key == "rim":
        return ["click"]
    if key == "pulse":
        return ["pulse"]
    if key in ("kick", "tkick"):
        return ["kick"]
    if key in ("snare", "tsnare"):
        return ["snare"]
    if key in ("bass", "tbass"):
        return ["note"]
    if key in ("gtr", "tgtr"):
        return ["chug"]
    if key == "ride":
        if k == K_BAND:
            return ["crash"]
        return ["bell"] if kind == "tall" else ["ride"]
    raise KeyError(key)


def is_group_start(k: int) -> bool:
    return t_of(k) in GROUP_STARTS


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def engaged(key: str, k: int) -> bool:
    g = BY_KEY[key]
    return g.k_in <= k < g.k_out


def motion(key: str, k: float):
    """Where the gear is: ("in", 1.0) engaged; ("enter", u) sliding in, u 0..1; ("drop", u) dropping into
    mesh; ("lift", u) lifting out; ("exit", u) sliding out; ("away", 0) not on screen."""
    g = BY_KEY[key]
    slide, drop = SLIDE_S / SIXTEENTH, DROP_S / SIXTEENTH
    if g.k_in == 0 and k < g.k_in:                       # the rim gear is simply there before the first click
        return ("in", 1.0)
    if k < g.k_in - slide - drop:
        return ("away", 0.0)
    if k < g.k_in - drop:
        return ("enter", (k - (g.k_in - slide - drop)) / slide)
    if k < g.k_in:
        return ("drop", (k - (g.k_in - drop)) / drop)
    if k < g.k_out:
        return ("in", 1.0)
    if g.k_out >= K_END:
        return ("in", 1.0)
    if k < g.k_out + drop:
        return ("lift", (k - g.k_out) / drop)
    if k < g.k_out + drop + slide:
        return ("exit", (k - g.k_out - drop) / slide)
    return ("away", 0.0)


def slip(k: float) -> int:
    """Teeth the beat is ahead of the cycle start, counted from the pulse's entry (0 = home)."""
    return int((k - K_PULSE) // CYCLE) % 4


# ---- captions: one per lap of the cycle, then the turnaround ----------------------------------
CAPTIONS = [
    ("WHY DOES THIS BEAT DRIFT?", "Gojira, The Art of Dying. The intro as a machine: one tooth per sixteenth."),
    ("ONE WHEEL, 45 TEETH", "Eleven are missing: the rests. Five, five, five, three, three, three, then five, five, five, three, three."),
    ("FIVES AND THREES", "x x x x · and x x ·. The dots on the left count each group as it strikes."),
    ("A SECOND WHEEL IS COMING", "Twelve teeth, three of them amber: the beat. It slides in next lap."),
    ("THE PULSE ENGAGES", "45 = 11 × 4 + 1. The small wheel turns 3¾ times a lap, so the beat lands one tooth later every lap."),
    ("ONE TOOTH BEHIND EVERY LAP", "45 = 11 × 4 + 1. The spare tooth piles up."),
    ("TWO TEETH", "The amber tooth now lands two teeth before the white one."),
    ("THREE TEETH", "One more lap and the slip is a whole beat."),
    ("HOME", "Four laps, 45 beats, 19.3 seconds. Now hear it drift again."),
    ("AGAIN: ONE TOOTH", "Same wheels, same slip. Only where you count from has changed."),
    ("TWO", "Fans call this riff 21/16 or 45/16. Both are half right: the 21 is the second half of the 45."),
    ("THREE", "Next lap it is home again, and the band is waiting on the inner tracks."),
    ("HOME. THE BAND.", "Five new wheels take the rack: kick, snare, bass, guitars, ride. The clicks and the pulse roll away."),
    ("KICK, SNARE, BASS: 45 TEETH EACH", "28 kicks, 11 snares, 34 bass notes. Short teeth are rests; a long tooth is a held note."),
    ("GUITARS AND RIDE: 12 TEETH", "The guitars strike every tooth. The ride's tall teeth are the bell, on the beat."),
    ("ONE MORE LAP, THEN A TURN", "Four small wheels swap in: K K K S · S, four times. A bar of 6/4."),
    ("THE TURNAROUND", "The 45s roll away. Six-tooth cells go round twice, then the riff starts over."),
]
assert len(CAPTIONS) == N_LAPS_CYCLE + 1


def caption_index(k: float) -> int:
    return min(int(max(k, 0) // CYCLE), N_LAPS_CYCLE)


def caption_window(idx: int) -> tuple[int, int]:
    """(k_start, k_end) of caption idx."""
    if idx < N_LAPS_CYCLE:
        return idx * CYCLE, (idx + 1) * CYCLE
    return K_TURN, K_END


# ---- self-check -------------------------------------------------------------------------------
def _check():
    assert sorted(RIM_RESTS) == [4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44]
    assert sorted(SNARES) == [3, 8, 13, 17, 20, 23, 27, 32, 37, 41, 44]
    assert len(KICKS) == 28 and len(RIM_HITS) == 34
    assert K_PULSE % BEAT == 0 and K_BAND % BEAT == 0 and K_TURN % BEAT == 0
    assert K_PULSE % CYCLE == 0 and K_BAND % CYCLE == 0 and K_TURN % CYCLE == 0
    assert K_TURN % 12 == 0, "the turnaround wheel (12 teeth) must be at slot 0 when it engages"
    assert all(h % 180 == 0 for h in HOMES)
    if BPM == 140.0 and FIRST_CLICK_OFFSET_S == 0.0:
        assert N_FRAMES == 4783, N_FRAMES
    assert slip(K_PULSE) == 0 and slip(K_PULSE + CYCLE) == 1 and slip(360) == 0 and slip(540) == 0
    assert record_bar(0) == 1 and record_bar(K_PULSE) == 12 and record_bar(K_BAND) == 42 and record_bar(K_TURN) == 53
    for g in GEARS:
        assert g.k_in % g.n == 0, (g.key, "must land on tooth 0")
        assert abs(g.x / PITCH - round(g.x / PITCH)) < 1e-9, g.key
    assert len(BY_KEY["rim"].tall) == 34 and len(BY_KEY["kick"].tall) == 28 and len(BY_KEY["snare"].tall) == 11
    assert len(BY_KEY["bass"].tall) == 34 and len(BY_KEY["bass"].wide) == 11
    assert hits_at("ride", K_BAND) == ["crash"] and hits_at("ride", K_BAND + 2) == ["ride"] and hits_at("ride", K_BAND + 4) == ["bell"]
    assert [hits_at("tkick", K_TURN + c) for c in range(6)] == [["kick"]] * 3 + [[], [], []]
    assert [hits_at("tsnare", K_TURN + c) for c in range(6)] == [[], [], [], ["snare"], [], ["snare"]]
    assert motion("pulse", 10)[0] == "away" and motion("pulse", K_PULSE + 1)[0] == "in"
    assert motion("rim", K_BAND + 20)[0] == "away" and motion("ride", K_END - 1)[0] == "in"
    # gears that are on screen together never overlap (tip radii from the renderer's module)
    import math
    m = PITCH / math.pi
    for a in GEARS:
        for b in GEARS:
            if a.key >= b.key or a.k_out <= b.k_in or b.k_out <= a.k_in or a.side != b.side:
                continue          # opposite sides are separated by the rack
            ra, rb = a.n * m / 2 + 18, b.n * m / 2 + 18
            ya, yb = a.side * (a.n * m / 2 + RACK_HALF), b.side * (b.n * m / 2 + RACK_HALF)
            d = math.hypot(a.x - b.x, ya - yb)
            assert d >= ra + rb + 4, (a.key, b.key, d, ra + rb)


if __name__ == "__main__":
    _check()
    print(f"BPM {BPM:.3f} · offset {FIRST_CLICK_OFFSET_S:.3f} s · sixteenth {SIXTEENTH * 1000:.4f} ms · lap {LAP:.4f} s")
    print(f"total {TOTAL:.4f} s · {N_FRAMES} frames @ {FPS} fps")
    for name, k in (("first click, bar 1", 0), ("pulse joins, bar 12", K_PULSE), ("home", 360),
                    ("cut + band, record bar 42", K_BAND), ("turnaround, bar 53", K_TURN), ("end", K_END)):
        print(f"  k = {k:3d}  film {FIRST_CLICK_OFFSET_S + k * SIXTEENTH:7.3f} s  record sixteenth {record_sixteenth(k):4d}  {name}")
    print("self-check passed")

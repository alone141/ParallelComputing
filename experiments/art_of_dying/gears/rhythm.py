"""Round 3: the picture follows Gojira's recording of "The Art of Dying" (The Way of All Flesh, 2008).

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


# ---- the parts --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Layer:
    key: str
    name: str
    wheel: str        # "riff" (45), "beat" (44) or "turn" (12)
    track: int        # 0 = the teeth, 1/2 = pin tracks
    colour: tuple
    k_in: int
    k_out: int        # exclusive


LAYERS = (
    Layer("rim",   "RIM CLICKS",   "riff", 0, (242, 234, 211), 0, K_BAND),
    Layer("pulse", "TRIBAL PULSE", "beat", 0, (255, 170, 60), K_PULSE, K_BAND),
    Layer("ks",    "KICK + SNARE", "riff", 1, (255, 92, 70), K_BAND, K_TURN),
    Layer("ride",  "RIDE",         "beat", 1, (108, 224, 255), K_BAND, K_END),
    Layer("bass",  "BASS",         "riff", 2, (182, 140, 255), K_BAND, K_TURN),
    Layer("gtr",   "GUITARS",      "beat", 2, (125, 255, 154), K_BAND, K_END),
    Layer("tks",   "TURNAROUND",   "turn", 1, (255, 92, 70), K_TURN, K_END),
    Layer("tbass", "TURN BASS",    "turn", 0, (182, 140, 255), K_TURN, K_END),
)
SNARE_COLOUR = (255, 255, 255)
BASS_C2_COLOUR = (222, 200, 255)
BY_KEY = {l.key: l for l in LAYERS}
KEYS = tuple(l.key for l in LAYERS)
STACK_KEYS = ("rim", "pulse", "ks", "ride", "bass", "gtr", "tks")   # rows in the left column


def hits_at(key: str, k: int) -> list[str]:
    """What part `key` plays on sixteenth k if engaged (the caller checks presence)."""
    t = t_of(k)
    if key == "rim":
        return [] if t in RIM_RESTS else ["click"]
    if key == "pulse":
        return ["pulse"] if k % BEAT == 0 else []
    if key == "ks":
        return ["snare"] if t in SNARES else (["kick"] if t in KICKS else [])
    if key == "ride":
        if k == K_BAND:
            return ["crash"]
        return ["bell"] if k % BEAT == 0 else (["ride"] if k % BEAT == 2 else [])
    if key == "bass":
        return [] if t in RIM_RESTS else ["note"]
    if key == "gtr":
        return ["chug"]
    if key == "tks":
        c = (k - K_TURN) % 6
        return ["kick"] if c in TURN_KICKS else (["snare"] if c in TURN_SNARES else [])
    if key == "tbass":
        return ["note"]
    raise KeyError(key)


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def presence(key: str, k: float) -> float:
    """0..1: the part's follower is engaged at sixteenth k (eased over 0.2 s at both ends)."""
    L = BY_KEY[key]
    e = 0.2 / SIXTEENTH
    return _smooth((k - L.k_in) / e + 1) * (1 - _smooth((k - L.k_out) / e + 1)) if k > L.k_in - e else 0.0


def engaged(key: str, k: int) -> bool:
    L = BY_KEY[key]
    return L.k_in <= k < L.k_out


def slip(k: float) -> int:
    """Teeth the beat is ahead of the cycle start, counted from the pulse's entry (0 = home)."""
    return int((k - K_PULSE) // CYCLE) % 4


# ---- captions: one per lap of the cycle, then the turnaround ----------------------------------
CAPTIONS = [
    ("WHY DOES THIS BEAT DRIFT?", "Gojira, The Art of Dying. The intro as a machine: one tooth per sixteenth."),
    ("ONE WHEEL, 45 TEETH", "Eleven are missing: the rests. Five, five, five, three, three, three, then five, five, five, three, three."),
    ("FIVES AND THREES", "x x x x · and x x ·. The dots on the left count each group as it strikes."),
    ("A SECOND WHEEL IS WAITING", "44 teeth, every fourth one amber: the beat. It engages next lap."),
    ("THE PULSE ENGAGES", "44 teeth is eleven beats. The riff wheel has 45: one tooth more."),
    ("ONE TOOTH BEHIND EVERY LAP", "45 = 11 × 4 + 1. The spare tooth piles up."),
    ("TWO TEETH", "The amber tooth now lands two teeth before the white one."),
    ("THREE TEETH", "One more lap and the slip is a whole beat."),
    ("HOME", "Four laps, 45 beats, 19.3 seconds. Now hear it drift again."),
    ("AGAIN: ONE TOOTH", "Same wheels, same slip. Only where you count from has changed."),
    ("TWO", "Fans call this riff 21/16 or 45/16. Both are half right: the 21 is the second half of the 45."),
    ("THREE", "Next lap it is home again, and the band is waiting on the inner tracks."),
    ("HOME. THE BAND.", "Kick, snare, ride, bass and both guitars enter together on the first tooth. The clicks and the pulse stop."),
    ("THE KIT RIDES THE RIFF WHEEL", "K K K S · and K K S. The bass plays the rim pattern on a low C."),
    ("THE RIDE RIDES THE BEAT WHEEL", "Bell on the beat, a ghost note between. Guitars on every tooth."),
    ("ONE MORE LAP, THEN A TURN", "The small wheel: six teeth, K K K S · S, four times. A bar of 6/4."),
    ("THE TURNAROUND", "The 45 pauses for six beats, then the riff starts over."),
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
    assert presence("rim", 10) == 1 and presence("pulse", 10) == 0 and presence("pulse", K_PULSE + 5) == 1
    assert presence("rim", K_BAND + 5) == 0 and presence("ks", K_BAND + 5) == 1 and presence("ks", K_TURN + 5) == 0
    assert presence("tks", K_TURN + 5) == 1 and presence("gtr", K_TURN + 5) == 1


if __name__ == "__main__":
    _check()
    print(f"BPM {BPM:.3f} · offset {FIRST_CLICK_OFFSET_S:.3f} s · sixteenth {SIXTEENTH * 1000:.4f} ms · lap {LAP:.4f} s")
    print(f"total {TOTAL:.4f} s · {N_FRAMES} frames @ {FPS} fps")
    for name, k in (("first click, bar 1", 0), ("pulse joins, bar 12", K_PULSE), ("home", 360),
                    ("cut + band, record bar 42", K_BAND), ("turnaround, bar 53", K_TURN), ("end", K_END)):
        print(f"  k = {k:3d}  film {FIRST_CLICK_OFFSET_S + k * SIXTEENTH:7.3f} s  record sixteenth {record_sixteenth(k):4d}  {name}")
    print("self-check passed")

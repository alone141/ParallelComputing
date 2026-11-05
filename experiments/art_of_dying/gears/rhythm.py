"""The rhythm and the timeline of the gears film: one source for the picture and the sound.

Gojira, "The Art of Dying" (The Way of All Flesh, 2008), intro. Every part is placed on one
45-sixteenth cycle at 140 bpm. Independent study; not affiliated with the band or their label.

Notation:  n = sixteenths since the film started;  t = n mod 45 (position in the lap);
           lap = n // 45.

    python3 rhythm.py      # self-check and a printed timeline
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- tempo and cycle --------------------------------------------------------------------------
BPM = 140
GROUPS = (5, 5, 5, 3, 3, 3, 5, 5, 5, 3, 3)                 # 24 + 21
CYCLE = sum(GROUPS)                                        # 45
BEAT = 4
GROUP_STARTS = tuple(sum(GROUPS[:i]) for i in range(len(GROUPS)))   # 0 5 10 15 18 21 24 29 34 39 42

RIM_RESTS = frozenset(s + g - 1 for s, g in zip(GROUP_STARTS, GROUPS))   # last slot of each group
RIM_HITS = tuple(t for t in range(CYCLE) if t not in RIM_RESTS)          # 34
SNARES = frozenset(s + (3 if g == 5 else 2) for s, g in zip(GROUP_STARTS, GROUPS))
KICK_RESTS = frozenset(s + 4 for s, g in zip(GROUP_STARTS, GROUPS) if g == 5)
KICKS = frozenset(range(CYCLE)) - SNARES - KICK_RESTS                    # 28
BEAT_PHASE = 0        # the beat wheel turns from frame 0, so beats fall on n = 0 (mod 4)

# ---- film timing ------------------------------------------------------------------------------
# 12 laps. Whole frames at 60 fps: 12 * 45 * (15/140) * 60 = 3471.43, so the film is 3471 frames
# and runs 0.012 % fast (139.98 bpm). Picture and sound both use SIXTEENTH below, so the loop is exact.
FPS = 60
N_LAPS = 12
N_SIXTEENTHS = N_LAPS * CYCLE                              # 540
N_FRAMES = round(N_SIXTEENTHS * (15 / BPM) * FPS)          # 3471
TOTAL = N_FRAMES / FPS                                     # 57.85 s
SIXTEENTH = TOTAL / N_SIXTEENTHS                           # 0.107130 s
LAP = CYCLE * SIXTEENTH                                    # 4.8208 s
FRAMES_PER_SIXT = N_FRAMES / N_SIXTEENTHS
HOME_EVERY = BEAT * CYCLE                                  # 180 sixteenths: lcm(4, 45)


def frame_to_n(f: int) -> float:
    return f / FRAMES_PER_SIXT


def t_of(n: int) -> int:
    return n % CYCLE


def group_of(t: int):
    """(index, start slot, size) of the group that slot t belongs to."""
    for i, (s, g) in enumerate(zip(GROUP_STARTS, GROUPS)):
        if s <= t < s + g:
            return i, s, g
    raise ValueError(t)


# ---- the six parts ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Layer:
    key: str
    name: str          # stack label
    wheel: str         # "riff" (45) or "beat" (44): which wheel carries its pins
    track: int         # 0 = the teeth themselves, 1/2 = inner pin tracks
    colour: tuple      # RGB
    entry_lap: int     # it enters (and has its focus lap) here


LAYERS = (
    Layer("rim",   "RIM CLICKS",   "riff", 0, (242, 234, 211), 0),
    Layer("pulse", "TRIBAL PULSE", "beat", 0, (255, 170, 60), 1),
    Layer("ks",    "KICK + SNARE", "riff", 1, (255, 92, 70), 5),
    Layer("ride",  "RIDE",         "beat", 1, (108, 224, 255), 6),
    Layer("bass",  "BASS",         "riff", 2, (182, 140, 255), 7),
    Layer("gtr",   "GUITARS",      "beat", 2, (125, 255, 154), 9),
)
SNARE_COLOUR = (255, 255, 255)
BY_KEY = {l.key: l for l in LAYERS}
KEYS = tuple(l.key for l in LAYERS)
FULL_MIX_LAPS = (10, 11)          # lap indices (0-based) where all six play without a focus


def hits_at(key: str, n: int) -> list[str]:
    """What part `key` plays on sixteenth n: a list of hit kinds (empty = silence).
    rim: click · pulse: pulse · ks: kick | snare · ride: ride | bell · bass: note · gtr: chug"""
    t = t_of(n)
    if key in ("rim", "bass"):
        return [] if t in RIM_RESTS else (["click"] if key == "rim" else ["note"])
    if key == "pulse":
        return ["pulse"] if n % BEAT == BEAT_PHASE else []
    if key == "ks":
        return ["snare"] if t in SNARES else (["kick"] if t in KICKS else [])
    if key == "ride":
        return ["bell"] if n % BEAT == BEAT_PHASE else (["ride"] if n % BEAT == (BEAT_PHASE + 2) % BEAT else [])
    if key == "gtr":
        return ["chug"]
    raise KeyError(key)


def is_group_start(n: int) -> bool:
    return t_of(n) in GROUP_STARTS


# ---- the timeline -----------------------------------------------------------------------------
PEEL = 3.2             # seconds before the end over which the parts above the rim clicks leave
PEEL_FADE = 0.6
FOCUS_DIM = 0.42       # picture: brightness of the other parts' pins during a focus lap
FOCUS_DUCK_DB = -8.0   # sound: the other parts during a focus lap
EASE = 0.3


def lap_start(lap: int) -> float:
    return lap * LAP


def entry_time(key: str) -> float:
    return lap_start(BY_KEY[key].entry_lap)


def focus_window(key: str) -> tuple[float, float]:
    s = entry_time(key)
    return s, s + LAP


def peel_window(key: str):
    i = KEYS.index(key)
    if i == 0:
        return None
    order = len(LAYERS) - i                    # guitars leave first, pulse last
    step = (PEEL - PEEL_FADE) / (len(LAYERS) - 2)
    start = TOTAL - PEEL + (order - 1) * step
    return start, start + PEEL_FADE


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def presence(key: str, time: float) -> float:
    """0..1: the part has entered and not yet peeled away (eased 0.3 s at its entry)."""
    e = entry_time(key)
    if time < e:
        return 0.0
    p = 1.0 if key == "rim" else _smooth((time - e) / 0.25)
    pw = peel_window(key)
    if pw is None:
        return p
    return p * (1.0 - _smooth((time - pw[0]) / (pw[1] - pw[0])))


def focus_weight(key: str, time: float) -> float:
    s, e = focus_window(key)
    if key == "rim":
        return 1.0 - _smooth((time - (e - EASE)) / EASE) if s <= time < e + EASE else 0.0
    if s - EASE <= time < e + EASE:
        return _smooth((time - s) / EASE) * (1.0 - _smooth((time - (e - EASE)) / EASE))
    return 0.0


def focused_layer(time: float):
    lap = int(time // LAP)
    for l in LAYERS:
        if l.entry_lap == lap:
            return l.key
    return None


def background_factor(key: str, time: float) -> float:
    return max((focus_weight(o.key, time) for o in LAYERS if o.key != key), default=0.0)


def pin_brightness(key: str, time: float) -> float:
    """Picture: 0..1 brightness of a part's pins (presence, dimmed while another part is in focus)."""
    return presence(key, time) * (1.0 - (1.0 - FOCUS_DIM) * background_factor(key, time))


def layer_gain(key: str, time: float) -> float:
    duck = 10 ** (FOCUS_DUCK_DB / 20)
    return presence(key, time) * (1.0 - (1.0 - duck) * background_factor(key, time))


@dataclass(frozen=True)
class Hit:
    n: int
    time: float
    key: str
    kind: str
    accent: bool
    gain: float


def events() -> list[Hit]:
    out = []
    for n in range(N_SIXTEENTHS):
        time = n * SIXTEENTH
        for l in LAYERS:
            g = layer_gain(l.key, time)
            if g <= 1e-4:
                continue
            for kind in hits_at(l.key, n):
                out.append(Hit(n, time, l.key, kind, is_group_start(n), g))
    return out


# ---- captions, one per lap --------------------------------------------------------------------
CAPTIONS = [
    ("WHY DOES THIS BEAT DRIFT?", "Gojira, The Art of Dying. The intro as a machine: one tooth per sixteenth."),
    ("44 TEETH IS ELEVEN BEATS", "The riff wheel has 45, one tooth more. Each lap, the beat ends up a tooth ahead."),
    ("ONE TOOTH EVERY LAP", "45 = 11 × 4 + 1. The spare tooth piles up."),
    ("FOUR TEETH IS ONE BEAT", "One more lap and the slip is a whole beat."),
    ("HOME", "Four laps, 45 beats, 19.3 seconds. Then it drifts again."),
    ("KICK AND SNARE", "On the riff wheel: K K K S · and K K S. The kick alternates feet."),
    ("THE RIDE", "On the beat wheel: the bell on every beat, a ghost note in between."),
    ("THE BASS", "The rim pattern on a low C. Each group's last note holds through the rest."),
    ("HOME AGAIN", "Five parts on two wheels. 45 against 44, every four laps."),
    ("TWO GUITARS", "Palm-muted C on every tooth, one left, one right."),
    ("THE WHOLE STACK", "A teaching mix: on the record the clicks never play with the kit."),
    ("BACK TO THE CLICKS", "It drifts one tooth a lap, and it comes home in four."),
]
assert len(CAPTIONS) == N_LAPS


# ---- self-check -------------------------------------------------------------------------------
def _check():
    assert sorted(RIM_RESTS) == [4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44]
    assert sorted(SNARES) == [3, 8, 13, 17, 20, 23, 27, 32, 37, 41, 44]
    assert sorted(KICK_RESTS) == [4, 9, 14, 28, 33, 38]
    assert len(KICKS) == 28 and len(SNARES) == 11 and len(RIM_HITS) == 34
    ks = "".join("S" if t in SNARES else "K" if t in KICKS else "." for t in range(CYCLE))
    assert ks == "KKKS.KKKS.KKKS.KKSKKSKKSKKKS.KKKS.KKKS.KKSKKS", ks
    rim = "".join("." if t in RIM_RESTS else "x" for t in range(CYCLE))
    assert rim == "xxxx.xxxx.xxxx.xx.xx.xx.xxxx.xxxx.xxxx.xx.xx.", rim
    assert N_LAPS % 4 == 0, "the beat wheel's paint must come round for the loop"
    assert N_FRAMES == 3471
    assert presence("rim", 0) == 1 and presence("pulse", 0) == 0
    assert all(presence(k, TOTAL - 1e-9) < 1e-6 for k in KEYS[1:])
    assert focused_layer(0.0) == "rim" and focused_layer(entry_time("gtr") + 1) == "gtr"
    assert focused_layer(lap_start(4) + 1) is None and focused_layer(TOTAL - 1) is None
    assert [BY_KEY[k].entry_lap for k in KEYS] == sorted(BY_KEY[k].entry_lap for k in KEYS)


if __name__ == "__main__":
    _check()
    ev = events()
    print(f"{BPM} bpm nominal · sixteenth {SIXTEENTH * 1000:.3f} ms · lap {LAP:.4f} s · {N_LAPS} laps")
    print(f"total {TOTAL:.4f} s · {N_FRAMES} frames @ {FPS} fps · {len(ev)} hits")
    for l in LAYERS:
        print(f"  {l.name:<13} enters lap {l.entry_lap + 1:2d} at {entry_time(l.key):6.2f} s · peel {peel_window(l.key)}")
    for i, (a, b) in enumerate(CAPTIONS):
        print(f"  lap {i + 1:2d} {lap_start(i):6.2f} s  {a}")
    print("self-check passed")

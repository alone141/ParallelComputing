"""The rhythm and the round-2 timeline: the single source for the picture and the sound.

Gojira, "The Art of Dying" (The Way of All Flesh, 2008), intro, at 140 bpm on one 45-sixteenth cycle.

    python3 rhythm.py        # self-check, prints the timeline, writes timeline.js for the game

Notation: n = sixteenths since the film started; t = n mod 45; lap = n // 45.
The film is one loop of 8 laps (360 sixteenths). It starts with the click on n = 0 already at the hit
line and ends 1/60 s before n = 360 would arrive there, so the last frame runs into the first.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

BPM = 140
SIXTEENTH = 15 / BPM                      # 0.107142857 s
GROUPS = (5, 5, 5, 3, 3, 3, 5, 5, 5, 3, 3)
CYCLE = sum(GROUPS)                       # 45
LAP = CYCLE * SIXTEENTH                   # 4.8214 s
GROUP_STARTS = tuple(sum(GROUPS[:i]) for i in range(len(GROUPS)))
RIM_RESTS = frozenset(s + g - 1 for s, g in zip(GROUP_STARTS, GROUPS))
SNARES = frozenset(s + (3 if g == 5 else 2) for s, g in zip(GROUP_STARTS, GROUPS))
KICK_RESTS = frozenset(s + 4 for s, g in zip(GROUP_STARTS, GROUPS) if g == 5)
KICKS = frozenset(range(CYCLE)) - SNARES - KICK_RESTS
BEAT_PHASE = 0                            # the cycle starts on a beat: pulse, bell on n = 0 (mod 4)

FPS = 60
N_LAPS = 8
N_STEPS = N_LAPS * CYCLE                  # 360 = 2 x 180: the loop point is a realignment
TOTAL_MUSIC = N_STEPS * SIXTEENTH         # 38.5714 s
N_FRAMES = round(TOTAL_MUSIC * FPS)       # 2314 (the film is 4.8 ms shorter than the music grid)
TOTAL = N_FRAMES / FPS
REALIGN = 4 * CYCLE                       # 180

FOCUS_DIM = 0.42                          # opacity of the other lanes during a focus lap
FOCUS_DUCK_DB = -8.0                      # the other layers' level during a focus lap
FOCUS_BOOST_DB = {"ride": 4.0}            # a part's own level in its focus lap (default +1.5 dB)
EASE = 0.3                                # s, ramps in and out of a focus lap


@dataclass(frozen=True)
class Layer:
    key: str
    name: str
    entry: int           # first sixteenth the layer plays (a lap line)
    peel: int | None     # first sixteenth the layer no longer plays (None: never leaves)


# one lap per part, in the record's order; full mix on lap 7 (n 270-314); the peel on lap 8
LAYERS = (
    Layer("rim", "CLICK", 0, None),
    Layer("pulse", "PULSE", 45, 339),
    Layer("ks", "KIT", 90, 333),
    Layer("ride", "RIDE", 135, 327),
    Layer("bass", "BASS", 180, 321),
    Layer("gtr", "GUITAR", 225, 315),
)
BY_KEY = {l.key: l for l in LAYERS}
FULL_MIX_LAP = 6


def t_of(n): return n % CYCLE


def hits_at(key, n):
    """Hit kinds layer `key` plays on sixteenth n (ignoring entry/peel)."""
    t = t_of(n)
    if key == "rim":
        return [] if t in RIM_RESTS else ["click"]
    if key == "bass":
        return [] if t in RIM_RESTS else ["note"]
    if key == "pulse":
        return ["pulse"] if n % 4 == BEAT_PHASE else []
    if key == "ks":
        return ["snare"] if t in SNARES else (["kick"] if t in KICKS else [])
    if key == "ride":
        return ["bell"] if n % 4 == BEAT_PHASE else (["ride"] if n % 4 == (BEAT_PHASE + 2) % 4 else [])
    if key == "gtr":
        return ["chug"]
    raise KeyError(key)


def present(key, n):
    l = BY_KEY[key]
    return n >= l.entry and (l.peel is None or n < l.peel)


def focus_lap(key):
    return BY_KEY[key].entry // CYCLE


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def focus_weight(key, time):
    """0..1: how much `key` is the focused layer at `time` (its entry lap, eased)."""
    s = BY_KEY[key].entry * SIXTEENTH
    e = s + LAP
    if key == "rim":
        return 1.0 - _smooth((time - (e - EASE)) / EASE) if time < e + EASE else 0.0
    if time < s - EASE or time >= e + EASE:
        return 0.0
    return _smooth((time - s) / EASE) * (1.0 - _smooth((time - (e - EASE)) / EASE))


def background(key, time):
    return max((focus_weight(o.key, time) for o in LAYERS if o.key != key), default=0.0)


def layer_gain(key, time):
    duck = 10 ** (FOCUS_DUCK_DB / 20)
    boost = 10 ** (FOCUS_BOOST_DB.get(key, 1.5) / 20 * focus_weight(key, time))
    return boost * (1.0 - (1.0 - duck) * background(key, time))


@dataclass(frozen=True)
class Hit:
    n: int
    time: float
    key: str
    kind: str
    accent: bool
    gain: float


def events():
    out = []
    for n in range(N_STEPS):
        time = n * SIXTEENTH
        for l in LAYERS:
            if not present(l.key, n):
                continue
            for kind in hits_at(l.key, n):
                out.append(Hit(n, time, l.key, kind, t_of(n) in GROUP_STARTS, layer_gain(l.key, time)))
    return out


def timeline():
    return dict(
        bpm=BPM, step=SIXTEENTH, cycle=CYCLE, laps=N_LAPS, steps=N_STEPS, fps=FPS, frames=N_FRAMES,
        total=TOTAL, realign=REALIGN, beatPhase=BEAT_PHASE,
        groups=list(GROUPS), groupStarts=list(GROUP_STARTS), rests=sorted(RIM_RESTS),
        snares=sorted(SNARES), kickRests=sorted(KICK_RESTS),
        focusDim=FOCUS_DIM, ease=EASE, fullMixLap=FULL_MIX_LAP,
        layers=[dict(key=l.key, name=l.name, entry=l.entry, peel=l.peel) for l in LAYERS],
    )


def _check():
    assert sorted(RIM_RESTS) == [4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44]
    assert sorted(SNARES) == [3, 8, 13, 17, 20, 23, 27, 32, 37, 41, 44]
    assert sorted(KICK_RESTS) == [4, 9, 14, 28, 33, 38]
    assert len(KICKS) == 28 and CYCLE - len(RIM_RESTS) == 34
    ks = "".join("S" if t in SNARES else "K" if t in KICKS else "." for t in range(CYCLE))
    assert ks == "KKKS.KKKS.KKKS.KKSKKSKKSKKKS.KKKS.KKKS.KKSKKS", ks
    assert N_STEPS % 4 == 0 and REALIGN % 4 == 0
    for l in LAYERS:
        assert l.entry % CYCLE == 0
        if l.peel is not None:
            assert l.peel >= 315 and l.peel <= 345, "the peel must be done before the loop's approach window"
    # after the peel only clicks remain, as at the start
    for n in range(345, N_STEPS):
        assert [l.key for l in LAYERS if present(l.key, n)] == ["rim"]


if __name__ == "__main__":
    _check()
    with open("timeline.js", "w") as f:
        f.write("// generated by rhythm.py: do not edit\nwindow.TIMELINE = " + json.dumps(timeline(), indent=1) + ";\n")
    ev = events()
    print(f"{BPM} bpm · sixteenth {SIXTEENTH*1000:.2f} ms · lap {LAP:.4f} s · {N_LAPS} laps · {N_STEPS} steps")
    print(f"music {TOTAL_MUSIC:.4f} s · film {TOTAL:.4f} s = {N_FRAMES} frames @ {FPS} fps · {len(ev)} hits")
    for l in LAYERS:
        print(f"  {l.name:<7} enters n={l.entry:3d} ({l.entry*SIXTEENTH:5.2f} s) lap {focus_lap(l.key)+1}  leaves n={l.peel}")
    from collections import Counter
    print("  hits per layer:", dict(Counter(h.key for h in ev)))
    print("wrote timeline.js · self-check passed")

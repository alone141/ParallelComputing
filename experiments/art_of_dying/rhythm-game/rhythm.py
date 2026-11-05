"""Round 3: the rhythm and the film timeline on the record's own clock. Picture only.

Gojira, "The Art of Dying" (The Way of All Flesh, 2008), intro. The owner muxes the original recording
later (it is not in the repository); this module fixes where every sixteenth of the film sits.

    python3 rhythm.py [--bpm 140.0] [--offset 0.0]     # self-check, prints the timeline, writes timeline.js

k = the film's sixteenth index from 0. Film time of sixteenth k = OFFSET + k * 15 / BPM.
Record mapping: film sixteenth k = record sixteenth k for k < 540, record sixteenth k + 128 for k >= 540
(the pulse-only bridge, record bars 34-41, is cut). The beat (quarter note) is on k = 0 (mod 4) throughout.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

GROUPS = (5, 5, 5, 3, 3, 3, 5, 5, 5, 3, 3)
CYCLE = sum(GROUPS)                                                   # 45
GROUP_STARTS = tuple(sum(GROUPS[:i]) for i in range(len(GROUPS)))
RIM_RESTS = frozenset(s + g - 1 for s, g in zip(GROUP_STARTS, GROUPS))
SNARES = frozenset(s + (3 if g == 5 else 2) for s, g in zip(GROUP_STARTS, GROUPS))
KICK_RESTS = frozenset(s + 4 for s, g in zip(GROUP_STARTS, GROUPS) if g == 5)
KICKS = frozenset(range(CYCLE)) - SNARES - KICK_RESTS

FPS = 60
PULSE_IN = 180            # bar 12: the tribal pulse joins (first realignment)
REALIGN_1 = 360           # bar 23: second realignment, pulse comes home
CUT = 540                 # bar 42 (record): the band enters; record bars 34-41 removed before it
TURN = 720                # bar 53: the 6/4 turnaround, 24 sixteenths
END = 744
TURN_PATTERN = "KKKS.S"   # kick kick kick snare rest snare, four times


@dataclass(frozen=True)
class Layer:
    key: str
    name: str
    entry: int
    exit: int             # first sixteenth the layer no longer plays


LAYERS = (
    Layer("rim", "CLICK", 0, CUT),
    Layer("pulse", "PULSE", PULSE_IN, CUT),
    Layer("ks", "KIT", CUT, END),
    Layer("ride", "RIDE", CUT, END),
    Layer("bass", "BASS", CUT, END),
    Layer("gtr", "GUITAR", CUT, END),
)
BY_KEY = {l.key: l for l in LAYERS}


def present(key, k):
    l = BY_KEY[key]
    return l.entry <= k < l.exit


def hits_at(key, k):
    """Hit kinds layer `key` plays on sixteenth k (ignoring entry/exit)."""
    if k >= TURN:                                      # the 6/4 turnaround
        p = (k - TURN) % 6
        if key == "ks":
            return {"K": ["kick"], "S": ["snare"], ".": []}[TURN_PATTERN[p]]
        if key == "bass":
            return ["note_c1"] if p < 3 else ["note_c2"]
        if key == "gtr":
            return ["chug_c2"] if p < 3 else ["chug_eb2"]
        if key == "ride":
            return ["bell"] if k % 4 == 0 else (["ride"] if k % 4 == 2 else [])
        return []
    t = k % CYCLE
    if key == "rim":
        return [] if t in RIM_RESTS else ["click"]
    if key == "bass":
        return [] if t in RIM_RESTS else ["note"]
    if key == "pulse":
        return ["pulse"] if k % 4 == 0 else []
    if key == "ks":
        return ["snare"] if t in SNARES else (["kick"] if t in KICKS else [])
    if key == "ride":
        if k == CUT:
            return ["crash"]
        return ["bell"] if k % 4 == 0 else (["ride"] if k % 4 == 2 else [])
    if key == "gtr":
        return ["chug"]
    raise KeyError(key)


def bar_of(k):
    """The tab's bar number (10 bars of 4/4 and one of 5/4 per 180 sixteenths) for film sixteenth k."""
    if k >= TURN:
        return 53
    rec = k if k < CUT else k + 128
    lapset, within = divmod(rec, 180)
    return lapset * 11 + (within // 16 + 1 if within < 160 else 11)


def timeline(bpm, offset):
    step = 15 / bpm
    total = offset + END * step
    frames = round(total * FPS)
    hits = 0
    for k in range(END):
        for l in LAYERS:
            if present(l.key, k):
                hits += len(hits_at(l.key, k))
    return dict(
        bpm=bpm, offset=offset, step=step, cycle=CYCLE, end=END, fps=FPS, frames=frames, total=total,
        pulseIn=PULSE_IN, realign1=REALIGN_1, cut=CUT, turn=TURN, turnPattern=TURN_PATTERN,
        groups=list(GROUPS), groupStarts=list(GROUP_STARTS), rests=sorted(RIM_RESTS),
        snares=sorted(SNARES), kickRests=sorted(KICK_RESTS), hits=hits,
        bars=[bar_of(k) for k in range(END)],
        layers=[dict(key=l.key, name=l.name, entry=l.entry, exit=l.exit) for l in LAYERS],
    )


def _check(tl):
    assert sorted(RIM_RESTS) == [4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44]
    assert sorted(SNARES) == [3, 8, 13, 17, 20, 23, 27, 32, 37, 41, 44]
    ks = "".join("S" if t in SNARES else "K" if t in KICKS else "." for t in range(CYCLE))
    assert ks == "KKKS.KKKS.KKKS.KKSKKSKKSKKKS.KKKS.KKKS.KKSKKS", ks
    assert PULSE_IN % 4 == 0 and REALIGN_1 % 4 == 0 and CUT % 4 == 0 and TURN % 4 == 0 and END - TURN == 24
    assert CUT % CYCLE == 0 and TURN % CYCLE == 0           # the band enters at t = 0; the turnaround too
    assert bar_of(0) == 1 and bar_of(PULSE_IN) == 12 and bar_of(CUT) == 42 and bar_of(TURN) == 53 and bar_of(CUT - 1) == 33
    if tl["bpm"] == 140.0 and tl["offset"] == 0.0:
        assert tl["frames"] == 4783, tl["frames"]
        assert abs(tl["total"] - 79.7142857) < 1e-6


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpm", type=float, default=140.0, help="tempo of the recording (default 140.000)")
    ap.add_argument("--offset", type=float, default=0.0, help="film time of the first rim click, s (FIRST_CLICK_OFFSET_S)")
    a = ap.parse_args()
    tl = timeline(a.bpm, a.offset)
    _check(tl)
    with open("timeline.js", "w") as f:
        f.write("// generated by rhythm.py: do not edit\nwindow.TIMELINE = " + json.dumps(tl) + ";\n")
    print(f"BPM {a.bpm:.3f} · FIRST_CLICK_OFFSET_S {a.offset:.3f} · sixteenth {tl['step']*1000:.3f} ms · lap {CYCLE*tl['step']:.4f} s")
    print(f"film {tl['total']:.4f} s = {tl['frames']} frames @ {FPS} fps · {tl['hits']} hits")
    for name, k in (("first click", 0), ("pulse in (bar 12)", PULSE_IN), ("realigned (bar 23)", REALIGN_1),
                    ("cut: band in (bar 42)", CUT), ("6/4 turnaround (bar 53)", TURN), ("end", END)):
        print(f"  k={k:3d}  {a.offset + k*tl['step']:8.3f} s  {name}")
    print("wrote timeline.js · self-check passed")

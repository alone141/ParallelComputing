#!/usr/bin/env python3
"""Three Doors: the story file. The only source of the film's numbers, strings and timings.

    python story.py            writes build/story.json (the picture and the score both read it)
    python story.py --table    prints the timeline and the round sequence

Pure Python 3, standard library only. Nothing here depends on the clock or on chance: the round sequence is
arranged (see SEQUENCE below), with a fixed linear congruential generator for the door assignments, so the
file is the same on every run.

The film (35.0 s, 2100 frames at 60 fps; frame 2099 is frame 0 again, so the video loops):

  0.00  frame 0: three closed doors, nothing else
  0.20  hook, four lines at 56 px, then the sub-line at 1.70; the hook frame (the cover) is 2.00; out by 3.00
  3.20  the three rules, one after another (rules 2 and 3 wrap to two lines); out by 8.60
  8.40  the scoreboard fades in: STAY 0, ROUND 0, SWITCH 0
  8.60  rounds 1 to 3, 2.0 s each, every step visible (switch, stay, switch); pause 14.60 to 16.60
 16.60  rounds 4 to 30, accelerating (durations from 30 to 6 frames); pause 21.60 to 23.60
 23.60  the sprint, rounds 31 to 300 in 5.0 s: the doors shimmer, the counts climb along the arranged sequence
 28.60  hold at ROUND 300, STAY 100, SWITCH 200
 29.40  "Switch."
 33.00  "Switch." leaves, the scoreboard fades, and by 34.80 everything is at its frame-0 state

Event kinds (the integer frame is always `frame`):
  stamp / unstamp   a text item comes in (with its bloom) / fades out            id names the text
  pick              your pick lights a door (round k, door d)                     k, door
  host_open         the host's door opens (round k, door d)                       k, door
  reveal            the two remaining doors open and the prize shows              k, prize, pick, outcome
  tally             the count ticks: STAY or SWITCH and ROUND                     k, outcome, stay, switch
  close             every open door closes over `frames` frames                   k
  pause             a hold with nothing moving but the glow                       k, end
  sprint            the blurred run: count_by_frame gives the rounds completed    start, end
  sprint_tick       the round count rose on this frame (sound only)               k
  scoreboard_in / scoreboard_out
  payoff            "Switch." stamps                                               (also a stamp event)
  loop              the frame from which the picture equals frame 0

Sound cues (`sound`): one entry per audible moment, {frame, cue, id}. Cues: hook, subline, rule, board, pick,
knock, reveal, chime_stay, chime_switch, close, tick, payoff, settle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "build", "story.json")

FPS = 60
W, H = 1080, 1920
FRAMES = 2100                       # 35.0 s
SLICES = 4
SLICE_FRAMES = FRAMES // SLICES     # 525
assert SLICE_FRAMES * SLICES == FRAMES


def F(seconds: float) -> int:
    """Frame of a time in seconds (exact for multiples of 1/60)."""
    f = round(seconds * FPS)
    assert abs(f - seconds * FPS) < 1e-6, seconds
    return f


# ---------------------------------------------------------------------------------------------- palette
COLOURS = {
    "bg": "#08061A",
    "starlight": "#F5F7FF",
    "mist_light": "#C9D1E6",
    "mist": "#8A93A8",
    "lavender": "#B9A6FF",     # the mechanic: the doors at rest, the hook's target phrase
    "cyan": "#6CF2FF",         # you: your pick, the STAY tally
    "amber": "#FFB547",        # the prize, the SWITCH tally
    "white": "#FFFFFF",
}
FACES = {
    "display": {"family": "Inter Display", "weight": 700},
    "medium": {"family": "Inter", "weight": 500},
    "regular": {"family": "Inter", "weight": 400},
    "mono": {"family": "JetBrains Mono", "weight": 500},
    "monoB": {"family": "JetBrains Mono", "weight": 700},
}
STAMP_BLOOM = {"rest_glow": 0.5, "peak_glow": 1.2, "frames_at_peak": 2, "settle_frames": 15, "fade_out_frames": 15}

# --------------------------------------------------------------------------------------------- geometry
DOOR_W, DOOR_H = 240, 400
DOOR_TOP = 980
DOOR_X = [280, 540, 800]            # centres; door 0 is the left door
GEOMETRY = {
    "doors": {"w": DOOR_W, "h": DOOR_H, "top": DOOR_TOP, "x": DOOR_X, "radius": 22, "frame_lw": 4,
              "panel_inset": 16, "open_frames": 12, "close_frames": 12},
    "prize": {"radius": 13, "halo": 110},
    "scoreboard": {"x": [250, 540, 830], "label_baseline": 400, "count_baseline": 520, "label_size": 26,
                   "label_tracking_em": 0.33, "count_size": 88, "fade_frames": 24},
    "hook": {"size": 56, "baselines": [600, 668, 736, 804], "subline_size": 34, "subline_baseline": 900,
             "x": 540, "align": "centre", "fade_out_frames": 30},
    "rules": {"size": 38, "baselines": [[600], [676, 726], [802, 852]], "x": 540, "align": "centre"},
    "payoff": {"size": 128, "baseline": 790, "x": 540, "align": "centre"},
    "nebula": {"centre": [540, 1180], "radius": 820, "peak": 0.14, "crossfade_frames": 72},
    "band": {"x0": 132, "x1": 948, "baseline_max": 1536, "ink_max": 1548, "button_column_x": 948,
             "button_column_y": [780, 1536], "cover_y": [240, 1680]},
    "mono": {"advance_em": 0.6},
}

# ------------------------------------------------------------------------------------------------ texts
TEXTS = {
    "hook": ["Three doors, one prize.", "You pick one.", "The host opens an empty one.", "Switch or stay?"],
    "hook_target": "Switch or stay?",
    "hook_you": "You",
    "subline": "Guess before it ends.",
    "rules": [["The host knows where the prize is."],
              ["The host always opens", "an empty door that is not yours."],
              ["Then you may switch", "to the other closed door."]],
    "labels": {"stay": "STAY", "round": "ROUND", "switch": "SWITCH"},
    "payoff": "Switch.",
}

# ------------------------------------------------------------------------------------------- timeline
T = {
    "hook_in": [F(0.20), F(0.50), F(0.80), F(1.10)],
    "subline_in": F(1.70),
    "cover": F(2.00),
    "hook_out": F(2.50),            # fade over hook.fade_out_frames; gone by 3.00
    "rules_in": [F(3.20), F(4.80), F(6.40)],
    "rules_out": F(8.40),           # fade over the stamp fade; gone by 8.65
    "board_in": F(8.40),
    "rounds_start": F(8.60),
    "pause1": [F(14.60), F(16.60)],
    "pause2": [F(21.60), F(23.60)],
    "sprint": [F(23.60), F(28.60)],
    "payoff_in": F(29.40),
    "payoff_out": F(33.00),
    "board_out": F(33.40),
    "loop_from": F(34.80),
}
SLOW_ROUND_FRAMES = 120
SLOW_PHASES = {"host": 30, "reveal": 66, "tally": 78, "close": 108}
ACCEL_FROM, ACCEL_TO = 30, 6        # frames per round, rounds 4 to 30

# ----------------------------------------------------------------------------------------- the sequence
# S = switching would have won (the prize was behind the other closed door); T = staying would have won
# (the prize was behind your door). Arranged so that the tallies are exactly 1:2, 10:20 and 100:200 at the
# three pauses, rounds 1 to 3 landing switch, stay, switch. The arrangement inside each block is a fixed
# shuffle from the generator below, so the pattern does not repeat visibly; it is not a random draw.


class LCG:
    """A small deterministic generator (the same numbers on every machine and Python version)."""

    def __init__(self, seed: int):
        self.x = seed & 0x7FFFFFFF

    def next(self) -> int:
        self.x = (1103515245 * self.x + 12345) & 0x7FFFFFFF
        return self.x

    def below(self, n: int) -> int:
        return (self.next() >> 8) % n

    def shuffle(self, items: list) -> list:
        a = list(items)
        for i in range(len(a) - 1, 0, -1):
            j = self.below(i + 1)
            a[i], a[j] = a[j], a[i]
        return a


def longest_run(s: str) -> int:
    best = run = 0
    prev = None
    for c in s:
        run = run + 1 if c == prev else 1
        prev = c
        best = max(best, run)
    return best


def arranged_block(n_stay: int, n_switch: int, gen: LCG, max_run: int, block: int, prefix: str = "") -> str:
    """n_stay T and n_switch S in blocks of `block` rounds, each block a fixed shuffle of its share (one third
    T), accepted only if no run of equal letters in prefix + the text so far exceeds max_run."""
    assert (n_stay + n_switch) % block == 0 and n_stay * 2 == n_switch and block % 3 == 0
    per_t, per_s = block // 3, 2 * block // 3
    out = ""
    while len(out) < n_stay + n_switch:
        for _ in range(10000):
            cand = "".join(gen.shuffle(["T"] * per_t + ["S"] * per_s))
            if longest_run(prefix + out + cand) <= max_run:
                out += cand
                break
        else:
            raise RuntimeError("no arrangement found")
    return out


def build_sequence() -> str:
    gen = LCG(20261028)
    seq = "STS"
    seq += arranged_block(9, 18, gen, 4, 9, seq)
    seq += arranged_block(90, 180, gen, 5, 6, seq)
    assert len(seq) == 300
    assert seq[:3].count("T") == 1 and seq[:30].count("T") == 10 and seq.count("T") == 100
    return seq


def assign_doors(seq: str) -> list[dict]:
    """Per round: your pick (moves every round), the prize door and the host's door, as door indices 0..2."""
    gen = LCG(1973)
    rounds = []
    pick = 0
    for k, outcome in enumerate(seq, start=1):
        if k > 1:
            pick = (pick + 1 + gen.below(2)) % 3            # a different door from the last round
        others = [d for d in range(3) if d != pick]
        if outcome == "T":
            prize = pick
            host = others[gen.below(2)]
        else:
            prize = others[gen.below(2)]
            host = [d for d in others if d != prize][0]
        assert len({pick, host, prize if prize != pick else host}) >= 2 and host != pick and host != prize
        rounds.append({"k": k, "outcome": "stay" if outcome == "T" else "switch", "pick": pick, "prize": prize, "host": host})
    return rounds


# ---------------------------------------------------------------------------------------------- timing
def accel_durations() -> list[int]:
    """Frames per round for rounds 4 to 30 (27 rounds), from ACCEL_FROM down to ACCEL_TO, summing exactly to
    the accelerating window (pause1 end to pause2 start)."""
    total = T["pause2"][0] - T["pause1"][1]
    n = 27
    # a power curve, then integer rounding with the remainder spread over the earliest rounds
    p = 3.4
    raw = [ACCEL_TO + (ACCEL_FROM - ACCEL_TO) * ((n - 1 - i) / (n - 1)) ** p for i in range(n)]
    scale = (total - n * ACCEL_TO) / (sum(raw) - n * ACCEL_TO)
    d = [max(ACCEL_TO, int(round(ACCEL_TO + (r - ACCEL_TO) * scale))) for r in raw]
    i = 0
    while sum(d) != total:
        if sum(d) < total:
            d[i % n] += 1
        else:
            j = i % n
            if d[j] > ACCEL_TO:
                d[j] -= 1
        i += 1
    assert sum(d) == total and all(x >= ACCEL_TO for x in d) and d[0] <= ACCEL_FROM + 2
    for a, b in zip(d, d[1:]):
        assert a >= b
    return d


def phases(duration: int) -> dict:
    """The frames (relative to the round start) of a round's steps, scaled from the slow round."""
    if duration == SLOW_ROUND_FRAMES:
        return dict(SLOW_PHASES)
    host = max(1, round(0.25 * duration))
    reveal = max(host + 1, round(0.55 * duration))
    tally = max(reveal + 1, round(0.65 * duration))
    close = max(tally + 1, round(0.88 * duration))
    assert close < duration, (duration, close)
    return {"host": host, "reveal": reveal, "tally": tally, "close": close}


def smoothstep(u: float) -> float:
    u = 0.0 if u < 0 else 1.0 if u > 1 else u
    return u * u * (3 - 2 * u)


def sprint_counts() -> list[int]:
    """Rounds completed at each frame of the sprint window (inclusive of both ends): 30 at the start, 300 at
    the end, following a smoothstep so the count accelerates in and eases out."""
    start, end = T["sprint"]
    n = end - start
    counts = []
    for i in range(n + 1):
        c = 30 + int(270 * smoothstep(i / n) + 1e-9)
        counts.append(min(300, c))
    counts[-1] = 300
    assert counts[0] == 30 and all(b >= a for a, b in zip(counts, counts[1:]))
    return counts


# ----------------------------------------------------------------------------------------- the story
def build() -> dict:
    seq = build_sequence()
    rounds = assign_doors(seq)
    stay = switch = 0
    for r in rounds:
        if r["outcome"] == "stay":
            stay += 1
        else:
            switch += 1
        r["stay"] = stay
        r["switch"] = switch
    assert rounds[2]["stay"] == 1 and rounds[29]["stay"] == 10 and rounds[299]["stay"] == 100

    events: list[dict] = []
    sound: list[dict] = []
    ev = lambda frame, kind, **kw: events.append({"frame": int(frame), "kind": kind, **kw})
    cue = lambda frame, c, ident: sound.append({"frame": int(frame), "cue": c, "id": ident})

    # the hook
    for i, f in enumerate(T["hook_in"]):
        ev(f, "stamp", id=f"hook{i + 1}")
        cue(f, "hook", f"hook{i + 1}")
    ev(T["subline_in"], "stamp", id="subline")
    cue(T["subline_in"], "subline", "subline")
    for i in range(4):
        ev(T["hook_out"], "unstamp", id=f"hook{i + 1}", frames=GEOMETRY["hook"]["fade_out_frames"])
    ev(T["hook_out"], "unstamp", id="subline", frames=GEOMETRY["hook"]["fade_out_frames"])
    # the rules
    for i, f in enumerate(T["rules_in"]):
        ev(f, "stamp", id=f"rule{i + 1}")
        cue(f, "rule", f"rule{i + 1}")
    for i in range(3):
        ev(T["rules_out"], "unstamp", id=f"rule{i + 1}", frames=STAMP_BLOOM["fade_out_frames"])
    # the scoreboard
    ev(T["board_in"], "scoreboard_in", frames=GEOMETRY["scoreboard"]["fade_frames"])
    cue(T["board_in"], "board", "board")

    # rounds 1 to 30, played
    f = T["rounds_start"]
    durations = [SLOW_ROUND_FRAMES] * 3 + accel_durations()
    for r, d in zip(rounds[:30], durations):
        k = r["k"]
        ph = phases(d)
        r["start"] = f
        r["duration"] = d
        r["pick_f"] = f
        r["host_f"] = f + ph["host"]
        r["reveal_f"] = f + ph["reveal"]
        r["tally_f"] = f + ph["tally"]
        ev(f, "pick", k=k, door=r["pick"])
        cue(f, "pick", f"pick{k}")
        ev(r["host_f"], "host_open", k=k, door=r["host"])
        cue(r["host_f"], "knock", f"host{k}")
        ev(r["reveal_f"], "reveal", k=k, prize=r["prize"], pick=r["pick"], outcome=r["outcome"])
        cue(r["reveal_f"], "reveal", f"reveal{k}")
        ev(r["tally_f"], "tally", k=k, outcome=r["outcome"], stay=r["stay"], switch=r["switch"])
        cue(r["tally_f"], "chime_stay" if r["outcome"] == "stay" else "chime_switch", f"tally{k}")
        if k in (3, 30):
            pause = T["pause1"] if k == 3 else T["pause2"]
            assert f + d == pause[0], (k, f + d, pause)
            close_f = pause[1] - GEOMETRY["doors"]["close_frames"]
            ev(pause[0], "pause", k=k, end=pause[1])
            ev(close_f, "close", k=k, frames=GEOMETRY["doors"]["close_frames"])
            cue(close_f, "close", f"close{k}")
            r["close_f"] = close_f
            r["end"] = pause[1]
            f = pause[1]
        else:
            close_f = f + ph["close"]
            ev(close_f, "close", k=k, frames=f + d - close_f)
            cue(close_f, "close", f"close{k}")
            r["close_f"] = close_f
            r["end"] = f + d
            f += d
    assert f == T["sprint"][0]

    # the sprint, rounds 31 to 300
    counts = sprint_counts()
    s0, s1 = T["sprint"]
    ev(s0, "sprint", start=s0, end=s1, from_round=30, to_round=300)
    ticks = []
    for i in range(1, len(counts)):
        if counts[i] > counts[i - 1]:
            ticks.append(s0 + i)
            ev(s0 + i, "sprint_tick", k=counts[i])
            cue(s0 + i, "tick", f"tick{counts[i]}")
    for r in rounds[30:]:
        r["start"] = None
    # the payoff and the ending
    ev(T["payoff_in"], "payoff")
    ev(T["payoff_in"], "stamp", id="payoff")
    cue(T["payoff_in"], "payoff", "payoff")
    ev(T["payoff_out"], "unstamp", id="payoff", frames=STAMP_BLOOM["fade_out_frames"] + 9)
    ev(T["board_out"], "scoreboard_out", frames=48)
    cue(T["board_out"], "settle", "settle")
    ev(T["loop_from"], "loop")
    events.sort(key=lambda e: (e["frame"], e["kind"]))
    sound.sort(key=lambda c: c["frame"])

    tallies = [{"k": r["k"], "stay": r["stay"], "switch": r["switch"]} for r in rounds]
    story = {
        "meta": {
            "title": "Three Doors", "fps": FPS, "frames": FRAMES, "width": W, "height": H, "total_s": FRAMES / FPS,
            "slices": SLICES, "slice_frames": SLICE_FRAMES, "colours": COLOURS, "faces": FACES, "stamp_bloom": STAMP_BLOOM,
            "loop": {"cover_frame": T["cover"], "loop_from": T["loop_from"], "last_frame": FRAMES - 1},
        },
        "geometry": GEOMETRY,
        "texts": TEXTS,
        "timeline": T,
        "sequence": seq,
        "rounds": rounds,
        "tallies": tallies,
        "sprint": {"start": s0, "end": s1, "count_by_frame": counts, "ticks": ticks},
        "events": events,
        "sound": sound,
    }
    return story


def table(story: dict) -> str:
    lines = []
    T_ = story["timeline"]
    lines.append(f"Three Doors: {story['meta']['frames']} frames at {story['meta']['fps']} fps = {story['meta']['total_s']:.2f} s, "
                 f"{story['meta']['slices']} slices of {story['meta']['slice_frames']}")
    lines.append("sequence (S = switch wins, T = stay wins):")
    seq = story["sequence"]
    lines.append("  " + seq[:3] + " | " + seq[3:30] + " |")
    for i in range(30, 300, 60):
        lines.append("  " + seq[i:i + 60])
    lines.append(f"  longest run {longest_run(seq)}; T at 3/30/300: {seq[:3].count('T')}/{seq[:30].count('T')}/{seq.count('T')}")
    lines.append("rounds 1 to 30:")
    lines.append("   k  out    pick host prize   start   dur  host reveal tally close   end  STAY SWITCH")
    for r in story["rounds"][:30]:
        lines.append(f"  {r['k']:2d}  {r['outcome']:6s} {r['pick']:3d} {r['host']:4d} {r['prize']:5d}  {r['start']:6d} {r['duration']:5d} "
                     f"{r['host_f'] - r['start']:5d} {r['reveal_f'] - r['start']:6d} {r['tally_f'] - r['start']:5d} {r['close_f'] - r['start']:5d} "
                     f"{r['end']:6d}  {r['stay']:4d} {r['switch']:6d}")
    sp = story["sprint"]
    lines.append(f"sprint: frames {sp['start']}-{sp['end']}, {len(sp['ticks'])} ticks, counts 30 -> 300")
    lines.append("timeline:")
    for k, v in T_.items():
        lines.append(f"  {k:14s} {v}")
    lines.append(f"events: {len(story['events'])}; sound cues: {len(story['sound'])}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Three Doors: write build/story.json")
    ap.add_argument("--table", action="store_true", help="print the timeline and the sequence")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args(argv)
    story = build()
    if args.table:
        print(table(story))
        return 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(story, fh, indent=1)
    print(f"{os.path.relpath(args.out, HERE)}: {story['meta']['frames']} frames, {len(story['events'])} events, "
          f"{len(story['sound'])} sound cues, sequence T at 3/30/300 = "
          f"{story['sequence'][:3].count('T')}/{story['sequence'][:30].count('T')}/{story['sequence'].count('T')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

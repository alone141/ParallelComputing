"""Rhythm model of the intro of Gojira's "The Art of Dying" (The Way of All Flesh, 2008).

Everything is counted in sixteenths at 140 bpm.  One cycle is 45 sixteenths,
grouped 5+5+5+3+3+3 | 5+5+5+3+3.  The quarter-note beat (every 4 sixteenths)
slips one sixteenth against the cycle per lap and realigns after 4 laps.
Independent study; not affiliated with the band or their label.
"""

BPM = 140
SIX = 60.0 / BPM / 4          # one sixteenth in seconds = 0.10714 s
CYCLE = 45
GROUPS = [5, 5, 5, 3, 3, 3, 5, 5, 5, 3, 3]

GROUP_START = []
_acc = 0
for g in GROUPS:
    GROUP_START.append(_acc)
    _acc += g
assert _acc == CYCLE

# Rim clicks: every sixteenth except the last of each group.
RESTS = {s + g - 1 for s, g in zip(GROUP_START, GROUPS)}
assert RESTS == {4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44}
RIM = [t for t in range(CYCLE) if t not in RESTS]            # 34 hits

# Full band (from bar 42): a 5 is "K K K S .", a 3 is "K K S".
SNARE = {s + g - 2 if g == 5 else s + g - 1 for s, g in zip(GROUP_START, GROUPS)}
assert SNARE == {3, 8, 13, 17, 20, 23, 27, 32, 37, 41, 44}
KICK_RESTS = {4, 9, 14, 28, 33, 38}                           # the rests of the 5s
KICK = [t for t in range(CYCLE) if t not in SNARE and t not in KICK_RESTS]
assert len(KICK) == 28


def slot_of(t):
    """(row, column, group_size) of sixteenth t in the stacked-rows layout."""
    for r, (s, g) in enumerate(zip(GROUP_START, GROUPS)):
        if s <= t < s + g:
            return r, t - s, g
    raise ValueError(t)


def count_of(t):
    """What a drummer says on sixteenth t: 1..5 inside the group."""
    return slot_of(t)[1] + 1


def is_beat(g):
    """Global sixteenth g (from the start of the music) carries a quarter-note beat."""
    return g % 4 == 0


def beat_count(g, first_beat):
    """Second voice: 1 2 3 4, counted from the first beat that sounds."""
    return ((g - first_beat) // 4) % 4 + 1


def kit_letter(t):
    if t in SNARE:
        return "S"
    if t in KICK_RESTS:
        return ""
    return "K"


if __name__ == "__main__":
    print("sixteenth =", round(SIX * 1000, 2), "ms; cycle =", round(CYCLE * SIX, 3), "s;",
          "4 laps =", round(4 * CYCLE * SIX, 2), "s =", 4 * CYCLE // 4, "beats")
    for lap in range(5):
        phases = sorted({(t) for t in range(CYCLE) if is_beat(lap * CYCLE + t)})
        print("lap", lap, "beats at t =", phases)

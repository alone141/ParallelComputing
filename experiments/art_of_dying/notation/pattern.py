"""The intro of Gojira's "The Art of Dying" (The Way of All Flesh, 2008) as data.

Everything is expressed on a 45-sixteenth cycle, t = 0..44, at 140 bpm.
Sources: Songsterr community tab, Raphael Pannier's transcription, Yogev Gabay's explainer
(see README).  This is an independent study, not affiliated with the band or label.
"""

BPM = 140
SIXTEENTH = 15.0 / BPM            # 0.10714 s
CYCLE = 45                        # sixteenths per cycle
GROUPS = [5, 5, 5, 3, 3, 3, 5, 5, 5, 3, 3]   # 24 + 21 = 45
assert sum(GROUPS) == CYCLE

# group start slot and group index per slot
GROUP_START = []
_s = 0
for g in GROUPS:
    GROUP_START.append(_s)
    _s += g
SLOT_GROUP = []                   # slot -> group index
SLOT_IN_GROUP = []                # slot -> position inside its group (0-based)
for gi, g in enumerate(GROUPS):
    for k in range(g):
        SLOT_GROUP.append(gi)
        SLOT_IN_GROUP.append(k)

# last slot of each group is a rest in the rim-click pattern
REST_SLOTS = {GROUP_START[i] + GROUPS[i] - 1 for i in range(len(GROUPS))}
assert REST_SLOTS == {4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44}
RIM = [t not in REST_SLOTS for t in range(CYCLE)]           # 34 hits

# full band (bars 42+): a 5 is "K K K S .", a 3 is "K K S"
SNARE_SLOTS = {GROUP_START[i] + (3 if GROUPS[i] == 5 else 2) for i in range(len(GROUPS))}
assert SNARE_SLOTS == {3, 8, 13, 17, 20, 23, 27, 32, 37, 41, 44}
KICK_SLOTS = {t for t in range(CYCLE) if t not in SNARE_SLOTS and t not in {4, 9, 14, 28, 33, 38}}
assert len(KICK_SLOTS) == 28 and len(SNARE_SLOTS) == 11

# beats: every 4 sixteenths; the cycle starts on a beat.  After n laps the
# beat sits at slot (−n) mod 4 ... i.e. it slips one sixteenth per lap.
def beat_offset(lap):
    """Slot (0..3) of the first beat inside lap `lap` (lap 0 starts on a beat)."""
    return (-45 * lap) % 4

def is_beat(lap, t):
    return (t - beat_offset(lap)) % 4 == 0

# tab view: ten bars of 4/4 then one bar of 5/4 = 45 beats = 4 laps
TAB_BARS = [4] * 10 + [5]
assert sum(TAB_BARS) == 45

# the two halves people call "24/16" and "21/16"
SYSTEMS = [(0, 24), (24, 45)]

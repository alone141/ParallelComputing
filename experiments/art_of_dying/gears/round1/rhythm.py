"""The facts of the intro of Gojira's "The Art of Dying" (The Way of All Flesh, 2008),
expressed as positions in a 45-sixteenth cycle, plus the film's timing.

Independent study; not affiliated with the band or their label.
"""
import math

BPM = 140.0
SIXTEENTH_S = 15.0 / BPM            # 107.14 ms
CYCLE = 45                          # sixteenths per riff cycle
BEAT = 4                            # sixteenths per quarter-note beat
LAPS = 4                            # laps until the beat realigns: lcm(45, 4) = 180
LOOP_SIXT = CYCLE * LAPS            # 180 sixteenths per film loop
GROUPS = [5, 5, 5, 3, 3, 3, 5, 5, 5, 3, 3]   # 24 + 21 = 45
RESTS = {4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44}   # last slot of each group
RIM_HITS = [t for t in range(CYCLE) if t not in RESTS]  # 34 per cycle

assert sum(GROUPS) == CYCLE
_s, _r = 0, set()
for g in GROUPS:
    _r.add(_s + g - 1)
    _s += g
assert _r == RESTS, _r

# Film timing: 60 fps, whole frames, so the loop is 1157 frames (180 * 15/140 * 60 = 1157.14).
FPS = 60
FRAMES = round(LOOP_SIXT * SIXTEENTH_S * FPS)          # 1157
LOOP_S = FRAMES / FPS                                   # 19.2833 s
FRAMES_PER_SIXT = FRAMES / LOOP_SIXT                    # 6.4278
SIXT_S_FILM = LOOP_S / LOOP_SIXT                        # 107.13 ms (0.01 % faster than the record)


def frame_to_sixt(f):
    """Continuous position in sixteenths (0 .. 180) for frame index f."""
    return f / FRAMES_PER_SIXT


def is_rim(T):
    return (T % CYCLE) not in RESTS


def is_beat(T):
    return T % BEAT == 0


def group_centres():
    """(centre slot, size) of each group, for engraving the 5s and 3s on the riff wheel."""
    out, s = [], 0
    for g in GROUPS:
        out.append((s + (g - 1) / 2.0, g))
        s += g
    return out

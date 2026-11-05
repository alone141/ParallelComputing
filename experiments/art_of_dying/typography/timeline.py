"""Film timeline: which lap does what, and the few words on screen.

Laps of the 45-sixteenth cycle (4.821 s each):
  0      rim clicks alone: the stack of counted digits builds
  1-4    the quarter-note beat joins (ghost digit + orange digits); it drifts
  5      the beat realigns; the kit plays the same 45 as K / S letters
  6      outro: the stack again, credits; ends on a lap boundary so it loops
"""
import rhythm as R

LAPS = ["rim", "beat", "beat", "beat", "beat", "band", "rim"]
FIRST_BEAT = 1 * R.CYCLE                     # the first tribal hit: start of lap 1
FADE = 0.22


def beat(g):
    """Global sixteenth g carries a quarter-note beat (the beat grid starts with lap 1)."""
    return g >= FIRST_BEAT and (g - FIRST_BEAT) % 4 == 0


def lap_mode(lap):
    if 0 <= lap < len(LAPS):
        return LAPS[lap]
    return "none"


def total_seconds():
    return len(LAPS) * R.CYCLE * R.SIX


def total_frames(fps):
    return int(round(total_seconds() * fps))


def glyph_for(lap, t):
    mode = lap_mode(lap)
    if mode in ("rim", "beat"):
        return "" if t in R.RESTS else str(R.count_of(t))
    if mode == "band":
        return R.kit_letter(t)
    return ""


def s(lap, t):
    """Seconds at sixteenth t of a lap."""
    return (lap * R.CYCLE + t) * R.SIX


# (kind, text, start, end)
WORDS = [
    ("title", "COUNT THIS RIFF", -1.0, s(0, 21)),
    ("kicker", "Gojira · The Art of Dying · 140 bpm", s(0, 23), s(4, 40)),
    ("caption", "every click is a count", s(0, 5), s(0, 41)),
    ("caption", "45 counts make one lap", s(0, 44), s(1, 3)),
    ("caption", "now the beat: 1  2  3  4", s(1, 4), s(1, 43)),
    ("sub", "a big hit every 4 counts", s(1, 8), s(1, 43)),
    ("caption", "the beat slips one count", s(2, 3), s(2, 43)),
    ("sub", "45 = 11 × 4 + 1", s(2, 7), s(2, 43)),
    ("caption", "and slips again", s(3, 2), s(3, 43)),
    ("caption", "and again", s(4, 1), s(4, 36)),
    ("headline", "the kit plays the same 45", s(5, 0), s(5, 40)),
    ("caption", "back together after 4 laps", s(5, 0), s(5, 20)),
    ("sub", "45 beats · 19.3 seconds", s(5, 4), s(5, 20)),
    ("caption", "K = kick   S = snare", s(5, 23), s(5, 41)),
    ("sub", "5 = K K K S .    3 = K K S", s(5, 27), s(5, 41)),
    ("headline", "5 5 5 3 3 3 · 5 5 5 3 3 = 45", s(6, 0), s(6, 44)),
    ("caption", "Gojira · The Art of Dying", s(6, 2), s(6, 44)),
    ("sub", "The Way of All Flesh (2008) · an independent study", s(6, 6), s(6, 44)),
]


def captions_at(time):
    out = []
    for kind, text, a, b in WORDS:
        if a - FADE <= time <= b + FADE:
            if time < a + FADE:
                alpha = (time - a) / FADE if time >= a else 0.0
            elif time > b:
                alpha = max(0.0, 1 - (time - b) / FADE)
            else:
                alpha = 1.0
            out.append((kind, text, alpha))
    return out

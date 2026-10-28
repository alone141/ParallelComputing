"""THE SHUFFLE: facts, stops, beats and the master timeline.

Everything that moves in the film, and every sound in the score, is driven by one
function, ``TIMELINE.n_at(t)``: the number of cards dealt from a full deck at time ``t``
seconds.  ``film.py`` and ``soundtrack.py`` both import this module, so picture and sound
cannot drift apart.  ``n`` is a count, not a size: it climbs from 0 to 52 and then stays
there, because 52 factorial is the ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CARDS = 52

# ---------------------------------------------------------------------------
# The numbers (README.md, "The numbers"; re-verified October 2026)
# ---------------------------------------------------------------------------

AGE_GYR = 13.787  # Planck 2018 VI, TT,TE,EE+lowE+lensing+BAO: 13.787 ± 0.020 Gyr
SECONDS_EXACT = AGE_GYR * 1e9 * 365.25 * 86400  # 4.3509 × 10¹⁷ s since the Big Bang
PEOPLE_EXACT = 8_300_678_396  # UN WPP 2024, World, 1 July 2026 (medium variant)
PLANCK_TIME = 5.391247e-44  # s, CODATA 2022 (numerically unchanged from CODATA 2018)
PLANCK_TIMES_EXACT = SECONDS_EXACT / PLANCK_TIME  # 8.0702 × 10⁶⁰
FACTORIAL_52 = math.factorial(CARDS)  # 80658175170943878571660636856403766975289505440883277824000000000000

# As the film shows them (one decimal below ten of a unit, otherwise three significant
# figures).  Counters, size texts and post text all use these, so every surface agrees.
SECONDS = 4.35e17
PEOPLE = 8.3e9
PLANCK_TIMES = 8.07e60
EVERYONE_SECONDS = 3.61e27  # 8.3 billion people × 4.35 × 10¹⁷ s


def orders(k: int) -> int:
    """Possible orders of ``k`` cards dealt from a full deck: 52 × 51 × … × (53 − k)."""
    return math.prod(range(CARDS - k + 1, CARDS + 1))


_LOG2_CUM = [0.0]
for _j in range(CARDS):
    _LOG2_CUM.append(_LOG2_CUM[-1] + math.log2(CARDS - _j))


def log2_orders(n: float) -> float:
    """log₂ of the orders, smooth and monotone in ``n``: between card ``k`` and ``k + 1`` the count
    multiplies by ``52 − k``, so log₂ climbs linearly, and the last card (×1) is flat.  (Γ-interpolation
    would overshoot 52 factorial by 13% between cards 51 and 52.)"""
    n = min(max(float(n), 0.0), float(CARDS))
    k = int(math.floor(n))
    if k >= CARDS:
        return _LOG2_CUM[CARDS]
    return _LOG2_CUM[k] + (n - k) * math.log2(CARDS - k)


def orders_f(n: float) -> float:
    """The orders as a float, smooth in ``n`` (for the live readout between cards)."""
    return 2.0 ** log2_orders(n)


LOG2_ALL = log2_orders(CARDS)  # ≈ 225.58


def ratio_per_shuffle(shuffles: float) -> float:
    """52 factorial ÷ shuffles: at most one order in this many is ever seen."""
    return FACTORIAL_52 / shuffles


# ---------------------------------------------------------------------------
# Colour roles (playbook/film-style.md §3)
# ---------------------------------------------------------------------------

LAVENDER = "#B9A6FF"  # the start and the mechanic
CYAN = "#6CF2FF"  # you, and people standing in for you
LIME = "#D4FF5C"  # time / the Big Bang (new in this film)
VIOLET = "#E08BFF"  # cards: the whole deck (new in this film)
MAGENTA = "#FF6FD8"  # the outro counter's glow only


# ---------------------------------------------------------------------------
# Stops and beats
# ---------------------------------------------------------------------------


@dataclass
class Row:
    """A pass row: the reference fills first in its colour with a live counter, then the
    orders overtake it in lavender.  ``ref`` is the figure as shown, so the counter ends
    on the size text's number."""

    ref: float
    noun: str
    color: str
    notation: str = "words"  # "words" (8.3 billion) or "sci" (4.35 × 10¹⁷), matching the size text


@dataclass
class Stop:
    key: str
    card: int
    title: str
    size_text: str
    compare: str
    accent: str
    dwell: float
    drawing: str | None = None  # crowd | spiral | None
    note: str | None = None
    note_delay: float = 1.2
    row: Row | None = None
    row_delay: float = 0.5
    you: tuple[float, float] | None = None  # YOU pin position inside the drawing (units)
    ref_exact: float | None = None  # for the README's margins


@dataclass
class Beat:
    """An "imagine" beat after the final stop: all those shuffles still reach at most a
    sliver of the orders.  A phase label with its own drawing and row; the HUD holds 52."""

    key: str
    title: str
    size_text: str
    compare: str
    accent: str
    drawing: str
    shuffles: float  # as shown
    shuffles_exact: float
    ratio_text: str  # the figure in the line
    dwell: float


STOPS: list[Stop] = [
    Stop("everyone", 6, "EVERYONE ALIVE", "8.3 billion people", "More orders than [people alive]", CYAN, 3.2,
         drawing="crowd", row=Row(PEOPLE, "people", CYAN, "words"), you=(0.42, 0.38), ref_exact=PEOPLE_EXACT),
    Stop("bigbang", 11, "SINCE THE BIG BANG", "4.35 × 10¹⁷ seconds", "More orders than [every second] so far",
         LIME, 5.4, drawing="spiral", note="Not even [a quarter] of the deck.", note_delay=2.5,
         row=Row(SECONDS, "seconds", LIME, "sci"), ref_exact=SECONDS_EXACT),
    Stop("deck", 52, "THE WHOLE DECK", "8.07 × 10⁶⁷ orders", "52 × 51 × 50 × … × 2 × 1", VIOLET, 4.2,
         note="Imagine a good shuffle\nevery second since the Big Bang.", note_delay=1.3),
]
BY_KEY = {s.key: s for s in STOPS}

BEATS: list[Beat] = [
    Beat("second", "A SHUFFLE A SECOND", "4.35 × 10¹⁷ shuffles", "At most [1 in 1.85 × 10⁵⁰] of the orders", LIME,
         "spiral", SECONDS, SECONDS_EXACT, "1.85 × 10⁵⁰", 3.0),
    Beat("everyone", "EVERYONE SHUFFLING", "3.61 × 10²⁷ shuffles", "At most [1 in 2.23 × 10⁴⁰] of the orders",
         CYAN, "crowd", EVERYONE_SECONDS, PEOPLE_EXACT * SECONDS_EXACT, "2.23 × 10⁴⁰", 3.0),
    Beat("planck", "EVERY PLANCK TIME", "8.07 × 10⁶⁰ shuffles", "Still about [1 in 10 million]", LAVENDER, "planck",
         PLANCK_TIMES, PLANCK_TIMES_EXACT, "10 million", 3.0),
]
BEAT_BY_KEY = {b.key: b for b in BEATS}

# The start and the phase labels (no chip; the size text takes the accent)
START_LABEL = ("52 cards", "A FULL DECK", "Every deal is one [order] of the cards.")
MECHANIC_LABEL = ("52 × 51 × 50 × …", "DEAL A CARD", "Each card has [one fewer] choice.")
SPRINT_LABEL = ("CARDS 12 → 52", "KEEP DEALING", "The last card has [no choice].")
HOOK_LINES = ["How many cards dealt", "have more orders than", "seconds since", "the Big Bang?"]
OUTRO_LINES = ["Every good shuffle", "is almost certainly new."]
HUD_UNIT = "CARDS DEALT"

# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

HOOK = 3.1  # the first card lands here; a 4-line hook gets 0.35 s more than The Doubling's 2.75 s
HOOK_OUT = 2.45  # the question and sub-line start leaving
SUB_IN = 0.85  # "Guess before it ends." enters
HUD_IN = (2.9, 3.35)  # the HUD fades in, after the hook's last glyph has gone (2.45 + 0.28 + 0.15)
START_LABEL_T = (0.45, 3.18)  # the start label rises, then has left before the mechanic's label
MECHANIC_IN = 3.5
STEP_TIMES = (0.70, 0.60, 0.46)  # cards 1-3, one at a time
SETTLE = 0.06
TEACH_HOLD = 0.3  # the hold on card 3 before the first stop
SPRINT = 2.7  # cards 12 → 52
ROW_DURATION = 1.5  # a pass row fills in 1.5 s: the reference over 0.7 s, then the orders over 0.8 s
NOTE_IN = 0.3  # a note enters this long after the comparison line starts leaving, so they never overlap
LABEL_GAP = 0.4  # a label block has left (0.28 s fade plus its last glyph's stagger) before the next one enters
ROW_REF_PART = 0.7
BEAT_ROW_DELAY = 0.5
OUTRO = 4.2
FPS_HINT = 60


def transition_time(dn: float) -> float:
    return 0.5 + 0.028 * dn


def _pchip(xs, ys):
    """Monotone cubic interpolation (Fritsch-Carlson) with flat ends."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.zeros_like(ys)
    for k in range(1, len(xs) - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    def f(x):
        x = np.clip(np.asarray(x, float), xs[0], xs[-1])
        k = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(h) - 1)
        s = (x - xs[k]) / h[k]
        s2, s3 = s * s, s * s * s
        return (
            (2 * s3 - 3 * s2 + 1) * ys[k]
            + (s3 - 2 * s2 + s) * h[k] * m[k]
            + (-2 * s3 + 3 * s2) * ys[k + 1]
            + (s3 - s2) * h[k] * m[k + 1]
        )

    return f


class Timeline:
    def __init__(self):
        kf = [(0.0, 0.0), (HOOK, 0.0)]
        t = HOOK
        for k, dt in enumerate(STEP_TIMES, start=1):
            t += dt
            kf.append((t, float(k)))
            t += SETTLE
            kf.append((t, float(k)))
        t += TEACH_HOLD  # a breath on card 3, so the mechanic's line can be read
        kf.append((t, float(len(STEP_TIMES))))
        self.steps_end = t
        self.arrive: dict[str, float] = {}
        self.leave: dict[str, float] = {}
        self.row_start: dict[str, float] = {}
        self.note_time: dict[str, float] = {}
        prev_n = float(len(STEP_TIMES))
        for s in STOPS:
            t += SPRINT if s.card == CARDS else transition_time(s.card - prev_n)
            self.arrive[s.key] = t
            if s.row:
                self.row_start[s.key] = t + s.row_delay
            if s.note:
                self.note_time[s.key] = t + s.note_delay
            kf.append((t, float(s.card)))
            t += s.dwell
            self.leave[s.key] = t
            kf.append((t, float(s.card)))
            prev_n = float(s.card)
        self.sprint_start = self.leave["bigbang"]
        self.beat_arrive: dict[str, float] = {}
        self.beat_leave: dict[str, float] = {}
        self.beat_row_start: dict[str, float] = {}
        for b in BEATS:
            self.beat_arrive[b.key] = t
            self.beat_row_start[b.key] = t + BEAT_ROW_DELAY
            t += b.dwell
            self.beat_leave[b.key] = t
        self.outro_start = t
        t += OUTRO
        kf.append((t, float(CARDS)))
        self.total = t
        self.keyframes = kf
        self._f = _pchip(*zip(*kf))
        grid = np.linspace(0.0, self.total, int(self.total * 2000) + 1)
        self._grid, self._n_grid = grid, self.n_at(grid)

    def n_at(self, t):
        """Cards dealt at ``t``: exact on every hold (the interpolant is snapped to the integer it
        sits on, so no consumer ever sees 5.999999999999999 on a plateau)."""
        n = np.minimum(self._f(t), float(CARDS))
        r = np.round(n)
        return np.where(np.abs(n - r) < 1e-9, r, n)

    def speed_at(self, t, eps=1e-3):
        return (self._f(t + eps) - self._f(t - eps)) / (2 * eps)

    def card_time(self, k: int) -> float:
        """When card ``k`` lands (the counter flips, the tick and the pop fire)."""
        return float(self._grid[np.argmax(self._n_grid >= k - 0.02)])

    def loop(self, t) -> float:
        """0 → 1 over the last 1.2 s: the glide back to the first frame."""
        t = float(t)
        e0, e1 = self.total - 1.2, self.total - 0.05
        u = min(1.0, max(0.0, (t - e0) / (e1 - e0)))
        return u * u * (3 - 2 * u)


TIMELINE = Timeline()

if __name__ == "__main__":
    tl = TIMELINE
    print(f"total {tl.total:.3f} s, {round(tl.total * FPS_HINT)} frames at {FPS_HINT} fps; teaching ends {tl.steps_end:.2f} s")
    for k in range(1, 4):
        print(f"  card {k} lands at {tl.card_time(k):.2f} s")
    for s in STOPS:
        extra = f"  row {tl.row_start[s.key]:.2f}" if s.row else ""
        extra += f"  note {tl.note_time[s.key]:.2f}" if s.note else ""
        print(f"  {s.key:9s} card {s.card:2d}  arrive {tl.arrive[s.key]:6.2f}  leave {tl.leave[s.key]:6.2f}{extra}")
    for b in BEATS:
        print(f"  beat {b.key:9s} {tl.beat_arrive[b.key]:6.2f} → {tl.beat_leave[b.key]:6.2f}")
    print(f"  outro {tl.outro_start:.2f} → {tl.total:.2f}")
    print()
    print("orders check:")
    for s in STOPS:
        if s.ref_exact:
            print(f"  card {s.card}: {orders(s.card):.4e} = ×{orders(s.card) / s.ref_exact:.3f} the reference; "
                  f"one card earlier ×{orders(s.card - 1) / s.ref_exact:.3f}")
    print(f"  52 factorial = {FACTORIAL_52} ≈ {FACTORIAL_52:.4e}; log₂ = {LOG2_ALL:.3f}")
    for b in BEATS:
        r = ratio_per_shuffle(b.shuffles_exact)
        print(f"  beat {b.key:9s}: {b.shuffles_exact:.4e} shuffles → 1 in {r:.5e}; shown as {b.ratio_text}")

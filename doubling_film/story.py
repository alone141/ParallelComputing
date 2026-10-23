"""Facts, milestones and the master timeline.

Everything that moves in the film (and every sound in the soundtrack) is driven
by one function: ``n_at(t)``, the number of doublings of the Planck length
reached at time ``t`` seconds.  Both ``film.py`` and ``soundtrack.py`` import
this module so picture and sound can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PLANCK = 1.616255e-35  # metres (CODATA 2018)
LIGHT_YEAR = 9.4607e15  # metres
AU = 1.495979e11  # metres

FPS_HINT = 60


def doublings(size_m: float) -> float:
    """How many times the Planck length must be doubled to reach ``size_m``."""
    return float(np.log2(size_m / PLANCK))


@dataclass
class Row:
    """The comparison, acted out: ``count`` copies of the previous thing lined
    up from ``start`` to ``end`` (units of this milestone's size)."""

    count: float
    noun: str
    start: tuple[float, float]
    end: tuple[float, float]
    color: str
    label_side: str = "up"  # where the live counter sits: up | down | left | right
    label_offset: float = 0.3  # distance of an up/down counter from the row


@dataclass
class Milestone:
    key: str
    size: float  # characteristic size in metres (drawn as the reference diameter)
    title: str
    size_text: str  # friendly size shown above the title
    compare: str  # the relative example
    accent: str
    dwell: float  # seconds the camera lingers
    note: str | None = None  # replaces ``compare`` half-way through the dwell
    note_delay: float = 1.2
    # Where the previous milestone sits inside this one, in units of this size
    # (0, 0) is the centre and 0.5 is the edge of the reference circle.
    sit: tuple[float, float] = (0.0, 0.0)
    # Which point of this object is placed at the next milestone's ``sit``.
    attach: tuple[float, float] = (0.0, 0.0)
    # Where "you" are inside this object (drives the YOU pin), or None.
    you: tuple[float, float] | None = None
    row: Row | None = None
    row_delay: float = 0.5  # seconds after arrival that the row starts filling
    n: float = field(init=False)

    def __post_init__(self):
        self.n = doublings(self.size)


# Everest's summit position in the drawing (x), see visuals.Everest.
EVEREST_SUMMIT_X = 0.04


def _earth_nepal() -> tuple[float, float]:
    """Everest's position on the Earth drawing (orthographic projection)."""
    lon0, lat0 = np.radians(45.0), np.radians(18.0)
    lon, lat = np.radians(86.92), np.radians(27.99)
    x = np.cos(lat) * np.sin(lon - lon0)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)
    return 0.5 * float(x), 0.5 * float(y)


EARTH_NEPAL = _earth_nepal()
# The galaxy texture is drawn GALAXY_SCALE x the nominal diameter so that its
# visible (bright) disc spans the 100,000 ly reference.  The Sun's place in it,
# in units of the nominal diameter: 26,700 ly out, on the Orion spur.
GALAXY_SCALE = 1.12
GALAXY_SUN = (0.5 * 0.534 * GALAXY_SCALE * np.cos(np.radians(-118)),
              0.5 * 0.534 * GALAXY_SCALE * np.sin(np.radians(-118)))
SOLAR_EARTH = (0.5 / 30.07, 0.0)  # 1 AU in units of Neptune's orbital diameter

MILESTONES: list[Milestone] = [
    Milestone(
        "planck", PLANCK, "THE PLANCK LENGTH", "1.6 × 10⁻³⁵ m",
        "Where known physics runs out", "#B9A6FF", 0.0,
    ),
    Milestone(
        "proton", 1.68e-15, "A PROTON", "0.0000000000000017 m",
        "About [100 quintillion] Planck lengths wide", "#FF8A5B", 3.2,
        row=Row(1e20, "Planck lengths", (-0.5, 0.0), (0.5, 0.0), "#B9A6FF"),
    ),
    Milestone(
        "atom", 1.06e-10, "A HYDROGEN ATOM", "0.1 nanometres",
        "[≈60,000×] wider than its proton", "#5CC8FF", 3.2,
        row=Row(60_000, "protons", (-0.5, 0.0), (0.5, 0.0), "#FF8A5B"),
    ),
    Milestone(
        "virus", 1.0e-7, "A VIRUS", "100 nanometres",
        "About [1,000] atoms across", "#FF5FA2", 3.2,
        row=Row(1_000, "atoms", (-0.5, 0.0), (0.5, 0.0), "#5CC8FF"),
    ),
    Milestone(
        "rbc", 7.5e-6, "A RED BLOOD CELL", "7.5 micrometres",
        "[75] viruses side by side", "#FF4D5E", 3.2,
        row=Row(75, "viruses", (-0.5, 0.0), (0.5, 0.0), "#FF8FCB"),
    ),
    Milestone(
        "ant", 4.0e-3, "AN ANT", "4 millimetres",
        "[530] blood cells, nose to tail", "#FFB547", 3.2,
        row=Row(530, "blood cells", (-0.48, -0.25), (0.39, -0.25), "#FF4D5E", label_side="down"),
    ),
    Milestone(
        "human", 1.7, "YOU", "1.7 metres",
        "[425] ants tall", "#6CF2FF", 3.8,
        note="Doubling [116]. Remember that.", note_delay=2.3,
        row=Row(425, "ants", (-0.3, -0.5), (-0.3, 0.5), "#FFB547", label_side="left"),
        sit=(0.16, -0.47), attach=(0.0, -0.5), you=(0.0, 0.51),
    ),
    Milestone(
        "everest", 8849.0, "MOUNT EVEREST", "8,849 metres",
        "[5,200] people stacked up", "#A9D6FF", 3.2,
        row=Row(5_200, "people", (EVEREST_SUMMIT_X, -0.5), (EVEREST_SUMMIT_X, 0.5), "#6CF2FF", label_side="right"),
        sit=(EVEREST_SUMMIT_X, 0.5), attach=(EVEREST_SUMMIT_X, -0.5),
        you=(EVEREST_SUMMIT_X, 0.5),
    ),
    Milestone(
        "earth", 1.2742e7, "EARTH", "12,742 km",
        "[1,440] Everests wide", "#4DA3FF", 3.2,
        row=Row(1_440, "Everests", (-0.5, 0.0), (0.5, 0.0), "#A9D6FF"),
        sit=EARTH_NEPAL, you=EARTH_NEPAL,
    ),
    Milestone(
        "sun", 1.3927e9, "THE SUN", "1.39 million km",
        "[109] Earths across", "#FFB13B", 3.2,
        row=Row(109, "Earths", (-0.5, 0.0), (0.5, 0.0), "#8CCBFF"),
        sit=(0.535, 0.0), you=(0.535, 0.0),
    ),
    Milestone(
        "solar", 2 * 30.07 * AU, "OUR SOLAR SYSTEM", "out to Neptune",
        "[6,500] Suns wide", "#8FB8FF", 3.2,
        row=Row(6_500, "Suns", (-0.5, 0.0), (0.5, 0.0), "#FFB13B", label_side="down"),
        you=SOLAR_EARTH,
    ),
    Milestone(
        "star", 4.2465 * LIGHT_YEAR, "THE NEAREST STAR", "4.2 light-years away",
        "[4,500] solar systems away", "#FF8C6B", 3.2,
        row=Row(4_500, "solar systems", (-0.47, 0.0), (0.47, 0.0), "#8FB8FF", label_side="down", label_offset=0.85),
        sit=(-0.5, 0.0), attach=(-0.5, 0.0), you=(-0.5, 0.0),
    ),
    Milestone(
        "milkyway", 1.0e5 * LIGHT_YEAR, "THE MILKY WAY", "100,000 light-years",
        "[23,500] trips to the nearest star", "#C7A6FF", 5.4,
        note="You're closer to [a galaxy]\nthan to the Planck length.",
        note_delay=2.5, sit=GALAXY_SUN, you=GALAXY_SUN,
        row=Row(23_500, "trips", (-0.5, 0.0), (0.5, 0.0), "#FF8C6B"),
    ),
    Milestone(
        "universe", 93e9 * LIGHT_YEAR, "THE OBSERVABLE UNIVERSE", "93 billion light-years",
        "Just [20] more doublings", "#FF6FD8", 4.6,
        note="[930,000] Milky Ways across", note_delay=1.3, you=(0.0, 0.0),
        row=Row(930_000, "Milky Ways", (-0.5, 0.0), (0.5, 0.0), "#C7A6FF", label_side="down"), row_delay=1.45,
    ),
]

BY_KEY = {m.key: m for m in MILESTONES}
N_HUMAN = BY_KEY["human"].n
N_GALAXY = BY_KEY["milkyway"].n
N_END = BY_KEY["universe"].n

# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

HOOK = 2.75  # seconds of question before the first doubling
STEP_TIMES = (0.40, 0.36, 0.33, 0.31, 0.29, 0.27, 0.26, 0.25)  # the first doublings, one by one
ROW_DURATION = 1.5  # seconds a comparison row takes to fill
ARRIVE_BEFORE = 0.15  # dwell starts this many doublings before a milestone...
LEAVE_AFTER = 0.25  # ...and ends this many after it
OUTRO = 4.2


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
            t += 0.06
            kf.append((t, float(k)))
        self.steps_end = t
        self.arrive: dict[str, float] = {"planck": 0.0}
        self.leave: dict[str, float] = {"planck": HOOK}
        self.row_start: dict[str, float] = {}
        t += 2.45  # the long sprint through the empty zone
        prev_n = float(len(STEP_TIMES))
        for m in MILESTONES[1:]:
            if m.key != "proton":
                t += transition_time(m.n - prev_n)
            self.arrive[m.key] = t
            if m.row:
                self.row_start[m.key] = t + m.row_delay
            kf.append((t, m.n - ARRIVE_BEFORE))
            t += m.dwell
            self.leave[m.key] = t
            kf.append((t, m.n + LEAVE_AFTER))
            prev_n = m.n
        self.outro_start = t
        t += OUTRO
        kf.append((t, N_END + 0.75))
        self.total = t
        self.keyframes = kf
        self._f = _pchip(*zip(*kf))

    def n_at(self, t):
        return self._f(t)

    def speed_at(self, t, eps=1e-3):
        return (self._f(t + eps) - self._f(t - eps)) / (2 * eps)


TIMELINE = Timeline()

if __name__ == "__main__":
    tl = TIMELINE
    print(f"total {tl.total:.2f}s")
    for m in MILESTONES:
        print(f"{m.key:9s} n={m.n:7.2f}  arrive={tl.arrive[m.key]:6.2f}s  leave={tl.leave[m.key]:6.2f}s"
              f"  row={tl.row_start.get(m.key, float('nan')):6.2f}s")

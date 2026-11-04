"""The rhythm and the timeline: the single source for the picture and the sound.

Gojira, "The Art of Dying" (The Way of All Flesh, 2008), intro. Every part is placed on one
45-sixteenth cycle at 140 bpm. Derived from the Songsterr transcription (song 71013, revision
9988433); see README.md for what was taken and what was simplified.

Notation used on screen and here:
    n  sixteenths since the film started (0, 1, 2, ...)
    t  position in the lap, t = n mod 45 (0 ... 44)
    lap = n // 45

    python rhythm.py        # self-check and a printed summary
"""

from __future__ import annotations

from dataclasses import dataclass

# ---- tempo and cycle --------------------------------------------------------------------------
BPM = 140
SIXTEENTH = 15 / BPM                     # seconds per sixteenth (0.10714... s)
GROUPS = (5, 5, 5, 3, 3, 3, 5, 5, 5, 3, 3)
HALVES = (24, 21)                        # 5+5+5+3+3+3, then 5+5+5+3+3
CYCLE = sum(GROUPS)                      # 45 sixteenths per lap
LAP = CYCLE * SIXTEENTH                  # 4.8214... s

GROUP_STARTS = tuple(sum(GROUPS[:i]) for i in range(len(GROUPS)))   # 0 5 10 15 18 21 24 29 34 39 42

# Rim clicks (and bass): every group is all hits except its last sixteenth.
#   5 -> x x x x .     3 -> x x .
RIM_RESTS = frozenset(s + g - 1 for s, g in zip(GROUP_STARTS, GROUPS))
# Kick + snare:  5 -> K K K S .     3 -> K K S
SNARES = frozenset(s + (3 if g == 5 else 2) for s, g in zip(GROUP_STARTS, GROUPS))
# KICK_RESTS: the slots where neither drum plays (the kick is also silent on every snare slot).
KICK_RESTS = frozenset(s + 4 for s, g in zip(GROUP_STARTS, GROUPS) if g == 5)
KICKS = frozenset(range(CYCLE)) - SNARES - KICK_RESTS


# The 4/4 grid (pulse, ride, bell). The pulse enters at n = 90 (lap 2), and 90 = 2 (mod 4), so the
# beat sits on n = 2 (mod 4): every 4/4 layer then enters on a hit, at t = 0.
BEAT_PHASE = 2


def t_of(n: int) -> int:
    return n % CYCLE


# ---- the six layers ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layer:
    key: str
    name: str            # lane label, CAPS
    colour: str          # lane colour
    equation: tuple      # lines shown once, during the layer's focus lap
    pitch_midi: int | None = None


LAYERS = (
    Layer("rim", "RIM CLICKS", "#FFB547",
          ("t = n mod 45",
           "rim(t) = 1  ⇔  t ∉ R",
           "R = {4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44}")),
    Layer("pulse", "TRIBAL PULSE", "#FF5A4E",
          ("pulse(n) = 1  ⇔  n ≡ 2 (mod 4)",
           "45 ≡ 1 (mod 4)")),
    Layer("ks", "KICK + SNARE", "#4DA3FF",
          ("S = {3, 8, 13, 17, 20, 23, 27, 32, 37, 41, 44}",
           "K = ℤ₄₅ ∖ (S ∪ {4, 9, 14, 28, 33, 38})")),
    Layer("ride", "RIDE", "#7CF0C0",
          ("ride(n) = 1  ⇔  n ≡ 0 (mod 4)",
           "bell(n) = 1  ⇔  n ≡ 2 (mod 4)")),
    Layer("bass", "BASS", "#B9A6FF",
          ("bass(t) = rim(t)",
           "pitch: C1 (32.7 Hz)"),
          pitch_midi=24),
    Layer("gtr", "GUITAR", "#A9D6FF",
          ("gtr(n) = 1  ∀ n",
           "muted, on C2 (65.4 Hz)"),
          pitch_midi=36),
)
SNARE_COLOUR = "#F5F7FF"
BY_KEY = {layer.key: layer for layer in LAYERS}


def hits_at(key: str, n: int) -> list[str]:
    """What layer `key` plays on sixteenth n. Returns a list of hit kinds (empty = silence).

    Kinds: rim: "click" · pulse: "pulse" (big stacked drum, with a stick grace note just before it)
    ks: "kick" | "snare" · ride: "ride" (soft, ghosted, off-beat) | "bell" (on the beat)
    bass: "note" · gtr: "chug" (palm-muted dead note)
    """
    t = t_of(n)
    if key in ("rim", "bass"):
        return [] if t in RIM_RESTS else (["click"] if key == "rim" else ["note"])
    if key == "pulse":
        return ["pulse"] if n % 4 == BEAT_PHASE else []
    if key == "ks":
        return ["snare"] if t in SNARES else (["kick"] if t in KICKS else [])
    if key == "ride":
        return ["bell"] if n % 4 == BEAT_PHASE else (["ride"] if n % 4 == (BEAT_PHASE + 2) % 4 else [])
    if key == "gtr":
        return ["chug"]
    raise KeyError(key)


def is_group_start(n: int) -> bool:
    """First sixteenth of a 5 or a 3 (a natural accent; an interpretive choice, not in the tab)."""
    return t_of(n) in GROUP_STARTS


# ---- the timeline -----------------------------------------------------------------------------
FPS = 60
LAPS_PER_LAYER = 2                       # lap 1: focus (equation on screen) · lap 2: settle
FULL_MIX_LAPS = 1
N_LAPS = LAPS_PER_LAYER * len(LAYERS) + FULL_MIX_LAPS          # 13
N_SIXTEENTHS = N_LAPS * CYCLE                                   # 585
TOTAL = N_LAPS * LAP                                            # 62.678... s: exactly 13 laps
N_FRAMES = round(TOTAL * FPS)                                   # 3761

# Loop: over the last PEEL seconds the layers above the rim clicks leave, top lane first, so the
# last frame matches frame 0 (rim clicks alone) and the music restarts on the grid when it loops.
PEEL = 1.2
PEEL_FADE = 0.45                         # each layer's own fade length
FOCUS_DIM = 0.45                         # opacity of the other lanes during a focus lap
FOCUS_DUCK_DB = -4.0                     # level of the other layers during a focus lap
EASE = 0.3                               # seconds for focus dim/duck ramps


def lap_start(lap: int) -> float:
    return lap * LAP


def entry_lap(key: str) -> int:
    return [layer.key for layer in LAYERS].index(key) * LAPS_PER_LAYER


def entry_time(key: str) -> float:
    return lap_start(entry_lap(key))


def focus_window(key: str) -> tuple[float, float]:
    """The layer's focus lap: (start, end) in seconds."""
    s = entry_time(key)
    return s, s + LAP


def equation_window(key: str) -> tuple[float, float, float, float]:
    """(in_start, in_end, out_start, out_end): about 3.8 s fully readable."""
    s, e = focus_window(key)
    return s + 0.45, s + 0.9, e - 0.65, e - 0.35


def peel_window(key: str) -> tuple[float, float] | None:
    """Fade-out window at the end of the film, or None for the rim clicks (they never leave)."""
    i = [layer.key for layer in LAYERS].index(key)
    if i == 0:
        return None
    order = len(LAYERS) - i              # guitar leaves first (order 1), pulse last (order 5)
    step = (PEEL - PEEL_FADE) / (len(LAYERS) - 2)
    start = TOTAL - PEEL + (order - 1) * step
    return start, start + PEEL_FADE


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def presence(key: str, time: float) -> float:
    """0..1: is the layer in the film at this time (entered, not yet peeled away)."""
    if time < entry_time(key):
        return 0.0
    pw = peel_window(key)
    if pw is None:
        return 1.0
    return 1.0 - _smooth((time - pw[0]) / (pw[1] - pw[0]))


def focus_weight(key: str, time: float) -> float:
    """0..1: how strongly `key` is the focused layer right now (eased at both ends)."""
    s, e = focus_window(key)
    if key == LAYERS[0].key:             # the opening focus starts at frame 0, so no ease-in
        return 1.0 - _smooth((time - (e - EASE)) / EASE) if s <= time < e + EASE else 0.0
    return _smooth((time - s) / EASE) * (1.0 - _smooth((time - (e - EASE)) / EASE)) if s - EASE <= time < e + EASE else 0.0


def focused_layer(time: float) -> str | None:
    lap = int(time // LAP)
    if lap < len(LAYERS) * LAPS_PER_LAYER and lap % LAPS_PER_LAYER == 0:
        return LAYERS[lap // LAPS_PER_LAYER].key
    return None


def background_factor(key: str, time: float) -> float:
    """0..1: how far `key` is pushed back because another layer is in focus."""
    return max((focus_weight(o.key, time) for o in LAYERS if o.key != key), default=0.0)


def lane_opacity(key: str, time: float) -> float:
    """Picture: the lane's overall opacity (presence, dimmed while another layer is in focus)."""
    return presence(key, time) * (1.0 - (1.0 - FOCUS_DIM) * background_factor(key, time))


def layer_gain(key: str, time: float) -> float:
    """Sound: linear gain for the layer at this time (presence x focus duck)."""
    duck = 10 ** (FOCUS_DUCK_DB / 20)
    return presence(key, time) * (1.0 - (1.0 - duck) * background_factor(key, time))


@dataclass(frozen=True)
class Hit:
    n: int
    time: float
    key: str
    kind: str
    accent: bool
    gain: float          # layer gain at that moment (presence x duck)


def events() -> list[Hit]:
    """Every hit in the film, in time order. The sound renders these; the picture flashes them."""
    out = []
    for n in range(N_SIXTEENTHS):
        time = n * SIXTEENTH
        for layer in LAYERS:
            if time < entry_time(layer.key):
                continue
            g = layer_gain(layer.key, time)
            if g <= 1e-4:
                continue
            for kind in hits_at(layer.key, n):
                out.append(Hit(n, time, layer.key, kind, is_group_start(n), g))
    return out


# ---- self-check -------------------------------------------------------------------------------
def _check():
    assert CYCLE == 45 and sum(HALVES) == CYCLE
    assert sorted(RIM_RESTS) == [4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44]
    assert sorted(SNARES) == [3, 8, 13, 17, 20, 23, 27, 32, 37, 41, 44]
    assert sorted(KICK_RESTS) == [4, 9, 14, 28, 33, 38]
    assert len(KICKS) == 28 and len(SNARES) == 11 and CYCLE - len(RIM_RESTS) == 34
    # the kick/snare lap as the tab spells it
    ks = "".join("S" if t in SNARES else "K" if t in KICKS else "." for t in range(CYCLE))
    assert ks == "KKKS.KKKS.KKKS.KKSKKSKKSKKKS.KKKS.KKKS.KKSKKS", ks
    rim = "".join("." if t in RIM_RESTS else "x" for t in range(CYCLE))
    assert rim == "xxxx.xxxx.xxxx.xx.xx.xx.xxxx.xxxx.xxxx.xx.xx.", rim
    # the beat slips one sixteenth per lap and is home after 4 laps
    assert CYCLE % 4 == 1 and (4 * CYCLE) % 4 == 0
    # the equation text matches the sets
    assert BY_KEY["rim"].equation[2] == "R = {" + ", ".join(map(str, sorted(RIM_RESTS))) + "}"
    assert BY_KEY["ks"].equation[0] == "S = {" + ", ".join(map(str, sorted(SNARES))) + "}"
    assert BY_KEY["ks"].equation[1] == "K = ℤ₄₅ ∖ (S ∪ {" + ", ".join(map(str, sorted(KICK_RESTS))) + "})"
    # timeline
    assert N_LAPS == 13 and N_FRAMES == round(TOTAL * FPS)
    assert presence("rim", 0) == 1 and presence("pulse", 0) == 0
    assert all(presence(layer.key, TOTAL) < 1e-6 for layer in LAYERS[1:])
    assert presence("rim", TOTAL) == 1
    assert focused_layer(0.0) == "rim" and focused_layer(entry_time("gtr") + 1) == "gtr"
    assert focused_layer(entry_time("gtr") + LAP + 1) is None and focused_layer(TOTAL - 1) is None
    # every layer enters on a hit, at t = 0
    for layer in LAYERS:
        n0 = entry_lap(layer.key) * CYCLE
        assert t_of(n0) == 0 and hits_at(layer.key, n0), layer.key
    assert BY_KEY["pulse"].equation[0].endswith(f"n ≡ {BEAT_PHASE} (mod 4)")


if __name__ == "__main__":
    _check()
    ev = events()
    print(f"{BPM} bpm · sixteenth {SIXTEENTH * 1000:.2f} ms · lap {LAP:.4f} s · {N_LAPS} laps")
    print(f"total {TOTAL:.4f} s · {N_FRAMES} frames @ {FPS} fps · {len(ev)} hits")
    for layer in LAYERS:
        s, e = focus_window(layer.key)
        print(f"  {layer.name:<13} enters {entry_time(layer.key):6.2f} s · focus {s:6.2f}–{e:6.2f} s · "
              f"peel {peel_window(layer.key)}")
    print("self-check passed")

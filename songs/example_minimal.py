"""A minimal, made-up song that exercises what Art of Dying does not: a tempo map (two bpms), an
eighth-note step, 5/8 + 7/8 bars, a compound "herta" tooth, and a part that leaves and comes back.
Not a real transcription. Use it as a template: copy, rename, replace every number with the tab's.

    python3 -m gears_film check songs/example_minimal.py
    python3 -m gears_film still songs/example_minimal.py 6.0 /tmp/x.png
"""
from gears_film.song import Song, Segment, Section, Gear, Moment, Counter, SlipSpec, Anchor

P = 42.0
BARS = (5, 7)                         # eighths per bar: a 5/8 and a 7/8, 12 steps per cycle
CYCLE = sum(BARS)
tall = lambda ps: {p: "tall" for p in ps}

GEARS = [
    # the riff gear: one tooth position per eighth of the two-bar cycle, strikes on 0 1 3 | 5 6 8 10
    Gear("riff", "RIFF", "RIFF", (226, 206, 166), 12, tall((0, 1, 3, 5, 6, 8, 10)), +1, 0 * P, ((0, 96),), sound="chug", white_position=0),
    # a quarter-note pulse: 2 steps per beat, a 4-tooth gear (4 = lcm-friendly: 96 % 4 == 0, 24 % 4 == 0)
    Gear("pulse", "PULSE", "PULSE", (255, 170, 60), 4, tall((0,)), -1, 0 * P, ((24, 72), (84, 96)), sound="pulse"),
    # a herta figure on every other eighth: a compound tooth strikes three times inside one step
    Gear("herta", "HERTA", "HERTA", (125, 255, 154), 12, {p: "herta" for p in range(0, 12, 2)}, +1, 8 * P, ((48, 96),), sound="kick"),
]

SONG = Song(
    slug="example_minimal", title="Example", artist="Nobody", album="A Template", year=2026,
    bpm=100.0, steps_per_beat=2,
    segments=[Segment(0, 48, bpm=100.0, label="slow"), Segment(72, 48, bpm=120.0, label="fast")],   # 24 record steps cut
    sections=[Section("A", 0, BARS * 4), Section("cut", 48, BARS * 2), Section("B", 72, BARS * 4)],
    gears=GEARS,
    moments=[Moment(0, "A TEMPLATE", "Replace everything here with the tab's facts.", "hook"),
             Moment(24, "THE PULSE ENGAGES", "12 against 4: three beats per cycle, no drift at all.", "big", "pulse"),
             Moment(48, "FASTER, AND A HERTA", "The second segment runs at 120 bpm; the green gear's teeth strike three times each.", "big", "herta")],
    card_lines=["A made-up example."],
    credit_lines=["Not a real song."],
    counters=[Counter(("riff",), BARS, 0, 96)],
    slip=SlipSpec(cycle_gear="riff", beat_steps=2, beat_gears=("pulse",)),
    cycle_steps=CYCLE, cycle_end=96,
    stack=["riff", "pulse", "herta"],
    anchors=[Anchor("pulse", (40, 160), ("pulse",))],
    cut_note="cut: two bars removed",
)

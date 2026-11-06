"""Gojira, "The Art of Dying" (The Way of All Flesh, 2008): the intro as gears.

The worked example for gears_film. Everything the engine needs to know about the song is here.
Independent study; not affiliated with the band or their label. The recording is not in the repo:
the film is silent and the owner muxes the record using record_map.json.

Facts (verified against Songsterr's community tab, Raphael Pannier's transcription and Yogev
Gabay's explainer): 140 bpm; one 45-sixteenth cycle grouped 5+5+5+3+3+3 then 5+5+5+3+3; rim
clicks on every sixteenth but the last of each group; a stacked "tribal" hit on every beat; from
bar 42 kick (K K K S . / K K S), snare, ghosted ride with the bell on the beat, bass on the rim
pattern on C1, two palm-muted guitars on every sixteenth; a 6/4 turnaround at bar 53.

Step = one sixteenth (4 per beat). Film: record steps 0-539 (bars 1-33), then 668-871 (bars
42-53); bars 34-41 (the pulse alone, cycle suspended) are cut.
"""
from gears_film.song import Song, Segment, Section, Gear, Moment, Counter, SlipSpec, CameraMove, Anchor

P = 42.0                                      # the rack pitch (world units): gear x must be a multiple
GROUPS = (5, 5, 5, 3, 3, 3, 5, 5, 5, 3, 3)
CYCLE = sum(GROUPS)                           # 45
STARTS = tuple(sum(GROUPS[:i]) for i in range(len(GROUPS)))
RESTS = frozenset(s + g - 1 for s, g in zip(STARTS, GROUPS))                  # 4 9 14 17 20 23 28 33 38 41 44
HITS = tuple(t for t in range(CYCLE) if t not in RESTS)                        # 34
SNARES = frozenset(s + (3 if g == 5 else 2) for s, g in zip(STARTS, GROUPS))   # 11
KICK_RESTS = frozenset(s + 4 for s, g in zip(STARTS, GROUPS) if g == 5)
KICKS = frozenset(range(CYCLE)) - SNARES - KICK_RESTS                          # 28
HELD = frozenset(t for t in HITS if (t + 1) % CYCLE in RESTS)                  # the group-final bass notes

K_PULSE, K_BAND, K_TURN, K_END = 180, 540, 720, 744

TAN, AMBER, RED, WHITE = (226, 206, 166), (255, 170, 60), (255, 92, 70), (255, 255, 255)
VIOLET, VIOLET2, GREEN, GREEN2, SKY = (182, 140, 255), (226, 206, 255), (125, 255, 154), (206, 255, 214), (108, 224, 255)

tall = lambda positions: {p: "tall" for p in positions}
ALL12 = tall(range(12))
BEAT12 = tall((0, 4, 8))
SECOND_HALF = frozenset(p for p in range(12) if (p % 6) // 3 == 1)             # C2 / E-flat 2 cells

GEARS = [
    Gear("rim", "RIM", "RIM CLICKS", TAN, 45, tall(HITS), +1, 0 * P, ((0, K_BAND),), sound="click", white_position=0),
    Gear("pulse", "PULSE", "PULSE", AMBER, 12, BEAT12, -1, 0 * P, ((K_PULSE, K_BAND),), sound="pulse"),
    Gear("snare", "SNARE", "SNARE", WHITE, 45, tall(SNARES), -1, 11 * P, ((K_BAND, K_TURN),), sound="snare"),
    Gear("kick", "KICK", "KICK", RED, 45, tall(KICKS), +1, 17 * P, ((K_BAND, K_TURN),), sound="kick", lead=0.1),
    Gear("bass", "BASS", "BASS", VIOLET, 45, {**tall(HITS), **{p: "long" for p in HELD}}, -1, 28 * P, ((K_BAND, K_TURN),),
         sound="note", lead=0.2),
    Gear("gtr", "GTRS", "GTRS", GREEN, 12, ALL12, +1, 31 * P, ((K_BAND, K_TURN),), sound="chug", lead=0.3, label_dx=-44),
    Gear("ride", "RIDE", "RIDE", SKY, 12, {**BEAT12, 2: "medium", 6: "medium", 10: "medium"}, +1, 36 * P, ((K_BAND, K_END),),
         sound="ride", first_hit="crash", lead=0.4, label_dx=44, label_dy=40),
    # the 6/4 turnaround: K K K S . S four times; bass C1 x3 / C2 x3; guitars C2 / E-flat 2 in threes
    Gear("tbass", "BASS", "BASS 6/4", VIOLET, 12, ALL12, -1, 39 * P, ((K_TURN, K_END),), sound="note",
         colour2=VIOLET2, colour2_positions=SECOND_HALF, stack="bass"),
    Gear("tkick", "KICK", "KICK 6/4", RED, 12, tall(p for p in range(12) if p % 6 in (0, 1, 2)), +1, 42 * P, ((K_TURN, K_END),),
         sound="kick", stack="kick", lead=0.1),
    Gear("tgtr", "GTRS", "GTRS 6/4", GREEN, 12, ALL12, -1, 45 * P, ((K_TURN, K_END),), sound="chug",
         colour2=GREEN2, colour2_positions=SECOND_HALF, stack="gtr", lead=0.2),
    Gear("tsnare", "SNARE", "SNARE 6/4", WHITE, 12, tall(p for p in range(12) if p % 6 in (3, 5)), +1, 48 * P, ((K_TURN, K_END),),
         sound="snare", stack="snare", lead=0.3),
]

MOMENTS = [
    Moment(0, "WHY DOES THIS BEAT DRIFT?", "Gojira, The Art of Dying. The intro as a machine: one tooth per sixteenth.", "hook"),
    Moment(45, "ONE WHEEL, 45 TEETH", "Eleven are missing: the rests. Five, five, five, three, three, three, then five, five, five, three, three."),
    Moment(90, "FIVES AND THREES", "x x x x · and x x ·. The dots on the left count each group as it strikes."),
    Moment(135, "A SECOND WHEEL IS COMING", "Twelve teeth, three of them amber: the beat. It slides in next lap."),
    Moment(180, "THE PULSE ENGAGES", "45 = 11 × 4 + 1. The small wheel turns 3¾ times a lap, so the beat lands one tooth later every lap.", "big", "pulse"),
    Moment(225, "ONE TOOTH BEHIND EVERY LAP", "45 = 11 × 4 + 1. The spare tooth piles up."),
    Moment(270, "TWO TEETH", "The amber tooth now lands two teeth before the white one."),
    Moment(315, "THREE TEETH", "One more lap and the slip is a whole beat."),
    Moment(360, "HOME", "Four laps, 45 beats, 19.3 seconds. Now hear it drift again.", "home"),
    Moment(405, "AGAIN: ONE TOOTH", "Same wheels, same slip. Only where you count from has changed."),
    Moment(450, "TWO", "Fans call this riff 21/16 or 45/16. Both are half right: the 21 is the second half of the 45."),
    Moment(495, "THREE", "Next lap it is home again, and the band is waiting on the inner tracks."),
    Moment(540, "HOME. THE BAND.", "Five new wheels take the rack: kick, snare, bass, guitars, ride. The clicks and the pulse roll away.", "home"),
    Moment(585, "KICK, SNARE, BASS: 45 TEETH EACH", "28 kicks, 11 snares, 34 bass notes. Short teeth are rests; a long tooth is a held note.", colour="kick"),
    Moment(630, "GUITARS AND RIDE: 12 TEETH", "The guitars strike every tooth. The ride's tall teeth are the bell, on the beat.", colour="ride"),
    Moment(675, "ONE MORE LAP, THEN A TURN", "Four small wheels swap in: K K K S · S, four times. A bar of 6/4.", colour="kick"),
    Moment(720, "THE TURNAROUND", "The 45s roll away. Six-tooth cells go round twice, then the riff starts over.", colour="kick"),
]

# the record's bars: each 180-sixteenth stretch is written as ten bars of 4/4 and one of 5/4
ELEVEN = (16,) * 10 + (20,)
SECTIONS = [
    Section("clicks alone (bars 1-11)", 0, ELEVEN),
    Section("clicks + pulse (bars 12-33)", 180, ELEVEN * 2),
    Section("pulse alone, cut (bars 34-41)", 540, (16,) * 8),
    Section("the band (bars 42-52)", 668, ELEVEN),
    Section("6/4 turnaround (bar 53)", 848, (24,)),
]

SONG = Song(
    slug="art_of_dying",
    title="The Art of Dying", artist="Gojira", album="The Way of All Flesh", year=2008,
    bpm=140.0, steps_per_beat=4, first_click_offset_s=0.0,
    segments=[
        Segment(record_start=0, steps=540, label="clicks and pulse"),
        Segment(record_start=668, steps=180, label="the band"),
        Segment(record_start=848, steps=24, label="6/4 TURN"),
    ],
    sections=SECTIONS,
    gears=GEARS,
    moments=MOMENTS,
    card_lines=["45 against 4: it drifts one tooth a lap and comes home in four."],
    credit_lines=["An independent study of the rhythm. Not affiliated with the band or label."],
    counters=[Counter(("rim",), GROUPS, 0, K_TURN), Counter(("tkick", "tsnare"), (6,), K_TURN, K_END)],
    slip=SlipSpec(cycle_gear="rim", beat_steps=4, beat_gears=("pulse", "ride")),
    cycle_steps=CYCLE, cycle_end=K_TURN,
    stack=["rim", "pulse", "kick", "snare", "ride", "bass", "gtr"],
    camera=[
        CameraMove(0, 0, cam_x=0.0, zoom=0.9),                                        # the opening frame
        CameraMove(K_PULSE - 12, K_PULSE - 6, cam_x=0.0, zoom=0.8),
        CameraMove(K_BAND - 13, K_BAND - 7, keys=("rim", "snare", "kick", "bass", "gtr", "ride")),
        CameraMove(K_BAND + 3, K_BAND + 8, keys=("snare", "kick", "bass", "gtr", "ride")),
        CameraMove(K_TURN - 13, K_TURN - 7, keys=("snare", "kick", "bass", "gtr", "ride", "tbass", "tkick", "tgtr", "tsnare")),
        CameraMove(K_TURN + 1.5, K_TURN + 5.5, keys=("ride", "tbass", "tkick", "tgtr", "tsnare")),
    ],
    anchors=[Anchor("tribal pulse", (40, 160), ("pulse",)), Anchor("snare", (150, 5000), ("snare", "tsnare"))],
    cut_note="cut: bars 34-41 removed",
)

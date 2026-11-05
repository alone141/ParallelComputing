"""Meshuggah, "Bleed" (obZen, 2008): the opening (bars 1-33) as gears.

Independent study; not affiliated with the band or their label. The recording is not in the repo:
the film is silent and the owner muxes the record using record_map.json.

Facts (verified against Songsterr's community tab s23825, seven tracks, bars 1-33 read on a 32nd
grid; see films/bleed/README.md for the bar-by-bar table): 115 bpm; 4/4 throughout except one
bar of 2/4 at bar 25. The two guitars (unison, low string), the bass and both kick drums play
xxx.x. on 32nds from bar 1: a six-32nd cell (two 32nds, a sixteenth, a sixteenth) that fits a
32-32nd bar 5⅓ times, so it realigns with the bar line every three bars (96 = 16 cells): 22, 21,
21 notes per bar. The kick alternates feet: right, left, right, left. The snare accents the cell
with a pattern three bars long (some hits a 32nd off the sixteenth grid); the china (bars 1-8,
17-24) or open hi-hat (9-16) rides every eighth; a crash on bar 1, 9 and 26. The vocal sections
are eight bars long, so each begins at a different phase of the three-bar cycle: "Beams of Fire"
(bar 9) two teeth in, "My Crimson Liquid" (bar 17) one tooth in. The band stops after beat 4 of
bar 24 (a home bar); bar 25 is 2/4, guitars alone: the new cell once, then the old one. From bar
26 ("Ripples Ascend") the cell is xxx.x.x.x. (ten 32nds): 20, 19, 19, 19, 19 notes per bar, home
every five bars; the snare's pushes follow the cell (a five-bar pattern); open hi-hat on eighths.

Step = one sixteenth (4 per beat). Film: record steps 0-519 (bars 1-33), nothing cut. On the
sixteenth grid the old cell is three teeth [double, tall, tall] and the new one five
[double, tall, tall, tall, tall]; `double` strikes twice inside its step (the two 32nds).
"""
from gears_film.song import Song, Segment, Section, Gear, Moment, Counter, SlipSpec, CameraMove, Anchor

P = 42.0                                      # the rack pitch (world units): gear x must be a multiple
BAR = 16                                      # sixteenths per 4/4 bar
K_STOP = 382                                  # the band's last stroke is step 381 (bar 24, 32nd 26)
K_24 = 384                                    # bar 25, the 2/4 bar (8 steps), guitars alone
K_NEW = 392                                   # bar 26, "Ripples Ascend"
K_END = 520                                   # end of bar 33

GREEN, VIOLET, RED, RED_L = (125, 255, 154), (182, 140, 255), (255, 92, 70), (255, 168, 128)
WHITE, SKY = (255, 255, 255), (108, 224, 255)


def cells(teeth, cell):
    """Tooth types for a gear whose pattern is `cell` (a tuple of types) repeated over `teeth`."""
    assert teeth % len(cell) == 0
    return {p: cell[p % len(cell)] for p in range(teeth)}


OLD = ("double", "tall", "tall")              # xxx.x. in 32nds, on the sixteenth grid
NEW = ("double", "tall", "tall", "tall", "tall")   # xxx.x.x.x.
LEFT_OLD = frozenset(p for p in range(12) if p % 3 == 2)        # the left foot's lone sixteenth
LEFT_NEW = frozenset(p for p in range(20) if p % 5 in (2, 4))

# The snare's three-bar pattern, in 32nds per bar (bars 1, 4, 7, ... / 2, 5, 8, ... / 3, 6, 9, ...).
SNARE_32 = ((1, 6, 10, 16, 22, 25, 30), (2, 5, 10, 16, 22, 26, 29), (2, 6, 9, 16, 21, 26, 30))
SNARE = {}
for bar_i, hits in enumerate(SNARE_32):
    for t in hits:
        SNARE[bar_i * BAR + t // 2] = "tall" if t % 2 == 0 else "late"     # an odd 32nd: a push
# From bar 26 the snare's pushes follow the new cell (a five-bar pattern); the picture keeps the
# hits on the sixteenth grid, bar 28's pattern: 32nds 2 6 10 16 22 26 30.
SNARE_NEW = {t // 2: "tall" for t in (2, 6, 10, 16, 22, 26, 30)}
CYM = {0: "tall", **{p: "medium" for p in range(2, BAR, 2)}}     # every eighth; the downbeat tall and white

S1 = ("riff", "cym", "kick", "bass", "snare")
S2 = ("riff2", "cym2", "kick2", "bass2", "snare2")

GEARS = [
    # bars 1-24: every wheel in mesh from frame 0
    Gear("riff", "GTRS", "GUITARS", GREEN, 12, cells(12, OLD), +1, 0 * P, ((0, K_STOP),), sound="chug"),
    Gear("cym", "CYMBAL", "CYMBAL", SKY, BAR, CYM, -1, 0 * P, ((0, K_STOP),), sound="ride", first_hit="crash",
         white_position=0),
    Gear("kick", "KICK", "KICK", RED, 12, cells(12, OLD), -1, 8 * P, ((0, K_STOP),), sound="kick",
         colour2=RED_L, colour2_positions=LEFT_OLD),
    Gear("bass", "BASS", "BASS", VIOLET, 12, cells(12, OLD), -1, 17 * P, ((0, K_STOP),), sound="note"),
    Gear("snare", "SNARE", "SNARE", WHITE, 3 * BAR, SNARE, +1, 26 * P, ((0, K_STOP),), sound="snare"),
    # bar 25, 2/4: the guitars alone, eight teeth: the new cell once, then the old one
    Gear("g24", "GTRS", "2/4", GREEN, 8, {**cells(5, NEW), **{5: "double", 6: "tall", 7: "tall"}}, -1, 24 * P,
         ((K_24, K_NEW),), sound="chug", stack="riff", label_dy=130),
    # bars 26-33: the new cell. 392 is not a multiple of 20 or 16: these wheels are keyed (phase 8)
    # so that their tooth 0 is on the contact at step 392.
    Gear("riff2", "GTRS", "GUITARS", GREEN, 20, cells(20, NEW), +1, 29 * P, ((K_NEW, K_END),), sound="chug",
         stack="riff", phase=8),
    Gear("cym2", "CYMBAL", "HAT", SKY, BAR, CYM, -1, 29 * P, ((K_NEW, K_END),), sound="ride", first_hit="crash",
         white_position=0, stack="cym", phase=8, lead=0.1),
    Gear("kick2", "KICK", "KICK", RED, 20, cells(20, NEW), -1, 45 * P, ((K_NEW, K_END),), sound="kick",
         colour2=RED_L, colour2_positions=LEFT_NEW, stack="kick", phase=8, lead=0.2),
    Gear("bass2", "BASS", "BASS", VIOLET, 20, cells(20, NEW), -1, 37 * P, ((K_NEW, K_END),), sound="note",
         stack="bass", phase=8, lead=0.3),
    Gear("snare2", "SNARE", "SNARE", WHITE, BAR, SNARE_NEW, +1, 45 * P, ((K_NEW, K_END),), sound="snare",
         stack="snare", phase=8, lead=0.4),
]

TOOTH_TYPES = {
    # a snare push: it strikes a 32nd after the sixteenth (subhit at 1/2), drawn narrower and lower
    "late": dict(add=14.0, width=0.7, bright=1.0, subhits=(0.5,)),
}


def bar(n):
    """Film step of the record's bar n (bar 25 is the 2/4 bar)."""
    return BAR * (n - 1) if n <= 25 else K_NEW + BAR * (n - 26)


MOMENTS = [
    Moment(0, "WHY DOES THIS RIFF NEVER LAND?",
           "Meshuggah, Bleed. The opening as a machine: one tooth per sixteenth. The guitars' cell is three teeth long.", "hook"),
    Moment(bar(2), "A BAR IS 16 TEETH. THE CELL IS 3.",
           "16 = 5 × 3 + 1. Bar 2 begins one tooth into the cell: the bracket on the guitar wheel shows it."),
    Moment(bar(3), "TWO TEETH",
           "Bar 3 begins two teeth into the cell. Every bar line lands one tooth later, and the snare leans with it."),
    Moment(bar(4), "HOME",
           "Three bars, 48 teeth, and the cell starts on the bar line again. The snare's wheel has 48 teeth: its pattern is three bars long.", "home"),
    Moment(bar(5), "KICK: BOTH FEET, ONE CELL",
           "Two 32nds, a sixteenth, a sixteenth: right, left, right, left. The split tooth is the two 32nds; the pale teeth are the left foot.", colour="kick"),
    Moment(bar(7), "HOME AGAIN",
           "Bars 1, 4, 7, 10: the cell lands on the bar line every third bar. A verse is eight bars, and 8 is not a multiple of 3.", "home"),
    Moment(bar(9), "BEAMS OF FIRE",
           "The voice enters on bar 9, the third bar of a cycle: two teeth into the cell. The cymbal opens to the hi-hat.", "big", "riff"),
    Moment(bar(10), "HOME",
           "Count with the readout: slip 1, slip 2, home. Three bars to a lap, eight bars to a verse.", "home"),
    Moment(bar(13), "HOME ON BAR 13",
           "Bars 13 and 16 land. The verse ends on a home bar, so the next verse cannot start on one.", "home"),
    Moment(bar(17), "MY CRIMSON LIQUID",
           "The second verse enters one tooth into the cell: a different lean from bar 9. 256 = 5 × 48 + 16.", "big", "riff"),
    Moment(bar(19), "HOME",
           "Every third bar still: 19, 22, 25. The china rides every eighth, so the bar never stops being 16.", "home"),
    Moment(bar(22), "THE LAST LAP",
           "Bar 22: the guitars climb through a lick on the same cell. Three bars to the reset.", colour="riff"),
    Moment(bar(25), "HOME. THEN A BAR OF 2/4.",
           "The band stops after beat 4 of bar 24, a home bar. Alone, the guitars play the new cell once, then the old one: eight teeth.", "home"),
    Moment(bar(26), "RIPPLES ASCEND: A NEW CELL",
           "Five teeth now: two 32nds, then four sixteenths. 16 = 3 × 5 + 1: one tooth of slip per bar, home every fifth bar.", "big", "riff2"),
    Moment(bar(28), "TWO TEETH, THREE, FOUR",
           "Same machine, longer cell: the bar line walks around the new wheel while the hi-hat holds the eighths.", colour="riff2"),
    Moment(bar(31), "HOME, FIVE BARS LATER",
           "Bar 31 lands on the cell. The next home would be bar 36, but the old cell returns at bar 34.", "home"),
    Moment(bar(33), "WHY IT NEVER LANDS",
           "A three-tooth cell under a sixteen-tooth bar: home every third bar, never where a verse begins."),
]

SECTIONS = [
    Section("the riff (bars 1-8)", 0, (BAR,) * 8),
    Section("Beams of Fire (bars 9-16)", bar(9), (BAR,) * 8),
    Section("My Crimson Liquid (bars 17-24)", bar(17), (BAR,) * 8),
    Section("2/4, guitars alone (bar 25)", bar(25), (8,)),
    Section("Ripples Ascend (bars 26-33)", bar(26), (BAR,) * 8),
]

SONG = Song(
    slug="bleed",
    title="Bleed", artist="Meshuggah", album="obZen", year=2008,
    bpm=115.0, steps_per_beat=4, first_click_offset_s=0.0,
    segments=[
        # one continuous stretch of the record; two segments so the audio fit can lock the band's
        # re-entry at bar 26 on its level jump (bar 25 is guitars alone, no anchor hits there)
        Segment(record_start=0, steps=K_NEW, label="2/4 · GUITARS ALONE"),       # the label shows only during bar 25
        Segment(record_start=K_NEW, steps=K_END - K_NEW, label="RIPPLES ASCEND", level_jump=True),
    ],
    sections=SECTIONS,
    gears=GEARS,
    moments=MOMENTS,
    card_lines=["A 3-tooth cell under a 16-tooth bar.", "Home every third bar, never where a verse begins."],
    credit_lines=["An independent study of the rhythm.", "Not affiliated with the band or label."],
    counters=[Counter(("snare", "cym"), (BAR,), 0, K_24), Counter(("g24",), (8,), K_24, K_NEW),
              Counter(("snare2", "cym2"), (BAR,), K_NEW, K_END)],
    # the bar (the cymbal wheel, 16 teeth) against the cell (3 teeth, then 5): slip 1 per bar
    slip=[SlipSpec(cycle_gear="cym", beat_steps=3, beat_gears=("riff",), k_from=0, k_to=K_24),
          SlipSpec(cycle_gear="cym2", beat_steps=5, beat_gears=("riff2",), k_from=K_NEW, k_to=K_END)],
    cycle_steps=BAR, cycle_end=K_24, lap_word="BAR", lap_label_pos=(490, 1446),
    stack=["riff", "bass", "kick", "snare", "cym"],
    tooth_types=TOOTH_TYPES,
    camera=[
        CameraMove(0, 0, keys=S1),                                             # the opening frame
        CameraMove(K_24 - 10, K_24 - 3, keys=S1 + ("g24",) + S2),              # wide: the 2/4 wheel and the new set arrive
        CameraMove(K_NEW + 9, K_NEW + 15, keys=S2),                            # settle once the leavers are gone
    ],
    anchors=[Anchor("snare", (150, 450), ("snare", "snare2")), Anchor("kick", (40, 180), ("kick", "kick2"))],
    first_step_guess_s=0.24,
)

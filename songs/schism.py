"""Tool, "Schism" (Lateralus, 2001): the intro, first verse and first chorus as gears.

Independent study; not affiliated with the band or their label. The recording is not in the repo:
the film is silent and the owner muxes the record using record_map.json.

Facts (Songsterr community tab s6700, revision in /tmp/tabs/schism/revisions.json at build time,
8 tracks, checked bar by bar with gears_film/transcribe/songsterr_parse.py; the tab's synced-video
bar times give the tempo per section):

- Tempo: the tab writes 109 bpm from bar 4, 108 from bar 11, 107 from bar 19, 108 at 41, 109 at
  43, 107 at 51. The synced video's bar times say 107.9 (bars 4-19), 107.1 (20-43), 107.3 (44-51);
  the owner's recording autocorrelates at about 107. The owner measured each segment against the
  recording (bass anchors: melody fundamentals 160-205 Hz, D2 pedal 60-90 Hz): 108.08 / 107.88 /
  108.06 / 107.79 bpm, record step 0 at 46.818 s of their file (the official-video audio; the album
  audio starts at 33.0 s in it). Those values are in the segments below. first_click_offset_s stays
  0: the picture starts on record step 0, and the owner's audio build starts the record there.
- Bars 1-3 (64 bpm, 5/4 5/4 4/4): a quiet guitar intro, not in the film. Record step 0 is the
  downbeat of bar 4 (46.818 s into the owner's file). The step is the eighth note (2 per beat).
- Bars 4-11 "Intro II - Bass": bass alone, bars alternating 5/8 and 7/8: 12 eighths per pair.
  The bass strikes EVERY eighth; the 5|7 is in the pitches. 5/8 = F3 D2 E3 D2 D2,
  7/8 = F3 D2 E3 D2 G3 D2 D2 (tab: s0f10 s1f0 s0f9 s1f0 s0f12, tuning [43, 38, 33, 26], so s1f0 is
  the open D string). Ghost pickup notes (grace notes off the grid) sit at the end of each bar.
- Bar 12 "Intro III - with Band": kick (36) 5/8 "x...x", 7/8 "x.x...x" in bars 13, 17, 21, ... and
  "x.xx..x" in bars 15, 19, ...: the kick's true cycle is two 5|7 pairs, 24 eighths. Snare (38) on
  the 3rd eighth of the five and the 5th eighth of the seven. Hi-hat (42) on every eighth except
  the one after an open hat (46): the open hat is the 3rd eighth of the first seven and the 4th of
  the second. A China (48) on the first downbeat. Guitar 1 doubles the riff a third up
  (Bb3 D3 A3 D3 D3 | Bb3 D3 A3 D3 C4 D3 D3, same shape: pedal, melody, one high note); guitar 2
  is silent here.
- Bar 20 "Verse 1": the voice enters (not drawn); the guitars hold chords (bars 20-35, not drawn
  as teeth) and from bar 36 play the riff shape on A (C4 A2 B3 A2 A2 | C4 A2 B3 A2 D4 A2 A2).
  Bass and drums keep the patterns above; the tab adds crashes and small fills in some bars
  (not drawn). Bar 43 ends on an A1 pickup into the chorus (drawn as the pedal D).
- Bar 44 "Chorus": 6/8 + 7/8 = 13 eighths per pair, four pairs. Bass C3 A1 B2 A1 A1 A1 |
  C3 A1 B2 A1 D3 A1 A1 (pedal A1 = open A); both guitars the same shape an octave up. Drums:
  kick "..xx.x" | "x...x..", snare on the 6/8 downbeat and the 3rd eighth of the seven, ride (51)
  on most eighths (the tab varies bar to bar: 6/8 "xx..xx" and 7/8 "xxxxxx." are used), a low
  tom (41) on the 6/8's 3rd eighth and crashes (not drawn). The film ends at the end of bar 51.

Film = record steps 0-291 (bars 4-51) with no cuts: 48 steps of bass alone, 48 of the band,
144 of verse, 52 of chorus. The 6-tooth "6/8 EAR" wheels are not instruments: they show where a
listener who hears 6/8 would tap (every dotted quarter), and they never flash.
"""
from gears_film.song import Song, Segment, Section, Gear, Moment, Counter, CameraMove, Anchor

P = 42.0                                   # the rack pitch (world units): gear x must be a multiple
BARS, CBARS = (5, 7), (6, 7)               # eighths per bar
CYCLE, CCYCLE = sum(BARS), sum(CBARS)      # 12, 13

K_EAR, K_BAND, K_VERSE, K_GTR2, K_CHORUS, K_END = 24, 48, 96, 192, 240, 292

VIOLET, VIOLET_DARK = (182, 140, 255), (118, 90, 168)
RED, WHITE, SKY, GREEN, GHOST = (255, 92, 70), (255, 255, 255), (108, 224, 255), (125, 255, 154), (196, 206, 230)

tall = lambda positions: {p: "tall" for p in positions}

# the riff, typed by pitch: 5/8 = F D E D D, 7/8 = F D E D G D D (every eighth strikes)
RIFF = {0: "melody", 1: "pedal", 2: "melody", 3: "pedal", 4: "pedal",
        5: "melody", 6: "pedal", 7: "melody", 8: "pedal", 9: "high", 10: "pedal", 11: "pedal"}
# the chorus riff: 6/8 = C A B A A A, 7/8 = C A B A D A A
CRIFF = {0: "melody", 1: "pedal", 2: "melody", 3: "pedal", 4: "pedal", 5: "pedal",
         6: "melody", 7: "pedal", 8: "melody", 9: "pedal", 10: "high", 11: "pedal", 12: "pedal"}
# drums over two 5|7 pairs (24 steps): bar A 0-4, B 5-11, C 12-16, D 17-23
KICK24 = tall((0, 4, 5, 7, 11, 12, 16, 17, 19, 20, 23))
SNARE12 = tall((2, 9))
HATS24 = {**{p: "medium" for p in range(24)}, 7: "tall", 20: "tall"}      # tall = open hat
del HATS24[8]                                                               # the open hat rings through
# the chorus drums over 6|7 (13 steps)
CKICK13 = tall((2, 3, 5, 6, 10))
CSNARE13 = tall((0, 8))
CRIDE13 = tall((0, 1, 4, 5, 6, 7, 8, 9, 10, 11))
EAR6 = {0: "tap", 3: "tap"}                                                 # where a 6/8 listener taps

GEARS = [
    Gear("bass", "BASS", "BASS", VIOLET, 12, RIFF, +1, 0 * P, ((0, K_CHORUS),), sound="note", white_position=0),
    Gear("ear", "6/8 EAR", "6/8 EAR", GHOST, 6, EAR6, -1, 0 * P, ((K_EAR, K_CHORUS),), sound="click"),
    Gear("kick", "KICK", "KICK", RED, 24, KICK24, +1, 8 * P, ((K_BAND, K_CHORUS),), sound="kick", lead=0.3),
    Gear("snare", "SNARE", "SNARE", WHITE, 12, SNARE12, -1, 5 * P, ((K_BAND, K_CHORUS),), sound="snare", lead=0.4, label_dy=90),
    Gear("hats", "CYMBAL", "HI-HAT", SKY, 24, HATS24, +1, 18 * P, ((K_BAND, K_CHORUS),), sound="ride",
         first_hit="crash", lead=0.1),
    Gear("gtr", "GTRS", "GTRS", GREEN, 12, RIFF, -1, 11 * P, ((K_BAND, K_VERSE), (K_GTR2, K_CHORUS)), sound="chug", lead=0.2),
    # the chorus: 6 + 7 = 13 teeth, second gears in free slots to the right
    Gear("cbass", "BASS", "BASS 6+7", VIOLET, 13, CRIFF, +1, 26 * P, ((K_CHORUS, K_END),), sound="note",
         white_position=0, stack="bass", origin=K_CHORUS, lead=0.5),
    Gear("cear", "6/8 EAR", "6/8 EAR", GHOST, 6, EAR6, -1, 26 * P, ((K_CHORUS, K_END),), sound="click", stack="ear", lead=0.4),
    Gear("ckick", "KICK", "KICK 6+7", RED, 13, CKICK13, +1, 33 * P, ((K_CHORUS, K_END),), sound="kick", stack="kick", lead=0.3, origin=K_CHORUS, label_dy=90),
    Gear("csnare", "SNARE", "SNARE 6+7", WHITE, 13, CSNARE13, -1, 31 * P, ((K_CHORUS, K_END),), sound="snare", stack="snare", lead=0.2, origin=K_CHORUS, label_dy=90),
    Gear("cride", "CYMBAL", "RIDE", SKY, 13, CRIDE13, +1, 40 * P, ((K_CHORUS, K_END),), sound="ride",
         first_hit="crash", stack="hats", lead=0.1, origin=K_CHORUS),
    Gear("cgtr", "GTRS", "GTRS 6+7", GREEN, 13, CRIFF, -1, 38 * P, ((K_CHORUS, K_END),), sound="chug", stack="gtr", lead=0.0, origin=K_CHORUS),
]

# the song's own tooth types (merged over the built-ins): the riff is typed by pitch
TOOTH_TYPES = {
    "pedal":  dict(add=9.0, bright=0.8, shade=VIOLET_DARK),   # short and dark: the open-string pedal note
    "melody": dict(add=18.0),                                  # tall: the melody notes (F, E / C, B)
    "high":   dict(add=23.0, width=0.85),                      # the tallest tooth: the high G / D
    "tap":    dict(add=14.0, bright=0.8, subhits=()),          # a painted tooth that never strikes: a ghost tap
}

MOMENTS = [
    Moment(0, "IS THIS 6/8?", "Tool, Schism. The bass riff as a wheel: one tooth per eighth note. Count the tall teeth.", "hook"),
    Moment(12, "ONE WHEEL, 12 TEETH", "Every tooth strikes: the bass plays every eighth. Short dark teeth are the open D; tall ones are the melody."),
    Moment(24, "WHERE A 6/8 EAR TAPS", "A ghost wheel with six teeth: 12 = 6 + 6, so it never drifts. That is why this feels like 6/8.", "big", "ear"),
    Moment(36, "BUT THE RIFF SAYS 5 + 7", "F comes back after five steps, then after seven. The counter below counts the bars: 5, then 7.", colour="bass"),
    Moment(48, "THE BAND: 5 + 7 ON THE DRUMS", "Kick and hi-hat have 24 teeth: the seven comes in two flavours. Snare on the 3rd of the five, the 5th of the seven.", "big", "kick"),
    Moment(60, "KICK: 24 TEETH, TWO SEVENS", "x · · · x | x · x · · · x, then x · · · x | x · x x · · x. One lap of the kick is two of the riff.", colour="kick"),
    Moment(72, "THE GUITAR DOUBLES THE RIFF", "Same teeth, a third higher. The hi-hat opens on the seven's 3rd eighth, then on its 4th.", colour="gtr"),
    Moment(84, "THE GHOST STILL FITS", "Two of its four taps land on the open D, the riff's weakest notes. In step, but not in time.", colour="ear"),
    Moment(96, "VERSE: THE VOICE ENTERS", "The guitars drop to held chords and roll away. Bass and drums keep 5 + 7 for twelve laps.", "big", "bass"),
    Moment(108, "WHY 5 + 7 AND NOT 12/8?", "Because the riff restarts on step 5: F, D, E, D, D, then F again. A 6/8 ear waits for step 6."),
    Moment(120, "THE DRUMS AGREE", "Kick on step 0 and step 5, the two bar lines. Snare on 2 and 9: the backbeat of each bar.", colour="kick"),
    Moment(132, "TWO KINDS OF SEVEN", "The kick's second seven adds a note: x · x x · · x. The hi-hat opens one step later to match.", colour="hats"),
    Moment(156, "HALFWAY", "Twelve laps in. The ghost wheel has turned 24 times and is still exactly in step: no drift at all."),
    Moment(180, "THE GUITARS COME BACK", "The riff moves to A for the pre-chorus. The green wheel returns on the next lap.", colour="gtr"),
    Moment(192, "PRE-CHORUS: THE RIFF ON A", "Guitars and bass play the same shape a fourth lower. The teeth do not change: 5 + 7.", "big", "gtr"),
    Moment(216, "TWO LAPS OF 12 TO GO", "At the chorus the five becomes a six. Wheels with 13 teeth wait on the right.", colour="bass"),
    Moment(228, "ONE MORE LAP", "Six new wheels slide in. Watch the ghost: 13 is not a multiple of 6.", colour="ear"),
    Moment(240, "CHORUS: 6 + 7 = 13", "Ride cymbal, snare on the one. Now the ghost wheel falls one tooth behind every lap.", "big", "cride"),
    Moment(253, "ONE TOOTH BEHIND", "13 = 6 + 6 + 1. The spare tooth piles up on the ghost, lap after lap.", colour="cear"),
    Moment(266, "TWO TEETH", "The wheel that never drifted in the verse drifts here. The riff keeps its own count.", colour="cear"),
    Moment(279, "THREE TEETH, THEN VERSE 2", "Not 6/8, not 12/8: 5 + 7, then 6 + 7. The pieces fit because the bass says where they go.", colour="cbass"),
]

SECTIONS = [
    Section("Intro II, bass alone (bars 4-11)", 0, BARS * 4),
    Section("Intro III, with band (bars 12-19)", 48, BARS * 4),
    Section("Verse 1 (bars 20-43)", 96, BARS * 12),
    Section("Chorus (bars 44-51)", 240, CBARS * 4),
]

SONG = Song(
    slug="schism",
    title="Schism", artist="Tool", album="Lateralus", year=2001,
    bpm=107.0, steps_per_beat=2, first_click_offset_s=0.0,
    # the tempo map: measured by the owner against their recording (boundary-defined: each segment's
    # length in the record divided into its steps, so the audio runs uncut and the clocks match).
    # The tab's synced-video bar times had said 107.9 / 107.9 / 107.1 / 107.3.
    segments=[
        Segment(record_start=0, steps=48, bpm=108.0833, label="BASS ALONE"),
        Segment(record_start=48, steps=48, bpm=107.8785, label="THE BAND", level_jump=True),
        Segment(record_start=96, steps=144, bpm=108.0641, label="VERSE 1"),
        Segment(record_start=240, steps=52, bpm=107.7855, label="CHORUS: 6 + 7", level_jump=True),
    ],
    sections=SECTIONS,
    gears=GEARS,
    moments=MOMENTS,
    card_lines=["Not 6/8: five eighths and seven, then six and seven in the chorus."],
    credit_lines=["An independent study of the rhythm. Not affiliated with the band or label."],
    counters=[Counter(("bass",), BARS, 0, K_BAND),
              Counter(("kick", "snare", "bass"), BARS, K_BAND, K_CHORUS),
              Counter(("ckick", "csnare", "cbass"), CBARS, K_CHORUS, K_END)],
    slip=None,                       # no steady beat to drift against: 12 = 6 + 6 (see the README)
    cycle_steps=CYCLE, cycle_end=K_CHORUS,
    rack_marks=tuple(range(0, K_CHORUS, CYCLE)) + tuple(range(K_CHORUS, K_END + 1, CCYCLE)),
    stack=["bass", "ear", "kick", "snare", "hats", "gtr"],
    tooth_types=TOOTH_TYPES,
    camera=[
        CameraMove(0, 0, cam_x=0.0, zoom=2.0),                                             # the opening frame: the riff wheel alone
        CameraMove(K_EAR - 6, K_EAR - 3, cam_x=0.0, zoom=1.3),                             # room for the ghost below
        CameraMove(K_BAND - 7, K_BAND - 3, keys=("bass", "ear", "kick", "snare", "hats", "gtr")),
        CameraMove(K_CHORUS - 8, K_CHORUS - 4, keys=("bass", "kick", "hats", "cbass", "ckick", "cride", "cgtr")),
        CameraMove(K_CHORUS + 2, K_CHORUS + 6, keys=("cbass", "cear", "ckick", "csnare", "cride", "cgtr"), zoom_cap=0.85),
    ],
    anchors=[Anchor("bass melody attack", (600, 2500), ("bass", "cbass"), tooth_types=("melody", "high")),
             Anchor("kick", (40, 180), ("kick", "ckick")),
             Anchor("snare", (150, 450), ("snare", "csnare"))],
    first_step_guess_s=46.82,         # record step 0 (the bar-4 downbeat) in the owner's file, as measured
    cut_note=None,
)

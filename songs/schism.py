"""Tool, "Schism" (Lateralus, 2001): the intro, first verse and first chorus as gears. Round 2.

Independent study; not affiliated with the band or their label. The recording is not in the repo:
the film is silent and the owner muxes the record using record_map.json.

THE MECHANISM (round 2). Schism is not a polymetre: its bars alternate in length, 5/8 then 7/8
(6/8 then 7/8 in the chorus). So the bar lengths are the gears. Each instrument's 12-step pattern
is split into its 5-step bar and its 7-step bar, drawn as a 5-tooth gear and a 7-tooth gear on a
seesaw (a `Rocker`) that tips one into mesh as the other lifts, on every bar line. The engaged gear
is bright and turns one tooth per eighth; the resting one is dim and still. A member turns exactly
once per bar, so its tooth 0 is on the contact at every one of its bar lines. The kick's two
different sevens live on one 14-tooth gear that turns half a turn per 7/8 bar. The guitar (the riff
on one wheel of 12 = 5 + 7) stays a plain gear on the rack. The hi-hat (verse) and ride (chorus)
play every eighth and carry no bar information: they are not drawn, so the camera can stay close
(phone legibility first). At the chorus the five becomes a six: new seesaws (6 + 7) glide in and
the old ones roll away.

THE CLOCK (round 2). Tool play without a click. The owner beat-tracked every eighth of the recording
from the bar-4 downbeat (46.8181 s in their file): `step_times` below is that list, and the picture
follows it step by step (the engine interpolates inside a step). Against one tempo per section the
record drifted up to 146 ms; against this list its eighths sit within ±10 ms in all 24 cycles.

Facts (Songsterr community tab s6700, 8 tracks, read bar by bar with
gears_film/transcribe/songsterr_parse.py; bar numbers are the tab's, bar 1 is the first bar):
- Bars 1-3 (64 bpm, 5/4 5/4 4/4): a quiet guitar intro, not in the film. Record step 0 is the
  downbeat of bar 4. The step is the eighth note (2 per beat).
- Bars 4-11 "Intro II - Bass": bass alone, 5/8 + 7/8 x 4. The bass strikes EVERY eighth; the 5|7
  is in the pitches: 5/8 = F3 D2 E3 D2 D2, 7/8 = F3 D2 E3 D2 G3 D2 D2 (tab s0f10 s1f0 s0f9 s1f0
  s0f12, tuning [43, 38, 33, 26]: s1f0 is the open D). Grace-note pickups at bar ends not drawn.
- Bar 12 "Intro III - with Band": kick (36) 5/8 "x...x", 7/8 "x.x...x" in bars 13, 17, 21, ... and
  "x.xx..x" in bars 15, 19, ...; snare (38) on the 3rd eighth of the five and the 5th of the seven;
  hi-hat (42) on every eighth except the one after an open hat (46): open on the 3rd eighth of the
  first seven and the 4th of the second; China (48) on the bar-12 downbeat. Guitar 1 doubles the
  riff a third up (Bb3 D3 A3 D3 D3 | Bb3 D3 A3 D3 C4 D3 D3); guitar 2 is silent here.
- Bar 20 "Verse 1": the voice enters (not drawn); the guitars hold chords (bars 20-35, not drawn as
  teeth) and from bar 36 play the riff shape on A. Bass and drums keep the patterns; the tab adds
  crashes and small fills in some bars (not drawn). Bar 43 ends on an A1 pickup (drawn as the pedal).
- Bar 44 "Chorus": 6/8 + 7/8 = 13 eighths a pair, four pairs. Bass C3 A1 B2 A1 A1 A1 | C3 A1 B2 A1
  D3 A1 A1 (pedal = open A); both guitars the same shape an octave up. Kick "..xx.x" | "x...x..",
  snare on the 6/8 downbeat and the 3rd eighth of the seven, ride (51) on most eighths (the tab
  varies bar to bar; 6/8 "xx..xx" and 7/8 "xxxxxx." are used), low tom and crashes not drawn.
  The film ends at the end of bar 51 (record step 292).
"""
from gears_film.song import Song, Segment, Section, Gear, Rocker, Moment, Counter, CameraMove, Anchor

P = 42.0                                   # the rack pitch (world units): gear x must be a multiple
BARS, CBARS = (5, 7), (6, 7)               # eighths per bar
CYCLE, CCYCLE = sum(BARS), sum(CBARS)      # 12, 13
BAR_LENGTHS = BARS * 20 + CBARS * 4        # 48 bars, 292 steps

K_BAND, K_VERSE, K_GTR2, K_CHORUS, K_END = 48, 96, 192, 240, 292

VIOLET, VIOLET_DARK = (182, 140, 255), (118, 90, 168)
RED, WHITE, SKY, GREEN = (255, 92, 70), (255, 255, 255), (108, 224, 255), (125, 255, 154)

tall = lambda positions: {p: "tall" for p in positions}

# the riff, typed by pitch: the 5/8 bar F D E D D and the 7/8 bar F D E D G D D (every eighth strikes)
RIFF5 = {0: "melody", 1: "pedal", 2: "melody", 3: "pedal", 4: "pedal"}
RIFF7 = {0: "melody", 1: "pedal", 2: "melody", 3: "pedal", 4: "high", 5: "pedal", 6: "pedal"}
RIFF12 = {**RIFF5, **{5 + p: t for p, t in RIFF7.items()}}                   # the guitar's one wheel
RIFF6 = {0: "melody", 1: "pedal", 2: "melody", 3: "pedal", 4: "pedal", 5: "pedal"}   # chorus 6/8: C A B A A A
CRIFF13 = {**RIFF6, **{6 + p: t for p, t in RIFF7.items()}}                   # chorus guitar wheel

# the band's seesaws: film ranges
R_BASS, R_DRUMS, R_CHORUS = ((0, K_CHORUS),), ((K_BAND, K_CHORUS),), ((K_CHORUS, K_END),)

ROCKERS = [
    Rocker("bass", "BASS", "BASS", VIOLET, {5: "bass5", 7: "bass7"}, R_BASS, side=-1),
    Rocker("kick", "KICK", "KICK", RED, {5: "kick5", 7: "kick14"}, R_DRUMS, side=+1, lead=0.3),
    Rocker("snare", "SNARE", "SNARE", WHITE, {5: "snare5", 7: "snare7"}, R_DRUMS, side=-1, lead=0.4),
    Rocker("cbass", "BASS", "BASS 6·7", VIOLET, {6: "cbass6", 7: "cbass7"}, R_CHORUS, side=-1, lead=0.5),
    Rocker("ckick", "KICK", "KICK 6·7", RED, {6: "ckick6", 7: "ckick7"}, R_CHORUS, side=+1, lead=0.3),
    Rocker("csnare", "SNARE", "SNARE 6·7", WHITE, {6: "csnare6", 7: "csnare7"}, R_CHORUS, side=-1, lead=0.2),
]

GEARS = [
    # the bass seesaw below the rack at x = 0: a 5 and a 7
    Gear("bass5", "BASS", "5", VIOLET, 5, RIFF5, -1, -2 * P, R_BASS, sound="note", white_position=0, rocker="bass"),
    Gear("bass7", "BASS", "7", VIOLET, 7, RIFF7, -1, 2 * P, R_BASS, sound="note", stack="bass5", rocker="bass"),
    # the band: kick seesaw (5 and 14 = two different sevens) and the guitar above; snare seesaw below
    Gear("kick5", "KICK", "5", RED, 5, tall((0, 4)), +1, -2 * P, R_DRUMS, sound="kick", first_hit="crash", rocker="kick"),
    Gear("kick14", "KICK", "14", RED, 14, tall((0, 2, 6, 7, 9, 10, 13)), +1, 3 * P, R_DRUMS, sound="kick", stack="kick5", rocker="kick"),
    Gear("snare5", "SNARE", "5", WHITE, 5, tall((2,)), -1, 8 * P, R_DRUMS, sound="snare", rocker="snare"),
    Gear("snare7", "SNARE", "7", WHITE, 7, tall((4,)), -1, 12 * P, R_DRUMS, sound="snare", stack="snare5", rocker="snare"),
    Gear("gtr", "GTRS", "GTRS", GREEN, 12, RIFF12, +1, 10 * P, ((K_BAND, K_VERSE), (K_GTR2, K_CHORUS)), sound="chug", lead=0.1),
    # the chorus: 6 + 7 seesaws and a 13-tooth guitar wheel in free slots to the right
    Gear("cbass6", "BASS", "6", VIOLET, 6, RIFF6, -1, 24 * P, R_CHORUS, sound="note", white_position=0, stack="bass5", rocker="cbass"),
    Gear("cbass7", "BASS", "7", VIOLET, 7, RIFF7, -1, 29 * P, R_CHORUS, sound="note", stack="bass5", rocker="cbass"),
    Gear("ckick6", "KICK", "6", RED, 6, tall((2, 3, 5)), +1, 24 * P, R_CHORUS, sound="kick", stack="kick5", rocker="ckick"),
    Gear("ckick7", "KICK", "7", RED, 7, tall((0, 4)), +1, 29 * P, R_CHORUS, sound="kick", stack="kick5", rocker="ckick"),
    Gear("csnare6", "SNARE", "6", WHITE, 6, tall((0,)), -1, 33 * P, R_CHORUS, sound="snare", first_hit="crash", stack="snare5", rocker="csnare"),
    Gear("csnare7", "SNARE", "7", WHITE, 7, tall((2,)), -1, 38 * P, R_CHORUS, sound="snare", stack="snare5", rocker="csnare"),
    Gear("cgtr", "GTRS", "GTRS 6+7", GREEN, 13, CRIFF13, +1, 35 * P, R_CHORUS, sound="chug", stack="gtr", origin=K_CHORUS, lead=0.0),
]

# the song's own tooth types (merged over the built-ins): the riff is typed by pitch
TOOTH_TYPES = {
    "pedal":  dict(add=9.0, bright=0.8, shade=VIOLET_DARK),   # short and dark: the open-string pedal note
    "melody": dict(add=18.0),                                  # tall: the melody notes (F, E / C, B)
    "high":   dict(add=23.0, width=0.85),                      # the tallest tooth: the high G / D
}

MOMENTS = [
    Moment(0, "IS THIS 6/8?", "Tool, Schism. Two wheels take turns: five teeth, then seven. One tooth per eighth note.", "hook"),
    Moment(12, "FIVE, THEN SEVEN", "The bass riff is F D E D D on the small wheel, F D E D G D D on the big one. Each turns once per bar, then tips out."),
    Moment(24, "A SEESAW, NOT A POLYMETRE", "Nothing drifts. The bar lengths just alternate, and the counter counts 5, then 7, with whichever wheel is down."),
    Moment(36, "WHY IT FEELS LIKE 6/8", "5 + 7 = 12 = two bars of 6/8. The sum is even; the halves are not. F lands on step 0 and step 5, never on 6.", colour="bass5"),
    Moment(48, "THE BAND: 5 AND 7 ON THE DRUMS", "Kick and snare get seesaws of their own. The kick's seven has two flavours, so its big wheel has 14 teeth: half a turn per bar.", "big", "kick5"),
    Moment(60, "KICK: 5, THEN HALF OF 14", "x · · · x on the five. x · x · · · x, then x · x x · · x: the fourteen shows one seven, then the other.", colour="kick5"),
    Moment(72, "THE GUITAR: ONE PLAIN WHEEL", "It doubles the riff a third up on a wheel of 12 = 5 + 7, no seesaw. The hi-hat plays every eighth and is not drawn.", colour="gtr"),
    Moment(84, "SNARE: ONE HIT PER WHEEL", "On the 3rd tooth of the five, the 5th of the seven. The backbeat moves with the bar.", colour="snare5"),
    Moment(96, "VERSE: THE VOICE ENTERS", "The guitar drops to held chords and rolls away. Bass and drums keep seesawing: 5, 7, 5, 7.", "big", "bass5"),
    Moment(108, "COUNT WITH THE SEESAWS", "Small wheel down: count to five. Big wheel down: count to seven. That is the whole trick."),
    Moment(132, "TWO KINDS OF SEVEN", "Watch the kick's 14: each seven uses the other half. The second seven adds a kick on its fourth eighth.", colour="kick5"),
    Moment(156, "HALFWAY", "Twelve laps of 5 + 7. The band plays without a click, so the record breathes; the picture follows every eighth."),
    Moment(180, "THE GUITAR COMES BACK", "The riff moves to A for the pre-chorus. The green wheel returns on the next lap.", colour="gtr"),
    Moment(192, "PRE-CHORUS: THE RIFF ON A", "The same shapes a fourth lower. Still five, then seven.", "big", "gtr"),
    Moment(216, "TWO LAPS TO THE CHORUS", "At the chorus the five becomes a six. New seesaws wait on the right.", colour="bass5"),
    Moment(228, "ONE MORE LAP", "Six and seven: 13 teeth a pair. Nothing there is 6/8 either.", colour="bass5"),
    Moment(240, "CHORUS: 6, THEN 7", "The small wheels now have six teeth. The snare lands on the one of the six; the ride cymbal (every eighth) is not drawn.", "big", "csnare6"),
    Moment(253, "SIX AND SEVEN", "The kick: · · x x · x on the six, x · · · x · · on the seven. The bass pedal moves to the open A.", colour="ckick6"),
    Moment(266, "THE SUM IS 13", "Odd again: no 6/8 ear can split it. The seesaw just keeps tipping.", colour="cbass6"),
    Moment(279, "THEN VERSE 2", "Not 6/8, not 12/8: 5 then 7, then 6 then 7. The pieces fit because the bar tells you where they go.", colour="cbass6"),
]

SECTIONS = [
    Section("Intro II, bass alone (bars 4-11)", 0, BARS * 4),
    Section("Intro III, with band (bars 12-19)", 48, BARS * 4),
    Section("Verse 1 (bars 20-43)", 96, BARS * 12),
    Section("Chorus (bars 44-51)", 240, CBARS * 4),
]

# The picture's clock: the record time of every eighth (film seconds from record step 0 = 46.8181 s of
# the owner's file), beat-tracked by the owner from their recording (Viterbi over the onset envelope,
# checked per cycle with pitch-specific bass combs and the chorus's snare and kick combs).
STEP_FILM_S = [0.0, 0.27378, 0.55156, 0.82933, 1.10711, 1.38489, 1.66267, 1.94044, 2.21822, 2.496, 2.77378, 3.05156, 3.32933, 3.60711, 3.88489, 4.16267, 4.44044, 4.71822, 4.996, 5.27378, 5.55156, 5.82933, 6.10711, 6.38489, 6.66267, 6.94044, 7.21822, 7.496, 7.77378, 8.05156, 8.32533, 8.59911, 8.87289, 9.14667, 9.42444, 9.70222, 9.98, 10.25778, 10.53556, 10.81333, 11.09111, 11.36889, 11.64667, 11.92444, 12.20222, 12.48, 12.75778, 13.03556, 13.30933, 13.58711, 13.86489, 14.14267, 14.42044, 14.70222, 14.984, 15.26578, 15.54356, 15.82133, 16.09911, 16.37689, 16.65467, 16.92844, 17.20222, 17.476, 17.75378, 18.03156, 18.30933, 18.58711, 18.86489, 19.14267, 19.42044, 19.69822, 19.976, 20.25378, 20.53156, 20.80933, 21.08711, 21.36489, 21.64267, 21.92044, 22.19822, 22.476, 22.75378, 23.03156, 23.30533, 23.57911, 23.85289, 24.12667, 24.40044, 24.67822, 24.956, 25.23378, 25.51156, 25.78933, 26.06711, 26.34889, 26.63067, 26.91244, 27.19422, 27.476, 27.75378, 28.03156, 28.30933, 28.58711, 28.86489, 29.14267, 29.42044, 29.69822, 29.976, 30.25778, 30.53956, 30.82133, 31.10311, 31.38489, 31.66667, 31.94444, 32.22222, 32.5, 32.78178, 33.06356, 33.34533, 33.62711, 33.90889, 34.19067, 34.47244, 34.75422, 35.036, 35.31778, 35.59556, 35.87333, 36.15111, 36.42889, 36.70667, 36.98844, 37.27022, 37.552, 37.83378, 38.11556, 38.39333, 38.67111, 38.94889, 39.22667, 39.50444, 39.78622, 40.068, 40.34978, 40.63156, 40.91333, 41.19511, 41.47689, 41.75867, 42.04044, 42.32222, 42.604, 42.88578, 43.16756, 43.44933, 43.72711, 44.00489, 44.28267, 44.56044, 44.83822, 45.12, 45.40178, 45.68356, 45.96533, 46.24711, 46.52889, 46.81067, 47.09244, 47.37022, 47.648, 47.92578, 48.20356, 48.48133, 48.75911, 49.03689, 49.31467, 49.59244, 49.87422, 50.156, 50.43778, 50.71956, 51.00133, 51.28311, 51.56489, 51.84667, 52.12844, 52.40622, 52.684, 52.96178, 53.23956, 53.51733, 53.79511, 54.06889, 54.34267, 54.61644, 54.89022, 55.168, 55.44578, 55.72756, 56.00933, 56.29111, 56.56889, 56.84667, 57.12444, 57.40222, 57.68, 57.95778, 58.23556, 58.51333, 58.79111, 59.06889, 59.34667, 59.62444, 59.90222, 60.18, 60.45778, 60.73556, 61.01333, 61.29111, 61.56889, 61.84667, 62.12844, 62.41022, 62.692, 62.97378, 63.25156, 63.52933, 63.80711, 64.08489, 64.36267, 64.64044, 64.91822, 65.196, 65.47378, 65.75156, 66.02933, 66.30711, 66.58489, 66.86667, 67.14844, 67.43022, 67.712, 67.99378, 68.27556, 68.55733, 68.83911, 69.12089, 69.40267, 69.68444, 69.96622, 70.248, 70.52978, 70.80756, 71.08533, 71.36311, 71.64089, 71.91867, 72.19644, 72.47822, 72.76, 73.04178, 73.32356, 73.60533, 73.88711, 74.16889, 74.45067, 74.72844, 75.00622, 75.284, 75.56178, 75.83956, 76.11733, 76.39511, 76.67289, 76.95067, 77.22844, 77.50622, 77.784, 78.06578, 78.34756, 78.62933, 78.91111, 79.19289, 79.47467, 79.75644, 80.03822, 80.316, 80.59378, 80.87156, 81.14933, 81.42711]
assert len(STEP_FILM_S) == K_END + 1


def avg_bpm(k0, k1):
    """The mean tempo of record steps k0..k1 from the step list (for the record map's segment lines)."""
    return round(60.0 / (2 * (STEP_FILM_S[k1] - STEP_FILM_S[k0]) / (k1 - k0)), 4)


SONG = Song(
    slug="schism",
    title="Schism", artist="Tool", album="Lateralus", year=2001,
    bpm=107.0, steps_per_beat=2, first_click_offset_s=0.0,
    # the musical segments (for labels, bar numbering and the audio fit); their bpm is the mean of the
    # tracked eighths, the picture itself runs on step_times
    segments=[
        Segment(record_start=0, steps=48, bpm=avg_bpm(0, 48), label="BASS ALONE"),
        Segment(record_start=48, steps=48, bpm=avg_bpm(48, 96), label="THE BAND", level_jump=True),
        Segment(record_start=96, steps=144, bpm=avg_bpm(96, 240), label="VERSE 1"),
        Segment(record_start=240, steps=52, bpm=avg_bpm(240, 292), label="CHORUS: 6 + 7", level_jump=True),
    ],
    step_times=STEP_FILM_S,
    bar_lengths=BAR_LENGTHS,
    sections=SECTIONS,
    rockers=ROCKERS,
    gears=GEARS,
    moments=MOMENTS,
    card_lines=["Not 6/8: a five and a seven taking turns, then a six and a seven."],
    credit_lines=["An independent study of the rhythm. Not affiliated with the band or label."],
    counters=[Counter(("bass5", "bass7"), BARS, 0, K_BAND),
              Counter(("kick5", "kick14", "snare5", "snare7", "bass5", "bass7"), BARS, K_BAND, K_CHORUS),
              Counter(("ckick6", "ckick7", "csnare6", "csnare7", "cbass6", "cbass7"), CBARS, K_CHORUS, K_END)],
    slip=None,                       # nothing drifts: the bars alternate, they do not run against a beat
    cycle_steps=CYCLE, cycle_end=K_CHORUS,
    rack_marks=tuple(range(0, K_CHORUS, CYCLE)) + tuple(range(K_CHORUS, K_END + 1, CCYCLE)),
    stack=["bass5", "kick5", "snare5", "gtr"],
    tooth_types=TOOTH_TYPES,
    camera=[
        CameraMove(0, 0, cam_x=0.0, zoom=1.5),                                             # the opening frame: the bass seesaw alone
        CameraMove(K_BAND - 7, K_BAND - 3, keys=("bass5", "bass7", "kick5", "kick14", "snare5", "snare7", "gtr"), zoom_cap=1.15),
        CameraMove(K_CHORUS - 8, K_CHORUS - 4, keys=("bass5", "kick5", "gtr", "snare7", "cbass7", "ckick7", "csnare7", "cgtr")),
        CameraMove(K_CHORUS + 2, K_CHORUS + 6, keys=("cbass6", "cbass7", "ckick6", "ckick7", "csnare6", "csnare7", "cgtr"), zoom_cap=1.15),
    ],
    anchors=[Anchor("bass melody attack", (600, 2500), ("bass5", "bass7", "cbass6", "cbass7"), tooth_types=("melody", "high")),
             Anchor("kick", (40, 180), ("kick5", "kick14", "ckick6", "ckick7")),
             Anchor("snare", (150, 450), ("snare5", "snare7", "csnare6", "csnare7"))],
    first_step_guess_s=46.82,        # record step 0 (the bar-4 downbeat) in the owner's file, as measured
    cut_note=None,
)

# Gears film: Tool, "Schism"

*Lateralus* (2001). The intro, the first verse and the first chorus as a machine: one tooth per
eighth note. The film's question is **"is this 6/8?"** It is not: the bars alternate, 5/8 then 7/8
(twelve eighths a pair), and in the chorus 6/8 then 7/8 (thirteen). On screen that is literal: two
wheels, five teeth and seven, taking turns on a seesaw.

Built with the `gears-film` skill from `songs/schism.py`; the engine is `gears_film/`. The render
is silent (`schism_silent.mp4`, 1080×1920, 60 fps, no audio track) and runs on the record's own
clock; the owner fits their recording to `record_map.json` and muxes it. **The recording is not in
this repository.**

## Round 2: what changed and why

Round 1 drew each instrument as one wheel per 12-step cycle (a 12-tooth bass, 24-tooth kick and
hi-hat, a 6-tooth "6/8 ear" ghost) and ran on one tempo per section. The owner watched it and two
things changed.

**The mechanism: 5 and 7 in turn.** Schism is not a polymetre; nothing drifts against anything.
Its bar lengths alternate, so the bar lengths are now the gears. Each instrument's 12-step pattern
is split into its 5-step bar and its 7-step bar, drawn as a 5-tooth gear and a 7-tooth gear on a
seesaw (the engine's new `Rocker`). On every bar line the seesaw tips: the incoming gear drops into
mesh in an eased 100 ms that lands its tooth 0 exactly on the line, the outgoing gear lifts off
the rack and rests, dim and still, until its next bar. The engaged gear is bright and turns one
tooth per eighth; because a member turns exactly once per bar, its tooth 0 is on the contact at
every one of its bar lines. The kick's two different sevens (`x.x...x` in one pair, `x.xx..x` in
the next) live on one 14-tooth gear that turns half a turn per 7/8 bar, showing one seven and then
the other. The bass keeps its pitch-typed teeth (pedal short and dark, melody tall, the high G
tallest). The guitar, which doubles the riff, stays one plain wheel of 12 = 5 + 7: the contrast
between the plain wheel and the seesaw is itself the lesson. The 6/8 ghost wheel is gone. At the
chorus the five becomes a six: new seesaws (6 + 7) and a 13-tooth guitar wheel glide in and the old
ones roll away.

For phone legibility the hi-hat (verse) and ride (chorus) are **not drawn**: they play every eighth,
carry no bar information, and as 24- and 13-tooth wheels they set the frame width. Without them
the camera sits a third closer to the small 5- and 7-tooth gears. The band-entry crash (a China on
the bar-12 downbeat) blooms on the kick's seesaw; the chorus crash on the snare's. The captions and
this README say so.

**The clock: every eighth of the record.** Tool play without a click, and the tempo breathes bar by
bar: against one bpm per section the record drifted up to 146 ms (half an eighth) within the film.
The owner beat-tracked every eighth of their recording from the bar-4 downbeat (a Viterbi tracker
over the onset envelope, `gears_film/audio/beat_track.py`), verified the grid's phase per cycle with
pitch-specific bass combs (F3 at steps 0 and 5, E3 at 2 and 7, G3 at 9) and the chorus's snare and
kick combs (within ±10 ms in all 24 cycles), and sent the record time of every step. The song module
carries that list as `step_times` (a new engine field: the picture interpolates inside each step)
and the film follows the band exactly: 292 steps, 81.427 s, 4886 frames, local tempo 106.5-109.3
bpm. The audio is one uncut stretch of the record from 46.8181 s of the owner's file.

Engine additions for round 2, all additive (`songs/art_of_dying.py` and `songs/bleed.py` render
pixel-identical on ten sampled frames): `Rocker` and `Gear(rocker=)`, `Song(bar_lengths=,
step_times=)`, the member phase (`tooth_at`, `member_Te`, `member_meshed`), the seesaw drawing
(beam, pivot, dimmed resting member, side tooth counts), `record_map.json["step_film_s"]`, and the
checks that go with them. Documented in `gears_film/README.md` and the skill.

## The facts (bar numbers are the tab's; bar 1 is the first bar of the song)

Source: Songsterr community tab s6700 (8 tracks), read bar by bar with
`gears_film/transcribe/songsterr_parse.py`. The tab is community work and is not in the repository.

| Bars | Section (tab marker) | Meter | What plays |
|---|---|---|---|
| 1-3 | Intro I | 5/4 5/4 4/4 at 64 bpm | quiet guitar: **not in the film** |
| 4-11 | Intro II - Bass | 5/8 + 7/8 × 4 | bass alone |
| 12-19 | Intro III - with Band | 5/8 + 7/8 × 4 | drums, guitar 1 doubling the riff a third up |
| 20-43 | Verse 1 | 5/8 + 7/8 × 12 | voice; guitars hold chords (20-35) then play the riff shape on A (36-43) |
| 44-51 | Chorus | 6/8 + 7/8 × 4 | bass and both guitars on the A riff shape, ride cymbal |
| 52- | Verse 2 | 5/8 + 7/8 | **not in the film** |

- **The step** is the eighth note (`steps_per_beat = 2`). Record step 0 is the downbeat of bar 4,
  the first bass note: 46.818 s into the owner's file (the official-video audio, which has a video
  intro before the song; the album audio starts at 33.0 s of it). `first_click_offset_s` stays 0:
  the picture starts on record step 0 and the owner's audio build starts the record there.
- **Tempo.** The tab writes 109 bpm from bar 4, 108 from bar 11, 107 from bar 19, 108 at bar 41,
  109 at bar 43, 107 at bar 51. The beat-tracked record (above) breathes between 106.5 and 109.3 bpm
  bar by bar; the means per section are 108.19 (bars 4-11), 108.10 (12-19), 107.37 (20-43), 107.14
  (44-51), and those are what `record_map.json` lists per segment.
- **The bass riff** strikes every eighth; the 5|7 is in the pitches. 5/8 = F3 D2 E3 D2 D2,
  7/8 = F3 D2 E3 D2 G3 D2 D2 (tab: s0f10 s1f0 s0f9 s1f0 s0f12 in tuning D1 A1 D2 G2; s1f0 is the open
  D). Grace-note pickups off the grid at the ends of bars are not drawn.
- **Drums from bar 12** (General MIDI numbers in the tab): kick 36 plays 5/8 `x...x` and 7/8
  `x.x...x` in bars 13, 17, 21, ... but `x.xx..x` in bars 15, 19, 23, ...: its true cycle is two
  5|7 pairs. Snare 38 on the 3rd eighth of the five and the 5th of the seven. Hi-hat 42 on every
  eighth except the one after an open hat 46 (not drawn). China 48 on the bar-12 downbeat.
- **Guitar 1 at bars 12-19** doubles the riff a third higher (Bb3 D3 A3 D3 D3 | Bb3 D3 A3 D3 C4 D3
  D3): same pedal / melody / high shape. Guitar 2 is silent until the chorus.
- **Chorus (bars 44-51).** Bass C3 A1 B2 A1 A1 A1 | C3 A1 B2 A1 D3 A1 A1 (pedal = open A), guitars
  the same shape an octave up. Kick `..xx.x` | `x...x..`; snare on the 6/8 downbeat and the 3rd
  eighth of the seven; ride 51 on most eighths (not drawn); low tom 41 on the 6/8's 3rd eighth and
  crashes on some downbeats (not drawn).

## The gears

Rack pitch 42 world units; `x` is the resting position along the rack; side +1 is above the rack,
−1 below. Film step = record step (no cuts). A seesaw's two gears sit either side of its pivot.

| Seesaw / gear | Label | Teeth | Strikes (own teeth) | Side, x | Engaged (film steps) | Bars |
|---|---|---|---|---|---|---|
| `bass` seesaw: `bass5` | BASS, 5 | 5 | F D E D D: melody 0 2, pedal 1 3 4 (0 white) | −1, −84 | 0-240, every 5/8 bar | 4-43 |
| `bass7` | 7 | 7 | F D E D G D D: melody 0 2, high 4, pedal 1 3 5 6 | −1, 84 | 0-240, every 7/8 bar | 4-43 |
| `kick` seesaw: `kick5` | KICK, 5 | 5 | 0 4; crash on entry | +1, −84 | 48-240, every 5/8 bar | 12-43 |
| `kick14` | 14 | 14 | 0 2 6 (first seven) · 7 9 10 13 (second seven), half a turn per bar | +1, 126 | 48-240, every 7/8 bar | 12-43 |
| `snare` seesaw: `snare5` | SNARE, 5 | 5 | 2 | −1, 336 | 48-240, every 5/8 bar | 12-43 |
| `snare7` | 7 | 7 | 4 | −1, 504 | 48-240, every 7/8 bar | 12-43 |
| `gtr` | GTRS | 12 | the riff shape, 12 = 5 + 7 | +1, 420 | 48-96, 192-240 | 12-19, 36-43 |
| `cbass` seesaw: `cbass6` | BASS 6·7, 6 | 6 | C A B A A A: melody 0 2, pedal elsewhere | −1, 1008 | 240-292, every 6/8 bar | 44-51 |
| `cbass7` | 7 | 7 | C A B A D A A: melody 0 2, high 4 | −1, 1218 | 240-292, every 7/8 bar | 44-51 |
| `ckick` seesaw: `ckick6` | KICK 6·7, 6 | 6 | 2 3 5 | +1, 1008 | 240-292, every 6/8 bar | 44-51 |
| `ckick7` | 7 | 7 | 0 4 | +1, 1218 | 240-292, every 7/8 bar | 44-51 |
| `csnare` seesaw: `csnare6` | SNARE 6·7, 6 | 6 | 0; crash on entry | −1, 1386 | 240-292, every 6/8 bar | 44-51 |
| `csnare7` | 7 | 7 | 2 | −1, 1596 | 240-292, every 7/8 bar | 44-51 |
| `cgtr` | GTRS 6+7 | 13 | the chorus riff shape, 13 = 6 + 7 | +1, 1470 | 240-292 | 44-51 |

Tooth types (song-defined, merged over the engine's built-ins): `pedal` short and dark
(add 9, a darker violet), `melody` tall (add 18), `high` the tallest (add 23, narrower). The riff
gears are typed by pitch, not by hit and rest, because the bass plays every eighth: the tall teeth
show where the melody falls and the dark ones where the pedal note does.

Readouts: the bar counter (5 dots, then 7; 6, then 7 in the chorus) mirrors which gear is down,
coloured by the first gear that strikes there (kick red, snare white, bass violet; pedal dots
darker). LAP n / 20 counts the 5 + 7 pairs; the rack's notched tooth marks every pair (every 12,
then every 13). No SLIP readout: nothing drifts.

## Timeline

Film seconds from the tracked eighths (film 0 = record step 0 = 46.818 s of the owner's file).

| Film s | Film step | Bar | What happens |
|---|---|---|---|
| 0.00 | 0 | 4 | **Hook** "IS THIS 6/8?": the bass seesaw alone, close up; the 5 is down, the 7 rests |
| 1.38 | 5 | 5 | the first tip: the 7 drops onto the bar line as the 5 lifts (then every bar line) |
| 3.33 | 12 | 6 | "FIVE, THEN SEVEN" |
| 6.66 | 24 | 8 | "A SEESAW, NOT A POLYMETRE" |
| 9.98 | 36 | 10 | "WHY IT FEELS LIKE 6/8" |
| 13.31 | 48 | 12 | **The band**: kick seesaw (5 + 14) and snare seesaw (5 + 7) drop in on the downbeat with the guitar wheel; crash on the kick |
| 26.63 | 96 | 20 | **Verse 1**: the voice enters (not drawn); the guitar wheel lifts and rolls away |
| 53.52 | 192 | 36 | the guitar wheel returns (the riff shape on A) |
| 66.87 | 240 | 44 | **Chorus**: three 6 + 7 seesaws and a 13-tooth guitar wheel take the rack; the 5 + 7 seesaws roll away; crash on the snare |
| 70.53, 74.17, 77.78 | 253, 266, 279 | 46, 48, 50 | "SIX AND SEVEN", "THE SUM IS 13", "THEN VERSE 2" |
| 81.43 | 292 | end of 51 | end; the last 1.2 s dim into the closing card |

**No cuts.** The film keeps record steps 0-291 (bars 4-51) straight through. The record map
splits it into four consecutive segments (bass alone / band / verse / chorus) for labels, bar
numbering and the audio fit; the picture's clock is `step_film_s`, the tracked eighths.

## What is arranged versus what the record does

- The hi-hat (every eighth in the verse, open on the 3rd eighth of one seven and the 4th of the
  next) and the ride (most eighths in the chorus) are not drawn. The band-entry China crash is
  drawn on the kick's seesaw and the chorus crash on the snare's.
- The voice is not drawn. The guitars' held chords in bars 20-35 are not drawn as teeth (the
  wheel leaves); guitar 2 (the double track) is silent in the tab until the chorus.
- The kick keeps the two-sevens alternation throughout; the fills in bars 42-43 (into the chorus)
  and 49-51 are not drawn. The tab's occasional crashes on verse downbeats are not drawn.
- Bar 43's last two eighths are an A1 pickup into the chorus; the wheel draws the D pedal. Bar 51's
  last eighth is a D2 pickup into verse 2; the wheel draws the A pedal.
- Chorus drums are simplified to the tab's bars 46-47: the low tom on the 6/8's third eighth, the
  crashes and bar 49's snare variant are not drawn.
- Grace-note pickups (off the grid at the ends of bars) are not drawn.
- The seesaw itself is a drawing device: the record's bars simply alternate in length; the beam
  and pivot show the hand-over, they do not claim a mechanism in the music.

## Files

- `schism_silent.mp4`: the engine's silent render, 4886 frames at 60 fps (81.43 s), H.264 yuv420p.
- `record_map.json`: segments, anchors, the tracked step times and tempos for the owner's audio tools.
- `stills/` and `contact_sheet.png`: the frames checked before and after the render.
- `../../songs/schism.py`: the song data, the only input.

## Credits

Tool, "Schism", *Lateralus* (Volcano Entertainment, 2001). Written and performed by Tool. An
independent study of the rhythm; not affiliated with the band or their label. The recording is
the owner's own copy and is not in this repository; the picture is silent.

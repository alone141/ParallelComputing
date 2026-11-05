# Bleed as gears

Meshuggah, "Bleed" (obZen, 2008): the opening, bars 1-33, drawn as a machine with the
[gears-film engine](../../gears_film/README.md). Song data: [`songs/bleed.py`](../../songs/bleed.py).
The question the film answers: *why does this riff feel like it never lands?* Because the guitars'
cell is three sixteenths long and the bar is sixteen: the bar line falls one tooth later in the cell
every bar and the two meet again only every third bar, while the verses are eight bars long.

An independent study of the rhythm. Not affiliated with the band or their label. The recording is
not in this repository: `bleed_silent.mp4` is the engine's silent picture, and whoever holds a copy
of the record aligns it with `record_map.json` and muxes it (see the skill's "Fit the recording").

## The facts (Songsterr tab s23825, read on a 32nd grid; bars 1-33)

| Fact | Bars |
|---|---|
| 115 bpm, 4/4 throughout except one bar of 2/4 | bar 25 is the 2/4 bar |
| Sections (tab markers): the riff, "Beams of Fire", "My Crimson Liquid", 2/4, "Ripples Ascend", "The Regurgitation" | 1, 9, 17, 25, 26, 34 |
| The riff: two guitars in unison on the low string, the bass and both kick drums play `xxx.x.` on 32nds, a six-32nd cell, from the first 32nd of bar 1; 22, 21, 21 notes per bar; home on the bar line every three bars (96 32nds = 16 cells) | 1-24 |
| The riff stops after one stroke on 32nd 26 of bar 24 (the first 32nd of a cell); 32nds 27-31 are silent | 24 |
| The 2/4 bar: guitars alone, `xxx.x.x.` then `x.xxx.x.`: the new ten-32nd cell once, then one old six-32nd cell; no bass, no drums | 25 |
| The new riff `xxx.x.x.x.` (ten 32nds): 20, 19, 19, 19, 19 notes per bar, home every five bars; bass and kick play it too | 26-33 |
| Kick: both feet, written as GM 36 / 35 alternating in the tab: right, left, right, left per cell (36 on cell 32nds 0 and 2, 35 on 1 and 4); from bar 21 the tab writes one foot only | 1-20 notated, assumed after |
| Snare (GM 38), a three-bar pattern in 32nds per bar: A `1 6 10 16 22 25 30`, B `2 5 10 16 22 26 29`, C `2 6 9 16 21 26 30` (bars 1, 4, 7, … are A). Odd 32nds are pushes a 32nd after the sixteenth grid | 1-24 |
| Snare variants against that pattern: bar 1 has no 32nd 1 (a crash sits there); bar 8 is a fill (`2 5 10 16 22 28`); bars 9 and 17 drop the 2; bars 16, 22 and 24 drop the 30; bar 19 drops the 1 | 1, 8, 9, 16, 17, 19, 22, 24 |
| Snare from bar 26: `6 10 16 21 26 30`, `2 6 9 16 22 26 29`, `2 6 10 16 22 26 30`, `2 5 10 16 22 25 30`, `2 6 10 16 22 26 30`, then bars 31-33 repeat 26-28 (bar 31 adds the 1): a five-bar pattern whose pushes follow the new cell | 26-33 |
| Cymbals on every eighth: china (GM 52) bars 1-8 and 17-24, open hi-hat (46) bars 9-16 and 26-33; crash (49 + 57) on beat 1 of bars 1, 9, 26; 57 alone on beat 1 of 17, 19, 21, 23 | 1-33 |
| The guitars' pitch alternates between two notes (the open low string and the next string's fifth fret) in runs of about 20-27 notes; bars 22-23 and 35 are a lick on the same rhythm | 1-24 |

Cross-check: the tab's seven tracks agree with each other (both guitars, the bass and the kick line
are identical in rhythm for bars 1-24 and 26-33); the bar start times of the tab's synced video give
114.95 bpm over bars 1-33 and 1.04 s for bar 25 (half a bar); the owner's recording measures 114.95 bpm
and starts at 0.24 s. The film is rendered at the nominal 115 bpm
until the measured tempo and offset are pasted into the song data.

## The gears

Step = one sixteenth (4 per beat). On the sixteenth grid the old cell is three teeth
`[double, tall, tall]` and the new one five `[double, tall, tall, tall, tall]`; a `double` tooth has two
tips and strikes twice in its step (the two 32nds). The snare's pushes are a song-defined `late`
tooth (one tip, strikes half a step late, drawn narrower and lower).

| Gear | Teeth | Pattern | Side, x | Engaged (film steps) | Notes |
|---|---|---|---|---|---|
| GUITARS | 12 | `[double tall tall]` × 4 | above, 0 | 0-382 | the slip bracket is drawn on it |
| CYMBAL | 16 | tooth 0 tall and white (the downbeat), every even tooth medium | below, 0 | 0-382 | the bar wheel: the cycle gear of the slip readout; crash on entry |
| KICK | 12 | as GUITARS, left-foot teeth paler | below, 8P | 0-382 | the double tooth is right then left |
| BASS | 12 | as GUITARS | below, 17P | 0-382 | |
| SNARE | 48 | the three-bar pattern: 21 teeth, 5 of them `late` | above, 26P | 0-382 | its wheel is the whole three-bar cycle |
| GTRS 2/4 | 8 | `[double tall tall tall tall double tall tall]` | below, 24P | 384-392 | bar 25, guitars alone |
| GUITARS (new) | 20 | `[double tall tall tall tall]` × 4 | above, 29P | 392-520 | phase 8 |
| HAT | 16 | as CYMBAL | below, 29P | 392-520 | phase 8; crash on entry |
| BASS (new) | 20 | as the new GUITARS | below, 37P | 392-520 | phase 8 |
| KICK (new) | 20 | as the new GUITARS, left-foot teeth paler | below, 45P | 392-520 | phase 8 |
| SNARE (new) | 16 | `2 6 10 16 22 26 30` in 32nds = teeth 1 3 5 8 11 13 15 | above, 45P | 392-520 | phase 8 |

P = 42 world units (the rack pitch). Readouts: the counter is the current bar's sixteen dots
(eight for bar 25), white where the snare strikes, cyan where the cymbal does; `BAR n / 24` for bars
1-24, then the section label; the slip readout counts the bar wheel against the cell: 16 = 5 × 3 + 1,
one tooth per bar, HOME every third bar (bars 4, 7, …, 25), and from bar 26 against the five-tooth
cell: 16 = 3 × 5 + 1, HOME at bar 31.

## Timeline

| Film s | Step | Bar | What happens |
|---|---|---|---|
| 0.0 | 0 | 1 | all five wheels in mesh from frame 0; crash; hook caption |
| 2.1 | 16 | 2 | the bracket: the bar line falls one tooth into the cell |
| 6.3 | 48 | 4 | HOME (gold ring); then every third bar |
| 16.7 | 128 | 9 | "Beams of Fire": the voice enters two teeth into the cell |
| 33.4 | 256 | 17 | "My Crimson Liquid": one tooth into the cell |
| 49.8 | 382 | 24 | the band stops after beat 4's first stroke; the five wheels lift and roll out |
| 50.1 | 384 | 25 | HOME; the 2/4 wheel drops in: guitars alone, eight teeth |
| 51.1 | 392 | 26 | "Ripples Ascend": five new wheels drop in with the crash; the new five-tooth cell |
| 61.6 | 472 | 31 | HOME, five bars later |
| 66.6-67.8 | 510-520 | 33 | fade into the closing card |

Nothing is cut: the film is record steps 0-519 (bars 1-33) in one piece; the two segments in the
record map are consecutive, split only so the audio fit can lock bar 26 on its level jump. The film
ends at the end of bar 33, where the record goes into "The Regurgitation" (a crash and the old cell
again).

## What is arranged, what the record does

- **The kick's feet** are drawn alternating right / left throughout; the tab writes both feet only in
  bars 1-20 and a single line afterwards.
- **The snare in bars 1-24** is drawn as its three-bar pattern on a 48-tooth wheel; the eight
  variant bars listed above (bar 1's crash, bar 8's fill, dropped hits before section changes) are not
  drawn.
- **The snare from bar 26** is drawn on a 16-tooth wheel with bar 28's hits on the sixteenth grid; the
  record's pushes (a 32nd early on one or two hits in most bars, a five-bar pattern) are not drawn.
- **Bar 24, beat 4.5:** the record plays a single stroke where the cell has a double; the wheels
  show the double's second tip.
- **Cymbals:** china and open hi-hat are one wheel, "CYMBAL", in bars 1-24 (the hat takes over for
  bars 9-16 on the record); the crashes on bars 17, 19, 21 and 23 are not drawn; the one at bar 9 is
  not either.
- **The guitars' pitch alternation and the lick in bars 22-23** are not drawn: the wheels show
  rhythm only.
- **The rack's notched teeth** mark every sixteenth tooth: the bar lines in bars 1-24; after the
  eight-step bar 25 they fall on half-bars.

## Files

| File | What |
|---|---|
| `bleed_silent.mp4` | the engine's silent render, 1080×1920, 60 fps, 4070 frames, no audio |
| `record_map.json` | segments and anchors for the audio tools |
| `stills/`, `contact_sheet.png` | frames at the hook, the bracket, each kind of HOME, the vocal entries, the swap, the new set, the closing card |

```
python3 -m gears_film check songs/bleed.py
python3 -m gears_film still songs/bleed.py 50.4 /tmp/swap.png
python3 -m gears_film render songs/bleed.py        # -> gears_film/out/bleed/
```

## Engine additions made for this film

All additive, defaults unchanged; `songs/art_of_dying.py` renders pixel-identical on sampled frames.

- `Gear.phase`: a wheel keyed so that its tooth 0 is on the contact at step `-phase mod teeth`. The
  new cell enters at step 392, which no multiple of 5 divides; the five bar-26 wheels use `phase=8`.
- `Segment.level_jump` and `Segment.search_s` pass through to `record_map.json`.
- `Song.slip` may be a list of `SlipSpec`, each with `k_from` / `k_to`: one stretch of the film gets
  its own cycle gear and beat; HOME also fires exactly where a ranged spec ends (bar 25 here).
- `Song.lap_word` ("BAR 7 / 24") and `Song.lap_label_pos` (the label off the counter row, which with
  sixteen dots would push it into the apps' button column).
- Anchors in the record map count only strikes on the grid (a `late` tooth is no anchor).

Credits: Meshuggah, "Bleed", obZen (Nuclear Blast, 2008). Transcription: Songsterr community tab
s23825 (its revision is named in the downloaded `revisions.json`, which is not committed). This is an
independent study of the rhythm; not affiliated with the band or the label.

# Gears film: Tool, "Schism"

*Lateralus* (2001). The intro, the first verse and the first chorus as a machine: one tooth per
eighth note. The film's question is **"is this 6/8?"** It is not: the riff is 5/8 + 7/8 (twelve
eighths a pair), and the chorus is 6/8 + 7/8 (thirteen).

Built with the `gears-film` skill from `songs/schism.py`; the engine is `gears_film/`. The render
is silent (`schism_silent.mp4`, 1080×1920, 60 fps, no audio track) and runs on the record's own
clock; the owner fits their recording to `record_map.json` and muxes it. **The recording is not in
this repository.**

## The facts (bar numbers are the tab's; bar 1 is the first bar of the song)

Source: Songsterr community tab s6700 (8 tracks), read bar by bar with
`gears_film/transcribe/songsterr_parse.py`; the tab's synced-video bar times checked the tempo.
The tab is community work and is not in the repository.

| Bars | Section (tab marker) | Meter | What plays |
|---|---|---|---|
| 1-3 | Intro I | 5/4 5/4 4/4 at 64 bpm | quiet guitar: **not in the film** |
| 4-11 | Intro II - Bass | 5/8 + 7/8 × 4 | bass alone |
| 12-19 | Intro III - with Band | 5/8 + 7/8 × 4 | drums, guitar 1 doubling the riff a third up |
| 20-43 | Verse 1 | 5/8 + 7/8 × 12 | voice; guitars hold chords (20-35) then play the riff shape on A (36-43) |
| 44-51 | Chorus | 6/8 + 7/8 × 4 | bass and both guitars on the A riff shape, ride cymbal |
| 52- | Verse 2 | 5/8 + 7/8 | **not in the film** |

- **Tempo.** The tab writes 109 bpm from bar 4, 108 from bar 11, 107 from bar 19, 108 at bar 41,
  109 at bar 43, 107 at bar 51. The synced video's bar times give 107.9 (bars 4-19), 107.1 (20-43)
  and 107.3 (44-51); the owner's recording autocorrelates at about 107. The song module carries the
  video-derived values per segment; the owner's measured bpm and offset replace them.
- **The step** is the eighth note (`steps_per_beat = 2`). Record step 0 is the downbeat of bar 4,
  the first bass note (about 12.8 s into the owner's recording).
- **The bass riff** strikes every eighth; the 5|7 is in the pitches. 5/8 = F3 D2 E3 D2 D2,
  7/8 = F3 D2 E3 D2 G3 D2 D2 (tab: s0f10 s1f0 s0f9 s1f0 s0f12 in tuning D1 A1 D2 G2; s1f0 is the open
  D). Grace-note pickups off the grid at the ends of bars are not drawn.
- **Drums from bar 12** (General MIDI numbers in the tab): kick 36 plays 5/8 `x...x` and 7/8
  `x.x...x` in bars 13, 17, 21, ... but `x.xx..x` in bars 15, 19, 23, ...: its true cycle is two
  5|7 pairs, 24 eighths. Snare 38 on the 3rd eighth of the five and the 5th of the seven. Hi-hat 42 on
  every eighth except the one after an open hat 46; the open hat is the 3rd eighth of the first seven
  and the 4th of the second. China 48 on the bar-12 downbeat.
- **Guitar 1 at bars 12-19** doubles the riff a third higher (Bb3 D3 A3 D3 D3 | Bb3 D3 A3 D3 C4 D3
  D3): same pedal / melody / high shape. Guitar 2 is silent until the chorus.
- **Chorus (bars 44-51).** Bass C3 A1 B2 A1 A1 A1 | C3 A1 B2 A1 D3 A1 A1 (pedal = open A), guitars
  the same shape an octave up. Kick `..xx.x` | `x...x..`; snare on the 6/8 downbeat and the 3rd
  eighth of the seven; ride 51 on most eighths (the tab varies bar to bar); low tom 41 on the 6/8's
  3rd eighth; crashes on some downbeats.

## The gears

Rack pitch 42 world units; `x` is the resting position along the rack; side +1 is above the rack,
−1 below. Film step = record step (no cuts).

| Key | Label | Teeth | Strikes | Side, x | Engaged (film steps) | Bars |
|---|---|---|---|---|---|---|
| `bass` | BASS | 12 | 12: 4 melody (F, E), 1 high (G), 7 pedal (D) | +1, 0 | 0-240 | 4-43 |
| `ear` | 6/8 EAR | 6 | 2 painted taps at 0 and 3, **never strike** | −1, 0 | 24-240 | 8-43 |
| `kick` | KICK | 24 | 11: 0 4 · 5 7 11 · 12 16 · 17 19 20 23 | +1, 336 | 48-240 | 12-43 |
| `snare` | SNARE | 12 | 2: 2, 9 | −1, 210 | 48-240 | 12-43 |
| `hats` | HI-HAT (stack row CYMBAL) | 24 | 23: closed (medium) on all but 8; open (tall) at 7 and 20; crash on entry | +1, 756 | 48-240 | 12-43 |
| `gtr` | GTRS | 12 | 12, the riff shape | −1, 462 | 48-96, 192-240 | 12-19, 36-43 |
| `cbass` | BASS 6+7 | 13 | 13: melody 0 2 6 8, high 10, pedal elsewhere | +1, 1092 | 240-292 | 44-51 |
| `cear` | 6/8 EAR | 6 | 2 painted taps, never strike | −1, 1092 | 240-292 | 44-51 |
| `ckick` | KICK 6+7 | 13 | 5: 2 3 5 · 6 10 | +1, 1386 | 240-292 | 44-51 |
| `csnare` | SNARE 6+7 | 13 | 2: 0, 8 | −1, 1302 | 240-292 | 44-51 |
| `cride` | RIDE (stack row CYMBAL) | 13 | 10: 0 1 4 5 · 6-11; crash on entry | +1, 1680 | 240-292 | 44-51 |
| `cgtr` | GTRS 6+7 | 13 | 13, the chorus riff shape | −1, 1596 | 240-292 | 44-51 |

Tooth types (song-defined, merged over the engine's built-ins): `pedal` short and dark
(add 9, a darker violet), `melody` tall (add 18), `high` the tallest (add 23, narrower), `tap`
(add 14, painted at 0.8 but with no strike times: it never flashes). The riff gears are typed by
pitch, not by hit and rest, because the bass plays every eighth: the tall teeth show where the
melody falls and the dark ones where the pedal note does, and that is what makes 5 + 7 audible.

The **6/8 EAR** wheels are not instruments. They show where a listener who hears 6/8 would tap:
every dotted quarter, every third eighth. 12 = 6 + 6, so in the verse they turn in step with the
riff for ever (two of the four taps fall on the open-D pedal, the riff's weakest notes). In the
chorus the pair is 13 eighths, and the ghost falls one tooth behind every lap: that is drawn by
the geometry (the ghost sits under the chorus bass, both start on tooth 0 at step 240). The
engine's SLIP readout is not used: it takes one cycle gear for the whole film, and 12 against 6
has no drift to read out.

## Timeline

Film seconds at the module's nominal tempos (they move when the owner's measured bpm is pasted in).

| Film s | Film step | Bar | What happens |
|---|---|---|---|
| 0.00 | 0 | 4 | **Hook** "IS THIS 6/8?": the riff wheel alone, 12 teeth, zoomed in |
| 3.34 | 12 | 6 | "ONE WHEEL, 12 TEETH" |
| 6.67 | 24 | 8 | the 6-tooth ghost glides in and drops under the riff |
| 10.01 | 36 | 10 | "BUT THE RIFF SAYS 5 + 7": the counter counts the bars |
| 13.35 | 48 | 12 | **The band**: kick (24), snare (12), hi-hat (24) and guitar (12) drop in on the downbeat; China crash on the hi-hat |
| 26.69 | 96 | 20 | **Verse 1**: the voice enters (not drawn); the guitar wheel lifts and rolls away |
| 53.58 | 192 | 36 | the guitar wheel returns (the riff shape on A) |
| 67.03 | 240 | 44 | **Chorus**: six new wheels with 13 teeth (and a new ghost) take the rack; the 12s and 24s roll away; crash on the ride |
| 70.66, 74.30, 77.93 | 253, 266, 279 | 46, 48, 50 | "ONE TOOTH BEHIND", "TWO TEETH", "THREE TEETH, THEN VERSE 2" |
| 81.57 | 292 | end of 51 | end; the last 1.2 s dim into the closing card |

**No cuts.** The film keeps record steps 0-291 (bars 4-51) straight through. The record map
splits it into four consecutive segments (bass alone / band / verse / chorus) only so the
owner's audio fit can measure each one's tempo; the band and chorus segments are flagged
`level_jump` for the fit.

## What is arranged versus what the record does

- The ghost wheels (6/8 EAR) are a teaching device, not a part of the record.
- The voice is not drawn. The guitars' held chords in bars 20-35 are not drawn as teeth (the
  wheel leaves); guitar 2 (the double track) is silent in the tab until the chorus.
- The hi-hat wheel uses the tab's bars 12-15. In the verse the tab adds crashes or a China on some
  downbeats and varies the hat (a rest here, an open there); those bars are not drawn.
- The kick keeps the two-sevens alternation throughout; the fills in bars 42-43 (into the chorus)
  and 49-51 are not drawn.
- Bar 43's last two eighths are an A1 pickup into the chorus; the wheel draws the D pedal. Bar 51's
  last eighth is a D2 pickup into verse 2; the wheel draws the A pedal.
- Chorus drums are simplified to the tab's bars 46-47: the ride varies bar to bar, the low tom on
  the 6/8's third eighth and the crashes are not drawn, nor bar 49's snare variant.
- Grace-note pickups (off the grid at the ends of bars) are not drawn.
- Tempo: the module's per-segment bpm comes from the tab's synced-video bar times, not from the
  recording; the owner's measurement replaces it.

## Engine additions made for this song (additive; `songs/art_of_dying.py` renders unchanged)

- `Segment(level_jump=..., search_s=...)` pass through to `record_map.json` (only when set).
- `Anchor(tooth_types=(...))`: an anchor counts only hits of those tooth types (here the bass's
  melody and high notes, so the comb is asymmetric and can lock the downbeat).
- `Gear(origin=k)`: the film step where the gear's strike position 0 is on the contact line. The rack
  drives every gear with one phase, so a 13-tooth gear entering at step 240 names `origin=240`.
- `Song(rack_marks=(...))`: explicit film steps for the rack's notched teeth (every 12 in the verse,
  every 13 in the chorus).
- The bar counter uses a tooth type's `shade` for its dot (the pedal dots are darker).

## Files

- `schism_silent.mp4`: the engine's silent render, 4894 frames at 60 fps (81.57 s), H.264 yuv420p.
- `record_map.json`: segments, anchors and tempos for the owner's audio tools.
- `stills/` and `contact_sheet.png`: the frames checked before the render.
- `../../songs/schism.py`: the song data, the only input.

## Credits

Tool, "Schism", *Lateralus* (Volcano Entertainment, 2001). Written and performed by Tool. An
independent study of the rhythm; not affiliated with the band or their label. The recording is
the owner's own copy and is not in this repository; the picture is silent.

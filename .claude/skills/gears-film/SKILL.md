---
name: gears-film
description: Build a gears-style rhythm-breakdown film of a song's odd-meter or polymetric section (one gear per instrument on a rack, teeth are hits, silent 1080x1920 60 fps picture for the real recording). Use it when asked for a "gears film", a rhythm explainer in the Art of Dying style, or to add a song to gears_film/.
---

# Gears film

## The idea in one paragraph

A song's rhythm section is drawn as a machine. A rack runs across the middle of a vertical phone
frame and moves one tooth per step (the step is the song's smallest grid: a sixteenth, an eighth,
a 32nd). Every instrument is a gear sitting on or hanging from the rack: its tooth count is its
cycle in steps, its tall teeth are its hits, its stub teeth are its rests, a long tooth is a held
note, a compound tooth is a herta. Gears are hidden until their part plays: they glide in, hang a
beat above their place, drop into mesh exactly on their first step, and lift and roll away when the
part stops. Because every gear advances one tooth per step, a polymetre is visible as geometry: a
45-tooth riff gear against a 12-tooth beat gear slips one tooth per lap and comes home after four.
The picture is silent and runs on the record's own clock; whoever holds the recording aligns it
with the exported `record_map.json` and muxes it. The engine is `gears_film/`; a song is one data
module `songs/<slug>.py`; the worked example is `songs/art_of_dying.py` (Gojira, "The Art of Dying").

## Conventions that work, and why

- **One gear per instrument**, kick and snare separate. Parts that change pattern (a turnaround) are
  a *second gear* with its own engaged range, in a free slot, not a gear that mutates.
- **Teeth are hits, stubs are rests.** Stubs keep every gear meshed, so a snare with 11 hits in 45
  still turns honestly. `long` teeth for held notes; `medium` for a ghost stroke (the ride's ping
  against the bell); `herta` for two 32nds + a 16th inside one eighth (three tips, three flashes).
- **The rack is the drive.** All gears share one phase: a gear's tooth 0 is on the contact line at
  every multiple of its tooth count, so each gear must *enter on a multiple of its tooth count*
  (`check` enforces it). The rack's every-cycle tooth is lighter and notched: the lap is visible on
  the rail itself.
- **Hidden until engaged.** Nothing dormant on screen; the opening frame is the first gear alone and
  the hook caption must land within 2 s with the sound off.
- **Motion:** 1.0 s smoothstep glide in, lifted; a hang of 0.38 s (+0.1 s stagger per gear when
  several arrive together, so the eye can follow); a 0.08 s drop that lands the first hit on the
  grid. Exits: lift at the part's end, 1.0 s eased glide out. No linear starts or stops, no flicks.
- **Camera breaths:** pull wide while gears glide in (so they are seen coming), hold through the
  landing and the leavers' exit, settle on the new engaged set only after the leavers are gone.
  Keep the engaged set inside screen x 60-860 (the right-hand column is the apps' buttons).
- **Flashes are local.** A hit flashes at the gear's contact in its colour; the crash is a bloom at
  the ride's first tooth with a white core; HOME is a gold ring at the cycle gear's contact. Never
  a full-frame wash or lift: on a near-black frame even a 5 % lift reads as a glitch frame.
- **Type sizes** (phone): captions 62 / 32 px at the top; hub number and name inside big gears scale
  with the camera (38-70 / 22-34 px); small gears get a two-line label beside them (36 + 24 px);
  the readout strip is three rows at y 1392 / 1446 / 1502 (counter 52 px, LAP 40, SLIP 30, stack
  30). Nothing essential below y = 1536 (the bottom 20 %) or right of x ≈ 880.
- **Format:** 1080×1920, 60 fps, H.264 yuv420p, no audio track; the frame count is
  round(total seconds × 60) and `render` checks it with ffprobe.
- **Honesty:** say on screen and in the README what is cut from the record and what is arranged
  (the engine shows a `cut_note` for 2.7 s after the first cut).

## Writing songs/<slug>.py, step by step

Start from a verified transcription (tab + a trusted explainer; count the record yourself).
Copy `songs/example_minimal.py` (a template with a tempo map, an eighth-note step, 5/8 + 7/8 bars
and a herta) or `songs/art_of_dying.py` (the full worked example) and replace every number.

1. **Identity:** `slug, title, artist, album, year`, `card_lines` (one line that answers the film's
   question), `credit_lines` (independent study, not affiliated).
2. **Time:** nominal `bpm`, `steps_per_beat` (4 for sixteenths, 2 for eighths, 8 for 32nds; triplet
   grids use 3 or 6), `first_click_offset_s` (0.0 until the recording is measured). Both bpm and
   offset can be overridden at render time with `GEARS_BPM` / `GEARS_OFFSET`.
3. **The record map:** `sections` (name, record step where it starts, the steps per bar in order,
   e.g. `(16,)*10 + (20,)` for ten 4/4 bars and one 5/4) for bar numbering; `segments` = the kept
   stretches of the record in film order, each `Segment(record_start, steps, bpm=None, label)`.
   A gap between consecutive segments is a cut. Give a segment its own `bpm` when the tab (or the
   measured recording) changes tempo: film time is the cumulative sum of segment step lengths.
4. **Gears:** for each instrument `Gear(key, name, short, colour, teeth, strikes, side, x, ranges,
   sound=..., ...)`. `strikes` is `{position: tooth type}` over `0..teeth-1`; unlisted positions
   are stubs. `side` +1 above / -1 below the rack. `x` is a multiple of 42 (the pitch); gears on
   the same side that are on screen together must not overlap (`check` tests it: centre distance
   ≥ sum of tip radii + 40, tip radius = teeth × 42/π/2 + 18). `ranges` are film step ranges,
   several allowed. `sound` picks the flash style (`click pulse kick snare note chug ride`); the
   ride's tall teeth ring the bell and its medium teeth ping; `first_hit="crash"` for a crash on
   entry. `white_position` paints one tooth white (the cycle start). `colour2` + `colour2_positions`
   for a second pitch (C1/C2). `stack` makes a second gear light an existing stack row. `lead`
   staggers simultaneous arrivals. `label_dx/dy` step a small gear's label aside.
5. **Choosing tooth counts** so every gear lands on tooth 0 at its entries and the slip stays
   readable: a gear's count must divide every one of its entry steps. A 4/4 part in a 45-step cycle
   that enters at 180, 540 and 720 fits a 12-tooth gear (three beats per turn; 180, 540, 720 are
   multiples of 12) better than a 44-tooth one, and makes the slip bracket 30° per tooth instead of
   8°. A six-step turnaround cell fits a 12-tooth gear that goes round twice. For Bleed-style 32nd
   figures use an eighth-note step and `herta` teeth (24-tooth gears, not 96). Prefer the smallest
   count that divides the entries and holds the pattern.
6. **Slip and HOME:** `SlipSpec(cycle_gear, beat_steps, beat_gears)`. Slip per lap = cycle teeth
   mod beat steps; laps to home = beat / gcd. The readout fades in when the first beat gear enters;
   HOME flashes where the slip returns to 0 while that gear is engaged. A non-drifting pair gets no
   HOME. The bracket is drawn on whichever `beat_gears` entry is engaged.
7. **Counters:** `Counter(gears, bars, k_from, k_to)`: the current bar's steps as dots, filled in the
   colour of the first listed gear that strikes there, hollow for a rest. Bars of unequal length
   (5 5 5 3 3 3 | 5 5 5 3 3, or 5/8 + 7/8) are just the list.
8. **Moments:** `Moment(step, title, subtitle, kind, colour)`: one caption per lap is a good rhythm;
   kinds `hook` (step 0, up from frame 0), `big` (appears at once, for an entry), `home` (gold),
   `caption`. Make the two biggest moments of the record the two biggest captions.
9. **Camera:** leave `camera` empty for automatic framing (wide while gears arrive, settle after
   the leavers go), or give `CameraMove` keyframes (`keys` to fit, or `cam_x`/`zoom`); the first
   entry is the opening frame. Explicit moves are better once the layout is known.
10. **Anchors:** one or two distinctive instruments with a frequency band, for the audio tools.
11. `cycle_steps`, `cycle_end` (where LAP n/N stops and the segment label takes over), `stack`
    (row order), `cut_note`.

## Deciding segments and cuts

Keep the stretch that carries the idea and nothing that only repeats it: for Art of Dying the eight
bars where the pulse plays alone were cut because the film is about the pulse *against* the riff.
Cut on a bar line where the record's downbeat lands on the cycle's step 0, so the picture
continues without a visible jump and the audio tools can crossfade (`crossfade_s`). Say what was
cut (`cut_note` and the README). Make the film end on something: a turnaround, a cadence; the
engine dims the last `end_fade_s` seconds into a closing card for the audio fade.

## The still-first loop

```
python3 -m gears_film check songs/x.py                      # data validation + printed timeline
python3 -m gears_film still songs/x.py 19.3 /tmp/a.png      # one frame at a film time
python3 -m gears_film render songs/x.py                     # frames (4 workers), silent MP4, frame-count check, record_map.json
python3 -m gears_film export-record-map songs/x.py          # the map alone
```

Render stills at: frame 0 (the hook), every entry (the glide, the hang, the landing), each HOME,
the widest camera frame, the busiest frame, an exit, the last second. Put them in a contact sheet.
Look for: labels overlapping, anything essential below y 1536 or right of x 880, a gear hanging
over another, a caption longer than two lines, the readout strip touching a lower gear. Then a
full render (~20 min on 4 CPUs for 80 s), then frame-by-frame checks around the big moments
(crop 10-20 consecutive frames into a strip): no washed-out frame, no jump, mean luma continuous.

## Checks to run before handing over

- `check` passes (teeth within counts, entries on tooth 0, ranges inside the film, no overlaps,
  moments on valid steps).
- `render` reports the frame count equal to round(total × 60) from ffprobe.
- `record_map.json` lists every segment with `film_step`, `record_step`, `steps`, `bpm`, `film_s`,
  `step_s`, and the anchors' `record_steps`.
- A README for the song: the gear table, the timeline with the cuts, what is arranged versus what
  the record does, credits and the disclaimer.

## What the audio tools need (record_map.json)

```
{"bpm": nominal, "steps_per_beat": n, "first_step_guess_s": 0.5,
 "segments": [{"film_step": k, "record_step": r, "steps": n, "bpm": b, "film_s": s, "step_s": d}, ...],
 "anchors": [{"name": "...", "band_hz": [lo, hi], "record_steps": [...]}, ...],
 "crossfade_s": 0.04, "fade_out_s": 1.2, "gain_db": -2.4}
```
The owner's tools measure the recording per segment (bpm and offset) against the anchors. **Paste
the measured values back into the song module** (`bpm=` on each `Segment`, `first_click_offset_s`),
re-run `check` (the frame count changes) and `render`.

## Transcription and the real recording

The picture is rendered silent from a verified transcription; the recording is fitted and muxed
afterwards by whoever holds a copy of it. Nothing here downloads music: the owner supplies the file.

### 1. Get the tab

```bash
node gears_film/transcribe/songsterr_fetch.js "tool schism" "schism" /tmp/tabs/schism
python3 -I gears_film/transcribe/songsterr_parse.py summary /tmp/tabs/schism 40
```

`summary` prints the tracks, the tempo map, every time-signature change and section marker, and
notes per bar per track: that is the song's structure (who plays where, where the parts enter).
Songsterr's tabs are community work; `revisions.json` names the revision you used. Keep the
downloaded JSON out of the repository (it is their content); commit only the rhythm you derive.
If the tab has `video_points.json`, its bar start times (for the synced YouTube video) are a free
check on the tempo.

### 2. Derive the patterns

```bash
python3 -I gears_film/transcribe/songsterr_parse.py grid   /tmp/tabs/schism 3 4-11 --res 8
python3 -I gears_film/transcribe/songsterr_parse.py stream /tmp/tabs/schism 3 4-11 --res 8
```

`grid` shows one row per note (General MIDI drum numbers on drum tracks: 35/36 kick, 38 snare,
37 side stick, 42/46 hi-hat, 49/57 crash, 51 ride, 53 ride bell, 52 China; string/fret on pitched
tracks) on a grid of `--res` notes per whole note. Choose `--res` so every hit lands on the grid
(`~` means off-grid: raise it, or the part uses tuplets). `stream` concatenates the bars, prints the
run lengths (the 5 5 5 3 3 3 of The Art of Dying) and the shortest period that repeats exactly:
that period in steps is the gear's tooth count.

Write the findings as facts before designing anything: tempo (and where it changes), the step unit,
the cycle length and grouping, each instrument's hits within the cycle (as step indices), where each
part enters and leaves (bar numbers and record steps), and every bar of odd length. Have a second,
independent pass check the facts against the raw JSON and at least one outside source (a drummer's
transcription or lesson); The Art of Dying's cross-check caught a wrong README claim and a snare rule
this way. Everything in the song data must trace to those facts.

### 3. Decide the film's segments

A mixed master cannot be layered one part at a time: the picture follows the record's own entries.
Choose which record steps the film keeps (`segments` in the song data): usually from the first hit
of the hook to the end of the first full-band block, with dead stretches removed at bar lines (The
Art of Dying drops its pulse-only bridge). A cut lands on a downbeat where something new enters, so
the entry masks the crossfade. Export the map the audio tools need:

```bash
python3 -m gears_film export-record-map songs/schism.py   # writes record_map.json
```

### 4. Fit the recording

```bash
ffmpeg -i Schism.mp3 -ar 48000 -c:a pcm_s24le record_48k.wav
python3 gears_film/audio/fit_record.py record_48k.wav record_map.json fit.json
```

The fit slides a comb of an anchor instrument's hits (from `record_map.json`) over a band-limited
onset envelope, per segment, and reports each segment's start time in the record and its tempo. Read
the "score vs half-step off" ratio: above about 2 the anchor locked; near 1 it did not (pick a more
percussive anchor or a better band: 2–9 kHz for sticks, rims, snares and cymbals; 40–180 Hz for kick
and low drums, which lock less sharply). Expect surprises: The Art of Dying's band section sits 45 ms
earlier than the grid extrapolated through its bridge, and drifts 0.2 bpm slower; sections of real
records are edits.

Two switches per segment in the song data (they pass through to `record_map.json`):
`"level_jump": true` on any segment that begins with a band entry or a cut, so the fit anchors on the
level jump (the downbeat sits 0–50 ms before it) and cannot be fooled by a one-step ambiguity; and
`"search_s"` to widen the search around the record's own clock when the record has a bigger edit
there. If the fit prints AMBIGUOUS and the segment has no level jump, choose a more percussive anchor
or two anchors in different bands.

Paste each segment's measured bpm into the song data's tempo map and the first segment's offset into
`first_click_offset_s`, then re-render the picture, so picture and sound share one clock.

### 5. Build, mux, verify

```bash
python3 gears_film/audio/edit_record.py build record_48k.wav record_map.json fit.json film_audio.wav
gears_film/audio/mux_verify.sh out/<slug>_record_silent.mp4 film_audio.wav out/<slug>_record.mp4 record_map.json
```

`build` cuts the segments at their measured starts, crossfades 40 ms on each boundary, fades the
last 1.2 s and trims the gain (−2.4 dB by default: a +0.6 dBTP master lands at −1.6 dBTP after AAC;
a +2 dBTP master needs about −4 dB). `mux_verify.sh` copies the video stream (never re-encode it,
never use `-shortest`), reports the frame count, loudness and true peak, then decodes the audio back
out of the MP4 and checks every anchor against the picture's grid: on-grid energy must beat
half-a-step-off clearly and the best shift must sit within a few ms. Three to nine milliseconds is
normal; tens of milliseconds means a segment is on the wrong grid.

The muxed film contains the recording: keep it out of the repository (commit the silent render) and
hand it to the owner. Expect the platforms to recognise the music: Instagram may mute it, YouTube
will claim it, TikTok usually keeps it; the silent render exists so the licensed track can be added
in-app instead.

## Known limits of the engine

Single rack (no second drive); a gear cannot change pattern without leaving (use a second gear);
steps are uniform within a segment (swing or tuplets inside a step need a compound tooth type,
add one to `TOOTH_TYPES`); the closing card and readout strip have fixed positions; the safe box
is tuned for 1080×1920 only.

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

_Placeholder: the owner's session fills this in with its own tools (how the tab was verified, how
the recording was measured, the crossfade and mux procedure)._

## Known limits of the engine

Single rack (no second drive); a gear cannot change pattern without leaving (use a second gear);
steps are uniform within a segment (swing or tuplets inside a step need a compound tooth type,
add one to `TOOTH_TYPES`); the closing card and readout strip have fixed positions; the safe box
is tuned for 1080×1920 only.

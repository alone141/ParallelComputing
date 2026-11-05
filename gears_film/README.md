# gears_film

A song-agnostic engine for gears-style rhythm-breakdown films: one gear per instrument on a rack
that moves one tooth per step, teeth as hits, stubs as rests, hidden until engaged, a silent
1080×1920 60 fps picture on the record's own clock. Grown out of
[`experiments/art_of_dying/gears/`](../experiments/art_of_dying/gears/README.md) (rounds 1-6); the
engine reproduces that round-6 film pixel for pixel from `songs/art_of_dying.py`.

```
python3 -m gears_film check songs/art_of_dying.py
python3 -m gears_film still songs/art_of_dying.py 57.9 /tmp/bar42.png
python3 -m gears_film render songs/art_of_dying.py          # -> gears_film/out/art_of_dying/art_of_dying_silent.mp4 + record_map.json
python3 -m gears_film export-record-map songs/art_of_dying.py
GEARS_BPM=139.85 GEARS_OFFSET=0.12 python3 -m gears_film render songs/art_of_dying.py
```

Needs `numpy`, `Pillow`, `ffmpeg`, and the Inter / Inter Display / JetBrains Mono fonts at the
paths in `render.py`. Outputs go to `gears_film/out/<slug>/` (ignored by git); a song's deliverables (README, the silent
MP4, stills, contact sheet, record_map.json) are committed under `films/<slug>/`.

| File | What |
|---|---|
| `song.py` | the data model (`Song`, `Segment`, `Section`, `Gear`, `Moment`, `Counter`, `SlipSpec`, `CameraMove`, `Anchor`), the derived timeline (tempo map, film→record steps, slip/HOME, captions, counters), `record_map()`, `check()`, `load_song()` |
| `render.py` | the frame renderer: rack, gear sprites, motion (glide / hang / drop / lift / exit), camera, flashes, labels, readout strip, closing card |
| `build.py` | parallel frame rendering, ffmpeg encode, frame-count check, record map |
| `__main__.py` | the CLI |
| `sound/` | the borrowed synthesised instruments and DSP (a library; the films are silent) |
| `../songs/art_of_dying.py` | the worked example (Gojira) |
| `../songs/schism.py`, `../films/schism/` | Tool, "Schism": 5/8 + 7/8 on an eighth-note step, teeth typed by pitch, a 6/8 ghost wheel, 13-tooth chorus gears with `origin` |
| `../songs/bleed.py`, `../films/bleed/` | Meshuggah, "Bleed": a 3-tooth cell under a 16-tooth bar, `double` and `late` teeth, every gear in mesh from frame 0, a ranged slip list, a 2/4 swap |
| `../songs/example_minimal.py` | a template with a tempo map, an eighth-note step, 5/8 + 7/8 bars, a herta tooth |
| `../.claude/skills/gears-film/SKILL.md` | how to build a new song |

## The data module in brief

`songs/<slug>.py` exports `SONG = Song(...)`:

- identity: `slug, title, artist, album, year, card_lines, credit_lines`
- time: `bpm` (nominal; env `GEARS_BPM` overrides), `steps_per_beat` (2, 4, 8, ...),
  `first_click_offset_s` (env `GEARS_OFFSET`)
- record map: `sections` (bar lengths in steps, for bar numbering), `segments` (kept record stretches
  in film order, each with an optional `bpm`: the tempo map; gaps are cuts; `level_jump` and
  `search_s` pass through to the record map for the audio fit)
- tooth types: built-in `tall stub long medium double herta`, plus the song's own
  `tooth_types={name: dict(add, width, bright, subhits, tips, ext, shade)}` (type teeth by hit/rest,
  by pitch, or by subdivision)
- gears: `Gear(key, name, short, colour, teeth, strikes={pos: tooth type name},
  side=±1, x, ranges=((k_in, k_out), ...), sound, first_hit, colour2, colour2_positions,
  white_position, stack, lead, label_dx, label_dy, origin)`; `origin` is the film step at which
  strike position 0 is on the contact line, for a gear whose count does not divide its entry step
  (`phase=p` is a deprecated alias for `origin = -p mod teeth`)
- rockers (bars that alternate in length, 5/8 + 7/8): `rockers=[Rocker(key, name, short, colour,
  members={bar_steps: gear_key}, ranges, side, lead, swap_s=0.1, lift=46, label_dx, label_dy)]` and
  `bar_lengths=(5, 7, 5, 7, ...)` over the whole film. A rocker is a seesaw carrying one gear per bar
  length; on every bar line it tips the bar's gear into mesh (an eased `swap_s` that lands its tooth
  on the line) and lifts the other, which rests dim and still. Members are Gears with `rocker=key`,
  the rocker's `side` and `ranges`, their own `x`; a member's tooth count is a multiple of its bar
  length (a 14-tooth gear serves alternating 7/8 bars half a turn each) and it keeps its own phase
  (it advances only while its bar is on), so it need not enter on a multiple of its teeth
- clock: `step_times=[...]` (film seconds of every step 0..total_steps, e.g. a beat-tracked record
  without a click) replaces the segments' bpm as the picture's clock; the record map then carries
  `step_film_s` and each segment's mean bpm
- readouts: `counters` (bar lengths per cycle, which gears fill the dots; a tooth type's `shade`
  colours its dot), `slip=SlipSpec(cycle_gear, beat_steps, beat_gears, k_from, k_to)` or a list of
  them (slip per lap and HOME are derived; a ranged spec also flashes HOME where it ends),
  `cycle_steps`, `cycle_end`, `rack_marks` (explicit notched-tooth steps when the cycle length
  changes), `lap_word`, `lap_label_pos`, `stack`, `cut_note`
- captions: `moments=[Moment(step, title, subtitle, kind, colour)]`
- camera: `camera=[CameraMove(...)]` or empty for automatic framing
- audio tools: `anchors` (`Anchor(name, band_hz, gears, tooth_types=(), on_grid=True)`),
  `first_step_guess_s`, `crossfade_s`, `end_fade_s`, `gain_db`. `first_click_offset_s` is leading
  picture before step 0, not the record's offset: it stays 0 unless a lead-in is wanted

Derived: `total_steps`, `total_s`, `n_frames = round(total_s × 60)`, `step_time(k)`, `k_at(frame)`,
`record_step(k)`, `record_bar(k)`, `hits_at(key, k)`, `slip_at(k)`, `homes`, `record_map()`.

## record_map.json

```
{"bpm": 140.0, "steps_per_beat": 4, "first_step_guess_s": 0.5,
 "segments": [{"film_step": 0, "record_step": 0, "steps": 540, "bpm": 140.0, "film_s": 0.0, "step_s": 0.107142857, "label": "..."}, ...],
 "anchors": [{"name": "tribal pulse", "band_hz": [40, 160], "record_steps": [180, 184, ...]}, ...],
 "crossfade_s": 0.04, "fade_out_s": 1.2, "gain_db": -2.4,
 "song": ..., "title": ..., "artist": ..., "first_click_offset_s": 0.0, "total_steps": 744, "total_s": 79.714286, "frames": 4783, "fps": 60}
```

## Regression against the round-6 film

`songs/art_of_dying.py` rendered through the engine was compared with the round-6 frames on ten
sampled frames (0, 600, 1159, 2025, 3440, 3473, 3482, 3700, 4650, 4782): mean absolute difference
0.0000 on all ten before the LAP-position fix; after it, frames inside the 6/4 turnaround differ only
by the LAP label moving 38 px right (it now clears the six-dot counter it used to touch). Frame
count 4783, as before.

## What it cannot do yet

One rack only (no second drive); a gear cannot change pattern without leaving (model it as a
second gear in a free slot, or as a rocker when the pattern follows the bar length); the step is
uniform inside a segment unless `step_times` is given (swing would need per-type
subhit fractions, which a song-defined tooth type can already carry); a rocker holds one gear per
bar length and its members cannot carry a SlipSpec; `steps_per_beat` is one
value per song; the readout strip, caption block and
closing card are at fixed positions tuned for 1080×1920; automatic camera framing is serviceable
but explicit `CameraMove`s look better; no synthesised preview mix (the sound library is there, the
mixer was not generalised).

# Experiment 5 · rhythm game (round 3: the record's own clock)

A vertical film (1080×1920, 60 fps, 79.71 s, **4783 frames, silent**) that explains the intro of
**Gojira – "The Art of Dying"** (*The Way of All Flesh*, 2008) as a rhythm game being played, timed to
the original recording. The audio is the band's recording, supplied and muxed by the repository owner
outside this repository; it is not in the repo and this session never had it. The film is an independent
study, not affiliated with Gojira, their label or their publishers.

Six lanes, one per part, light up exactly as the record brings the parts in. The lap line marks every
45-sixteenth cycle; because 45 = 11 × 4 + 1 it lands one sixteenth later against the beat every lap,
and the game judges each lap as it crosses (+1 LATE, +2 LATE, +3 LATE, PERFECT). The two big moments
are the record's own: **bar 12**, where the tribal pulse enters on the beat exactly as the lap comes home
(PULSE IN · PERFECT), and **bar 42**, where the whole band drops in on the second homecoming
(REALIGNED · BAND IN) while the clicks and the pulse go dark. The 6/4 turnaround at bar 53 is the final
flourish, and a FULL COMBO results card sits under the owner's 1.2 s audio fade.

## Files (round 3)

| File | What |
|---|---|
| `art_of_dying_rhythm_game_record_silent.mp4` | the silent film: H.264 yuv420p, 1080×1920, 60 fps, 4783 frames, no audio stream |
| `stills/*.png` | eight frames: hook, lap 4 (+3), bar 12, bar 23, bar 42, lap 16, the 6/4, the results card |
| `contact_sheet.png` | one frame every 3.2 s |
| `rhythm.py` | the timeline: pattern sets, entries and exits, the cut, the turnaround, bar numbers; `--bpm` and `--offset`; writes `timeline.js` |
| `game.html`, `game.js`, `timeline.js` | the game, every frame a pure function of film time; every timing a function of the sixteenth index k |
| `render.js`, `contact_sheet.py`, `build.sh` | the recorder and the build |
| `round2/` | the round-2 synthesised, looping film and its complete build (sound engine included) |

## Timeline (140.000 bpm, first click at 0.000 s)

One sixteenth = 15/140 s = 107.143 ms; one lap = 45 sixteenths = 4.8214 s. k is the film's sixteenth
index; film time of k = `OFFSET + k × 15 / BPM`. Record mapping: film k = record k for k < 540, and
record k + 128 for k ≥ 540 (the cut). The beat (quarter note) is on k ≡ 0 (mod 4) throughout.

| Film time | k | Tab bar | What happens | Picture |
|---|---|---|---|---|
| 0.000 s | 0 | 1 | first rim click; clicks alone over the held choir note | hook over the running game; the PULSE lane shows the grid as ghost notes and the laps are judged +1/+2/+3 LATE against it |
| 19.286 s | 180 | 12 | tribal pulse joins on every beat, with its stick grace note | PULSE IN · PERFECT (first homecoming), lane flash |
| 38.571 s | 360 | 23 | the pulse comes home | REALIGNED card (4 laps = 180 steps = 45 beats) |
| 57.857 s | 540 | 42 | **the cut**: record bars 34–41 (pulse alone, 128 sixteenths, 13.714 s) removed; the band enters together on a beat at t = 0 of the cycle: kick + snare, ride (crash on beat 1, bell on beats, ghost between), bass on C1, two palm-muted guitars; clicks and pulse stop | REALIGNED · BAND IN · BAR 42; KIT, RIDE, BASS, GUITAR flash on, CLICK and PULSE go dark; the drift bracket and judgement move to the RIDE lane (the bell carries the beat) |
| 77.143 s | 720 | 53 | the 6/4 turnaround, 24 sixteenths: K K K S . S ×4, bass C1 ×3 then C2 ×3, guitars C2 ×3 then E♭2 ×3 | 6/4 TURNAROUND banner · PERFECT; brackets of 6; C1/C2 and E♭ labels on the notes |
| 78.429 s | 732 | 53 | | FULL COMBO results card (1140 hits, 16 laps + 6/4) under the audio fade |
| 79.714 s | 744 | | end | |

Bar numbers follow the tab (ten bars of 4/4 and one of 5/4 per 180 sixteenths), so the HUD's BAR counter
jumps from 33 to 42 at the cut.

## Re-rendering with a measured tempo or offset

```bash
cd experiments/art_of_dying/rhythm-game
BPM=139.85 OFFSET=0.080 ./build.sh      # about 9 minutes; prints the frame count
```

`rhythm.py --bpm B --offset O` writes `timeline.js`; the game reads `step = 15 / B` and `offset = O`
and places every note, lap line, judgement, banner and caption from k. The frame count becomes
`round((O + 744 × 15 / B) × 60)`. Captions are scheduled in film seconds at 140 bpm and scaled by the
step, so they stay on their laps.

## What changed from round 2

- No synthesis, no loop, no one-lane-per-lap unlocks: the parts enter as the record enters them.
  The round-2 film and its sound engine are preserved in `round2/`.
- The pulse-only bridge (bars 34–41) is cut; the film is 744 sixteenths, 4783 frames.
- The beat lane is the pulse until bar 42 and the ride bell after it, so the drift bracket and the
  lap judgement never lose their reference.
- A BAR counter replaces PARTS in the HUD; the 6/4 turnaround has its own brackets, pattern banner and
  note labels; a results card ends the film.
- `BPM` and `FIRST_CLICK_OFFSET_S` are build parameters.

## What worked

- Laps 1–12 are the record's own drama and the game vocabulary fits them: the ghost grid gets judged
  LATE three times, the pulse arrives on the first PERFECT, slips again, and the band drops on the second.
- The lane switch at bar 42 (CLICK and PULSE dark, four lanes flash on) reads as a level change.
- The turnaround's K K K S . S is visible as three kicks, a white snare, a hole, a snare.
- Rebuilding for a different tempo is one command; nothing in the picture is hand-timed.

## What failed or was cut

- **Not checked against the recording.** No copy of the record was available here, so the alignment is
  by arithmetic (140.000 bpm, first click at 0). If the record's tempo is 139.9 the picture drifts
  about 0.4 s by the end; hence the BPM/OFFSET parameters.
- **Nineteen seconds of clicks alone.** That is the record. The hook, the group brackets, the ghost grid
  and three LATE judgements have to carry it; I could not test whether they do.
- **The cut is invisible in the picture.** The bar counter jumps from 33 to 42 and nothing else marks
  it; an honest "bars 34–41 cut" label would have cost a caption slot in the busiest moment.
- **Round-2 postmortem items remain**: the judgements are about the lap, not a fallible player; the
  caption band covers the top of the lanes; the drum fills and ride details of the actual recording
  (bars 42–65 and after) are reduced to the tab's steady pattern.
- **The ending is a guess at the audio fade**: the results card starts 1.3 s before the end so it is
  fully up when the fade begins.

## With another day

1. Align to the recording (onset-detect the first click and the bar-42 crash) and lock BPM/OFFSET.
2. A fallible player for laps 1–4: show a tap on every ghost beat and let it visibly miss the groups.
3. A small "bars 34–41 cut" marker at the lane switch.
4. Watch it with the record.

## Credits

Music analysed: Gojira, "The Art of Dying", from *The Way of All Flesh* (Listenable / Prosthetic,
2008). Rhythm facts cross-checked against the Songsterr community tab, Raphael Pannier's transcription
and Yogev Gabay's explainer. This is an independent, non-commercial study; it is not affiliated with
or endorsed by Gojira or their label. The recording is the owner's licensed copy, used in the app, and
is not in this repository. The round-2 instruments and DSP (`round2/sound_voices.py`,
`round2/sound_dsp.py`) are the repository owner's own work, copied with attribution.

---

# Round 2 (superseded: the synthesised, looping film)

Everything in this section now lives in `round2/` (its own `build.sh` rebuilds it as
`round2/art_of_dying_rhythm_game_synth.mp4`). The text is unchanged.

A looping vertical film (1080×1920, 60 fps, 38.6 s, one loop) that explains the intro of
**Gojira – "The Art of Dying"** (*The Way of All Flesh*, 2008) as a rhythm game being played. Six lanes,
one per part, unlock one lap at a time in the record's order: rim clicks, tribal pulse, kick + snare,
ride, bass, guitar. A lap line marks every 45-step cycle; because 45 = 11 × 4 + 1 the line lands one
step later against the pulse every lap, and the game judges each lap as it crosses: +1 LATE, +2 LATE,
+3 LATE, then **PERFECT** on lap 5, where the screen celebrates **REALIGNED · 4 laps = 180 steps =
45 pulses** and the bass enters. Lap 7 is the full mix; on lap 8 the parts peel away in reverse until
only the clicks remain, which is exactly where the film started, so it loops.

This is an independent study of the song's rhythm. It is not affiliated with Gojira, their label or
their publishers. No part of the recording is used; every sound is synthesised.

**The stack is a teaching arrangement, not a moment from the song.** On the record the rim clicks never
play together with the kit (they stop at bar 42 when the band enters), and the tribal pulse rests in
bars 42–65. The film layers all six parts on one cycle so the eye can compare them; the README of the
owner's own version makes the same choice.

## Files

| File | What |
|---|---|
| `art_of_dying_rhythm_game.mp4` | the film (H.264 yuv420p, AAC 256k, 1080×1920, 60 fps, 2314 frames, one loop) |
| `stills/*.png` | six frames: hook (0.6 s), pulse lap (7.0 s), kit focus (13.0 s), realignment (19.45 s), guitar in (26.0 s), full mix (31.0 s) |
| `rhythm.py` | the single source of truth: pattern sets, layer entries and peel, focus windows; writes `timeline.js` |
| `game.html`, `game.js`, `timeline.js` | the game: every frame a pure function of film time, on a 2D canvas |
| `render.js` | Playwright recorder: steps the clock per frame, pipes PNGs into ffmpeg |
| `soundtrack.py` | the mix: places every hit from `rhythm.events()`, calibrates layer balance, reverb, true-peak limiter |
| `sound_voices.py`, `sound_dsp.py` | the instruments and the DSP, copied unchanged from the owner's version (see Credits) |
| `build.sh`, `contact_sheet.py`, `loop_check.py` | rebuild everything; review tools |
| `contact_sheet.png` | one frame every 1.6 s of the final film |

## How to run

```bash
cd experiments/art_of_dying/rhythm-game
./build.sh            # about 5 minutes on 4 cores; prints ffprobe, ebur128 and the loop-seam check
```

`python3 rhythm.py` alone prints the timeline and self-checks the pattern sets against the tab
(the kit lap must spell `KKKS.KKKS.KKKS.KKSKKSKKSKKKS.KKKS.KKKS.KKSKKS`). Opening `game.html` in a
browser plays the game in real time, silent.

### Timeline

| Lap | Steps | What happens |
|---|---|---|
| 1 | 0–44 | hook, rim clicks alone; the pulse lane shows the grid as hollow ghost notes |
| 2 | 45–89 | PULSE IN · the lap is judged +1 LATE |
| 3 | 90–134 | KIT IN (kick L/R alternating, snare white) · +2 LATE |
| 4 | 135–179 | RIDE IN (bell on the beat, ghost ping between) · +3 LATE |
| 5 | 180–224 | **REALIGNED** · PERFECT · BASS IN (hold tails through each group's rest) |
| 6 | 225–269 | GUITAR IN (two chugs per sixteenth) |
| 7 | 270–314 | FULL MIX |
| 8 | 315–359 | FULL COMBO banked, then guitar, bass, ride, kit and pulse leave every six steps; clicks only by step 345 |

Each entering part gets a focus lap: its lane glows, the others dim to 42 % and duck −8 dB. The film
is 2314 frames = 38.5667 s; the music grid is 360 × 15/140 s = 38.5714 s, so the loop seam is 4.8 ms
early (a twentieth of a sixteenth). Film time 0 = song 0:00 (bar 1 beat 1).

### Sound

Every instrument is numpy: side-stick rim click + stick click; the stacked tribal hit (surdo, floor
tom, block, castanets, sticks) with its stick grace note a 32nd early; a tight gated kick; a rimshot
snare; a ghosted ride ping and the ride bell; a picked bass on C1 (32.7 Hz) with the group's last note
held; two palm-muted guitars on C2 through a high-gain stage and a cabinet, panned left and right; a
low choir on C that breathes but never ducks. Layer balance is set by measurement (each layer alone over
the full-mix lap, K-weighted, relative to the kit). A circular reverb and a circular true-peak limiter
make the loop seamless. Master: −14.0 LUFS integrated, −2.0 dBTP before AAC; the final MP4 measures −14.0 LUFS and −1.5 dBTP
(ffmpeg `ebur128=peak=true`).

## What changed from round 1

- Six lanes instead of two; every part of the intro is in the picture, in the record's order.
- The film starts with the game already running (step 0 on the hit line), not after a 1.5 s pre-roll:
  action on frame 1, and the loop needs it.
- Judgements now mean something: each lap is judged against the pulse (+1/+2/+3 LATE, PERFECT). The
  per-hit PERFECT text is gone.
- The four laps of waiting are gone: a new part arrives every lap, each with its own banner and
  focus lap, and the realignment lands on lap 5 with the bass.
- Captions are shorter and fewer words carry the "why"; the brackets, the dashed holes, the drift gap
  and the lane unlocks do most of the work.
- The soundtrack is the owner's instrument set through my own timeline, properly mastered.
- The film loops.

## What worked

- **A part per lane, a lap per part.** Six unlocks in 30 s keeps a payoff every 4.8 s, and the lock
  glyphs on the empty lanes promise more to come. The PARTS 3/6 counter and the coloured dots give the
  sound-off viewer the structure.
- **Lap judgement.** "+2 LATE" popping in the pulse lane as the line crosses turns the drift into the
  game's own vocabulary; PERFECT on lap 5 is the celebration's cue.
- **The kit as feet.** Kicks alternate L/R positions in the lane, so the "feet alternate" fact is
  visible without a word, and the snare in white reads as the accent.
- **Hold tails for the bass** show the group's last note held through the rest, which was the one
  bass fact worth showing.
- **Shared timeline.** `rhythm.py` writes `timeline.js`; the JS hit count (848) matches the Python
  event count, so the picture cannot drift from the sound.
- **The owner's sound engine** gave a real kick, a cracking snare, a deep pulse with its grace note and
  guitars that chug instead of fizz, and it masters to spec in nine seconds.

## What failed or was cut

- **Judgements are still about the lap, not the player.** The brief's other option (a player who
  misses on the odd groups until taught) was cut: a miss means silence, and silence breaks the music the
  film exists to teach. So nobody ever misses, and FULL COMBO is a foregone conclusion.
- **The focus lap for the ride is too quiet.** The ride is balanced −8 LU under the kit; when every
  other part ducks −8 dB for its focus lap, lap 4 drops about 2.5 LU below its neighbours (per-lap
  loudness: −18.5, −13.5, −12.1, −15.6, −14.9, −14.1, −12.3, −14.6 LUFS). I added a +4 dB boost to the
  ride in its own focus lap (+1.5 dB for the others); it moved the number by 0.1 LU, because the duck on
  the loud parts dominates. A smaller duck for that one lap is the real fix.
- **The caption band covers the top of the lanes** for about 0.3 s of every note's approach, and it
  hides the LAP tag on lines that are still high up. The alternative (captions below the hit line) is in
  the apps' UI zone.
- **The combo hides on lap 8** so the seam shows no number jump, but the HUD still flips from "4/4 ·
  +3 STEPS" to "1/4 · ALIGNED" at the seam. That flip is a lap line crossing, so it is honest, but a
  viewer who notices the loop will notice it.
- **Not checked by ear.** Levels, balance, true peak and the seam are measured (own BS.1770 meter and
  ffmpeg's), not listened to.
- **The record's dynamics are flattened.** The real intro builds over 2:16; this stack builds over 29 s
  and then peels in 5 s. The 6/4 turnarounds, bars 34–41 (pulse alone) and the verse are not here.

## With another day

1. A fallible player on an optional pass: the game shows a second run where a "4/4 player" taps the
   pulse and the chart shows their MISS on every lap line until lap 5, then the real run.
2. Captions out of the lanes: a narrower play area with a caption column, or a voice-over.
3. Per-layer focus levels (the ride louder in its focus lap), and a short riser into the realignment.
4. Listen to it.

---

# Round 1 (superseded)

A short vertical film (1080×1920, 60 fps, about 43 s) that explains the rhythm of the intro of
**Gojira – "The Art of Dying"** (*The Way of All Flesh*, 2008) by showing a rhythm game being played.
Notes fall down two lanes toward a hit line, exactly in time with the music. One lane is the rim-click
pattern (groups of 5 and 3), the other is the quarter-note pulse. A lap line marks each 45-step cycle.
Because 45 = 11 × 4 + 1, the lap line lands one step later against the pulse every lap, and the game
counts the drift (+1, +2, +3) until the fourth lap, when the line and the pulse coincide again. That is
the moment the game celebrates: **REALIGNED · 4 laps = 180 steps = 45 pulses · DRUM UNLOCKED**, and
the big drum enters, exactly where it enters on the record.

This is an independent study of the song's rhythm for a visual-explainer channel. It is not affiliated
with Gojira, their label or their publishers. The audio in the film is synthesised from scratch (no
recording is used); see "Adding the official track" below.

## The seed and how I read it

The seed was "a rhythm game you watch being played": lanes, notes approaching a hit line, judgements,
a combo counter, the odd grouping visible in the note patterns, and the 4-lap realignment as the moment
the game celebrates. The question behind it: can the language of a game (approach, hit, judgement,
combo) explain the 45-cycle and the drifting beat better than a diagram?

My reading:

- **Two lanes, not four.** The intro is really two things disagreeing: a 45-step pattern and a 4-step
  pulse. One lane each. A kick/snare lane would have been faithful to bars 42+ but would have buried
  the point.
- **The grouping is drawn on the notes.** Each 5 or 3 gets a bracket with its number, and the skipped
  last step of every group is drawn as a dashed empty slot. So "x x x x ." and "x x ." are literally
  visible as four notes and a hole, two notes and a hole.
- **The drift is a gap you can see.** Every lap line carries a red bracket from the previous pulse up
  to the line, labelled +1, +2, +3. On lap 5 the bracket collapses and the HUD flips to ALIGNED.
- **The game's own grid is the pulse.** For the first four laps the record has rim clicks only, so the
  pulse lane shows hollow "ghost" notes: the metronome the game counts in. When the four laps resolve,
  the ghosts become solid drum notes ("DRUM UNLOCKED"), which is what the record does at bar 12.
- **Combo as evidence.** The combo counter is the running count of hits; the final screen reads
  "364 HITS · 8 LAPS · 90 PULSES". The hit counts are real: 34 clicks per lap, one pulse every 4 steps.
- **Game-faithful timeline.** Film time 1.5 s = cycle step 0 = bar 1 beat 1 of the tab. Eight laps
  (bars 1–23 of the tab's 10×4/4 + 1×5/4 bar structure) are played, then a 3.3 s end card.

## Round-1 files

Round 1 used `synth.py` (a cruder numpy synth, deleted in round 2) and a two-lane `game.js`; its
text below is unchanged.

## Files (round 1)

| File | What |
|---|---|
| `art_of_dying_rhythm_game.mp4` | the film (H.264 yuv420p, AAC, 1080×1920, 60 fps) |
| `stills/*.png` | six frames: hook (0.9 s), lap 2 (7.2 s), lap 4 line with the +3 gap (15.3 s, the best one), the realignment (20.95 s), drum in (25.0 s), end card (40.4 s) |
| `game.html`, `game.js` | the game: a pure function of film time, drawn on a 2D canvas |
| `render.js` | Playwright recorder: steps the clock per frame, pipes PNGs into ffmpeg |
| `synth.py` | the soundtrack, synthesised with numpy only |
| `build.sh` | runs everything and prints the ffprobe/ebur128 checks |

## How to run

```bash
cd experiments/art_of_dying/rhythm-game
./build.sh             # ~5 minutes on 4 cores; FPS=30 CRF=24 ./build.sh for a quick pass
```

Or open `game.html` in any browser to watch the game play in real time (silent). The recorder never
uses the real-time loop: it calls `renderFrame(t)` with `t = frame / fps`, so the output is
deterministic and every note is hit on the exact sample.

Timing model (all in `game.js` and `synth.py`): 140 bpm, one sixteenth = 15/140 s = 107.14 ms,
cycle = 45 sixteenths = 4.8214 s, 4 laps = 180 sixteenths = 19.2857 s. Rests at
t ∈ {4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44}; pulse on every t ≡ 0 (mod 4) counted from step 0.

## Adding the official track

The film's clock is the record's clock: **film time 1.5 s = 0:00.000 of the song (bar 1 beat 1)**, and
the film plays bars 1–23 (eight 45-step laps), ending on the downbeat of lap 9 (bar 23 beat 2 of a
4/4 count; step 360). If you replace the synthesised audio with the official recording inside
TikTok/Instagram, align the first rim click to 1.5 s and keep the tempo at 140. The record has no
metronome thump during laps 1–4; my soundtrack adds a very quiet one so the drift can be heard before
the drum enters. With the real track the ghost-note lane is silent, which is also fine.

## What worked

- **The drift bracket.** Watching the red gap between the pulse note and the lap line grow by one
  note-height per lap, and then vanish, is the clearest thing in the film. It needs no words.
- **Holes in the notes.** The dashed empty slot at the end of each group makes "5" and "3" readable
  as shapes (four bars and a hole, two bars and a hole) even without the bracket numbers.
- **Determinism.** Stepping the clock per frame and screenshotting gave an exact 60 fps film with
  notes landing on the frame; no dropped frames, no audio drift. 2603 frames in under 4 minutes
  (the MP4 carries 2602 after `-shortest` trims to the audio length).
- **Ghost → solid as the celebration.** The drum entering on the first realignment is a fact of the
  record that the game format turns into a reward ("DRUM UNLOCKED"). That was the best fit between
  seed and song.
- **numpy drums** were enough: rim click (band-passed noise + 2.3 kHz "tock"), stacked hit (swept
  sines for kick/tom/surdo + wood block + castanet bursts + a stick grace note a 32nd early), a
  choir-ish drone on C, a crash on the two realignments. Mixed to about −15 LUFS integrated with a
  peak limiter only (true peak about −1 dBFS); the dynamics are left alone.

## What failed or was cut

- **Judgements are decoration.** A bot plays the chart perfectly, so every hit is PERFECT and the
  combo never breaks. The seed's "hit judgement" idea carries no information here. A real version
  would let the player be the drummer who keeps 4/4 and show the *misses* when the chart drifts.
  I cut that for time: it needs a scripted "human" with errors and a second lap of design.
- **The game only half explains; the diagram does the rest.** The drift bracket and the HUD readout
  (LINE VS PULSE: +2 STEPS) are diagrams smuggled into the game. Pure game language (notes, hits,
  combo) did not say "one step late per lap" on its own. Verdict on the experiment's question: the
  game makes the drift *felt* (you watch the lap line slide) but the *why* still needed a caption.
- **Seven captions in 40 s** is a lot of reading on a phone. I kept them because the sound-off viewer
  has nothing else; a voice-over would let most of them go.
- **Length.** 43 s. The first four laps are honest to the record (clicks only) but risk the 2-second
  test. The hook ("THE PULSE SLIPS. Every lap. Watch it come home.") over already-falling notes is my
  bet; I have no way to test it here.
- **Only laps 1–8.** Bars 34–41 (4/4 only), the full band from bar 42, the 6/4 turnarounds, the
  snare pattern and the feet-alternating kick are not in the film. Two lanes could not show them
  without becoming a transcription.
- **Synthesised sound is crude.** The stacked hit has no room, the drone is not a choir, and the
  first balance was a broadband wall (−8 LUFS) that I had to tame by shortening decays and dropping
  the saturation. I checked it with ebur128 and spectrograms, not with ears. A one-pass `loudnorm`
  also crushed the dynamics to 3.8 LU of range and resampled to 96 kHz; I replaced it with a plain
  limiter at 48 kHz.
- **No sidechain between seed and song build.** The record's own build (clicks → clicks + drum →
  drum alone → full band) would have been a natural "level 1 / 2 / 3" structure; I only used the
  first step of it.

## With another day

1. Make the player fallible: a scripted "drummer" who holds 4/4 and *misses* the clicks that fall off
   the grid, so the judgements (EARLY / LATE / MISS) do the explaining and the realignment is the
   first PERFECT lap.
2. Add a third lane from bar 42 (kick + snare as one lane of two colours) and run the 6/4 turnaround
   as a "bonus bar".
3. A 15-second cut: one lap of teaching, then time-lapse through laps 2–4 to the celebration.
4. A proper voice-over, which would remove most captions.
5. Render at 2× and downsample for cleaner glow; it is slightly soft at 1×.

## Credits

Music analysed: Gojira, "The Art of Dying", from *The Way of All Flesh* (Listenable / Prosthetic,
2008). Rhythm facts cross-checked against the Songsterr community tab, Raphael Pannier's transcription
and Yogev Gabay's explainer. This is an independent, non-commercial study; it is not affiliated with
or endorsed by Gojira or their label. No part of the recording is used in the film.

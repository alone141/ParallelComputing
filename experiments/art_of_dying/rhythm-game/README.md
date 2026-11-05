# Experiment 5 · rhythm game

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

## Files

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

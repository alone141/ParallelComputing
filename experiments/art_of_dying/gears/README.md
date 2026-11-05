# Experiment 1: gears

A 19-second looping film for TikTok and Instagram Reels about the rhythm of the intro of
**Gojira, "The Art of Dying"** (*The Way of All Flesh*, 2008). It answers one question on
screen: *why does the beat drift, and when does it come back?*

This is an independent study of the song's rhythm. It is not affiliated with Gojira, their
management or their label. The audio in the film is synthesised from scratch; no part of the
recording is used.

Deliverables in this folder:

| File | What |
|---|---|
| `art_of_dying_gears.mp4` | the film, 1080×1920, 60 fps, 19.28 s, H.264 + AAC |
| `stills/*.png` | five frames (frame 0, 300, 760, 1120, 1156) |
| `rhythm.py` | the facts of the intro as data, and the film's timing |
| `render.py` | the frame renderer (Pillow + numpy) |
| `audio.py` | the drum synth (numpy only) |
| `build.sh` | renders frames, builds the audio, muxes the MP4, exports the stills |

## The idea

Two gears mesh at the centre of the screen. Both advance exactly one tooth per sixteenth note.

- **The riff wheel (top, steel, 45 slots).** One lap is one cycle of the riff. Eleven of its
  45 teeth are missing: the last slot of every 5 and every 3 (`x x x x .` and `x x .`). When a
  tooth strikes the mesh point you hear a rim click; when a gap passes, silence. The group sizes
  `5 5 5 3 3 3 5 5 5 3 3` are engraved on its rim band. Tooth 0, the cycle start, is painted white.
- **The beat wheel (bottom, bronze, 44 teeth).** Every fourth tooth is painted amber. A painted
  tooth at the mesh point is the big stacked "tribal" hit that falls on every quarter note.

Because 45 = 44 + 1, the riff wheel finishes each lap one tooth after the beat wheel has
completed its eleven beats. The white tooth arrives at the mesh one tooth after an amber one,
then two, then three. After four laps the slip is four teeth, which is one whole beat, and the
white tooth and an amber tooth meet at the mesh together: the gears come home. That takes
180 sixteenths = 45 beats = 19.3 s, and it is exactly where the film loops, so the loop point
*is* the payoff.

The picture carries the idea with the sound off: a `LAP n / 4` readout, a four-box `SLIP`
meter that fills one box per lap, a bracket at each cycle start that measures how far ahead the
amber tooth already is, and four captions, one per lap:

1. WHY DOES THIS BEAT DRIFT? (the hook, visible from frame 0)
2. 45 TEETH AGAINST 44
3. ONE TOOTH EVERY LAP (45 = 11 × 4 + 1)
4. FOUR TEETH IS ONE BEAT

With the sound on, the synthesised rim clicks and the tribal hit are sample-locked to the teeth.

## The seed and how I read it

The seed was *gears*: the 45-sixteenth cycle and the 4-sixteenth beat as two wheels that turn
together, hits where teeth meet, the four-lap realignment as the gears coming home.

Calls I made:

- **45 and 44 teeth, not 45 and 4.** A 4-tooth pinion against a 45-tooth wheel is the more
  literal reading (it turns 11¼ times per lap), but a 4-tooth gear is a tiny star next to a big
  wheel and nothing on it is legible on a phone. Two wheels of almost the same size stacked
  vertically fill a 9:16 frame, put the mesh point in the middle of the screen, and make the
  "one tooth too many" argument visible: you can see that one rim is a little longer.
- **The rests are missing teeth.** This is the part of the seed I like best. The rim pattern
  does not need its own cam: the riff wheel *is* the pattern, and the 5+5+5+3+3+3+5+5+5+3+3
  grouping becomes a physical object you could 3D-print.
- **No kick/snare wheels.** The film covers the bars 12-33 texture (rim clicks + tribal hit)
  only. A third wheel for the full-band pattern from bar 42 was planned and cut (see below).
- **Loop length = exactly four laps.** The home moment is frame 0. On first viewing frame 0 is
  the hook plus a gold flash and a small "home" readout you cannot yet understand; on the second
  pass it is the payoff. That is a risk: the viewer gets the answer by watching it twice, which
  is also what TikTok rewards.
- **Sound: synthesised, not the record.** Rim click = filtered noise knock plus two short
  partials; tribal hit = pitched kick, floor tom, a 58 Hz surdo, wood block, castanet noise and a
  skin transient, soft-clipped together, with a stick grace note a 32nd before; a held C2 drone
  stands in for the choir. If the owner prefers the original, the film is bar-aligned: frame 0 is
  the downbeat of bar 1 of the intro at 140 bpm, and the film's tempo is 140.02 bpm (one loop is
  1157 frames instead of 1157.14, a 2 ms difference over 19 s). Add the track at 0:00. The
  record's cycle structure also repeats every 180 sixteenths, so any later loop can start at
  0:19.29 × n (bar 12 beat 2, bar 23 beat 3, ...). Note that on the record the big hit only
  enters at bar 12, so over the first loop the amber wheel marks beats the record keeps silent.

## How to run

```
cd experiments/art_of_dying/gears
./build.sh            # ~5 min on 4 CPUs: frames (Pillow), audio (numpy), ffmpeg mux, stills
```

Only `numpy`, `Pillow` and `ffmpeg` are needed. `python3 render.py --still 0 300` renders single
frames; `python3 audio.py out.wav` builds the track alone. Fonts: Inter Display, Inter and
JetBrains Mono from the system font directories (paths at the top of `render.py`).

The MP4 is 1080×1920, 60 fps, yuv420p, loudness-normalised to −14 LUFS. Bottom ~20 % of the
frame and the right-hand column carry no essential text (readouts sit left of the mesh; the
bottom wheel runs under the app UI on purpose).

## What worked

- The mechanism is honest. There is no animation cheat: both gears rotate at a constant rate,
  tooth positions are computed from the sixteenth count, and every flash and every sound comes
  from a tooth crossing the mesh line. The drift and the homecoming fall out of the geometry.
- The missing-teeth idea makes the 11 rests visible as shape, not as notation.
- Pre-rendering each gear once as an RGBA sprite and rotating it per frame keeps a pure Pillow
  pipeline at ~0.6 s per 2× supersampled frame; the whole film renders in about four minutes.
- The loop. Four laps is 1157 whole frames at 60 fps, the beat wheel's paint pattern has 11-fold
  symmetry, so the last frame flows into the first without a visible jump, and the audio tails
  wrap (the track is rendered over two loops and the second is kept).

## What failed or was cut

- **A true 11¼-turn pinion.** The physically literal "45 against 4" was dropped for legibility.
  Something is lost: with 44 teeth the "slip" has to be measured against *any* amber tooth,
  and a careful viewer may ask why the amber tooth that comes home is not the one that started.
  (It is a different amber tooth, four slots on, and they are indistinguishable by design.)
- **Teeth meshing exactly at the hit.** Real meshing gears have their teeth interleaved, so a
  tooth of one wheel is centred at the mesh point when a *gap* of the other is. I split the
  difference: the riff tooth is centred 27 ms after the audio hit and the beat tooth 27 ms before.
  That is under two frames and invisible, but it is a fudge.
- **The third wheel.** A 44-tooth kick/snare wheel (kick on every slot except six, snare at the
  end of each group) for bars 42+ was in the plan. It would have needed a second act, a second
  layout and twice the sound design; with 90 minutes it was cut rather than half-done.
- **Phone-size legibility of the engraved 5s and 3s.** They are 30 px tall and rotate with the
  wheel; half of them are upside down at any moment. They are texture, not a reading aid.
- **The first second is busy.** The gold home flash, the hook title and the readouts all land
  at once. I kept the flash because the loop needs it, but it fights the title for a beat.
- **Sound.** The tribal hit is a decent thud but nothing like the record's stack of surdo, toms
  and wood; the drone is a thin stand-in for the choir. Nobody will mistake it for Gojira.

## With another day

- Build the second act: the kick/snare wheel sliding in for bars 42+, the 6/4 turnaround as a
  visible extra half-tooth, then the verse's tempo change as the gears audibly slowing.
- Add a slow-motion insert at each cycle start (a 0.5 s "freeze" with the bracket), then catch up,
  so the slip measurement gets screen time without breaking the loop.
- Render the gears in 3D (Blender) with real involute teeth and shadows at the mesh; the flat
  sprites read as diagram, which is fine for the idea but not as satisfying as metal.
- A proper cover: bass on C1, the two palm-muted guitars as a tick on every sixteenth, ride bell
  on beats, and a real mix.
- Test on a phone and tune the type sizes; everything was judged from stills on a desktop.

Credits: Gojira, "The Art of Dying", *The Way of All Flesh* (Listenable / Prosthetic, 2008).
Pattern facts cross-checked against the Songsterr community tab, Raphael Pannier's transcription
and Yogev Gabay's explainer, as provided in the brief.

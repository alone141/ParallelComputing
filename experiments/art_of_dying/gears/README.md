# Experiment 1: gears

A 58-second looping film for TikTok and Instagram Reels about the rhythm of the intro of
**Gojira, "The Art of Dying"** (*The Way of All Flesh*, 2008). Two meshing wheels that are also
music-box discs play the intro part by part and answer one question on screen: *why does the beat
drift, and when does it come back?*

This is an independent study of the song's rhythm. It is not affiliated with Gojira, their
management or their label. The audio is synthesised from scratch; no part of the recording is used.

Deliverables in this folder (round 2; round 1 is kept under `round1/`):

| File | What |
|---|---|
| `art_of_dying_gears.mp4` | the film: 1080×1920, 60 fps, 3471 frames = 57.85 s, H.264 + AAC, −14 LUFS |
| `stills/*.png` | six frames: the hook, the pulse entering, home, kick + snare, bass, guitars |
| `rhythm.py` | the facts of the intro and the film's timeline (parts, focus laps, duck, peel): one source for picture and sound |
| `render.py` | the frame renderer (Pillow + numpy) |
| `soundtrack.py` | the mixer and master (adapted from the owner's study, see credits) |
| `sound_voices.py`, `sound_dsp.py` | the synthesised instruments and the DSP (borrowed unchanged, see credits) |
| `build.sh` | renders frames, the soundtrack, the MP4 and the stills (about 15 min on 4 CPUs) |
| `round1/` | the 19-second round-1 film, its stills and its code |

## Round 2: the machine plays the whole intro

### The idea

Two gears mesh in the middle of the screen; both advance exactly one tooth per sixteenth.

- **The riff wheel** (top, steel, 45 slots) is one cycle of the riff per lap. Its outer teeth are
  the rim clicks, and eleven teeth are missing: the last slot of every 5 and every 3. Inside the rim
  it carries two pin tracks, like a music-box disc: **kick (red) and snare (white)** on the first,
  **bass (violet)** on the second, with each group's last note drawn as a held arc through its rest.
- **The beat wheel** (bottom, bronze, 44 teeth) carries the 4/4 parts: every fourth tooth is painted
  amber for the **tribal pulse**; its first pin track is the **ride** (a big pin for the bell on the
  beat, a small one for the ghosted stroke between), its second the **guitars** (a pin on every tooth).
- Every track has a **follower** on the line between the two centres. A pin or tooth passing under an
  engaged follower is a hit: a flash in that part's colour, and its sound. A lifted follower is a
  part that has not entered yet (or has left): the pins are there, nothing strikes them.
- Because 45 = 44 + 1, the beat wheel is one tooth further round every time the riff wheel completes
  a lap. A `SLIP` meter fills one box per lap, a bracket on the beat wheel measures the gap at each
  cycle start, and after four laps the slip is a whole beat and the white cycle-start tooth meets an
  amber tooth at the mesh again: **home**, with a gold flash.

The film is twelve laps (57.85 s). One part enters per focus lap; the drift story gets its four
clean laps first:

| Lap | Caption | What happens |
|---|---|---|
| 1 | WHY DOES THIS BEAT DRIFT? | rim clicks alone; the 5s and 3s count up in the left column |
| 2 | 44 TEETH IS ELEVEN BEATS | the pulse engages; the 45:44 choice is explained in one line |
| 3-4 | ONE TOOTH EVERY LAP / FOUR TEETH IS ONE BEAT | slip 2, slip 3 |
| 5 | HOME | the first homecoming (19.3 s), nothing new enters |
| 6 | KICK AND SNARE | focus lap on the riff wheel's first track |
| 7 | THE RIDE | focus lap on the beat wheel's first track |
| 8 | THE BASS | focus lap on the riff wheel's second track |
| 9 | HOME AGAIN | five parts, second homecoming (38.6 s) |
| 10 | TWO GUITARS | the last part enters |
| 11 | THE WHOLE STACK | full mix, no focus |
| 12 | BACK TO THE CLICKS | full mix, then the parts peel away in reverse over the last 3.2 s |

The last frame is the first frame (rim clicks alone, every other follower lifted), so the film
loops; the third homecoming is the loop seam itself, which is why it is silent.

During a focus lap the new part's pins are the brightest thing on screen, its follower carries a
name chip, its row in the left-hand stack grows, the caption takes its colour, and every other part
is dimmed to 42 % in the picture and ducked by 8 dB in the sound; then it settles into the stack.

**This is a teaching arrangement, not a moment from the record.** On the record the rim clicks play
alone (bars 1-11), then with the tribal hit (12-33); the hit plays alone in bars 34-41; from bar 42
the kit, bass and guitars take over and the clicks stop, and the tribal hit rests until bar 66. The
film stacks all six so that each one can be heard against the others and against the wheels.

### Sound

Everything is synthesised in numpy at 48 kHz: side-stick rim clicks doubled with a stick click, the
stacked tribal hit (bass drum, surdo, floor tom, wood block, castanets, sticks) with its stick grace
note a 32nd before, a tight gated kick, a rimshot snare, ghosted ride pings and the ride bell, a
picked bass on C1 whose group-final notes are held through the rest, two palm-muted guitars chugging
C2 left and right (the right one 1-4 ms late), and a low choir on C under everything. A circular
reverb and a true-peak limiter make the loop seamless; the balance is set by K-weighted measurement
per part. The instruments, the DSP and the mixer are the owner's own (see credits); what is mine is
the timeline they play and the film tempo. Mastered to −14.0 LUFS integrated and −2.0 dBTP in the
WAV so the AAC stays under −1 dBTP; the final MP4 measures are in the build log below.

Picture and sound share one clock: 12 laps = 540 sixteenths = 3471 whole frames at 60 fps, so one
sixteenth is 107.130 ms (139.98 bpm, 0.012 % slower than the record) and the loop is exact in both.

### How to run

```
cd experiments/art_of_dying/gears
./build.sh            # ~15 min on 4 CPUs: frames (Pillow), soundtrack (numpy), ffmpeg mux, stills
python3 rhythm.py     # prints the timeline and runs the self-check
python3 render.py --still 0 1170 --out stills      # single frames
python3 soundtrack.py build/x.wav --start 24 --dur 10   # an excerpt of the soundtrack
```

Only `numpy`, `Pillow` and `ffmpeg` are needed. Fonts: Inter Display, Inter, JetBrains Mono (system
paths at the top of `render.py`). The MP4 is yuv420p; the bottom 20 % of the frame and the right-hand
column carry no essential text (readouts sit in the left column; the wheels run under the app UI).

### Round-1 postmortem items, and what became of them

- **The busy first second.** Fixed. Frame 0 is the hook title, the riff wheel clicking, and the group
  counter. No home flash, no lap or slip readout (they fade in when the pulse enters in lap 2). The
  homecomings happen at 19.3 s and 38.6 s, where they are earned, and the loop seam is quiet.
- **The engraved 5s and 3s.** Dropped, and replaced with something legible: a group counter in the
  left column that shows the current group's size as a numeral and its slots as dots, filled click by
  click, the rest as a hollow ring. It counts `5 ●●●●○` and `3 ●●○` all film long.
- **45 against 44, or 45 against 4.** Kept 45:44 and explained it on screen in one line at the top
  of lap 2: "44 teeth is eleven beats. The riff wheel has 45, one tooth more." A 4-tooth pinion is
  the truer picture of "the beat" but a four-tooth star next to a 45-tooth wheel is illegible on a
  phone and has nowhere to carry the ride and guitar tracks.
- **Sound.** Replaced wholesale with the owner's instruments (credited below): a real grace note,
  a kick with a beater click, a snare that cracks, a bass that sits under the guitars.

### What worked

- The music-box reading of the seed. Pins on tracks with one follower each turned out to be the
  honest way to add parts without adding wheels: the two wheels still carry the whole argument (45
  against 44), and every part sits on the wheel whose period it actually has.
- Nothing is faked. Pins, teeth, flashes and sounds are all computed from the same sixteenth count;
  the drift, the slip bracket and the homecomings fall out of the geometry.
- Rendering stayed cheap: two pre-rendered wheel bodies rotated per frame, about 160 pins drawn as
  polygons, additive glow sprites. About 0.5 s per 2× supersampled frame, four workers.
- The soundtrack renders in 23 s and the mix builds the way the picture does (short-term loudness
  rises from about −20 LUFS in lap 1 to about −13 LUFS in the full stack).

### What failed or was cut

- **No true pinion.** The literal 45-against-4 seed is still not on screen; it is explained, not shown.
- **The 27 ms fudge** from round 1 remains: teeth interleave, so a tooth is centred under the mesh a
  quarter-sixteenth from the audio grid. Invisible, but it is there.
- **A lot of small type.** The left column (group counter, lap, slip, six-row stack) is set in
  21-24 px mono. It is readable on a phone held normally, but it is a column of UI, and a viewer who
  only watches the wheels will miss the names of the parts. The name chips on the followers during
  focus laps are the backup.
- **The dormant tracks are visible from frame 0.** Dim pins for parts that have not entered add
  texture to the hook. I kept them because they promise more and because the loop needs the last frame
  to equal the first; a cleaner hook would show a bare wheel and add pins as parts enter.
- **The bass hold arcs** read as little "H" shapes at phone size rather than as a sustained note.
- **Nothing was auditioned.** The instruments were tuned by their author from measurement, not by ear,
  and I could not listen either; the limiter works hard on the pulse (up to 7.8 dB on transients).
- **The record's own arrangement is not reproduced** (clicks never play with the kit; the pulse rests in
  bars 42-65). This is said in the film's lap-11 caption and here, not hidden, but a viewer who wants
  the song's structure gets a teaching stack instead.
- **No phone test.** Everything was judged from stills and contact sheets on a desktop.

### With another day

- Show the true 45:4 somewhere: a small four-lobed cam on the beat wheel's axle, turning 11¼ times
  per riff lap, with its own counter.
- Play the record's arrangement as a second act: the clicks dropping out at bar 42, the 6/4
  turnarounds as a visible half-tooth, the pulse returning at bar 66.
- Pins that rise out of the disc as a part enters (and sink as it leaves), instead of dim pins
  present from frame 0.
- Listen to it, on a phone, and remix.

## Round 1 (kept for the record)

The 19-second first version, in `round1/`: the two wheels and the drift story only, a numpy drum
synth of my own, and the loop point as the payoff.


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

### The seed and how I read it (round 1)

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

### How round 1 was run

```
cd experiments/art_of_dying/gears/round1
# the round-1 renderer and synth are kept there; see the file headers
```

Only `numpy`, `Pillow` and `ffmpeg` are needed. `python3 render.py --still 0 300` renders single
frames; `python3 audio.py out.wav` builds the track alone. Fonts: Inter Display, Inter and
JetBrains Mono from the system font directories (paths at the top of `render.py`).

The MP4 is 1080×1920, 60 fps, yuv420p, loudness-normalised to −14 LUFS. Bottom ~20 % of the
frame and the right-hand column carry no essential text (readouts sit left of the mesh; the
bottom wheel runs under the app UI on purpose).

### What worked (round 1)

- The mechanism is honest. There is no animation cheat: both gears rotate at a constant rate,
  tooth positions are computed from the sixteenth count, and every flash and every sound comes
  from a tooth crossing the mesh line. The drift and the homecoming fall out of the geometry.
- The missing-teeth idea makes the 11 rests visible as shape, not as notation.
- Pre-rendering each gear once as an RGBA sprite and rotating it per frame keeps a pure Pillow
  pipeline at ~0.6 s per 2× supersampled frame; the whole film renders in about four minutes.
- The loop. Four laps is 1157 whole frames at 60 fps, the beat wheel's paint pattern has 11-fold
  symmetry, so the last frame flows into the first without a visible jump, and the audio tails
  wrap (the track is rendered over two loops and the second is kept).

### What failed or was cut (round 1)

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

### With another day (written after round 1)

- Build the second act: the kick/snare wheel sliding in for bars 42+, the 6/4 turnaround as a
  visible extra half-tooth, then the verse's tempo change as the gears audibly slowing.
- Add a slow-motion insert at each cycle start (a 0.5 s "freeze" with the bracket), then catch up,
  so the slip measurement gets screen time without breaking the loop.
- Render the gears in 3D (Blender) with real involute teeth and shadows at the mesh; the flat
  sprites read as diagram, which is fine for the idea but not as satisfying as metal.
- A proper cover: bass on C1, the two palm-muted guitars as a tick on every sixteenth, ride bell
  on beats, and a real mix.
- Test on a phone and tune the type sizes; everything was judged from stills on a desktop.


## Credits

Gojira, "The Art of Dying", *The Way of All Flesh* (Listenable / Prosthetic, 2008). Pattern facts
cross-checked against the Songsterr community tab, Raphael Pannier's transcription and Yogev Gabay's
explainer, as provided in the brief.

`sound_voices.py` and `sound_dsp.py` are borrowed unchanged, and `soundtrack.py` adapted, from the
repository owner's own Art of Dying study (branch `ccr-03f861e3-540mn1`, folder `art_of_dying/`), with
their permission for this round. The picture, the gears and the timeline are this experiment's.

# Experiment 1: gears

A vertical film for TikTok and Instagram Reels about the rhythm of the intro of **Gojira, "The Art
of Dying"** (*The Way of All Flesh*, 2008). Every instrument of the intro is a gear on one common
drive; each gear's teeth are its hits and its stub teeth are its rests; it slides into mesh when
its part enters and out when it stops. The question answered on screen: *why does the beat drift,
and when does it come back?*

This is an independent study of the song's rhythm. It is not affiliated with Gojira, their
management or their label. **The round-3 and round-4 films are silent: their audio is the original
recording, supplied and muxed by the repository owner, and it is not in this repository.** The
round-1 and round-2 films use synthesised sound only.

Deliverables in this folder (round 4 is current; earlier rounds are kept under `round1/`, `round2/`, `round3/`):

| File | What |
|---|---|
| `art_of_dying_gears_record_silent.mp4` | **round 4**: the picture for the recording, 1080×1920, 60 fps, 4783 frames = 79.714 s, no audio track |
| `stills/*.png`, `stills/contact_sheet.png` | eight round-4 frames: the hook, bar 12 engaging, mid clicks + pulse with the slip bracket, the bar-42 breath, bar 42 landed, lap 13, bar 53, the closing card; and a contact sheet |
| `rhythm.py` | the facts of the intro, the gear table and the record-synced timeline (`BPM`, `FIRST_CLICK_OFFSET_S`) |
| `render.py` | the round-4 frame renderer (Pillow + numpy): rack, gears, camera |
| `build.sh` | renders the frames, the silent MP4 and the stills (about 20 min on 4 CPUs) |
| `round3/` | round 3: the same timeline on the two music-box wheels plus a turnaround wheel |
| `round2/` | the 58-second synthesised teaching film and its code |
| `round1/` | the 19-second first film and its code |

## Round 4: one gear per instrument

### The mechanism

The drive is a **double-sided rack** running across the middle of the screen, moving one tooth
(one pitch) per sixteenth on the record's clock. Each instrument is a gear that either sits on
the rack or hangs from it. A gear's teeth come in two heights: **tall teeth are hits** (they reach
the rack's floor and strike it, with a flash in the instrument's colour) and **stub teeth are
rests** (they still mesh, so the gear is always driven, but they strike nothing). The bass's held
notes are **long teeth** that run on into the rest. The rim gear's tooth 0 is painted white: the
cycle start.

A gear is hidden until its part enters. It slides in from the right, lifted clear of the rack
(0.28 s), then drops into mesh (0.08 s) landing exactly on its first sixteenth, tooth 0 on the
contact line. Its teeth interleave the rack's teeth with no phase fudge (the round-1 to round-3
quarter-sixteenth offset is gone). When its part stops it lifts off the rack and slides out to
the left. While lifted it does not turn. A camera pans and zooms to keep the engaged set inside
the left 60-860 px of the frame, so nothing essential sits in the right-hand button column, and
all readouts live in a strip above the bottom 20 %.

| Gear | Positions | Striking teeth | Side | Engaged (film k) | Record |
|---|---|---|---|---|---|
| RIM CLICKS | 45 | 34 tall (rests at 4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44); tooth 0 white | above | 0 - 540 | bars 1-33 |
| PULSE | 12 | 3 tall amber (every fourth) | below | 180 - 540 | bars 12-33 |
| SNARE | 45 | 11 tall white | below | 540 - 720 | bars 42-52 |
| KICK | 45 | 28 tall red | above | 540 - 720 | bars 42-52 |
| BASS | 45 | 34 tall violet, the 11 group-final ones long (held through the rest) | below | 540 - 720 | bars 42-52 |
| GUITARS | 12 | 12 tall green (every sixteenth; both guitars, one gear) | above | 540 - 720 | bars 42-52 |
| RIDE | 12 | 3 tall (bell, on the beat) + 3 medium (ghost ping, off the beat); a crash on its first tooth | above | 540 - 744 | bars 42-53 |
| BASS 6/4 | 12 | 12 tall, C1 dark violet ×3 / C2 light violet ×3 | below | 720 - 744 | bar 53 |
| KICK 6/4 | 12 | 6 tall: K K K · · · twice | above | 720 - 744 | bar 53 |
| GUITARS 6/4 | 12 | 12 tall, C2 green ×3 / E♭2 light green ×3 | below | 720 - 744 | bar 53 |
| SNARE 6/4 | 12 | 4 tall: · · · S · S twice | above | 720 - 744 | bar 53 |

Why 12 for the 4/4 parts: a gear's period in sixteenths must divide its tooth count, and every
gear lands on tooth 0 at its entry, so a 12-tooth beat gear (three beats per turn) lands at 180,
540 and 720, all multiples of 12. The 45-against-4 story survives directly: the pulse gear makes
3¾ turns per lap of the riff gear, so at every cycle start its amber tooth is one more tooth past
the contact (a bracket on the gear measures it, the SLIP meter counts it), and after four laps the
white tooth and an amber tooth meet on the contact line again: HOME, at 38.571 s and at 57.857 s.
The turnaround cell is six sixteenths, so 12-tooth gears go round twice in the 6/4 bar.

### The timeline

Unchanged from round 3: 140.000 bpm, first click at 0.000 s (both parameters), 744 sixteenths,
4783 frames at 60 fps, the cut at k = 540. Film k is record sixteenth k for k < 540 and k + 128 after.

| Film k | Film time | Record | On screen |
|---|---|---|---|
| 0 | 0.000 s | bar 1 | the rim gear alone on the rack, clicking; hook caption; group counter counting 5s and 3s |
| 178-180 | 19.07 s | bar 12 | the pulse gear slides in under the rim gear and drops into mesh at 19.286 s: **big moment 1**, amber flash, THE PULSE ENGAGES; camera eases out a little; LAP and SLIP fade in |
| 225 ... | laps 6-8 | bars 13-22 | at each cycle start a bracket on the pulse gear shows the amber tooth 1, 2, 3 teeth past the contact |
| 360 | 38.571 s | bar 23 | **HOME**: gold flash where the two gears meet the rack |
| 536-540 | 57.4-57.9 s | bars 33 → 42 (the cut) | the camera breathes out; five gears slide in from the right and drop into mesh together at 57.857 s: **big moment 2**, HOME again, a crash; the rim and pulse gears lift and roll away left; camera settles on the band |
| 540-720 | laps 13-16 | bars 42-52 | kick, snare, bass (45 each), guitars and ride (12 each); the slip bracket moves to the ride gear |
| 716-720 | 76.7-77.1 s | bar 53 | four 12-tooth turnaround gears slide in to the right of the ride gear; kick, snare, bass and guitars lift and roll away; camera pans to the turnaround train; counter switches to the 6-dot cell |
| 744 | 79.714 s | end | the lights dim with the owner's 1.2 s audio fade into the closing card |

Re-render on a measured tempo or offset exactly as in round 3:

```
cd experiments/art_of_dying/gears
AOD_BPM=139.85 AOD_OFFSET=0.12 ./build.sh    # ~20 min: frames, silent MP4 (H.264 CRF 28, no audio), stills
```

### What changed from round 3

- Seven instruments became seven gears (eleven with the turnaround set) instead of pin tracks on
  two wheels; nothing dormant is on screen, so the hook is one gear and the rack.
- The drive is a rack, not the wheels meshing each other, so gears of any size mesh honestly and a
  gear can enter or leave without touching another.
- Tall / stub teeth replaced "missing teeth" and painted pins: every gear is always driven, and the
  rests are visible as short teeth.
- A camera: zoom 0.9 for the single gear, 0.8 for two, a 0.41 breath at bar 42, 0.55 for the band,
  0.95 for the turnaround train.
- The pulse gear is 12 teeth (three amber) rather than 44 (eleven painted): the 45:4 story is now
  literal, and the slip bracket spans 30° per tooth instead of 8°.

### What worked

- Engagement as a physical act: a gear lifting off and rolling away is the record's arrangement
  made visible (the pulse resting from bar 42, the clicks stopping) without a word of caption.
- Landing on the grid: every entry is at a multiple of the gear's tooth count, so each gear lands
  tooth 0 on the contact line; the rack and all gears share one phase and there is no visual fudge.
- Bar 42 as a single gesture: breath out, five gears drop in, two roll off, settle. It is the
  biggest thing in the film and it happens where the record does it.
- The layout survived the phone constraints: big gears above and below the rack, the two small
  ones in the gap, everything inside the left 800 px, readouts in a strip above the UI zone.

### What failed or was cut

- **Three slides per lap are not a conveyor.** Gears travel 1400 world units in 0.28 s: it reads as
  a flick, not as a glide, and a viewer may not see where a gear came from. A longer slide would
  have to start before the previous part's last hit, which the music does not allow.
- **The bar-42 swap overlaps on screen for a moment.** The five band gears land to the right of the
  rim and pulse gears while those are still on the rack (their last strikes are at k = 538 and 536,
  and HOME at 540 needs them in place), so for about half a second the camera shows seven gears and
  the big ones are small. It is honest; it is also the busiest half-second of the film.
- **Small type again.** The strip (group counter, lap, slip, the seven-name stack) is 21-24 px mono;
  the gear labels are 22 px. The gear colours and the hub counts carry most of the meaning.
- **The turnaround's crossing rhythm is not drawn.** Kick 6/4, snare 6/4 and the ride all mesh the
  same rack, so the 6-against-4 cross-rhythm is visible only as the ride's bell landing on a
  different cell slot each time; no caption points at it.
- **The rack is plain** and moves fast (one pitch every 107 ms); it is a reliable metronome but not
  beautiful, and at zoom 0.41 its teeth are 17 px.
- **No gear for the choir**, which is a held note and not a rhythm; the first 19 seconds of the
  record have more going on than the picture admits.
- **Not checked against the recording.** As in round 3, the first click and the tempo come from the
  brief; the re-render path exists for a measured tempo, but I could not listen.

## Round 3: the picture for the recording (kept in `round3/`)

`round3/art_of_dying_gears_record_silent_round3.mp4`, with its `rhythm.py`, `render.py`, `build.sh` and stills.

### What changed

The owner decided the film's audio will be Gojira's recording. A mixed master cannot be layered one
part at a time, so the round-2 teaching stack is gone: **parts now engage exactly when the record
plays them**, on the record's clock, and the film no longer loops. The gears stay: the rim teeth and
the pulse followers carry laps 1-12, then at bar 42 the inner pin tracks engage all at once while the
rim and pulse followers lift. A third, small wheel was added for the 6/4 turnaround.

### The timeline

Film seconds at 140.000 bpm with the first click at 0.000 s. One sixteenth = 15/140 s = 107.143 ms,
one lap of the 45-cycle = 4.8214 s. `k` is the film's sixteenth index; the beat is on `k ≡ 0 (mod 4)`
throughout. Film sixteenth `k` is record sixteenth `k` for `k < 540` and record sixteenth `k + 128`
after the cut.

| Film k | Film time | Record | On screen |
|---|---|---|---|
| 0 | 0.000 s | bar 1, the first rim click | hook: WHY DOES THIS BEAT DRIFT? The riff wheel clicks; the group counter counts 5s and 3s; the beat wheel and the small wheel turn dormant |
| 0-179 | laps 1-4 | bars 1-11, clicks over the choir | captions introduce the wheel, the rests, and the waiting beat wheel |
| 180 | 19.286 s | bar 12: the tribal pulse joins | **big moment 1**: the beat wheel's followers drop, amber flash, THE PULSE ENGAGES, the 45:44 line; LAP and SLIP readouts fade in |
| 180-359 | laps 5-8 | bars 12-22 | the slip grows one tooth a lap: bracket on the beat wheel, SLIP meter 1, 2, 3 |
| 360 | 38.571 s | bar 23 | **HOME**: gold flash, the white and amber teeth meet; then the slip grows again (laps 9-12) |
| 540 | 57.857 s | **the cut**: record bars 34-41 (pulse alone, 13.714 s, cycle suspended) removed; film 57.857 s = record bar 42 (1:11.57) | **big moment 2**: HOME again and the band: kick + snare, bass (riff wheel tracks), ride with a crash, guitars (beat wheel tracks) engage together on the first tooth; rim and pulse followers lift; a small note says what was cut |
| 540-719 | laps 13-16 | bars 42-52 | the kit rides the riff wheel, the ride rides the beat wheel; the slip meter keeps counting against the ride bell |
| 720 | 77.143 s | bar 53: the 6/4 turnaround, 24 sixteenths | the riff wheel's followers lift; the 12-tooth turnaround wheel engages: K K K S · S four times on its pin track, bass C1 ×3 / C2 ×3 as dark and light violet teeth; the counter switches to a 6-dot cell |
| 744 | 79.714 s | end (the audio fades over the last 1.2 s) | the lights go down with the fade and a closing card comes up: Gojira, The Art of Dying, The Way of All Flesh 2008, the one-line answer, the disclaimer |

Total: 744 sixteenths = 79.7142857 s = round(79.714 × 60) = **4783 frames** at 60 fps.

Not shown: the guitars' C2 / E♭2 alternation in the turnaround (their pins live on the beat wheel,
which has no pitch), and the 6/4 turnaround at bar 65 and the pulse's return at bar 66, which are
after the film ends.

### Re-rendering with a measured tempo or offset

Every timing is a function of one sixteenth counter and two parameters at the top of `rhythm.py`:

```
BPM = 140.000                 # the recording's tempo
FIRST_CLICK_OFFSET_S = 0.0    # film seconds before the first rim click
```

Override them from the environment; the frame count follows:

```
cd experiments/art_of_dying/gears
AOD_BPM=139.85 AOD_OFFSET=0.12 python3 rhythm.py     # prints the timeline and the frame count
AOD_BPM=139.85 AOD_OFFSET=0.12 ./build.sh            # ~15 min: frames, silent MP4, stills
```

With an offset, the wheels turn but nothing is engaged before the first click (k < 0). The silent
MP4 is H.264 CRF 28, yuv420p, no audio track; the owner aligns frame 0 (or the offset) with the
record's first click and muxes.

### What worked

- The record's own arrangement is more dramatic than the teaching stack: eight laps of clicks and
  pulse give the slip room to be felt, and bar 42 arrives as HOME and the band in one blow.
- The music-box tracks survived the change without a redesign: engagement is just a follower
  dropping, so "the pulse rests in bars 42-65" is a lifted follower and nothing else.
- A 12-tooth wheel is enough for the turnaround: the 6-sixteenth cell goes round twice, so the small
  wheel reaches slot 0 exactly at k = 720 (720 is a multiple of 12) with no slide-in.
- One sixteenth counter and two parameters: a re-render on a measured tempo is one command.

### What failed or was cut

- **The cut is invisible as music.** The picture simply continues at 57.857 s; only a small line in
  the left column says that eight bars were removed. Anyone who knows the record will feel the jump
  in the audio and may not read the note.
- **Dormant wheels from frame 0.** The beat wheel and the turnaround wheel are visible, dim, before
  they matter. They promise what is coming but they are two more things in the hook.
- **The turnaround wheel sits in the right-hand column** (upper right of the riff wheel), where the
  app's buttons are. Its name chip is transient and the stack row duplicates it, but the wheel itself
  is partly under UI on some phones.
- **No pitch on the beat wheel**, so the guitars' turnaround figure is not shown.
- **The slip meter after bar 42** counts the ride bell against the cycle start, which is correct but
  no caption says so; the readout is there for continuity more than for teaching.
- **Not auditioned against the recording.** The timeline is built from the brief's bar map; if the
  record's tempo is not exactly 140.000 the owner's measurement and a re-render will fix the drift,
  but I could not check the first click or the bar-42 downbeat myself.

## Round 2: the synthesised teaching film (kept in `round2/`)

The 58-second looping film `round2/art_of_dying_gears_synth.mp4`, with its own `rhythm.py`, `render.py`, `soundtrack.py`, borrowed sound modules and `build.sh` (run it from inside `round2/`).

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
WAV so the AAC stays under −1 dBTP. Measured on the final MP4 with `ffmpeg -af ebur128=peak=true`:

| Measure | Value |
|---|---|
| integrated loudness | −14.0 LUFS |
| true peak | −1.3 dBTP (the WAV is at −2.8; AAC at 256 kb/s overshot a −2.0 ceiling by 1.2 dB on the first try) |
| loop seam | sample jump 0.003 full scale against a 99.9th-percentile jump of 0.157; the seam window peaks at −2.8 dBTP |
| file | 17.6 MB, H.264 CRF 27, yuv420p, 3471 frames at 60 fps |

Picture and sound share one clock: 12 laps = 540 sixteenths = 3471 whole frames at 60 fps, so one
sixteenth is 107.130 ms (139.98 bpm, 0.012 % slower than the record) and the loop is exact in both.

### How to run

```
cd experiments/art_of_dying/gears/round2
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
  and I could not listen either; the limiter works hard on the pulse (up to 9.4 dB on transients, more than 3 dB on 16 % of samples) to reach −14 LUFS under a −2.8 dBTP ceiling.
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

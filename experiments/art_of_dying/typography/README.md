# Experiment 3 · Typography — *Count this riff*

A 34-second vertical film (1080×1920, 60 fps) that counts the intro of
**Gojira — "The Art of Dying"** (*The Way of All Flesh*, 2008) the way a
drummer would, using nothing but type.

> This is an independent study of a piece of music, made for learning and
> discussion. It is not affiliated with, endorsed by, or connected to Gojira,
> Roadrunner Records or Listenable Records. No audio from the record is used:
> every sound here is synthesised.

- Film: [`art_of_dying_typography.mp4`](art_of_dying_typography.mp4)
- Stills: [`stills/`](stills/) — `01_hook` (0.6 s), `02_one_lap` (4.7 s), `03_beat_joins` (7.2 s),
  `04_the_slip` (14.6 s), `05_the_kit` (25.0 s), `06_outro` (31.5 s)
- Code: `rhythm.py`, `timeline.py`, `render.py`, `audio.py`, `voices.py`, `build.sh`

## The idea

The intro runs on a 45-sixteenth cycle at 140 bpm, grouped
**5 5 5 3 3 3 | 5 5 5 3 3**. A 5 is "x x x x ." and a 3 is "x x .": the rim
clicks play every sixteenth except the last of each group. The quarter-note
beat (a stacked "tribal" drum hit) falls every 4 sixteenths. Because
45 = 11 × 4 + 1, the beat slips one sixteenth against the cycle on every lap
and lines up with it again after 4 laps (45 beats, 19.3 s).

The film makes those three facts visible with type alone:

1. **The groups are the shape of the text.** Every rim click drops a counted
   digit into a stack of eleven rows. Rows of 5 and rows of 3 give the stack a
   silhouette: 5 5 5 3 3 3 / 5 5 5 3 3. The rest at the end of each group is
   the digit a drummer counts but does not play, so it is drawn thin and dim:
   a shape without weight.
2. **The beat is a second voice.** A huge ghost digit behind the stack counts
   1 2 3 4 on every beat, and the small digit the beat lands on turns orange
   and heavy.
3. **The slip is the drift of the orange.** Lap by lap the orange digits move
   one column, because the beat's 4 does not divide the cycle's 45. After four
   laps the picture is back where it started and the words say so.

Then the kit arrives: the same stack re-spelled as **K K K S . / K K S**
(kick kick kick snare, kick kick snare), the way the full band plays the
cycle from bar 42. The outro restates the sum and the film ends on a lap
boundary so it loops into its own first frame.

## The seed and how I read it

Seed: *kinetic typography. Nothing on screen but type: the counted numbers
and a few words. Weight, size, position and timing do the work of the drums.
Rests are silence or space. A second voice counting 1 2 3 4 that drifts
across the first.*

Reading:

- **Position** carries the structure: the stack is a bar chart made of
  digits, and the column a digit sits in *is* its count.
- **Weight and colour** carry the beat: ExtraBold orange versus Bold grey,
  Light and dim for the rest.
- **Size** carries the hit: every digit lands at 145 % and settles in
  90 ms; beat digits land larger and stay at 118 %.
- **Timing** is literal: each glyph appears exactly on its sixteenth
  (107.14 ms). At 60 fps that is 6.43 frames per sixteenth, so onsets are
  never more than 8 ms off.
- **The second voice** is the ghost digit *and* a spoken voice. I used two
  voices from one TTS model: a low voice says the size of each group on its
  rest ("five… five… five… three, three, three"), which is the count a
  drummer makes silently, and a higher voice counts "one two three four" on
  the beat, drifting across the first.
- **Rests are space**: in the band lap the kick rest is empty; in the rim laps
  the rest is a dim thin digit. I chose the dim digit over pure space because
  without it the rows of 5 read as rows of 4, which breaks the premise.

I added one thing the seed did not ask for: the letters K and S for the kit
lap. They are still type, and they show that the band plays the same shape.

## Timeline (7 laps × 4.821 s = 33.75 s)

| Lap | Time | On screen | Sound |
|---|---|---|---|
| 0 | 0.0–4.8 | "COUNT THIS RIFF"; the stack builds digit by digit; tallies 24, 21 | rim clicks, choir pad, group voice |
| 1 | 4.8–9.6 | ghost 1 2 3 4 joins; beat digits orange on t ≡ 0 (mod 4) | + tribal hit with 32nd-note stick grace, beat voice |
| 2 | 9.6–14.5 | orange on t ≡ 3: "the beat slips one count · 45 = 11 × 4 + 1" | same |
| 3 | 14.5–19.3 | orange on t ≡ 2: "and slips again" | same |
| 4 | 19.3–24.1 | orange on t ≡ 1: "and again" | same |
| 5 | 24.1–28.9 | back on t ≡ 0: "back together after 4 laps"; stack re-spelled K K K S / K K S | kick, snare, ride + bell, bass on C1 holding through the rests, dead-note guitars, beat voice |
| 6 | 28.9–33.75 | the stack once more; "5 5 5 3 3 3 · 5 5 5 3 3 = 45"; credit | rim clicks, choir pad, group voice; loops into lap 0 |

The record builds more slowly (11 bars of rim clicks, 22 with the tribal hit,
8 of the hit alone, then the band at bar 42). I compressed it to one lap of
rim clicks and four with the beat so the full slip-and-return fits in 20 s.
The film is at exactly 140 bpm with the cycle starting on a beat, so a
section of the official track can be laid under it in the TikTok/Instagram
editor: bar 12 beat 1 of the record aligns with 4.821 s of the film (start of
lap 1), and the band entry at bar 42 does *not* align with the film's band
lap (which starts at 24.1 s); if the record is used, mute the film's audio.

## How to run

```
pip install piper-tts            # only for the counting voices
cd experiments/art_of_dying/typography
./build.sh                       # ≈ 2–3 min on 4 CPUs → art_of_dying_typography.mp4
python3 render.py still 12.9 x.png   # one frame
python3 rhythm.py                # prints the beat positions per lap
```

`voices.py` downloads the piper voice `en_US-lessac-medium` (≈ 60 MB) into
`piper_data/` the first time and writes the five words to `words/`. If piper
is not available, the pre-rendered `words/*.wav` in this folder are used.
Everything else is numpy, Pillow and ffmpeg.

## What worked

- **The stack silhouette.** Rows of 5 and 3 made from the counted digits
  themselves show the 45 at a glance, with sound off, inside the first lap.
  Nobody has to read "5+5+5+3+3+3" to see it.
- **The drift of the orange.** The slip is usually explained with arithmetic.
  Here it is a column of orange digits walking one step per lap, with last
  lap's positions left as a faint orange residue so the step is visible.
- **A model you can assert on.** `rhythm.py` derives the rests, snares and
  kicks from the group list and asserts them against the published pattern;
  it caught my own mistake (I had put the 3-group's snare on the wrong slot).
- **Frame-exact timing.** Rendering from the sixteenth index rather than from
  keyframes meant every glyph, every hit and every word share one clock. A
  check on the finished MP4 finds the first big hit's energy in the
  4.821 s window and the stick grace 40 ms before it, as designed.
- **A bug the stills caught.** My first build anchored the beat grid to the
  film's frame 0, so the beat entered three sixteenths into lap 1 instead of
  on its first "1" and the "back together" lap landed on the wrong phase.
  A contact sheet of 16 frames showed it; the grid now starts with lap 1.
- **The voice on the rest.** Saying "five" on the silent fifth count was the
  one idea that let a spoken voice keep up: one word per group instead of
  one per sixteenth.

## What failed or was cut

- **A voice that counts every sixteenth.** The seed's "1 2 3 4 5" spoken at
  9.3 syllables per second is beyond speech; piper's words time-compressed
  to 107 ms turn into clicks. I cut it before building it in and let the
  picture do that count.
- **I could not listen.** The mix was tuned from RMS numbers, waveform images
  and an EBU R128 reading, not by ear. The drums are plausible numpy drums,
  not a Gojira sound; the "choir" is a detuned pad. Expect the voices to need
  a level change on first listen.
- **The record itself is absent.** Without the track the film cannot prove
  its own claim about the song; it demonstrates the rhythm the transcriptions
  describe.
- **The hook is quiet.** Rim clicks alone are a thin first two seconds for a
  feed. I boosted them and front-loaded the title, but a sound-on viewer
  gets the big hit only at 4.8 s.
- **The ghost digit hides behind the stack.** At 1150 px it is legible as a
  shape but its identity (which of 1 2 3 4) is partly covered. It works as a
  pulse more than as a readable count.

## With another day

- Listen, then remix: a real reference ear, sidechain the pad under the hit,
  and re-cut the voices.
- Give lap 0 a sound-on hook: start on a single tribal hit with the title,
  then let the rim clicks count.
- Animate the slip itself: between laps, slide the orange column one step
  rather than letting it reappear, so the "+1" is motion rather than
  comparison.
- Try the bars-34–41 moment (the hit alone, cycle suspended) as a typographic
  blackout: only the ghost digits, counting in empty space.
- A 15-second cut for the feed: lap 0 and two beat laps, with the return
  implied by the caption.

## Credits

Music analysed: Gojira, "The Art of Dying", *The Way of All Flesh* (Listenable /
Prosthetic, 2008), composed by Joe Duplantier, Mario Duplantier, Christian
Andreu and Jean-Michel Labadie. Rhythm facts cross-checked against Songsterr's
community tab, Raphael Pannier's transcription and Yogev Gabay's explainer.
Fonts: Inter Display (words), JetBrains Mono (digits). Voice: piper-tts,
`en_US-lessac-medium`. Everything else: numpy, Pillow, ffmpeg.

# Experiment 4 · notation: *The Art of Dying* intro, engraved and animated

A 45-second vertical film (1080 × 1920, 30 fps) that typesets the drum intro of
Gojira's **"The Art of Dying"** (*The Way of All Flesh*, 2008; drums Mario Duplantier)
as real drum notation, then plays through it with a cursor and shows a drummer how to
count it. This is an independent study made for a creative experiment. It is not
affiliated with Gojira, their label or their publishers, and nothing from the record is
sampled: every sound in the film is synthesised in numpy.

| | |
|---|---|
| Film | [`out/art_of_dying_notation.mp4`](out/art_of_dying_notation.mp4) · 44.9 s · 1346 frames · 30 fps · yuv420p · AAC · about 3 MB |
| Stills | [`stills/`](stills/) (six frames from the film + the LilyPond reference page) |
| Reference chart | [`lilypond/art_of_dying_intro.pdf`](lilypond/art_of_dying_intro.pdf) / `.png` (engraved with LilyPond, generated from the same data as the film) |
| Code | `pattern.py` (the rhythm as data), `engrave.py` (the engraver), `synth.py` (the sounds), `film.py` (timeline, frames, audio, mux), `make_lily.py` (LilyPond reference) |

## The idea

Everyone argues about whether this riff is "in 21/16" or "in 45/16". A drummer does not
need the argument; they need a chart they can count. So the film puts one bar of 45/16 on
screen, beamed the way the pattern is actually felt, **5 + 5 + 5 + 3 + 3 + 3 | 5 + 5 + 5 + 3 + 3**,
colours the fives amber and the threes teal, writes the count under every sixteenth, and
then lets the quarter-note beat walk across the chart so you can *see* why the tab needs
a 5/4 bar every eleventh bar.

Structure of the film:

1. **0.0–1.7 s · hook.** "THIS DRUM INTRO IS IN 45/16." Four stick clicks count in while the
   staff wipes on and the score draws itself group by group, beams growing left to right.
2. **1.7–11.4 s · count it.** Two laps of side stick alone over a held choir note. The cursor
   runs, each hit flashes, the counts 1 2 3 4 (5) / 1 2 (3) appear as they are played, and the
   active group's beam lights up.
3. **11.4–30.6 s · the beat slips.** Four laps with the tribal hit (bass drum + floor tom +
   surdo + wood block + castanets, with its stick grace note) on every quarter note. The hit is
   engraved as a stems-down quarter-note voice under the rim clicks, and a row of red dots
   above the staff marks the beat. Each lap the dots and the quarter notes step one slot to the
   left, because 45 = 11 × 4 + 1. A row of eleven chips (4/4 × 10, then 5/4) fills beat by beat
   to show how the tab writes the same 45 beats.
4. **30.6–40.3 s · bar 42, the whole band.** The chart switches to a two-voice kit groove:
   hands up (ride eighths, bell on the beat, snare, a crash on beat 1), feet down (kick sixteenths
   with stemlets over the rests, R L R under the first group). The second lap is re-engraved
   with the beat one slot earlier, so the bell and ride move while kick and snare stay put.
   Bass (C1, following the rim pattern) and two palm-muted guitars (C2) join the audio.
5. **40.3–44.9 s · outro.** "SCREENSHOT THIS." The kit chart stays up while the band fades.

## The seed and how I read it

The seed was *drum notation for real drummers*: typeset the intro properly, then animate it.
I took "properly" seriously in two ways:

- **Two engravings from one source of truth.** `pattern.py` holds the rhythm as data (the
  45-slot cycle, the groups, the rests, the snare/kick slots, the beat drift). `make_lily.py`
  turns it into a LilyPond score (45/16 with `beatStructure = 5,5,5,3,3,3,5,5,5,3,3`, stemlets
  over the rests, the two-voice kit groove, and the tab's 4/4 × 10 + 5/4 view). `engrave.py` is my
  own engraver on Pillow that lays out the same data for the film. The LilyPond page is the
  check: if my beams, rests or noteheads disagree with it, mine are wrong.
- **Animatable geometry.** I wrote the film's engraver myself instead of animating LilyPond's SVG
  because every notehead, stem and beam is then a coordinate I control: beams can grow, groups
  can light up, the cursor knows where slot 23 is, and the beat voice can be re-engraved per lap.
  That decision is the whole experiment; see "What failed" for its cost.

Notation decisions a drummer might query, and why:

- **Beamed 5+5+5+3+3+3+5+5+5+3+3 with stemlets over the rests**, rather than ending each beam at
  the rest. Stemlets keep the group visibly a unit (Gould's *Behind Bars* recommends them for
  rhythmic notation), which is the point of the chart.
- **Side stick as a cross on the snare space.** Standard.
- **The tribal hit as bass drum + floor tom, stems down, quarter notes**, with a slashed cross
  grace note before it for the stick click. The surdo, wood block and castanets are in the sound
  but not on the staff; a drum chart shows what the drummer plays.
- **The kit groove in two voices**, hands up / feet down, eighths with a secondary beam only
  where two notes are a sixteenth apart (how an engraver would beam it, and what LilyPond does
  with the same input), and no rests in the hands voice because the feet voice carries the grid.
- **45/16 kept as one bar with repeat signs**, not 24/16 + 21/16, because the beat drift only
  makes sense against the full 45.

## How to run it

```
pip install lilypond            # only for the reference page; the film does not need it
cd experiments/art_of_dying/notation
python3 film.py                 # ~4 min on 4 cores: frames -> out/frames, audio, mux -> out/art_of_dying_notation.mp4
python3 film.py --still 24.6 x.png   # one frame at 24.6 s
python3 film.py --mux           # re-mux existing frames after an audio change
python3 make_lily.py            # regenerate lilypond/art_of_dying_intro.{ly,pdf,png}
```

Only numpy, Pillow and ffmpeg are needed for the film. Set `AOD_FRAMES=/some/dir` to keep
the 1346 frames (about 1.3 GB as PNG) out of the repo folder.

**Adding the real track in the app.** The film's audio is a synthesised mock-up, deliberately.
If the owner prefers the record, the alignment is: the riff starts at **1.714 s** of the film
(after a four-beat count-in at 140 bpm); one lap of the cycle is 4.821 s; the tribal hit joins
at lap 3 (11.36 s), which corresponds to **bar 12** of the record, and the kit section at
30.64 s corresponds to **bar 42** (about 1:10 on the record). Lining up the record's bar 12
with 11.36 s keeps laps 3–8 in sync; the first two laps then show bars 10–11 instead of 1–2,
which look identical.

## What worked

- **The engraver.** A few hundred lines of Pillow produce a chart that holds up next to the
  LilyPond page: the beams, stemlets, two voices, ledger lines, grace notes, repeat barlines and
  the stacked time signature all read as notation. The prototype still took 15 minutes; the
  two-voice kit chart came out right at the first render.
- **One data file, two engravings.** Generating the LilyPond score from `pattern.py` caught a
  real bug: my first snare rule put the snare on the penultimate slot of *every* group, which is
  wrong for the threes ("K K S" has no rest), and an assertion against the verified slot list
  failed immediately.
- **The beat drift is visible.** Red dots and quarter notes stepping one slot left per lap, with
  the 4/4 chips filling underneath, makes the "why is there a 5/4 bar" question answer itself
  without a word of narration.
- **Colour-coding fives and threes** is the thing a drummer would screenshot: amber for 5,
  teal for 3, counts under every slot.
- **Pipeline speed.** 0.18 s per frame at 2× supersampling, four processes, about four minutes
  for the whole film, so I could re-render the final film twice after seeing problems.

## What failed or was cut

- **I never heard the film.** The container has no audio output, so the mix was balanced by
  numbers alone (per-section RMS, EBU R128 loudness, two-pass `loudnorm` aimed at −14 LUFS; the peak limit stops it at about −19.5 LUFS integrated, −1.3 dBFS peak). The
  rim click, the ride and the guitar chug are plausible by construction but I cannot promise
  they sound good; the 32.7 Hz bass will be inaudible on a phone speaker, which is honest to
  the record but may read as "nothing there".
- **The caption about the drift was wrong in the first full render.** I wrote "the beat lands
  one sixteenth *later* in the pattern". In the chart (where the pattern is the fixed frame) the
  beat steps one slot to the *left*, i.e. the pattern runs one sixteenth behind the beat each
  lap. I caught it reviewing the contact sheet and re-rendered. It is the kind of mistake that
  gets you corrected in the comments by every drummer watching.
- **The glyphs are drawn, not from a music font.** I did not install Bravura, so the sixteenth
  rest is a geometric approximation that reads as a "7" at phone size, and the percussion clef is
  two bars. LilyPond's page is prettier; mine is more legible on a dark phone screen, which is
  what the brief needed. A SMuFL font would be the first upgrade.
- **The 6/4 turnaround (bars 53 and 65), the China ride (66–76), the tribal hit returning in 66
  and the 4/4 "cycle suspended" passage (34–41) are not in the film.** 45 seconds was the right
  length for a Reel; those belong in a part two. They are in the README's facts, not the chart.
- **Noteheads are small.** 24 sixteenths across a 1000 px line leaves 36 px per slot, so cross
  noteheads are about 25 px wide. Legible on a phone held normally, but not generous. Breaking
  the cycle into four systems (5+5+5 | 3+3+3 | 5+5+5 | 3+3) would be larger and lose the
  24 + 21 reading; I kept the two systems.
- **The hook is a claim, not a question.** "THIS DRUM INTRO IS IN 45/16" is the strongest
  two-second opener I found, but it hands the "actually it's 21/16" crowd a cue. The film
  addresses that in lap 2 ("both are half right"); whether that is early enough is a test to run.
- **No voice-over.** A counting voice ("one two three four, one two three four five") would help
  sound-on viewers; there is no TTS in the container and I chose not to spend the time
  installing one. The on-screen counts carry it for sound-off viewers, which is most of them.
- **Grace notes are cramped.** With 36 px per slot the slashed grace before the tribal hit sits
  tight against the chord. Correct, but not elegant.

## With another day

1. Install Bravura (SMuFL) and draw rests, clef, flags and the time signature from the font.
2. Record or synthesise a counting voice, and add a one-line "sticking" row (R L R L …).
3. A part two: bars 34–41 (plain 4/4), bar 42's crash, the 6/4 turnaround after every four laps,
   the China in 66–76, the verse at 138 bpm.
4. Playback-tested audio: listen on a phone, fix the ride bell (probably too bright) and the rim
   click (probably too thin).
5. An interactive version of the same engraver in the browser (the layout code is pure
   arithmetic), so a drummer can tap the tempo down to 60 bpm and watch the cursor.

## Sources and credit

Gojira, "The Art of Dying", *The Way of All Flesh* (Listenable / Prosthetic, 2008). Drums by
Mario Duplantier. The transcription follows the Songsterr community tab, Raphael Pannier's
transcription and Yogev Gabay's explainer, as summarised in the brief for this experiment.
Independent study; not affiliated with, endorsed by or connected to the band, their label or
publishers. No audio from the record is used.

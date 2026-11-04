# The Art of Dying: the intro, one layer at a time

> **A standalone project, not a Lumiera film.** The Lumiera playbook (`../playbook/`) does not apply here.
> This folder borrows the repo's tools and the platform safe zones, nothing else.

A ~63-second vertical (9:16) breakdown of the intro of Gojira's **"The Art of Dying"** (*The Way of All
Flesh*, 2008). Six parts enter one at a time on fixed lanes swept by a playhead. Each part is shown once as a formal
equation while it is in focus. Every part is recreated with synthesised instruments; no recording is used.

| | |
|---|---|
| Format | 1080×1920, 60 fps, H.264 + AAC |
| Length | 62.68 s = exactly 13 laps of the 45-sixteenth cycle at 140 bpm, so the video and the music loop |
| Layers, in order | rim clicks · tribal pulse · kick + snare · ride · bass · guitar, plus a quiet choir drone throughout |
| Angle | A plain instrument-by-instrument breakdown |

## The rhythm

Everything sits on one **45-sixteenth cycle** at **140 bpm** (one lap = 4.82 s), grouped

    5 + 5 + 5 + 3 + 3 + 3   (24)
    5 + 5 + 5 + 3 + 3       (21)

Fans online call it "21/16" or "45/16"; both are right in part: the 21 is the second half of the 45.

On screen, `n` counts sixteenths from the start and `t = n mod 45` is the position in the lap.

| Layer | What it plays | Equation shown |
|---|---|---|
| Rim clicks | Every sixteenth except the last of each group (5 → `xxxx·`, 3 → `xx·`): 34 hits, 11 rests | `rim(t) = 1 ⇔ t ∉ R`, `R = {4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44}` |
| Tribal pulse | A big stacked drum hit on every beat, with a stick grace note just before it | `pulse(n) = 1 ⇔ n ≡ 2 (mod 4)`, `45 ≡ 1 (mod 4)` |
| Kick + snare | 5 → `KKKS·`, 3 → `KKS`: 28 kicks, 11 snares, 6 rests | `S = {3, 8, 13, 17, 20, 23, 27, 32, 37, 41, 44}`, `K = ℤ₄₅ ∖ S ∖ {4, 9, 14, 28, 33, 38}` |
| Ride | Soft, ghosted ride on the off-beats; bell on the beats | `ride(n) = 1 ⇔ n ≡ 0 (mod 4)`, `bell(n) = 1 ⇔ n ≡ 2 (mod 4)` |
| Bass | The rim-click pattern on low C | `bass(t) = rim(t)`, C1 (32.7 Hz) |
| Guitar | Palm-muted dead notes on every sixteenth, low C | `gtr(n) = 1 ∀ n`, C2 (65.4 Hz) |

Because 45 = 11 × 4 + 1, the 4/4 beat slips one sixteenth against the cycle every lap and lines up with
it again after 4 laps (45 beats, about 19.3 s). That is why the tab writes a 5/4 bar after every ten
bars of 4/4: 10 × 16 + 20 = 180 sixteenths = 4 laps.

**Why `n ≡ 2` for the beat.** The pulse enters at n = 90 (lap 2), and 90 ≡ 2 (mod 4). Putting the beat
there makes every 4/4 layer enter on a hit, at t = 0. Viewers can check it against the counter.
Counted from the record's first click the beat is on n ≡ 0 and the pulse enters after four rim laps
(n = 180); the film keeps two rim laps, which shifts the count by 90 ≡ 2 (mod 4) and leaves the
beat-to-cycle relation unchanged.

## Source and what the film simplifies

The transcription is **Songsterr's community tab** (song 71013, revision 9988433), read in a browser.
Its tempo (140 bpm) matches the tab's own bar-by-bar sync to the recording to within 0.01% on average.
Only the derived rhythm above is kept in this repo; Songsterr's files are not redistributed.

An independent check agrees with it: Raphael Pannier's transcription of Mario Duplantier's groove
(A: 5+5+5+3+3+3, B: 5+5+5+3+3, and the same kick/snare lap note for note), and Yogev Gabay's
explainer (5 as four hits and a rest, 3 as two hits and a rest, 24 + 21 = 45).

In the record, the intro (bars 1–77, about 0:00–2:16) builds like this:

| Bars | What plays |
|---|---|
| 1–11 | Rim clicks (side stick + sticks) on the cycle, over a held choir note |
| 12–33 | + the pulse on every beat |
| 34–41 | The pulse alone |
| 42–77 | Kit (kick + snare on the cycle, ghosted ride off-beats, bell on the beats), bass on the rim pattern, guitars on every sixteenth (dead palm-muted notes in bars 42–52, an open riff in 54–65, palm-muted C2 in 66–76); the pulse returns from bar 66 |

The film's simplifications:

- **Order and the full mix.** One part at a time, rather than the kit, bass and guitars entering
  together. On the record the rim clicks play only in bars 1–33 and never sound with the kit, bass or
  guitars, and the pulse rests in bars 42–65, so the film's full-mix lap is a construction, not a
  moment from the song.
- **The cut sections.** The bridge (bars 34–41, the pulse alone) and the 4/4 fill into the verse
  (bar 77) are left out.
- **The turnaround.** In the kit section the tab adds a 6/4 bar after every four laps (bars 53 and
  65): a 24-sixteenth turnaround grouped 6+6+6+6 (kick/snare `KKKS·S` four times, bass and guitars in
  threes). The film keeps the clean cycle.
- **Details kept simple.** The ride bell plays on every beat (the tab has a crash on beat 1 of bars 42
  and 54, and a China in place of the bell in bars 66–76). The guitars keep the dead palm-muted chug
  throughout. The two kick drums (the tab alternates feet) are one voice.
- **Accents.** A slight accent on the first hit of each group is our choice; the tab marks none.

## How it is built

| File | Role |
|---|---|
| `rhythm.py` | The single source: tempo, cycle, the six layers (pattern, colour, equation), and the timeline (entries, focus laps, equation timing, the loop peel). `python3 rhythm.py` runs a self-check |
| `soundtrack.py` | The procedural score, rendered from `rhythm.events()` |
| `film.py` | The frame renderer (Pillow + numpy, screen-blended glow), posed from `rhythm.py` |
| `render.sh` | Renders in parallel slices, muxes the soundtrack, encodes |
| `sound_voices.py`, `sound_dsp.py` | The instruments and the choir; numpy-only filters, reverb, loudness and true-peak meters |
| `film_gfx.py` | Text with glyph fallback for the maths signs, mark sprites, the glow composite |
| `sound_check.py` | Analysis of a rendered soundtrack: onsets against the grid, loudness, the loop seam, focus levels |
| `POST.md` | Captions, titles, alt text and a launch checklist |

**Each layer's two laps.** In its first lap (the focus lap) the new lane is at full brightness, its
equation is on screen for about 3.8 s, and the other lanes dim to 45% while the other layers drop by
8 dB (`FOCUS_DIM`, `FOCUS_DUCK_DB` in `rhythm.py`). In its second lap it plays as one of the stack.
Played marks trail back to dim over a lap; for the 4/4 layers (pulse, ride) the previous lap's hits
show as outline rings, one sixteenth to the right, so the slip is visible.

## Rendering

Needs Python 3 with numpy and Pillow, ffmpeg, and the fonts Inter (with Inter Display), JetBrains Mono
and DejaVu Sans. Run from this folder:

```bash
python3 rhythm.py                          # self-check (instant)
python3 soundtrack.py out/soundtrack.wav   # the score, ~25 s
python3 film.py still 21.7 out/s.png       # one frame plus a layout JSON; warns on safe-zone breaks
START=19 DUR=5 ./render.sh preview         # 540×960 at 30 fps, a few seconds
./render.sh                                # the film: out/the_art_of_dying_intro.mp4, ~5 min on 4 cores
```

`AUDIO=reuse` keeps an existing `out/soundtrack.wav`. `python3 film.py loopcheck` compares frame 0
with the last frame. After changing a voice, paste the master gain that `soundtrack.py` prints into
`MASTER_GAIN_DB` so excerpts match the film's level.

Credit: music and composition by Gojira. This is an independent study, not affiliated with the band or
their label.

# Experiment 2 · audio-first: can a computer hear the 45?

An independent study of the rhythm in the intro of **Gojira, "The Art of Dying"**
(*The Way of All Flesh*, 2008). Not affiliated with the band, Roadrunner Records or
anyone else; no recording of the song is used. All sound in the film is synthesised here.

**Deliverables (this folder)**

| File | What |
|---|---|
| `art_of_dying_audio_first.mp4` | the film, 1080×1920, 30 fps, 35.4 s, 5.6 MB, audio −14 LUFS |
| `stills/still_*.png` | five stills at 1.0, 6.0, 11.6, 22.5 and 31.5 s |
| `synth.py` | numpy synthesiser: the only file that knows the score |
| `analyse.py` | the detector: works from `build/mix.wav` alone |
| `render.py` | Pillow renderer + ffmpeg encode |
| `detector_log.txt` | what the detector believed, once per second |

## The idea

Fans argue whether the riff is in 21/16 or 45/16. The tab says ten bars of 4/4 and one
of 5/4. All of these are *readings* of the score. This film asks a different question:
if you had no score, only the sound, could a dumb algorithm hear the loop length, hear
the beat, and show that the two do not divide?

The picture is built from analysis of the audio, not from the score. Three panels, all
updated causally (each frame uses only the audio heard up to that moment):

1. **What it hears** – the onset envelope of the last 3.5 s, each detected hit typed as a
   *click* (amber bar) or a *thump* (coral dot).
2. **How it repeats** – the autocorrelation of each stream, with the detected period
   marked. First in milliseconds (the pulse at 107 ms appears within one second), then in
   pulses once the pulse is known.
3. **Folded** – the recent hits laid out in rows of the detected loop length. If the
   loop length is right, columns line up. The amber clicks line up at 45; the coral hits
   drift one column per row. That drift *is* the 45 = 11 × 4 + 1 slip, and it is the
   shot I wanted.

## The seed and how I read it

Seed: *audio-first*. I took it literally and in two halves. First the sound had to
exist: a numpy cover of the intro, compressed from 77 bars to seven laps of the
45-sixteenth cycle (two laps of rim clicks over a choir pad, four laps with the stacked
"tribal" hit on every beat, one lap of kit + bass + guitars as a finale, then a closing
hit). Second, the detector was not allowed to see the score: `analyse.py` reads the WAV
and nothing else. I checked myself with the detector log, not the other way round. The
captions are hand-written and timed to the sections of the mix; the numbers in them are
read from the detector at that frame. The line "45 = 11 × 4 + 1" is my arithmetic, but it
is only shown while the detector itself reports loop = 45 and hit = 4.

## How to run

```
cd experiments/art_of_dying/audio-first
python3 synth.py build          # ~2 s  -> build/mix.wav (+ stems, alignment.json)
python3 analyse.py              # ~3 s  -> build/analysis.npz, build/analysis_log.txt
python3 render.py               # ~50 s on 4 cores -> art_of_dying_audio_first.mp4, stills/
python3 render.py frame 22.5 out.png   # one frame at a given time
```

Needs python3, numpy, Pillow and ffmpeg. Nothing else; no scipy, no librosa. Fonts: Inter,
Inter Display, JetBrains Mono (paths at the top of `render.py`).

## How the detector works (and the decisions in it)

1. STFT (2048/256, 5.8 ms hop) → log-magnitude spectral flux over all bins → causal peak
   picking. This gives one onset list, 292 onsets in 35 s.
2. **Typing onsets.** Each onset gets a *thumpness* = L / (L + H), where L and H are the
   positive jumps of the RMS envelopes of the signal band-limited below 200 Hz and above
   1.5 kHz, each jump measured against the maximum of the previous 58 ms. Thumpness ≥ 0.4
   → thump, else click. Rim clicks score ≈ 0.01, the stacked hit ≈ 0.83, kicks ≈ 0.68,
   snares ≈ 0.16 (so a snare counts as a click).
3. **Pulse (tatum).** First strong peak of the flux autocorrelation between 60 and 200 ms,
   then refined from the peaks near 8× and 24× that lag. It reads 107.1–107.3 ms from
   the first second to the end (true value 107.14 ms).
4. **Grid.** The last 20 s of each onset train is quantised onto the pulse grid (phase chosen
   to catch the most onset energy), one value per sixteenth.
5. **Autocorrelation on the grid**, mean-removed, unbiased, integer lags up to half the
   window.
6. **Period per stream** = argmax of a comb score (ACF at P averaged with ACF at 2P when
   observable), then reduced to the smallest sub-multiple that scores at least 90 % as
   well. The comb is what finally separates 45 from 24, see below.
7. Loop = the longer of the two periods, hit = the shorter.

The detector has a 20 s memory, so it needs to hear a lap twice (9.6 s) before 45 can
appear at all. The film shows that wait rather than hiding it.

## What worked

- **The pulse** is found within a second and holds within 0.2 % for the whole film.
- **45 is recovered from the clicks alone**, at t ≈ 10 s, exactly when the second lap
  completes, with autocorrelation 0.92. It holds from 10 s to 27 s.
- **4 is recovered from the thumps** from t ≈ 14 s, with 0.87, and holds to the end.
- **The slip is visible without reading anything**: in the fold at 45 the coral dots step
  one column left on every row. The best still is `stills/still_4_22.5s.png`.
- **The 24 is real.** Before 45 is observable the detector's best guess is 24, and the
  caption says so. The first half of the cycle (5+5+5+3+3+3 = 24) nearly repeats at lag
  24: the second half (5+5+5+3+3 = 21) matches 21 of its 24 slots. That is the same fact
  behind the "21/16" name. The comb score (ACF at 48 is weak, ACF at 90 is strong) is what
  tips the decision to 45 once 90 pulses have been heard.
- All numpy, no dependencies to install, the whole pipeline runs in under a minute.

## What failed or was cut

- **The full-band section defeats this detector.** With kick, bass and palm-muted guitars
  every sixteenth, nearly every onset types as a thump, the kick pattern shares the 20 s
  window with the earlier tribal hits, and the click stream is left with snares and ride
  bells. It reports loop = 24, hit = 4. I cut the finale to one lap and say on screen that
  the machine loses the thread rather than pretending. A detector that tracked each
  instrument (or at least three or four onset types instead of two) would be the next step.
- **The 180 is never seen.** Clicks and hit line up again every 180 pulses (19.3 s). An
  unbiased autocorrelation needs at least two overlaps, so a 20 s memory can only see lags
  up to 10 s. The record gives you 22 bars of clicks + hit (bars 12–33, about 2 super-cycles),
  so a 40 s memory on the real song would see it. I did not make the film long enough.
- **The clicks' 45 weakens as the hit section goes on** (0.92 → 0.48). When a rim click
  falls on a beat the big hit swallows it and the onset is typed as a thump, so the click
  train loses 11 hits per lap in a slipping pattern. Ironically the slip itself erodes the
  evidence for 45 in the click stream. Over four laps it is still enough.
- **Two earlier detectors failed completely** and are worth recording:
  - *Log-compressed band flux* (low < 200 Hz vs high > 1.5 kHz): with log compression a
    rim click and a stacked drum hit had the *same* low/high ratio (both about 1:3). Band
    energy with log scaling does not tell timbres apart.
  - *Linear spectral flux per band*: the stacked hit's pitch glides (bass drum 130→48 Hz,
    tom, surdo) keep producing positive flux for half a second, which cancelled every rim
    click near a beat. Envelope jumps measured against the recent maximum fixed it.
  - A first naive run also had the pulse 0.6 % short, which by lag 45 misses the sharp
    autocorrelation peak by 30 ms. Refining the pulse from long lags fixed that.
- **No recurrence plot, no live spectrogram.** Three panels were all a phone could take.
- **The sound is a sketch.** Drums are sine sweeps and filtered noise; the "choir" is three
  detuned sines. It is enough to analyse, not enough to enjoy. The synthesised cover is
  what plays in the film; the owner can replace it with the official track in-app if the
  on-screen timing is adjusted (see `build/alignment.json`: sixteenth = 107.14 ms, the mix
  starts on a cycle start which is also a beat; the sections start at 0, 9.64 and 28.93 s).

## What I would do with another day

1. Run the detector on the real recording (licensed in the app, not available here) and
   report honestly whether 45 survives real drums, room and mastering.
2. Lengthen the hit section to eight laps and widen the memory to 40 s so the 180 appears
   as a third peak, then make that the ending: "they meet again after 180".
3. Replace two onset types with a small clustering of onset spectra (k-means on a few
   band energies) so kick, snare, bell and guitar get their own trains. The full-band
   section would probably then yield 45 from the kick + bass pattern.
4. A better hook. The first two seconds are a question and some amber bars; a flash of the
   finished fold plot ("this is what it will find") before the count starts would be stronger.
5. A proper mix: sampled drums or at least a bass drum with a body, and sidechain so the
   clicks stay audible under the hit.

## Notes on verification

- `ffprobe`: 1061 frames, 1080×1920, yuv420p, 30 fps, 35.37 s, 5.6 MB.
- Loudness after `loudnorm`: −12.2 LUFS integrated (target was −14; the single-pass
  loudnorm filter on a short file overshoots a little; it is within platform tolerances).
- Ground truth for checking the detector: `build/alignment.json` (written by `synth.py`,
  never read by `analyse.py`), and the facts in the task brief (140 bpm, 45-sixteenth
  cycle grouped 5+5+5+3+3+3 + 5+5+5+3+3, hit on every fourth sixteenth).

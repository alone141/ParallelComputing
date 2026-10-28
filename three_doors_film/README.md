# Three Doors

The second Lumiera film (Pure Patterns pillar). A prize behind one of three doors; you pick one; the host, who
knows where the prize is, opens an empty door that is not yours; switch or stay? The film answers by playing
rounds with two tallies, STAY and SWITCH, until the shares settle, and says one word.

35.0 s, 1080 × 1920, 60 fps, 2100 frames; frame 2099 equals frame 0, so it loops. Built with Canvas 2D in
headless Chromium via Playwright, with its own score, independently of The Doubling's engine and toolkit.
The approved script is `SCRIPT.md`; this file is the treatment as built.

## The film

| t (s) | Frames | What happens |
|---|---|---|
| 0.00 | 0 | Three closed doors, glowing lavender outlines, nothing else |
| 0.20 to 1.70 | 12 to 102 | The hook stamps in, four lines at 56 px: "Three doors, one prize." / "You pick one." / "The host opens an empty one." / "Switch or stay?", then "Guess before it ends." The hook frame (the cover) is 2.00 s, frame 120 |
| 2.50 to 3.00 | 150 to 180 | The hook fades |
| 3.20 to 8.60 | 192 to 516 | The three rules, one after another, held together: "The host knows where the prize is." / "The host always opens an empty door that is not yours." / "Then you may switch to the other closed door." |
| 8.40 | 504 | The scoreboard fades in: STAY 0, ROUND 0, SWITCH 0 |
| 8.60 to 14.60 | 516 to 876 | Rounds 1 to 3, 2.0 s each: your pick lights a door cyan; the host's door slides open, empty, and dims; the two remaining doors open and an amber point of light shows the prize; the tally that would have won ticks; the doors close. They land switch, stay, switch |
| 14.60 to 16.60 | 876 to 996 | Pause: ROUND 3, STAY 1, SWITCH 2, round 3's reveal held, then the doors close |
| 16.60 to 21.60 | 996 to 1296 | Rounds 4 to 30, accelerating from 28 frames a round to 6 |
| 21.60 to 23.60 | 1296 to 1416 | Pause: ROUND 30, STAY 10, SWITCH 20 |
| 23.60 to 28.60 | 1416 to 1716 | The sprint, rounds 31 to 300: the counts climb on a smoothstep while the doors shimmer (panels half open, colours flickering, the prize's light flickering between doors, all scaled by the speed of the count). No round is drawn individually here |
| 28.60 to 29.40 | 1716 to 1764 | Hold: ROUND 300, STAY 100, SWITCH 200 |
| 29.40 | 1764 | "Switch." stamps in at 128 px; the SWITCH tally blooms with it |
| 33.00 to 34.80 | 1980 to 2088 | "Switch." fades, the scoreboard fades, the nebula returns to lavender |
| 34.80 to 34.98 | 2088 to 2099 | Frame 0 again (the layout of every frame from 2088 is identical to frame 0) |

## The numbers

| Figure | Value | On screen |
|---|---|---|
| Rounds at the pauses | 3, 30, 300 | ROUND 3 / 30 / 300 |
| STAY at the pauses | 1, 10, 100 | exact thirds |
| SWITCH at the pauses | 2, 20, 200 | exact two thirds |
| Why | Staying wins only when the first pick was right, 1 time in 3; switching wins whenever it was wrong, 2 times in 3, because the host's rules never move the prize and always leave the other closed door as the alternative | never stated; the tallies carry it |

**The rounds are arranged, not drawn.** `story.py` fixes the outcome of all 300 rounds so the tallies are
exactly 1 : 2 at round 3, 10 : 20 at round 30 and 100 : 200 at round 300 (rounds 1 to 3 are switch, stay,
switch; the rest is a fixed shuffle in blocks with one third stay wins, no run longer than 5). The doors
your pick, the prize and the host open are assigned by a small fixed generator so the picture is the same on
every machine. No line on screen calls the rounds random or a test, and the post package says the sequence is
arranged. Rounds 1 to 30 are each played in full; the sprint shows the counts and a blur, not the rounds.

## The look

- Deep Space background with a lavender nebula (14 %), amber through the hold and the payoff.
- **Doors:** three rounded outlines, 240 × 400 px at x = 280, 540, 800, top at y = 980, 4 px frames with a baked
  halo, a faint lit panel inside each. Opening slides the panel up over 12 frames and shows a darker room; the
  prize is a point of Starlight with an amber halo, drawn beneath the panel so it appears as the panel rises.
- **Colours:** doors lavender (the mechanic); your pick cyan (you); the host's opened door Mist; the prize and
  the SWITCH tally amber; the STAY tally cyan; the ROUND count Starlight. The hook's target phrase
  "Switch or stay?" is lavender and "You" is cyan.
- **Scoreboard:** one row near the top. Labels STAY, ROUND, SWITCH in Inter Medium 26, caps, tracked +33 % of
  the em, Mist, at baseline 400; counts in JetBrains Mono Bold 88 at baseline 520, centred at x = 250, 540, 830.
  A count that ticks punches 12 % and blooms; in the sprint the counts hold a raised glow instead.
- **Type:** hook Inter Display Bold 56 (four lines at baselines 600 to 804, centred), sub-line Inter Regular 34
  Mist at 900; rules Inter Medium 38 Mist Light (rules 2 and 3 wrap to two lines) at baselines 600 to 852;
  payoff Inter Display Bold 128 at baseline 790. Everything inside x 132 to 948 and above y 1536; the widest
  line is the hook's third at 776 px.
- **Motion:** text stamps in (ink in one frame with a bloom that settles over 15 frames) and fades out; doors
  slide; the sprint's flicker is a hash of the frame number, scaled by the count's speed, so it is deterministic.

## The sound

The film's own palette, quiet, from `soundtrack.py` on the same timeline: a soft noise stamp per line; a dry
tick as your pick lights; a 200 Hz knock with a 900 Hz edge as the host's door opens; a short breath of noise at
the reveal; a small bell per tally step (E5 for STAY, G♯5 for SWITCH); a thud as the doors close; in the sprint
one tick per round over a noise texture that follows the count's speed; one low E minor chord with a knock
under it on "Switch." (the loudest moment); a quiet low tone as the scoreboard fades. In rounds 4 to 30 the
knock, breath and thud get quieter as the rounds shorten; the bells keep their level. A short convolution room
on everything but the chord. Silence at both ends, so the score loops with the picture. Master: −15 LUFS
integrated by ffmpeg's ebur128, limiter at −1.5 dBFS; the phone check (300 Hz high-pass, every cue above
−30 dBFS within 50 ms) passes.

## What it keeps and what it breaks (`playbook/film-style.md`)

Part A, the general aesthetics, is met, with these decided exceptions, all by the owner:
- §2 payoff: no reframing line after "Switch.". §2 ending: no closing line; the return to frame 0 is the ending.
- §3: the prize's accent is amber, which is also the ant / warm life accent; a shared accent by decision.
- §4: the payoff "Switch." is a sentence-case lead in a film that also has caps labels (STAY, ROUND, SWITCH).
- §1 length: 35 s, below the 45 to 75 s default, as a short cut whose beats are the three pauses.
- The hook is four lines at 56 px (the second sentence does not fit the band on one line at any legible size).

Part B, The Doubling's toolkit, is not used: no HUD counter, STEP chip, measuring row, YOU pin, progress
meter, warp or outro; its sound palette is not used. The scoreboard is this film's own device.

The loop is measured on the final MP4 as PSNR between the last and first decoded frames, threshold 45 dB.

## Rendering

Needs Node 22 with Playwright 1.63 and its Chromium (`PLAYWRIGHT_BROWSERS_PATH`, default `/opt/pw-browsers`;
this pipeline never runs `playwright install`), Python 3 with numpy, ffmpeg with libx264 and the ebur128
filter, and the fonts Inter, Inter Display and JetBrains Mono at the paths in `page.html`. Fonts are gated by
measurement (JetBrains Mono "0" at 100 px = 60.0 ± 0.1 px; Inter Display Bold 56 px "Switch or stay?" =
401.4 ± 1 px); a mismatch aborts.

```
python3 story.py --table          # the timeline and the arranged sequence
python3 story.py                  # build/story.json
node render.mjs --check           # layout checks over every event frame (±1) and every sixth frame (~20 s)
node render.mjs --stills          # 32 named moments, 360 × 640 downscales, a contact sheet (~10 s)
node render.mjs --verify          # slice-boundary frames identical across two browsers (~30 s)
python3 soundtrack.py             # out/soundtrack.wav and out/soundtrack_report.json (~20 s)
bash render.sh preview            # out/three_doors_preview.mp4, 540 × 960, 30 fps, with sound (~1 min)
bash render.sh final              # out/three_doors_final.mp4, the silent copy and the cover (~5 min)
python3 checks.py                 # the definition-of-done checks on the final MP4
```

`layout.js` is the one layout: `page.html` draws what it returns and `render.mjs --check` tests the same
objects (band, safe zones, text overlaps, text over doors, strings against `story.json`, STAY + SWITCH = ROUND
at every frame, the counts at the pauses, one prize at a time, the loop). Frames leave the page as PNG and are
piped into lossless x264 slices, four in parallel; nothing depends on the clock or on chance.

## Files

| File | Role |
|---|---|
| `SCRIPT.md` | The approved script |
| `story.py` | Numbers, strings, the arranged sequence, the integer-frame events and sound cues; writes `build/story.json` |
| `layout.js` | `layout(frame)` and the checker (UMD; Node or browser) |
| `page.html` | The renderer: doors, text sprites, nebula, vignette |
| `render.mjs` | The Playwright driver: `--still`, `--frames`, `--slice`, `--preview`, `--check`, `--verify`, `--stills`, `--time` |
| `soundtrack.py` | The score and its master, `measure()` and `phone_check()` |
| `render.sh` | `preview` or `final`: story, slices, concat, score, encode, silent copy, cover, checks |
| `checks.py` | Size, rate, frame count, codec, loudness, true peak, loop PSNR, phone check, the answer search over `POST.md`, the silent copy and the cover |
| `POST.md` | The post package |

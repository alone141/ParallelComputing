# The Art of Dying: the intro, one layer at a time

> **A standalone project, not a Lumiera film.** The Lumiera playbook (`../playbook/`) does not apply here.
> This folder borrows the repo's tools and the platform safe zones, nothing else.

A ~63-second vertical (9:16) breakdown of the intro of Gojira's **"The Art of Dying"** (*The Way of All
Flesh*, 2008). Six parts enter one at a time on scrolling lanes. Each part is shown once as a formal
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
| Kick + snare | 5 → `KKKS·`, 3 → `KKS`: 28 kicks, 11 snares, 6 rests | `S = {3, 8, 13, 17, 20, 23, 27, 32, 37, 41, 44}`, `K = ℤ₄₅ ∖ (S ∪ {4, 9, 14, 28, 33, 38})` |
| Ride | Soft, ghosted ride on the off-beats; bell on the beats | `ride(n) = 1 ⇔ n ≡ 0 (mod 4)`, `bell(n) = 1 ⇔ n ≡ 2 (mod 4)` |
| Bass | The rim-click pattern on low C | `bass(t) = rim(t)`, C1 (32.7 Hz) |
| Guitar | Palm-muted dead notes on every sixteenth, low C | `gtr(n) = 1 ∀ n`, C2 (65.4 Hz) |

Because 45 = 11 × 4 + 1, the 4/4 beat slips one sixteenth against the cycle every lap and lines up with
it again after 4 laps (45 beats, about 19.3 s). That is why the tab writes a 5/4 bar after every ten
bars of 4/4: 10 × 16 + 20 = 180 sixteenths = 4 laps.

**Why `n ≡ 2` for the beat.** The pulse enters at n = 90 (lap 2), and 90 ≡ 2 (mod 4). Putting the beat
there makes every 4/4 layer enter on a hit, at t = 0. Viewers can check it against the counter.

## Source and what the film simplifies

The transcription is **Songsterr's community tab** (song 71013, revision 9988433), read in a browser.
Its tempo (140 bpm) matches the tab's own bar-by-bar sync to the recording to within 0.01% on average.
Only the derived rhythm above is kept in this repo; Songsterr's files are not redistributed.

In the record, the intro (bars 1–77, about 0:00–2:16) builds like this:

| Bars | What plays |
|---|---|
| 1–11 | Rim clicks (side stick + sticks) on the cycle, over a held choir note |
| 12–33 | + the pulse on every beat |
| 34–41 | The pulse alone |
| 42–77 | Kit (kick + snare on the cycle, ghosted ride off-beats, bell on the beats), bass on the rim pattern, both guitars palm-muted; the pulse returns from bar 66 |

The film's simplifications:

- **Order.** One part at a time, rather than the kit, bass and guitars entering together.
- **The turnaround.** In bars 42–53 the tab adds a 6/4 bar after four laps (a fill). The film keeps
  the clean cycle.
- **Accents.** A slight accent on the first hit of each group is our choice; the tab marks none.

## How it is built

| File | Role |
|---|---|
| `rhythm.py` | The single source: tempo, cycle, the six layers (pattern, colour, equation), and the timeline (entries, focus laps, equation timing, the loop peel). `python3 rhythm.py` runs a self-check |
| `soundtrack.py` | The procedural score, rendered from `rhythm.events()` |
| `film.py` | The frame renderer (Pillow + numpy, screen-blended glow), posed from `rhythm.py` |
| `render.sh` | Renders in parallel slices, muxes the soundtrack, encodes |
| `POST.md` | Captions, titles, alt text and a launch checklist |

Credit: music and composition by Gojira. This is an independent study, not affiliated with the band or
their label.

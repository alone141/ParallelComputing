# The Doubling

> Style rules for all Lumiera films are in [`../playbook/film-style.md`](../playbook/film-style.md). This film is their reference implementation.

A ~68-second vertical (9:16) film for TikTok and Instagram Reels, made with
[Manim](https://www.manim.community/). It starts at the Planck length and keeps
doubling. After **116** steps you reach a human, after **185** the Milky Way,
and after **205** the whole observable universe. Every stop shows its step
number and acts out a relative example, such as "425 ants tall" or "109 Earths across".

| | |
|---|---|
| Format | 1080×1920, 60 fps, H.264 + AAC (also a silent version) |
| Length | 68.0 s; the last frame matches the first, so the video loops cleanly |
| Hook | "How many times must you double this to reach a galaxy? Guess before it ends." |
| Payoff | "You're closer to a galaxy than to the Planck length" (116 steps up, 69 to go) |

## Structure

1. **Hook (0–2.75 s).** The question appears over a single glowing dot.
2. **Counted steps (2.75–5.7 s).** The first 8 doublings are shown one at a
   time. The dot splits 1 → 2 → 4 → … → 256, and each step gets a "×2" pop and a pluck.
3. **The great desert (5.7–8.2 s).** A fast sprint through steps 9–66. The desert
   label is shown only while the scale is still below what any experiment has
   resolved: colliders reach about 10⁻²⁰ m, roughly step 50.
4. **13 stops, about 3.2 s each.** Each stop has:
   - a **STEP N** chip;
   - a punch of the HUD counter;
   - a **measuring row** acting out the comparison. Copies of the previous thing
     line up across the new one while a live counter climbs, e.g. "0 → 425 ants".
   The YOU pin follows your position from the human onward: on the summit,
   on Earth, beside the Sun, in the galaxy's spiral arm, and at the centre of the
   observable universe.
5. **Outro (63.8–68 s).** The counter flies to the centre as "205", then the
   image glides back to the first frame so the video loops.

## Rendering

```bash
pip install manim            # needs Cairo/Pango dev headers (e.g. libpango1.0-dev)
./render.sh preview          # 540x960 @ 30 fps, a few minutes
./render.sh                  # 1080x1920 @ 60 fps, the upload version
```

The output goes to `out/the_doubling_final.mp4` (with sound) and
`out/the_doubling_final_silent.mp4` (add a trending sound in-app instead).
To use a virtualenv, set `MANIM=/path/to/venv/bin/manim PYTHON=/path/to/venv/bin/python`.
The film is rendered as 4 frame-exact slices in parallel (`JOBS=4`), then joined without re-encoding.

Fonts: Inter, Inter Display and JetBrains Mono (all open-source).

## How it is built

| File | Role |
|---|---|
| `story.py` | Facts, milestone texts, comparison-row specs, and the master timeline `n_at(t)` (doublings reached at time *t*). The picture and the sound both read it. |
| `film.py` | The scene. Each frame is posed from `n_at(t)`: zoom (`2**(n_i - n)`), an exponential pan between stops, the HUD, labels with STEP chips, the live row counters, the YOU pin, and the outro. |
| `visuals.py` | The 14 drawings, each authored at one reference size, plus `MeasureRow`, which acts out each comparison. |
| `assets.py` | Procedural textures in numpy: glows, electron cloud, the Sun, a barred spiral galaxy, the cosmic web. They are cached in `media/`. |
| `engine.py` | A faster camera. It composites only a sprite's own region, uses "screen" blending for light, mip-maps big textures, and adds a vignette. Also rigs, per-glyph text animation and fast monospaced counters. |
| `soundtrack.py` | A procedural score from the same timeline: a tick per doubling, plucks on the counted steps, speed-driven whooshes, a pentatonic bell per stop, a ratchet and chime for each counter, sub-bass impacts on the reveals, and reverb. |

## The numbers

Doublings are `log2(size / 1.616e-35 m)`. The STEP chip shows the nearest whole step.

| Stop | Size used | Doublings → STEP | Acted-out comparison (vs the previous stop) |
|---|---|---|---|
| Planck length | 1.616 × 10⁻³⁵ m (CODATA 2018): the scale where known physics runs out | 0 | — |
| Proton | 1.68 fm (2 × 0.841 fm charge radius) | 66.49 → 66 | 1.04 × 10²⁰ ≈ 100 quintillion Planck lengths |
| Hydrogen atom | 0.106 nm (2 × Bohr radius) | 82.44 → 82 | 63,000 ≈ 60,000 protons |
| Virus | 100 nm (influenza / coronavirus scale) | 92.32 → 92 | 943 ≈ 1,000 atoms |
| Red blood cell | 7.5 µm | 98.55 → 99 | 75 viruses |
| Ant | 4 mm (a typical small ant) | 107.61 → 108 | 533 ≈ 530 blood cells |
| Human | 1.7 m | 116.34 → 116 | 425 ants |
| Mount Everest | 8,849 m (2020 survey) | 128.69 → 129 | 5,205 ≈ 5,200 people |
| Earth | 12,742 km mean diameter | 139.18 → 139 | 1,440 Everests |
| Sun | 1,392,700 km | 145.95 → 146 | 109 Earths |
| Solar system | Neptune's orbit, 2 × 30.07 AU | 158.61 → 159 | 6,460 ≈ 6,500 Suns |
| Nearest star | Proxima Centauri, 4.2465 ly | 170.73 → 171 | 4,465 ≈ 4,500 solar systems |
| Milky Way | 100,000 ly (the common figure; estimates vary) | 185.26 → 185 | 23,549 ≈ 23,500 trips to Proxima |
| Observable universe | 93 billion ly diameter | 205.08 → 205 | 930,000 Milky Ways |

"You're closer to a galaxy than to the Planck length" is measured in doublings:
116 steps separate you from the Planck length, and only 69 separate you from the Milky Way.

## Posting

Posting rules are in [`../playbook/brand.md`](../playbook/brand.md) §6 (cover, captions, hashtags,
the answer rule) and the safe zones in [`../playbook/film-style.md`](../playbook/film-style.md) §1.
The launch package for this film, with its cover `the_doubling_cover_hook.jpg` (the hook frame at
1.9 s), is [`../lumiera_brand/FIRST_POST.md`](../lumiera_brand/FIRST_POST.md).

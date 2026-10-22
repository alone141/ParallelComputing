# The Doubling

A ~46-second vertical (9:16) film for TikTok and Instagram Reels, made with
[Manim](https://www.manim.community/). It starts at the Planck length and keeps
doubling. After **116** doublings you reach a human, after **185** the Milky Way,
and after **205** the whole observable universe. Each stop gives a relative
example ("109 Earths across", "75 viruses side by side", ...).

| | |
|---|---|
| Format | 1080×1920, 60 fps, H.264 + AAC (also a silent version) |
| Length | 46.4 s; the last frame matches the first, so the video loops cleanly |
| Hook | "How many times must you double this to reach a galaxy? Guess before it ends." |
| Payoff | "You're closer to a galaxy than to the Planck length" (116 doublings up, 69 to go) |

## Rendering

```bash
pip install manim            # needs Cairo/Pango dev headers (e.g. libpango1.0-dev)
./render.sh preview          # 540x960 @ 30 fps, a few minutes
./render.sh                  # 1080x1920 @ 60 fps, the upload version
```

The output goes to `out/the_doubling_final.mp4` (with sound) and
`out/the_doubling_final_silent.mp4` (add a trending sound in-app instead).
To use a virtualenv, set `MANIM=/path/to/venv/bin/manim PYTHON=/path/to/venv/bin/python`.

Fonts: Inter, Inter Display and JetBrains Mono (all open-source, from Google Fonts / JetBrains).

## How it is built

| File | Role |
|---|---|
| `story.py` | Facts, milestone texts, and the master timeline `n_at(t)`, which gives the doublings reached at time *t*. The picture and the sound both read it. |
| `film.py` | The scene. Each frame is posed from `n_at(t)`: zoom (`2**(n_i - n)`), an exponential pan between milestones, the HUD, labels and the YOU pin. |
| `visuals.py` | The 14 drawings (proton, atom, virus, ... galaxy), each authored at one reference size. |
| `assets.py` | Procedural textures in numpy: glows, electron cloud, the Sun, a barred spiral galaxy, the cosmic web. They are cached in `media/`. |
| `engine.py` | A faster camera. It composites only a sprite's own region, uses "screen" blending for light, mip-maps big textures, and adds vignette and film grain. Also rigs, per-glyph text animation and fast monospaced counters. |
| `soundtrack.py` | A procedural score: a tick per doubling, speed-driven whooshes, a pentatonic bell per milestone, sub-bass impacts on the reveals, and a reverb. |

## The numbers

Doublings are `log2(size / 1.616e-35 m)`.

| Stop | Size used | Doublings | Relative example |
|---|---|---|---|
| Planck length | 1.616 × 10⁻³⁵ m (CODATA 2018) | 0 | — |
| Proton | 1.68 fm (2 × 0.841 fm charge radius) | 66.5 | 1.04 × 10²⁰ Planck lengths |
| Hydrogen atom | 0.106 nm (2 × Bohr radius) | 82.4 | 63,000 × its proton |
| Virus | 100 nm (influenza / coronavirus scale) | 92.3 | 943 atoms |
| Red blood cell | 7.5 µm | 98.5 | 75 viruses |
| Ant | 4 mm (a typical small ant) | 107.6 | 533 blood cells |
| Human | 1.7 m | 116.3 | 425 ants |
| Mount Everest | 8,849 m (2020 survey) | 128.7 | 5,205 people |
| Earth | 12,742 km mean diameter | 139.2 | 1,440 Everests |
| Sun | 1,392,700 km | 146.0 | 109 Earths |
| Solar system | Neptune's orbit, 2 × 30.07 AU | 158.6 | 6,460 Suns |
| Nearest star | Proxima Centauri, 4.2465 ly | 170.7 | 4,465 solar systems |
| Milky Way | 100,000 ly (the common figure; estimates vary) | 185.3 | 23,500 trips to Proxima |
| Observable universe | 93 billion ly diameter | 205.1 | 930,000 Milky Ways |

"The great desert" (doublings 4–50) is shown only while the scale is still below
what any experiment has resolved (colliders reach about 10⁻²⁰ m, roughly doubling 50).

## Posting notes

* All text sits inside the TikTok/Reels safe area. Nothing important is in the
  bottom ~18% (caption) or the lower right (buttons).
* Suggested cover: the Milky Way frame (~35 s) or the "205" frame (~43 s).
* Caption idea: *"Double the smallest length in physics 205 times and you get… everything. Where were you? 👇"*

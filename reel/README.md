# Parallel computing — at every scale

A 55 s vertical (1080×1920, 60 fps) animated short for TikTok and Instagram
Reels, made with [Manim Community](https://www.manim.community/) 0.22.

| File | What it is |
|---|---|
| `out/parallel_scales_scored.mp4` | Final film with the procedural soundtrack (−14.5 LUFS, loops seamlessly) |
| `out/parallel_scales_silent.mp4` | Same picture, no audio, for adding a trending sound in-app |

## The film

It's a "Powers of Ten" zoom-out. Each scale collapses into one glowing token,
and that token opens as a single unit of the next scale. A counter in one honest
unit, *calculations at once* (simultaneous FP32 lanes), climbs as it goes, and
a lime line under it explains every jump.

Pacing rule: every idea lands on a calm stage and is read before the animation
it describes fires, and each payoff holds. Cards stay up at least
0.3 s per word + 0.8 s; the lime equation lines stay up at least 1.5 s.

| Film time | Beat | Scale | Mechanism on screen | Counter |
|---|---|---|---|---|
| 0.0 | `b01_hook` | Hook | one worker, one task at a time → 320 workers, one step | — |
| 5.0 | `b02_simd` | 1/6 one core | SIMD: one `vaddps` adds 16 number pairs in lockstep | 1 → 16 |
| 11.0 | `b03_cpu` | 2/6 one CPU | 16 cores, each running its own thread over shared memory | 256 |
| 16.5 | `b04_gpu` | 3/6 one GPU | H100: 16,896 lanes; warps of 32 threads, one instruction | 16,896 |
| 24.0 | `b05_node` | 4/6 one server | 8 GPUs exchanging results through NVSwitch | 135,168 |
| 29.5 | `b06_datacenter` | 5/6 data center | no shared memory: GPUs pass messages; 100,000+ GPUs train one AI | 1,689,600,000+ |
| 36.5 | `b07_planet` | 6/6 the planet | Folding@home 2020: work units out, results back | (retired) |
| 42.5 | `b08_amdahl` | the catch | Amdahl's law: 10% serial work caps speedup at 10× | speedup meter |
| 51.5 | `b09_loop` | loop | "So why build 100,000 GPUs?" → back to frame 0 | — |

The last frame of `b09` is the first frame of `b01`, and the worker's hop
clock runs straight through the cut, so the video loops seamlessly. Every
number is in [`design/facts.json`](design/facts.json) with its source.

## Posting kit

- **Title:** How parallel computing works — at every scale
- **Caption:** One worker vs 320. Then 16 lanes inside one CPU core, 16 cores,
  an H100's 16,896 math lanes, 8 GPUs on NVLink, 100,000+ GPUs training one
  AI, and in 2020 a planet of home PCs… until one law no budget can beat.
  So why build 100,000-GPU clusters? Wrong answers only ↓
- **Pinned comment:** Hint: Gustafson's law (1988). Grow the problem, not just the workers.
- **Hashtags:** #parallelcomputing #computerscience #gpu #cuda #simd #nvidia
  #supercomputer #ai #amdahlslaw #programming #coding #techtok #learnontiktok #stem #manim
- **Cover frame:** film time ≈ 34.6 s (b06 local ~5.1): the hall right after
  the big drop, with "100,000+" up top and "1,689,600,000+" below.
- **Accuracy notes for replies:**
  - The counter counts FP32 lanes only: no tensor cores, no FMA-as-2-FLOPs.
  - The Folding@home comparison holds only for April 2020, and it was not the HPL benchmark.
  - The 10% serial fraction is illustrative.

## Build

Setup. ffmpeg plus the Pango headers come from the system; everything else is in a venv.

```bash
sudo apt-get install -y ffmpeg libpango1.0-dev pkg-config
python3 -m venv ~/.venvs/manim
~/.venvs/manim/bin/pip install manim==0.22.0 "pillow<12" scipy   # pillow 12+ breaks manim 0.22 ImageMobject
```

Fonts: Inter, Inter Display and JetBrains Mono must be installed. There is no
LaTeX dependency.

```bash
PY=~/.venvs/manim/bin/python        # build.py defaults to /home/user/.venvs/manim; override with
export MANIM_BIN=~/.venvs/manim/bin/manim REEL_PYTHON=$PY

$PY reel/build.py contact b04_gpu          # 540x960 preview + time-stamped contact sheet PNG
$PY reel/build.py frames b04_gpu 1.3 4.55  # exact preview frames at beat-local times
$PY reel/build.py film                     # whole film at preview quality (~3 min)
$PY reel/build.py film --final             # 1080x1920 @ 60 fps masters into reel/out (~15 min)
$PY reel/tools/check_handoffs.py           # bit-exact check of all 9 cuts incl. the loop seam
```

## How it's built

- `timeline.py`: beat order, durations (every cut sits on the 120 BPM
  half-second grid), and music energy per beat.
- `style.py`: the 9×16-unit frame, safe zone checks (the TikTok/Reels UI
  covers the top, the bottom and the right rail), palette, typography and easing.
  - `BeatScene` counts frames exactly as manim writes them and pads each beat
    to its slot.
  - It logs audio cues with film-absolute times and drives a dust field from
    film time, so the dust continues across cuts and loops.
- `motif.py`: the shared visual grammar.
  - Tokens, cards, the HUD counter, camera EXPAND/COLLAPSE, flashes, the hook grid.
  - `handoff_state(name)`: the boundary frames. Adjacent beats build each cut
    from the same function, so they match by construction.
- `b01_hook.py` … `b09_loop.py`: one Scene per beat.
- `build.py`: renders the beats, concatenates them, and grades.
  - Bloom is computed on fixed-size intermediates, so previews and masters
    look the same.
  - The grade also adds a background lift, vignette and film grain, then
    synthesises the audio and muxes both MP4s.
- `audio/synth.py`: numpy/scipy soundtrack.
  - A D-minor bed (pad, arp, hats, kick) follows the per-beat energy.
  - Sound effects (hits, sub drops, whooshes, risers, ticks, glitches,
    shimmers) are placed on the logged cues, with a three-tier hit hierarchy.
  - Silences are true silences, effect tails wrap onto the start for the
    loop, and mastering uses BS.1770 loudness.
- `design/`: the storyboard (a judge panel's synthesis of three competing
  drafts), the verified fact sheet and the judging notes.

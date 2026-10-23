# Parallel computing — at every scale

A 55 s vertical (1080×1920, 60 fps) animated short for TikTok and Instagram
Reels, made with [Manim Community](https://www.manim.community/) 0.22.

| File | What it is |
|---|---|
| `out/parallel_scales_scored.mp4` | Final film with the procedural soundtrack (−14.5 LUFS, loops seamlessly) |
| `out/parallel_scales_silent.mp4` | Same picture, no audio, for adding a trending sound in-app |

## The film

It's a "level-up run" through three scales. Each scale collapses into one glowing
token, and that token opens as a single unit of the next scale. The final boss
is Amdahl's law. A counter in one honest unit, *cores working at once* (a
literal spec count of what the maker calls a core), climbs as it goes, and a
lime line under it explains every jump. The script is
[`design/script_v2.json`](design/script_v2.json), approved together with the
decisions recorded in its `user_decisions` field.

Pacing rule: every idea lands on a calm stage and is read before the animation
it describes fires, and each payoff holds. Cards stay up at least
0.3 s per word + 0.8 s; the lime equation lines stay up at least 1.5 s.

| Film time | Beat | Level | On screen | Counter |
|---|---|---|---|---|
| 0.0 | `b01_hook` | Hook | "One at a time." → "All at once." (320 tasks light up) → "It scales… then stalls." | — |
| 5.0 | `b03_cpu` | 1/3 one CPU | "That worker? One CPU core." → "16 cores. 16 different jobs." → "They share one memory." → "Need more?" | 1 → 16 (16-CORE RYZEN 9 9950X) |
| 14.5 | `b04_gpu` | 2/3 one GPU | "16? Cute. Meet the GPU." → "Thousands of tiny cores." → "Groups of 32. One instruction." → "Same math, every number." → "For AI? Not enough." | 16,896 (H100: 16,896 CUDA CORES) |
| 26.5 | `b06_datacenter` | 3/3 data center | "Many boxes. No shared memory." → "So they pass messages." → "Go bigger." → "100,000+ GPUs. One AI." → "More GPUs…" | 1,689,600,000+ (100,000+ GPUs × 16,896) |
| 39.0 | `b08_amdahl` | final boss | record-scratch → "So… infinite speed?" → "No." → "Some work can't be split." → "A million GPUs?" → "10% serial = max 10×." → "Forever." | speedup meter 1.0× → 9.9× → 10× |
| 51.5 | `b09_loop` | loop | "Why buy 100,000 GPUs?" · "Wrong answers only ↓" → back to frame 0 | — |

The last frame of `b09` is the first frame of `b01`, and the worker's hop
clock runs straight through the cut, so the video loops seamlessly. Every
number is in [`design/facts.json`](design/facts.json) with its source.

## Posting kit

- **Title:** How parallel computing works — at every scale
- **Caption:** 1 worker → 16 CPU cores → 16,896 CUDA cores → 1.7 BILLION CUDA
  cores training ONE AI. Then the final boss shows up: if 10% of the job can't
  be split, you max out at 10× faster. Forever. So why buy 100,000 GPUs?
  Wrong answers only ↓
- **Pinned comment:** How to beat the final boss: Gustafson's law (1988).
  Amdahl's 10× cap assumes the job stays the same size. AI labs grow the job with
  the machine (bigger models, more data), so the parallel part keeps growing and
  100,000 GPUs still pay off. Nerd notes: a "CUDA core" is NVIDIA's name for one
  tiny FP32 math lane. They run 32 at a time on one instruction (a warp), so they
  are not CPU-style cores. H100 SXM5 = 132 SMs × 128 = 16,896. The CPU count is
  physical cores (a 9950X has 16 cores / 32 threads). The 10% serial share is
  just an example.
- **Hashtags:** #parallelcomputing #computerscience #gpu #cpu #nvidia #cuda
  #datacenter #ai #amdahlslaw #programming #coding #techtok #learnontiktok #stem #manim
- **Cover frame:** film time ≈ 34.5 s (b06 local ~8.0): the hall right after
  the big drop, with "100,000+ GPUs. One AI." up top and "1,689,600,000+" below.
- **Accuracy notes for replies:**
  - The counter counts cores as each maker defines them: CPU physical cores, then NVIDIA FP32 CUDA cores.
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
$PY reel/tools/check_handoffs.py           # bit-exact check of every cut incl. the loop seam
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
- `b01_hook.py`, `b03_cpu.py`, `b04_gpu.py`, `b06_datacenter.py`, `b08_amdahl.py`,
  `b09_loop.py`: one Scene per beat (file numbers kept from the first cut).
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

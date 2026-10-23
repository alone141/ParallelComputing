# Film & visual style

How a Lumiera film looks, moves, reads and sounds. The reference implementation is
`doubling_film/` ("The Doubling"). When this guide and the code disagree, this guide wins:
fix the code, or change the rule here in the same pull request. Never let them silently drift.

Keywords: **MUST** (non-negotiable), **SHOULD** (the default; deviate only for a reason you can
state), **MAY** (optional).

**Units used in this guide**
- Positions and lengths are **scene units** on a frame **9 units wide × 16 tall** (Manim
  `frame_width=9`, `frame_height=16`). x runs from −4.5 to +4.5 and y from −8 (bottom) to +8 (top).
  At 1080×1920, 1 unit = 120 px, so `x_px = (x + 4.5)·120` and `y_px = (8 − y)·120`.
- Font sizes are **Manim `font_size`** values on that frame, not px or pt. One size unit is ≈ 1.7 px
  of em at 1080 wide (size 31 ≈ 51 px em; size 60 ≈ 100 px em, ≈ 62 px cap height).
- "Steps" are the film's counted unit (doublings in The Doubling). `n` is the current position on
  that scale, which may be fractional; a stop sits at `n_i`.

---

## 1. Format

| Property | Value |
|---|---|
| Aspect / resolution | **9:16, 1080×1920, 60 fps** (MUST) |
| Length | SHOULD be 45–75 s, with ~60–70 s as the default. MAY be shorter (down to ~30 s) when the sourced stops run out. Never pad with unsourced stops or with holds longer than §2 allows |
| Delivery | H.264 High, yuv420p, CRF ~18, AAC 256 kbps 48 kHz, `+faststart`. Ship a **with-sound** and a **silent** version. Intermediate renders MAY use higher quality (the reference renders at CRF 15) |
| Grain | Light luma grain is added **at encode time** (ffmpeg `noise=c0s=3:c0f=t`), never baked into frames. Why: baked grain made files 10× larger and turns to mush after TikTok/Instagram re-encode |
| Loop | The last frame SHOULD match the first frame, so the video replays seamlessly |
| Logo / end card | **None inside films.** No end card, lockup, watermark, handle or call to action. The loop is the sign-off; the brand lives on the profile. To introduce an end card, change this rule first. It would have to fit inside the outro and still end on the first frame |

### Safe zones (MUST)
TikTok and Reels cover parts of the frame with their own UI. These rules apply to every film frame
and therefore to every cover, since a cover is a film frame.
- **Bottom ~20%** (y < −4.8, i.e. below 1536 px): caption and sound bar. No text baseline below
  y = −4.8; descenders may dip to ≈ −4.9.
- **Right column** (x > +3.4 for y between −4.8 and +1.5, i.e. right of 948 px and 780–1536 px down):
  like, comment and share buttons. No text there.
- **Top ~8%** (y > +6.7): the tab bar. Only decorative elements.
- Centred text MUST fit within **6.8 units** (±3.4). Fit it down rather than letting it run wide.
- A cover's key text MUST sit inside **Instagram's central 3:4 crop** (1080×1440, i.e. y between
  +6.0 and −6.0).

---

## 2. Structure of a film

Every film is **one honest question, answered on screen**.

1. **Hook (0–~2.5 s).** A question the viewer can answer in their head, with the signature
   sub-line "Guess before it ends." (see `brand.md` §1).
   - The glyphs rise in from frame 0, 0.32 s each with a 0.011 s stagger, so the whole question is in by ≈ 0.8 s.
   - The sub-line enters at ≈ 0.85 s.
   - Both leave from 2.1 s and MUST be fully gone before the HUD fades in (2.5–2.95 s).
   - The starting object (e.g. the Planck dot with its dashed reticle) is visible underneath from frame 0.
   - **The hook frame** is the moment the question and sub-line are both fully on screen and not yet
     leaving (1.9 s in The Doubling). Frame 0 is **not** the hook frame.
2. **Teach the mechanic (~3 s).** Show the rule slowly before speeding up. The Doubling counts its
   first 8 doublings one at a time: the first step takes 0.40 s, tapering to 0.25 s, each followed by
   a 0.06 s settle. Each step gets a ×2 pop, the digit change and a pluck, all at the same moment.
   A phase label explains it ("DOUBLE IT. AGAIN." / "Every step makes it twice as long.").
3. **Journey.** A sequence of **stops** with **holds**, joined by fast transitions (a warp; near-empty
   frames are intended there). A long stretch with no stop gets a phase label (e.g. "THE GREAT
   DESERT"), which is a label block without a STEP chip.
4. **Payoff.** The answer to the hook, plus one reframing line that makes it personal
   ("You're closer to a galaxy than to the Planck length."), with the meter brackets (§4).
5. **Outro (~4.2 s).**
   - The HUD counter flies to the centre (y ≈ +0.95, 2.15× larger, over 0.85 s), turning `#FFD1F3`
     with a Magenta glow.
   - Under it come a unit label (e.g. "DOUBLINGS", Inter Medium 26, Mist Light, tracked +0.13 units,
     y = −0.78) and a one- or two-line summary (Inter Medium 34, Starlight, centred at y = −1.95).
   - Over the last ~1.2 s, everything glides back to the first frame so the video loops. The
     background, glow and particles return to their frame-0 state.

The answer to the hook MUST NOT appear in the cover or any post text (full list: `brand.md` §6).

### Pacing (SHOULD)
| Element | Timing |
|---|---|
| Hold at a stop | **~3.2 s**. Use up to ~3.8 s when a note follows, ~5.4 s for the payoff stop and ~4.6 s for the final stop |
| Transition between stops | `0.5 + 0.028 × (steps travelled)` seconds (≈ 0.7–1.1 s) |
| First sprint (end of teaching → first stop) | ≈ 2.5 s |
| Labels | Enter 0.3 s **before** the object settles; exit when the next transition starts |
| Note (a second line that replaces the comparison) | Enters 1.2–2.5 s into the hold, after the comparison has been read |
| Comparison row | Starts 0.5 s after arrival and fills in **1.5 s** (ease-in-out). A row that belongs to a note starts as the note enters |
| Motion | Never fully stops. During a hold the camera keeps drifting slowly in the direction of travel: from `n_i − 0.15` to `n_i + 0.25` |

---

## 3. The visual language

**Light on deep space.** Subjects are drawn as glowing light: crisp vector lines and shapes, soft
additive glows, and procedural textures for the Sun, galaxies and so on. Backgrounds are near-black.

- **Background.** The film base is near-black indigo (`#08061A`; brand Deep Space `#07061A`). It MAY
  shift subtly in hue by scene (warm near the Sun, blue near Earth), but it MUST stay near-black, with
  no channel above `0x14`. The Doubling's per-scene ramp is `BG_KEYS` in `doubling_film/film.py`.
- A soft **nebula glow** (≈ 20% strength) sits behind the featured object, tinted with its accent.
  Dial it down (to ~25–60%) where it would read as mud, e.g. empty frames between two stars.
- Glows MUST add like light ("screen" blend) and MUST fade to zero before the edge of their sprite.
  A visible square edge on a glow is a defect.
- A light vignette is fine. Banding is avoided by the encode-time grain.

### Colour roles
- **The scene's accent** colours the scene's own elements: the STEP chip, the size text when there is
  no chip, the `[highlighted]` number, the nebula tint and the counter flash. One accent per scene.
- **Fixed meanings that override the scene accent:**
  - **Cyan `#6CF2FF` = you.** It's used for the YOU pin and tag, copies of people standing in for
    you, your distance to the answer (the "69" bracket), and the hook's target word ("galaxy?").
    It's never the accent of any other subject.
  - **Measuring rows** take the colour of **the thing being counted** (§4), lightened where it would
    not read against the current subject.
  - **Lavender `#B9A6FF`** is the start: the starting object, the mechanic phase, the ×2 pops, and the
    mechanic word in the hook ("double").
  - **Magenta `#FF6FD8`** is the finale glow and the outro counter's glow.
  - The **progress meter** fill is a gradient from the start accent to the final accent
    (lavender → cyan → magenta in The Doubling).
- **Text colours.** Starlight `#F5F7FF` for titles and main text. Mist Light `#C9D1E6` for readable
  secondary lines (comparisons, notes, size text beside a chip, captions inside drawings, the outro
  label). Mist `#8A93A8` for small labels and hints (HUD label, hook sub-line, the meter's "0").
  The HUD counter is the only pure white `#FFFFFF`.

### Stop accents (reuse for the same subjects in every film)
| Subject | Accent |
|---|---|
| Planck / quantum | `#B9A6FF` lavender |
| Proton | `#FF8A5B` |
| Atom | `#5CC8FF` |
| Virus | `#FF5FA2` (row copies `#FF8FCB`) |
| Blood | `#FF4D5E` |
| Ant / warm life | `#FFB547` amber |
| You / people | `#6CF2FF` cyan |
| Mountains | `#A9D6FF` |
| Earth | `#4DA3FF` (row copies `#8CCBFF`) |
| Sun | `#FFB13B` |
| Solar system | `#8FB8FF` |
| Stars | `#FF8C6B` |
| Galaxy | `#C7A6FF` |
| Universe / finale | `#FF6FD8` magenta |

A **new subject** gets its accent once: at least ~30° of hue away from neighbouring stops, never cyan
or magenta. Add it to this table in the same pull request, and reuse it in every later film.

Drawing style per subject (SHOULD):
- Spheres get a night-side shade, a rim light on the lit edge and a small specular glow, with light
  from the upper left.
- Small life gets neon line art over a warm fill.
- People are a clean pictogram.
- Mountains are low-poly facets with snow.
- Space objects are procedural textures (galaxy, cosmic web, Sun with limb darkening and spots).

### Movement
- **Zoom by scale.** An object at stop `i` is drawn at `2^(n_i − n)` of its reference size (a
  diameter of 5.2 units, centred at y = +0.22).
  - It's fully visible from about 1/10 to about 1.5× its reference size.
  - It fades in between 3× and 1.5× (`n_i − 1.6` to `n_i − 0.55` when growing).
  - It fades out between 1/10 and 1/85 (`n_i + 3.4` to `n_i + 6.4`).
  - Nothing pops in at full size.
- **Direction.** These rules are defined on `n`, so they hold both ways. In a film that zooms in
  (`n` falls; e.g. The Halving):
  - objects arrive as a point and grow, and leave by growing past the frame;
  - streaks flow outward and the doubling rings expand;
  - the hold drift and the idle particle drift follow the direction of travel;
  - the pops read "÷2".
  Drift MUST NOT carry `n` past a physical limit: if the last stop is the Planck length, the final
  hold ends on it and the outro holds `n` constant.
- **Exponential pan.** When the next object is off-centre (the human standing on Everest, Earth beside
  the Sun), the camera pans exponentially, so the previous object glides smoothly to its place in the new one.
- **Warp particles.** About 150 radial streaks, with length tracking the zoom speed. Doubling rings
  (radii ×2) move with the zoom. Both stay faint during holds and brighten during sprints.
- **Easing.** Use ease-in-out for counts and fills, ease-out for text entering, and smoothstep for fades.
  No linear starts or stops on visible motion.

---

## 4. The devices (the Lumiera toolkit)

Reuse these. They are what makes a film recognisably ours. The geometry and code are in `doubling_film/film.py` and `visuals.py`.

| Device | Rule |
|---|---|
| **HUD counter** | Big number at top centre (JetBrains Mono Bold, size 104, y = +5.6, white `#FFFFFF`). Above it, a label at y = +6.55 (Inter Medium 19, CAPS, Mist, tracked +0.09 units) names the film's unit ("DOUBLINGS", "HALVINGS"). It fades in over 2.5–2.95 s, after the hook. While a stop holds, it shows that stop's step. On arrival it **punches** (+16% and back over 0.5 s) and flashes 85% toward the accent, decaying over ~0.45 s |
| **Size readout + ruler** | A live scientific readout ("1.7 × 10⁰ m"): JetBrains Mono Medium 30, `#9FB3D9`, y = +4.5, mantissa to one decimal, with the exponent drawn as raised 0.6× digits and a true minus. It stays live during holds, because it labels the ruler; the stop's nominal size is the label's size text. Below it, a ruler at y = +4.12 (`#5D6B8C`, 1.6 px) whose width equals the reference size (5.2 units) |
| **STEP chip** | At every stop, a pill ("STEP 116") left of the size text and above the title. JetBrains Mono Bold 22 in the accent, on an accent fill at 16% with a 1.6 px accent outline. **Step number = `round(|n_stop − n_start|)` on unrounded values**, in the film's own unit counted from its own start. The HUD shows the same number during the hold. Never subtract rounded steps taken from another film |
| **Measuring row** (acted-out comparison) | Copies of the **smaller of two neighbouring stops** line up across the larger one at its stop, or stack for heights, while a live counter climbs ("0 → 425 ants"). When the film grows, that's the previous stop; when it shrinks, it's the next one. Draw individual dots (radius 0.46 × spacing) when each copy is at least ~4 px (row length in px ÷ count); otherwise draw a glowing line (core 3.2 px, glow 10 px at 22%) with ~90 white beads (radius 0.022 units) and a small star at the growing end. The finished count MUST equal the number in the line on screen while it fills (the comparison, or the note at a final stop). The row fades out with its counter |
| **Live counter** | JetBrains Mono Bold 21, in the row colour mixed 40% toward white, with a black halo (8 px stroke, 78%, round joins so there are no spikes). The format is commas for counts, plus a noun ("4,500 solar systems"), and words for totals of ten million or more ("100 quintillion Planck lengths"). For horizontal rows it rides the middle of the filled part, 0.3 units above or below the row (further if captions are in the way), clamped to x ∈ [−3.4, +3.3]. For vertical rows it sits beside the growing end. It MUST NOT collide with the YOU pin, the meter or other text |
| **YOU pin** | A cyan capsule (0.9 × 0.4 units) reading "YOU" (Inter Display Bold 21, `#03121A`), with a stem and a small target ring, marking the viewer's position. Show it only while that position is inside the frame and the human is no bigger than ~1.2× the reference size: from the human stop onward when the film grows, or until the human leaves the frame when it shrinks. The meter's YOU tag remains throughout |
| **Progress meter** | A **size axis** at the left edge: x = −3.95, from y = −2.45 (the smallest size shown) to y = +3.45 (the largest). The base line is `#27304A`, 2 px; ticks at each stop are in that stop's accent, at 30% opacity until passed. The fill runs from the start stop toward the current `n` (downward in a film that shrinks), with the accent gradient from §3. A "0" in Mist sits at the start, and a "YOU" tag (Inter Display Bold 16, cyan) sits at the viewer's step. For the payoff, brackets go to the right of the YOU tag (x = −3.21): mono bold 20, the part below "you" in Mist Light, the part above in cyan ("116" / "69"). They appear with the punchline and stay until 0.6 s into the next stop |
| **Answer ring** | When the payoff stop arrives, a thin ring in its accent grows around the HUD counter (1.4 → 5.0 units over 1.2 s, ease-out) and fades |
| **×2 / ÷2 pops** | While the mechanic is taught, "×2" (Inter Display Bold 40, lavender) pops beside the counter at (+1.55, +5.62): +25% scale, then fading over ~0.3 s. It happens together with the digit change and the pluck |

---

## 5. Typography & text

| Role | Font | Size | Notes |
|---|---|---|---|
| Hook question | Inter Display Bold | 50 | **Sentence case.** 3 short lines, block centred at y = +4.75 with a 0.16 line gap. The mechanic word in lavender, the target word in cyan, the rest Starlight |
| Hook sub-line | Inter Regular | 27 | "Guess before it ends." in Mist, at y = +3.3 |
| Stop / phase title | Inter Display Bold, CAPS | 60, fit to ≤ 6.8 units | "THE MILKY WAY" |
| Comparison / note line | Inter Medium | 31 (29 when two lines) | Sentence case, Mist Light. The `[bracketed]` number is in the accent. Fit down to ≤ 6.8 units, never below 26; if it still doesn't fit, shorten the line. Only notes may use two lines |
| STEP chip | JetBrains Mono Bold | 22 | See §4 |
| Size text | JetBrains Mono Medium | 25 | In Mist Light beside a chip, or in the accent when there's no chip (phase labels). If it contains Unicode superscript exponents ("1.6 × 10⁻³⁵ m"), set it in Inter Medium 27, because JetBrains Mono has no superscript minus |
| Small captions inside drawings | Inter | 17–20 | "Sun", "Proxima Centauri", planet names. Mist Light |
| HUD label | Inter Medium, CAPS | 19 | Tracked **+0.09 units per glyph** (≈ +33% of the em) |

**Letter spacing.** Film tracking is given in scene units per glyph (`engine.text(tracking=…)`).
The brand wordmark's "+32%" (`brand.md` §2) is a percentage of the em.

**Label block positions** (line centres, y in units):
- STEP chip and size text at −2.86.
- Title at −3.51.
- Comparison (first line) at −4.21.

A two-line note keeps its first line where a single line sits and grows downward. Its last baseline
MUST stay at or above y = −4.8.

**Punctuation & spelling.**
- Comparison lines that are noun phrases take no full stop ("[425] ants tall").
- Full sentences end with one ("Doubling 116. Remember that.", "Smaller than any experiment can probe.").
- `→` and `…` appear only inside counting sequences ("2 → 4 → 8 → …", "STEPS 9 → 50").
- No emoji or ✦ on screen.
- British spelling (metres, centre, colour).

**Text animation.** Glyphs rise in one at a time: rise 0.22 units over 0.42 s, stagger 0.014 s
(titles 0.018 s, comparisons 0.008 s). Exit is a short lift (0.16) and fade over 0.28 s.

Text MUST NOT overlap other text, the YOU pin or the meter at any time. Check with stills.

---

## 6. Numbers & claims

- **One source of sizes.** Each film's README holds a "The numbers" table, and its code (`story.py`)
  is the only source of its sizes. Captions, descriptions, sources pages and comments quote from that
  table, to at least 3 significant figures (1.68 fm, 0.106 nm, 4.25 ly). A later film reuses the same
  size for the same subject (you 1.7 m, hydrogen atom 0.106 nm, proton 1.68 fm).
- **Rounding.** Round to the nearest value, choosing precision for readability:
  - Prefer **2 significant figures** (533 → 530, 5,205 → 5,200, 6,460 → 6,500, 4,465 → 4,500). A third
    MAY be kept (23,549 → 23,500).
  - Whole counts below 1,000 MAY be exact (109, 425).
  - A rounder figure MAY be used when the line says **"about" or "≈"** (943 → "about 1,000";
    63,095 → "≈ 60,000×"; 1.04 × 10²⁰ → "about 100 quintillion").
  - Any figure more than ~2% off the computed value MUST carry "about" or "≈", on screen included.
  - Never choose the direction or the precision because the result sounds bigger.
- **Comparisons** are between a stop and its smaller neighbour ("75 viruses side by side"), so they
  match the measuring row.
- **Keep honest qualifiers:** *observable* universe, *where known physics runs out* (never "the
  smallest length in physics"), *on average*, *in theory*. The outro MAY say "to the edge of
  everything" once "THE OBSERVABLE UNIVERSE" has been on screen in the same film.
- **Range claims.** A claim tied to a range is on screen only while the scale is inside that range, in
  either direction. It leaves before the range ends: "Smaller than any experiment can probe" goes
  before step 50, since colliders resolve ~10⁻²⁰ m. If that leaves less than ~1.5 s to read it,
  move the claim into the next stop's note.
- **Notation.** Write powers of ten as "1.6 × 10⁻³⁵ m": a multiplication sign and a raised exponent.
  Never use `^`, e-notation or "x".

---

## 7. Sound

The score is generated from the same timeline as the picture, so it's always in sync
(`doubling_film/soundtrack.py`).

| Layer | Rule |
|---|---|
| Hook | A soft sub bloom (70 → 40 Hz) with an A4 bell at frame 0, then a 1.4 s filtered-noise riser into the first step |
| Tick | One soft tick per step. It turns into a buzz in sprints, with gain falling as the rate rises |
| Plucks | A rising pentatonic pluck (E4 G4 A4 C5 D5 E5 G5 A5) on each step of the mechanic phase |
| Whoosh | Filtered noise that follows the zoom speed. Keep it under the bells, never harsh |
| Stop bell | A bell on each regular stop's arrival, climbing an A-minor pentatonic scale, over a small sub thump (120 → 60 Hz) |
| Big reveals | No bell. Instead, a sub-bass boom (95 → 34 Hz) plus a chord. "You" is A major (A3 E4 A4 C♯5), the payoff is A minor add9 (A2 E3 A3 C4 E4 B4) and the finale is F major 7/9 (F2 C3 A3 E4 G4 C5). These are the loudest moments |
| Answer shimmer | Four quiet high bells (E5 A5 B5 E6) as the payoff line lands |
| Counter ratchet | ~22 soft clicks (1.5 → 3.2 kHz) that follow the row's easing, plus a quiet E6 chime when it lands |
| Outro | A filtered swell into the counter's flight, then a boom and an A-minor chord (A2 E3 B3 C4 E4 A4) |
| Pad | A quiet A-minor pad whose brightness follows size, so it darkens as things shrink. Thinner in empty stretches |
| Master | **−14 to −16 LUFS integrated, true peak ≤ −1.0 dBTP, measured on the final MP4** (ffmpeg `ebur128=peak=true`). Master the WAV to about −1.5 dBFS sample peak so AAC encoding stays under −1 dBTP. Convolution reverb |

In a film that shrinks, the plucks and stop bells still rise with **progress**, while the pad follows
**size**. When the film starts at you, the frame-0 bloom is the "you" moment.

---

## 8. Definition of done (check every film against this)

- [ ] 1080×1920, 60 fps. Frame count = `round(total × 60)` ± 1 (slice rounding), checked with
      `ffprobe -count_frames`. With-sound and silent versions exported.
- [ ] Hook in by ≈ 0.8 s and fully gone before the HUD appears. The hook frame is identified for the cover.
- [ ] No text baseline below y = −4.8 and nothing in the right button column. Centred text ≤ 6.8 units wide.
- [ ] No text-on-text, text-on-pin or text-on-meter collisions at any stop. Check stills at arrival,
      mid-row, row end and late hold.
- [ ] STEP chip = HUD counter during each hold. Each row's final count = the number in the line on
      screen while it fills.
- [ ] No glow with a square edge. No black spikes on text halos.
- [ ] Every number traceable to the film's "The numbers" table and rounded per §6. The answer to the
      hook is absent from the cover and every post text.
- [ ] Loudness −14 to −16 LUFS and true peak ≤ −1 dBTP on the final MP4.
- [ ] Last frame ≈ first frame (loop).

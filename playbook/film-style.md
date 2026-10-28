# Film & visual style

How a Lumiera film looks, moves, reads and sounds. The guide has two parts:

- **Part A, the general aesthetics (§1–§7),** which every film MUST meet: the format and safe zones, the
  structure of one honest question, light and colour, the faces and their case, numbers and claims, the
  master targets, the loop, and the definition of done. They are written so that they presuppose no device.
- **Part B, The Doubling's toolkit (§8),** the Scale pillar's default: The Doubling's structure and pacing,
  its devices, its label block and text animation, and its sound palette. A film MAY reuse the toolkit in
  whole, in part or not at all, and says which in its README.

The reference implementations are `doubling_film/` (*The Doubling*: the general aesthetics and the whole
toolkit) and `three_doors_film/` (*Three Doors*: the general aesthetics, built independently of the toolkit).
The shelved *The Long Number* (`archive/shuffle_film_long_number/`) is cited below where a rule was first
written for it; its text choices stay valid examples.
When this guide and the code disagree, this guide wins: fix the code, or change the rule here in the same
pull request. Never let them silently drift.

Keywords: **MUST** (non-negotiable), **SHOULD** (the default; deviate only for a reason you can
state), **MAY** (optional).

**Units used in this guide**
- Positions and lengths are **scene units** on a frame **9 units wide × 16 tall** (x from −4.5 to +4.5,
  y from −8 at the bottom to +8 at the top; Manim `frame_width=9`, `frame_height=16`). At 1080×1920,
  1 unit = 120 px, so `x_px = (x + 4.5)·120` and `y_px = (8 − y)·120`. A film that lays out in pixels uses
  the px figures given beside each rule.
- Font sizes are **Manim `font_size`** values on that frame, not px or pt. One size unit is ≈ 1.7 px
  of em at 1080 wide (size 31 ≈ 51 px em; size 60 ≈ 100 px em, ≈ 62 px cap height). A film that sets its
  type in px quotes px em.

---

# Part A. The general aesthetics (every film MUST meet these)

## 1. Format

| Property | Value |
|---|---|
| Aspect / resolution | **9:16, 1080×1920, 60 fps** (MUST) |
| Length | SHOULD be 45–75 s, with ~60–70 s as the default. MAY be shorter (down to ~30 s) when the sourced stops run out, or as a deliberately short cut that lands its payoff early and whose every beat adds a new comparison (*The Long Number*, ≈ 37 s). Never pad with unsourced stops or with holds longer than the reading needs |
| Delivery | H.264 High, yuv420p, CRF ~18, AAC 256 kbps 48 kHz, `+faststart`. Ship a **with-sound** and a **silent** version. Intermediate renders MAY use higher quality (the reference renders at CRF 15; lossless slices are fine) |
| Grain | Light luma grain is added **at encode time** (ffmpeg `noise=c0s=3:c0f=t`), never baked into frames. Why: baked grain made files 10× larger and turns to mush after TikTok/Instagram re-encode |
| Loop | The last frame MUST match the first frame, so the video replays seamlessly: **the loop is the sign-off.** After the encode and the grain the two frames are not bit-identical, so the match is measured on the final MP4 against a threshold the film's README states. The score loops with the picture (§6) |
| Logo / end card | **None inside films.** No end card, lockup, watermark, handle or call to action. The brand lives on the profile. To introduce an end card, change this rule first. It would have to fit inside the ending and still end on the first frame |

### Safe zones (MUST)
TikTok and Reels cover parts of the frame with their own UI. These rules apply to every film frame
and therefore to every cover, since a cover is a film frame.
- **Bottom ~20%** (y < −4.8, i.e. below 1536 px): caption and sound bar. No text baseline below
  y = −4.8; descenders may dip to ≈ −4.9 (1548 px).
- **Right column** (x > +3.4 for y between −4.8 and +1.5, i.e. right of 948 px and 780–1536 px down):
  like, comment and share buttons. No text there.
- **Top ~8%** (y > +6.7, i.e. above 156 px): the tab bar. Only decorative elements.
- Text MUST fit within the **6.8-unit band** (±3.4, i.e. x 132–948 px), whether it is centred or set flush.
  Fit it down rather than letting it run wide.
- A cover's key text MUST sit inside **Instagram's central 3:4 crop** (1080×1440, i.e. y between
  +6.0 and −6.0, 240–1680 px).

---

## 2. Structure: one honest question

Every film is **one honest question, answered on screen**. Whatever its devices, it has these beats, in
this order.

1. **Hook.** A question the viewer can answer in their head, with the signature sub-line
   "Guess before it ends." (`brand.md` §1). The question is fully on screen within the first second or
   so, the viewer has time to read both lines, and both are fully gone before the first answering element
   appears. The starting object MAY be visible underneath from frame 0.
   - **The hook frame** is the moment the question and sub-line are both fully on screen and not yet
     leaving (1.9 s in The Doubling). Frame 0 is **not** the hook frame. The hook frame is the cover
     (`brand.md` §6), so it carries nothing of the answer.
   - The mechanic word is lavender and the target phrase follows §3; the rest is Starlight.
2. **Teach the mechanic.** Show the rule slowly before speeding up, with as few words as the picture
   needs. The mechanic is seen happening, not asserted.
3. **The journey.** A sequence of stops, each with a comparison **the picture itself makes** (copies of a
   smaller thing filling a larger one, a length set against a length, a count climbing to a figure), held
   long enough to read. A comparison is shown, never only stated, and its figure is the one in the line on
   screen (§5).
4. **Payoff.** The answer to the hook, plus one reframing line that makes it land ("You're closer to a
   galaxy than to the Planck length.").
5. **Ending and loop.** A short ending (a closing line, or the toolkit's outro, §8.1), then everything
   returns to its frame-0 state so the video loops (§1). No end card.

The answer to the hook MUST NOT appear in the cover or any post text (full list: `brand.md` §6).

How long a hold lasts, how a transition moves and exactly when the hook enters and leaves are the film's
own; the toolkit's figures (§8.1) are the Scale pillar's default. Whatever the timings, every line is on
screen long enough to be read on a phone, a line has fully left before another takes its place, and
nothing is cut off mid-fade.

---

## 3. The visual language

**Light on deep space.** Subjects are drawn as glowing light: crisp vector lines and shapes, type, soft
additive glows, and procedural textures where a subject needs them. Backgrounds are near-black.

- **Background.** The film base is near-black indigo (`#08061A`; brand Deep Space `#07061A`). It MAY
  shift subtly in hue by scene (warm near the Sun, blue near Earth, the scene's accent in a film of type),
  but it MUST stay near-black, with no channel above `0x1A` (the base's own blue). The Doubling's
  per-scene ramp is `BG_KEYS` in `doubling_film/film.py`.
- A soft **nebula glow** (≈ 10–20% opacity) MAY sit behind the featured object or the composition's
  centre, tinted with the scene's accent. Where it would read as mud or wash out the subject, scale it
  down, never above 1 (The Doubling: star 0.25, solar system 0.6, Everest 0.7, Earth 0.8; `NEBULA` in
  `film.py`).
- Glows MUST add like light ("screen" blend) and MUST fade to zero before the edge of their sprite.
  A visible square edge on a glow is a defect.
- A light vignette is fine. Banding is avoided by the encode-time grain.

### Colour roles
- **The scene's accent** colours the scene's own elements: its title or lead, its highlighted number, its
  glow tint, and whatever marks the stop (a chip, a rule, a flash). One accent per scene.
- **Fixed meanings that override the scene accent:**
  - **Cyan `#6CF2FF` = you.** It is used for the YOU pin and tag, copies of people standing in for you,
    everyone alive, your distance to the answer, and the hook's target word when the target is you
    ("galaxy?"). It is never the accent of any other subject, and nothing that is not you is ever cyan.
  - **The hook's target phrase** is cyan when the target is you. When the target is not you, it MAY take
    its own subject's accent (lime for "the Big Bang?" in The Long Number) where the film's README says
    so. It is never a third colour.
  - **Comparisons** take the colour of **the thing being counted or measured against**, lightened where
    it would not read against the current subject.
  - **Lavender `#B9A6FF`** is the start and the mechanic: the starting object, the mechanic phase, the
    marks of the mechanic (the ×2 / ÷2 pops, the multiplier cells, the dealt card) and the mechanic word
    in the hook ("double", "possible deals"). When the starting object is you, cyan wins: the human stays
    cyan, and lavender marks only the mechanic.
  - **Magenta `#FF6FD8`** is the observable universe's accent and, in a film that flies its counter to
    the centre at the end (§8.1), that counter's glow. It is never used for a mid-film reveal. A final
    stop other than the universe keeps its own accent, and a film with neither has no magenta.
- **Text colours.** Starlight `#F5F7FF` for titles and main text. Mist Light `#C9D1E6` for readable
  secondary lines (comparisons, notes, size text beside a chip, captions inside drawings, an outro
  label). Mist `#8A93A8` for small labels and hints (a HUD label, the hook sub-line, a meter's "0").
  Pure white `#FFFFFF` appears only on a HUD counter (§8.3), never on other text.

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
| Universe | `#FF6FD8` magenta |
| Time / the Big Bang | `#D4FF5C` lime (first used in The Long Number) |
| Cards / the whole deck | `#E08BFF` violet (The Long Number; 31° of hue from lavender, 32° from magenta) |

A **new subject** gets its accent once: at least ~30° of hue away from neighbouring stops, never cyan
or magenta. Add it to this table in the same pull request, and reuse it in every later film.

Drawing style per subject, when a subject is drawn (SHOULD):
- Spheres get a night-side shade, a rim light on the lit edge and a small specular glow, with light
  from the upper left.
- Small life gets neon line art over a warm fill.
- People are a clean pictogram.
- Mountains are low-poly facets with snow.
- Space objects are procedural textures (galaxy, cosmic web, Sun with limb darkening and spots).

The Doubling's movement (zoom by scale, the exponential pan, warp particles and its easing rules) is in
§8.2. A film chooses its own motion language and describes it in its README; what every film shares is
the light above.

---

## 4. Typography & text

Three faces and no others (`brand.md` §4). Check they are installed before rendering: renderers
substitute a missing family without warning.

| Role | Face | Case and colour |
|---|---|---|
| Hook question | Inter Display Bold | **Sentence case.** The mechanic word lavender, the target phrase per §3, the rest Starlight |
| Titles, stop leads, phase titles | Inter Display Bold | **CAPS** by default ("THE MILKY WAY"). A film that has no caps labels at all MAY set its titles and leads in sentence case ("Eleven cards dealt.") where its README says so (The Long Number) |
| Hook sub-line, comparison, note and teaching lines, captions inside drawings | Inter Regular / Medium | **Sentence case**, Mist Light (the sub-line Mist). The key phrase or `[bracketed]` number in the accent |
| Numbers: counters, readouts, chips, cells, standalone figures | JetBrains Mono | A numeral inside a sentence stays in the sentence's face |
| Small tracked labels (a HUD label) | Inter Medium, CAPS | Tracked **+0.09 units per glyph** (≈ +33% of the em) |

**Fit.** Text MUST fit within the 6.8-unit band (§1). Fit a line down rather than letting it run wide; if
it still does not fit, shorten the line. Set every line so it reads on a phone, and check stills at phone
size (a 360×640 downscale) as well as full size.

**Letter spacing.** Film tracking is given in scene units per glyph (`engine.text(tracking=…)`).
The brand wordmark's "+32%" (`brand.md` §2) is a percentage of the em.

**Punctuation & spelling.**
- Comparison lines that are noun phrases take no full stop ("[425] ants tall").
- Full sentences end with one ("Doubling 116. Remember that.", "Smaller than any experiment can probe.").
- `→` and `…` appear only inside counting sequences ("2 → 4 → 8 → …", "STEPS 9 → 50", "52 × 51 × 50 × …").
- No emoji or ✦ on screen.
- British spelling (metres, centre, colour).

Text MUST NOT overlap other text, or any pin, meter, rule or card, at any time. Check with stills.

The sizes, positions and text animation of The Doubling's label block are in §8.4.

---

## 5. Numbers & claims

- **One source of numbers.** Each film's README holds a "The numbers" table, and its code (`story.py`)
  is the only source of its figures. Post text (captions, titles, descriptions, comments) that repeats a
  figure shown on screen writes it exactly as the film shows it ("1.6 × 10⁻³⁵ m"). Lists of sizes (the
  YouTube "Sizes used" line, a sources page) quote the table to at least 3 significant figures, or as
  the table gives it when it has fewer (1.616 × 10⁻³⁵ m, 1.68 fm, 0.106 nm, 60.1 AU, 4.25 ly; 4 mm, 1.7 m). A later film reuses the same
  figure for the same subject (you 1.7 m, hydrogen atom 0.106 nm, proton 1.68 fm, 8.3 billion people).
- **Rounding.** Round to the nearest value, choosing precision for readability:
  - Prefer **2 significant figures** (533 → 530, 5,205 → 5,200, 6,460 → 6,500, 4,465 → 4,500). A third
    MAY be kept (23,549 → 23,500).
  - Whole counts below 1,000 MAY be exact (109, 425).
  - A rounder figure MAY be used when the line says **"about" or "≈"** (943 → "about 1,000";
    63,095 → "≈ 60,000×"; 1.04 × 10²⁰ → "≈ 100 quintillion").
  - Any figure more than ~2% off the computed value MUST carry "about" or "≈", on screen included.
  - Never choose the direction or the precision because the result sounds bigger.
- **Bounds.** A figure written as a bound ("at most 1 in N") rounds N in the direction that keeps the bound
  true (down, for "at most 1 in N"); a figure rounded to the nearest value says "about" instead.
- **Comparisons** are the ones the picture makes (§2), so the words and the picture never disagree, and
  the figure in the line is the figure the picture reaches. A stop SHOULD beat the thing it is compared
  with clearly (by ~20% or more), and a threshold answer ("the eleventh card") holds from below as well:
  the step before falls clearly short.
- **Keep honest qualifiers:** *observable* universe, *where known physics runs out* (never "the
  smallest length in physics"), *on average*, *in theory*, *dealt from a full deck*, *a good shuffle*.
  The ending MAY say "to the edge of everything" once "THE OBSERVABLE UNIVERSE" has been on screen in
  the same film.
- **Range claims.** A claim tied to a range is on screen only while the scale is inside that range, in
  either direction. It leaves before the range ends: "Smaller than any experiment can probe" goes
  before step 50, since colliders resolve ~10⁻²⁰ m. If that leaves less than ~1.5 s to read it,
  move the claim into the next stop's note.
- **Notation.** Write powers of ten as "1.6 × 10⁻³⁵ m": a multiplication sign and a raised exponent.
  Never use `^`, e-notation or "x". A film MAY instead write its numbers **positionally** on screen, in
  full, with digits in groups of three and "≈" (or "about") on every rounded figure, where its README
  justifies it (The Long Number, whose numbers are its picture); its post text keeps the house notation
  for the same figures.
- **Factorials and permutations.** On screen, write the product in the film's order
  ("52 × 51 × 50 × … × 2 × 1"); in post text, "52 factorial". Never "52!": it reads as an exclamation
  mark (`brand.md` §5). The arrangements of a dealt deck are **"possible deals"**, on screen and in post
  text, never "orders".

---

## 6. Sound

The score MUST be generated from the same timeline as the picture (the film's story file), so it is
always in sync. Its palette MAY be the toolkit's (§8.5) or the film's own.

| Layer | Rule |
|---|---|
| Master | **−14 to −16 LUFS integrated, true peak ≤ −1.0 dBTP, measured on the final MP4** (ffmpeg `ebur128=peak=true`). Master the WAV to about −1.5 dBFS sample peak, or measure the WAV with `ebur128` and gain it to about −15 LUFS before the limiter, so the AAC encode lands inside the range |
| Loudest moments | The payoff and the final stop are the film's most marked moments, however the film marks them (a boom and a chord, a silence and a single knock) |
| Loop | By the last frame the score is back at its frame-0 state, so the sound loops with the picture |
| Phone | Every event SHOULD still be audible on a phone speaker, which reproduces little below ~300 Hz; nothing load-bearing lives only below that |

---

## 7. Definition of done (check every film against this)

- [ ] 1080×1920, 60 fps. Frame count = `round(total × 60)` ± 1 (slice rounding), checked with
      `ffprobe -count_frames`. With-sound and silent versions exported.
- [ ] Hook fully on screen within the first second or so and fully gone before the film starts answering.
      The hook frame is identified for the cover and carries nothing of the answer.
- [ ] No text baseline below y = −4.8 and nothing in the right button column. Text within the 6.8-unit band.
- [ ] No text-on-text or text-on-device collisions at any moment. **Stills first:** check them at every
      entry and exit and mid-hold before any full render, and script the check so it runs on the film's own
      layout.
- [ ] Every number traceable to the film's "The numbers" table and rounded per §5. The figure a device
      reaches equals the figure in the line on screen. The answer to the hook is absent from the cover and
      every post text (a scripted check where possible).
- [ ] No glow with a square edge. No black spikes on text halos.
- [ ] Loudness −14 to −16 LUFS and true peak ≤ −1 dBTP on the final MP4.
- [ ] Last frame ≈ first frame (loop), measured.
- [ ] The film's README lists what it keeps and what it breaks from §8, with the reason for each break.

---

# Part B. The Doubling's toolkit (the Scale pillar's default)

## 8. The Doubling's toolkit (the Scale pillar's default)

These are The Doubling's structure and pacing, its devices, its label block and text animation, and its
sound palette. They are the **default for the Scale pillar** (`brand.md` §7): a Scale film is expected to
reuse them, so the series stays recognisable from film to film. Any other film MAY reuse them in whole, in
part or not at all. A device that is reused keeps its rules as written here; a device that is changed is a
new device, described in the film's README. The geometry and code are in `doubling_film/film.py`,
`visuals.py` and `soundtrack.py`.

**Declaring what a film keeps and breaks.** Each film's README has a section that lists, rule by rule
and device by device, what the film keeps from this part and what it breaks, with one line of reason for
each break (*The Long Number*'s README is the pattern). The general aesthetics of Part A are never on that
list: they are met, not broken. Where a break here shows that a Part A rule quietly presupposed a device,
the rule is rewritten in Part A in the same pull request, not waived for one film.

Units in this part: "steps" are the film's counted unit (doublings in The Doubling). `n` is the current
position on that scale, which may be fractional; a stop sits at `n_i`.

### 8.1 Structure and pacing

1. **Hook (0–2.75 s; the first step starts at 2.75 s, `HOOK` in `story.py`).**
   - The glyphs rise in from frame 0, 0.32 s each with a 0.011 s stagger, so the whole question is in by ≈ 0.8 s.
   - The sub-line enters at ≈ 0.85 s.
   - Both leave from 2.1 s and are fully gone before the HUD fades in (2.5–2.95 s).
   - A 4-line hook (§8.4) MAY take 0.35 s more: the first step at 3.1 s, the question leaving from 2.45 s, the
     HUD fading in at 2.9–3.35 s, the start label leaving at ≈ 3.2 s and the mechanic's label entering at
     ≈ 3.5 s, so the four lines are fully on screen for ≈ 1.6 s (the archived Manim build of *The Shuffle*).
   - The starting object (e.g. the Planck dot with its dashed reticle) is visible underneath from frame 0.
   - The starting object is **not a stop**. Its label block rises in from ≈ 0.45 s at the usual
     positions (§8.4): size text in its accent with **no STEP chip**, the title and one line. It has no
     measuring row. It starts leaving at ≈ 3.3 s, just before the mechanic's phase label (3.45 s), so
     it is part of the hook frame and the cover. The HUD shows 0 when it fades in. When the start is
     you, the YOU pin fades in with the HUD (2.5–2.95 s), never under the hook.
   - The hook frame is at 1.9 s in The Doubling.
2. **Teach the mechanic (~3 s).** The Doubling counts its first 8 doublings one at a time: the first step
   takes 0.40 s, tapering to 0.25 s, each followed by a 0.06 s settle. Each step gets a ×2 pop, the digit
   change and a pluck, all at the same moment. A phase label explains it ("DOUBLE IT. AGAIN." / "Every step
   makes it twice as long.").
3. **Journey.** A sequence of **stops** with **holds**, joined by fast transitions (a warp; near-empty
   frames are intended there). A long stretch with no stop gets a phase label (e.g. "THE GREAT
   DESERT"), which is a label block without a STEP chip.
4. **Payoff.** The answer to the hook, plus one reframing line that makes it personal
   ("You're closer to a galaxy than to the Planck length."), with the meter brackets (§8.3).
5. **Outro (~4.2 s).**
   - The HUD counter flies to the centre (y ≈ +0.95, 2.15× larger, over 0.85 s), turning `#FFD1F3`
     with a Magenta glow.
   - Under it come a unit label (e.g. "DOUBLINGS", Inter Medium 26, Mist Light, tracked +0.13 units,
     y = −0.78) and a one- or two-line summary (Inter Medium 34, Starlight, centred at y = −1.95).
   - Over the last ~1.2 s, everything glides back to the first frame so the video loops. The
     background, glow and particles return to their frame-0 state.

**Pacing (the toolkit's default)**
| Element | Timing |
|---|---|
| Hold at a stop | **~3.2 s**. Use up to ~3.8 s when a note follows, ~5.4 s for the payoff stop and ~4.6 s for the final stop |
| Transition between stops | `0.5 + 0.028 × (steps travelled)` seconds (≈ 0.7–1.1 s) |
| First sprint (end of teaching → first stop) | ≈ 2.5 s when it crosses a stretch with no stop (The Doubling's desert, steps 9–66). If the first stop follows the teaching directly, it is an ordinary transition (row above) |
| Labels | Enter 0.3 s **before** the object settles; exit when the next transition starts |
| Note (a second line that replaces the comparison) | Enters 1.2–2.5 s into the hold, after the comparison has been read |
| Comparison row | Starts 0.5 s after arrival and fills in **1.5 s** (ease-in-out). A row that belongs to a note starts as the note enters |
| Motion | Never fully stops. During a hold the camera keeps drifting slowly in the direction of travel: from `n_i − 0.15` to `n_i + 0.25` (when `n` falls: from `n_i + 0.15` to `n_i − 0.25`) |

### 8.2 Movement

- **Zoom by scale.** An object at stop `i` is drawn at `2^(n_i − n)` of its reference size (a
  diameter of 5.2 units, centred at y = +0.22).
  - It's fully visible from about 1/10 to about 1.5× its reference size (`n_i + 3.4` to `n_i − 0.55`)
    and fades to zero by 1/85 (`n_i + 6.4`) and by 3× (`n_i − 1.6`). The window is defined on `n`, so
    it is the same in both directions; only the order changes.
  - When growing (`n` rises): it fades in from 3× to 1.5× and fades out from 1/10 to 1/85.
  - When zooming in (`n` falls): it fades in from 1/85 to 1/10 (arriving as a point) and fades out
    from 1.5× to 3×.
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
  (radii ×2) move with the zoom. Both stay faint during holds and brighten during sprints. These faint
  full-circle rings are a film device, not the brand mark (`brand.md` §2).
- **Easing.** Use ease-in-out for counts and fills, ease-out for text entering, and smoothstep for fades.
  No linear starts or stops on visible motion, and motion never fully stops (§8.1).

### 8.3 The devices

| Device | Rule |
|---|---|
| **HUD counter** | Big number at top centre (JetBrains Mono Bold, size 104, y = +5.6, white `#FFFFFF`). Above it, a label at y = +6.55 (Inter Medium 19, CAPS, Mist, tracked +0.09 units) names the film's unit ("DOUBLINGS", "HALVINGS"). It fades in over 2.5–2.95 s, after the hook. While a stop holds, it shows that stop's step. On arrival it **punches** (+16% and back over 0.5 s) and flashes 85% toward the accent, decaying over ~0.45 s |
| **Size readout + ruler** | A live scientific readout ("1.7 × 10⁰ m"): JetBrains Mono Medium 30, `#9FB3D9`, y = +4.5, mantissa to one decimal, with the exponent drawn as raised 0.6× digits and a true minus. It stays live during holds, because it labels the ruler; the stop's nominal size is the label's size text. Below it, a ruler at y = +4.12 (`#5D6B8C`, 1.6 px) whose width equals the reference size (5.2 units) |
| **STEP chip** | At every stop after the start (§8.1), a pill ("STEP 116") left of the size text and above the title. JetBrains Mono Bold 22 in the accent, on an accent fill at 16% with a 1.6 px accent outline. **Step number = `round(|n_stop − n_start|)` on unrounded values**, in the film's own unit counted from its own start. The HUD shows the same number during the hold. Never subtract rounded steps taken from another film |
| **Measuring row** (acted-out comparison) | Copies of the **smaller of two neighbouring stops** line up across the larger one at its stop, or stack for heights, while a live counter climbs ("0 → 425 ants"). When the film grows, that's the previous stop; when it shrinks, it's the next one. The row takes the colour of the thing being counted (§3). Draw individual dots (radius 0.46 × spacing) when each copy is at least ~4 px (row length in px ÷ count); otherwise draw a glowing line (core 3.2 px, glow 10 px at 22%) with ~90 white beads (radius 0.022 units) and a small star at the growing end. The finished count MUST equal the number in the line on screen while it fills (the comparison, or the note at a final stop). The row fades out with its counter |
| **Live counter** | JetBrains Mono Bold 21, in the row colour mixed 40% toward white, with a black halo (8 px stroke, 78%, round joins so there are no spikes). The format is commas for counts, plus a noun ("4,500 solar systems"), and words for totals of ten million or more ("100 quintillion Planck lengths"). For horizontal rows it rides the middle of the filled part, 0.3 units above or below the row (further if captions are in the way), clamped to x ∈ [−3.4, +3.3]. For vertical rows it sits beside the growing end. It MUST NOT collide with the YOU pin, the meter or other text |
| **YOU pin** | A cyan capsule (0.9 × 0.4 units) reading "YOU" (Inter Display Bold 21, `#03121A`), with a stem and a small target ring, marking the viewer's position. Show it only while that position is inside the frame. Its opacity is set on `n`, so it works in both directions: full while the human is at most ~1.2× its reference size (`n ≥ n_human − 0.25`), fading to zero by ~1.5× (`n_human − 0.6`). So it appears as the human arrives when the film grows, and it leaves as the human grows from 1.2× to 1.5× when the film shrinks. The meter's YOU tag remains throughout |
| **Progress meter** | A **size axis** at the left edge: x = −3.95, from y = −2.45 (the smallest size shown) to y = +3.45 (the largest). The base line is `#27304A`, 2 px; ticks at each stop are in that stop's accent, at 30% opacity until passed. The fill runs from the start stop toward the current `n` (downward in a film that shrinks), as a gradient from the start accent to the final accent (lavender → cyan → magenta in The Doubling; it starts from the starting object's colour, cyan in a film that starts at you). A "0" in Mist sits at the start, and a "YOU" tag (Inter Display Bold 16, cyan) sits at the viewer's step. For the payoff, two brackets go to the right of the YOU tag (x = −3.21), mono bold 20: a cyan bracket from you to the answer, labelled with your distance to it ("69"), and a bracket in Mist with its label in Mist Light for the distance the reframing line compares it with ("116", the start to you, in The Doubling). Each spans its own stretch of the axis, above or below YOU as the film's direction puts it. They appear with the punchline and stay until 0.6 s into the next stop |
| **Answer ring** | When the payoff stop arrives, a thin ring in its accent grows around the HUD counter (1.4 → 5.0 units over 1.2 s, ease-out) and fades |
| **×2 / ÷2 pops** | While the mechanic is taught, "×2" (Inter Display Bold 40, lavender) pops beside the counter at (+1.55, +5.62): +25% scale, then fading over ~0.3 s. It happens together with the digit change and the pluck |
| **Outro counter flight** | The HUD counter's flight to the centre with the Magenta glow, the unit label and the summary (§8.1, outro) |

### 8.4 Label block and text animation

| Role | Font | Size | Notes |
|---|---|---|---|
| Hook question | Inter Display Bold | 50 | 3 short lines, block centred at y = +4.75 with a 0.16 line gap. A question that needs 4 lines MAY use size 42, spaced 1.04 em baseline to baseline (place lines by baseline, not by stacking ink boxes, or the spacing turns uneven), block centred at y = +4.75, with the glyph stagger shortened (0.0085 s for 58 glyphs) so it is in by ≈ 0.8 s; its top then sits at ≈ +5.93, inside the cover crop (§1), and the hook MAY run to 3.1 s (§8.1) |
| Hook sub-line | Inter Regular | 27 | "Guess before it ends." in Mist, at y = +3.3 |
| Stop / phase title | Inter Display Bold, CAPS | 60, fit to ≤ 6.8 units | "THE MILKY WAY" |
| Comparison / note line | Inter Medium | 31 (29 when two lines) | Sentence case, Mist Light. The `[bracketed]` number is in the accent. Fit down to ≤ 6.8 units, never below 26; if it still doesn't fit, shorten the line. Only notes may use two lines |
| STEP chip | JetBrains Mono Bold | 22 | See §8.3 |
| Size text | JetBrains Mono Medium | 25 | In Mist Light beside a chip, or in the accent when there's no chip (phase labels and the start). If it contains Unicode superscript exponents ("1.6 × 10⁻³⁵ m"), set it in Inter Medium 27, because JetBrains Mono has no superscript minus |
| Small captions inside drawings | Inter | 17–20 | "Sun", "Proxima Centauri", planet names. Mist Light |
| HUD label | Inter Medium, CAPS | 19 | Tracked **+0.09 units per glyph** (≈ +33% of the em) |

**Label block positions** (line centres, y in units):
- STEP chip and size text at −2.86.
- Title at −3.51.
- Comparison (first line) at −4.21.

A two-line note keeps its first line where a single line sits and grows downward. Its last baseline
MUST stay at or above y = −4.8.

**Text animation.** Glyphs rise in one at a time: rise 0.22 units over 0.42 s, stagger 0.014 s
(titles 0.018 s, comparisons 0.008 s). Exit is a short lift (0.16) and fade over 0.28 s.

### 8.5 Sound palette

The score is generated from the same timeline as the picture (`doubling_film/soundtrack.py`).

| Layer | Rule |
|---|---|
| Hook | A soft sub bloom (70 → 40 Hz) with an A4 bell at frame 0, then a 1.4 s filtered-noise riser into the first step |
| Tick | One soft tick per step. It turns into a buzz in sprints, with gain falling as the rate rises |
| Plucks | A rising pentatonic pluck (E4 G4 A4 C5 D5 E5 G5 A5) on each step of the mechanic phase |
| Whoosh | Filtered noise that follows the zoom speed. Keep it under the bells, never harsh |
| Stop bell | A bell on each regular stop's arrival, climbing an A-minor pentatonic scale, over a small sub thump (120 → 60 Hz) |
| Big reveals | No bell. Instead, a sub-bass boom (95 → 34 Hz) plus a chord. "You" is A major (A3 E4 A4 C♯5), the payoff is A minor add9 (A2 E3 A3 C4 E4 B4) and the finale (the film's final stop, whatever its subject; the universe in The Doubling) is F major 7/9 (F2 C3 A3 E4 G4 C5). If the payoff stop is also the final stop, it takes the payoff chord. These are the loudest moments |
| Answer shimmer | Four quiet high bells (E5 A5 B5 E6) as the payoff line lands |
| Counter ratchet | ~22 soft clicks (1.5 → 3.2 kHz) that follow the row's easing, plus a quiet E6 chime when it lands |
| Outro | A filtered swell into the counter's flight, then a boom and an A-minor chord (A2 E3 B3 C4 E4 A4) |
| Pad | A quiet A-minor pad whose brightness follows size, so it darkens as things shrink. Thinner in empty stretches |
| Reverb | Convolution reverb on the mix. The master targets are in §6 |

In a film that shrinks, the plucks and stop bells still rise with **progress**, while the pad follows
**size**. When the film starts at you, the frame-0 hook layer is the "you" moment: the soft bloom
(70 → 40 Hz) stays, the A4 bell becomes the "You" chord (A3 E4 A4 C♯5) at the bell's level, and
there is no 95 → 34 Hz boom and no later "you" reveal.

### 8.6 Toolkit checks (in addition to §7)

- [ ] STEP chip = HUD counter during each hold.
- [ ] Each measuring row's final count = the number in the line on screen while it fills.
- [ ] The live counter, the YOU pin and the meter collide with nothing. Check stills at arrival, mid-row,
      row end and late hold.
- [ ] The meter's brackets agree with the reframing line.

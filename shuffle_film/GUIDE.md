# The Shuffle: a guide for the agent who builds it

> **The film has since been built** (October 2026). [`README.md`](README.md) is the current treatment and
> numbers table, and the playbook's *Films that count* section holds the rules that were agreed. This guide is
> kept as the record of the exploration that preceded it. Figures and notation here pre-date the decisions in
> the README (for example "at most 1 in 10 million", now "about", and "52!", now written out); the README wins
> where they differ.

> **These are suggestions, not rules.** This guide collects what one round of exploration with
> the channel's owner found in October 2026: an idea, checked numbers, a possible shape, layouts
> that worked, and mistakes worth not repeating. Take what helps, change what doesn't, and ask the
> owner about anything that matters.
>
> The repo's own instructions (`AGENTS.md`, `playbook/`) still apply as they are written. Where an
> idea here would bend the playbook, that is a question for the owner, not an instruction from this
> file.

## The idea

A film for the **Pure Patterns** pillar about how many ways a deck of 52 cards can be ordered.

- **The mechanic.** Deal cards one at a time from a full, shuffled deck. The first card can be any
  of 52, the next any of the 51 left, then 50, and so on. Each card multiplies the possible orders.
- **The question.** How many cards must you deal before the possible orders outnumber the seconds
  since the Big Bang?
- **The answer** is surprisingly small: **11**.
- **The full deck.** All 52 cards give 52 factorial ≈ 8.07 × 10⁶⁷ orders.

It fits the channel's pattern: one honest question, answered on screen, and a number to guess
before the answer arrives.

## The direction the owner chose last

The owner reviewed several versions and settled, for now, on:

- **A short cut, about 30 s**, shorter than the playbook's 45–75 s default.
- **The dealing count, answer 11.**
- **An "imagine" ending.** After the full deck, imagine shuffling on cosmic scales: each attempt
  still sees almost none of the orders.
- **A simple ending line**, such as "Every good shuffle is almost certainly new."

Treat this as the latest preference, not a fixed brief. Versions that were tried and set aside, in
case they are useful:

- **A deck that grows card by card (answer 20).** A deck of *k* cards has *k* factorial orders, and
  20 cards beat the seconds since the Big Bang.
  - For: it is unambiguous and teaches "each new card has one more place to go".
  - Set aside: the owner preferred 11 and a shorter film.
- **A longer cut (about 44–51 s)** with more stops (atoms in your body, Earth, the Sun, the Planck
  time). It was set aside for length.

## The maths, and the trap in it

- **Dealing** *k* cards from 52: 52 × 51 × … × (53 − *k*) = 52! / (52 − *k*)! possible orders.
- **A deck of only** *k* **cards:** *k*! orders.

These are easy to confuse.

| Count | 11 cards | Beats 4.35 × 10¹⁷ s? |
|---|---|---|
| Dealt from a full deck | 2.41 × 10¹⁸ | yes, at card 11 |
| A deck of only 11 cards | 11! ≈ 4.0 × 10⁷ | no; needs 20 cards (20! ≈ 2.43 × 10¹⁸) |

A viewer who works out 11 × 10 × … × 1 will say the film is wrong unless the wording makes the
dealing clear. Some ways to keep it clear:

- put "dealt" in the question;
- show that the deck is full;
- show the multipliers falling: ×52, ×51, ×50.

While we were checking, even a web-search summary slipped on this.

## Numbers to start from (re-check before using)

These were computed in October 2026 from the sources at the end. Re-verify anything before it goes
on screen.

### The dealing count

| Card | Orders (52 × 51 × …) | Reference | × ref at the card | × ref one card earlier |
|---|---|---|---|---|
| 1, 2, 3 | 52 · 2,652 · 132,600 | (teaching) | — | — |
| 6 | 1.47 × 10¹⁰ | 8.3 billion people alive (UN WPP 2024, 2026 projection) | 1.77 | 0.038 |
| 7 | 6.74 × 10¹¹ | Milky Way stars, 100–400 billion (a range) | 1.69 (vs 400 bn) | 0.15 (vs 100 bn) |
| **11** | **2.41 × 10¹⁸** | **4.35 × 10¹⁷ s since the Big Bang** (13.787 Gyr, Planck 2018) | **5.54** | **0.13** |
| 15 | 5.86 × 10²⁴ | a tablespoon of water, 5.0 × 10²³ molecules (15 mL) | 11.7 | 0.31 |
| 15 | 5.86 × 10²⁴ | stars in the observable universe, 10²²–10²⁴ (a range) | 5.9 (vs 10²⁴) | inside the range |
| 33 | 6.63 × 10⁵⁰ | atoms in Earth, ≈ 1.33 × 10⁵⁰ | 4.99 | 0.25 |
| 42 | 2.22 × 10⁶¹ | Planck times since the Big Bang, 8.07 × 10⁶⁰ | 2.75 | 0.25 |
| 43 | 2.22 × 10⁶² | Planck lengths across the observable universe, 5.44 × 10⁶¹ (The Doubling's figure) | 4.08 | 0.41 |
| 52 | 8.07 × 10⁶⁷ | 52 factorial, exactly 80,658,175,170,943,878,571,660,636,856,403,766,975,289,505,440,883,277,824,000,000,000,000 | — | — |

Two references sit too close to their threshold to claim cleanly:

- **Atoms in a 70 kg body (≈ 7 × 10²⁷):** card 17 beats it by only × 1.12.
- **Atoms in the Sun (≈ 10⁵⁷):** card 38 lands at × 0.97, almost exactly on it.

One idea that served well was to keep a stop only if it beats its reference by about 20% or more,
and the answer only if the card before falls short by a similar margin.

### The "imagine" beats

| Imagine | Shuffles | Orders per shuffle (52! ÷ shuffles) | As a line |
|---|---|---|---|
| One shuffle a second since the Big Bang | 4.35 × 10¹⁷ | 1.854 × 10⁵⁰ | at most 1 in 1.85 × 10⁵⁰ of the orders |
| Everyone alive (8.3 billion) doing that | 3.61 × 10²⁷ | 2.234 × 10⁴⁰ | at most 1 in 2.23 × 10⁴⁰ |
| One shuffle every Planck time since the Big Bang | 8.07 × 10⁶⁰ | 9.995 × 10⁶ | still at most 1 in 10 million |

"At most" is honest because S shuffles can show at most S different orders.

### The growing-deck alternative (*k*!), if it comes back

The stops it would have:

| Card | Passes |
|---|---|
| 14 | people alive |
| 15 | Milky Way stars, across the whole range |
| **20** | **seconds since the Big Bang** |
| 24 | a tablespoon of water (× 1.24, thin) |
| 27 | atoms in a 70 kg body (× 1.56) |
| 42 | atoms in Earth |
| 46 | atoms in the Sun |
| 48 | Planck times since the Big Bang |

## A possible shape (about 32 s)

One timing that read well in an animatic, using the playbook's pacing as a base. Adjust freely.

| ≈ Time | Beat | Idea |
|---|---|---|
| 0–2.75 s | Hook | The question over a full deck; the sub-line "Guess before it ends." |
| 2.75–4.7 s | Teach | Deal cards 1–3 slowly: ×52, ×51, ×50 and a readout of 52 → 2,652 → 132,600 orders |
| 5.3 s | Card 6 | More orders than everyone alive (a natural place for the YOU pin) |
| 9.1 s | **Card 11** | The answer. A reframing line such as "Not even a quarter of the deck." (11 of 52) |
| 14.5–17 s | Sprint | Cards 12–52. "The last card has no choice." (×1) |
| 17 s | Card 52 | The whole deck, 8.07 × 10⁶⁷, then a setup line: "Imagine shuffling since the Big Bang." |
| 20 / 22.6 / 25.2 s | Three beats | The table above, about 2.6 s each (3 s for the last) |
| 28.2–32.4 s | Outro | The final count, a summary line, and a glide back to the first frame for the loop |

- **Possibly tight:** 2.6 s per beat may be short for reading "At most 1 in 1.85 × 10⁵⁰ of the
  orders". Another second across the beats would help.
- **Possibly redundant:** the readout already shows the count, so a beat's line may not need to
  repeat it.

## Wording that fitted

Widths were measured on the repo's 9 × 16-unit frame (`playbook/film-style.md` §1), where centred
text should stay within 6.8 units.

- **Hook (4 lines):** "How many cards dealt / have more orders than / seconds since / the Big Bang?"
  The widest line is 5.98 units at size 42. Putting "seconds since the Big Bang?" on one line runs to
  7.7 units, too wide.
- **Labels:**
  - "52 cards / A FULL DECK / Deal it one card at a time."
  - "DEAL A CARD / Each card has one fewer choice."
  - "EVERYONE ALIVE / More orders than people alive"
  - "SINCE THE BIG BANG / More orders than every second", then "Not even a quarter of the deck."
  - "KEEP DEALING / The last card has no choice."
  - "THE WHOLE DECK / 52 × 51 × 50 × … × 2 × 1"
- **Beat titles:** "A SHUFFLE A SECOND", "EVERYONE, EVERY SECOND", "EVERY PLANCK TIME". Size text in
  the form "4.35 × 10¹⁷ shuffles".
- **Too long for a beat's size text:** "4.35 × 10¹⁷ shuffles since the Big Bang". Stating "since the
  Big Bang" once, in the setup line before the beats, kept the beats short.
- **Shorts title ideas** that keep the answer out: "Deal from a full deck: how many cards beat the
  Big Bang?" · "Cards dealt against the age of the universe".

## Visual ideas that worked in mock-ups

- **The deck as hero.**
  - It starts as a full deck: a lavender line-art card over a few stepped edges.
  - Dealt cards fill a **ring of 52 card slots** around the centre (radius ≈ 2.55 units at y = +0.22,
    slots ≈ 0.19 × 0.27), clockwise from the top.
  - A radius of 2.7 ran into the label block's chip.
- **No zoom by scale.** The Doubling zooms because its stops are sizes. Counts aren't sizes, and the
  references don't come in size order, so a cross-fade or a "gather from points of light" arrival
  read better.
- **A pass row.** A horizontal track sits under the count readout, where The Doubling's ruler was.
  The reference fills first in its colour, with a live counter ("8.3 billion people"). Then the
  orders overtake it in lavender.
- **A copies row.** When the reference would fill a tiny fraction of the track, as in the beats,
  copies of the reference line up across the orders instead: "1.85 × 10⁵⁰ per shuffle".
- **A meter on a log₂ axis** of the orders. At the payoff, brackets read 11 (cards to the answer) and
  41 (cards left).
- **Colours tried:**
  - lime `#D4FF5C` for time and the Big Bang;
  - violet `#E08BFF` for the full deck.

  The playbook's existing ones were kept: lavender for the start and mechanic, cyan for you and
  people, and magenta for the outro's glow.
- **Stand-in drawings** that read clearly enough in stills:
  - a lime spiral of time from a bright point (the Big Bang);
  - a disc of cyan points (everyone alive);
  - the Planck dot in a dashed reticle;
  - a faint lavender dust of points for "the orders".

## Things that went wrong, and what fixed them

- **The 4-line hook broke the cover crop.** At size 46, centred at y = +5.0, its top reached +6.32,
  outside Instagram's central 3:4 crop (+6.0), which the playbook requires for covers. Size 42,
  spaced 1.04 em baseline to baseline and centred at +4.78, put the top at +5.96.
- **Line spacing went uneven.** Stacking lines by their ink boxes packs lines without descenders
  closer together. Placing each line by its baseline fixed it.
- **The hook landed late.** The 58-glyph hook took 0.83 s to rise in at the reference stagger. A
  stagger of 0.0085 s brought it back to about 0.8 s.
- **A counter ran into a bracket.** A live counter clamped at x ≥ −3.4 ran into the payoff bracket's
  line at x = −3.21. Starting counters no further left than their track fixed it. A text-only overlap
  check missed this, because the bracket is a line, so it's worth checking lines too.
- **The readout and size text disagreed.** A one-decimal readout ("8.1 × 10⁶⁷") sat over a size text
  reading "8.07 × 10⁶⁷". Three significant figures made them agree.
- **A counter ended on the wrong number.** A copies-row counter ended on "9,994,565" while its line
  said "1 in 10 million". Ending counters on the line's rounded number fixed it.
- **Number words run out.** Words like quintillion stop being useful above about 10²¹. Scientific
  notation worked from there.
- **Counters dropped decimals.** "8 billion people" against a size text of "8.3 billion". Keeping
  one decimal below ten of a unit fixed it.
- **A note ran close to the caption bar.** A two-line note at size 29 put its last baseline at
  −4.796, against the playbook's −4.8 limit. That leaves almost no room to lengthen it.

## Honesty notes worth keeping

- **Keep "dealt" wherever 11 appears,** including post text (see the maths above).
- **"A good shuffle"** means well mixed: about seven riffle shuffles (Bayer & Diaconis 1992). A deck
  fresh from the box is always in the same order.
- **52 factorial is about 10¹² times fewer** than the atoms in the observable universe (about
  10⁸⁰). Viral posts often get this backwards.
- **Watch for a wrong value.** One online summary quotes 52! as 8.65 × 10⁶⁷; the right figure is
  8.07 × 10⁶⁷.
- **The Planck time** is "where known physics runs out", not "the shortest possible time".
- **"52!" reads as an exclamation mark,** and the brand voice avoids those. "52 factorial", or the
  written-out product, avoids the clash.
- **The brand guide keeps the hook's answer (11) out** of covers, titles, captions, descriptions,
  tags and comments.

## How the playbook fits, and where a count film strains it

The playbook was written around The Doubling, and a few of its rules assume sizes, not counts:

- zoom by scale;
- the STEP chip and HUD in doublings;
- a 3-line hook;
- a 45–75 s length;
- one decimal on the readout.

In October 2026 a "films that count" variant was drafted into the playbook and later withdrawn, so
the owner could start fresh. If the build needs rules like these, it's worth agreeing them with the
owner and writing them into the playbook in the same pull request, as `playbook/README.md` suggests.

## Tooling notes

- **The reference film,** `doubling_film/`, has a reusable engine you could use directly:
  - `engine.py`: a fast camera with screen-blended glow sprites, mono counters (`MonoFont`,
    `MonoLine`, `sci_spec`), per-glyph text animation (`AnimText`) and `lines`;
  - `visuals.py` (`sprite`) and `assets.py` (procedural textures: glow, star, ring, disc, sun, galaxy,
    cosmic web);
  - a parallel render script (`render.sh`).

  Importing it with `sys.path` worked for stills, but a film named `story.py` would clash with The
  Doubling's own `story.py`, so copying or renaming may be cleaner.
- **Manim Community 0.22** worked. It needs `libcairo2-dev`, `libpango1.0-dev` and `pkg-config`, plus
  the fonts Inter 4 (with Inter Display) and JetBrains Mono.
- **Speed:** a single 1080×1920 still renders in a few seconds.
- **`t2c` gotcha:** Manim's colour mapping matches plain substrings, so a key like "Big Bang?" colours
  the question mark too.
- **Stills first.** Rendering single frames was a cheap way to check text widths, the cover crop and
  collisions before animating. Moments worth checking at each stop are arrival, mid-row, row end and
  late in the hold (`playbook/film-style.md` §8).
- **An animatic.** A timed storyboard of held key frames, with crossfades, a burned-in timecode and
  a click track built from the same timeline, took about 40 s to build. It was useful for judging
  pacing.

## A workflow that worked

1. A treatment with the question, the numbers table and a timeline.
2. Stills for the riskiest frames: the hook (also the cover), the payoff and the final stop.
3. An animatic for pacing.
4. The full build, with a timeline function the picture and the sound both read, as The Doubling
   does.
5. The playbook's definition-of-done checks (`playbook/film-style.md` §8).

The owner liked seeing the upsides and downsides of each choice, being asked before long renders,
and having anything that changed the answer or the hook confirmed first.

## Sources

- Bayer, D. & Diaconis, P. (1992). "Trailing the dovetail shuffle to its lair". *Annals of Applied
  Probability* 2(2). Summary: [Harvey Mudd Math Fun Facts, "Seven Shuffles"](https://math.hmc.edu/funfacts/seven-shuffles/).
- Planck Collaboration (2018). "Planck 2018 results. VI. Cosmological parameters" (age 13.787 ± 0.020 Gyr).
- CODATA 2018 recommended values (Planck time 5.391247 × 10⁻⁴⁴ s). SI 2019 (Avogadro constant).
- UN DESA, *World Population Prospects 2024*. The 2026 figure was seen via aggregators
  ([StatisticsTimes](https://www.statisticstimes.com/m/demographics/world-population.php)); worth
  confirming on the UN site.
- Milky Way stars, 100–400 billion: [Space.com](https://www.space.com/25959-how-many-stars-are-in-the-milky-way.html), [NASA Goddard](https://asd.gsfc.nasa.gov/blueshift/?p=7791).
- Atoms in Earth: [Jefferson Lab Science Education](https://education.jlab.org/qa/mathatom_05.html).
- Existing coverage of the topic: TED-Ed's "every shuffle is unique" lesson, Vsauce's "Math Magic",
  [Puzzlewocky](https://puzzlewocky.com/brain-teasers/probability-puzzles/every-shuffle-of-a-deck-of-cards-is-probably-unique-in-history/),
  [Mental Floss](https://www.mentalfloss.com/article/92408/how-many-ways-can-you-sort-deck-cards).
  Most frame it around the full deck. The "how many cards beat the Big Bang" threshold did not turn
  up in our searches.

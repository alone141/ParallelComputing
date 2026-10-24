# The Shuffle

> **Status: treatment.** Nothing here is built or rendered yet. Style rules for all Lumiera films are
> in [`../playbook/film-style.md`](../playbook/film-style.md); this film follows them, using the
> count-film variants in §4 (*Films that count*). Re-check every source in "The numbers" before
> scripting (`playbook/ideas.md`).

A ~44-second vertical (9:16) film for the **Pure Patterns** pillar. It starts with one card and adds
cards one at a time. A deck of 2 cards can lie in 2 orders, 3 cards in 6, 4 cards in 24: each new
card has one more place to go, so it multiplies the orders by the deck's new size. The count passes
the people alive at **14** cards, the seconds since the Big Bang at **20**, the atoms in Earth at
**42**, and a full deck of 52 reaches about 8.07 × 10⁶⁷ orders.

| | |
|---|---|
| Format | 1080×1920, 60 fps, with-sound and silent versions (`film-style.md` §1) |
| Length | ≈ 43.6 s. That is under the 45 s default on purpose: the two weakest stops were cut (see [Decided](#decided)), and holds may not be padded (`film-style.md` §1) |
| Hook | "How many cards have more orders than seconds since the Big Bang? Guess before it ends." |
| Answer | 20 cards. **Never** in the cover, title, caption, description, tags or comments (`brand.md` §6) |
| Payoff | "Not even half a deck." (20 of 52 cards) |
| Outro | "52" CARDS · "Every good shuffle / is almost certainly new." |
| Mechanic | A deck of *k* cards has 1 × 2 × 3 × … × *k* orders (*k* factorial) |

## Why this film

- **The question is honest and answerable in your head.** We expect many guesses near a full
  deck. The true answer, 20, lands well before that, and 19 cards fall clearly short (1.2 × 10¹⁷ against
  4.35 × 10¹⁷), so the answer does not hang on a rounding.
- **It teaches why.** The film shows the *one more place* rule, so the viewer sees why the count
  explodes, not just that it does. That is the Patterns pillar's promise.
- **It reuses the toolkit.** The ×*k* pops play the role of The Doubling's ×2, and it keeps the HUD,
  the STEP chip (as CARD), the YOU pin, the meter, the payoff brackets, the answer ring and the loop.
- **It links to The Doubling.** A full deck's orders are about 226 doublings, against the 205 that
  span the observable universe. That makes a good fact for the description and comments.

Risks are listed under [Open decisions](#open-decisions).

## Structure

Times are from the pacing rules in `film-style.md` §2 and are approximate until `story.py` exists.

| ≈ Time | Beat | On screen |
|---|---|---|
| 0–2.75 s | **Hook** | One glowing card (lavender) at the centre. The question rises in over 4 lines (size 46, block centred at y = +5.0, glyph stagger ≈ 0.009 s so all 53 glyphs are in by ≈ 0.8 s): "How many cards / have more [orders] / than seconds since / the [Big Bang?]" ("orders" lavender, "Big Bang?" cyan). Sub-line "Guess before it ends." The start label rises from ≈ 0.45 s: size text "1 order", title ONE CARD, line "It can only lie [one] way." HUD fades in at 2.5–2.95 s showing **1** |
| 2.75–5.6 s | **Teach the mechanic** | Cards 2 to 6, one at a time (0.70, 0.60, 0.46, 0.40, 0.36 s, each with a 0.06 s settle). The new card hovers while its *k* possible places glow (before, between and after the cards), then slides into one. "×2", "×3" … "×6" pops beside the HUD, together with the digit change and a pluck. The readout goes 2 → 6 → 24 → 120 → 720 orders. For cards 2 and 3, the 2 and then 6 possible orders flash as tiny ghost rows, so the count is seen, not just read. Phase label: size text "1 × 2 × 3 × 4 × …", title ADD A CARD, line "Each new card has [one more] place." (fit 28.9) |
| 5.6–7.2 s | Sprint | Cards 7–13. The row of cards rises into a **ring of 52 slots** around the centre, filling one slot per card. The readout races 5,040 → 6.2 × 10⁹ |
| 7.2 s | Stop · card 14 | EVERYONE ALIVE, with the YOU pin on Earth's night side |
| 11.0 s | **Payoff** · card 20 | SINCE THE BIG BANG. Answer ring around the HUD, then the note "Not even [half] a deck." with the meter brackets |
| 17.1 s | Stop · card 27 | YOU (the atoms in you), YOU pin on the pictogram |
| 20.3–22.8 s | Long stretch | Cards 28–41 in a ≈ 2.5 s sprint. Phase label: size text "CARDS 28 → 41", title KEEP ADDING, line "Each card multiplies by its [number]." (fit 29.4) |
| 22.8 s | Stop · card 42 | EARTH |
| 26.7 s | Stop · card 46 | THE SUN |
| 30.4 s | Stop · card 48 | THE PLANCK TIME (with a note) |
| 34.8 s | **Final stop** · card 52 | THE WHOLE DECK. The ring closes |
| 39.4–43.6 s | **Outro** | The HUD "52" flies to the centre (Magenta glow). Label CARDS, summary "Every good shuffle / is almost certainly new." The ring gathers back into one card, and the frame glides back to frame 0 |

## The stops

Each stop has the label block of `film-style.md` §5: chip, size text, title, line, and sometimes a
note. All lines are measured at their font size against the 6.8-unit limit. Where one needs fitting,
the size it fits at is given (never below 26).

| Card | Chip | Size text | Title | Line (and note) | Accent | Row |
|---|---|---|---|---|---|---|
| 1 | none (the start) | 1 order | ONE CARD | It can only lie [one] way. | lavender `#B9A6FF` | none |
| 14 | CARD 14 | 8.3 billion people | EVERYONE ALIVE (fit 58.9) | More orders than [people alive] | cyan `#6CF2FF` (people) | pass row: 0 → 8.3 billion people |
| 20 | CARD 20 | 4.35 × 10¹⁷ seconds | SINCE THE BIG BANG (fit 48.8) | More orders than [every second]. Note at ≈ 2.5 s: "Not even [half] a deck." | time `#D4FF5C` (new) | pass row: 0 → 4.35 × 10¹⁷ seconds |
| 27 | CARD 27 | ≈ 7 × 10²⁷ atoms | YOU | More than the [atoms] in you | cyan `#6CF2FF` | pass row: 0 → 7 × 10²⁷ atoms |
| 42 | CARD 42 | ≈ 1.3 × 10⁵⁰ atoms | EARTH | More orders than its [atoms] | Earth `#4DA3FF` | pass row: 0 → 1.3 × 10⁵⁰ atoms |
| 46 | CARD 46 | ≈ 10⁵⁷ atoms | THE SUN | More orders than its [atoms] | Sun `#FFB13B` | pass row: 0 → 10⁵⁷ atoms |
| 48 | CARD 48 | 8.07 × 10⁶⁰ since the Big Bang | THE PLANCK TIME (fit 55.5) | More than the [Planck times] so far (fit 30.6). Note at ≈ 1.8 s: "Where known physics [runs out]." | Planck `#B9A6FF` | pass row: 0 → 8.07 × 10⁶⁰ Planck times |
| 52 | CARD 52 | 8.07 × 10⁶⁷ orders | THE WHOLE DECK (fit 56.0) | 1 × 2 × 3 × … × 52. Note at ≈ 1.3 s: "[10 million] orders for every / Planck time since the Big Bang." | cards `#E08BFF` (new) | copies row with the note: 0 → 10 million |

- **Hue spacing.** The two new accents are at least 30° of hue from their neighbouring stops:
  - time (76°) sits between two cyan stops (186°), 110° from each;
  - cards (284°) sits 31° from the Planck lavender (253°) and 32° from the outro's magenta (316°).
- **Two YOU moments.** The pin is shown at card 14 (you among everyone alive) and card 27 (the atoms
  in you).
- **Payoff meter.** At card 20 the meter brackets read **20** (cyan, start to answer) and **32**
  (Mist Light, answer to the full deck).

## How it looks

- **The hero is the deck.**
  - During the teaching it is a short row of cards at the centre, with the possible places drawn as
    glowing slots.
  - From card 7 the row rises into a ring of 52 card slots (radius ≈ 2.7 units, centred at
    y = +0.22), which fills one slot per card. Its bottom (y ≈ −2.48) stays above the label block,
    and its left edge (x ≈ −2.7) stays clear of the meter.
- **Inside the ring:**
  - Each stop's reference is drawn in light: Earth's night side with city lights, a cosmic
    timeline spiralling out from a bright point, the people pictogram, Earth, the Sun, and the Planck dot with
    its dashed reticle (an echo of The Doubling's start).
  - The objects do **not** zoom by scale. Counts are not sizes, and the Planck time after the Sun
    would read as a lie on a size zoom.
  - Each object gathers from points of light as it arrives and scatters into the warp as it leaves.
- **The orders** are a faint lavender cloud of points behind the reference. They brighten and swirl
  with the rate of growth, and the warp streaks follow the same rate.
- **Background:** near-black. It shifts to blue near Earth, warm near the Sun and violet at the
  Planck time and the full deck, always within `film-style.md` §3.

## Sound

The palette is The Doubling's (`film-style.md` §7):
- **Plucks:** cards 2–6 get the rising pluck.
- **Ticks:** one tick per card in the sprints.
- **Stop bells:** each regular stop gets a bell.
- **Big reveals:**
  - the "You" chord at card 27 (YOU);
  - the payoff chord and answer shimmer at card 20;
  - the finale chord at card 52.
- **Pad:** its brightness follows the count (log₂ of the orders) rather than size.

## The numbers

A deck of *k* cards has *k* factorial orders. The orders of 52 cards are exactly
80,658,175,170,943,878,571,660,636,856,403,766,975,289,505,440,883,277,824,000,000,000,000
≈ 8.07 × 10⁶⁷. A stop sits at the first card whose orders beat the reference.

- **The ×above column** is the margin at the stop: the orders at the stop divided by the reference.
- **The ×below column** is the same ratio one card earlier: the orders one card earlier divided by
  the reference.

| Card | Orders | Reference used | Source | × above | × below |
|---|---|---|---|---|---|
| 14 | 8.72 × 10¹⁰ | 8.3 billion people alive (mid-2026) | UN DESA, *World Population Prospects 2024* (projection for 2026) | 10.5 | 0.75 |
| **20** | **2.43 × 10¹⁸** | **4.35 × 10¹⁷ s since the Big Bang** (13.787 Gyr × 3.156 × 10⁷ s per Julian year) | Planck Collaboration 2018, VI (13.787 ± 0.020 Gyr) | **5.6** | **0.28** |
| 27 | 1.09 × 10²⁸ | ≈ 7 × 10²⁷ atoms in a 70 kg adult | Freitas, *Nanomedicine* Vol. I (1999); the usual textbook figure. **Verify before scripting** | 1.56 | 0.06 |
| 42 | 1.41 × 10⁵¹ | ≈ 1.33 × 10⁵⁰ atoms in Earth | Jefferson Lab Science Education, "How many atoms are in the Earth?" | 10.6 | 0.25 |
| 46 | 5.50 × 10⁵⁷ | ≈ 10⁵⁷ atoms in the Sun (computed: M☉ = 1.988 × 10³⁰ kg with the surface mix X = 0.738, Y = 0.249, Z = 0.013 gives 9.5 × 10⁵⁶) | Asplund et al. 2009 (composition); IAU 2015 (mass) | 5.8 | 0.13 |
| 48 | 1.24 × 10⁶¹ | 8.07 × 10⁶⁰ Planck times since the Big Bang (4.35 × 10¹⁷ s ÷ 5.391247 × 10⁻⁴⁴ s) | CODATA 2018 (Planck time) | 1.54 | 0.03 |
| 52 | 8.07 × 10⁶⁷ | 52 factorial ÷ 8.07 × 10⁶⁰ = 9.99 × 10⁶ → "10 million orders for every Planck time" (0.05% off) | exact | — | — |

- **Every stop beats its reference by at least 54%** (the Planck time, × 1.54), as the count-film
  rule in `film-style.md` §4 asks.
- **The hook's answer is robust from below too:** 19 cards give only 0.28 of the seconds since the
  Big Bang.
- **The YOU stop (card 27) holds for anyone under about 109 kg**, since bodies carry about 10²⁶
  atoms per kg. The description says "a 70 kg adult".
- **The Doubling callback, for post text only:** 52 factorial ≈ 2^225.6, about **226 doublings**,
  against The Doubling's 205.

### Qualifiers to keep

- **Every good shuffle is almost certainly new.** "Good" means well mixed: about seven riffle
  shuffles of a 52-card deck (Bayer & Diaconis 1992). A deck fresh from the box is always in the same
  order, so the first good shuffle is what makes it new.
- **Why "almost certainly":** suppose all 8.3 billion people had shuffled once a second since the
  Big Bang (3.6 × 10²⁷ shuffles). The chance that any two of those shuffles ever matched would still
  be about 8 × 10⁻¹⁴, about 1 in 12 trillion. This goes in the description, not on screen.
- **Not more than the atoms in the universe.** 52 factorial is about 10¹² times *fewer* than the
  atoms in the observable universe (about 10⁸⁰). Never imply otherwise. It makes a good honest
  answer to the comments that will claim it.
- **The Planck time** is "where known physics runs out", never "the shortest possible time".
- **No "52!" in any post text or on screen.** It reads as an exclamation mark (`brand.md` §4).
  Write "52 factorial", or the product.

## Posting notes (draft)

These are for the post package, which is written later in the order of `lumiera_brand/FIRST_POST.md`.

- **Cover:** the hook frame (≈ 1.9 s): the card, the 4-line question and the sub-line.
- **Shorts title ideas** (no answer): "How many cards out-count the seconds since the Big Bang?"
  (56 characters) · "A deck of cards against the age of the universe" (47 characters).
- **Pinned comment:** "Lock in your guess: how many cards does it take? One number." (62 characters)
- **Description stops:**
  - 14 cards · everyone alive
  - ??? · every second since the Big Bang (your guess)
  - 27 · the atoms in you
  - 42 · Earth
  - 46 · the Sun
  - 48 · the Planck time
  - 52 · the whole deck
- **Hashtags** (the Patterns set, at most 5): #math #mathematics #probability #science
  #scienceexplained.

## Research notes (web check, October 2026)

- **The topic is well covered.** The big versions are TED-Ed's "every shuffle is unique" lesson,
  Vsauce's "Math Magic", and many written explainers (Puzzlewocky, Mental Floss). Their framing is
  almost always the full deck ("8 × 10⁶⁷", "how long would 52 factorial seconds take").
- **What is ours:**
  - a deck that grows card by card, showing *why* the count explodes;
  - the "how many cards beat the Big Bang" threshold, which our searches did not turn up;
  - honest qualifiers that the viral versions skip (a *good* shuffle; fewer than the atoms in the
    universe).
- **Errors to avoid:**
  - a summary seen in the check quotes 8.65 × 10⁶⁷, where the right figure is 8.07 × 10⁶⁷;
  - it is easy to confuse dealt sequences (52 × 51 × …) with the orders of a smaller deck
    (*k* factorial). Even the search tool's own summary slipped on this.

  The second is why this film counts a growing deck: "11 cards" is true for deals from a full deck
  but false for 11 cards' own orders.

## Decided

- **Optional stops dropped (October 2026).** Two stops were cut: the Milky Way at card 15
  (1.3 × 10¹² orders against 100–400 billion stars) and a tablespoon of water at card 24
  (6.2 × 10²³ against 5.0 × 10²³ molecules).
  - The first followed card 14 too closely, and the second had the thinnest margin (× 1.24).
  - The figures are kept in this file's git history in case a later film wants them.

## Open decisions

1. **A 4-line hook.** At size 46 it fits the width (widest line 5.69 units), but its height is
   checked only on paper (block centred at y = +5.0, below y = +6.7). Confirm with a still, the first
   time a test frame is rendered.
2. **New accents** (time `#D4FF5C`, cards `#E08BFF`). Check them in stills against
   the deep-space ground before they're final.
3. **The ring of 52 slots.** This is the main new drawing work, along with gathering from points of
   light.
   - Simpler fallback: keep a growing row of cards under the subject.
   - Cost of the fallback: it gets crowded past ~20 cards and has to compress into a fanned deck.

## Sources

- Bayer, D. & Diaconis, P. (1992). "Trailing the dovetail shuffle to its lair". *Annals of Applied
  Probability* 2(2). Summary: [Harvey Mudd Math Fun Facts, "Seven Shuffles"](https://math.hmc.edu/funfacts/seven-shuffles/).
- Planck Collaboration (2018). "Planck 2018 results. VI. Cosmological parameters" (age 13.787 ± 0.020 Gyr).
- CODATA 2018 recommended values (Planck time 5.391247 × 10⁻⁴⁴ s).
- UN DESA Population Division, *World Population Prospects 2024*. The 2026 figure is via aggregators
  ([StatisticsTimes](https://www.statisticstimes.com/m/demographics/world-population.php)); confirm on the UN site.
- Atoms in Earth: [Jefferson Lab Science Education](https://education.jlab.org/qa/mathatom_05.html).
- Solar composition: Asplund, M. et al. (2009). "The chemical composition of the Sun". *ARA&A* 47.
- Atoms in a 70 kg body: Freitas, R. A. (1999). *Nanomedicine, Volume I*, Landes Bioscience.
- Coverage: [Puzzlewocky](https://puzzlewocky.com/brain-teasers/probability-puzzles/every-shuffle-of-a-deck-of-cards-is-probably-unique-in-history/),
  [Mental Floss](https://www.mentalfloss.com/article/92408/how-many-ways-can-you-sort-deck-cards),
  [Vsauce "Math Magic" transcript](https://rosetta.to/u/vsauce/math-magic).

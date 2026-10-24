# The Shuffle

> **Status: treatment, with test stills.** The film itself is not built yet. Style rules for all
> Lumiera films are in [`../playbook/film-style.md`](../playbook/film-style.md); this film follows
> them, using the count-film variants in §4 (*Films that count*). Re-check every source in "The
> numbers" before scripting (`playbook/ideas.md`).

A ~32-second vertical (9:16) film for the **Pure Patterns** pillar. Cards are dealt one at a time from
a full shuffled deck. The first card can be any of 52, the next any of the 51 left, then 50, so each
card multiplies the possible orders.

- **6 cards:** more orders than people alive.
- **11 cards:** more orders than seconds since the Big Bang.
- **All 52 cards:** about 8.07 × 10⁶⁷ orders.

The film ends by imagining shuffles on cosmic scales, each of which still sees almost none of them.

| | |
|---|---|
| Format | 1080×1920, 60 fps, with-sound and silent versions (`film-style.md` §1) |
| Length | ≈ 32.4 s. A deliberately short cut, as `film-style.md` §1 allows |
| Hook | "How many cards dealt have more orders than seconds since the Big Bang? Guess before it ends." |
| Answer | 11 cards. **Never** in the cover, title, caption, description, tags or comments (`brand.md` §6) |
| Payoff | "Not even a quarter of the deck." (11 of 52 cards) |
| Ending | Three "imagine" beats, then the outro: "52" CARDS DEALT · "Every good shuffle / is almost certainly new." |
| Mechanic | Dealing *k* cards from a full deck gives 52 × 51 × … × (53 − *k*) possible orders |

## Why this film

- **The question is honest and answerable in your head.** We expect many guesses near the full deck.
  The true answer, 11, lands at about 9 s, and 10 cards fall clearly short (5.7 × 10¹⁶ against
  4.35 × 10¹⁷), so the answer does not hang on a rounding.
- **It is short.** At about 32 s the payoff lands early, and the "imagine" beats give the ending its
  lift.
- **It teaches why.** The ×52, ×51, ×50 pops show that each card multiplies by the cards still left.
- **It reuses the toolkit.** It keeps the HUD, the STEP chip (as CARD), the YOU pin, the meter, the
  payoff brackets, the answer ring and the loop.

Risks are listed under [Open decisions](#open-decisions).

## Structure

Times follow the pacing rules in `film-style.md` §2, except for the beats (§4, *Films that count*).
They are approximate until `story.py` exists.

| ≈ Time | Beat | On screen |
|---|---|---|
| 0–2.75 s | **Hook** | A full deck (lavender line art) at the centre. The question rises in over 4 lines (size 42, 1.04 em baseline to baseline, block centred at y = +4.78, glyph stagger 0.0085 s so all 58 glyphs are in by 0.80 s): "How many cards [dealt] / have more orders than / seconds since / the [Big Bang?]" ("dealt" lavender, "Big Bang?" cyan). Sub-line "Guess before it ends." The start label rises from ≈ 0.45 s: size text "52 cards", title A FULL DECK, line "Deal it [one card] at a time." HUD fades in at 2.5–2.95 s showing **0** (CARDS DEALT) |
| 2.75–4.7 s | **Teach the mechanic** | Cards 1 to 3 are dealt one at a time (0.70, 0.60, 0.46 s, each with a 0.06 s settle) into the first slots of a **ring of 52**. "×52", "×51", "×50" pop beside the HUD with the digit change and a pluck. The readout goes 52 → 2,652 → 132,600 orders. Phase label: size text "52 × 51 × 50 × …", title DEAL A CARD, line "Each card has [one fewer] choice." |
| 5.3 s | Stop · card 6 | EVERYONE ALIVE, with the YOU pin on Earth's night side |
| 9.1 s | **Payoff** · card 11 | SINCE THE BIG BANG. Answer ring around the HUD, then the note "Not even [a quarter] of the deck." with the meter brackets |
| 14.5–17.0 s | Long stretch | Cards 12–52 in a ≈ 2.5 s sprint. Phase label: size text "CARDS 12 → 52", title KEEP DEALING, line "The last card has [no choice]." |
| 17.0 s | **Final stop** · card 52 | THE WHOLE DECK. The ring closes. Line "52 × 51 × 50 × … × 2 × 1", then at ≈ 1.2 s the note "Imagine [shuffling] since the Big Bang." |
| 20.0 s | Imagine 1 | A SHUFFLE A SECOND |
| 22.6 s | Imagine 2 | EVERYONE, EVERY SECOND |
| 25.2 s | Imagine 3 | EVERY PLANCK TIME |
| 28.2–32.4 s | **Outro** | The HUD "52" flies to the centre (Magenta glow). Label CARDS DEALT, summary "Every good shuffle / is almost certainly new." The ring gathers back into the deck, and the frame glides back to frame 0 |

## The stops and beats

Each has the label block of `film-style.md` §5. Stops carry a chip; the start, the phase labels and
the beats don't, so their size text takes the accent. All lines are measured at their font size
against the 6.8-unit limit. Where one needs fitting, the size it fits at is given (never below 26).

| At | Chip | Size text | Title | Line (and note) | Accent | Row |
|---|---|---|---|---|---|---|
| start | none | 52 cards | A FULL DECK | Deal it [one card] at a time. | lavender `#B9A6FF` | none |
| card 6 | CARD 6 | 8.3 billion people | EVERYONE ALIVE (fit 58.9) | More orders than [people alive] | cyan `#6CF2FF` (people) | pass row: 0 → 8.3 billion people |
| card 11 | CARD 11 | 4.35 × 10¹⁷ seconds | SINCE THE BIG BANG (fit 48.8) | More orders than [every second]. Note at ≈ 2.5 s: "Not even [a quarter] of the deck." | time `#D4FF5C` | pass row: 0 → 435 quadrillion seconds |
| card 52 | CARD 52 | 8.07 × 10⁶⁷ orders | THE WHOLE DECK (fit 56.0) | 52 × 51 × 50 × … × 2 × 1. Note at ≈ 1.2 s: "Imagine [shuffling] since the Big Bang." (fit 27.7) | cards `#E08BFF` | none |
| imagine 1 | none | 4.35 × 10¹⁷ shuffles | A SHUFFLE A SECOND | At most [1 in 1.85 × 10⁵⁰] of the orders (fit ≈ 28) | time `#D4FF5C` | copies row: 0 → 1.85 × 10⁵⁰ per shuffle |
| imagine 2 | none | 3.61 × 10²⁷ shuffles | EVERYONE, EVERY SECOND (fit) | At most [1 in 2.23 × 10⁴⁰] of the orders (fit ≈ 28) | cyan `#6CF2FF` (people) | copies row: 0 → 2.23 × 10⁴⁰ per shuffle |
| imagine 3 | none | 8.07 × 10⁶⁰ shuffles | EVERY PLANCK TIME | Still at most [1 in 10 million] | Planck `#B9A6FF` | copies row: 0 → 10 million per shuffle |

- **The beats** are phase labels after the final stop, about 2.6 s each (3.0 s for the last).
  - The HUD stays on 52 and the closed ring stays in the cards accent.
  - Each beat has its own drawing inside the ring: the Big Bang spiral, a crowd of cyan points, and
    the Planck dot in its dashed reticle.
  - Each beat's row lines up copies of all those shuffles across the orders, so its counter ends on
    the line's number.
- **The payoff meter brackets** at card 11 read **11** (cyan, start to answer) and **41** (Mist Light,
  answer to the full deck).
- **YOU** appears at card 6, as one of everyone alive. The crowd in imagine 2 is cyan, because people
  stand in for you.

## How it looks

- **The hero is the deck.**
  - It starts as a full deck at the centre: the top card over a few stepped edges.
  - Dealt cards fill a ring of 52 card slots (radius 2.55 units, centred at y = +0.22, slots
    0.19 × 0.27), clockwise from the top.
  - Its bottom slot ends at y ≈ −2.47, 0.17 above the chip, and its left edge (x ≈ −2.69) stays
    clear of the meter and the bracket labels.
- **Inside the ring:**
  - Each stop's reference is drawn in light: Earth's night side with city lights, the Big Bang
    spiral, the orders cloud at card 52, and the beats' drawings.
  - The objects do **not** zoom by scale: counts are not sizes.
  - Each object gathers from points of light as it arrives and scatters into the warp as it leaves.
- **The row** sits under the count readout, where The Doubling's ruler was (y = +3.75, 5.2 wide).
  - On a pass row, the reference fills over 0.7 s with its live counter, then the orders overtake it
    in lavender over 0.8 s.
  - The beats use copies rows instead, because their reference would fill less than 2% of the track.
  - The counter rides 0.3 below the track and starts no further left than the track.
- **The orders** are a faint lavender cloud of points behind the reference. They brighten and swirl
  with the rate of growth, and the warp streaks follow the same rate.
- **Background:** near-black, warm-green near the Big Bang and violet from card 52, always within
  `film-style.md` §3.

## Sound

The palette is The Doubling's (`film-style.md` §7):
- **Plucks:** cards 1–3 get the rising pluck.
- **Ticks:** one tick per card in the sprints.
- **Stop bells:** each regular stop gets a bell, and each beat a quieter one.
- **Big reveals:**
  - the "You" chord at card 6;
  - the payoff chord and answer shimmer at card 11;
  - the finale chord at card 52.
- **Pad:** its brightness follows the count (log₂ of the orders) rather than size.

## The numbers

Dealing *k* cards from a full deck gives 52 × 51 × … × (53 − *k*) possible orders. All 52 give
exactly 80,658,175,170,943,878,571,660,636,856,403,766,975,289,505,440,883,277,824,000,000,000,000
≈ 8.07 × 10⁶⁷ (52 factorial). A stop sits at the first card whose orders beat the reference.

- **The ×above column** is the margin at the stop: the orders at the stop divided by the reference.
- **The ×below column** is the same ratio one card earlier.

| Card | Orders | Reference used | Source | × above | × below |
|---|---|---|---|---|---|
| 6 | 1.47 × 10¹⁰ | 8.3 billion people alive (mid-2026) | UN DESA, *World Population Prospects 2024* (projection for 2026) | 1.77 | 0.038 |
| **11** | **2.41 × 10¹⁸** | **4.35 × 10¹⁷ s since the Big Bang** (13.787 Gyr × 3.156 × 10⁷ s per Julian year) | Planck Collaboration 2018, VI (13.787 ± 0.020 Gyr) | **5.54** | **0.13** |
| 52 | 8.07 × 10⁶⁷ | — | exact | — | — |

| Beat | Shuffles | Orders per shuffle (52 factorial ÷ shuffles) | On screen | Source |
|---|---|---|---|---|
| A shuffle a second since the Big Bang | 4.35 × 10¹⁷ | 1.854 × 10⁵⁰ | 1 in 1.85 × 10⁵⁰ (0.2% off) | Planck 2018 |
| Everyone alive, a shuffle a second since the Big Bang | 8.3 × 10⁹ × 4.35 × 10¹⁷ = 3.61 × 10²⁷ | 2.234 × 10⁴⁰ | 1 in 2.23 × 10⁴⁰ (0.2% off) | UN WPP 2024, Planck 2018 |
| A shuffle every Planck time since the Big Bang | 4.35 × 10¹⁷ s ÷ 5.391247 × 10⁻⁴⁴ s = 8.07 × 10⁶⁰ | 9.995 × 10⁶ | 1 in 10 million (0.05% off) | CODATA 2018 |

- **Both stops beat their reference by at least 77%** (card 6, × 1.77), as the count-film rule in
  `film-style.md` §4 asks.
- **The hook's answer is robust from below too:** 10 cards give only 0.13 of the seconds since the
  Big Bang.
- **The beats say "at most"** because S shuffles can show at most S different orders. Repeats only
  lower it.
- **The Doubling callback, for post text only:** 52 factorial ≈ 2^225.6, about **226 doublings**,
  against The Doubling's 205.

### Qualifiers to keep

- **"Dealt" is what makes 11 true.** Dealing 11 cards from a full deck gives 52 × 51 × … × 42
  ≈ 2.41 × 10¹⁸ orders. A deck of only 11 cards has just 11 factorial ≈ 4.0 × 10⁷.
  - The hook says "dealt", the start label says A FULL DECK, and the pops show ×52, ×51, ×50, so the
    film never invites the second reading.
  - Post text should keep "dealt from a full deck" too.
- **Every good shuffle is almost certainly new.** "Good" means well mixed: about seven riffle
  shuffles of a 52-card deck (Bayer & Diaconis 1992). A deck fresh from the box is always in the same
  order, so the first good shuffle is what makes it new.
- **Why "almost certainly":** even imagine 2's 3.6 × 10²⁷ shuffles give only about an 8 × 10⁻¹⁴
  chance that any two ever matched, about 1 in 12 trillion. This goes in the description, not on
  screen.
- **Not more than the atoms in the universe.** 52 factorial is about 10¹² times *fewer* than the
  atoms in the observable universe (about 10⁸⁰). Never imply otherwise; it makes a good honest answer
  in the comments.
- **The Planck time** is "where known physics runs out", never "the shortest possible time".
- **No "52!" in any post text or on screen.** It reads as an exclamation mark (`brand.md` §4).
  Write "52 factorial", or the product.

## Posting notes (draft)

These are for the post package, which is written later in the order of `lumiera_brand/FIRST_POST.md`.

- **Cover:** the hook frame (≈ 1.9 s): the deck, the 4-line question and the sub-line.
- **Shorts title ideas** (no answer): "Deal from a full deck: how many cards beat the Big Bang?"
  (56 characters) · "Cards dealt against the age of the universe" (43 characters).
- **Pinned comment:** "Lock in your guess: how many cards does it take? One number." (60 characters)
- **Description stops:**
  - 6 cards · everyone alive
  - ??? · every second since the Big Bang (your guess)
  - 52 · the whole deck
  - then the three "imagine" beats
- **Hashtags** (the Patterns set, at most 5): #math #mathematics #probability #science
  #scienceexplained.

## Research notes (web check, October 2026)

- **The topic is well covered.** The big versions are TED-Ed's "every shuffle is unique" lesson,
  Vsauce's "Math Magic", and many written explainers (Puzzlewocky, Mental Floss). Their framing is
  almost always the full deck ("8 × 10⁶⁷", "how long would 52 factorial seconds take").
- **What is ours:**
  - the deal shown card by card, showing *why* the count explodes;
  - the "how many cards beat the Big Bang" threshold, which our searches did not turn up;
  - the escalating "imagine" beats;
  - honest qualifiers that the viral versions skip (a *good* shuffle; "at most"; fewer than the atoms
    in the universe).
- **Errors to avoid:**
  - a summary seen in the check quotes 8.65 × 10⁶⁷, where the right figure is 8.07 × 10⁶⁷;
  - it is easy to confuse dealt sequences (52 × 51 × …) with the orders of a smaller deck
    (*k* factorial). Even the search tool's own summary slipped on this, which is why "dealt" stays in
    every line.

## Decided

- **A short dealing cut, answer 11 (October 2026).**
  - The first treatment grew a deck card by card (answer 20, ≈ 44 s, with stops at the atoms in you,
    Earth, the Sun and the Planck time).
  - It was cut to a ≈ 32 s film that deals from a full deck and ends on three "imagine" beats.
  - The earlier figures are in this file's git history.
- **The 4-line hook layout (checked in stills).** Size 42, spaced 1.04 em baseline to baseline,
  block centred at y = +4.78.
  - Measured: top +5.96 (the cover crop is +6.0), 0.15 above the sub-line, widest line 5.98 units.
  - Size 46 centred at +5.0 broke the cover crop. Stacking the lines by ink boxes spaced them
    unevenly.
- **The payoff, final-stop and beat layouts (checked in stills).** There is no overlapping text at
  the payoff's arrival, mid-row, row end and late hold, at the final stop's arrival and note, or in
  the three beats.
  - The pass row's counter starts no further left than the track, after a first render ran it into a
    bracket line.
  - The count readout shows three significant figures, so it matches the size text at card 52.
  - Each beat's counter ends on its line's rounded number. The first render ended imagine 3 on
    "9,994,565" against a line of "10 million".

## Open decisions

1. **The new accents.** Time (`#D4FF5C`) and cards (`#E08BFF`) are both in the stills. They need
   your verdict.
2. **The beat pacing.** 2.6 s per beat is tight for "At most 1 in 1.85 × 10⁵⁰ of the orders". Check
   it in the first animatic, and lengthen the beats if needed (≈ +1 s in all).
3. **Stand-in drawings.** The Big Bang spiral, the crowd, the Planck dot and the orders cloud are
   placeholders for the build.
4. **The ring of 52 slots.** It is the main new drawing work, along with gathering from points of
   light.
   - Simpler fallback: a growing row of dealt cards under the subject.
   - Cost of the fallback: it gets crowded past ~20 cards and has to compress into a fanned deck.

## Test stills

`stills.py` poses single frames with The Doubling's engine (imported from `../doubling_film`) to check
layout before the film is built: the hook frame, the payoff (card 11), the final stop (card 52) and
the three beats.

```bash
manim -r 1080,1920 -s --disable_caching stills.py HookFrame            # the hook frame (≈ 1.9 s, the cover)
manim -r 1080,1920 -s --disable_caching stills.py PayoffFrame          # card 11, 3.6 s into the hold
manim -r 1080,1920 -s --disable_caching stills.py FinalFrame           # card 52, with its note
manim -r 1080,1920 -s --disable_caching stills.py ImagineSecond        # the first beat (also ImagineEveryone, ImaginePlanck)
GUIDES=1 manim -r 1080,1920 -s --disable_caching stills.py PayoffFrame # with the safe zones drawn over it
```

- **Other moments:** `PayoffArrival`, `PayoffMidRow`, `PayoffRowEnd` and `FinalArrival` pose the
  other moments from `film-style.md` §8.
- **Printed checks:** the hook scene prints its measured extents; the stop scenes print every text box
  and any overlap with other text, the ring's slots, the meter or the bracket lines.
- **Layout overrides:** `HOOK_SIZE`, `HOOK_PITCH` (em) and `HOOK_Y` override the hook layout for
  comparisons.
- **Requirements:** Manim 0.18 or later with the Pango and Cairo headers, as in
  `../doubling_film/README.md`.

## Sources

- Bayer, D. & Diaconis, P. (1992). "Trailing the dovetail shuffle to its lair". *Annals of Applied
  Probability* 2(2). Summary: [Harvey Mudd Math Fun Facts, "Seven Shuffles"](https://math.hmc.edu/funfacts/seven-shuffles/).
- Planck Collaboration (2018). "Planck 2018 results. VI. Cosmological parameters" (age 13.787 ± 0.020 Gyr).
- CODATA 2018 recommended values (Planck time 5.391247 × 10⁻⁴⁴ s).
- UN DESA Population Division, *World Population Prospects 2024*. The 2026 figure is via aggregators
  ([StatisticsTimes](https://www.statisticstimes.com/m/demographics/world-population.php)); confirm on the UN site.
- Coverage: [Puzzlewocky](https://puzzlewocky.com/brain-teasers/probability-puzzles/every-shuffle-of-a-deck-of-cards-is-probably-unique-in-history/),
  [Mental Floss](https://www.mentalfloss.com/article/92408/how-many-ways-can-you-sort-deck-cards),
  [Vsauce "Math Magic" transcript](https://rosetta.to/u/vsauce/math-magic).

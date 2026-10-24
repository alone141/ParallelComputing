# The Shuffle

> Style rules for all Lumiera films are in [`../playbook/film-style.md`](../playbook/film-style.md). This film
> follows them, with the count-film rules in §4 (*Films that count*). Every number below was re-verified in
> October 2026; re-check the sources before any re-post that quotes a figure.

A ≈ 34-second vertical (9:16) film for the **Pure Patterns** pillar. Cards are dealt one at a time from a
full, shuffled deck. The first card can be any of 52, the next any of the 51 left, then 50, so each card
multiplies the possible orders.

- **6 cards:** more orders than people alive.
- **11 cards:** more orders than seconds since the Big Bang.
- **All 52 cards:** about 8.07 × 10⁶⁷ orders.

The film ends by imagining a good shuffle on cosmic scales, each of which still reaches almost none of them.

| | |
|---|---|
| Format | 1080×1920, 60 fps, H.264 + AAC; with-sound and silent versions (`film-style.md` §1) |
| Length | 33.96 s (2,038 frames). A deliberately short cut that lands its payoff early, as `film-style.md` §1 allows |
| Hook | "How many cards dealt have more orders than seconds since the Big Bang? Guess before it ends." |
| Answer | 11 cards dealt. **Never** in the cover, title, caption, description, tags or comments (`brand.md` §6) |
| Payoff | "Not even a quarter of the deck." (11 of 52 cards) |
| Ending | Three "imagine" beats, then the outro: "52" CARDS DEALT · "Every good shuffle / is almost certainly new." |
| Mechanic | Dealing *k* cards from a full deck gives 52 × 51 × … × (53 − *k*) possible orders |

## Why this film

- **The question is honest and answerable in your head.** Most guesses will sit near the full deck. The
  answer, 11, lands at 9.5 s, and 10 cards fall clearly short (5.74 × 10¹⁶ against 4.35 × 10¹⁷), so it does not
  hang on a rounding.
- **It teaches why.** The ×52, ×51, ×50 pops show that each card multiplies by the cards still left.
- **It reuses the toolkit.** The HUD, the chip (as CARD), the YOU pin, the meter, the payoff brackets, the
  answer ring, the loop and the score are The Doubling's, driven by one timeline.
- **"Dealt" is what makes 11 true.** A deck of only 11 cards has just 11 factorial ≈ 4.0 × 10⁷ orders. The hook
  says "dealt", the start label says A FULL DECK, and the pops show ×52, ×51, ×50, so the film never invites
  the second reading. Post text keeps "dealt from a full deck" too.

## Structure

Times come from `story.py` (`python story.py` prints them).

| Time | Beat | On screen |
|---|---|---|
| 0–3.1 s | **Hook** | A full deck (lavender line art) at the centre. The question rises in over 4 lines (size 42, 1.04 em baseline to baseline, block centred at y = +4.75, glyph stagger 0.0085 s so all 58 glyphs are in by 0.80 s): "How many cards [dealt] / have more orders than / seconds since / the [Big Bang?]" ("dealt" lavender, "Big Bang?" cyan). Sub-line "Guess before it ends." at 0.85 s. Both leave from 2.45 s. The start label rises from 0.45 s: "52 cards" · A FULL DECK · "Every deal is one [order] of the cards.", and leaves at 3.5 s. The HUD fades in at 2.85–3.3 s showing **0** CARDS DEALT |
| 3.1–5.04 s | **Teach the mechanic** | Cards 1, 2 and 3 land at 3.74, 4.41 and 4.94 s, each flying out of the deck into the first slots of a ring of 52. "×52", "×51", "×50" pop beside the HUD with the digit change and a pluck. The readout appears with the first card: 52 → 2,652 → 132,600 orders. Phase label from 3.8 s: "52 × 51 × 50 × …" · DEAL A CARD · "Each card has [one fewer] choice." |
| 5.62 s | Stop · card 6 | EVERYONE ALIVE, a disc of cyan points of light gathering at the centre, with the YOU pin. Pass row: 8.3 billion people fill in cyan, then the orders overtake them in lavender. Hold 3.2 s |
| 9.46 s | **Payoff** · card 11 | SINCE THE BIG BANG: the Big Bang spiral in lime. Answer ring around the HUD. Pass row: 4.35 × 10¹⁷ seconds (18% of the track), then the orders. At 11.96 s the note "Not even [a quarter] of the deck." with the meter brackets 11 and 41. Hold 5.4 s |
| 14.86–17.36 s | Long stretch | Cards 12–52 stream out of the deck in a 2.5 s sprint; the deck's edges thin and the top card leaves with the last one. Phase label: "CARDS 12 → 52" · KEEP DEALING · "The last card has [no choice]." |
| 17.36 s | **Final stop** · card 52 | THE WHOLE DECK: the ring closes and turns violet; the orders dust brightens. Line "52 × 51 × 50 × … × 2 × 1", then at 18.66 s the note "Imagine a good shuffle / every second since the Big Bang." Hold 3.2 s |
| 20.56 s | Imagine 1 | A SHUFFLE A SECOND · "4.35 × 10¹⁷ shuffles" · "At most [1 in 1.85 × 10⁵⁰] of the orders". 3.0 s |
| 23.56 s | Imagine 2 | EVERYONE, EVERY SECOND · "3.61 × 10²⁷ shuffles" · "At most [1 in 2.23 × 10⁴⁰] of the orders". 3.0 s |
| 26.56 s | Imagine 3 | EVERY PLANCK TIME · "8.07 × 10⁶⁰ shuffles" · "Still about [1 in 10 million]". 3.2 s |
| 29.76–33.96 s | **Outro** | The HUD "52" flies to the centre with the Magenta glow. Label CARDS DEALT, summary "Every good shuffle / is almost certainly new." The ring gathers back into the deck's glow; over the last 1.2 s the deck re-forms and the frame glides back to frame 0 |

## The stops and beats

Each has the label block of `film-style.md` §5. Stops carry a chip; the start, the phase labels and the beats
don't, so their size text takes the accent. Every line is measured against the 6.8-unit limit by
`stills.py --check`, which also checks the safe zones and every text, pin and meter collision at 24 moments.

| At | Chip | Size text | Title | Line (and note) | Accent | Row |
|---|---|---|---|---|---|---|
| start | none | 52 cards | A FULL DECK | Every deal is one [order] of the cards. | lavender `#B9A6FF` | none |
| cards 1–3 | none | 52 × 51 × 50 × … | DEAL A CARD | Each card has [one fewer] choice. | lavender | none |
| card 6 | CARD 6 | 8.3 billion people | EVERYONE ALIVE | More orders than [people alive] | cyan `#6CF2FF` (people) | pass row: 0 → 8.3 billion people |
| card 11 | CARD 11 | 4.35 × 10¹⁷ seconds | SINCE THE BIG BANG | More orders than [seconds] since the Big Bang. Note at 2.5 s: Not even [a quarter] of the deck. | lime `#D4FF5C` (time) | pass row: 0 → 4.35 × 10¹⁷ seconds |
| cards 12–52 | none | CARDS 12 → 52 | KEEP DEALING | The last card has [no choice]. | lavender | none |
| card 52 | CARD 52 | 8.07 × 10⁶⁷ orders | THE WHOLE DECK | 52 × 51 × 50 × … × 2 × 1. Note at 1.3 s: Imagine a good shuffle / every second since the Big Bang. | violet `#E08BFF` (the deck) | none |
| imagine 1 | none | 4.35 × 10¹⁷ shuffles | A SHUFFLE A SECOND | At most [1 in 1.85 × 10⁵⁰] of the orders | lime | the shuffles as one point of light; the orders sweep past |
| imagine 2 | none | 3.61 × 10²⁷ shuffles | EVERYONE, EVERY SECOND | At most [1 in 2.23 × 10⁴⁰] of the orders | cyan (people) | same |
| imagine 3 | none | 8.07 × 10⁶⁰ shuffles | EVERY PLANCK TIME | Still about [1 in 10 million] | lavender (Planck) | same |

- **The beats** are phase labels after the final stop, with their own drawing inside the ring (the spiral, the
  crowd, the Planck dot in its reticle). The HUD holds 52 and the count readout dims to 40%, so one exponent
  number is live per frame. Their rows carry no live counter: the line carries the figure.
- **The payoff brackets** read **11** (start to the answer, in the time accent) and **41** (the answer to the
  whole deck, Mist Light). Neither is the viewer's distance, so neither is cyan.
- **YOU** is one of everyone alive: the pin sits on the crowd at card 6, and the meter's YOU tag at card 6.

## How it looks

- **The deck is the hero.** It starts as a full deck: the top card over four stepped edges. Dealt cards fly out
  of it, shrinking into a **ring of 52 slots** (radius 2.55 units about y = +0.22, slots 0.19 × 0.27), clockwise
  from the top. The edges thin as the cards go (one edge per 13 cards) and the top card leaves with card 52.
  While a reference drawing holds the centre, the deck's cards fade and its halo stays.
- **No zoom by scale.** Counts are not sizes. Each reference gathers from 36 points of light as it arrives and
  scatters as it leaves (0.7 s in, 0.35 s out).
- **The pass row** sits under the count readout, where The Doubling's ruler was (y = +3.75, 5.2 units wide).
  The reference fills first in its colour over 0.7 s with a live counter, then the orders overtake it in
  lavender over 0.8 s. The counter rides 0.3 below the track, in the notation of the size text ("8.3 billion
  people", "4.35 × 10¹⁷ seconds"), starts no further left than the track, and ends on the size text's figure.
- **The meter** is linear in cards dealt, so its brackets agree with "Not even a quarter of the deck".
- **The orders** are a faint lavender dust behind everything inside the ring, thickening with log₂ of the count
  and turning violet when the deck is whole.
- **Colours:** lavender for the start, the mechanic and the dealt cards; cyan for people; lime for time; violet
  for the whole deck; magenta only for the outro glow. Backgrounds stay within `film-style.md` §3.
- **The loop:** the ring gathers into the deck's glow as the HUD leaves; over the last 1.2 s the deck re-forms,
  the dust fades, and the background, nebula and particles return to their frame-0 state. `checks.py` measures
  the last frame against the first.

## Sound

The score is The Doubling's palette (`film-style.md` §7), generated from the same timeline:

- a soft bloom and an A4 bell at frame 0, a riser into the first card;
- a tick on every card (a buzz in the sprint), plucks E4 G4 A4 on cards 1–3;
- whooshes that follow the rate of dealing;
- a ratchet while each pass-row counter climbs and a chime when it lands; a quiet sweep for each beat's row;
- the "You" chord at card 6, the payoff chord with the answer shimmer at card 11, the finale chord at card 52;
  a quieter bell on each beat (C5, D5, E5);
- a pad whose brightness follows log₂ of the orders;
- a master pass that measures the mix with ffmpeg's `ebur128` and lands it at −15 LUFS with the true peak
  under −1.3 dBTP, so the final MP4 sits inside −16 to −14 LUFS and under −1 dBTP.

## The numbers

Dealing *k* cards from a full deck gives 52 × 51 × … × (53 − *k*) possible orders. All 52 give exactly
80,658,175,170,943,878,571,660,636,856,403,766,975,289,505,440,883,277,824,000,000,000,000 ≈ 8.07 × 10⁶⁷
(52 factorial; copy the integer, never retype it). A stop sits at the first card whose orders beat the reference.
*× at the stop* is the orders divided by the reference; *× one card earlier* is the same ratio one card earlier.

| Card | Orders | Reference (exact) | As shown | Source | × at the stop | × one card earlier |
|---|---|---|---|---|---|---|
| 6 | 1.466 × 10¹⁰ | 8,300,678,396 people, 1 July 2026 | 8.3 billion people | UN DESA, *World Population Prospects 2024*, medium variant | 1.77 | 0.038 |
| **11** | **2.411 × 10¹⁸** | **4.3509 × 10¹⁷ s** (13.787 Gyr × 365.25 × 86,400) | 4.35 × 10¹⁷ seconds | Planck Collaboration 2018, VI, TT,TE,EE+lowE+lensing+BAO: 13.787 ± 0.020 Gyr | **5.54** | **0.13** |
| 52 | 8.066 × 10⁶⁷ | — | 8.07 × 10⁶⁷ orders | exact | — | — |

| Beat | Shuffles (exact) | As shown | 52 factorial ÷ shuffles | On screen |
|---|---|---|---|---|
| A shuffle a second since the Big Bang | 4.3509 × 10¹⁷ | 4.35 × 10¹⁷ shuffles | 1.8539 × 10⁵⁰ | at most 1 in 1.85 × 10⁵⁰ (a valid bound: 1.85 ≤ 1.8539) |
| Everyone alive, a shuffle a second since the Big Bang | 8.3007 × 10⁹ × 4.3509 × 10¹⁷ = 3.6115 × 10²⁷ | 3.61 × 10²⁷ shuffles | 2.2334 × 10⁴⁰ | at most 1 in 2.23 × 10⁴⁰ (valid: 2.23 ≤ 2.2334) |
| A shuffle every Planck time since the Big Bang | 4.3509 × 10¹⁷ s ÷ 5.391247 × 10⁻⁴⁴ s = 8.0702 × 10⁶⁰ | 8.07 × 10⁶⁰ shuffles | 9,994,565 | **about** 1 in 10 million: "at most 1 in 10 million" would round the bound the wrong way by 0.05%, so the line says "about" |

- **Both stops beat their reference by at least 77%**, and the hook's answer holds from below: 10 cards give
  0.13 of the seconds since the Big Bang.
- **"At most"** is honest because *S* shuffles can show at most *S* different orders; repeats only lower it.
- **The 8.07 coincidence.** Beat 3's 8.07 × 10⁶⁰ shuffles and the deck's 8.07 × 10⁶⁷ orders share a mantissa
  by chance (8.0702 against 8.0658). Their quotient is 9.99 × 10⁶, which is why the last beat lands on
  10 million. Expect a "typo" comment; the reply is in `POST.md`.
- **Why "almost certainly new".** With *S* = 3.61 × 10²⁷ shuffles (everyone, every second since the Big Bang)
  the chance that any two well-mixed shuffles ever matched is about *S*² ÷ (2 × 52 factorial) ≈ 8 × 10⁻¹⁴, about
  1 in 12 trillion. This goes in the description, not on screen.
- **The Doubling callback, for post text only:** 52 factorial ≈ 2^225.6, about 226 doublings, against The
  Doubling's 205.

### Qualifiers to keep

- **"Dealt"** wherever 11 appears, including post text.
- **A good shuffle** means a well-mixed deck: about seven riffle shuffles (Bayer & Diaconis 1992, whose
  theorem is 3/2 log₂ *n* + θ riffles, with the deck still 0.33 from uniform in total variation after seven).
  A deck fresh from the box is always in the same order, and one riffle of a new deck is not almost certainly
  new. Never "seven shuffles make it random".
- **"Where known physics runs out"** for the Planck time, never "the shortest possible time". The on-screen beat
  has no room for it, so it lives in the description and alt text.
- **52 factorial is about a trillion times fewer** than the atoms in the observable universe (about 10⁸⁰, a
  figure that spans 10⁷⁸–10⁸²). Never sharper than "about a trillion", and never the other way round.
- **No "52!"** on screen or in post text. It reads as an exclamation mark. Write "52 factorial", or the product
  in the film's order: 52 × 51 × … × 1.
- **The age** is the Planck 2018 value with BAO (13.787 Gyr). The Planck-only value (13.797 Gyr) rounds to the
  same 4.35 × 10¹⁷ s.

## Rendering

```bash
# once: Pango and Cairo headers, then a pinned environment
sudo apt-get install libpango1.0-dev libcairo2-dev pkg-config
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

PYTHON=.venv/bin/python MANIM=.venv/bin/manim ./render.sh preview   # 540x960 @ 30 fps, ≈ 4 min on 4 cores
PYTHON=.venv/bin/python MANIM=.venv/bin/manim ./render.sh           # 1080x1920 @ 60 fps, ≈ 8 min on 4 cores
PYTHON=.venv/bin/python MANIM=.venv/bin/manim ./render.sh range 9 12  # one full-resolution slice, no sound
```

The output goes to `out/the_shuffle_final.mp4` (with sound) and `out/the_shuffle_final_silent.mp4`. The film
is rendered as 4 frame-exact slices in parallel (`JOBS=4`), joined without re-encoding, muxed with the score
and encode-time grain, then checked (`checks.py`: resolution, frame count, loudness, true peak, loop).

Fonts: Inter, Inter Display and JetBrains Mono. A single 1080×1920 frame renders in about 0.6 s; a 540×960
frame in about 0.3 s.

## How it is built

| File | Role |
|---|---|
| `story.py` | The numbers, the stops, the beats, the label texts and the master timeline `n_at(t)` (cards dealt at time *t*), plus `orders(k)` and the smooth, monotone `log2_orders(n)` (log-linear between cards: the last card multiplies by 1, so it is flat there). The picture and the sound both read it |
| `film.py` | The scene. Each frame is posed from `n_at(t)` and *t*: the deck, the ring, the drawings gathering from light, the pass rows and their counters, the HUD with its count readout, the meter and brackets, the pops, the labels and the outro. `report(t)` checks a frame for collisions and safe zones |
| `visuals.py` | The drawings: the deck, the ring of 52 slots, the crowd, the time spiral, the Planck dot, the orders dust, the row lines |
| `textures.py` | Three procedural textures (the point disc, the spiral, the dust), cached in `media/` |
| `soundtrack.py` | The score from the same timeline, with the loudness-measured master pass |
| `stills.py` | Renders named frames of the real film and runs `report()` across 24 moments (`--check`) |
| `checks.py` | The definition-of-done checks on a rendered MP4 |
| `render.sh` | Parallel slices, concat, score, mux, silent copy, checks |
| `POST.md` | The post package (captions, hashtags, pinned comment, alt text, title, description, tags, checklist) |

The Doubling's engine and textures are imported as a package (`doubling_film.engine`, `doubling_film.assets`);
its story, visuals and soundtrack are never imported, because their flat imports would bind to this film's
modules. The one engine change for this film, a `decimals` parameter on `sci_spec`, keeps its default, and
three Doubling frames were re-rendered pixel-identical to prove it.

## Checks

- `python stills.py --check --all`: 24 moments (frame 0, the hook frame, the start, cards 1–3, each stop at
  arrival, mid-row, row end and late in the hold, the sprint, the final stop and its note, the three beats,
  the outro, the glide and the last frame) report no text-on-text, text-on-pin or text-on-meter collisions,
  no baseline below −4.8 (descenders above −4.9), nothing in the button column, no centred text wider than
  6.8 units, and the hook's top at +5.93, inside the cover crop.
- `python checks.py out/the_shuffle_final.mp4`: 1080×1920, 2,038 frames ± 1, −16 to −14 LUFS, true peak
  ≤ −1 dBTP, last frame ≈ first frame.

## Decided (October 2026, with the owner)

- **A short dealing cut, answer 11**, with three "imagine" beats at 3.0, 3.0 and 3.2 s and a one-line ending.
  The earlier growing-deck version (answer 20) and the 44–51 s cut are in `GUIDE.md` and the git history.
- **The hook keeps "orders"**, and the start label defines it ("Every deal is one order of the cards."). The
  first card lands at 3.1 s rather than 2.75 s, so a 4-line question has 1.6 s fully on screen.
- **"Still about 1 in 10 million"**, not "at most": the exact ratio is 1 in 9,994,565.
- **"Every good shuffle is almost certainly new."** stays, with "a good shuffle" in the setup note and "good"
  defined in the description and alt text.
- **The meter is linear in cards dealt.** A log₂ axis would show the "11" bracket spanning 27% of the meter
  under a line saying "not even a quarter".
- **Accents:** lime for time and the Big Bang; violet for the whole deck, 31° of hue from lavender and 32°
  from magenta, which clears the §3 rule.
- **No zoom by scale.** References gather from points of light.
- **Counters match the size text's notation** on the same frame.
- **Layout measured in stills:** the 4-line hook at size 42, 1.04 em, centred at +4.75 (top +5.93, widest
  line 5.98 units); ring radius 2.55; the two-line note's descenders at −4.89.

## Posting

Posting rules are in [`../playbook/brand.md`](../playbook/brand.md) §6. The package for this film is
[`POST.md`](POST.md); its cover is the hook frame at 1.9 s, `out/the_shuffle_cover_hook.jpg`.

## Sources

- Planck Collaboration (2020). "Planck 2018 results. VI. Cosmological parameters", *A&A* 641, A6, Table 2
  (TT,TE,EE+lowE+lensing+BAO): age 13.787 ± 0.020 Gyr. [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
- UN DESA Population Division, *World Population Prospects 2024*, medium variant, World, 1 July 2026:
  8,300,678,396. [population.un.org/wpp](https://population.un.org/wpp/).
- CODATA 2022 recommended values: Planck time 5.391247(60) × 10⁻⁴⁴ s (unchanged from CODATA 2018).
  [arXiv:2409.03787](https://arxiv.org/abs/2409.03787).
- Bayer, D. & Diaconis, P. (1992). "Trailing the dovetail shuffle to its lair". *Annals of Applied
  Probability* 2(2), 294–313. Summary: [Harvey Mudd Math Fun Facts, "Seven Shuffles"](https://math.hmc.edu/funfacts/seven-shuffles/).
- 52 factorial: [Wikipedia, "Factorial"](https://en.wikipedia.org/wiki/Factorial) (table), checked against
  Python's `math.factorial(52)`.
- Atoms in the observable universe, ~10⁸⁰: [Universe Today](https://www.universetoday.com/36302/atoms-in-the-universe/).
- Existing coverage of the full-deck version: TED-Ed's "every shuffle is unique" lesson, Vsauce's "Math Magic",
  [Puzzlewocky](https://puzzlewocky.com/brain-teasers/probability-puzzles/every-shuffle-of-a-deck-of-cards-is-probably-unique-in-history/),
  [Mental Floss](https://www.mentalfloss.com/article/92408/how-many-ways-can-you-sort-deck-cards),
  [NRICH "Card Shuffle"](https://nrich.maths.org/5402). The "how many cards dealt beat the Big Bang" threshold
  did not turn up in any search.

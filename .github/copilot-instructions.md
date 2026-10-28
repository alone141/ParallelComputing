# Copilot instructions

This file mirrors `../AGENTS.md`. Edit `AGENTS.md` and the `playbook/` files, then copy the change here, prefixing relative links with `../`.

This repository is the baseline for **Lumiera**, a science-explained-visually channel on TikTok,
Instagram Reels and YouTube Shorts. It holds the films, the brand and the guide that keeps them consistent.

## Read first

**[`playbook/`](../playbook/README.md) is the source of truth for style.** Before any creative work
(a film, a cover, captions, bios, brand assets), read:

1. [`playbook/film-style.md`](../playbook/film-style.md): how films look, move, read and sound.
2. [`playbook/brand.md`](../playbook/brand.md): identity, logo, colour, type, voice, posting rules.

If anything here or elsewhere conflicts with the playbook, the playbook wins.

## The essentials (summary of the playbook)

- **Format:** vertical 9:16, 1080×1920, 60 fps. No text baseline below y = −4.8 (the bottom ~20%; descenders may dip
  to ≈ −4.9) or in the right button column; text within the 6.8-unit band (±3.4, centred or flush) on a 9×16-unit frame. No logo or end card
  inside films; the loop is the sign-off.
- **Look:** glowing light on near-black deep space (`#07061A`). One accent colour per scene. Cyan
  `#6CF2FF` means "you" (never another subject's accent). Fonts: Inter Display (hook and titles),
  Inter (lines, sentence case), JetBrains Mono (numbers); titles are CAPS by default.
- **Structure:** one honest question, answered on screen. Hook (with "Guess before it ends.") →
  teach the mechanic → stops whose comparisons the picture makes → payoff → an ending that loops back
  to the first frame.
- **Toolkit (MAY):** The Doubling's devices (HUD counter, STEP chip, a measuring row with a live counter,
  YOU pin, progress meter), its pacing and its sound palette are the Scale pillar's default, not a rule
  for every film. A film reuses them in whole, in part or not at all, and its README lists what it keeps
  and what it breaks ([`playbook/film-style.md`](../playbook/film-style.md) §8).
- **Numbers:** one numbers table per film, rounded to nearest, "about"/"≈" when more than ~2% off, and
  qualifiers kept. The hook's answer never appears in the cover, title, caption, description, tags
  or comments; alt text is the only exception.
- **Voice:** calm, precise, awe-filled, British spelling. No hype words, no exclamation marks, no
  emoji, at most one ✦, at most 5 hashtags.
- **Logo:** a point of light with rings that double in radius, an open outer ring and a cyan
  "you are here" dot. Generate it with `lumiera_brand/build_brand.py` (geometry in `brandlib.MARK`);
  never redraw it by eye.

## Where things live

| Path | What |
|---|---|
| [`playbook/`](../playbook/) | Style and brand guide, idea backlog |
| [`doubling_film/`](../doubling_film/) | *The Doubling*: the reference film (Manim). See its README for rendering |
| [`three_doors_film/`](../three_doors_film/) | *Three Doors*: the second film (Patterns pillar), a Monty Hall round played with two tallies, rendered with Canvas 2D in headless Chromium via Playwright; see its README |
| [`lumiera_brand/`](../lumiera_brand/) | Brand assets (generated), profile texts, launch-post package |
| [`archive/`](../archive/) | Old material kept for reference, not part of the baseline |
| [`archive/shuffle_film_manim/`](../archive/shuffle_film_manim/), [`archive/shuffle_film_long_number/`](../archive/shuffle_film_long_number/) | Two shelved attempts at a card-shuffle film (Manim on The Doubling's engine; a Canvas 2D ledger). Records: do not edit or re-render |

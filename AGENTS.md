# AGENTS.md

This repository is the baseline for **Lumiera**, a science-explained-visually channel on TikTok,
Instagram Reels and YouTube Shorts. It holds the films, the brand and the guide that keeps them consistent.

## Read first

**[`playbook/`](playbook/README.md) is the source of truth for style.** Before any creative work
(a film, a cover, captions, bios, brand assets), read:

1. [`playbook/film-style.md`](playbook/film-style.md): how films look, move, read and sound.
2. [`playbook/brand.md`](playbook/brand.md): identity, logo, colour, type, voice, posting rules.

If anything here or elsewhere conflicts with the playbook, the playbook wins.

## The essentials (summary of the playbook)

- **Format:** vertical 9:16, 1080×1920, 60 fps. No text baseline below y = −4.8 (the bottom ~20%; descenders may dip
  to ≈ −4.9) or in the right button column; centred text ≤ 6.8 units wide on a 9×16-unit frame. No logo or end card
  inside films; the loop is the sign-off.
- **Look:** glowing light on near-black deep space (`#07061A`). One accent colour per scene. Cyan
  `#6CF2FF` means "you" (never another subject's accent). Fonts: Inter Display (titles, CAPS),
  Inter (lines, sentence case), JetBrains Mono (numbers).
- **Structure:** one honest question, answered on screen. Hook (with "Guess before it ends.") →
  teach the mechanic → stops with ~3.2 s holds → payoff → outro that loops back to the first frame.
- **Devices:** HUD counter, STEP chip, a measuring row with a live counter (copies of the smaller
  neighbouring stop line up across the larger one), YOU pin, progress meter.
- **Numbers:** one sizes table per film, rounded to nearest, "about"/"≈" when more than ~2% off, and
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
| `playbook/` | Style and brand guide, idea backlog |
| `doubling_film/` | *The Doubling*: the reference film (Manim). See its README for rendering |
| `lumiera_brand/` | Brand assets (generated), profile texts, launch-post package |
| `archive/` | Old material kept for reference, not part of the baseline |

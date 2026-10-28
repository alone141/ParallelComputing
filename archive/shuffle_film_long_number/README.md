# The Long Number (archived, unfinished)

The second attempt at the second film: cards dealt one at a time from a full deck, and after each one the
possible deals so far (52 × 51 × 50 × …) written out in full as the next line of a typeset ledger, against
two references (everyone alive, every second since the Big Bang). Canvas 2D in headless Chromium via
Playwright, with its own score.

It was built to a working 540 × 960 preview that passed its scripted checks (`node render.mjs --check`,
`--verify`, `checks.py`), and then shelved before the final render because its script was dropped. It is kept
as a record of the pipeline (`render.mjs`, `layout.js`, `page.html`, `soundtrack.py`, `render.sh`,
`checks.py`), which the third film, *Three Doors* (`three_doors_film/`), adapts. Do not edit or render it.

The first attempt, on The Doubling's Manim engine, is beside it in `archive/shuffle_film_manim/`.

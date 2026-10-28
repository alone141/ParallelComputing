# Three Doors: script

Working title *Three Doors*. Pure Patterns pillar. One honest question: with a prize behind one of three
doors, after the host opens an empty door, should you switch or stay? The answer is shown by playing rounds
with two tallies, never asserted first.

Every on-screen string below is final text (approved); the one open item is the colour of the target phrase, marked *proposal*. Timings are a first sketch for a film of
about 35 s at 60 fps (2 100 frames) and will be tuned on the preview.

## 1. Frame 0

Three closed doors on deep space, side by side, nothing else. No text. The last frame of the film is this
frame again, so the video loops.

## 2. Hook (0.0 to 3.0 s)

Over the three doors, three lines in Inter Display Bold, sentence case, then the sub-line:

| t (s) | Text | Colour |
|---|---|---|
| 0.20 | Three doors, one prize. | Starlight |
| 0.50 | You pick one. The host opens an empty one. | Starlight, "You" cyan |
| 0.80 | Switch or stay? | the target phrase: *proposal* lavender, the colour of the mechanic |
| 1.70 | Guess before it ends. | Mist (Inter Regular) |

The hook frame, for the cover, is about 2.0 s: all four lines in, none leaving. Everything is out by 3.0 s.

## 3. Teach: the rules (3.2 to 8.6 s)

Three rule lines, Inter Medium, Mist Light, one appearing after another and held together, then gone before
the first round starts:

| t (s) | Text |
|---|---|
| 3.20 | The host knows where the prize is. |
| 4.80 | The host always opens an empty door that is not yours. |
| 6.40 | Then you may switch to the other closed door. |

All three leave at 8.6 s. Nothing in these lines calls the rounds random or a test.

## 4. Journey: the rounds (8.6 to 29.0 s)

One round is: your pick lights on one door, the host opens an empty door, the prize is revealed behind one of
the two closed doors. If it is behind your door, STAY counts one; if it is behind the other closed door,
SWITCH counts one. One tally climbs per round, and STAY + SWITCH = ROUND at every moment.

On screen throughout: the three doors; a ROUND counter; two tallies with small caps labels, STAY and SWITCH,
each above its count (JetBrains Mono). No other line during the rounds.

| Rounds | t (s) | Pace | At the pause |
|---|---|---|---|
| 1 to 3 | 8.6 to 14.6 | about 2 s a round, every step visible | hold 2 s: ROUND 3, STAY 1, SWITCH 2 |
| 4 to 30 | 16.6 to 21.6 | accelerating | hold 2 s: ROUND 30, STAY 10, SWITCH 20 |
| 31 to 300 | 23.6 to 28.6 | a sprint, doors flickering through rounds | hold: ROUND 300, STAY 100, SWITCH 200 |

The outcome sequence is arranged, not drawn: it is fixed so that the tallies are exactly one third and two
thirds at each of the three pauses. The README says so plainly. No line on screen says "about", because the
shares shown are exact.

Rounds 1 to 3 land switch, stay, switch, so the first round already shows a switch win and the pause reads
STAY 1, SWITCH 2. Your pick moves from door to door between rounds rather than always being the same door.

## 5. Payoff (29.0 to 33.0 s)

The tallies hold at 100 and 200 of 300. One word, Inter Display Bold, Starlight:

| t (s) | Text |
|---|---|
| 29.4 | Switch. |

No second line and no reframing line. The tallies are the argument.

## 6. Ending and loop (33.0 to 35.0 s)

No closing line. "Switch." leaves, the tallies and the ROUND counter fade, the open door closes, your cyan
pick fades, and the three closed doors are exactly as at frame 0 by 35.0 s.

## 7. The numbers

| Figure | Value | On screen as |
|---|---|---|
| Rounds at the pauses | 3, 30, 300 | ROUND 3, ROUND 30, ROUND 300 |
| STAY at the pauses | 1, 10, 100 | exact thirds |
| SWITCH at the pauses | 2, 20, 200 | exact two thirds |
| Why | staying wins only when the first pick was right, 1 time in 3; switching wins otherwise, 2 times in 3 | never stated, shown by the tallies |

## 8. Playbook notes

- The hook's target phrase is "Switch or stay?". The answer, "Switch.", never appears in the cover, title,
  caption, description, tags or comments; alt text is the only exception.
- Two beats differ from `playbook/film-style.md` §2 by decision: the payoff has no reframing line, and the
  ending has no closing line (the return to frame 0 is the ending). The README lists both.
- The rounds are arranged to land on exact thirds; the README states that the sequence is fixed, and no
  on-screen line calls it random.
- British spelling, no hype, no exclamation marks, no emoji, sentence case except the two tally labels and
  ROUND, which are the playbook's small caps labels.

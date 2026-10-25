#!/usr/bin/env python3
"""The Long Number: the story file.

`python story.py` writes `build/story.json`: the numbers, every on-screen string and the
integer-frame event list that the picture (`page.html` / `layout.js` / `render.mjs`) and the sound
(`soundtrack.py`) both read. It is the only writer of that file. Pure Python 3, standard library
only; nothing is imported from `doubling_film/` or `archive/`.

`python story.py --table` also prints the timeline as a table (frame, seconds, event).

Everything below is in frames at 60 fps; the film is 2208 frames (36.80 s). Every event sits on the
6-frame grid except: the opening stamp at frame 1 (one frame after the loop point), the run's rows
12-52 and their flights and cells (a 3-frame grid), the groups of a typed number (one group every
3 frames = 60 digits per second) and the fold's staggered group starts (0.02 s apart).

----------------------------------------------------------------------------------------------------
story.json
----------------------------------------------------------------------------------------------------
meta        fps, frames, width, height, total_s, loop (frame 0 == frame 2207), the brand colours by
            role, the faces (CSS family / weight / file), the stamp bloom and unstamp decay, the
            typing rate, and the cell / row text styles.
geometry    spine, left margin, cell column, row baselines (640 + 60 (k - 1)), the deck's two boxes
            (hook and column head) with their number baselines (centre y + 0.365 x size), the card's
            look, the two rules (x = spine - reference width, y_top, bottom rule), bands, the Mono
            metrics (advance 0.6 em, group gap 0.3 em), the safe zones and the camera fit.
numbers     rows[k-1]: k, digits (exact product 52 x 51 x ... x (53 - k)), groups (threes from the
            right), display (groups joined by spaces), ndigits, new_digits, width48 / left48 (at
            48 px), multiplier (53 - k), cell_text ("x n", k >= 2), reads (the number on the card's
            back = 53 - k), deck_after (52 - k), flight_start, land, cell (None for k = 1),
            rests_as_card, rest_frames, rate_per_s (60 / frames since the previous landing),
            camera {s, tx, ty} (the page transform target at the landing; the still page for
            rows 1-11), partials (= ndigits). rate_per_s is 60 / the shorter of the gaps to the
            previous and the next landing (5, 10, 20 in the run; 2.5 for rows 49-52).
            references: seconds, people, planck_shuffles, bound1, bound2 and factorial: digits,
            sig (bright leading digits; null = uniformly bright), groups, display, label, source.
texts       every worded or numeric string on screen, keyed by a stable id:
            text, face (display | inter | interR | mono | monoB), size (px), colour (role),
            align (left | right | centre), x, baseline (or baselines for multi-line numbers),
            layer (screen | page), group (hook, deck, reference, teach, stop, run, block, note,
            beat1, beat2, ending), spans [[string, colour], ...] when a phrase takes the accent,
            and for numbers: ref (into numbers.references), sig, prefix ("≈", drawn one Mono
            advance left of the first digit), line_groups (groups per line), width_hint (px, from
            the treatment's Pillow measurements; layout.js re-measures).
camera      the run's page transform: screen = T(page) with x' = 800 + (x - 800) s, y' = y s + ty;
            targets per row (s = min(1, 660 / width48), ty = 1240 - rowBase(k) s) plus the still
            anchors before and after; splines are PCHIP through the targets.
sound       cues: the percussion cue list derived from the events (knock, stamp, flick, flick_long,
            click, typing, pulse, silence, fold_tick, knock at the block, deck_tick, deck_out);
            partials: the tone's partial-count targets over time {frame, partials, ramp_frames,
            swell, dull, cut, dim_db}.
events      the list below, sorted by frame. Every event carries an integer `frame` and a `kind`.

----------------------------------------------------------------------------------------------------
Event kinds and their fields
----------------------------------------------------------------------------------------------------
nebula      colour (role), fade_frames (cross-fade from the previous colour starting at `frame`),
            end. The first, at frame 0, has fade_frames 0.
stamp       id (a texts key): the item appears whole in one frame with the bloom. knock:
            null | "knock" | "big" (the percussion body; "big" is the film's biggest, at
            "Eleven cards dealt."). A stamp of a multi-line text id stamps the whole item.
unstamp     ids: ink gone in one frame, glow decaying over glow_decay_frames (9 = 0.15 s).
type        id (a texts key), ref (numbers.references key), groups, frames_per_group (3),
            group_frames (the frame each group appears), end (first frame after the last group),
            lines [{baseline, groups: [first, last]}] (group indices per line). A typing click per
            group.
deck_travel end, from / to (deck boxes from geometry), ease "out": the deck moves from the hook
            box to the column head (3.00 to 3.40 s) reading its number; one long quiet flick.
flight      id ("card{k}"), k, reads (number on the card's back), land, duration (frames),
            deck_after (what the deck reads once the card has left; 0 = face with no number),
            becomes: "row" (card 1: the card becomes row 1 on landing), "cell" (becomes its cell
            on the landing frame: rows 12-48), "card" (rests as a card for rest_frames, then its
            face stamps out and the cell stamps in at `cell`), rest_frames, cell (frame or null),
            rate_per_s (the deal rate at this card, for the flick's gain 1 / (1 + rate / 8)),
            pan (+0.3), to: "spine" | "cell". The flight falls straight down the column from the
            deck to the cell (or the spine for card 1), exponential ease-out, scaling from the
            screen size to the page's scale, drawn beneath the page's ink.
row         k, ndigits, new_digits, click true, muted (row 10 only: the near miss), partials
            (the tone's count after this row), dull (row 10: its new partials enter at -9 dB until
            row 11 lands), silent (row 52: the click and nothing changes in the tone).
cell        k, text ("x n"): the card's face stamps out and the multiplier cell stamps in at
            x 826 on the row's baseline (Mono Medium 40, lavender).
rule        ref (seconds | people), action: "drop" (the rule grows from y_top to the bottom over
            [frame, end]), "full" (100 % from frame to `end`), "fade" (to 0 over [frame, end]),
            "pulse_dim" (one dim pulse, row 10's near miss, over [frame, end]).
band        ref, k (the crossing row), action: "fill" (from 0 to alpha over [frame, end]) or
            "fade" (to 0 over [frame, end]); alpha (0.18), x [rule_x, spine], y [baseline - 40,
            baseline + 14] in page coordinates.
pass        k, ref, colour, rule_x, row_left, overshoot_px, digits_left_of_rule (the digits that
            cross, coloured and bloomed for the hold), hold_until, rule_full [start, end],
            band {fill, fade, alpha}, dim_rows {rows, alpha, start, end}, pulse {start, end,
            from_y, to_y} (light running down the rule), swell_partial (the tone's newest partial
            swells +6 dB for 72 frames), silence_until (card 11 only: the tone is cut and no stamp
            sounds from the landing to this frame, except the big knock at "Eleven cards dealt.").
bloom       id ("deck_number"), end, glow (x 2): the deck's 41 blooms at the reframe.
lead        piece (0-3), text (the piece), text_so_far, k: the run's lead "x 4  x 3  x 2  x 1"
            grows one piece per landing of rows 49-52 at screen scale (Mono Medium 64, lavender).
deck_leave  the deck's face and number stamp out (21.20 s); a dry tick.
deck_return the deck's empty shape returns at the hook box (36.40 s); a dry tick.
fold        end, groups (23), stagger_s, group_starts (frame each group leaves, from the left),
            ease "in-out", all groups settle at `end` (so the first travels 0.8 s), from {size,
            baseline, x_right, page_scale}, to {id: "block", size, baselines, line_groups},
            fade_out {ids, start, end} (rows 1-51, cells and references), knock_at (the knock as
            the block settles), tick per group (typing click, 23 of them).
fade        ids, end, to_alpha (0 = out; 0.45 = dimmed): light changes in place.
tone        action: "cut" (10 ms), "return" (partials, ramp_frames, swell_partial), "target"
            (partials, ramp_frames), "dim" (dim_db over [frame, end]).
marker      id: "cover" (frame 144), "last" (frame 2207 == frame 0).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from fractions import Fraction

# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

FPS = 60
TOTAL_FRAMES = 2208
TOTAL_S = Fraction(TOTAL_FRAMES, FPS)  # 36.8
WIDTH, HEIGHT = 1080, 1920

SPINE = 800          # numbers flush right here
LEFT = 140           # words flush left here
CELL_X = 826         # multiplier cells flush left here; the deck column is 826-922
COLUMN_RIGHT = 922
ROW_BASE_0 = 640
ROW_PITCH = 60
MONO_ADVANCE = 0.6   # em
MONO_GAP = 0.3       # em between groups of three

COLOURS = {
    "bg": "#07061A",
    "starlight": "#F5F7FF",
    "mistLight": "#C9D1E6",
    "mist": "#8A93A8",
    "lavender": "#B9A6FF",
    "cyan": "#6CF2FF",
    "lime": "#D4FF5C",
    "violet": "#E08BFF",
}

FACES = {
    "display": {"family": "Inter Display", "weight": 700,
                "file": "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"},
    "inter": {"family": "Inter", "weight": 500,
              "file": "/usr/share/fonts/opentype/inter/Inter-Medium.otf"},
    "interR": {"family": "Inter", "weight": 400,
               "file": "/usr/share/fonts/opentype/inter/Inter-Regular.otf"},
    "mono": {"family": "JetBrains Mono", "weight": 500,
             "file": "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Medium.ttf"},
    "monoB": {"family": "JetBrains Mono", "weight": 700,
              "file": "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf"},
}

HOOK_DECK = {"x": 730, "y": 230, "w": 192, "h": 268, "num_size": 96}
HEAD_DECK = {"x": 826, "y": 220, "w": 96, "h": 134, "num_size": 48}
for _d in (HOOK_DECK, HEAD_DECK):
    _d["cx"] = _d["x"] + _d["w"] / 2
    _d["cy"] = _d["y"] + _d["h"] / 2
    _d["num_baseline"] = round(_d["cy"] + 0.365 * _d["num_size"], 2)   # cap-height centring


def T(seconds: str) -> int:
    """Seconds (as a decimal string) to an exact integer frame; refuses anything off a frame."""
    f = Fraction(seconds) * FPS
    if f.denominator != 1:
        raise ValueError(f"{seconds} s is not an integer frame at {FPS} fps")
    return int(f)


def row_base(k: int) -> int:
    return ROW_BASE_0 + ROW_PITCH * (k - 1)


# ------------------------------------------------------------------------------------------------
# Numbers
# ------------------------------------------------------------------------------------------------

def orders(k: int) -> int:
    """Possible deals after k cards: 52 x 51 x ... x (53 - k). Exact."""
    return math.prod(range(52, 52 - k, -1))


FACT52 = math.factorial(52)


def groups_of_three(digits: str) -> list[str]:
    return [digits[max(0, i - 3):i] for i in range(len(digits), 0, -3)][::-1]


def rounded_figure(mantissa: str, exponent: int) -> str:
    """The digit string of mantissa x 10^exponent (mantissa given as its digits, e.g. '435' for 4.35)."""
    digits = mantissa + "0" * (exponent + 1 - len(mantissa))
    assert len(digits) == exponent + 1, (mantissa, exponent)
    assert digits[0] != "0"
    return digits


def mono_width(ndigits: int, size: float) -> float:
    ngroups = -(-ndigits // 3)
    return ndigits * MONO_ADVANCE * size + (ngroups - 1) * MONO_GAP * size


def num_entry(digits: str, sig, label, source, **extra) -> dict:
    g = groups_of_three(digits)
    e = {"digits": digits, "ndigits": len(digits), "sig": sig, "groups": g, "display": " ".join(g),
         "label": label, "source": source}
    e.update(extra)
    return e


REFERENCES = {
    "seconds": num_entry(rounded_figure("435", 17), 3, "Every second since the Big Bang",
                         "4.35 x 10^17 s (README: 13.787 Gyr x 365.25 x 86 400 = 4.3509 x 10^17)",
                         mantissa="4.35", exponent=17),
    "people": num_entry(rounded_figure("83", 9), 2, "Everyone alive",
                        "8.3 x 10^9 (README: UN DESA WPP 2024, 8 300 678 396 on 1 July 2026)",
                        mantissa="8.3", exponent=9),
    "planck_shuffles": num_entry(rounded_figure("807", 60), 3, "Shuffles, every Planck time",
                                 "8.07 x 10^60 (README: 4.3509 x 10^17 s / 5.391247 x 10^-44 s)",
                                 mantissa="8.07", exponent=60),
    "bound1": num_entry(rounded_figure("185", 50), None, "At most 1 in",
                        "1.85 x 10^50 (README: 52 factorial / 4.3509 x 10^17 = 1.8539 x 10^50; a valid bound)",
                        mantissa="1.85", exponent=50),
    "bound2": num_entry(rounded_figure("1", 7), None, "Still about 1 in",
                        "10^7 (README: the computed ratio is 1 in 9 994 565, so 'about')",
                        mantissa="1", exponent=7),
    "factorial": num_entry(str(FACT52), None, "Possible deals of the whole deck",
                           "math.factorial(52), exact; README 8.07 x 10^67"),
}


def check_numbers() -> dict:
    """Assert the README table's figures against the exact integers; return the measured ratios."""
    sec = int(REFERENCES["seconds"]["digits"])
    ppl = int(REFERENCES["people"]["digits"])
    pl = int(REFERENCES["planck_shuffles"]["digits"])
    b1 = int(REFERENCES["bound1"]["digits"])
    b2 = int(REFERENCES["bound2"]["digits"])
    # the ≈ figures have exponent + 1 digits
    for key in ("seconds", "people", "planck_shuffles", "bound1", "bound2"):
        r = REFERENCES[key]
        assert r["ndigits"] == r["exponent"] + 1, key
    # 52! and its README rounding
    s = str(FACT52)
    assert len(s) == 68 and orders(52) == FACT52 == orders(51)
    mant3 = round(Fraction(FACT52, 10 ** 65))  # 806.58... -> 807
    assert mant3 == 807 and len(s) - 1 == 67, "README says 8.07 x 10^67"
    # the passes are real and the misses are short (README: 1.77 / 0.038 and 5.54 / 0.13)
    ratios = {
        "row6_over_people": orders(6) / ppl,
        "row5_over_people": orders(5) / ppl,
        "row11_over_seconds": orders(11) / sec,
        "row10_over_seconds": orders(10) / sec,
        "factorial_over_seconds": FACT52 / sec,
        "factorial_over_planck_shuffles": FACT52 / pl,
    }
    assert abs(ratios["row6_over_people"] - 1.77) < 0.01
    assert abs(ratios["row5_over_people"] - 0.038) < 0.001
    assert abs(ratios["row11_over_seconds"] - 5.54) < 0.01
    assert abs(ratios["row10_over_seconds"] - 0.13) < 0.005
    assert b1 <= FACT52 // sec, "1.85 x 10^50 must be a valid lower bound of 52 factorial / seconds"
    assert abs(ratios["factorial_over_seconds"] / 1.85e50 - 1) < 0.01
    assert abs(ratios["factorial_over_planck_shuffles"] / b2 - 1) < 0.01, "about 1 in 10 million"
    # digit counts the treatment relies on
    assert [len(str(orders(k))) for k in (5, 6, 10, 11, 12)] == [9, 11, 17, 19, 20]
    assert REFERENCES["seconds"]["ndigits"] == 18 and REFERENCES["people"]["ndigits"] == 10
    assert REFERENCES["bound1"]["ndigits"] == 51 and REFERENCES["planck_shuffles"]["ndigits"] == 61
    assert len(groups_of_three(s)) == 23
    return ratios


# ------------------------------------------------------------------------------------------------
# Geometry derived from the numbers
# ------------------------------------------------------------------------------------------------

RULE_X = {ref: round(SPINE - mono_width(REFERENCES[ref]["ndigits"], 48), 2) for ref in ("seconds", "people")}
RULE_Y_TOP = {"seconds": 318 + 12, "people": 444 + 12}
BAND_ALPHA = 0.18
CAMERA_FIT_PX = 660
CAMERA_NEWEST_Y = 1240


def camera_target(k: int) -> dict:
    w = mono_width(len(str(orders(k))), 48)
    s = min(1.0, CAMERA_FIT_PX / w)
    return {"s": round(s, 6), "tx": round(SPINE * (1 - s), 3), "ty": round(CAMERA_NEWEST_Y - row_base(k) * s, 3),
            "width48": round(w, 2)}


# ------------------------------------------------------------------------------------------------
# Texts
# ------------------------------------------------------------------------------------------------

TEXTS: dict[str, dict] = {}


def text(tid, s, face, size, colour, align, x, baseline, layer, group, **extra):
    assert tid not in TEXTS, tid
    e = {"text": s, "face": face, "size": size, "colour": colour, "align": align, "x": x,
         "baseline": baseline, "layer": layer, "group": group}
    e.update(extra)
    TEXTS[tid] = e


def number_text(tid, ref, size, colour, align, x, baseline, layer, group, **extra):
    r = REFERENCES[ref]
    prefix = extra.pop("prefix", None)
    shown = (prefix + " " if prefix else "") + r["display"]
    text(tid, shown, "mono", size, colour, align, x, baseline, layer, group,
         ref=ref, digits=r["digits"], sig=r["sig"], prefix=prefix, **extra)


def build_texts():
    # the deck
    text("deck52", "52", "monoB", 96, "lavender", "centre", HOOK_DECK["cx"], HOOK_DECK["num_baseline"],
         "screen", "deck", head={"size": 48, "x": HEAD_DECK["cx"], "baseline": HEAD_DECK["num_baseline"]},
         note="the number on the deck's back; it reads 52 - (cards dealt) and travels with the deck")
    text("deck_label", "The whole deck, face down.", "inter", 32, "mist", "right", COLUMN_RIGHT, 548,
         "screen", "deck", width_hint=430)
    # the hook: Inter Display Bold 67, pitch 78, flush left
    text("hook1", "How many cards", "display", 67, "starlight", "left", LEFT, 640, "screen", "hook")
    text("hook2", "make more possible deals", "display", 67, "starlight", "left", LEFT, 718, "screen", "hook",
         spans=[["make more ", "starlight"], ["possible deals", "lavender"]], width_hint=798)
    text("hook3", "than seconds since", "display", 67, "starlight", "left", LEFT, 796, "screen", "hook")
    text("hook4", "the Big Bang?", "display", 67, "lime", "left", LEFT, 874, "screen", "hook")
    text("hook_sub", "Guess before it ends.", "interR", 34, "mist", "left", LEFT, 980, "screen", "hook",
         width_hint=342)
    # the pinned references (page layer)
    text("ref_seconds_label", REFERENCES["seconds"]["label"], "inter", 36, "lime", "right", SPINE, 262,
         "page", "reference", width_hint=563)
    number_text("ref_seconds_number", "seconds", 48, "lime", "right", SPINE, 318, "page", "reference",
                prefix="≈", line_groups=[6])
    text("ref_people_label", REFERENCES["people"]["label"], "inter", 36, "cyan", "right", SPINE, 388,
         "page", "reference")
    number_text("ref_people_number", "people", 48, "cyan", "right", SPINE, 444, "page", "reference",
                prefix="≈", line_groups=[4])
    # the teach
    text("teach", "Each card multiplies the possible deals.", "inter", 40, "mistLight", "left", LEFT, 1400,
         "screen", "teach", spans=[["Each card multiplies the ", "mistLight"], ["possible deals", "lavender"],
                                   [".", "mistLight"]], width_hint=754)
    # the stops
    text("six_lead", "Six cards dealt.", "display", 56, "cyan", "left", LEFT, 1420, "screen", "stop", width_hint=395)
    text("six_line", "More possible deals than everyone alive.", "inter", 40, "mistLight", "left", LEFT, 1500,
         "screen", "stop", spans=[["More possible deals than ", "mistLight"], ["everyone alive", "cyan"],
                                  [".", "mistLight"]], width_hint=779)
    text("near_miss", "Ten cards. Still short.", "inter", 40, "mistLight", "left", LEFT, 1420, "screen", "stop",
         width_hint=403)
    text("eleven_lead", "Eleven cards dealt.", "display", 56, "lime", "left", LEFT, 1420, "screen", "stop",
         width_hint=487)
    text("eleven_line", "Past every second since the Big Bang.", "inter", 40, "mistLight", "left", LEFT, 1500,
         "screen", "stop", spans=[["Past ", "mistLight"], ["every second since the Big Bang", "lime"],
                                  [".", "mistLight"]], width_hint=731)
    text("reframe_line", "41 cards still in the deck.", "inter", 40, "mistLight", "left", LEFT, 1500,
         "screen", "stop", spans=[["41", "lavender"], [" cards still in the deck.", "mistLight"]], width_hint=469)
    # the run
    text("run_line", "One fewer place for every card.", "inter", 40, "mistLight", "left", LEFT, 1500,
         "screen", "run", spans=[["One fewer", "lavender"], [" place for every card.", "mistLight"]], width_hint=603)
    text("lead", "× 4  × 3  × 2  × 1", "mono", 64, "lavender", "left", LEFT, 1420, "screen", "run",
         pieces=["× 4", "× 3", "× 2", "× 1"], separator="  ", width_hint=691)
    text("whole_deck_lead", "The whole deck, dealt.", "display", 52, "violet", "left", LEFT, 1420, "screen", "stop",
         width_hint=534)
    # the block
    text("block_label", REFERENCES["factorial"]["label"], "inter", 36, "violet", "right", SPINE, 480,
         "screen", "block", width_hint=564)
    text("product_line", "52 × 51 × 50 × … × 2 × 1", "mono", 34, "lavender", "right", SPINE, 540, "screen", "block")
    number_text("block", "factorial", 80, "violet", "right", SPINE, None, "screen", "block",
                baselines=[640 + 96 * i for i in range(6)], line_groups=[4, 4, 4, 4, 4, 3], pitch=96,
                note="appears by the fold (kind fold), not by a stamp")
    text("note1", "Now a good shuffle", "inter", 44, "mistLight", "left", LEFT, 1430, "screen", "note", width_hint=409)
    text("note2", "every second since the Big Bang.", "inter", 44, "mistLight", "left", LEFT, 1485, "screen", "note",
         width_hint=700)
    # beat 1: a shuffle a second
    text("beat1_label", "Shuffles, one a second", "inter", 36, "lime", "right", SPINE, 262, "screen", "beat1",
         width_hint=393)
    number_text("beat1_number", "seconds", 48, "lime", "right", SPINE, 318, "screen", "beat1",
                prefix="≈", line_groups=[6], note="the same digits as ref_seconds_number, in the same place")
    text("beat1_atmost", "At most 1 in", "inter", 40, "mistLight", "left", LEFT, 600, "screen", "beat1", width_hint=219)
    number_text("beat1_wall", "bound1", 40, "starlight", "right", SPINE, None, "screen", "beat1",
                baselines=[660, 710, 760], line_groups=[7, 7, 3], pitch=50)
    text("beat1_of", "of the possible deals.", "inter", 40, "mistLight", "left", LEFT, 830, "screen", "beat1",
         width_hint=408)
    # beat 2: every Planck time
    text("beat2_label", REFERENCES["planck_shuffles"]["label"], "inter", 36, "lime", "right", SPINE, 262,
         "screen", "beat2", width_hint=468)
    number_text("beat2_number", "planck_shuffles", 40, "lime", "right", SPINE, None, "screen", "beat2",
                prefix="≈", baselines=[318, 366, 414], line_groups=[7, 7, 7], pitch=48)
    text("beat2_still", "Still about 1 in", "inter", 40, "mistLight", "left", LEFT, 600, "screen", "beat2",
         width_hint=262)
    number_text("beat2_tenmillion", "bound2", 64, "starlight", "right", SPINE, 680, "screen", "beat2",
                line_groups=[3], width_hint=345.6)
    text("beat2_of", "of the possible deals.", "inter", 40, "mistLight", "left", LEFT, 755, "screen", "beat2",
         width_hint=408)
    # the ending
    text("end1", "Every good shuffle", "display", 60, "starlight", "left", LEFT, 880, "screen", "ending", width_hint=517)
    text("end2", "is almost certainly new.", "display", 60, "starlight", "left", LEFT, 952, "screen", "ending",
         width_hint=646)


# ------------------------------------------------------------------------------------------------
# Events
# ------------------------------------------------------------------------------------------------

EVENTS: list[dict] = []
_seq = 0


def ev(frame: int, kind: str, **fields) -> dict:
    global _seq
    assert isinstance(frame, int) and 0 <= frame < TOTAL_FRAMES, (frame, kind)
    e = {"frame": frame, "kind": kind}
    e.update(fields)
    e["_seq"] = _seq
    _seq += 1
    EVENTS.append(e)
    return e


def stamp(frame, tid, knock=None):
    assert tid in TEXTS, tid
    return ev(frame, "stamp", id=tid, knock=knock)


def unstamp(frame, *tids):
    for t in tids:
        assert t in TEXTS, t
    return ev(frame, "unstamp", ids=list(tids), glow_decay_frames=9)


def type_(frame, tid, ref, line_groups, baselines):
    n = len(REFERENCES[ref]["groups"])
    assert sum(line_groups) == n, (tid, line_groups, n)
    fpg = 3  # 60 digits per second = one group of three every 3 frames
    gf = [frame + fpg * i for i in range(n)]
    lines, i0 = [], 0
    for b, c in zip(baselines, line_groups):
        lines.append({"baseline": b, "groups": [i0, i0 + c - 1]})
        i0 += c
    return ev(frame, "type", id=tid, ref=ref, groups=n, frames_per_group=fpg, group_frames=gf,
              end=frame + fpg * n, lines=lines)


def nebula(frame, colour, fade_frames=72):
    return ev(frame, "nebula", colour=colour, fade_frames=fade_frames, end=frame + fade_frames)


ROWS: list[dict] = []


def deal(k: int, land: int, duration: int, becomes: str, rest_frames: int = 0, muted=False):
    """Card k flies for `duration` frames, lands at `land` (the row stamps), then becomes its cell."""
    start = land - duration
    digits = str(orders(k))
    prev_digits = len(str(orders(k - 1))) if k >= 2 else 2
    reads = 53 - k
    cell_frame = None
    if becomes == "cell":
        cell_frame = land
    elif becomes == "card":
        cell_frame = land + rest_frames
    cam = camera_target(k)
    if k <= 11:
        cam = {"s": 1.0, "tx": 0.0, "ty": 0.0, "width48": cam["width48"]}  # the page is still until the run
    rate = None  # the deal rate at this card; filled in by set_rates() once every landing is known
    row = {
        "k": k, "digits": digits, "ndigits": len(digits), "new_digits": len(digits) - prev_digits,
        "groups": groups_of_three(digits), "display": " ".join(groups_of_three(digits)),
        "width48": cam["width48"], "left48": round(SPINE - cam["width48"], 2),
        "multiplier": reads, "cell_text": f"× {reads}" if k >= 2 else None,
        "reads": reads, "deck_after": 52 - k,
        "flight_start": start, "land": land, "flight_frames": duration, "cell": cell_frame,
        "becomes": becomes, "rests_as_card": becomes == "card", "rest_frames": rest_frames if becomes == "card" else 0,
        "rate_per_s": rate, "camera": {"s": cam["s"], "tx": cam["tx"], "ty": cam["ty"]},
        "partials": len(digits),
    }
    ROWS.append(row)
    row["flight_event"] = ev(start, "flight", id=f"card{k}", k=k, reads=reads, land=land, duration=duration,
                             deck_after=52 - k, becomes=becomes, rest_frames=row["rest_frames"], cell=cell_frame,
                             rate_per_s=rate, pan=0.3, to="spine" if k == 1 else "cell")
    r = ev(land, "row", k=k, ndigits=len(digits), new_digits=row["new_digits"], click=True,
           muted=muted, partials=len(digits))
    if k == 10:
        r["dull"] = True
    if k == 52:
        r["silent"] = True  # nothing changes in the tone: the number did not change
    if cell_frame is not None:
        ev(cell_frame, "cell", k=k, text=row["cell_text"])
    return row


def build_events():
    # ---- frame 0 and the knock ------------------------------------------------------------------
    nebula(0, "lavender", fade_frames=0)
    stamp(1, "deck52", knock="knock")
    stamp(1, "deck_label")

    # ---- the hook, on dotted eighths at 150 BPM ----------------------------------------------
    stamp(T("0.20"), "hook1", knock="knock")
    stamp(T("0.50"), "hook2", knock="knock")
    stamp(T("0.80"), "hook3", knock="knock")
    stamp(T("1.10"), "hook4", knock="knock")
    stamp(T("1.70"), "hook_sub")
    ev(T("2.40"), "marker", id="cover")

    # ---- 3.00 the re-set: hook out, the deck travels to the column head --------------------------
    unstamp(T("3.00"), "hook1", "hook2", "hook3", "hook4", "hook_sub", "deck_label")
    ev(T("3.00"), "deck_travel", end=T("3.40"), **{"from": "hook", "to": "head"}, ease="out",
       flick_long=True, reads=52)

    # ---- 3.40 the target types in; the lime rule drops 3.70-3.90 --------------------------------
    stamp(T("3.40"), "ref_seconds_label")
    t = type_(T("3.40"), "ref_seconds_number", "seconds", [6], [318])
    assert t["end"] == T("3.70")
    ev(T("3.70"), "rule", ref="seconds", action="drop", end=T("3.90"), x=RULE_X["seconds"],
       y_top=RULE_Y_TOP["seconds"], alpha=0.45)

    # ---- the teach: cards 1-5 --------------------------------------------------------------------
    deal(1, T("4.10"), 18, "row")                       # 3.80 -> 4.10; the card becomes row 1
    deal(2, T("5.00"), 18, "card", rest_frames=24)      # 4.70 -> 5.00; cell x 51 at 5.40
    stamp(T("5.40"), "teach")                           # the frame the first multiplier cell appears
    deal(3, T("5.80"), 18, "card", rest_frames=6)       # 5.50 -> 5.80; x 50 at 5.90
    deal(4, T("6.50"), 18, "card", rest_frames=6)       # 6.20 -> 6.50; x 49 at 6.60
    stamp(T("6.50"), "ref_people_label")
    t = type_(T("6.50"), "ref_people_number", "people", [4], [444])
    assert t["end"] == T("6.70")
    ev(T("6.70"), "rule", ref="people", action="drop", end=T("6.90"), x=RULE_X["people"],
       y_top=RULE_Y_TOP["people"], alpha=0.45)
    deal(5, T("7.10"), 18, "card", rest_frames=6)       # 6.80 -> 7.10; x 48 at 7.20
    unstamp(T("7.10"), "teach")

    # ---- 7.70 stop: card 6 passes everyone alive -------------------------------------------------
    row6 = deal(6, T("7.70"), 18, "card", rest_frames=6)  # 7.40 -> 7.70; x 47 at 7.80
    make_pass(row6, "people", "cyan", hold_until=T("10.10"), rule_full_until=T("10.10"),
              swell_partial=11)
    nebula(T("7.70"), "cyan")
    stamp(T("8.00"), "six_lead", knock="knock")
    stamp(T("8.20"), "six_line")

    # ---- 10.10 cards 7-9; the cyan rule and band have done their work --------------------------
    unstamp(T("10.10"), "six_lead", "six_line")
    ev(T("10.10"), "band", ref="people", k=6, action="fade", end=T("10.50"), alpha=BAND_ALPHA)
    ev(T("10.10"), "rule", ref="people", action="fade", end=T("10.50"))
    nebula(T("10.10"), "lavender")
    deal(7, T("10.40"), 18, "card", rest_frames=6)      # 10.10 -> 10.40; x 46 at 10.50
    deal(8, T("10.80"), 18, "card", rest_frames=6)      # 10.50 -> 10.80; x 45 at 10.90
    deal(9, T("11.20"), 18, "card", rest_frames=6)      # 10.90 -> 11.20; x 44 at 11.30

    # ---- 11.60 the near miss: card 10 -----------------------------------------------------------
    deal(10, T("11.60"), 18, "card", rest_frames=6, muted=True)   # 11.30 -> 11.60; x 43 at 11.70
    ev(T("11.60"), "rule", ref="seconds", action="pulse_dim", end=T("11.80"), alpha_peak=0.7)
    stamp(T("11.80"), "near_miss")

    # ---- 12.80 the payoff: card 11 passes every second since the Big Bang ----------------------
    row11 = deal(11, T("12.80"), 18, "card", rest_frames=6)       # 12.50 -> 12.80; x 42 at 12.90
    make_pass(row11, "seconds", "lime", hold_until=T("16.80"), rule_full_until=T("17.00"),
              swell_partial=19, silence_until=T("13.20"))
    nebula(T("12.80"), "lime")
    unstamp(T("12.80"), "near_miss")
    ev(T("12.80"), "tone", action="cut", fade_ms=10)
    stamp(T("13.10"), "eleven_lead", knock="big")
    ev(T("13.20"), "tone", action="return", partials=19, ramp_frames=36, swell_partial=19, swell_db=6,
       swell_frames=72)
    stamp(T("13.30"), "eleven_line")

    # ---- 15.20 the reframe: the deck's own number says what is left ----------------------------
    unstamp(T("15.20"), "eleven_line")
    stamp(T("15.20"), "reframe_line")
    ev(T("15.20"), "bloom", id="deck_number", end=T("15.60"), glow=2.0, reads=41)

    # ---- 16.80 the run ----------------------------------------------------------------------------
    unstamp(T("16.80"), "eleven_lead", "reframe_line")
    ev(T("16.80"), "band", ref="seconds", k=11, action="fade", end=T("17.20"), alpha=BAND_ALPHA)
    stamp(T("16.80"), "run_line")
    nebula(T("16.80"), "lavender")
    # rows 12-16 every 0.2 s, 17-24 every 0.1 s, 25-48 every 0.05 s; flights 0.25 s; cells on landing
    land = T("16.80")
    for k in range(12, 17):
        deal(k, land, 15, "cell")
        land += 12
    land = T("17.70")
    for k in range(17, 25):
        deal(k, land, 15, "cell")
        land += 6
    land = T("18.45")
    for k in range(25, 49):
        deal(k, land, 15, "cell")
        land += 3
    assert ROWS[-1]["land"] == T("19.60")
    ev(T("17.00"), "rule", ref="seconds", action="fade", end=T("17.40"))
    # the riffle ritardando: rows 49-51 at 0.4 s, the lead stamping at screen scale
    for piece, (k, when) in enumerate([(49, "20.00"), (50, "20.40"), (51, "20.80"), (52, "21.20")]):
        deal(k, T(when), 15, "card", rest_frames=6)
        pieces = TEXTS["lead"]["pieces"]
        ev(T(when), "lead", piece=piece, text=pieces[piece], text_so_far=TEXTS["lead"]["separator"].join(pieces[:piece + 1]),
           k=k, id="lead")
    # 21.20 the final stop: row 52 identical to row 51; the deck is gone
    ev(T("21.20"), "deck_leave", tick=True)

    # ---- 21.60 the whole deck ----------------------------------------------------------------------
    unstamp(T("21.60"), "lead", "run_line")
    stamp(T("21.60"), "whole_deck_lead", knock="knock")

    # ---- 23.20 the fold -----------------------------------------------------------------------------
    unstamp(T("23.20"), "whole_deck_lead")
    fold_start, fold_end = T("23.20"), T("24.00")
    n_groups = len(REFERENCES["factorial"]["groups"])
    stagger = Fraction(2, 100) * FPS  # 1.2 frames
    group_starts = [fold_start + round(stagger * i) for i in range(n_groups)]
    assert group_starts[-1] < fold_end
    cam52 = ROWS[51]["camera"]
    ev(fold_start, "fold", end=fold_end, groups=n_groups, stagger_s=0.02, group_starts=group_starts,
       ease="in-out", **{"from": {"size": 48, "baseline": row_base(52), "x_right": SPINE, "page_scale": cam52["s"],
                                  "page_tx": cam52["tx"], "page_ty": cam52["ty"]},
                         "to": {"id": "block", "size": 80, "baselines": TEXTS["block"]["baselines"],
                                "line_groups": TEXTS["block"]["line_groups"], "x_right": SPINE}},
       fade_out={"ids": ["rows_1_51", "cells", "ref_seconds_label", "ref_seconds_number",
                         "ref_people_label", "ref_people_number"],
                 "start": fold_start, "end": fold_start + 24},
       knock_at=fold_end, tick_per_group=True)
    nebula(T("23.20"), "violet")
    stamp(T("24.00"), "block_label")
    stamp(T("24.00"), "product_line")

    # ---- 24.60 the set-up note ---------------------------------------------------------------------
    stamp(T("24.60"), "note1")
    stamp(T("24.60"), "note2")

    # ---- 26.80 beat 1: a shuffle a second ------------------------------------------------------------
    unstamp(T("26.80"), "block", "block_label", "product_line", "note1", "note2")
    nebula(T("26.80"), "lime")
    stamp(T("27.00"), "beat1_label")
    t = type_(T("27.00"), "beat1_number", "seconds", [6], [318])
    assert t["end"] == T("27.30")
    stamp(T("27.40"), "beat1_atmost")
    t = type_(T("27.60"), "beat1_wall", "bound1", [7, 7, 3], [660, 710, 760])
    assert t["end"] == T("28.45")
    ev(T("27.60"), "tone", action="target", partials=51, ramp_frames=30)
    stamp(T("28.50"), "beat1_of")

    # ---- 30.30 beat 2: every Planck time --------------------------------------------------------------
    unstamp(T("30.30"), "beat1_atmost", "beat1_wall", "beat1_of", "beat1_label", "beat1_number")
    stamp(T("30.30"), "beat2_label")
    t = type_(T("30.30"), "beat2_number", "planck_shuffles", [7, 7, 7], [318, 366, 414])
    assert t["end"] == T("31.35")
    stamp(T("31.40"), "beat2_still")
    stamp(T("31.60"), "beat2_tenmillion", knock="knock")
    ev(T("31.60"), "tone", action="target", partials=8, ramp_frames=30)
    stamp(T("31.90"), "beat2_of")

    # ---- 33.40 the ending -------------------------------------------------------------------------------
    unstamp(T("33.40"), "beat2_still", "beat2_tenmillion", "beat2_of")
    ev(T("33.40"), "fade", ids=["beat2_label", "beat2_number"], end=T("33.80"), to_alpha=0.45)
    stamp(T("33.40"), "end1", knock="knock")
    ev(T("33.40"), "tone", action="dim", dim_db=3, end=T("35.80"))
    stamp(T("33.80"), "end2", knock="knock")

    # ---- 35.80 out; 36.40 the deck returns; 36.80 the loop ----------------------------------------------
    unstamp(T("35.80"), "end1", "end2")
    ev(T("35.80"), "fade", ids=["beat2_label", "beat2_number"], end=T("36.20"), to_alpha=0.0)
    nebula(T("35.80"), "lavender", fade_frames=54)   # 0.9 s: complete before the last frame
    ev(T("35.80"), "tone", action="target", partials=2, ramp_frames=36)
    ev(T("36.40"), "deck_return", tick=True, box="hook")
    ev(TOTAL_FRAMES - 1, "marker", id="last", equals_frame=0)


def set_rates():
    """The deal rate at each card: 60 / the shorter of the gaps to the previous and the next landing."""
    lands = [r["land"] for r in ROWS]
    for i, r in enumerate(ROWS):
        gaps = []
        if i > 0:
            gaps.append(lands[i] - lands[i - 1])
        if i + 1 < len(ROWS):
            gaps.append(lands[i + 1] - lands[i])
        r["rate_per_s"] = round(FPS / min(gaps), 4)
        r.pop("flight_event")["rate_per_s"] = r["rate_per_s"]


def make_pass(row: dict, ref: str, colour: str, hold_until: int, rule_full_until: int,
              swell_partial: int, silence_until: int | None = None):
    k, land = row["k"], row["land"]
    rule_x = RULE_X[ref]
    row_left = row["left48"]
    assert row_left < rule_x, (k, ref)
    crossing = [i for i in range(row["ndigits"]) if digit_left_edge(row["ndigits"], i) < rule_x - 1e-6]
    assert crossing == [0], "exactly the leading digit crosses the rule"
    fill_end = land + 12
    band_fade = {"start": hold_until, "end": hold_until + 24}
    base = row_base(k)
    ev(land, "band", ref=ref, k=k, action="fill", end=fill_end, alpha=BAND_ALPHA,
       x=[rule_x, SPINE], y=[base - 40, base + 14])
    ev(land, "rule", ref=ref, action="full", end=rule_full_until)
    p = ev(land, "pass", k=k, ref=ref, colour=colour, rule_x=rule_x, row_left=row_left,
           overshoot_px=round(rule_x - row_left, 2), digits_left_of_rule=1, hold_until=hold_until,
           rule_full=[land, rule_full_until],
           band={"fill": [land, fill_end], "fade": [band_fade["start"], band_fade["end"]], "alpha": BAND_ALPHA,
                 "x": [rule_x, SPINE], "y": [base - 40, base + 14]},
           dim_rows={"rows": list(range(1, k)), "alpha": 0.4, "start": land, "end": land + 30},
           pulse={"start": land, "end": land + 18, "from_y": RULE_Y_TOP[ref], "to_y": base + 12},
           lead_digit_bloom={"start": land, "end": land + 15, "glow": 1.2},
           swell_partial=swell_partial, swell_db=6, swell_frames=72)
    if silence_until is not None:
        p["silence_until"] = silence_until
    return p


def digit_left_edge(ndigits: int, i: int, size: float = 48) -> float:
    """Left edge (page x) of digit i (0-based from the left) of a right-aligned n-digit number."""
    w = mono_width(ndigits, size)
    left = SPINE - w
    ngroups = -(-ndigits // 3)
    first_group_len = ndigits - 3 * (ngroups - 1)
    if i < first_group_len:
        g, j = 0, i
    else:
        g = 1 + (i - first_group_len) // 3
        j = (i - first_group_len) % 3
    digits_before = (first_group_len if g > 0 else 0) + 3 * max(0, g - 1) + j
    return left + digits_before * MONO_ADVANCE * size + g * MONO_GAP * size


# ------------------------------------------------------------------------------------------------
# Camera and sound (derived)
# ------------------------------------------------------------------------------------------------

def build_camera() -> dict:
    still = {"s": 1.0, "tx": 0.0, "ty": 0.0}
    targets = [dict(frame=ROWS[10]["land"], k=11, **still, note="row 11: the page is still"),
               dict(frame=ROWS[11]["flight_start"], k=11, **still, note="card 12 leaves the deck: the camera starts")]
    for row in ROWS[11:]:
        targets.append(dict(frame=row["land"], k=row["k"], **row["camera"]))
    targets.append(dict(frame=T("23.20"), k=52, **ROWS[51]["camera"], note="held until the fold"))
    assert all(a["frame"] < b["frame"] for a, b in zip(targets, targets[1:]))
    assert ROWS[11]["camera"]["s"] < 1.0, "the page pulls back from row 12"
    return {
        "spine_x": SPINE, "newest_baseline_screen_y": CAMERA_NEWEST_Y, "fit_width_px": CAMERA_FIT_PX,
        "rule": "s = min(1, 660 / width48); tx = 800 (1 - s); ty = 1240 - rowBase(k) s; screen = (800 + (x - 800) s, y s + ty)",
        "still_until": ROWS[11]["flight_start"], "still_after": T("23.20"),
        "spline": "PCHIP (monotone cubic) in s and ty through the targets; flat before the first and after the last",
        "targets": targets,
    }


def build_sound() -> dict:
    cues, partials = [], []
    by_id = {}
    for e in EVENTS:
        k = e["kind"]
        f = e["frame"]
        if k == "stamp":
            cues.append({"frame": f, "cue": "stamp", "id": e["id"]})
            if e.get("knock"):
                cues.append({"frame": f, "cue": "knock", "level": e["knock"], "id": e["id"]})
        elif k == "type":
            for g in e["group_frames"]:
                cues.append({"frame": g, "cue": "typing", "id": e["id"]})
        elif k == "deck_travel":
            cues.append({"frame": f, "cue": "flick_long", "frames": e["end"] - f})
        elif k == "flight":
            cues.append({"frame": f, "cue": "flick", "k": e["k"], "rate_per_s": e["rate_per_s"], "pan": e["pan"]})
        elif k == "row":
            cues.append({"frame": f, "cue": "click", "k": e["k"], "muted": e.get("muted", False),
                         "run_index": e["k"] - 12 if e["k"] >= 12 else None})
            partials.append({"frame": f, "partials": e["partials"], "ramp_frames": 15,
                             "new": e["new_digits"], "dull": e.get("dull", False), "unchanged": e.get("silent", False)})
        elif k == "pass":
            cues.append({"frame": f, "cue": "pulse", "ref": e["ref"], "frames": e["pulse"]["end"] - f})
            if "silence_until" in e:
                cues.append({"frame": f, "cue": "silence", "until": e["silence_until"],
                             "except": "the big knock at 'Eleven cards dealt.'"})
        elif k == "tone":
            if e["action"] == "cut":
                partials.append({"frame": f, "partials": 0, "ramp_frames": 0, "cut": True, "fade_ms": e["fade_ms"]})
            elif e["action"] == "return":
                partials.append({"frame": f, "partials": e["partials"], "ramp_frames": e["ramp_frames"],
                                 "swell": {"partial": e["swell_partial"], "db": e["swell_db"], "frames": e["swell_frames"]}})
            elif e["action"] == "target":
                partials.append({"frame": f, "partials": e["partials"], "ramp_frames": e["ramp_frames"]})
            elif e["action"] == "dim":
                partials.append({"frame": f, "dim_db": e["dim_db"], "end": e["end"]})
        elif k == "fold":
            for g in e["group_starts"]:
                cues.append({"frame": g, "cue": "fold_tick"})
            cues.append({"frame": e["knock_at"], "cue": "knock", "level": "knock", "id": "block"})
        elif k == "deck_leave":
            cues.append({"frame": f, "cue": "deck_out"})
        elif k == "deck_return":
            cues.append({"frame": f, "cue": "deck_tick"})
    # the card-6 swell lives on the pass (the tone is not cut there)
    for e in EVENTS:
        if e["kind"] == "pass" and "silence_until" not in e:
            partials.append({"frame": e["frame"], "swell": {"partial": e["swell_partial"], "db": e["swell_db"],
                                                            "frames": e["swell_frames"]}})
    partials.insert(0, {"frame": 0, "partials": 2, "ramp_frames": 0, "note": "the deck's two digits"})
    cues.sort(key=lambda c: c["frame"])
    partials.sort(key=lambda c: c["frame"])
    return {"cues": cues, "partials": partials,
            "notes": ["150 BPM; the stamps are the percussion; the tone's fundamental is E2 (82.4 Hz)",
                      "row 10's new partials enter dull (-9 dB) and come up to level when row 11 lands",
                      "the tone is cut at 768 and returns at 792 with 19 partials; row 52 adds nothing",
                      "the click in the run is 1 dB quieter per row to a floor of -30 dBFS (run_index)",
                      "the flick's gain in the run follows 1 / (1 + rate_per_s / 8)",
                      "render 2.0 s past 36.80 and fold the tail onto 0.00-2.00"]}


# ------------------------------------------------------------------------------------------------
# Consistency
# ------------------------------------------------------------------------------------------------

RUN_GRID_KINDS = {"flight", "row", "cell"}


def check_events():
    frames = [e["frame"] for e in EVENTS]
    assert frames == sorted(frames), "events are sorted by frame"
    assert all(isinstance(f, int) for f in frames)
    assert min(frames) == 0 and max(frames) == TOTAL_FRAMES - 1
    # the grid
    for e in EVENTS:
        f = e["frame"]
        if f == 1 or (e["kind"] == "marker" and e["id"] == "last"):
            continue  # the knock one frame after the loop point; the last frame itself
        if e["kind"] in RUN_GRID_KINDS and e["k"] >= 12:
            assert f % 3 == 0, e
        else:
            assert f % 6 == 0, ("off the 6-frame grid", e)
        for key in ("end", "land", "cell", "hold_until", "silence_until", "knock_at"):
            if e.get(key) is not None:
                assert isinstance(e[key], int) and 0 <= e[key] <= TOTAL_FRAMES, (key, e)
    # no two stamps / types of one id; every stamped id is unstamped (or folds / fades) before it is re-used
    seen = {}
    for e in EVENTS:
        if e["kind"] in ("stamp", "type"):
            assert e["id"] not in seen, ("stamped twice", e["id"])
            seen[e["id"]] = e["frame"]
    # every text id is referenced by an event
    referenced = set()
    for e in EVENTS:
        if "id" in e and e["id"] in TEXTS:
            referenced.add(e["id"])
        for t in e.get("ids", []):
            referenced.add(t)
        if e["kind"] == "fold":
            referenced.add(e["to"]["id"])
            referenced.update(t for t in e["fade_out"]["ids"] if t in TEXTS)
    missing = set(TEXTS) - referenced
    assert not missing, ("texts never used", sorted(missing))
    # every unstamp refers to something on screen at that frame
    on = {}
    for e in EVENTS:
        if e["kind"] in ("stamp", "type"):
            on[e["id"]] = e["frame"]
        elif e["kind"] == "fold":
            on[e["to"]["id"]] = e["end"]
            for t in e["fade_out"]["ids"]:
                if t in TEXTS:
                    assert t in on, ("fold fades something not on screen", t)
                    del on[t]
        elif e["kind"] == "lead":
            on.setdefault(e["id"], e["frame"])
        elif e["kind"] == "deck_leave":
            del on["deck52"]
        elif e["kind"] == "fade" and e["to_alpha"] == 0:
            for t in e["ids"]:
                assert t in on, ("fade of something not on screen", t)
                del on[t]
        elif e["kind"] == "unstamp":
            for t in e["ids"]:
                assert t in on and on[t] <= e["frame"], ("unstamp of something not on screen", t, e["frame"])
                del on[t]
    assert not on, ("still on screen at the end", on)
    # rows
    assert len(ROWS) == 52 and [r["k"] for r in ROWS] == list(range(1, 53))
    for r in ROWS:
        assert int(r["digits"]) == orders(r["k"]) and r["digits"] == str(orders(r["k"]))
        assert r["flight_start"] < r["land"] and (r["cell"] is None or r["cell"] >= r["land"])
    assert ROWS[51]["digits"] == str(FACT52) == ROWS[50]["digits"]
    lands = [r["land"] for r in ROWS]
    assert lands == sorted(lands) and len(set(lands)) == 52
    assert [r["land"] for r in ROWS[11:16]] == [1008, 1020, 1032, 1044, 1056]
    assert [r["land"] for r in ROWS[16:24]] == list(range(1062, 1110, 6))
    assert [r["land"] for r in ROWS[24:48]] == list(range(1107, 1177, 3))
    assert [r["land"] for r in ROWS[48:52]] == [1200, 1224, 1248, 1272]
    assert all(r["becomes"] == "cell" for r in ROWS[11:48])
    assert all(r["becomes"] == "card" and r["rest_frames"] == 6 for r in ROWS[48:52])
    assert ROWS[0]["becomes"] == "row" and ROWS[1]["rest_frames"] == 24
    assert all(r["rest_frames"] == 6 for r in ROWS[2:11])
    assert max(sum(1 for o in ROWS if o["flight_start"] < r["land"] <= o["land"]) for r in ROWS) == 5, "five cards in the air"
    assert [ROWS[k - 1]["rate_per_s"] for k in (12, 17, 25, 48, 49, 52)] == [5.0, 10.0, 20.0, 20.0, 2.5, 2.5]
    assert all(r["camera"] == {"s": 1.0, "tx": 0.0, "ty": 0.0} for r in ROWS[:11])
    # geometry the passes rely on
    assert abs(RULE_X["seconds"] - 209.6) < 1e-6 and abs(RULE_X["people"] - 468.8) < 1e-6
    assert [ROWS[k - 1]["left48"] for k in (5, 6, 10, 11)] == [512.0, 440.0, 238.4, 166.4]
    assert abs(ROWS[51]["camera"]["s"] - 0.29) < 0.005
    # holds
    hold = lambda a, b: T(b) - T(a)
    assert hold("7.70", "10.10") == 144    # 2.4 s at card 6
    assert hold("11.60", "12.80") == 72    # 1.2 s on the near miss
    assert hold("12.80", "16.80") == 240   # 4.0 s at the payoff
    assert hold("21.20", "23.20") == 120   # 2.0 s on the wedge
    assert hold("24.00", "26.80") == 168   # 2.8 s on the block
    assert hold("28.50", "30.30") == 108   # 1.8 s on beat 1's sentence
    assert hold("31.90", "33.40") == 90    # 1.5 s on beat 2's
    assert hold("33.80", "35.80") == 120   # 2.0 s on the ending
    # wording
    for tid, t in TEXTS.items():
        assert "order" not in t["text"].lower(), (tid, "the word 'orders' must not appear on screen")
        assert "!" not in t["text"], tid
        if "spans" in t:
            assert "".join(s for s, _ in t["spans"]) == t["text"], tid
            assert all(c in COLOURS for _, c in t["spans"]), tid
        assert t["face"] in FACES and t["colour"] in COLOURS, tid
    assert TEXTS["lead"]["separator"].join(TEXTS["lead"]["pieces"]) == TEXTS["lead"]["text"]
    # nebula: the last cross-fade is complete before the last frame, so frame 2207 == frame 0
    neb = [e for e in EVENTS if e["kind"] == "nebula"]
    assert neb[0]["frame"] == 0 and neb[0]["colour"] == "lavender"
    assert neb[-1]["colour"] == "lavender" and neb[-1]["end"] <= TOTAL_FRAMES - 1
    for a, b in zip(neb, neb[1:]):
        assert a["end"] <= b["frame"], ("nebula cross-fades overlap", a, b)
    # everything that is lit is out before the last frame
    for e in EVENTS:
        if e["kind"] == "fade" and e["to_alpha"] == 0:
            assert e["end"] <= TOTAL_FRAMES - 1


# ------------------------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------------------------

def describe(e: dict) -> str:
    k = e["kind"]
    if k == "stamp":
        s = TEXTS[e["id"]]["text"]
        return f"stamp {e['id']}: \"{s}\"" + (f"  [{e['knock']}]" if e.get("knock") else "")
    if k == "unstamp":
        return "out: " + ", ".join(e["ids"])
    if k == "type":
        return f"type {e['id']}: {e['groups']} groups to {e['end']}"
    if k == "flight":
        return f"card {e['k']} (reads {e['reads']}) flies, lands {e['land']}; deck reads {e['deck_after'] or 'nothing'}"
    if k == "row":
        extra = " muted" if e.get("muted") else (" silent" if e.get("silent") else "")
        return f"row {e['k']}: {ROWS[e['k'] - 1]['display']} ({e['ndigits']} digits){extra}"
    if k == "cell":
        return f"cell {e['k']}: {e['text']}"
    if k == "pass":
        return f"PASS row {e['k']} crosses {e['ref']} ({e['colour']}) by {e['overshoot_px']} px" + \
               (f"; silence to {e['silence_until']}" if "silence_until" in e else "")
    if k == "rule":
        return f"rule {e['ref']} {e['action']} to {e.get('end')}"
    if k == "band":
        return f"band {e['ref']} {e['action']} to {e['end']}"
    if k == "nebula":
        return f"nebula -> {e['colour']} over {e['fade_frames']} frames"
    if k == "lead":
        return f"lead: \"{e['text_so_far']}\""
    if k == "fold":
        return f"fold: 23 groups to the block, settles {e['end']}"
    if k == "fade":
        return f"fade {', '.join(e['ids'])} to {e['to_alpha']} by {e['end']}"
    if k == "tone":
        return "tone " + " ".join(f"{a}={b}" for a, b in e.items() if a not in ("frame", "kind", "_seq"))
    if k == "deck_travel":
        return f"deck travels hook -> column head, to {e['end']}"
    if k == "bloom":
        return f"bloom {e['id']} (reads {e['reads']}) to {e['end']}"
    if k == "marker":
        return f"marker {e['id']}"
    return k


def table() -> str:
    lines = [f"{'frame':>6} {'s':>7}  event", "-" * 90]
    for e in EVENTS:
        lines.append(f"{e['frame']:>6} {e['frame'] / FPS:>7.3f}  {describe(e)}")
    return "\n".join(lines)


def build() -> dict:
    ratios = check_numbers()
    build_texts()
    build_events()
    set_rates()
    EVENTS.sort(key=lambda e: (e["frame"], e["_seq"]))
    check_events()
    for e in EVENTS:
        del e["_seq"]
    camera = build_camera()
    sound = build_sound()
    kinds = sorted({e["kind"] for e in EVENTS})
    return {
        "meta": {
            "title": "The Long Number", "fps": FPS, "frames": TOTAL_FRAMES, "width": WIDTH, "height": HEIGHT,
            "total_s": float(TOTAL_S), "slices": 4, "slice_frames": TOTAL_FRAMES // 4,
            "loop": {"first_frame": 0, "last_frame": TOTAL_FRAMES - 1, "identical": True,
                     "cover_frame": T("2.40")},
            "colours": COLOURS, "faces": FACES,
            "stamp_bloom": {"peak_glow": 1.2, "frames_at_peak": 2, "settle_frames": 15, "rest_glow": 0.5,
                            "bloom_stroke_factor": 2.5},
            "unstamp_glow_decay_frames": 9,
            "typing": {"digits_per_s": 60, "frames_per_group": 3},
            "row_style": {"face": "mono", "size": 48, "colour": "starlight", "align": "right", "x": SPINE,
                          "layer": "page", "placeholder_alpha": 0.45,
                          "placeholder_rule": "only the three ≈ figures (seconds, people, planck_shuffles) dim their zeros"},
            "cell_style": {"face": "mono", "size": 40, "colour": "lavender", "align": "left", "x": CELL_X,
                           "layer": "page"},
            "lead_style": {"face": "mono", "size": 64, "colour": "lavender", "align": "left", "x": LEFT,
                           "baseline": 1420, "layer": "screen"},
            "event_kinds": kinds,
            "ratios": {k: float(f"{v:.6g}") for k, v in ratios.items()},
        },
        "geometry": {
            "spine_x": SPINE, "left_x": LEFT, "cell_x": CELL_X, "column_right_x": COLUMN_RIGHT,
            "row_baseline": {"first": ROW_BASE_0, "pitch": ROW_PITCH, "rule": "640 + 60 (k - 1)"},
            "mono": {"advance_em": MONO_ADVANCE, "group_gap_em": MONO_GAP},
            "deck": {"hook": HOOK_DECK, "head": HEAD_DECK,
                     "number": {"face": "monoB", "colour": "lavender", "baseline_rule": "centre y + 0.365 x size"},
                     "card": {"ratio": "poker (96 x 134)", "face": "lavender at 8 % over the background",
                              "edge": "2 px lavender at 60 %", "glow": "10 px at 30 %", "corner_radius": "8 % of width"}},
            "rules": {ref: {"x": RULE_X[ref], "y_top": RULE_Y_TOP[ref], "width_px": 3, "alpha_rest": 0.45,
                            "alpha_pass": 1.0, "bottom": "newest row baseline + 12",
                            "bottom_before_rows": row_base(1) + 12, "colour": "lime" if ref == "seconds" else "cyan"}
                      for ref in ("seconds", "people")},
            "band": {"alpha": BAND_ALPHA, "x": "rule x to the spine", "y": "crossing row baseline - 40 to + 14"},
            "nebula": {"centre": [700, 800], "radius": 900, "peak": 0.10, "cross_fade_frames": 72},
            "safe": {"baseline_max_y": 1536, "ink_max_y": 1548, "x_min": 132, "x_max": 948,
                     "button_column": {"x_min": 948, "y": [780, 1536]}, "max_text_width": 808,
                     "cover_key_text_y": [240, 1680]},
            "camera_fit_px": CAMERA_FIT_PX, "camera_newest_baseline_y": CAMERA_NEWEST_Y,
        },
        "numbers": {"rows": ROWS, "references": REFERENCES},
        "texts": TEXTS,
        "camera": camera,
        "sound": sound,
        "events": EVENTS,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "build", "story.json"))
    ap.add_argument("--table", action="store_true", help="print the timeline table")
    args = ap.parse_args(argv)
    story = build()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(story, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    if args.table:
        print(table())
    n = len(EVENTS)
    print(f"wrote {args.out}: {TOTAL_FRAMES} frames, {n} events, {len(TEXTS)} texts, "
          f"{len(story['sound']['cues'])} sound cues, {len(story['meta']['event_kinds'])} event kinds")
    return 0


if __name__ == "__main__":
    sys.exit(main())

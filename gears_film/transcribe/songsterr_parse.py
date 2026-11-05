"""Read a Songsterr tab (the files songsterr_fetch.js saved) and show its rhythm on a step grid.

    python3 -I songsterr_parse.py summary <dir> [nbars]
        tracks, tempo map, time signatures and section markers, notes per bar per track.
    python3 -I songsterr_parse.py grid <dir> <track> <bars> [--res 16] [--notes 36,38]
        one row per note (General MIDI drum number, or string/fret for pitched tracks) per bar, on a
        grid of 1/res notes: x hit, ~ off-grid, g grace note, d dead note, h held (tie), . nothing.
        <bars> is like 12-19 (1-based, inclusive).
    python3 -I songsterr_parse.py stream <dir> <track> <bars> [--res 16] [--notes 37]
        the bars concatenated into one step stream, its run lengths (hits between rests), and the
        shortest period that repeats exactly: the cycle you are looking for.

Songsterr's format, as far as this needs it: measures[].voices[].beats[]; beat.duration = [num, den]
of a whole note; beat.dots; beat.tuplet (3 = triplet); beat.graceNote (takes no time); beat.palmMute;
notes[].fret (drum number for drum tracks), notes[].string, .rest, .tie, .dead, .ghost; a measure's
"signature" holds from there on; "marker" names a section; automations.tempo is the tempo map.
Run with `python3 -I`: the files are downloaded data.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from fractions import Fraction as F

TUPLET = {3: F(2, 3), 5: F(4, 5), 6: F(4, 6), 7: F(4, 7), 9: F(8, 9)}


def load(d):
    files = sorted(glob.glob(os.path.join(d, "track_*.json")), key=lambda p: int(os.path.basename(p)[6:-5]))
    if not files:
        sys.exit(f"no track_*.json in {d}")
    return [json.load(open(f)) for f in files]


def is_drums(t):
    return t.get("instrumentId") == 1024 or t.get("instrument") == "Drums"


def signatures(t):
    out, sig = [], (4, 4)
    for m in t["measures"]:
        if "signature" in m:
            sig = tuple(m["signature"])
        out.append(sig)
    return out


def marker(m):
    mk = m.get("marker")
    return mk.get("text") if isinstance(mk, dict) else mk


def beat_len(b):
    d = F(b["duration"][0], b["duration"][1])
    if b.get("tuplet"):
        d *= TUPLET.get(b["tuplet"], F(1))
    if b.get("dots"):
        d *= F(3, 2) if b["dots"] == 1 else F(7, 4)
    return d


def events(t, mi, drums):
    """(position as a fraction of a whole note, label, flag) for measure mi (0-based)."""
    out = []
    for v in t["measures"][mi]["voices"]:
        pos = F(0)
        for b in v["beats"]:
            for n in b["notes"]:
                if n.get("rest"):
                    continue
                lab = str(n.get("fret")) if drums else f"s{n.get('string')}f{n.get('fret')}"
                flag = ("g" if b.get("graceNote") else "h" if n.get("tie") else "d" if n.get("dead")
                        else "x")
                out.append((pos, lab, flag))
            if not b.get("graceNote"):
                pos += beat_len(b)
    return out


def bar_grid(t, mi, res, drums, labels=None):
    """{label: row} for one bar; the row has one cell per 1/res note."""
    num, den = signatures(t)[mi]
    n = int(F(num, den) * res)
    rows = {}
    for pos, lab, flag in events(t, mi, drums):
        if labels and lab not in labels:
            continue
        cell = pos * res
        row = rows.setdefault(lab, ["."] * n)
        i = int(cell)
        if i >= n:
            continue
        if cell != int(cell):
            row[i] = "~" if row[i] == "." else row[i]
        elif row[i] in (".", "~") or flag == "x":
            row[i] = flag
    return rows, (num, den)


def opt(args, name, default):
    if name in args:
        return args[args.index(name) + 1]
    return default


def cmd_summary(d, nbars):
    tracks = load(d)
    for i, t in enumerate(tracks):
        print(f"track {i}: {t.get('name')} | {t.get('instrument')} | bars {len(t['measures'])} | tuning {t.get('tuning')}")
    ref = tracks[0]
    print("tempo map:", json.dumps(ref.get("automations", {}).get("tempo")))
    sig, last = None, None
    print("time signatures and markers (bar: signature marker):")
    for i, m in enumerate(ref["measures"]):
        if "signature" in m:
            sig = tuple(m["signature"])
        mk = marker(m)
        if mk or sig != last:
            print(f"  {i + 1}: {sig[0]}/{sig[1]} {mk or ''}")
            last = sig

    def notes(m):
        return sum(1 for v in m["voices"] for b in v["beats"] if not b.get("rest")
                   for x in b["notes"] if not x.get("rest") and not x.get("tie"))
    print(f"notes per bar, first {nbars} bars:")
    print("bar   " + " ".join(f"{str(t.get('name'))[:9]:>9}" for t in tracks))
    for i in range(min(nbars, len(ref["measures"]))):
        print(f"{i + 1:>3}   " + " ".join(f"{notes(t['measures'][i]):>9}" for t in tracks))


def cmd_grid(d, ti, bars, res, labels):
    t = load(d)[ti]
    drums = is_drums(t)
    lo, hi = (int(x) for x in bars.split("-")) if "-" in bars else (int(bars), int(bars))
    for mi in range(lo - 1, hi):
        rows, (num, den) = bar_grid(t, mi, res, drums, labels)
        print(f"-- bar {mi + 1}  {num}/{den}  {marker(t['measures'][mi]) or ''}")
        per_beat = max(1, res // den)
        for lab in sorted(rows, key=lambda s: (len(s), s)):
            row = rows[lab]
            cells = "".join(c if k % per_beat else "|" + c for k, c in enumerate(row))
            print(f"   {lab:>8} {cells}")


def cmd_stream(d, ti, bars, res, labels):
    t = load(d)[ti]
    drums = is_drums(t)
    lo, hi = (int(x) for x in bars.split("-")) if "-" in bars else (int(bars), int(bars))
    s = []
    for mi in range(lo - 1, hi):
        rows, (num, den) = bar_grid(t, mi, res, drums, labels)
        n = int(F(num, den) * res)
        bar = [0] * n
        for row in rows.values():
            for i, c in enumerate(row):
                if c in ("x", "d"):
                    bar[i] = 1
        s += bar
    print("steps:", len(s), "hits:", sum(s))
    print("".join("x" if v else "." for v in s))
    runs, cur = [], 0
    for v in s:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    print("run lengths (hits between rests):", runs)
    for p in range(1, len(s) // 2 + 1):
        if all(s[i] == s[i + p] for i in range(len(s) - p)):
            print(f"shortest exact period: {p} steps ({F(p, res)} whole notes)")
            break
    else:
        print("no exact period in this range")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd, d = argv[0], argv[1]
    res = int(opt(argv, "--res", 16))
    labels = opt(argv, "--notes", None)
    labels = set(labels.split(",")) if labels else None
    if cmd == "summary":
        return cmd_summary(d, int(argv[2]) if len(argv) > 2 and argv[2].isdigit() else 40)
    if cmd == "grid":
        return cmd_grid(d, int(argv[2]), argv[3], res, labels)
    if cmd == "stream":
        return cmd_stream(d, int(argv[2]), argv[3], res, labels)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

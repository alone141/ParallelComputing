"""Track every step of a record that does not play to a click, and write a per-bar tempo map.

    python3 beat_track.py <record_48k.wav> <first_step_s> <bars> <nominal_bpm> <steps_per_beat> <out.json>
        [--bands 3000-9000:1,60-205:1,600-2500:0.6] [--window 0.40] [--pin 0.03]

<bars> is the kept stretch's bar lengths in steps, as a comma list with optional repeats, e.g.
"5,7*20,6,7*4" is not supported; write "5,7x20" for twenty 5+7 pairs: each "a,bxN" group repeats the
group N times. Example for Tool's Schism, bars 4-51: "5,7x20;6,7x4".
<first_step_s> is the record time of step 0 (from fit_record.py: the first segment's start).

How it works: a Viterbi tracker over candidate times (±window around each step's nominal time, 4 ms
apart). Emission = onset strength, a weighted sum of band-limited onset envelopes (each normalised to
its 99th percentile), max within ±6 ms. Transition = a strong penalty on the change of interval from
the previous step (the local tempo may drift, but smoothly) plus a weak pull to the nominal step. Step
0 is pinned within ±pin seconds of first_step_s. A step-level tracker cannot tell the grid from the
grid shifted by whole steps; that phase comes from first_step_s and from smoothness, and must be
checked with a pattern-specific comb (see the skill).

Output: {"record_first_step_s", "steps_per_beat", "step_record_times_s": [...], "bars": [{bar index,
record_step, steps, bpm (boundary-defined), record_start_s, film_s}], "film_total_s", "frames_60fps"}
and a printed per-bar list to paste into the song module as one Segment per bar (contiguous; the
audio is then one uncut stretch of the record from first_step_s, and the picture follows the band).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_record import Envelopes, read_wav  # noqa: E402


def parse_bars(spec):
    out = []
    for group in spec.split(";"):
        group = group.strip()
        if not group:
            continue
        reps = 1
        if "x" in group:
            group, r = group.rsplit("x", 1)
            reps = int(r)
        vals = [int(v) for v in group.split(",") if v.strip()]
        out += vals * reps
    return out


def opt(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def track(mono, sr, t0, n, bpm, spb, bands, window=0.40, pin=0.03, lam_local=4000.0, lam_nom=4.0, res=0.004):
    a0 = max(0.0, t0 - 2.0)
    step0 = 60.0 / (bpm * spb)
    seg = mono[int(a0 * sr): int((t0 + n * step0 * 1.08 + 2.0) * sr)]
    envs = Envelopes(seg, sr)
    onset = sum(w * envs.band(lo, hi) / (np.percentile(envs.band(lo, hi), 99) + 1e-12) for lo, hi, w in bands)
    tf = envs.t + a0
    hop = tf[1] - tf[0]
    w6 = max(1, int(round(0.006 / hop)))
    oms = sliding_window_view(np.pad(onset, w6, mode="edge"), 2 * w6 + 1).max(axis=1)[: len(onset)]
    offs = np.arange(-window, window + 1e-9, res)
    K = len(offs)
    back = np.zeros((n + 1, K), dtype=np.int32)
    prev = t0 + offs
    score = np.interp(prev, tf, oms) - 200.0 * (offs / pin) ** 2
    prev_prev = None
    for i in range(1, n + 1):
        cand = t0 + i * step0 + offs
        dt = cand[:, None] - prev[None, :]
        valid = dt > 0.5 * step0
        pen = lam_nom * np.log(np.where(valid, dt / step0, 1.0)) ** 2
        if prev_prev is not None:
            prev_int = prev - prev_prev[back[i - 1]]
            pen = pen + lam_local * np.log(np.where(valid, dt / prev_int[None, :], 1.0)) ** 2
        tot = score[None, :] - pen
        tot[~valid] = -1e9
        back[i] = np.argmax(tot, axis=1)
        score = tot[np.arange(K), back[i]] + np.interp(cand, tf, oms)
        prev_prev, prev = prev, cand
    idx = np.zeros(n + 1, dtype=int)
    idx[n] = int(np.argmax(score))
    for i in range(n, 0, -1):
        idx[i - 1] = back[i][idx[i]]
    return np.array([t0 + i * step0 + offs[idx[i]] for i in range(n + 1)])


def main(argv):
    if len(argv) < 6:
        print(__doc__)
        return 2
    wav, t0, bars_spec, bpm, spb, out = argv[:6]
    t0, bpm, spb = float(t0), float(bpm), int(spb)
    bars = parse_bars(bars_spec)
    n = sum(bars)
    bands = [(float(b.split(":")[0].split("-")[0]), float(b.split(":")[0].split("-")[1]),
              float(b.split(":")[1]) if ":" in b else 1.0)
             for b in opt(argv, "--bands", "3000-9000:1,60-205:1,600-2500:0.6").split(",")]
    sr, x = read_wav(wav)
    T = track(x.mean(1), sr, t0, n, bpm, spb, bands, float(opt(argv, "--window", 0.40)), float(opt(argv, "--pin", 0.03)))
    iv = np.diff(T)
    const = T[0] + np.arange(n + 1) * (T[-1] - T[0]) / n
    print(f"tracked {n} steps: {T[0]:.4f} → {T[-1]:.4f} s; step {iv.min() * 1000:.1f}-{iv.max() * 1000:.1f} ms; "
          f"largest drift from one constant tempo {np.abs(T - const).max() * 1000:.0f} ms")
    segs, k = [], 0
    for b, steps in enumerate(bars):
        a, z = T[k], T[k + steps]
        segs.append({"bar": b, "record_step": k, "steps": steps, "bpm": round(steps * 60.0 / (spb * (z - a)), 4),
                     "record_start_s": round(float(a), 5), "film_s": round(float(a - T[0]), 5)})
        k += steps
    total = float(T[-1] - T[0])
    json.dump({"record_first_step_s": round(float(T[0]), 5), "steps_per_beat": spb, "film_total_s": round(total, 5),
               "frames_60fps": round(total * 60), "bars": segs,
               "step_record_times_s": [round(float(t), 5) for t in T]}, open(out, "w"), indent=1)
    print(f"{len(segs)} bars · film {total:.4f} s = {round(total * 60)} frames · wrote {out}")
    print("per-bar segments (record_step, steps, bpm):")
    for s in segs:
        print(f"    ({s['record_step']}, {s['steps']}, {s['bpm']:.4f}),")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

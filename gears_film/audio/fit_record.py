"""Fit a real recording to the film's step grid, using the engine's record_map.json.

    python3 fit_record.py <record_48k.wav> <record_map.json> <fit.json>

record_map.json (exported by `python3 -m gears_film export-record-map songs/<slug>.py`) gives the
nominal bpm, steps per beat, the film segments (which record steps are kept) and one or two "anchors":
a distinctive instrument, its frequency band, and the RECORD steps on which it strikes.

What this does, per segment: a comb of the anchor's hits is slid over a band-limited onset envelope of
the recording to find the segment's start time in the record and its tempo (a ±2.5% search around the
nominal bpm, refined to 0.001 bpm). The first segment's fit also gives the record's first-step offset.
It prints how peaked each fit is (score on the grid vs half a step off: a ratio under ~1.5 means the
anchor did not lock and you should pick another anchor or band) and writes fit.json:

    {"offset_s": ..., "bpm": ..., "segments": [{"record_step": ..., "start_s": ..., "bpm": ...,
     "score": ..., "score_half_off": ...}, ...]}

Paste each segment's bpm into the song data's tempo map and re-render the picture, then build the
audio with edit_record.py. The WAV must be 48 kHz PCM (16 or 24 bit); convert with
`ffmpeg -i song.mp3 -ar 48000 -c:a pcm_s24le record_48k.wav`.
"""
from __future__ import annotations

import json
import sys
import wave

import numpy as np

N, H = 512, 64                       # 10.7 ms analysis window, 1.33 ms hop: tight enough for ±1 ms fits


def read_wav(path):
    with wave.open(path) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / (1 << 15)
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        x = np.where(x >= 1 << 23, x - (1 << 24), x).astype(np.float64) / (1 << 23)
    else:
        raise SystemExit("use 16- or 24-bit PCM")
    return sr, x.reshape(-1, ch)


class Envelopes:
    """Band-limited log-spectral flux (onset strength) of the mono mix, by frequency band."""

    def __init__(self, mono, sr):
        self.sr = sr
        self.mono = np.asarray(mono, float)
        win = np.hanning(N)
        nfr = (len(mono) - N) // H
        idx = np.arange(N)[None, :] + H * np.arange(nfr)[:, None]
        self.spec = np.abs(np.fft.rfft(mono[idx] * win, axis=1))
        self.freqs = np.fft.rfftfreq(N, 1 / sr)
        self.t = np.arange(nfr) * H / sr            # frame START: a rise shows at or after the attack
        self.cache = {}

    def band(self, lo, hi):
        """Onset strength in [lo, hi) Hz. Wide bands above 150 Hz come from the short STFT (its bins
        are sr/512 = 93.75 Hz apart). A band that is low or narrower than two bins (a bass pedal, one
        pitch) is band-passed exactly in the frequency domain instead, and its analytic envelope is
        followed: no frame latency, no empty bins."""
        key = (lo, hi)
        if key in self.cache:
            return self.cache[key]
        if lo < 150 or hi - lo < 2 * self.freqs[1]:
            env = self._analytic_envelope(lo, hi)
            nfr = len(self.t)
            e = env[: nfr * H].reshape(nfr, H).mean(1)
            le = np.log1p(e / (np.median(e) + 1e-12))
            self.cache[key] = np.maximum(np.diff(le, prepend=le[:1]), 0)
        else:
            b = np.log1p(300 * self.spec[:, (self.freqs >= lo) & (self.freqs < hi)])
            self.cache[key] = np.maximum(np.diff(b, axis=0, prepend=b[:1]), 0).sum(1)
        return self.cache[key]

    def _analytic_envelope(self, lo, hi):
        X = np.fft.rfft(self.mono)
        f = np.fft.rfftfreq(len(self.mono), 1 / self.sr)
        X[(f < lo) | (f >= hi)] = 0
        X[1:-1] *= 2                                  # analytic signal: positive frequencies doubled
        return np.abs(np.fft.ifft(X, n=len(self.mono)))[: len(self.mono)]

    def score(self, env, times):
        return float(np.interp(times, self.t, env).mean())


class Comb:
    """The anchors that cover one segment: a list of (envelope, its mean, relative hit steps). The
    score of a candidate grid is the sum over anchors of the mean onset strength at the hits,
    each normalised by its band's overall level, so a cymbal band and a drum-body band weigh alike."""

    def __init__(self, envs, parts):
        self.envs = envs
        self.parts = [(env, float(env.mean()) + 1e-9, np.asarray(rel, float)) for env, rel in parts]

    def score(self, t0, step_s, shift_steps=0.0):
        return sum(self.envs.score(env, t0 + (rel + shift_steps) * step_s) / m for env, m, rel in self.parts)


def fit_segment(comb, bpm0, spb, t_guess, search_s, bpm_pct=2.5):
    """Best (start_s, bpm) for a segment, starting near t_guess."""
    best = (-1.0, t_guess, bpm0)
    for bpm in np.arange(bpm0 * (1 - bpm_pct / 100), bpm0 * (1 + bpm_pct / 100) + 1e-9, bpm0 * 0.0005):
        step_s = 60.0 / (bpm * spb)
        for t0 in np.arange(t_guess - search_s, t_guess + search_s, 0.002):
            s = comb.score(t0, step_s)
            if s > best[0]:
                best = (s, t0, bpm)
    for d_bpm, d_t in ((bpm0 * 0.0001, 0.0005), (bpm0 * 0.00002, 0.0001)):
        _, t0, bpm = best
        for bb in np.arange(bpm - 10 * d_bpm, bpm + 10 * d_bpm, d_bpm):
            for tt in np.arange(t0 - 10 * d_t, t0 + 10 * d_t, d_t):
                s = comb.score(tt, 60.0 / (bb * spb))
                if s > best[0]:
                    best = (s, tt, bb)
    s, t0, bpm = best
    half = comb.score(t0, 60.0 / (bpm * spb), 0.5)
    return t0, bpm, s, half


def level_jump_time(mono, sr, t_center, span_s=0.6, min_db=3.0):
    """Where the level steps up near t_center, on MEDIANS of 5 ms RMS windows (0.3 s before, 0.5 s
    after), so one loud hit cannot move it: the earliest window within ±span_s whose step is within
    1 dB of the largest. Returns (time, step in dB), or (None, best step) when no step reaches
    min_db: then there is no jump to anchor on."""
    w = int(0.005 * sr)
    i0, i1 = max(0, int((t_center - span_s - 0.3) * sr)), int((t_center + span_s + 0.5) * sr)
    seg = mono[i0:i1]
    nw = len(seg) // w
    rms = np.sqrt((seg[:nw * w].reshape(nw, w) ** 2).mean(1)) + 1e-9
    nb, na = 60, 100                           # 0.3 s before, 0.5 s after, in 5 ms windows
    db = 20 * np.log10(rms)

    def trimmed(v):                            # mean without the loudest quarter: one hit cannot move it
        s = np.sort(v)
        return s[: max(1, int(len(s) * 0.75))].mean()
    steps = np.full(nw, -99.0)
    for j in range(nb, nw - na):
        steps[j] = trimmed(db[j:j + na]) - trimmed(db[j - nb:j])
    best = float(steps.max())
    if best < min_db:
        return None, best
    j = int(np.argmax(steps))                  # the contrast peaks where both windows are pure
    return (i0 + j * w) / sr, best


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    wav, map_path, out = argv
    rm = json.load(open(map_path))
    sr, x = read_wav(wav)
    envs_mono = x.mean(1)
    envs = Envelopes(envs_mono, sr)
    bpm0, spb = float(rm["bpm"]), int(rm["steps_per_beat"])
    anchors = rm["anchors"]
    print(f"{wav}: {len(x) / sr:.2f} s · nominal {bpm0} bpm · {spb} steps per beat · anchors: "
          + ", ".join(a["name"] for a in anchors))
    segs = rm["segments"]
    MIN_HITS = 16                              # an anchor with fewer hits in a stretch does not vote

    def comb_for(r0, n, bpm_centre=None):
        parts, names = [], []
        for a in anchors:
            inside = [s - r0 for s in a["record_steps"] if r0 <= s < r0 + n]
            if len(inside) >= MIN_HITS:
                parts.append((envs.band(*a["band_hz"]), inside))
                names.append(a["name"])
        return (Comb(envs, parts) if parts else None), " + ".join(names)

    # Contiguous segments (no record steps cut between them) form a RUN: one stretch of the record
    # that plays straight through. Tempo is fitted over the whole run (a long comb pins it), then
    # each segment is refined a little; the segment boundaries inside a run then DEFINE each
    # segment's tempo, so the audio can be used uncut and the picture's clock matches it exactly.
    runs = []
    for i, seg in enumerate(segs):
        if runs and seg["record_step"] == segs[i - 1]["record_step"] + segs[i - 1]["steps"]:
            runs[-1].append(i)
        else:
            runs.append([i])
    result = {"segments": [None] * len(segs), "runs": []}
    prev_end_s, prev_end_step, prev_bpm = None, None, bpm0
    for run in runs:
        r0 = int(segs[run[0]]["record_step"])
        n = sum(int(segs[i]["steps"]) for i in run)
        bpm_nom = float(segs[run[0]].get("bpm", bpm0))
        comb, name = comb_for(r0, n)
        if comb is None:
            print(f"run at record step {r0} ({n} steps): no anchor with {MIN_HITS}+ hits, skipped")
            for i in run:
                result["segments"][i] = {"record_step": int(segs[i]["record_step"]), "start_s": None, "bpm": bpm_nom}
            continue
        if prev_end_s is None:
            guess, search = float(rm.get("first_step_guess_s", 0.5)), float(rm.get("first_search_s", 1.5))
        else:                                  # continue the record's own clock across the cut
            step_prev = 60.0 / (prev_bpm * spb)
            guess = prev_end_s + (r0 - prev_end_step) * step_prev
            # Within half a step of the clock, unless the map says the record has a bigger edit
            # here: a comb of hits is often ambiguous by exactly one step, and the clock is the
            # best tie-breaker.
            search = float(segs[run[0]].get("search_s", 0.45 * step_prev))
        if segs[run[0]].get("level_jump"):
            # The stretch begins with a band entry or a cut: the level jump is the strongest
            # evidence of its downbeat, which precedes the jump by the attack time (0 to ~50 ms).
            tj, step_db = level_jump_time(envs_mono, sr, guess)
            if tj is None:
                print(f"record step {r0}: level_jump set but no step of 3 dB or more near {guess:.3f} s "
                      f"(best {step_db:.1f} dB); fitting on the clock window instead")
            else:
                print(f"record step {r0}: level jump of {step_db:.1f} dB at {tj:.4f} s ({(tj - guess) * 1000:+.0f} ms from the clock)")
                guess, search = tj - 0.025, 0.03
        t0, bpm, s, half = fit_segment(comb, bpm_nom, spb, guess, search)
        step_s = 60.0 / (bpm * spb)
        s_m1, s_p1 = comb.score(t0, step_s, -1.0), comb.score(t0, step_s, 1.0)
        print(f"run: record steps {r0}-{r0 + n} ({len(run)} segment{'s' if len(run) > 1 else ''}, anchors {name}): "
              f"starts {t0:.4f} s · {bpm:.4f} bpm · score {s:.2f} vs half-step off {half:.2f} (ratio {s / max(half, 1e-9):.2f})")
        if max(s_m1, s_p1) > 0.9 * s:
            print(f"   AMBIGUOUS by one step: score one step earlier {s_m1:.2f}, later {s_p1:.2f}. "
                  f"If the stretch begins with an entry or a cut, set \"level_jump\": true on its first "
                  f"segment; otherwise use a more percussive anchor.")
        tj, step_db = level_jump_time(envs_mono, sr, t0)
        print(f"   level: a {step_db:.1f} dB step at {tj:.4f} s ({(tj - t0) * 1000:+.0f} ms from the fitted start)"
              if tj is not None else
              f"   level: no step near the start (largest {step_db:.1f} dB): the music continues across this boundary")
        # per-segment refinement inside the run: a small window around the run's clock
        starts, bpms = [], []
        for i in run:
            rs, ns = int(segs[i]["record_step"]), int(segs[i]["steps"])
            g = t0 + (rs - r0) * step_s
            comb_i, name_i = comb_for(rs, ns)
            if comb_i is not None and len(run) > 1:
                ti, bi, si, hi = fit_segment(comb_i, bpm, spb, g, 0.06, bpm_pct=0.6)
                note = f"anchors {name_i}, ratio {si / max(hi, 1e-9):.2f}"
            else:
                ti, bi, note = g, bpm, "from the run"
            starts.append(ti)
            bpms.append(bi)
            print(f"   segment {i} (record step {rs}, {ns} steps): starts {ti:.4f} s · {bi:.4f} bpm · {note}")
        # the boundaries define the tempos inside the run (the last segment keeps its fitted bpm)
        for j, i in enumerate(run):
            ns = int(segs[i]["steps"])
            if j + 1 < len(run):
                bpm_i = ns * 60.0 / (spb * (starts[j + 1] - starts[j]))
                if abs(bpm_i / bpms[j] - 1) > 0.012:
                    print(f"   note: segment {i}'s boundary tempo {bpm_i:.3f} differs from its fitted {bpms[j]:.3f} by more than 1.2%")
            else:
                bpm_i = bpms[j]
            result["segments"][i] = {"record_step": int(segs[i]["record_step"]), "start_s": round(starts[j], 5),
                                     "bpm": round(bpm_i, 4), "contiguous_with_previous": j > 0}
            print(f"   -> segment {i}: start {starts[j]:.4f} s, bpm {bpm_i:.4f}{' (boundary-defined)' if j + 1 < len(run) else ''}")
        last = run[-1]
        prev_bpm = bpms[-1]
        prev_end_s = starts[-1] + int(segs[last]["steps"]) * 60.0 / (prev_bpm * spb)
        prev_end_step = r0 + n
        result["runs"].append({"record_step": r0, "steps": n, "start_s": round(t0, 5), "bpm": round(bpm, 4),
                               "score": round(s, 3), "score_half_off": round(half, 3)})
    first = result["segments"][0]
    result["offset_s"] = first["start_s"]
    result["bpm"] = first["bpm"]
    json.dump(result, open(out, "w"), indent=1)
    print("wrote", out)
    print("paste into the song module: first_click_offset_s =", first["start_s"], "· per segment bpm =",
          [sg["bpm"] for sg in result["segments"]])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

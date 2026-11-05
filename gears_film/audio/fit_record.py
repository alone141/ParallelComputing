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
        win = np.hanning(N)
        nfr = (len(mono) - N) // H
        idx = np.arange(N)[None, :] + H * np.arange(nfr)[:, None]
        self.spec = np.abs(np.fft.rfft(mono[idx] * win, axis=1))
        self.freqs = np.fft.rfftfreq(N, 1 / sr)
        self.t = np.arange(nfr) * H / sr            # frame START: a rise shows at or after the attack
        self.cache = {}

    def band(self, lo, hi):
        key = (lo, hi)
        if key not in self.cache:
            b = np.log1p(300 * self.spec[:, (self.freqs >= lo) & (self.freqs < hi)])
            self.cache[key] = np.maximum(np.diff(b, axis=0, prepend=b[:1]), 0).sum(1)
        return self.cache[key]

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


def level_jump_time(mono, sr, t_center):
    """The first 5 ms window within ±0.3 s of t_center whose RMS reaches -3 dB of the RMS of the two
    seconds that follow t_center: where a band entry or a cut lands."""
    w = int(0.005 * sr)
    a0, a1 = max(0, int((t_center - 0.3) * sr)), int((t_center + 2.0) * sr)
    seg_rms = np.sqrt((mono[a0:a1] ** 2).mean()) + 1e-9
    win = mono[a0:int((t_center + 0.3) * sr)]
    nw = len(win) // w
    rms = np.sqrt((win[:nw * w].reshape(nw, w) ** 2).mean(1))
    j = int(np.argmax(rms > seg_rms * 10 ** (-3 / 20)))
    return (a0 + j * w) / sr


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
    result = {"segments": []}
    first_guess = float(rm.get("first_step_guess_s", 0.5))
    for i, seg in enumerate(rm["segments"]):
        r0, n = int(seg["record_step"]), int(seg["steps"])
        bpm_seg = float(seg.get("bpm", bpm0))
        parts, names = [], []
        for a in anchors:                      # every anchor with hits inside this segment
            inside = [s - r0 for s in a["record_steps"] if r0 <= s < r0 + n]
            if len(inside) >= 8:
                parts.append((envs.band(*a["band_hz"]), inside))
                names.append(a["name"])
        if not parts:
            print(f"segment {i} (record step {r0}, {n} steps): no anchor hits, skipped")
            result["segments"].append({"record_step": r0, "start_s": None, "bpm": bpm_seg})
            continue
        comb = Comb(envs, parts)
        name = " + ".join(names)
        if i == 0:
            guess, search = first_guess, max(first_guess, 1.0)
        else:                                  # continue the record's own clock from the last fit
            prev = result["segments"][-1]
            step_prev = 60.0 / (prev["bpm"] * spb)
            guess = prev["start_s"] + (r0 - prev["record_step"]) * step_prev
            # Within half a step of the clock, unless the map says the record has a bigger edit
            # here: a comb of hits is often ambiguous by exactly one step (off-beat pings, kicks
            # under snare slots), and the clock is the best tie-breaker.
            search = float(seg.get("search_s", 0.45 * step_prev))
        if seg.get("level_jump"):
            # The segment begins with a band entry or a cut: the level jump is the strongest evidence
            # of its downbeat. A downbeat precedes the jump by the attack time (0 to ~50 ms), so the
            # comb only refines inside that window; one-step ambiguities cannot reach it.
            tj = level_jump_time(envs_mono, sr, guess)
            guess, search = tj - 0.025, 0.03
        t0, bpm, s, half = fit_segment(comb, bpm_seg, spb, guess, search)
        step_s = 60.0 / (bpm * spb)
        s_m1, s_p1 = comb.score(t0, step_s, -1.0), comb.score(t0, step_s, 1.0)
        print(f"segment {i} (record step {r0}, {n} steps, anchor {name}): starts {t0:.4f} s · "
              f"{bpm:.4f} bpm · score {s:.2f} vs half-step off {half:.2f} (ratio {s / max(half, 1e-9):.2f})")
        if max(s_m1, s_p1) > 0.9 * s:
            print(f"   AMBIGUOUS by one step: score one step earlier {s_m1:.2f}, later {s_p1:.2f}. "
                  f"If the segment begins with an entry or a cut, set \"level_jump\": true on it; "
                  f"otherwise use a more percussive anchor.")
        tj = level_jump_time(envs_mono, sr, t0)
        print(f"   level: first 5 ms window within -3 dB of the segment's level at {tj:.4f} s "
              f"({(tj - t0) * 1000:+.0f} ms from the fitted start)")
        result["segments"].append({"record_step": r0, "start_s": round(t0, 5), "bpm": round(bpm, 4),
                                   "score": round(s, 3), "score_half_off": round(half, 3),
                                   "score_minus_1_step": round(s_m1, 3), "score_plus_1_step": round(s_p1, 3)})
    first = result["segments"][0]
    result["offset_s"] = first["start_s"]
    result["bpm"] = first["bpm"]
    json.dump(result, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

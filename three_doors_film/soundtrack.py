#!/usr/bin/env python3
"""Three Doors: the score, generated from build/story.json (numpy only; ffmpeg for the loudness measurement).

    python soundtrack.py [--story build/story.json] [--out out/soundtrack.wav] [--report out/soundtrack_report.json]

The palette is the film's own and quiet: the picture's events are the only instruments.
  hook, subline, rule, board   a soft stamp of band-passed noise as a line lands
  pick                         a dry tick (2.6 kHz) as your pick lights
  knock                        the host's door opening: a 200 Hz knock with a 900 Hz edge so phones hear it
  reveal                       a short breath of noise (1.5 kHz) as the two doors open
  chime_stay / chime_switch    a small bell on each tally step, E5 for STAY and G#5 for SWITCH
  close                        a quiet thud as the doors close
  tick                         one dry tick per round in the sprint, over a noise texture that follows the speed
                               of the count (silent at the pauses, gone when the count settles)
  payoff                       one low E minor chord with a knock under it, the film's loudest moment
  settle                       a quiet low tone as the scoreboard fades
In the accelerating rounds (4 to 30) the knock, breath and thud get quieter as the rounds shorten; the chimes
keep their level, because they are the information. The room is a short convolution reverb on everything but
the payoff chord. Nothing rings past the last frame and nothing sounds before the first cue, so the score is at
its frame-0 state (silence) when the picture loops.

Master: the mix is measured with ffmpeg's ebur128, gained to -15 LUFS integrated, passed through a soft-knee
look-ahead limiter at -1.5 dBFS and measured again; the report records every pass. The phone check high-passes
the mix at 300 Hz and requires every cue to peak above -30 dBFS within 50 ms of its frame.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 48000
FPS = 60
SPF = SR // FPS

LEVELS = {   # peak dBFS of each element before the master
    "hook": -18.0, "subline": -24.0, "rule": -22.0, "board": -26.0,
    "pick": -22.0, "knock": -12.0, "reveal": -26.0, "chime_stay": -16.0, "chime_switch": -16.0, "close": -26.0,
    "tick": -24.0, "texture": -30.0, "payoff": -7.0, "payoff_knock": -14.0, "settle": -24.0,
}
ROOM_WET = 0.12
ROOM_SEED = 1392
NOISE_SEED = 7
TARGET_LUFS = -15.0
CEILING_DB = -1.5
LOOKAHEAD_MS = 5.0
PHONE_HP_HZ = 300.0
PHONE_THRESHOLD_DB = -30.0
PHONE_WINDOW_MS = 50.0
E5, GS5 = 659.255, 830.609


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-12))


def amp(d: float) -> float:
    return 10.0 ** (d / 20.0)


# ------------------------------------------------------------------------------------------------ filters
def biquad_coeffs(kind: str, f0: float, q: float, sr: int = SR):
    w = 2.0 * math.pi * f0 / sr
    s, c = math.sin(w), math.cos(w)
    alpha = s / (2.0 * q)
    if kind == "bandpass":
        b = (alpha, 0.0, -alpha)
    elif kind == "lowpass":
        b = ((1 - c) / 2, 1 - c, (1 - c) / 2)
    elif kind == "highpass":
        b = ((1 + c) / 2, -(1 + c), (1 + c) / 2)
    else:
        raise ValueError(kind)
    a0 = 1 + alpha
    return (b[0] / a0, b[1] / a0, b[2] / a0), (1.0, -2 * c / a0, (1 - alpha) / a0)


def biquad_run(x: np.ndarray, b, a) -> np.ndarray:
    b0, b1, b2 = b
    _, a1, a2 = a
    y = np.empty_like(x)
    z1 = z2 = 0.0
    for i in range(len(x)):
        xi = float(x[i])
        yi = b0 * xi + z1
        z1 = b1 * xi - a1 * yi + z2
        z2 = b2 * xi - a2 * yi
        y[i] = yi
    return y


_IR_CACHE: dict = {}


def biquad_ir(kind: str, f0: float, q: float, floor: float = 1e-9, max_len: int = SR) -> np.ndarray:
    key = (kind, round(f0, 6), round(q, 6))
    if key not in _IR_CACHE:
        b, a = biquad_coeffs(kind, f0, q)
        imp = np.zeros(max_len)
        imp[0] = 1.0
        h = biquad_run(imp, b, a)
        idx = np.nonzero(np.abs(h) > floor)[0]
        _IR_CACHE[key] = h[: (idx[-1] + 1) if len(idx) else 1].copy()
    return _IR_CACHE[key]


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    nfft = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[:n]


def biquad(x: np.ndarray, kind: str, f0: float, q: float, keep_tail: bool = True) -> np.ndarray:
    y = fft_convolve(x, biquad_ir(kind, f0, q))
    return y if keep_tail else y[: len(x)]


# ----------------------------------------------------------------------------------------------- elements
def env_exp(n: int, tau_s: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (tau_s * SR))


def fade_tail(sig: np.ndarray, ms: float = 1.0) -> np.ndarray:
    k = max(1, min(len(sig), int(SR * ms / 1000)))
    sig[-k:] *= np.linspace(1.0, 0.0, k)
    return sig


def normalise_peak(sig: np.ndarray, level_db: float) -> np.ndarray:
    p = float(np.max(np.abs(sig)))
    return sig * (amp(level_db) / p) if p > 0 else sig


def stamp_noise(rng, level_db: float, f0: float = 2600.0, dur_s: float = 0.025, tau_s: float = 0.008) -> np.ndarray:
    n = int(dur_s * SR)
    x = fade_tail(rng.standard_normal(n) * env_exp(n, tau_s))
    return normalise_peak(biquad(x, "bandpass", f0, 4.0), level_db)


def tick(rng, level_db: float) -> np.ndarray:
    n = int(0.008 * SR)
    x = fade_tail(rng.standard_normal(n) * env_exp(n, 0.002), 0.5)
    return normalise_peak(biquad(x, "bandpass", 2600.0, 6.0), level_db)


def knock(level_db: float, edge: float = 0.35) -> np.ndarray:
    """A 200 Hz damped sine over a 60 Hz one, with a short 900 Hz edge so the knock survives a phone speaker."""
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    x = (np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.05) + 0.8 * np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.08)
         + edge * np.sin(2 * np.pi * 900 * t) * np.exp(-t / 0.012))
    return normalise_peak(fade_tail(x, 5.0), level_db)


def breath(rng, level_db: float) -> np.ndarray:
    """The reveal: 90 ms of noise around 1.5 kHz with a soft attack."""
    n = int(0.09 * SR)
    env = np.minimum(1.0, np.arange(n) / (0.012 * SR)) * env_exp(n, 0.03)
    x = fade_tail(rng.standard_normal(n) * env, 4.0)
    return normalise_peak(biquad(x, "bandpass", 1500.0, 2.0), level_db)


def chime(level_db: float, f0: float) -> np.ndarray:
    """A small bell: the fundamental with a 2.76x partial, tau 0.22 s."""
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f0 * t) * np.exp(-t / 0.22) + 0.35 * np.sin(2 * np.pi * f0 * 2.76 * t) * np.exp(-t / 0.08)
    x *= np.minimum(1.0, t / 0.002)
    return normalise_peak(fade_tail(x, 10.0), level_db)


def thud(level_db: float) -> np.ndarray:
    """The doors closing: a 120 Hz thump with a 1.1 kHz tick on top."""
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 120 * t) * np.exp(-t / 0.04) + 0.5 * np.sin(2 * np.pi * 1100 * t) * np.exp(-t / 0.006)
    return normalise_peak(fade_tail(x, 5.0), level_db)


def chord(level_db: float, dur_s: float = 3.2) -> np.ndarray:
    """E minor, low: E2 B2 E3 G3 B3 with a few harmonics each, 0.25 s attack, decaying over dur_s."""
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f0, w in ((82.407, 1.0), (123.47, 0.8), (164.81, 0.7), (196.0, 0.55), (246.94, 0.5)):
        for h in range(1, 7):
            out += w / h ** 1.1 * np.sin(2 * np.pi * f0 * h * t + 0.37 * h)
    env = np.minimum(1.0, t / 0.25) * np.exp(-t / (dur_s / 3.2))
    out *= env
    return normalise_peak(fade_tail(out, 40.0), level_db)


def settle_tone(level_db: float, dur_s: float = 1.3) -> np.ndarray:
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    x = (np.sin(2 * np.pi * 164.81 * t) + 0.5 * np.sin(2 * np.pi * 329.63 * t) + 0.3 * np.sin(2 * np.pi * 494.44 * t))
    x *= np.minimum(1.0, t / 0.08) * np.exp(-t / (dur_s / 3.5))
    return normalise_peak(fade_tail(x, 30.0), level_db)


def place(bus: np.ndarray, start: int, sig: np.ndarray, gain: float = 1.0, pan: float = 0.0) -> None:
    theta = (pan + 1.0) * math.pi / 4.0
    gl, gr = math.cos(theta), math.sin(theta)
    m = max(gl, gr)
    gl, gr = gl / m, gr / m
    if start >= bus.shape[1] or start < 0:
        return
    end = min(start + len(sig), bus.shape[1])
    seg = sig[: end - start] * gain
    bus[0, start:end] += seg * gl
    bus[1, start:end] += seg * gr


def room_ir(seed: int = ROOM_SEED) -> np.ndarray:
    n = int(0.4 * SR)
    rng = np.random.default_rng(seed)
    out = np.zeros((2, n))
    for c in range(2):
        x = rng.standard_normal(n) * env_exp(n, 0.10)
        x = fade_tail(x, 10.0)
        y = biquad(x, "lowpass", 4000.0, math.sqrt(0.5), keep_tail=False)
        out[c] = y / math.sqrt(float(np.sum(y * y)))
    return out


# --------------------------------------------------------------------------------------------- the score
def door_pan(door: int) -> float:
    return (-0.35, 0.0, 0.35)[door]


def build(story: dict):
    n = story["meta"]["frames"] * SPF
    rng = np.random.default_rng(NOISE_SEED)
    dry = np.zeros((2, n))        # goes through the room
    direct = np.zeros((2, n))     # the chord, no room
    rounds = {r["k"]: r for r in story["rounds"]}
    durations = {r["k"]: r.get("duration") for r in story["rounds"] if r.get("duration")}
    placed = []

    def short_round_db(k: int) -> float:
        d = durations.get(k)
        if not d or d >= 120:
            return 0.0
        return 10.0 * math.log10(d / 120.0)      # -6 dB at 30 frames, -13 dB at 6

    for cue in story["sound"]:
        f, c, ident = cue["frame"], cue["cue"], cue["id"]
        s = f * SPF
        k = int(re.sub(r"\D", "", ident) or 0)
        r = rounds.get(k)
        pan = door_pan(r["pick"]) if r else 0.0
        if c == "hook":
            place(dry, s, stamp_noise(rng, LEVELS["hook"]))
        elif c == "subline":
            place(dry, s, stamp_noise(rng, LEVELS["subline"], 2200.0))
        elif c == "rule":
            place(dry, s, stamp_noise(rng, LEVELS["rule"], 2400.0))
        elif c == "board":
            place(dry, s, stamp_noise(rng, LEVELS["board"], 1800.0, 0.05, 0.015))
        elif c == "pick":
            place(dry, s, tick(rng, LEVELS["pick"]), pan=pan)
        elif c == "knock":
            place(dry, s, knock(LEVELS["knock"] + short_round_db(k)), pan=door_pan(r["host"]))
        elif c == "reveal":
            place(dry, s, breath(rng, LEVELS["reveal"] + short_round_db(k)))
        elif c == "chime_stay":
            place(dry, s, chime(LEVELS["chime_stay"], E5), pan=-0.25)
        elif c == "chime_switch":
            place(dry, s, chime(LEVELS["chime_switch"], GS5), pan=0.25)
        elif c == "close":
            place(dry, s, thud(LEVELS["close"] + short_round_db(k)))
        elif c == "tick":
            place(dry, s, tick(rng, LEVELS["tick"]), pan=0.3 * math.sin(k * 0.9))
        elif c == "payoff":
            place(direct, s, chord(LEVELS["payoff"]))
            place(dry, s, knock(LEVELS["payoff_knock"], 0.5))
        elif c == "settle":
            place(dry, s, settle_tone(LEVELS["settle"]))
        else:
            raise ValueError(f"unknown cue {c}")
        placed.append(c)

    # the sprint's texture: band-passed noise whose level follows the speed of the count
    sp = story["sprint"]
    counts = np.array(sp["count_by_frame"], dtype=float)
    speed = np.gradient(counts)
    speed = speed / max(1e-9, speed.max())
    t0 = sp["start"] * SPF
    m = (sp["end"] - sp["start"]) * SPF
    env = np.interp(np.arange(m) / SPF, np.arange(len(speed)), speed)
    noise = rng.standard_normal(m) * env
    tex = biquad(biquad(noise, "highpass", 800.0, 0.8, keep_tail=False), "lowpass", 3000.0, 0.8, keep_tail=False)
    tex = normalise_peak(tex, LEVELS["texture"]) * env     # the envelope shapes the texture after the filters too
    place(dry, t0, fade_tail(tex, 20.0))

    # the room on the dry bus
    ir = room_ir()
    wet = np.stack([fft_convolve(dry[c], ir[c])[:n] for c in range(2)])
    mix = dry + ROOM_WET * wet + direct
    return mix, {"cues_placed": len(placed), "texture_peak_db": LEVELS["texture"]}


# --------------------------------------------------------------------------------------- WAV, measure
def write_wav(path: str, data: np.ndarray, sr: int = SR) -> None:
    x = np.clip(np.asarray(data, dtype=np.float64), -1.0, 1.0)
    inter = np.ascontiguousarray(x.T).reshape(-1)
    ints = np.round(inter * (2 ** 23 - 1)).astype("<i4")
    raw = ints.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[0])
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(raw)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        ch, sw, sr, nfr = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nfr)
    if sw == 3:
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        ints = np.where(ints >= 2 ** 23, ints - 2 ** 24, ints)
        x = ints / 2.0 ** 23
    elif sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 2.0 ** 15
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2.0 ** 31
    else:
        raise ValueError(f"unsupported sample width {sw}")
    return x.reshape(-1, ch).T, sr


def measure(path: str) -> tuple[float, float]:
    """Integrated loudness (LUFS) and true peak (dBTP) of an audio file, by ffmpeg's ebur128."""
    proc = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-i", path, "-af", "ebur128=peak=true",
                           "-f", "null", "-"], capture_output=True, text=True)
    text = proc.stderr
    summary = text[text.rfind("Summary:"):] if "Summary:" in text else text
    m_i = re.search(r"I:\s+(-?[\d.]+|-inf)\s+LUFS", summary)
    m_tp = re.search(r"True peak:\s*\n\s*Peak:\s+(-?[\d.]+|-inf)\s+dBFS", summary)
    if not m_i or not m_tp:
        raise RuntimeError(f"could not parse ebur128 output for {path}:\n{text[-2000:]}")
    return float(m_i.group(1)), float(m_tp.group(1))


def _phone_filter(mono: np.ndarray) -> np.ndarray:
    return fft_convolve(mono, biquad_ir("highpass", PHONE_HP_HZ, math.sqrt(0.5)))[: len(mono)]


def event_peaks(stereo: np.ndarray, story: dict, window_ms: float = PHONE_WINDOW_MS) -> list[dict]:
    mono = stereo.mean(axis=0)
    hp = _phone_filter(mono)
    half = int(SR * window_ms / 1000)
    out = []
    for cue in story["sound"]:
        c = cue["frame"] * SPF
        lo, hi = max(0, c - half), min(len(mono), c + half + 1)
        out.append({"frame": cue["frame"], "t": round(cue["frame"] / FPS, 4), "cue": cue["cue"], "id": cue["id"],
                    "peak_db": round(db(np.max(np.abs(mono[lo:hi]))), 2),
                    "hp300_peak_db": round(db(np.max(np.abs(hp[lo:hi]))), 2)})
    return out


def phone_check(wav_path: str, story_json_path: str, threshold_db: float = PHONE_THRESHOLD_DB):
    with open(story_json_path) as fh:
        story = json.load(fh)
    stereo, sr = read_wav(wav_path)
    if sr != SR:
        raise ValueError(f"{wav_path} is {sr} Hz, expected {SR}")
    peaks = event_peaks(stereo, story)
    misses = [p for p in peaks if not (p["hp300_peak_db"] > threshold_db)]
    return len(misses) == 0, misses


# ------------------------------------------------------------------------------------------------ master
def dc_block(stereo: np.ndarray) -> np.ndarray:
    h = biquad_ir("highpass", 10.0, math.sqrt(0.5))
    return np.stack([fft_convolve(ch, h)[: stereo.shape[1]] for ch in stereo])


def trailing_max(a: np.ndarray, w: int) -> np.ndarray:
    m = a.copy()
    c = 1
    while c < w:
        s = min(c, w - c)
        m = np.maximum(m, np.concatenate([np.zeros(s), m[:-s]]))
        c += s
    return m


def release_envelope(att: np.ndarray, r_per_sample: float) -> np.ndarray:
    e = att.copy()
    s = 1
    n = len(e)
    while s < n:
        shifted = np.concatenate([np.full(s, -np.inf), e[:-s]]) - r_per_sample * s
        e = np.maximum(e, shifted)
        s *= 2
    return e


def trailing_mean(a: np.ndarray, w: int) -> np.ndarray:
    cs = np.cumsum(np.concatenate([np.full(w, a[0]), a]))
    return (cs[w:] - cs[:-w]) / w


def limiter(x: np.ndarray, ceiling_db: float = CEILING_DB, lookahead_ms: float = LOOKAHEAD_MS,
            knee_db: float = 4.0, release_db_per_s: float = 40.0) -> tuple[np.ndarray, float]:
    L = int(SR * lookahead_ms / 1000)
    peak = np.max(np.abs(x), axis=0)
    lvl = 20 * np.log10(np.maximum(peak, 1e-9))
    lo, hi = ceiling_db - knee_db / 2, ceiling_db + knee_db / 2
    out_lvl = np.where(lvl < lo, lvl, np.where(lvl > hi, ceiling_db, lvl - (lvl - lo) ** 2 / (2 * knee_db)))
    att = np.maximum(lvl - out_lvl, 0.0)
    att = trailing_max(att, L + 1)
    att = release_envelope(att, release_db_per_s / SR)
    g = trailing_mean(10 ** (-att / 20.0), L + 1)
    gain = np.concatenate([g[L:], np.full(L, g[-1])])
    return x * gain, max(0.0, float(-20 * np.log10(max(gain.min(), 1e-9))))


def master(mix: np.ndarray, tmp_path: str, log) -> tuple[np.ndarray, dict]:
    passes = []
    mix = dc_block(mix)
    write_wav(tmp_path, mix)
    lufs0, tp0 = measure(tmp_path)
    gain_db = TARGET_LUFS - lufs0
    limited, work = limiter(mix * amp(gain_db))
    write_wav(tmp_path, limited)
    lufs1, tp1 = measure(tmp_path)
    passes.append({"pre_gain_lufs": lufs0, "pre_gain_true_peak": tp0, "gain_db": round(gain_db, 3), "limiter_work_db": round(work, 3), "lufs": lufs1, "true_peak_dbtp": tp1})
    log(f"  master: pre {lufs0:.2f} LUFS / {tp0:.2f} dBTP, gain {gain_db:+.2f} dB, limiter {work:.2f} dB -> {lufs1:.2f} LUFS / {tp1:.2f} dBTP")
    if abs(lufs1 - TARGET_LUFS) > 0.2:
        gain_db += TARGET_LUFS - lufs1
        limited, work = limiter(mix * amp(gain_db))
        write_wav(tmp_path, limited)
        lufs1, tp1 = measure(tmp_path)
        passes.append({"gain_db": round(gain_db, 3), "limiter_work_db": round(work, 3), "lufs": lufs1, "true_peak_dbtp": tp1})
        log(f"  master: corrected gain {gain_db:+.2f} dB, limiter {work:.2f} dB -> {lufs1:.2f} LUFS / {tp1:.2f} dBTP")
    return limited, {"final": passes[-1], "passes": passes}


# ------------------------------------------------------------------------------------------------- main
def render(story_path: str, out_path: str, report_path: str, quiet: bool = False) -> dict:
    log = (lambda *a: None) if quiet else print
    with open(story_path) as fh:
        story = json.load(fh)
    n = story["meta"]["frames"] * SPF
    log(f"soundtrack: {story['meta']['frames']} frames, {len(story['sound'])} cues")
    mix, info = build(story)
    assert mix.shape == (2, n)
    tail_db = db(np.max(np.abs(mix[:, -SPF * 12:])))
    head_db = db(np.max(np.abs(mix[:, : SPF * 12])))
    log(f"  mix peak {db(np.max(np.abs(mix))):.2f} dBFS; first 12 frames {head_db:.1f} dBFS, last 12 frames {tail_db:.1f} dBFS")
    tmp = os.path.join(os.path.dirname(os.path.abspath(out_path)), "_master_tmp.wav")
    final, mrep = master(mix, tmp, log)
    write_wav(out_path, final)
    os.remove(tmp)
    ok, misses = phone_check(out_path, story_path)
    peaks = event_peaks(final, story)
    lufs, tp = mrep["final"]["lufs"], mrep["final"]["true_peak_dbtp"]
    loop_db = db(np.max(np.abs(final[:, -SPF * 12:])))
    report = {"out": os.path.abspath(out_path), "frames": story["meta"]["frames"], "seconds": n / SR, "master": mrep,
              "phone_check": {"ok": ok, "misses": misses, "threshold_db": PHONE_THRESHOLD_DB},
              "loop_tail_peak_db": loop_db, "head_peak_db": head_db, "cues": peaks, **info}
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=1)
    log(f"  {os.path.relpath(out_path, HERE)}: {n / SR:.2f} s, {lufs:.2f} LUFS, {tp:.2f} dBTP; last 12 frames {loop_db:.1f} dBFS")
    log(f"  phone check: {'passed' if ok else 'FAILED'} ({len(misses)} cues below {PHONE_THRESHOLD_DB} dBFS after a 300 Hz high-pass)")
    for m in misses[:10]:
        log(f"    frame {m['frame']} {m['cue']} {m['id']}: {m['hp300_peak_db']} dBFS")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Three Doors: the score")
    ap.add_argument("--story", default=os.path.join(HERE, "build", "story.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "soundtrack.wav"))
    ap.add_argument("--report", default=os.path.join(HERE, "out", "soundtrack_report.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    rep = render(args.story, args.out, args.report, args.quiet)
    ok = rep["phone_check"]["ok"] and -16.0 <= rep["master"]["final"]["lufs"] <= -14.0 and rep["master"]["final"]["true_peak_dbtp"] <= -1.0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

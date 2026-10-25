#!/usr/bin/env python3
"""The Long Number: the soundtrack.

`python soundtrack.py` reads `build/story.json` (written by `story.py`, the only writer) and
writes `out/soundtrack.wav` (48 kHz, 24-bit, stereo, 36.80 s plus one video frame) and
`out/soundtrack_report.json` (per-event peak levels, the master's figures and the phone check,
so a reviewer can audit the score without listening). numpy only: the biquads, the envelopes
and the FFT convolution are written out below; loudness and true peak are measured by ffmpeg's
ebur128 filter. Nothing is imported from `doubling_film/` or `archive/`.

One idea (treatment §9). The ledger has a voice: one sustained tone whose partials are the
digits. Fundamental E2 (82.4 Hz); partial n at n x 82.4 Hz with amplitude 1 / n^0.8, a fixed
seeded detune of +-3 cents per partial and a shared +-4 cent vibrato at about 0.15 Hz. The tone
has as many partials as the newest row has digits: two at frame 0 (the deck's own two digits),
each new digit fading in over 0.25 s as its row stamps, sixty-eight for the whole deck; in the
beats it re-targets to the ratio wall's digit count (51, then 8), dims 3 dB under the ending and
returns to two partials over 0.6 s at 35.80 s. The level follows the count: -24 dBFS RMS with
two partials and -18 dBFS with sixty-eight, interpolated in the logarithm of the count between.
The percussion is only the picture's events: word stamps, the knock on the big stamps, card
flicks, row clicks, typing clicks, the pulse down the rule, the fold's ticks and the deck's two
dry ticks. A small synthesised room (12 % wet) sits on the percussion only; the tone is dry.

The loop. The tone is strictly periodic in 36.80 s: the vibrato completes 5.5 cycles in the loop
(0.1495 Hz, so its value is continuous at the join) and each partial's frequency is nudged by
less than 0.014 Hz so it completes an integer number of cycles in the period. Every envelope is
back at its frame-0 value by 36.40 s. The percussion is rendered 2.0 s past 36.80 s and the tail
is folded onto 0.00-2.00 s (wrap-around mix); nothing sounds after the deck's dry tick at
36.40 s, so the join is seamless and the knock at frame 1 is heard once per play. The file is
one frame longer than the film (the first 800 samples repeated) so a mux with -shortest never
drops a video frame.

Cues derived here rather than read from story.json (story.py is not modified):
  - the lead's stamps x 4, x 3, x 2 (from the `lead` events; at x 1 the row's click and the
    deck's dry tick alone, as §9 says);
  - the fold's 23 ticks 35 ms apart from the fold's first frame, each 0.5 dB quieter (§9),
    instead of the picture's group-start frames (0.02 s apart) listed as `fold_tick` cues;
  - the payoff's pulse down the rule sounds when the tone returns (13.20 s), not inside the
    silence (§9: from the landing to 13.20 s only the room's decay);
  - the 3 dB ending dim is released over the final re-target (35.80-36.40 s) so the tone is at
    its frame-0 level by the loop point;
  - the run's click rule (-1 dB per row to a floor of -30 dBFS) is void as written because the
    click already sits at -30 dBFS; it is implemented literally and the clicks stay at -30 dBFS.

Levels quoted in dBFS are peak levels of the element before the master gain.

Public functions for checks.py:
  measure(path) -> (integrated_lufs, true_peak_dbtp)
  phone_check(wav_path, story_json_path) -> (ok, misses)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 48000
FPS = 60
SPF = SR // FPS                 # 800 samples per frame

# The tone
E2_HZ = 82.4
AMP_EXPONENT = 0.8
DETUNE_CENTS = 3.0
VIBRATO_CENTS = 4.0
VIBRATO_HZ_NOMINAL = 0.15
TONE_RMS_DB_AT_2 = -24.0
TONE_RMS_DB_AT_68 = -18.0
TONE_SEED = 824
PARTIAL_CAP_OPTION = 40         # the §12 fallback: audible partials capped, top partials rise instead
CAP_BOOST_DB = 6.0              # how much the top quarter rises when the count passes the cap
MUD_MARGIN_DB = 6.0             # the run click must sit this far above the tone's band RMS

# The percussion (peak dBFS before the master gain)
LEVELS = {
    "stamp": -26.0, "knock": -22.0, "knock_big": -16.0,
    "flick": -18.0, "flick_long": -24.0,
    "click": -30.0, "click_floor": -30.0, "click_muted_db": -6.0,
    "typing": -32.0, "pulse": -24.0, "tick": -26.0, "lead_stamp": -26.0,
    "fold_tick_step_db": -0.5,
}
FLICK_PAN = 0.3
ROOM_WET = 0.12
ROOM_SEED = 1392
TAIL_S = 2.0

# The master
TARGET_LUFS = -15.0
FALLBACK_LUFS = -16.0
CEILING_DB = -1.5
LOOKAHEAD_MS = 5.0
LIMITER_WORK_MAX_DB = 3.0
TONE_RAISE_DB = 3.0

# The phone check
PHONE_HP_HZ = 300.0
PHONE_THRESHOLD_DB = -30.0
PHONE_WINDOW_MS = 50.0


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-12))


def amp(d: float) -> float:
    return 10.0 ** (d / 20.0)


# ------------------------------------------------------------------------------------------------
# Filters (RBJ biquads written out; applied as FFT convolutions with their impulse responses)
# ------------------------------------------------------------------------------------------------

def biquad_coeffs(kind: str, f0: float, q: float, sr: int = SR):
    w = 2.0 * math.pi * f0 / sr
    s, c = math.sin(w), math.cos(w)
    alpha = s / (2.0 * q)
    if kind == "bandpass":          # constant 0 dB peak gain
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
    """Direct form II transposed, sample by sample (short signals and impulse responses)."""
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


def circular_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolution modulo len(x): a filter applied to one period of a periodic signal."""
    n = len(x)
    hp = np.zeros(n)
    hp[: len(h)] = h[:n]
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(hp), n)


def biquad(x: np.ndarray, kind: str, f0: float, q: float, keep_tail: bool = True) -> np.ndarray:
    y = fft_convolve(x, biquad_ir(kind, f0, q))
    return y if keep_tail else y[: len(x)]


def sweep_bandpass(x: np.ndarray, f_start: float, f_end: float, q: float) -> np.ndarray:
    """A band-pass whose centre glides exponentially from f_start to f_end over the signal."""
    n = len(x)
    y = np.empty(n)
    z1 = z2 = 0.0
    ratio = f_end / f_start
    for i in range(n):
        (b0, b1, b2), (_, a1, a2) = biquad_coeffs("bandpass", f_start * ratio ** (i / (n - 1)), q)
        xi = float(x[i])
        yi = b0 * xi + z1
        z1 = b1 * xi - a1 * yi + z2
        z2 = b2 * xi - a2 * yi
        y[i] = yi
    return y


# ------------------------------------------------------------------------------------------------
# Percussion elements (each returned at an exact peak level)
# ------------------------------------------------------------------------------------------------

def env_exp(n: int, tau_s: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (tau_s * SR))


def fade_tail(sig: np.ndarray, ms: float = 1.0) -> np.ndarray:
    k = max(1, min(len(sig), int(SR * ms / 1000)))
    sig[-k:] *= np.linspace(1.0, 0.0, k)
    return sig


def normalise_peak(sig: np.ndarray, level_db: float) -> np.ndarray:
    p = float(np.max(np.abs(sig)))
    return sig * (amp(level_db) / p) if p > 0 else sig


def stamp_noise(rng, level_db: float) -> np.ndarray:
    """A word stamp: 25 ms of white noise through a 2-pole band-pass (2.6 kHz, Q 4), tau 8 ms."""
    n = int(0.025 * SR)
    x = fade_tail(rng.standard_normal(n) * env_exp(n, 0.008))
    return normalise_peak(biquad(x, "bandpass", 2600.0, 4.0), level_db)


def knock(level_db: float) -> np.ndarray:
    """The knock: a 200 Hz damped sine (tau 50 ms) over a 60 Hz one (tau 80 ms)."""
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.05) + 0.8 * np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.08)
    return normalise_peak(fade_tail(x, 5.0), level_db)


def flick(rng, level_db: float, dur_s: float = 0.040, tau_s: float = 0.012, thump_s: float = 0.025) -> np.ndarray:
    """A card off the deck: noise band-passed 1.8-5 kHz with a tau 12 ms decay plus a 150 Hz thump."""
    n = int(dur_s * SR)
    x = fade_tail(rng.standard_normal(n) * env_exp(n, tau_s))
    y = biquad(biquad(x, "highpass", 1800.0, math.sqrt(0.5)), "lowpass", 5000.0, math.sqrt(0.5))
    y /= np.max(np.abs(y))
    m = int(thump_s * SR)
    t = np.arange(m) / SR
    thump = np.sin(2 * np.pi * 150 * t) * np.sin(np.pi * t / thump_s)
    out = np.zeros(max(len(y), m))
    out[: len(y)] += y
    out[:m] += 0.6 * thump
    return normalise_peak(out, level_db)


def click(rng, level_db: float, f0: float, dur_s: float, tau_s: float, q: float) -> np.ndarray:
    """A pitched click: a short noise burst through a narrow band-pass."""
    n = max(8, int(dur_s * SR))
    x = fade_tail(rng.standard_normal(n) * env_exp(n, tau_s), 0.5)
    return normalise_peak(biquad(x, "bandpass", f0, q), level_db)


def row_click(rng, level_db):
    return click(rng, level_db, 3400.0, 0.006, 0.0015, 8.0)


def typing_click(rng, level_db):
    return click(rng, level_db, 3200.0, 0.004, 0.001, 8.0)


def dry_tick(rng, level_db):
    return click(rng, level_db, 2600.0, 0.008, 0.002, 6.0)


def pulse(rng, level_db: float, dur_s: float = 0.3) -> np.ndarray:
    """The pulse down the rule: band-passed noise sweeping 600 -> 2400 Hz over 0.3 s."""
    n = int(dur_s * SR)
    env = np.minimum(1.0, np.arange(n) / (0.005 * SR)) * env_exp(n, 0.10)
    x = fade_tail(rng.standard_normal(n) * env, 5.0)
    return normalise_peak(sweep_bandpass(x, 600.0, 2400.0, 3.0), level_db)


def place(bus: np.ndarray, start: int, sig: np.ndarray, gain: float = 1.0, pan: float = 0.0) -> None:
    """Add `sig` to the stereo bus at sample `start`; the louder channel carries the element's level."""
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


# ------------------------------------------------------------------------------------------------
# The percussion bus from the cues
# ------------------------------------------------------------------------------------------------

def build_percussion(story: dict, n_render: int):
    cues = story["sound"]["cues"]
    events = story["events"]
    room_bus = np.zeros((2, n_render))      # through the room
    dry_bus = np.zeros((2, n_render))       # the deck's two dry ticks
    placed: list[dict] = []
    derived: list[str] = []
    passes = {e["ref"]: e for e in events if e["kind"] == "pass"}
    returns = [e["frame"] for e in events if e["kind"] == "tone" and e["action"] == "return"]

    def record(frame, t_s, cue, level_db, gain_db=0.0, pan=0.0, room=True, **extra):
        placed.append({"frame": frame, "t": round(t_s, 5), "cue": cue, "level_db": round(level_db + gain_db, 2),
                       "pan": pan, "room": room, **extra})

    for i, cue in enumerate(cues):
        rng = np.random.default_rng([52, i])
        kind = cue["cue"]
        f = cue["frame"]
        start = f * SPF
        pan, gain_db, bus = 0.0, 0.0, room_bus
        extra = {k: cue[k] for k in ("id", "k") if k in cue}
        if kind == "stamp":
            level = LEVELS["stamp"]
            sig = stamp_noise(rng, level)
        elif kind == "knock":
            level = LEVELS["knock_big"] if cue["level"] == "big" else LEVELS["knock"]
            sig = knock(level)
            extra["level"] = cue["level"]
        elif kind == "typing":
            level = LEVELS["typing"]
            sig = typing_click(rng, level)
        elif kind == "flick":
            level = LEVELS["flick"]
            sig = flick(rng, level)
            pan = cue.get("pan", FLICK_PAN)
            if cue["k"] >= 12:                               # the run: gain 1 / (1 + rate / 8)
                gain_db = db(1.0 / (1.0 + cue["rate_per_s"] / 8.0))
            extra["rate_per_s"] = cue["rate_per_s"]
        elif kind == "flick_long":
            level = LEVELS["flick_long"]
            sig = flick(rng, level, dur_s=cue["frames"] / FPS, tau_s=0.12)
            pan = FLICK_PAN
        elif kind == "click":
            level = LEVELS["click"]
            ri = cue.get("run_index")
            if ri is not None:
                level = max(LEVELS["click"] - 1.0 * ri, LEVELS["click_floor"])
            if cue.get("muted"):
                gain_db = LEVELS["click_muted_db"]
                extra["muted"] = True
            sig = row_click(rng, level)
        elif kind == "pulse":
            level = LEVELS["pulse"]
            sig = pulse(rng, level)
            p = passes.get(cue["ref"], {})
            if "silence_until" in p:
                ret = [r for r in returns if r >= p["silence_until"]]
                moved = ret[0] if ret else p["silence_until"]
                derived.append(f"pulse for the {cue['ref']} pass moved from frame {f} to frame {moved} (the tone's return)")
                f, start = moved, moved * SPF
            extra["ref"] = cue["ref"]
        elif kind == "silence":
            placed.append({"frame": f, "t": round(f / FPS, 5), "cue": "silence", "until": cue["until"],
                           "level_db": None, "room": False})
            continue
        elif kind == "fold_tick":
            continue                                          # derived from the fold event below
        elif kind in ("deck_out", "deck_tick"):
            level = LEVELS["tick"]
            sig = dry_tick(rng, level)
            bus = dry_bus
        else:
            raise ValueError(f"unknown cue {kind}")
        place(bus, start, sig, amp(gain_db), pan)
        record(f, start / SR, kind, level, gain_db, pan, bus is room_bus, **extra)

    # Derived: the lead's stamps (x 4, x 3, x 2; nothing extra at x 1) and the fold's ticks.
    for j, e in enumerate(events):
        if e["kind"] == "lead" and e["piece"] < 3:
            rng = np.random.default_rng([53, j])
            sig = stamp_noise(rng, LEVELS["lead_stamp"])
            place(room_bus, e["frame"] * SPF, sig)
            record(e["frame"], e["frame"] / FPS, "lead_stamp", LEVELS["lead_stamp"], id="lead", text=e["text"])
        elif e["kind"] == "lead":
            derived.append(f"lead piece '{e['text']}' at frame {e['frame']}: no stamp (the row's click and the dry tick alone)")
        elif e["kind"] == "fold":
            t0 = e["frame"] / FPS
            for g in range(e["groups"]):
                rng = np.random.default_rng([54, g])
                level = LEVELS["typing"] + LEVELS["fold_tick_step_db"] * g
                t = t0 + 0.035 * g
                place(room_bus, int(round(t * SR)), typing_click(rng, level))
                record(e["frame"], t, "fold_tick", level, group=g)
            derived.append(f"fold ticks: {e['groups']} ticks 35 ms apart from frame {e['frame']} "
                           f"({t0:.3f} s to {t0 + 0.035 * (e['groups'] - 1):.3f} s), each 0.5 dB quieter")
    derived.append("lead stamps x 4, x 3, x 2 derived from the lead events at -26 dBFS")
    placed.sort(key=lambda p: p["t"])
    return room_bus, dry_bus, placed, derived


def room_ir(seed: int = ROOM_SEED) -> np.ndarray:
    """0.5 s of white noise with tau 0.12 s, low-passed at 4 kHz, decorrelated stereo, unit energy."""
    n = int(0.5 * SR)
    rng = np.random.default_rng(seed)
    out = np.zeros((2, n))
    for c in range(2):
        x = rng.standard_normal(n) * env_exp(n, 0.12)
        x = fade_tail(x, 10.0)
        y = biquad(x, "lowpass", 4000.0, math.sqrt(0.5), keep_tail=False)
        out[c] = y / math.sqrt(float(np.sum(y * y)))
    return out


# ------------------------------------------------------------------------------------------------
# The tone
# ------------------------------------------------------------------------------------------------

def smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def swell_shape(n: int) -> np.ndarray:
    """+1 over n control points: rises over the first 15 %, falls over the last 40 %."""
    u = np.arange(n) / max(n - 1, 1)
    return smoothstep(u / 0.15) * (1.0 - smoothstep((u - 0.6) / 0.4))


def steady_mean_square(n_count: np.ndarray, amps: np.ndarray, cap: int | None) -> np.ndarray:
    """Mean square of the fully faded tone as a function of the (fractional) partial count."""
    n_max = len(amps)
    grid = np.linspace(0.0, n_max, n_max * 20 + 1)
    idx = np.arange(1, n_max + 1)[:, None]
    w = np.clip(grid[None, :] - (idx - 1), 0.0, 1.0)                     # partial n present when count >= n
    a = amps[:, None] * np.ones_like(w)
    if cap is not None:
        w = np.where(idx > cap, 0.0, w)
        top = (idx > cap - cap // 4) & (idx <= cap)
        boost = amp(CAP_BOOST_DB) ** np.clip((grid[None, :] - cap) / (n_max - cap), 0.0, 1.0)
        a = np.where(top, a * boost, a)
    ms = np.sum((a * w) ** 2 / 2.0, axis=0)
    return np.interp(n_count, grid, ms)


def build_tone(story: dict, n_period: int, cap: int | None = None) -> tuple[np.ndarray, dict]:
    rows = story["numbers"]["rows"]
    n_max = max(len(r["digits"]) for r in rows)                          # 68
    partial_events = sorted(story["sound"]["partials"], key=lambda e: e["frame"])
    cr = 1000                                                            # control rate, Hz
    period_s = n_period / SR
    n_ctrl = int(math.ceil(period_s * cr)) + 1
    t_ctrl = np.arange(n_ctrl) / cr

    env = np.zeros((n_max, n_ctrl))
    env[:2, :] = 1.0                                                     # frame 0: the deck's two digits
    gain_db = np.zeros((n_max, n_ctrl))                                  # per-partial swell / dull
    dim_db = np.zeros(n_ctrl)
    notes = []

    def ctrl_index(t_s):
        return int(round(t_s * cr))

    for ev in partial_events:
        t0 = ev["frame"] / FPS
        if ev.get("cut"):
            target_count, ramp = 0, ev.get("fade_ms", 10) / 1000.0
        elif "partials" in ev:
            target_count, ramp = ev["partials"], ev.get("ramp_frames", 0) / FPS
        else:
            target_count = None
        if target_count is not None:
            i0, i1 = ctrl_index(t0), max(ctrl_index(t0), ctrl_index(t0 + ramp))
            for n in range(n_max):
                target = 1.0 if n < target_count else 0.0
                v0 = env[n, i0]
                if i1 > i0:
                    env[n, i0:i1] = np.linspace(v0, target, i1 - i0, endpoint=False)
                env[n, i1:] = target
            if ev.get("dull"):                                           # row 10: the new partial at -9 dB
                new_from = target_count - ev.get("new", 1)
                cut_after = [e["frame"] for e in partial_events if e.get("cut") and e["frame"] > ev["frame"]]
                i_end = ctrl_index(cut_after[0] / FPS) if cut_after else n_ctrl
                for n in range(new_from, target_count):
                    gain_db[n, i0:i_end] += -9.0
                notes.append(f"partial {target_count} enters at -9 dB from frame {ev['frame']} "
                             f"until frame {cut_after[0] if cut_after else 'end'}")
        if "swell" in ev:
            sw = ev["swell"]
            i0 = ctrl_index(t0)
            i1 = min(n_ctrl, ctrl_index(t0 + sw["frames"] / FPS))
            gain_db[sw["partial"] - 1, i0:i1] += sw["db"] * swell_shape(i1 - i0)
        if "dim_db" in ev:
            i0, i1 = ctrl_index(t0), ctrl_index(ev["end"] / FPS)
            dim_db[i0:i1] = np.linspace(0.0, -ev["dim_db"], i1 - i0, endpoint=False)
            # release over the re-target that follows (derived; the loop needs the frame-0 level)
            nxt = [e for e in partial_events if e["frame"] >= ev["end"] and "partials" in e]
            rel_end = ctrl_index((nxt[0]["frame"] + nxt[0].get("ramp_frames", 0)) / FPS) if nxt else n_ctrl
            dim_db[i1:rel_end] = np.linspace(-ev["dim_db"], 0.0, rel_end - i1, endpoint=False)
            dim_db[rel_end:] = 0.0
            notes.append(f"ending dim -{ev['dim_db']} dB over frames {ev['frame']}-{ev['end']}, "
                         f"released by frame {int(round(rel_end / cr * FPS))}")

    if cap is not None:
        env[cap:, :] = 0.0

    amps = np.arange(1, n_max + 1) ** -AMP_EXPONENT
    n_eff = env.sum(axis=0)
    n_c = np.clip(n_eff, 2.0, None)
    target_db = TONE_RMS_DB_AT_2 + (TONE_RMS_DB_AT_68 - TONE_RMS_DB_AT_2) * np.log(n_c / 2.0) / np.log(68.0 / 2.0)
    ms = steady_mean_square(n_c, amps, cap)
    master_gain = 10 ** (target_db / 20.0) / np.sqrt(ms)
    cap_boost = np.zeros(n_ctrl)
    if cap is not None:
        cap_boost = CAP_BOOST_DB * np.clip((n_eff - cap) / (n_max - cap), 0.0, 1.0)

    # Oscillators: the shared vibrato, then each partial with a seeded detune and random phase.
    rng = np.random.default_rng(TONE_SEED)
    detune = rng.uniform(-DETUNE_CENTS, DETUNE_CENTS, n_max)
    phase0 = rng.uniform(0.0, 2 * np.pi, n_max)
    vib_cycles = round(VIBRATO_HZ_NOMINAL * period_s * 2) / 2.0          # 5.5: continuous at the join
    vib_hz = vib_cycles / period_s
    t = np.arange(n_period) / SR
    v = 2.0 ** (VIBRATO_CENTS * np.sin(2 * np.pi * vib_hz * t) / 1200.0)
    cum_v = (np.cumsum(v) - v) / SR                                      # integral of v up to each sample
    int_v = float(np.sum(v)) / SR                                        # over the whole period
    tone = np.zeros(n_period)
    nudges = []
    phase_err = 0.0
    for n in range(n_max):
        e_n = env[n]
        if not np.any(e_n > 0):
            continue
        g_db = gain_db[n]
        if cap is not None and (cap - cap // 4) < n + 1 <= cap:
            g_db = g_db + cap_boost
        f_n = (n + 1) * E2_HZ * 2.0 ** (detune[n] / 1200.0)
        cycles = f_n * int_v
        delta = (round(cycles) - cycles) / period_s                      # Hz, so the period holds an integer count
        nudges.append(delta)
        phase = 2 * np.pi * (f_n * cum_v + delta * t) + phase0[n]
        total = 2 * np.pi * (f_n * int_v + delta * period_s)
        phase_err = max(phase_err, abs(total / (2 * np.pi) - round(total / (2 * np.pi))))
        a_n = amps[n] * np.interp(t, t_ctrl, e_n * 10 ** (g_db / 20.0))
        tone += a_n * np.sin(phase)
    tone *= np.interp(t, t_ctrl, master_gain * 10 ** (dim_db / 20.0))

    info = {
        "partial_cap": cap,
        "max_partials": n_max,
        "vibrato_hz": round(vib_hz, 6),
        "vibrato_cycles_per_loop": vib_cycles,
        "detune_cents": [round(float(d), 3) for d in detune],
        "max_frequency_nudge_hz": round(max(abs(x) for x in nudges), 6),
        "loop_phase_error_cycles": phase_err,
        "notes": notes,
        "level_law": f"{TONE_RMS_DB_AT_2} dBFS RMS at 2 partials, {TONE_RMS_DB_AT_68} dBFS at 68, log-interpolated",
    }
    return tone, info


# ------------------------------------------------------------------------------------------------
# WAV in and out (24-bit, standard library)
# ------------------------------------------------------------------------------------------------

def write_wav(path: str, data: np.ndarray, sr: int = SR) -> None:
    """data: (channels, samples) floats in [-1, 1]; written as 24-bit little-endian PCM."""
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
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
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


# ------------------------------------------------------------------------------------------------
# Measurement (ffmpeg ebur128) and the phone check
# ------------------------------------------------------------------------------------------------

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
    """For every sound cue: the mix's peak (full band and high-passed at 300 Hz) within +-window of its frame."""
    mono = stereo.mean(axis=0)
    hp = _phone_filter(mono)
    half = int(SR * window_ms / 1000)
    out = []
    for cue in story["sound"]["cues"]:
        if cue["cue"] == "silence":
            continue
        c = cue["frame"] * SPF
        lo, hi = max(0, c - half), min(len(mono), c + half + 1)
        out.append({"frame": cue["frame"], "t": round(cue["frame"] / FPS, 4), "cue": cue["cue"],
                    "id": cue.get("id", cue.get("k")),
                    "peak_db": round(db(np.max(np.abs(mono[lo:hi]))), 2),
                    "hp300_peak_db": round(db(np.max(np.abs(hp[lo:hi]))), 2)})
    return out


def phone_check(wav_path: str, story_json_path: str, threshold_db: float = PHONE_THRESHOLD_DB):
    """High-pass the mix at 300 Hz; every sound cue must peak above -30 dBFS within 50 ms of its frame."""
    with open(story_json_path) as fh:
        story = json.load(fh)
    stereo, sr = read_wav(wav_path)
    if sr != SR:
        raise ValueError(f"{wav_path} is {sr} Hz, expected {SR}")
    peaks = event_peaks(stereo, story)
    misses = [p for p in peaks if not (p["hp300_peak_db"] > threshold_db)]
    return len(misses) == 0, misses


# ------------------------------------------------------------------------------------------------
# The master: DC block, gain to target, soft-knee look-ahead limiter
# ------------------------------------------------------------------------------------------------

def dc_block(stereo: np.ndarray) -> np.ndarray:
    """A 2-pole high-pass at 10 Hz, applied circularly so the period stays periodic."""
    h = biquad_ir("highpass", 10.0, math.sqrt(0.5))
    return np.stack([circular_convolve(ch, h) for ch in stereo])


def trailing_max(a: np.ndarray, w: int) -> np.ndarray:
    m = a.copy()
    c = 1
    while c < w:
        s = min(c, w - c)
        m = np.maximum(m, np.concatenate([np.zeros(s), m[:-s]]))
        c += s
    return m


def release_envelope(att: np.ndarray, r_per_sample: float) -> np.ndarray:
    """e[n] = max over d >= 0 of (att[n - d] - r d): a linear-in-dB release after every peak."""
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
    """Stereo-linked soft-knee look-ahead peak limiter. Returns (output, maximum gain reduction in dB)."""
    L = int(SR * lookahead_ms / 1000)
    peak = np.max(np.abs(x), axis=0)
    lvl = 20 * np.log10(np.maximum(peak, 1e-9))
    lo, hi = ceiling_db - knee_db / 2, ceiling_db + knee_db / 2
    out_lvl = np.where(lvl < lo, lvl, np.where(lvl > hi, ceiling_db, lvl - (lvl - lo) ** 2 / (2 * knee_db)))
    att = np.maximum(lvl - out_lvl, 0.0)
    att = trailing_max(att, L + 1)
    att = release_envelope(att, release_db_per_s / SR)
    g = trailing_mean(10 ** (-att / 20.0), L + 1)
    gain = np.concatenate([g[L:], np.full(L, g[-1])])              # look-ahead alignment, no delay
    return x * gain, max(0.0, float(-20 * np.log10(max(gain.min(), 1e-9))))


def master(tone: np.ndarray, perc: np.ndarray, tmp_path: str, log) -> tuple[np.ndarray, dict]:
    """measure -> gain -> limit -> re-measure; the tone rises 3 dB while the limiter works > 3 dB."""
    tone_boost_db = 0.0
    passes = []
    best = None
    for target in (TARGET_LUFS, FALLBACK_LUFS):
        for attempt in range(3):
            mix = dc_block(perc + tone[None, :] * amp(tone_boost_db))
            write_wav(tmp_path, mix)
            lufs0, tp0 = measure(tmp_path)
            gain_db = target - lufs0
            limited, work = limiter(mix * amp(gain_db))
            write_wav(tmp_path, limited)
            lufs1, tp1 = measure(tmp_path)
            if abs(lufs1 - target) > 0.2:                            # the limiter took loudness: correct once
                gain_db += target - lufs1
                limited, work = limiter(mix * amp(gain_db))
                write_wav(tmp_path, limited)
                lufs1, tp1 = measure(tmp_path)
            rec = {"target_lufs": target, "attempt": attempt, "tone_boost_db": tone_boost_db,
                   "pre_gain_lufs": lufs0, "pre_gain_true_peak": tp0, "gain_db": round(gain_db, 3),
                   "limiter_work_db": round(work, 3), "lufs": lufs1, "true_peak_dbtp": tp1}
            passes.append(rec)
            log(f"  master: target {target} LUFS, tone +{tone_boost_db:.0f} dB: pre {lufs0:.2f} LUFS / {tp0:.2f} dBTP, "
                f"gain {gain_db:+.2f} dB, limiter {work:.2f} dB -> {lufs1:.2f} LUFS / {tp1:.2f} dBTP")
            best = (limited, rec)
            if work <= LIMITER_WORK_MAX_DB:
                return limited, {"final": rec, "passes": passes}
            tone_boost_db += TONE_RAISE_DB
    return best[0], {"final": best[1], "passes": passes, "warning": "limiter work above 3 dB on every pass"}


# ------------------------------------------------------------------------------------------------
# Listening by measurement: does the wedge turn to mud?
# ------------------------------------------------------------------------------------------------

def band_rms_db(x: np.ndarray, f_lo: float, f_hi: float) -> float:
    spec = np.fft.rfft(x * np.hanning(len(x)))
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    sel = (freqs >= f_lo) & (freqs < f_hi)
    total = np.sum(np.abs(spec) ** 2)
    rms_all = math.sqrt(np.mean((x * np.hanning(len(x))) ** 2)) * math.sqrt(8 / 3)
    frac = float(np.sum(np.abs(spec[sel]) ** 2) / total) if total > 0 else 0.0
    return db(rms_all * math.sqrt(frac))


def spectral_flatness(x: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Geometric / arithmetic mean of the power in third-octave bands between f_lo and f_hi."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    edges = [f_lo]
    while edges[-1] * 2 ** (1 / 3) <= f_hi:
        edges.append(edges[-1] * 2 ** (1 / 3))
    powers = np.array([np.sum(spec[(freqs >= a) & (freqs < b)]) for a, b in zip(edges, edges[1:])])
    powers = powers[powers > 0]
    return float(np.exp(np.mean(np.log(powers))) / np.mean(powers))


def mud_metrics(tone: np.ndarray, story: dict) -> dict:
    """Measured on the wedge hold (the tone at its fullest) and on the run."""
    fold = next(e for e in story["events"] if e["kind"] == "fold")
    whole = next(e for e in story["events"] if e["kind"] == "stamp" and e["id"] == "whole_deck_lead")
    a, b = whole["frame"] * SPF + SPF * 6, fold["frame"] * SPF
    seg = tone[a:b]
    run_rows = [e for e in story["events"] if e["kind"] == "row" and 24 <= e["k"] <= 48]
    r0, r1 = run_rows[0]["frame"] * SPF, run_rows[-1]["frame"] * SPF
    run_seg = tone[r0:r1]
    click_band = band_rms_db(run_seg, 2500.0, 4500.0)
    rms = db(math.sqrt(np.mean(seg ** 2)))
    return {
        "wedge_rms_db": round(rms, 2),
        "wedge_crest_db": round(db(np.max(np.abs(seg))) - rms, 2),
        "wedge_flatness_300_6000": round(spectral_flatness(seg, 300.0, 6000.0), 4),
        "wedge_band_rms_2500_4500_db": round(band_rms_db(seg, 2500.0, 4500.0), 2),
        "run_band_rms_2500_4500_db": round(click_band, 2),
        "run_click_margin_db": round(LEVELS["click"] - click_band, 2),
        "wedge_frames": [a // SPF, b // SPF],
        "run_frames": [r0 // SPF, r1 // SPF],
    }


def choose_cap(story: dict, n_period: int, mode: str, log):
    """mode 'auto': cap only if it buys the run click >= 3 dB of margin and the margin is short of 6 dB."""
    if mode in ("0", "none", "off"):
        tone, info = build_tone(story, n_period, None)
        info["mud"] = mud_metrics(tone, story)
        info["cap_decision"] = "no cap (forced)"
        return tone, info
    if mode != "auto":
        tone, info = build_tone(story, n_period, int(mode))
        info["mud"] = mud_metrics(tone, story)
        info["cap_decision"] = f"cap {mode} (forced)"
        return tone, info
    tone_a, info_a = build_tone(story, n_period, None)
    mud_a = mud_metrics(tone_a, story)
    tone_b, info_b = build_tone(story, n_period, PARTIAL_CAP_OPTION)
    mud_b = mud_metrics(tone_b, story)
    log(f"  tone uncapped: wedge RMS {mud_a['wedge_rms_db']} dBFS, crest {mud_a['wedge_crest_db']} dB, "
        f"flatness {mud_a['wedge_flatness_300_6000']}, run click margin {mud_a['run_click_margin_db']} dB")
    log(f"  tone capped {PARTIAL_CAP_OPTION}: wedge RMS {mud_b['wedge_rms_db']} dBFS, crest {mud_b['wedge_crest_db']} dB, "
        f"flatness {mud_b['wedge_flatness_300_6000']}, run click margin {mud_b['run_click_margin_db']} dB")
    short = mud_a["run_click_margin_db"] < MUD_MARGIN_DB
    gain = mud_b["run_click_margin_db"] - mud_a["run_click_margin_db"]
    use_cap = short and gain >= 3.0
    chosen, info, mud = (tone_b, info_b, mud_b) if use_cap else (tone_a, info_a, mud_a)
    info["mud"] = mud
    info["mud_alternative"] = mud_a if use_cap else mud_b
    info["cap_decision"] = (f"cap {PARTIAL_CAP_OPTION}: the run click margin was {mud_a['run_click_margin_db']} dB "
                            f"uncapped and the cap adds {gain:.1f} dB" if use_cap else
                            f"no cap: the cap would change the run click margin by {gain:+.1f} dB "
                            f"(uncapped {mud_a['run_click_margin_db']} dB), so 68 partials stand")
    return chosen, info


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

def render(story_path: str, out_path: str, report_path: str, cap_mode: str = "auto", quiet: bool = False) -> dict:
    t_start = time.time()

    def log(msg):
        if not quiet:
            print(msg, flush=True)

    with open(story_path) as fh:
        story = json.load(fh)
    meta = story["meta"]
    assert meta["fps"] == FPS, meta["fps"]
    n_frames = meta["frames"]                                            # 2208
    n_period = n_frames * SPF                                            # 36.80 s: the loop
    n_render = n_period + int(TAIL_S * SR)
    log(f"story: {n_frames} frames, loop {n_period / SR:.2f} s, {len(story['sound']['cues'])} cues")

    log("tone")
    tone, tone_info = choose_cap(story, n_period, cap_mode, log)
    log(f"  {tone_info['cap_decision']}")
    first = tone[: SPF * 24]
    last = tone[-SPF * 24:]
    tone_info["rms_first_0_4s_db"] = round(db(math.sqrt(np.mean(first ** 2))), 2)
    tone_info["rms_last_0_4s_db"] = round(db(math.sqrt(np.mean(last ** 2))), 2)
    tone_info["join_step"] = round(db(abs(tone[-1] - tone[0])), 1)
    log(f"  RMS first 0.4 s {tone_info['rms_first_0_4s_db']} dBFS, last 0.4 s {tone_info['rms_last_0_4s_db']} dBFS, "
        f"wedge {tone_info['mud']['wedge_rms_db']} dBFS; loop phase error {tone_info['loop_phase_error_cycles']:.2e} cycles")

    log("percussion")
    room_bus, dry_bus, placed, derived = build_percussion(story, n_render)
    ir = room_ir()
    wet = np.stack([fft_convolve(room_bus[c], ir[c]) for c in range(2)])
    perc_full = np.zeros((2, wet.shape[1]))
    perc_full[:, :n_render] += room_bus + dry_bus
    perc_full += ROOM_WET * wet
    tail = perc_full[:, n_period:]
    perc = perc_full[:, :n_period].copy()
    perc[:, : tail.shape[1]] += tail                                     # the wrap-around mix
    tail_peak = db(np.max(np.abs(tail)))
    tail_peak = tail_peak if tail_peak > -200 else float("-inf")           # numerical residue only
    log(f"  {len(placed)} elements placed; tail past 36.80 s peaks at {tail_peak:.1f} dBFS (folded onto 0-2 s)"
        if tail_peak > -200 else f"  {len(placed)} elements placed; nothing sounds past 36.80 s (the folded tail is silent)")

    log("master")
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(out_path)), "_measure_tmp.wav")
    final, master_info = master(tone, perc, tmp_path, log)
    out = np.concatenate([final, final[:, :SPF]], axis=1)                # one extra frame
    write_wav(out_path, out)
    lufs, tp = measure(out_path)
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    log(f"  {out_path}: {out.shape[1]} samples = {out.shape[1] / SR:.6f} s, {lufs:.2f} LUFS, {tp:.2f} dBTP")

    log("phone check")
    ok, misses = phone_check(out_path, story_path)
    stereo, _ = read_wav(out_path)
    peaks = event_peaks(stereo, story)
    quietest = sorted(peaks, key=lambda p: p["hp300_peak_db"])[:12]
    log(f"  {'passed' if ok else 'FAILED'}: {len(peaks)} cues, {len(misses)} below {PHONE_THRESHOLD_DB} dBFS after a 300 Hz high-pass")
    for m in misses:
        log(f"    miss: frame {m['frame']} {m['cue']} {m['id']}: {m['hp300_peak_db']} dBFS")
    log("  quietest cues (high-passed peak):")
    for q in quietest[:6]:
        log(f"    frame {q['frame']:5d} {q['cue']:10s} {str(q['id']):22s} {q['hp300_peak_db']:7.2f} dBFS (full band {q['peak_db']:7.2f})")

    run_time = time.time() - t_start
    report = {
        "file": os.path.abspath(out_path),
        "sample_rate": SR, "bits": 24, "channels": 2,
        "samples": int(out.shape[1]), "duration_s": out.shape[1] / SR,
        "loop_s": n_period / SR, "extra_frames": 1,
        "lufs": lufs, "true_peak_dbtp": tp,
        "master": master_info,
        "tone": tone_info,
        "room": {"ir_s": 0.5, "tau_s": 0.12, "lowpass_hz": 4000, "wet": ROOM_WET, "stereo": "decorrelated"},
        "levels_dbfs_peak_pre_master": LEVELS,
        "tail_folded": {"from_s": n_period / SR, "length_s": tail.shape[1] / SR, "peak_db": tail_peak},
        "phone_check": {"ok": ok, "threshold_db": PHONE_THRESHOLD_DB, "highpass_hz": PHONE_HP_HZ,
                        "window_ms": PHONE_WINDOW_MS, "cues_checked": len(peaks), "misses": misses,
                        "quietest": quietest},
        "elements": placed,
        "cue_peaks": peaks,
        "derived_cues": derived,
        "run_time_s": round(run_time, 2),
    }
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=1)
    log(f"wrote {report_path}; {run_time:.1f} s")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="The Long Number: render the soundtrack from build/story.json")
    ap.add_argument("--story", default=os.path.join(HERE, "build", "story.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "soundtrack.wav"))
    ap.add_argument("--report", default=os.path.join(HERE, "out", "soundtrack_report.json"))
    ap.add_argument("--cap", default="auto", help="partial cap: auto (default), 0 for none, or a number such as 40")
    ap.add_argument("--measure", metavar="WAV", help="only measure a file: integrated LUFS and true peak")
    ap.add_argument("--phone-check", metavar="WAV", help="only run the phone check on a file against --story")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.measure:
        lufs, tp = measure(args.measure)
        print(f"{lufs:.2f} LUFS, {tp:.2f} dBTP")
        return 0
    if args.phone_check:
        ok, misses = phone_check(args.phone_check, args.story)
        print("passed" if ok else f"FAILED: {len(misses)} misses")
        for m in misses:
            print(m)
        return 0 if ok else 1
    report = render(args.story, args.out, args.report, args.cap, args.quiet)
    return 0 if report["phone_check"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

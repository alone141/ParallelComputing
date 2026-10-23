#!/usr/bin/env python3
"""Procedural soundtrack for the reel: a music bed on the timeline's BPM grid
plus sound effects placed at the cue times every beat logs while rendering
(build/cues/*.json). Pure numpy/scipy; output is a loopable 48 kHz stereo WAV
(the tail of the film wraps onto the start, so a replay has no seam).

  python reel/audio/synth.py --out build/soundtrack.wav [--seconds 46.0]

Cue types (BeatScene.cue(kind, dt=..., gain=..., **params)):
  hit, sub_drop, whoosh(dur, direction="up"|"down"), riser(dur),
  tick, ticks_burst(dur, rate0, rate1), glitch(dur), shimmer(dur),
  pad_swell(dur), pop, silence(dur)  # silence ducks the music bed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

REEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REEL))
import timeline  # noqa: E402

SR = 48_000
RNG = np.random.default_rng(1234)
BPM = getattr(timeline, "BPM", 120)
SPB = 60.0 / BPM  # seconds per beat

# D minor world. MIDI note numbers.
CHORDS = [
    [50, 57, 60, 64, 65],  # Dm9    D A C E F
    [46, 53, 57, 60, 62],  # Bbmaj9 Bb F A C D
    [41, 48, 55, 57, 60],  # Fadd9  F C G A C
    [48, 55, 57, 62, 64],  # C6/9   C G A D E
]
PENTA = [62, 65, 67, 69, 72, 74, 77, 79, 81, 84]  # D minor pentatonic, high


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def n_samples(sec: float) -> int:
    return max(int(round(sec * SR)), 1)


def tvec(sec: float) -> np.ndarray:
    return np.arange(n_samples(sec)) / SR


def delay(x: np.ndarray, n: int) -> np.ndarray:
    """Delay by n samples with silence in front (np.roll would wrap the tail
    onto the start, which clicks at the film's loop seam)."""
    return np.concatenate([np.zeros(n), x[:-n]]) if n > 0 else x


def stereo(x: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """Equal-power pan, pan in [-1, 1]."""
    a = (pan + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], axis=1)


def butter(x: np.ndarray, kind: str, freq, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, freq, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def swept_bandpass(x: np.ndarray, f0: float, f1: float, q: float = 1.2, block: int = 256) -> np.ndarray:
    """Bandpass whose centre glides exponentially f0 -> f1 (blockwise, state kept)."""
    out = np.zeros_like(x)
    nblocks = int(np.ceil(len(x) / block))
    zi = None
    for b in range(nblocks):
        frac = b / max(nblocks - 1, 1)
        fc = f0 * (f1 / f0) ** frac
        lo, hi = fc / (1 + 1 / (2 * q)), min(fc * (1 + 1 / (2 * q)), SR / 2 * 0.95)
        sos = signal.butter(2, [lo, hi], btype="band", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        seg = x[b * block:(b + 1) * block]
        out[b * block:(b + 1) * block], zi = signal.sosfilt(sos, seg, zi=zi)
    return out


def env_ad(n: int, attack: float, decay: float, curve: float = 4.0) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    d = np.exp(-curve * np.clip(t - attack, 0, None) / max(decay, 1e-4))
    return a * d


# --------------------------------------------------------------------------
# Instruments. Each returns (stereo audio, anchor sample): the anchor is the
# sample that lands on the cue time (e.g. the peak of a whoosh).
# --------------------------------------------------------------------------
def inst_hit(gain=1.0, **_):
    dur = 1.6
    t = tvec(dur)
    f = 38 + 110 * np.exp(-t * 28)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    body = np.tanh(body * 2.2) * 0.8
    click = butter(RNG.standard_normal(len(t)), "high", 1800) * np.exp(-t * 90) * 0.5
    noise = butter(RNG.standard_normal(len(t)), "band", [200, 3000]) * np.exp(-t * 14) * 0.25
    x = (body + click) * gain
    air = np.stack([noise, delay(noise, 180)], axis=1) * gain  # widen only the noise layer
    return stereo(x) + air, 0


def inst_sub_drop(gain=1.0, dur=1.8, **_):
    t = tvec(dur)
    f = 30 + 70 * np.exp(-t * 2.2)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_ad(len(t), 0.01, dur, 2.5)
    return stereo(np.tanh(x * 1.6) * 0.9 * gain), 0


def inst_whoosh(gain=1.0, dur=0.7, direction="up", **_):
    n = n_samples(dur)
    noise = RNG.standard_normal(n)
    f0, f1 = (250, 7000) if direction == "up" else (7000, 250)
    x = swept_bandpass(noise, f0, f1, q=0.9)
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * t) ** 1.6
    x = x * env / (np.max(np.abs(x * env)) + 1e-9) * 0.6 * gain
    pan = np.linspace(-0.7, 0.7, n) if direction == "up" else np.linspace(0.7, -0.7, n)
    a = (pan + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], axis=1), n // 2


def inst_riser(gain=1.0, dur=1.5, **_):
    n = n_samples(dur)
    t = np.arange(n) / SR
    frac = t / dur
    noise = swept_bandpass(RNG.standard_normal(n), 400, 9000, q=0.7)
    f = 180 * 2 ** (frac * 2.2)
    tone = signal.sawtooth(2 * np.pi * np.cumsum(f) / SR) * 0.25
    tone = butter(tone, "low", 3500)
    trem = 0.75 + 0.25 * np.sin(2 * np.pi * np.cumsum(4 + 22 * frac ** 2) / SR)
    env = frac ** 2.2
    x = (noise / (np.max(np.abs(noise)) + 1e-9) * 0.5 + tone) * env * trem * gain * 0.8
    wide = np.stack([x, delay(x, 240)], axis=1)
    return wide, 0


def inst_tick(gain=1.0, freq=None, pan=None, **_):
    dur = 0.06
    t = tvec(dur)
    f = freq or RNG.choice([3200, 3800, 4400])
    x = np.sin(2 * np.pi * f * t) * np.exp(-t * 160) * 0.35
    x += butter(RNG.standard_normal(len(t)), "high", 5000) * np.exp(-t * 400) * 0.25
    return stereo(x * gain, RNG.uniform(-0.5, 0.5) if pan is None else pan), 0


def inst_pop(gain=1.0, **_):
    dur = 0.18
    t = tvec(dur)
    f = 900 * np.exp(-t * 12) + 500
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_ad(len(t), 0.002, 0.12, 5) * 0.4
    return stereo(x * gain), 0


def inst_ticks_burst(gain=1.0, dur=1.2, rate0=6.0, rate1=40.0, **_):
    n = n_samples(dur + 0.1)
    out = np.zeros((n, 2))
    tt = 0.0
    while tt < dur:
        frac = tt / dur
        tick, _ = inst_tick(gain=0.55 + 0.45 * frac)
        i = n_samples(tt)
        m = min(len(tick), n - i)
        out[i:i + m] += tick[:m]
        tt += 1.0 / (rate0 + (rate1 - rate0) * frac ** 1.5)
    return out * gain, 0


def inst_glitch(gain=1.0, dur=0.35, **_):
    n = n_samples(dur)
    out = np.zeros(n)
    i = 0
    while i < n:
        seg = int(RNG.integers(SR // 120, SR // 25))
        kind = RNG.integers(0, 3)
        t = np.arange(min(seg, n - i)) / SR
        if kind == 0:
            s = signal.square(2 * np.pi * RNG.choice([110, 220, 440, 880]) * t) * 0.3
        elif kind == 1:
            s = np.round(RNG.standard_normal(len(t)) * 3) / 3 * 0.25
        else:
            s = np.zeros(len(t))
        out[i:i + len(t)] = s
        i += seg
    out = butter(out, "band", [150, 9000])
    st = np.stack([out, delay(out, 90)], axis=1)
    return st * gain, 0


def inst_shimmer(gain=1.0, dur=1.6, **_):
    n = n_samples(dur + 1.0)
    out = np.zeros((n, 2))
    notes = RNG.permutation(PENTA)[:6]
    for k, m in enumerate(sorted(notes)):
        start = k * dur / 8
        t = tvec(dur + 1.0 - start)
        f = midi_hz(m)
        x = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 6)) \
            * env_ad(len(t), 0.004, 0.9, 3.5) * 0.16
        i = n_samples(start)
        out[i:i + len(t)] += stereo(x, np.sin(k * 1.7) * 0.6)[: n - i]
    return out * gain, 0


def inst_pad_swell(gain=1.0, dur=2.0, chord=0, **_):
    n = n_samples(dur)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for m in CHORDS[chord % len(CHORDS)]:
        for det in (-0.08, 0.0, 0.08):
            x += signal.sawtooth(2 * np.pi * midi_hz(m + 12 + det) * t) * 0.04
    x = swept_bandpass(x, 300, 6000, q=0.5) * 2.0 + butter(x, "low", 800) * 0.4
    env = (t / dur) ** 1.8
    return np.stack([x, delay(x, 300)], axis=1) * env[:, None] * gain, 0


INSTRUMENTS = {
    "hit": inst_hit, "sub_drop": inst_sub_drop, "whoosh": inst_whoosh, "riser": inst_riser,
    "tick": inst_tick, "ticks_burst": inst_ticks_burst, "glitch": inst_glitch,
    "shimmer": inst_shimmer, "pad_swell": inst_pad_swell, "pop": inst_pop,
}


# --------------------------------------------------------------------------
# Music bed
# --------------------------------------------------------------------------
def energy_curve(total: float) -> np.ndarray:
    """Per-sample energy 0..1 from timeline.ENERGY (beat class -> value)."""
    energy = getattr(timeline, "ENERGY", {})
    n = n_samples(total)
    e = np.zeros(n)
    t0 = 0.0
    for cls, _mod, dur in timeline.BEATS:
        a, b = n_samples(t0), n_samples(t0 + dur)
        e[a:b] = energy.get(cls, 0.5)
        t0 += dur
    if t0 < total:
        e[n_samples(t0):] = e[n_samples(t0) - 1] if n_samples(t0) > 0 else 0.5
    k = n_samples(0.35)  # glide between sections
    return np.convolve(np.pad(e, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]


def pluck(freq: float, dur: float) -> np.ndarray:
    t = tvec(dur)
    x = (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t) * np.exp(-t * 10)
         + 0.12 * signal.sawtooth(2 * np.pi * freq * t)) * env_ad(len(t), 0.003, dur, 5)
    return butter(x, "low", 4200)


def kick() -> np.ndarray:
    t = tvec(0.45)
    f = 45 + 120 * np.exp(-t * 35)
    return np.tanh(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) * 1.8) * 0.75


def music_bed(total: float, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (dry stereo bed, reverb send)."""
    n = n_samples(total)
    pad = np.zeros(n)
    arp = np.zeros((n, 2))
    drums = np.zeros(n)
    hats = np.zeros((n, 2))
    bar = 4 * SPB
    chord_len = 2 * bar

    # Pad: detuned saws per chord, crossfaded, low-passed; brightness follows energy.
    t_all = np.arange(n) / SR
    ci = 0
    t0 = 0.0
    while t0 < total:
        seg_len = min(chord_len + 0.6, total - t0 + 0.6)
        a = n_samples(t0)
        t = tvec(seg_len)
        x = np.zeros(len(t))
        for m in CHORDS[ci % len(CHORDS)]:
            for det in (-0.11, 0.0, 0.13):
                x += signal.sawtooth(2 * np.pi * midi_hz(m + det) * t + RNG.uniform(0, 6.28)) * 0.05
        fade = np.minimum(np.clip(t / 0.6, 0, 1), np.clip((seg_len - t) / 0.6, 0, 1))
        m_ = min(len(x), n - a)
        pad[a:a + m_] += (x * fade)[:m_]
        t0 += chord_len
        ci += 1
    pad_dark = butter(pad, "low", 700)
    pad_bright = butter(pad, "low", 3200)
    pad = pad_dark * (1 - e) + pad_bright * e
    pad *= 0.55 + 0.25 * e
    pad_st = np.stack([pad, delay(pad, 420)], axis=1)

    # Arp (16ths) from energy >= 0.3, hats (8ths) from >= 0.5, kick from >= 0.72
    step = SPB / 4
    k_ = kick()
    for i in range(int(total / step) + 1):
        ts = i * step
        a = n_samples(ts)
        if a >= n:
            break
        ev = e[a]
        ci = int(ts // chord_len) % len(CHORDS)
        if ev >= 0.3:
            notes = CHORDS[ci][1:] + [x + 12 for x in CHORDS[ci][1:3]]
            m = notes[(i * 3) % len(notes)] + 12
            p = pluck(midi_hz(m), 0.35) * (0.08 + 0.10 * ev) * (1.0 if i % 4 == 0 else 0.7)
            seg = stereo(p, np.sin(i * 0.9) * 0.5)
            mm = min(len(seg), n - a)
            arp[a:a + mm] += seg[:mm]
        if ev >= 0.5 and i % 2 == 1:
            h, _ = inst_tick(gain=0.25 + 0.2 * ev, freq=6000)
            mm = min(len(h), n - a)
            hats[a:a + mm] += h[:mm]
        if ev >= 0.72 and i % 4 == 0:
            mm = min(len(k_), n - a)
            drums[a:a + mm] += k_[:mm] * (0.6 + 0.4 * ev)

    dry = pad_st + arp + hats + stereo(drums)
    send = pad_st * 0.5 + arp * 0.8
    return dry, send


def reverb_ir(seconds: float = 2.4) -> np.ndarray:
    t = tvec(seconds)
    irs = []
    for _ in range(2):
        x = RNG.standard_normal(len(t)) * np.exp(-t * 3.0)
        x = butter(x, "low", 5000)
        x[: n_samples(0.012)] *= np.linspace(0, 1, n_samples(0.012))
        irs.append(x / np.sqrt(np.sum(x ** 2)))
    return np.stack(irs, axis=1)


def load_cues() -> list[dict]:
    cues = []
    for f in sorted((REEL / "build" / "cues").glob("*.json")):
        data = json.loads(f.read_text())
        if data.get("scene") not in {b[0] for b in timeline.BEATS}:
            continue
        cues.extend(data.get("cues", []))
    return sorted(cues, key=lambda c: c["t"])


def render(total: float, cues: list[dict]) -> np.ndarray:
    tail = 3.0
    n = n_samples(total + tail)
    e = energy_curve(total + tail)

    duck = np.ones(n)  # sidechain + silences
    for c in cues:
        a = n_samples(c["t"])
        if c["type"] in ("hit", "sub_drop"):
            t = np.arange(n - a) / SR
            duck[a:] *= 1 - 0.65 * c.get("gain", 1.0) * np.exp(-t / 0.22)
        if c["type"] == "silence":
            d = c.get("dur", 0.5)
            b = n_samples(c["t"] + d)
            ramp = n_samples(0.06)
            duck[max(a - ramp, 0):a] *= np.linspace(1, 0.05, len(duck[max(a - ramp, 0):a]))
            duck[a:b] *= 0.05
            duck[b:b + ramp] *= np.linspace(0.05, 1, len(duck[b:b + ramp]))

    dry, send = music_bed(total + tail, e)
    dry *= duck[:, None]
    send *= duck[:, None]

    sfx = np.zeros((n, 2))
    sfx_send = np.zeros((n, 2))
    for c in cues:
        kind = c["type"]
        if kind not in INSTRUMENTS:
            continue
        params = {k: v for k, v in c.items() if k not in ("t", "local_t", "type", "note")}
        audio, anchor = INSTRUMENTS[kind](**params)
        a = n_samples(c["t"]) - anchor
        lo, hi = max(a, 0), min(a + len(audio), n)
        if hi <= lo:
            continue
        sfx[lo:hi] += audio[lo - a:hi - a]
        sfx_send[lo:hi] += audio[lo - a:hi - a] * (0.6 if kind in ("shimmer", "pop", "tick") else 0.3)

    ir = reverb_ir()
    wet_src = send + sfx_send
    wet = np.stack([signal.fftconvolve(wet_src[:, i], ir[:, i])[:n] for i in range(2)], axis=1)
    mix = dry * 0.9 + sfx * 1.0 + wet * 0.35

    # Loop: wrap the tail back onto the start so a replay is seamless.
    nt = n_samples(total)
    out = mix[:nt].copy()
    over = mix[nt:]
    out[: len(over)] += over * np.linspace(1, 0, len(over))[:, None] ** 0.5

    out = butter(out, "high", 28)
    peak = np.max(np.abs(out)) + 1e-9
    out = np.tanh(out / peak * 1.4) / np.tanh(1.4) * 0.89  # gentle soft clip, ~-1 dBFS
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=timeline.FILM_SECONDS)
    a = ap.parse_args()
    cues = load_cues()
    audio = render(a.seconds, cues)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(a.out, SR, (audio * 32767).astype(np.int16))
    print(f"soundtrack: {a.out} ({a.seconds:.2f}s, {len(cues)} cues)")


if __name__ == "__main__":
    main()

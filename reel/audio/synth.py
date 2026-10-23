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


def inst_reverse_swell(gain=1.0, dur=0.05, **_):
    """Reversed noise+tone bloom that ends exactly on the cue (pre-hit suck-in)."""
    t = tvec(dur)
    x = butter(RNG.standard_normal(len(t)), "band", [300, 6000]) * np.exp(-t * 18) * 0.5
    x += np.sin(2 * np.pi * 110 * t) * np.exp(-t * 12) * 0.4
    x = x[::-1] * np.linspace(0.2, 1.0, len(t)) ** 2
    return np.stack([x, delay(x, 120)], axis=1) * gain, len(t)


def inst_noise_burst(gain=1.0, **_):
    """Wide, bright transient layered on climax hits."""
    t = tvec(0.6)
    x = butter(RNG.standard_normal(len(t)), "band", [500, 12000]) * np.exp(-t * 9) * 0.35
    return np.stack([x, delay(x, 260)], axis=1) * gain, 0


INSTRUMENTS = {
    "hit": inst_hit, "sub_drop": inst_sub_drop, "whoosh": inst_whoosh, "riser": inst_riser,
    "tick": inst_tick, "ticks_burst": inst_ticks_burst, "glitch": inst_glitch,
    "shimmer": inst_shimmer, "pad_swell": inst_pad_swell, "pop": inst_pop,
    "reverse_swell": inst_reverse_swell, "noise_burst": inst_noise_burst,
}

# Hit hierarchy: the master limiter flattens equal-gain hits, so tier them
# before mixing. A hit that coincides with a long sub_drop is a climax.
TIER_ROUTINE, TIER_LANDING, TIER_CLIMAX = 0.6, 0.8, 1.0


def tier_hits(cues: list[dict]) -> list[dict]:
    drops = [c for c in cues if c["type"] == "sub_drop"]
    out = []
    for c in cues:
        if c["type"] != "hit":
            out.append(c)
            continue
        paired = [d for d in drops if abs(d["t"] - c["t"]) < 0.02]
        climax = any(d.get("dur", 0) >= 1.4 for d in paired) and c.get("gain", 1.0) >= 1.0
        tier = TIER_CLIMAX if climax else (TIER_LANDING if paired else TIER_ROUTINE)
        out.append({**c, "gain": c.get("gain", 1.0) * tier})
        if climax:
            out.append({"t": c["t"], "type": "reverse_swell", "gain": 0.9})
            out.append({"t": c["t"], "type": "noise_burst", "gain": 1.0})
    return sorted(out, key=lambda c: c["t"])


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
    nt = n_samples(total)
    e = energy_curve(total + tail)
    cues = tier_hits(cues)

    side = np.ones(n)  # sidechain ducking of the bed under hits
    for c in cues:
        if c["type"] in ("hit", "sub_drop"):
            a = n_samples(c["t"])
            t = np.arange(n - a) / SR
            side[a:] *= 1 - 0.65 * min(c.get("gain", 1.0), 1.0) * np.exp(-t / 0.22)

    dry, send = music_bed(total + tail, e)
    # The bed ends with the film (short fade); it must not wrap onto the start,
    # or the replay would hear two beds at once.
    fade = n_samples(0.03)
    bed_env = np.zeros(n)
    bed_env[:nt] = 1.0
    bed_env[nt - fade:nt] = np.linspace(1, 0, fade)
    dry *= (side * bed_env)[:, None]
    send *= (side * bed_env)[:, None]

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

    # Loop: SFX and reverb tails that run past the end wrap onto the start.
    ambience = dry * 0.9 + wet * 0.35
    out = ambience[:nt].copy()
    over = ambience[nt:] + sfx[nt:]
    out[: len(over)] += over * np.linspace(1, 0, len(over))[:, None] ** 0.5

    # Silences are applied after the wrap so nothing leaks into them. Short
    # ones (freeze frames, pre-drop gaps) are true silence; long ones duck.
    hush = np.ones(nt)
    for c in cues:
        if c["type"] != "silence":
            continue
        d = c.get("dur", 0.5)
        depth = c.get("depth", 0.003 if d <= 0.35 else 0.1)
        a, b = n_samples(c["t"]), min(n_samples(c["t"] + d), nt)
        r_in, r_out = n_samples(0.015), n_samples(0.01)
        lo = max(a - r_in, 0)
        hush[lo:a] = np.minimum(hush[lo:a], np.linspace(1, depth, a - lo))
        hush[a:b] = np.minimum(hush[a:b], depth)
        hush[b:b + r_out] = np.minimum(hush[b:b + r_out], np.linspace(depth, 1, len(hush[b:b + r_out])))
    out *= hush[:, None]
    out += sfx[:nt]  # cue sounds themselves are placed after the hush (they are intentional)

    out = butter(out, "high", 28)
    return master(out)


TARGET_LUFS = -14.5  # TikTok/Reels-friendly; platforms normalise around -14
CEILING = 0.72       # ~ -2.9 dBFS sample peak: AAC adds ~1.7 dB of inter-sample overshoot, so true peak stays under -1 dBTP


def lufs(x: np.ndarray) -> float:
    """Integrated loudness per ITU-R BS.1770-4 (K-weighting + gating), 48 kHz."""
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    k = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=0), axis=0)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    starts = range(0, len(k) - blk + 1, hop)
    z = np.array([np.mean(k[i:i + blk] ** 2, axis=0).sum() for i in starts])
    lk = -0.691 + 10 * np.log10(z + 1e-12)
    z = z[lk > -70]
    rel = -0.691 + 10 * np.log10(z.mean() + 1e-12) - 10
    z = z[(-0.691 + 10 * np.log10(z + 1e-12)) > rel]
    return float(-0.691 + 10 * np.log10(z.mean() + 1e-12))


def master(x: np.ndarray) -> np.ndarray:
    """Soft-clip limiter with its drive solved for TARGET_LUFS. The hit tiers are
    set before this, so the hierarchy survives (peaks compress, order holds)."""
    x = x / (np.max(np.abs(x)) + 1e-9)

    def limit(g):
        return np.tanh(x * g) / np.tanh(max(g, 1e-3)) * CEILING if g > 0 else x

    lo, hi = 0.3, 8.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if lufs(limit(mid)) < TARGET_LUFS:
            lo = mid
        else:
            hi = mid
    return limit((lo + hi) / 2)


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

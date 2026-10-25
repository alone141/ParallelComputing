"""Procedural score for The Shuffle, generated from the same timeline as the picture.

    python soundtrack.py media/soundtrack.wav

* a soft bloom and an A4 bell at frame 0, a riser into the first card
* a tick on every card dealt (a buzz in the sprint), a rising pluck on cards 1-3
* whooshes whose brightness follows the rate of dealing
* a soft ratchet while each pass-row counter climbs, and a chime when it lands
* the "You" chord at card 6, the payoff chord and answer shimmer at card 11,
  the finale chord at card 52, a quieter bell on each "imagine" beat
* a pad whose brightness follows the count (log₂ of the orders), and a convolution reverb
* a master pass that measures the mix with ffmpeg's ebur128 and lands it at -15 LUFS,
  true peak under -1 dBTP (film-style §7)
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from story import BEATS, BY_KEY, CARDS, HOOK, LOG2_ALL, NOTE_IN, ROW_REF_PART, STEP_TIMES, STOPS, TIMELINE, log2_orders  # noqa: E402

SR = 48_000
TARGET_LUFS = -15.0


def note(name: str) -> float:
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    pitch, octave = name[:-1], int(name[-1])
    return 440.0 * 2 ** ((names[pitch] + 12 * (octave - 4)) / 12)


class Mix:
    def __init__(self, seconds):
        self.n = int(seconds * SR) + SR
        self.dry = np.zeros((self.n, 2))
        self.send = np.zeros((self.n, 2))  # goes to the reverb

    def add(self, t0, mono, gain=1.0, pan=0.0, send=0.3):
        i = int(t0 * SR)
        if i >= self.n or i < 0:
            return
        mono = mono[: self.n - i]
        left, right = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        st = np.stack([mono * left, mono * right], 1) * gain
        self.dry[i : i + len(mono)] += st
        self.send[i : i + len(mono)] += st * send


def bell(freq, dur=3.0, decay=1.0, bright=1.0):
    t = np.arange(int(dur * SR)) / SR
    parts = [(1.0, 1.0, 1.0), (2.0, 0.5 * bright, 0.62), (3.01, 0.28 * bright, 0.42),
             (4.2, 0.14 * bright, 0.3), (5.43, 0.07 * bright, 0.2)]
    out = sum(a * np.sin(2 * np.pi * freq * k * t) * np.exp(-t / (decay * d)) for k, a, d in parts)
    attack = np.clip(t / 0.004, 0, 1)
    return out * attack * 0.35


def pluck(freq, dur=0.9):
    t = np.arange(int(dur * SR)) / SR
    out = sum(np.sin(2 * np.pi * freq * k * t) / k**1.6 for k in range(1, 7)) * np.exp(-t / 0.18)
    return out * np.clip(t / 0.002, 0, 1) * 0.3


def boom(dur=2.4, f0=95.0, f1=36.0, decay=0.7):
    t = np.arange(int(dur * SR)) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.12)
    phase = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(phase) * np.exp(-t / decay)
    click = np.random.default_rng(3).normal(size=len(t)) * np.exp(-t / 0.004) * 0.15
    return (body + click) * np.clip(t / 0.003, 0, 1)


def tick(rng):
    t = np.arange(int(0.035 * SR)) / SR
    env = np.exp(-t / 0.0055)
    tone = np.sin(2 * np.pi * 2650 * t) + 0.45 * np.sin(2 * np.pi * 3980 * t)
    noise = rng.normal(size=len(t)) * np.exp(-t / 0.0012) * 0.6
    return (tone * env + noise) * 0.5


def lowpass_blocks(x, cutoff, block=4096):
    """Overlap-add FFT low-pass whose cutoff (Hz, per sample) changes over time."""
    hop = block // 2
    win = np.hanning(block)
    out = np.zeros(len(x) + block)
    xp = np.concatenate([x, np.zeros(block)])
    freqs = np.fft.rfftfreq(block, 1 / SR)
    cutoff = np.broadcast_to(np.asarray(cutoff, float), (len(x),)) if np.ndim(cutoff) == 0 else cutoff
    for start in range(0, len(x), hop):
        seg = xp[start : start + block] * win
        c = cutoff[min(start + hop, len(cutoff) - 1)]
        mask = 1 / (1 + (freqs / max(c, 20)) ** 4)
        out[start : start + block] += np.fft.irfft(np.fft.rfft(seg) * mask, n=block)
    return out[: len(x)]


def reverb_ir(seconds=3.2, seed=5):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.normal(size=(n, 2)) * np.exp(-t / 0.75)[:, None]
    ir[:, 0] = lowpass_blocks(ir[:, 0], 9000 * np.exp(-t / 1.2) + 800)
    ir[:, 1] = lowpass_blocks(ir[:, 1], 9000 * np.exp(-t / 1.2) + 800)
    ir[: int(0.012 * SR)] = 0  # pre-delay
    return ir / np.sqrt((ir**2).sum(axis=0))


def ratchet(mix, rs, duration, clicks=22, gain=0.05, chime=True):
    """Soft clicks following the picture's ease-in-out over ``duration``, then a chime."""
    u = np.linspace(0, 1, 4000)
    p = np.where(u < 0.5, 4 * u**3, 1 - (-2 * u + 2) ** 3 / 2)
    for i in range(1, clicks + 1):
        tc = rs + duration * float(u[np.argmax(p >= i / clicks)])
        ct = np.arange(int(0.02 * SR)) / SR
        f = 1500 + 1700 * i / clicks
        click = np.sin(2 * np.pi * f * ct) * np.exp(-ct / 0.004)
        mix.add(tc, click, gain=gain, pan=-0.3 + 0.6 * i / clicks, send=0.15)
    if chime:
        mix.add(rs + duration, bell(note("E6"), dur=1.6, decay=0.5, bright=0.3), gain=0.09, send=0.5)


def build():
    tl = TIMELINE
    total = tl.total
    mix = Mix(total)
    rng = np.random.default_rng(11)
    t = np.arange(mix.n) / SR
    tc = np.clip(t, 0, total)
    grid = np.linspace(0, total, int(total * 400) + 1)
    n_grid = tl.n_at(grid)
    v_grid = np.abs(tl.speed_at(grid))
    l2_grid = np.array([log2_orders(x) for x in n_grid])
    n_t = np.interp(tc, grid, n_grid)
    v_t = np.interp(tc, grid, v_grid)
    l2_t = np.interp(tc, grid, l2_grid)
    speed = np.clip(v_t / 20.0, 0, 1)
    outro = tl.outro_start

    # pad: A minor, opening up as the count climbs ---------------------------------
    bright = 1.4 + 4.5 * np.clip(l2_t / LOG2_ALL, 0, 1)
    pad = np.zeros(mix.n)
    for f in (note("A2"), note("E3"), note("A3"), note("C4"), note("E4")):
        for det in (-0.004, 0.0035):
            ff = f * (1 + det)
            for k in range(1, 9):
                pad += np.sin(2 * np.pi * ff * k * t + rng.uniform(0, 6.3)) * np.exp(-k / bright) / k
    env = np.clip(t / 1.2, 0, 1) * (0.75 + 0.25 * np.sin(2 * np.pi * 0.11 * t))
    env *= np.where(tc > total - 1.5, np.clip((total - tc) / 1.5, 0, 1) * 0.7 + 0.3, 1.0)
    mix.add(0, pad * env * 0.018, send=0.5)

    # whoosh: follows the rate of dealing ---------------------------------------------
    for side, pan in ((0, -0.6), (1, 0.6)):
        noise = rng.normal(size=mix.n)
        w = lowpass_blocks(noise, 150 + 4200 * speed**1.6)
        mix.add(0, w * speed**1.5 * 0.055, pan=pan, send=0.15)

    # a tick on every card, at the moment the picture lands it ----------------------------
    for k in range(1, CARDS + 1):
        tt = tl.card_time(k)
        if tt >= outro:
            break
        rate = abs(float(tl.speed_at(tt)))
        g = 0.16 * min(1.0, (6.0 / (rate + 1e-3)) ** 0.55)
        mix.add(tt, tick(rng), gain=g, pan=0.25 if k % 2 else -0.25, send=0.12)

    # hook: soft bloom at frame one, riser into the first card ------------------------
    mix.add(0.0, boom(dur=2.0, f0=70, f1=40, decay=0.5), gain=0.35, send=0.2)
    mix.add(0.02, bell(note("A4"), decay=1.4, bright=0.5), gain=0.22, send=0.6)
    rt = np.arange(int(1.4 * SR)) / SR
    riser = lowpass_blocks(rng.normal(size=len(rt)), 300 + 5000 * (rt / 1.4) ** 2) * (rt / 1.4) ** 2.2
    mix.add(HOOK - 1.4, riser, gain=0.09, send=0.3)
    steps = ("E4", "G4", "A4", "C5", "D5", "E5", "G5", "A5")[: len(STEP_TIMES)]
    for k, nm in enumerate(steps):
        mix.add(tl.card_time(k + 1), pluck(note(nm)), gain=0.42, pan=-0.35 + 0.7 * k / max(len(steps) - 1, 1), send=0.4)

    # pass rows: a ratchet while the counter climbs, a chime when it lands -------------
    for s in STOPS:
        if s.key in tl.row_start:
            ratchet(mix, tl.row_start[s.key], ROW_REF_PART)  # the counter climbs with the reference fill
    # beat rows: a quiet sweep as the orders pass the shuffles' point of light
    for b in BEATS:
        rs = tl.beat_row_start[b.key] + 0.2
        st = np.arange(int(1.3 * SR)) / SR
        sweep = lowpass_blocks(rng.normal(size=len(st)), 400 + 3000 * (st / 1.3) ** 2) * np.sin(np.pi * st / 1.3) ** 1.5
        mix.add(rs, sweep, gain=0.035, send=0.5)
        mix.add(rs + 1.3, bell(note("E6"), dur=1.4, decay=0.45, bright=0.3), gain=0.05, send=0.6)

    # the stops: every one is a big reveal (film-style §7) --------------------------------
    for key, chord, g, boom_gain, decay in (("everyone", ("A3", "E4", "A4", "C#5"), 0.32, 0.45, 0.55),
                                            ("bigbang", ("A2", "E3", "A3", "C4", "E4", "B4"), 0.36, 0.75, 0.9),
                                            ("deck", ("F2", "C3", "A3", "E4", "G4", "C5"), 0.4, 0.75, 0.9)):
        ta = tl.arrive[key]
        mix.add(ta - 0.01, boom(dur=3.0, f0=95, f1=34, decay=decay), gain=boom_gain, send=0.15)
        for j, nm in enumerate(chord):
            mix.add(ta + 0.025 * j, bell(note(nm), dur=4.5, decay=2.0, bright=0.6), gain=g / len(chord) ** 0.5,
                    pan=(j / (len(chord) - 1) - 0.5) * 0.8, send=0.6)
    # the answer: a shimmer when "Not even a quarter of the deck." lands
    t_note = tl.note_time["bigbang"] + NOTE_IN
    for j, nm in enumerate(("E5", "A5", "B5", "E6")):
        mix.add(t_note + 0.07 * j, bell(note(nm), decay=0.9, bright=0.4), gain=0.1, pan=(-0.4, 0.4)[j % 2], send=0.8)
    # the beats: a quieter bell each, climbing, over a small thump
    for b, nm in zip(BEATS, ("C5", "D5", "E5")):
        ta = tl.beat_arrive[b.key]
        mix.add(ta, bell(note(nm), decay=1.1, bright=0.6), gain=0.2, send=0.5)
        mix.add(ta - 0.012, boom(dur=0.9, f0=120, f1=60, decay=0.18), gain=0.08, send=0.1)

    # outro: swell, then the chord for "52" -------------------------------------------------
    sw = np.arange(int(0.9 * SR)) / SR
    swell = lowpass_blocks(rng.normal(size=len(sw)), 400 + 6000 * (sw / 0.9) ** 2) * (sw / 0.9) ** 3
    mix.add(outro - 0.25, swell, gain=0.12, send=0.4)
    mix.add(outro + 0.65, boom(dur=3.5, f0=80, f1=32, decay=1.2), gain=0.6, send=0.2)
    for j, nm in enumerate(("A2", "E3", "B3", "C4", "E4", "A4")):
        mix.add(outro + 0.65 + 0.03 * j, bell(note(nm), dur=5.0, decay=2.4, bright=0.5), gain=0.17,
                pan=(j / 5 - 0.5) * 0.9, send=0.7)

    # reverb and a first master ----------------------------------------------------------------
    ir = reverb_ir()
    wet = np.stack([fftconvolve(mix.send[:, c], ir[:, c])[: mix.n] for c in range(2)], 1)
    out = mix.dry + wet * 0.55
    # a frame longer than the picture at 30 or 60 fps, so the mux's -shortest trims the sound, never a frame
    out = out[: int((total + 1 / 30) * SR)]
    fade = np.clip((total - np.arange(len(out)) / SR) / 0.35, 0, 1)[:, None]
    out *= fade
    rms = np.sqrt((out**2).mean())
    out *= 10 ** (-14 / 20) / (rms + 1e-12)
    out = np.tanh(out * 1.1) / np.tanh(1.1)
    out *= 10 ** (-1.5 / 20) / np.abs(out).max()
    return out


def write_wav(path, data):
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def measure(path):
    """Integrated loudness (LUFS) and true peak (dBTP) of an audio or video file, via ffmpeg ebur128."""
    r = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    err = r.stderr
    summary = err[err.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    p = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)
    if not (i and p):
        raise RuntimeError("could not parse ebur128 output:\n" + err[-2000:])
    return float(i.group(1)), float(p.group(1))


def master(data, target=TARGET_LUFS, peak_limit=-1.3):
    """Land the mix at ``target`` LUFS with the true peak under ``peak_limit`` dBTP (so the AAC
    encode stays under -1 dBTP), by gain and, if needed, gentle soft clipping."""
    for _ in range(4):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)
        write_wav(tmp, data)
        lufs, peak = measure(tmp)
        tmp.unlink()
        if abs(lufs - target) < 0.15 and peak <= peak_limit:
            return data, lufs, peak
        data = data * 10 ** ((target - lufs) / 20)
        if peak + (target - lufs) > peak_limit:
            k = 1.0 + 0.35 * (peak + (target - lufs) - peak_limit)
            data = np.tanh(data * k) / np.tanh(k)
            data *= min(1.0, 10 ** (peak_limit / 20) / (np.abs(data).max() + 1e-12))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    write_wav(tmp, data)
    lufs, peak = measure(tmp)
    tmp.unlink()
    return data, lufs, peak


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "media/soundtrack.wav"
    audio, lufs, peak = master(build())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_path, audio)
    print(f"wrote {out_path}: {len(audio) / SR:.2f}s, {lufs:.1f} LUFS, true peak {peak:.1f} dBTP")

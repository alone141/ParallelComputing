#!/usr/bin/env python3
"""Synthesise the intro of Gojira's "The Art of Dying" (compressed) with numpy only.

This is the ONLY file that knows the score. Everything downstream (analyse.py,
render.py) works from the rendered audio alone.

Timeline (140 bpm, one sixteenth = 15/140 s = 107.14 ms, 45-sixteenth cycle):
  laps 0-1  : rim clicks alone over a held choir pad            (record: bars 1-11)
  laps 2-5  : rim clicks + tribal stacked hit on every beat     (record: bars 12-33)
  lap 6     : kit + bass + two guitars, rim clicks stop         (record: bars 42-77)
  + a final tribal hit on the downbeat after lap 6, and a 1.6 s tail.
"""
import numpy as np, sys, os, json

SR = 44100
BPM = 140.0
SIXT = 15.0 / BPM                # 107.142857 ms
CYCLE = 45
RESTS = {4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44}          # rim-click rests
SNARE = {3, 8, 13, 17, 20, 23, 27, 32, 37, 41, 44}
KICK_RESTS = {4, 9, 14, 28, 33, 38}
LAPS_A, LAPS_B, LAPS_C = 2, 4, 1
LAPS = LAPS_A + LAPS_B + LAPS_C
N16 = LAPS * CYCLE                                           # 405 sixteenths
TAIL = 1.6
TOTAL = N16 * SIXT + TAIL

rng = np.random.default_rng(7)

def env(n, attack, decay, curve=1.0):
    t = np.arange(n) / SR
    e = np.exp(-t / decay) ** curve
    a = np.minimum(1.0, t / max(attack, 1e-5))
    return e * a

def lowpass(x, fc):
    """One-pole lowpass, vectorised via FFT (fine for short one-shots and long pads)."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    H = 1 / np.sqrt(1 + (f / fc) ** 4)
    return np.fft.irfft(X * H, n)

def highpass(x, fc):
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    H = (f / fc) ** 2 / np.sqrt(1 + (f / fc) ** 4)
    return np.fft.irfft(X * H, n)

def bandpass(x, lo, hi):
    return highpass(lowpass(x, hi), lo)

def sine(freq, n, phase=0.0):
    return np.sin(2 * np.pi * freq * np.arange(n) / SR + phase)

def sweep(f0, f1, n, tau):
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / tau)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)

# ---------------------------------------------------------------- one-shots
def side_stick():
    n = int(0.045 * SR)
    tone = sine(1750, n) * env(n, 0.0005, 0.006) + 0.5 * sine(3900, n) * env(n, 0.0003, 0.003)
    nz = bandpass(rng.standard_normal(n), 1500, 9000) * env(n, 0.0003, 0.004)
    return 0.8 * tone + 0.6 * nz

def sticks():
    n = int(0.03 * SR)
    nz = bandpass(rng.standard_normal(n), 2500, 12000) * env(n, 0.0002, 0.0025)
    return 0.9 * nz + 0.3 * sine(5200, n) * env(n, 0.0002, 0.002)

def rim_click():
    s = side_stick(); k = sticks()
    out = np.zeros(max(len(s), len(k)))
    out[:len(s)] += s; out[:len(k)] += k
    return out * 0.9

def tribal_hit():
    n = int(0.8 * SR)
    bd = sweep(130, 48, n, 0.03) * env(n, 0.001, 0.22)
    tom = sweep(110, 82, n, 0.08) * env(n, 0.002, 0.35)
    surdo = sweep(64, 55, n, 0.1) * env(n, 0.003, 0.55)
    wood = sine(1150, n) * env(n, 0.0005, 0.012) + 0.4 * sine(2300, n) * env(n, 0.0005, 0.008)
    cast = bandpass(rng.standard_normal(n), 2000, 8000) * env(n, 0.0005, 0.012)
    cast2 = np.zeros(n); d = int(0.012 * SR)
    cast2[d:] = cast[:-d] * 0.7
    out = 1.0 * bd + 0.7 * tom + 0.8 * surdo + 0.35 * wood + 0.35 * (cast + cast2)
    return np.tanh(1.3 * out) * 0.95

def kick():
    n = int(0.3 * SR)
    body = sweep(160, 46, n, 0.025) * env(n, 0.0005, 0.11)
    click = highpass(rng.standard_normal(n), 2500) * env(n, 0.0002, 0.003)
    return np.tanh(1.5 * (body + 0.25 * click)) * 0.9

def snare():
    n = int(0.25 * SR)
    nz = bandpass(rng.standard_normal(n), 300, 9000) * env(n, 0.0005, 0.07)
    tone = (sine(190, n) + 0.5 * sine(330, n)) * env(n, 0.0005, 0.05)
    return np.tanh(1.4 * (0.9 * nz + 0.7 * tone)) * 0.85

def ride_ghost():
    n = int(0.12 * SR)
    nz = bandpass(rng.standard_normal(n), 4000, 14000) * env(n, 0.0005, 0.03)
    return 0.22 * nz

def ride_bell():
    n = int(0.5 * SR)
    parts = [(2630, 1.0), (3410, 0.6), (4850, 0.45), (6120, 0.3), (1870, 0.3)]
    out = sum(a * sine(f, n, rng.uniform(0, 6.28)) for f, a in parts)
    nz = bandpass(rng.standard_normal(n), 3000, 12000) * env(n, 0.0003, 0.01)
    return (out * env(n, 0.0005, 0.18) * 0.35 + 0.4 * nz) * 0.7

def guitar_dead_note(n_samples):
    """Palm-muted dead note on C2 (65.4 Hz): short, thuddy, lots of pick noise."""
    n = int(0.11 * SR)
    f0 = 65.41
    t = np.arange(n) / SR
    saw = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    body = lowpass(saw, 900) * env(n, 0.0005, 0.03)
    pick = bandpass(rng.standard_normal(n), 1200, 7000) * env(n, 0.0002, 0.004)
    out = np.tanh(2.5 * (body + 0.5 * pick))
    return out * 0.5

def bass_note(dur):
    """C1 (32.7 Hz) bass, slightly driven, holds for dur seconds."""
    n = int(dur * SR)
    f0 = 32.70
    t = np.arange(n) / SR
    saw = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 20))
    x = lowpass(saw, 700)
    e = env(n, 0.004, 0.6)
    e[-int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))
    return np.tanh(2.0 * x) * e * 0.45

def choir_pad(dur):
    """Held airy note (C3 + G3 + C4) with slow vibrato: the record's choir drone."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f, a in [(130.81, 1.0), (196.0, 0.5), (261.63, 0.5), (523.25, 0.15)]:
        for dv, pa in [(0, 1.0), (0.4, 0.6), (-0.5, 0.6)]:
            vib = 1 + 0.003 * np.sin(2 * np.pi * 0.27 * t + pa)
            out += a * np.sin(2 * np.pi * (f + dv) * vib * t + pa)
    breath = lowpass(rng.standard_normal(n), 1800) * 0.4
    out = lowpass(out, 2500) + 0.06 * breath
    out /= np.max(np.abs(out)) + 1e-9
    fade = np.ones(n)
    fi = int(1.5 * SR); fade[:fi] = np.linspace(0, 1, fi)
    fo = int(1.0 * SR); fade[-fo:] = np.linspace(1, 0, fo)
    return out * fade * 0.10

# ---------------------------------------------------------------- score -> mix
def place(buf, sample, start_s, gain=1.0, pan=0.0):
    i = int(round(start_s * SR))
    if i >= buf.shape[1]:
        return
    seg = sample[: buf.shape[1] - i]
    l = (1 - pan) / 2 + 0.5; r = (1 + pan) / 2 + 0.5
    buf[0, i:i + len(seg)] += seg * gain * min(1.0, l)
    buf[1, i:i + len(seg)] += seg * gain * min(1.0, r)

def main(out_dir):
    n_total = int(TOTAL * SR)
    stems = {k: np.zeros((2, n_total)) for k in ["rims", "tribal", "kit", "bass", "guitars", "pad"]}
    events = []     # (time, part) for the README / alignment notes only
    RIM = rim_click(); TRIB = tribal_hit(); STK = sticks()
    KICK = kick(); SN = snare(); GH = ride_ghost(); BELL = ride_bell()
    for n in range(N16):
        t0 = n * SIXT
        lap, slot = divmod(n, CYCLE)
        on_beat = (n % 4 == 0)
        if lap < LAPS_A + LAPS_B:                     # rim clicks (laps 0-5)
            if slot not in RESTS:
                g = 1.0 + 0.04 * rng.standard_normal()
                place(stems["rims"], RIM, t0 + rng.normal(0, 0.0015), g, pan=0.15)
                events.append((t0, "rim"))
        if LAPS_A <= lap < LAPS_A + LAPS_B and on_beat:    # tribal hit on beats (laps 2-5)
            place(stems["tribal"], STK, t0 - SIXT / 2, 0.8, pan=-0.1)   # 32nd grace note
            place(stems["tribal"], TRIB, t0, 1.0)
            events.append((t0, "tribal"))
        if lap >= LAPS_A + LAPS_B:                     # full band (laps 6-8)
            if slot in SNARE:
                place(stems["kit"], SN, t0, 1.0)
                events.append((t0, "snare"))
            elif slot not in KICK_RESTS:
                place(stems["kit"], KICK, t0, 1.0 if n % 2 == 0 else 0.92)
                events.append((t0, "kick"))
            if on_beat:
                place(stems["kit"], BELL, t0, 1.0, pan=0.35)
            elif n % 2 == 0:
                place(stems["kit"], GH, t0, 1.0, pan=0.35)
            # bass: rim-click pattern on C1, last note of each group held through the rest
            if slot not in RESTS:
                hold = 1 if (slot + 1) not in RESTS else 2
                place(stems["bass"], bass_note(hold * SIXT), t0, 1.0)
            # two guitars, palm-muted dead notes on every sixteenth
            place(stems["guitars"], guitar_dead_note(0), t0 + rng.normal(0, 0.002), 1.0, pan=-0.6)
            place(stems["guitars"], guitar_dead_note(0), t0 + rng.normal(0, 0.002), 1.0, pan=0.6)
    # closing downbeat
    t_end = N16 * SIXT
    place(stems["tribal"], STK, t_end - SIXT / 2, 0.8)
    place(stems["tribal"], TRIB, t_end, 1.1)
    place(stems["kit"], KICK, t_end, 1.0)
    place(stems["bass"], bass_note(1.2), t_end, 1.0)
    events.append((t_end, "tribal"))
    # pad under section A (and fading into B)
    pad = choir_pad(LAPS_A * CYCLE * SIXT + 3.0)
    stems["pad"][0, :len(pad)] += pad; stems["pad"][1, :len(pad)] += pad

    gains = {"rims": 0.9, "tribal": 1.0, "kit": 0.95, "bass": 0.9, "guitars": 0.55, "pad": 1.0}
    mix = sum(stems[k] * gains[k] for k in stems)
    # gentle bus saturation + peak normalise
    mix = np.tanh(1.2 * mix)
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 0.89
    os.makedirs(out_dir, exist_ok=True)
    write_wav(os.path.join(out_dir, "mix.wav"), mix)
    for k in stems:
        s = stems[k] * gains[k]
        write_wav(os.path.join(out_dir, f"stem_{k}.wav"), np.tanh(1.2 * s) * 0.89)
    with open(os.path.join(out_dir, "alignment.json"), "w") as f:
        json.dump({"bpm": BPM, "sixteenth_s": SIXT, "cycle": CYCLE, "laps": [LAPS_A, LAPS_B, LAPS_C],
                   "total_s": TOTAL, "n_sixteenths": N16,
                   "note": "ground truth for the README only; analyse.py never reads this"}, f, indent=1)
    print(f"wrote mix.wav: {TOTAL:.2f} s, {N16} sixteenths, {LAPS} laps")

def write_wav(path, x):
    import wave
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.T.tobytes(order="C"))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")

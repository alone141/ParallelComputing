"""Synthesised soundtrack for the rhythm-game film (numpy only).

Follows the record's arrangement for the first eight laps of the intro:
  laps 1-4  rim clicks alone over a held choir-like drone
            (plus a very quiet metronome thump on every pulse, which the record
            does not have: it makes the drift audible before the drum enters)
  laps 5-8  rim clicks + the big stacked "tribal" hit on every pulse,
            with a stick grace note a 32nd before each hit.
The film ends on step 360 (second realignment) with a crash.
"""
import numpy as np, sys, wave

SR = 48000
BPM = 140
STEP = 15 / BPM
CYCLE = 45
RESTS = {4, 9, 14, 17, 20, 23, 28, 33, 38, 41, 44}
LAPS = 8
END_STEP = CYCLE * LAPS
DRUM_IN = CYCLE * 4
T0 = 1.5
END_HOLD = 3.3
DUR = T0 + END_STEP * STEP + END_HOLD

rng = np.random.default_rng(45)

def env(n, attack, decay, curve=1.0):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-5), 0, 1)
    d = np.exp(-t / decay) ** curve
    return a * d

def sine(freq, n, f_end=None, phase=0.0):
    t = np.arange(n) / SR
    if f_end is None:
        return np.sin(2 * np.pi * freq * t + phase)
    # exponential sweep from freq to f_end over the buffer
    k = np.log(f_end / freq)
    ph = 2 * np.pi * freq * (np.expm1(k * t / t[-1]) / (k / t[-1]) if k != 0 else t)
    return np.sin(ph + phase)

def noise(n):
    return rng.standard_normal(n)

def onepole_lp(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x); acc = 0.0
    for i in range(len(x)):
        acc = a * acc + (1 - a) * x[i]; y[i] = acc
    return y

def bandpass_fft(x, lo, hi):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))

# ---- instruments ----
def rim_click():
    n = int(0.06 * SR)
    body = bandpass_fft(noise(n), 1500, 7000) * env(n, 0.0003, 0.006)
    tock = sine(2300, n) * env(n, 0.0002, 0.012) * 0.6 + sine(1150, n) * env(n, 0.0002, 0.02) * 0.4
    return 0.55 * (body * 0.9 + tock)

def stick_grace():
    n = int(0.04 * SR)
    return 0.25 * (bandpass_fft(noise(n), 2500, 9000) * env(n, 0.0002, 0.004) + sine(3100, n) * env(n, 0.0002, 0.006) * 0.4)

def soft_pulse():
    n = int(0.25 * SR)
    return 0.22 * sine(80, n, 42) * env(n, 0.002, 0.08)

def tribal_hit():
    n = int(0.9 * SR)
    kick = sine(120, n, 46) * env(n, 0.001, 0.10) * 1.0
    tom = sine(96, n, 78) * env(n, 0.001, 0.16) * 0.7
    surdo = sine(58, n, 52) * env(n, 0.004, 0.22) * 0.8
    block = (sine(1450, n) * env(n, 0.0002, 0.018) + sine(2150, n) * env(n, 0.0002, 0.012) * 0.5) * 0.35
    cast = bandpass_fft(noise(n), 2000, 6000) * (env(n, 0.0002, 0.008) + np.roll(env(n, 0.0002, 0.006), int(0.018 * SR))) * 0.3
    sticks = bandpass_fft(noise(n), 1200, 5000) * env(n, 0.0003, 0.02) * 0.3
    x = kick + tom + surdo + block + cast + sticks
    x = x / max(1e-9, np.max(np.abs(x)))
    return 0.75 * np.tanh(1.3 * x) / np.tanh(1.3)

def crash():
    n = int(2.2 * SR)
    x = bandpass_fft(noise(n), 3000, 14000) * env(n, 0.002, 0.55)
    return 0.22 * x

def drone(n):
    """Held choir-like chord on C: slow attack, detuned partials, slow vibrato."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f0, amp in [(130.81, 1.0), (196.0, 0.7), (261.63, 0.5), (329.63, 0.3)]:
        for det in (-0.35, 0.0, 0.35):
            vib = 1 + 0.004 * np.sin(2 * np.pi * (0.17 + 0.05 * det) * t + det)
            for k in range(1, 7):
                out += amp * (1 / k ** 1.8) * np.sin(2 * np.pi * (f0 + det) * k * vib * t + k * det)
    out /= np.max(np.abs(out))
    # formant-ish colouring: a touch of band emphasis around 700 Hz and 2.5 kHz
    out = 0.7 * out + 0.2 * bandpass_fft(out, 500, 900) + 0.1 * bandpass_fft(out, 2200, 2900)
    return out

def place(buf, sample, t):
    i = int(round(t * SR))
    if i < 0: return
    j = min(len(buf), i + len(sample))
    buf[i:j] += sample[: j - i]

def main(out):
    n = int(DUR * SR)
    mix = np.zeros(n)
    click, grace, soft, hit, cr = rim_click(), stick_grace(), soft_pulse(), tribal_hit(), crash()
    for s in range(END_STEP + 1):
        t = T0 + s * STEP
        if s % CYCLE not in RESTS:
            # tiny humanisation: +-1.5 ms, slightly louder on group starts
            place(mix, click * (1.0 if s % CYCLE in (0, 5, 10, 15, 18, 21, 24, 29, 34, 39, 42) else 0.85), t + rng.normal(0, 0.0015))
        if s % 4 == 0:
            if s < DRUM_IN:
                place(mix, soft, t)
            else:
                place(mix, grace, t - STEP / 2)
                place(mix, hit, t)
        if s in (DRUM_IN, END_STEP):
            place(mix, cr, t)
    # drone under everything, fading in over 2 s, out over the end hold
    d = drone(n)
    tt = np.arange(n) / SR
    fade = np.clip((tt - T0 + 0.5) / 2.5, 0, 1) * np.clip((DUR - tt) / 2.0, 0, 1)
    mix += 0.12 * d * fade
    # end: everything fades with the picture
    mix *= np.clip((DUR - tt) / 0.6, 0, 1)
    # gentle master saturation and normalisation
    mix = np.tanh(mix)
    mix /= max(1e-9, np.max(np.abs(mix))) / 0.95
    # stereo: clicks slightly left, drum centre, drone wide-ish (cheap trick: delay one channel of the drone)
    left = mix.copy(); right = mix.copy()
    dr = 0.12 * d * fade
    right += np.roll(dr, int(0.011 * SR)) * 0.5 - dr * 0.5
    stereo = np.stack([left, right], axis=1)
    stereo /= max(1e-9, np.max(np.abs(stereo))) / 0.95
    pcm = (stereo * 32767).astype('<i2')
    with wave.open(out, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    print(f'wrote {out}: {DUR:.3f} s, {n} samples')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'out/soundtrack.wav')

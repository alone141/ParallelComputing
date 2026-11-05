"""Synthesised drums, bass and guitar for the intro, in pure numpy.

Nothing here is sampled from the record.  Every hit is a few lines of
additive / noise synthesis; the aim is a clear, countable mock-up, not a
replica of Mario Duplantier's kit.
"""
import numpy as np

SR = 44100
rng = np.random.default_rng(7)


def env(n, attack=0.0005, decay=0.1, curve=6.0):
    t = np.arange(n) / SR
    a = np.minimum(1.0, t / max(attack, 1e-5))
    d = np.exp(-curve * t / decay)
    return a * d


def bandpass_noise(n, f_lo, f_hi, seed=None):
    """White noise, band-limited with an FFT mask (no scipy needed)."""
    r = rng if seed is None else np.random.default_rng(seed)
    x = r.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    mask = ((f >= f_lo) & (f <= f_hi)).astype(float)
    # soften the edges
    k = max(3, n // 400)
    mask = np.convolve(mask, np.ones(k) / k, mode="same")
    return np.fft.irfft(X * mask, n)


def sweep_sine(n, f0, f1, tau):
    """Sine whose frequency decays exponentially from f0 to f1."""
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / tau)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph)


def metallic(n, base, ratios, decay, amps=None):
    t = np.arange(n) / SR
    out = np.zeros(n)
    amps = amps or [1.0 / (i + 1) ** 0.6 for i in range(len(ratios))]
    for r, a in zip(ratios, amps):
        out += a * np.sin(2 * np.pi * base * r * t) * np.exp(-t / (decay * (0.6 + 0.4 / r)))
    return out / sum(amps)


# ---- the instruments ------------------------------------------------------

def side_stick():
    n = int(0.09 * SR)
    body = bandpass_noise(n, 1500, 5000) * env(n, 0.0003, 0.018)
    wood = np.sin(2 * np.pi * 1850 * np.arange(n) / SR) * env(n, 0.0002, 0.012)
    knock = np.sin(2 * np.pi * 620 * np.arange(n) / SR) * env(n, 0.0002, 0.03)
    return 0.9 * body + 0.5 * wood + 0.35 * knock


def sticks():
    n = int(0.05 * SR)
    return bandpass_noise(n, 2500, 9000) * env(n, 0.0002, 0.01) * 0.9


def kick():
    n = int(0.35 * SR)
    body = sweep_sine(n, 160, 48, 0.035) * env(n, 0.0005, 0.16, 5)
    click = bandpass_noise(n, 1000, 5000) * env(n, 0.0002, 0.008)
    return 1.1 * body + 0.35 * click


def floor_tom():
    n = int(0.5 * SR)
    body = sweep_sine(n, 130, 82, 0.05) * env(n, 0.001, 0.3, 5)
    skin = bandpass_noise(n, 300, 2500) * env(n, 0.0005, 0.03)
    return 0.9 * body + 0.3 * skin


def surdo():
    n = int(0.8 * SR)
    body = sweep_sine(n, 70, 52, 0.08) * env(n, 0.002, 0.5, 5)
    return 1.0 * body


def wood_block():
    n = int(0.08 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1180 * t) + 0.5 * np.sin(2 * np.pi * 2300 * t)) * env(n, 0.0002, 0.025)


def castanets():
    n = int(0.05 * SR)
    return bandpass_noise(n, 3000, 10000) * env(n, 0.0002, 0.008)


def snare():
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * 185 * t) + 0.6 * np.sin(2 * np.pi * 330 * t)) * env(n, 0.0005, 0.09, 5)
    wires = bandpass_noise(n, 1500, 10000) * env(n, 0.0003, 0.14, 5)
    return 0.6 * tone + 0.8 * wires


def ride_ghost():
    n = int(0.6 * SR)
    m = metallic(n, 860, [1, 1.47, 2.09, 2.76, 3.42, 4.5, 5.9], 0.25)
    tick = bandpass_noise(n, 4000, 12000) * env(n, 0.0002, 0.01)
    return 0.35 * m + 0.25 * tick


def ride_bell():
    n = int(1.0 * SR)
    m = metallic(n, 1720, [1, 1.33, 1.78, 2.41, 3.07, 3.9], 0.6, amps=[1, 0.8, 0.7, 0.5, 0.35, 0.25])
    tick = bandpass_noise(n, 3000, 10000) * env(n, 0.0002, 0.012)
    return 0.8 * m + 0.4 * tick


def crash():
    n = int(1.8 * SR)
    wash = bandpass_noise(n, 2500, 14000) * env(n, 0.002, 0.9, 4)
    m = metallic(n, 640, [1, 1.52, 2.3, 2.9, 3.8, 5.1, 6.6], 0.9)
    return 0.7 * wash + 0.4 * m


def china():
    n = int(1.2 * SR)
    wash = bandpass_noise(n, 1800, 12000) * env(n, 0.001, 0.45, 4)
    m = metallic(n, 420, [1, 1.7, 2.5, 3.3, 4.9], 0.5)
    return 0.8 * wash + 0.5 * m


def tribal_hit():
    """Bass drum + floor tom + surdo + wood block + castanets, stacked."""
    parts = [kick() * 0.9, floor_tom() * 0.8, surdo() * 0.9, wood_block() * 0.5, castanets() * 0.4]
    n = max(len(p) for p in parts)
    out = np.zeros(n)
    for p in parts:
        out[: len(p)] += p
    return out


def guitar_chug(f=65.41, length=0.11):
    """Palm-muted dead note: a damped, clipped saw with a bit of pick noise."""
    n = int(length * SR)
    t = np.arange(n) / SR
    saw = 2 * ((t * f) % 1.0) - 1
    saw2 = 2 * ((t * f * 1.004) % 1.0) - 1
    x = (saw + saw2) * env(n, 0.001, 0.045, 5)
    x = np.tanh(2.5 * x)
    pick = bandpass_noise(n, 2000, 6000) * env(n, 0.0002, 0.006)
    return 0.6 * x + 0.3 * pick


def bass_note(f=32.70, length=0.11):
    n = int(length * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) + 0.15 * np.sin(2 * np.pi * 3 * f * t)
    e = np.ones(n)
    rel = int(0.02 * SR)
    e[-rel:] = np.linspace(1, 0, rel)
    e[: int(0.004 * SR)] = np.linspace(0, 1, int(0.004 * SR))
    return np.tanh(1.5 * x) * e * 0.8


def choir_pad(length, f=130.81):
    """A soft held note (the choir under bars 1-11): detuned saws through a crude low-pass."""
    n = int(length * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for det, a in ((1.0, 1), (1.005, 0.7), (0.996, 0.7), (1.5, 0.5), (2.0, 0.35)):
        x += a * (2 * ((t * f * det) % 1.0) - 1)
    # crude one-pole low-pass via FFT
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1 / SR)
    X *= 1 / (1 + (fr / 500.0) ** 2)
    x = np.fft.irfft(X, n)
    return x / (np.max(np.abs(x)) + 1e-9)


# ---- mixing helpers -------------------------------------------------------

class Mix:
    def __init__(self, length):
        self.n = int(length * SR) + SR
        self.buf = np.zeros(self.n)

    def add(self, sample, at, gain=1.0):
        i = int(round(at * SR))
        if i < 0:
            sample = sample[-i:]
            i = 0
        j = min(self.n, i + len(sample))
        self.buf[i:j] += gain * sample[: j - i]

    def render(self, length, peak=0.89):
        x = self.buf[: int(length * SR)]
        x = x / (np.max(np.abs(x)) + 1e-9) * peak
        return x


def write_wav(path, x):
    import wave
    y = np.clip(x, -1, 1)
    pcm = (y * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

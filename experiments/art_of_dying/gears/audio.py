"""Synthesised drum track for the film (numpy only), 48 kHz stereo WAV.

Parts (bars 12-33 texture of the record): rim clicks on every sixteenth except the last of
each group; the stacked "tribal" hit on every beat with a stick grace note a 32nd before;
a held choir-like drone on C. One film loop = 180 sixteenths; tails wrap so the loop is seamless.
"""
import numpy as np
import wave
import sys
import rhythm as R

SR = 48000
SIXT = R.SIXT_S_FILM
LOOP_N = R.FRAMES * SR // R.FPS          # samples per loop (exact: 1157 * 800)
assert R.FRAMES * SR % R.FPS == 0
rng = np.random.default_rng(7)


def env(n, attack, decay, curve=1.0):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-5), 0, 1)
    d = np.exp(-t / decay) ** curve
    return a * d


def bandpass_noise(n, lo, hi, seed=None):
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    w = np.exp(-0.5 * ((np.log(np.maximum(f, 1)) - np.log(np.sqrt(lo * hi))) / (0.5 * np.log(hi / lo))) ** 2)
    return np.fft.irfft(X * w, n)


def sine(freq, n, phase=0.0):
    t = np.arange(n) / SR
    if np.isscalar(freq):
        return np.sin(2 * np.pi * freq * t + phase)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR + phase)


def rim_click(level=1.0):
    """Side stick: a short bright knock with a woody body."""
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    knock = bandpass_noise(n, 1800, 6500) * env(n, 0.0003, 0.006)
    body = sine(2340, n) * env(n, 0.0002, 0.012) * 0.6
    body2 = sine(1160, n) * env(n, 0.0002, 0.03) * 0.35
    low = sine(380, n) * env(n, 0.0005, 0.02) * 0.25
    x = knock * 1.1 + body + body2 + low
    return x / np.max(np.abs(x)) * 0.42 * level


def stick_grace():
    n = int(0.05 * SR)
    x = bandpass_noise(n, 2500, 9000) * env(n, 0.0002, 0.004) + sine(3100, n) * env(n, 0.0002, 0.006) * 0.4
    return x / np.max(np.abs(x)) * 0.22


def tribal_hit():
    """Bass drum + floor tom + surdo + wood block + castanets + sticks, all at once."""
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    kick_f = 48 + 90 * np.exp(-t / 0.045)
    kick = sine(kick_f, n) * env(n, 0.0005, 0.22)
    tom_f = 92 + 30 * np.exp(-t / 0.08)
    tom = sine(tom_f, n) * env(n, 0.001, 0.38) * 0.7
    surdo = sine(58 + 6 * np.exp(-t / 0.2), n) * env(n, 0.002, 0.55) * 0.75
    block = (sine(1180, n) + 0.5 * sine(1760, n)) * env(n, 0.0002, 0.018) * 0.5
    casta = bandpass_noise(n, 3000, 9000) * env(n, 0.0002, 0.007) * 0.8
    skin = bandpass_noise(n, 150, 1500) * env(n, 0.0005, 0.03) * 0.9
    x = kick * 1.2 + tom + surdo + block + casta + skin
    # soft saturation for weight
    x = np.tanh(x * 1.6) / np.tanh(1.6)
    return x / np.max(np.abs(x)) * 1.0


def drone(n):
    """Held choir-ish C: detuned partials with slow movement, low in the mix."""
    t = np.arange(n) / SR
    base = 65.41  # C2
    x = np.zeros(n)
    for mult, amp in [(1, 0.5), (2, 0.9), (3, 0.5), (4, 0.6), (5, 0.25), (6, 0.3), (8, 0.15)]:
        for det, ph in [(-0.0025, 0.0), (0.0, 1.3), (0.0025, 2.1)]:
            f = base * mult * (1 + det)
            vib = 1 + 0.0015 * np.sin(2 * np.pi * (0.13 + 0.05 * mult) * t + ph)
            x += amp * np.sin(2 * np.pi * np.cumsum(f * vib) / SR + ph)
    x *= 1 + 0.08 * np.sin(2 * np.pi * 0.21 * t)
    return x / np.max(np.abs(x))


def place(buf, sample, at_s, gain=1.0, pan=0.0):
    """Add a mono sample into the stereo buffer at time at_s (seconds), constant-power panned."""
    i = int(round(at_s * SR))
    seg = sample * gain
    l, r = np.sqrt((1 - pan) / 2) * 1.414, np.sqrt((1 + pan) / 2) * 1.414
    if i < 0:
        seg, i = seg[-i:], 0
    n = min(len(seg), buf.shape[1] - i)
    if n <= 0:
        return
    buf[0, i:i + n] += seg[:n] * l
    buf[1, i:i + n] += seg[:n] * r


def build():
    N = LOOP_N
    buf = np.zeros((2, 2 * N))
    rim = rim_click()
    hit = tribal_hit()
    grace = stick_grace()
    # two loops of events; we keep the second so the first's tails wrap in
    for T in range(0, 2 * R.LOOP_SIXT):
        at = T * SIXT
        if R.is_rim(T):
            # slight humanisation: alternate hands a hair louder/softer, +-2 ms
            lvl = 1.0 if (T % R.CYCLE) % 5 == 0 or (T % R.CYCLE) in (24,) else 0.85
            place(buf, rim, at + rng.uniform(-0.0015, 0.0015), lvl, pan=0.18)
        if R.is_beat(T):
            place(buf, grace, at - SIXT / 2, 1.0, pan=-0.1)
            place(buf, hit, at, 1.0, pan=-0.05)
    out = buf[:, N:2 * N].copy()
    # drone: crossfade its own head into its tail so the loop seam is silent
    d = drone(N)
    xf = int(0.5 * SR)
    ramp = np.linspace(0, 1, xf)
    d2 = d.copy()
    d2[-xf:] = d[-xf:] * (1 - ramp) + d[:xf] * ramp
    out[0] += d2 * 0.11
    out[1] += d2 * 0.11
    out = np.tanh(out * 0.9) / np.tanh(0.9)
    out *= 0.89 / np.max(np.abs(out))
    return out


def write_wav(path, x):
    pcm = (np.clip(x.T, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
    x = build()
    write_wav(path, x)
    print(path, x.shape[1] / SR, "s")

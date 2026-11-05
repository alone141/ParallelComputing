"""Borrowed unchanged from the owner's own Art of Dying study (branch ccr-03f861e3-540mn1,
art_of_dying/sound_dsp.py), with permission, for round 2 of the gears experiment.

Signal processing for the soundtrack, in numpy only (scipy is not available here).

Filters are written as smooth magnitude responses |H(f)| and realised as minimum-phase FIR filters (the
cepstral method). Minimum phase means causal: a filtered transient never leaks ahead of its hit, so every
onset stays where rhythm.py puts it.

Everything that must loop (the reverb, the master filter, the limiter, the peak meter) can run circularly:
whatever rings past the end of the film folds back onto its start.
"""

from __future__ import annotations

import wave

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SR = 48_000


def fft_size(n: int) -> int:
    """The smallest power of two >= n."""
    return 1 << max(1, (int(n) - 1).bit_length())


# ---- magnitude responses (f in Hz; arrays welcome) ------------------------------------------------
def lowpass(f, fc, order=2):
    """Butterworth-shaped low-pass, 6 dB/octave per order."""
    return 1.0 / np.sqrt(1.0 + (np.asarray(f, float) / fc) ** (2 * order))


def highpass(f, fc, order=2):
    """Butterworth-shaped high-pass, 6 dB/octave per order."""
    f = np.maximum(np.asarray(f, float), 1e-3)
    return 1.0 / np.sqrt(1.0 + (fc / f) ** (2 * order))


def eq(f, fc, gain_db, width=1.0):
    """Peaking EQ: gain_db at fc, a Gaussian bump in log frequency (width = standard deviation, octaves)."""
    octaves = np.log2(np.maximum(np.asarray(f, float), 1e-3) / fc)
    return 10.0 ** (gain_db * np.exp(-0.5 * (octaves / width) ** 2) / 20)


def resonance(f, fc, q):
    """Band-pass resonator: 1 at fc, bandwidth about fc / q."""
    f = np.maximum(np.asarray(f, float), 1e-3)
    return 1.0 / np.sqrt(1.0 + (q * (f / fc - fc / f)) ** 2)


# ---- minimum-phase filtering and convolution ------------------------------------------------------
def min_phase(mag):
    """Minimum-phase spectrum (rfft layout of an even FFT size) with the given magnitude."""
    nfft = 2 * (len(mag) - 1)
    cep = np.fft.irfft(np.log(np.maximum(mag, 1e-7)), nfft)    # real cepstrum
    cep[1 : nfft // 2] *= 2                                     # fold the anticausal half onto the causal half
    cep[nfft // 2 + 1 :] = 0
    return np.exp(np.fft.rfft(cep))


def filt(x, response):
    """Filter x (along axis 0) with the causal minimum-phase filter |H(f)| = response(f)."""
    n = len(x)
    nfft = fft_size(n + 16384)
    h = min_phase(response(np.fft.rfftfreq(nfft, 1 / SR)))
    if np.ndim(x) == 2:
        h = h[:, None]
    return np.fft.irfft(np.fft.rfft(x, nfft, axis=0) * h, nfft, axis=0)[:n]


def fir(response, taps=8192):
    """Minimum-phase FIR impulse response for |H(f)| = response(f), faded out over its last tenth."""
    nfft = 4 * taps
    h = np.fft.irfft(min_phase(response(np.fft.rfftfreq(nfft, 1 / SR))), nfft)[:taps]
    k = taps // 10
    h[-k:] *= 0.5 * (1 + np.cos(np.pi * np.arange(k) / k))
    return h


def convolve(x, h, period=None, tail=False):
    """FFT convolution along axis 0. x: (n,) or (n, c); h: (m,) or (m, c) (one response per channel).

    period=None  the first n samples (an excerpt: its pre-roll absorbs the edge)
    period=P     the signal loops every P samples: what rings past P folds back onto the start
    tail=True    the whole linear result, n + m - 1 samples
    """
    n, m = len(x), len(h)
    nfft = fft_size(n + m)
    X = np.fft.rfft(x, nfft, axis=0)
    H = np.fft.rfft(h, nfft, axis=0)
    if H.ndim < X.ndim:
        H = H[:, None]
    y = np.fft.irfft(X * H, nfft, axis=0)[: n + m - 1]
    if tail:
        return y
    return y[:n] if period is None else fold(y, period)


def fold(y, period):
    """Wrap a signal longer than `period` onto one period: the loop."""
    out = y[:period].copy()
    for s in range(period, len(y), period):
        seg = y[s : s + period]
        out[: len(seg)] += seg
    return out


def shape(x, fn, factor=4):
    """Apply the waveshaper fn at `factor` x the sample rate (band-limited by FFT), so distortion
    cannot alias. Meant for short hits that start and end near zero."""
    n = len(x)
    X = np.fft.rfft(x)
    up = np.zeros(factor * n // 2 + 1, complex)
    up[: len(X)] = X
    y = fn(np.fft.irfft(up, factor * n) * factor)
    return np.fft.irfft(np.fft.rfft(y)[: len(X)] / factor, n)


# ---- reverb -------------------------------------------------------------------------------------
def reverb_ir(seconds=2.4, seed=5):
    """Stereo impulse response of a large, warm room: a handful of early reflections, then a diffuse tail
    whose highs die away faster than its mids (mid RT60 about 1.9 s). The send EQ is built in (no lows
    to muddy the kit, no fizz). Unit energy per channel."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    bands = ((lambda f: lowpass(f, 400), 0.22),
             (lambda f: highpass(f, 400) * lowpass(f, 2500), 0.27),
             (lambda f: highpass(f, 2500) * lowpass(f, 6000), 0.17),
             (lambda f: highpass(f, 6000), 0.09))
    ir = np.zeros((n, 2))
    for c in range(2):                                   # two independent tails: a wide, decorrelated room
        noise = rng.standard_normal(n)
        for response, tau in bands:
            ir[:, c] += filt(noise, response) * np.exp(-t / tau)
    ir *= (np.clip((t - 0.014) / 0.035, 0, 1) ** 2)[:, None]    # 14 ms pre-delay, then the tail builds up
    for c in range(2):                                   # early reflections: walls, floor, ceiling
        for ms in np.sort(rng.uniform(7, 40, 8)):
            ir[int(ms * SR / 1000), c] += rng.choice((-1.0, 1.0)) * rng.uniform(8, 16) * (12 / ms) ** 0.5
    ir = filt(ir, lambda f: highpass(f, 220) * lowpass(f, 8000))
    return ir / np.sqrt((ir ** 2).sum(axis=0))


# ---- metering -----------------------------------------------------------------------------------
# ITU-R BS.1770-4 K-weighting at 48 kHz: a high shelf (the head) and the RLB high-pass.
_K_SHELF = ((1.53512485958697, -2.69169618940638, 1.19839281085285), (1.0, -1.69065929318241, 0.73248077421585))
_K_HIGHPASS = ((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621))


def _biquad(coeffs, nfft):
    (b0, b1, b2), (a0, a1, a2) = coeffs
    z = np.exp(-2j * np.pi * np.fft.rfftfreq(nfft))
    return (b0 + b1 * z + b2 * z * z) / (a0 + a1 * z + a2 * z * z)


def loudness(x):
    """Integrated loudness in LUFS (BS.1770-4: K-weighting, 400 ms blocks with 75% overlap, absolute and
    relative gates). The K filters are applied in the frequency domain with their exact responses."""
    x = x[:, None] if x.ndim == 1 else x
    n = len(x)
    block, hop = int(0.4 * SR), int(0.1 * SR)
    if n < block:
        return float("-inf")
    nfft = fft_size(n + SR)
    k = _biquad(_K_SHELF, nfft) * _biquad(_K_HIGHPASS, nfft)
    y = np.fft.irfft(np.fft.rfft(x, nfft, axis=0) * k[:, None], nfft, axis=0)[:n]
    c = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y * y, axis=0)])
    starts = np.arange(0, n - block + 1, hop)
    z = ((c[starts + block] - c[starts]) / block).sum(axis=1)
    lk = -0.691 + 10 * np.log10(np.maximum(z, 1e-20))
    gate = lk > -70
    if not gate.any():
        return float("-inf")
    gate &= lk > -0.691 + 10 * np.log10(z[gate].mean()) - 10
    return float(-0.691 + 10 * np.log10(z[gate].mean()))


def true_peak_env(x, periodic=False, factor=4, chunk=1 << 16, pad=1024):
    """Per-sample true-peak estimate: the largest |x| over channels of a `factor` x band-limited
    interpolation (FFT, in overlapping chunks). periodic=True treats x as one loop."""
    x = x[:, None] if x.ndim == 1 else x
    n = len(x)
    edge = np.zeros((pad, x.shape[1]))
    xp = np.concatenate([x[-pad:] if periodic else edge, x, x[:pad] if periodic else edge])
    out = np.empty(n)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        seg = xp[s : e + 2 * pad]
        spec = np.fft.rfft(seg, axis=0)
        up = np.zeros((factor * len(seg) // 2 + 1, seg.shape[1]), complex)
        up[: len(spec)] = spec
        y = np.abs(np.fft.irfft(up, factor * len(seg), axis=0)).max(axis=1) * factor
        out[s:e] = y[pad * factor : (pad + e - s) * factor].reshape(e - s, factor).max(axis=1)
    return out


# ---- limiter ------------------------------------------------------------------------------------
def limit(x, ceiling_db, peak_env, periodic=False, lookahead=0.002, release_db_per_s=30.0):
    """Offline look-ahead brick-wall limiter, driven by the true-peak envelope of x.

    Gain reduction (dB) is held over the look-ahead window, released linearly in dB, then averaged over the
    look-ahead window, so it is already fully down when the peak arrives (never a hard clip). The release is
    the max-plus recursion r[i] = max(need[i], r[i-1] - c), solved in closed form with a running maximum.
    Returns (limited x, gain reduction in dB per sample).
    """
    n = len(x)
    L = max(1, int(round(lookahead * SR)))
    need = np.maximum(0.0, 20 * np.log10(np.maximum(peak_env, 1e-12)) - ceiling_db)
    pad = SR                                                  # one second of history for the release
    if periodic:
        ext = np.concatenate([need[-pad:], need, need[:L]])
    else:
        ext = np.concatenate([np.zeros(pad), need, np.zeros(L)])
    hold = sliding_window_view(ext, L).max(axis=1)            # hold[j] = max(ext[j : j + L])
    c = release_db_per_s / SR
    j = np.arange(len(hold))
    rel = np.maximum.accumulate(hold + c * j) - c * j
    cs = np.concatenate([[0.0], np.cumsum(rel)])
    smooth = (cs[L:] - cs[:-L]) / L                           # smooth[m] = mean(rel[m : m + L])
    gr = smooth[pad - L + 1 : pad - L + 1 + n]                # mean(rel[i - L + 1 : i + 1]) at sample i
    return x * (10 ** (-gr / 20))[:, None], gr


# ---- files --------------------------------------------------------------------------------------
def write_wav(path, x):
    """24-bit PCM WAV at SR; x is (n, channels) in [-1, 1]."""
    pcm = np.round(np.clip(x, -1.0, 1.0 - 2.0 ** -23) * 2 ** 23).astype("<i4")
    raw = np.ascontiguousarray(pcm).reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(raw)


def read_wav(path):
    """(sample rate, float array (n, channels)) from a 16- or 24-bit PCM WAV."""
    with wave.open(path, "rb") as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v) / 2.0 ** 23
    elif width == 2:
        x = np.frombuffer(raw, "<i2") / 32768.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    return sr, x.reshape(-1, ch)

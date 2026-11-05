#!/usr/bin/env python3
"""Recover the rhythm of build/mix.wav from the SOUND ALONE.

Nothing here reads the score, alignment.json or synth.py. Pipeline, all numpy, causal:

  1. STFT (2048/256) -> log-magnitude spectral flux (all bins) -> onset picking.
  2. Each onset is typed by its linear-magnitude flux below 200 Hz (L) and above 1.5 kHz (H):
         thumpness = L / (L + 2H)   -> THUMP if >= 0.4 else CLICK
     (a rim click is ~0.1, a stacked drum hit ~0.9).  Two impulse trains result.
  3. Tatum (the sixteenth) = first strong peak of the log-flux autocorrelation between
     60 and 200 ms, refined from the peaks near 8x and 24x that lag.
  4. The last 20 s of each train is quantised onto the tatum grid (phase chosen to catch
     the most onset energy), giving a sequence g[m], one value per sixteenth.
  5. Autocorrelation of g (mean-removed, unbiased) at integer lags 1..M/2.
  6. Per train, the fundamental period = argmax of a comb score (ACF[P] averaged with
     ACF[2P] when observable), reduced to the smallest sub-multiple nearly as strong.
  7. Cycle = the longer of the two fundamentals, beat = the shorter.

Everything is saved per video frame so render.py can show the estimate evolving.
"""
import numpy as np, wave, sys, os

SR = 44100
NFFT, HOP = 2048, 256                      # hop = 5.8 ms
FPS = 30
LOW, HIGH = (0, 200), (1500, 20000)
KMAX = 200
MIN_TATUM, MAX_TATUM = 0.060, 0.200        # s
WINDOW_S = 20.0                            # the detector remembers the last 20 s
THUMP_T = 0.4
THUMP_C = 1.0                              # weight of the high envelope in the thumpness ratio

def load(path):
    w = wave.open(path)
    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels()) / 32768.0
    return x.mean(1)

def stft_mag(x):
    pad = np.concatenate([np.zeros(NFFT // 2), x, np.zeros(NFFT)])
    frames = np.lib.stride_tricks.sliding_window_view(pad, NFFT)[::HOP]
    return np.abs(np.fft.rfft(frames * np.hanning(NFFT), axis=1))

def smooth3(f):
    return np.convolve(f, [0.25, 0.5, 0.25], mode="same")

def band_env(x, lo, hi, hop_s):
    """RMS envelope (per hop) of x band-limited to [lo, hi] Hz, via FFT brick-wall filter."""
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f >= hi)] = 0
    y = np.fft.irfft(X, len(x))
    n = len(y) // HOP
    e = np.sqrt((y[: n * HOP].reshape(n, HOP) ** 2).mean(1))
    return e

def fluxes(mag, x):
    """total: log-magnitude spectral flux (for onsets and the tatum).
    L, H: positive jumps of the band-limited RMS envelopes (< 200 Hz, > 1.5 kHz), for typing
    onsets. Envelope jumps, unlike spectral flux, are blind to decaying tails and pitch glides."""
    Lg = np.log1p(50.0 * mag)
    total = np.concatenate([[0.0], np.maximum(0.0, Lg[1:] - Lg[:-1]).mean(1)])
    total = smooth3(total); total /= np.percentile(total, 99.5) + 1e-9
    def jumps(lo, hi):
        e = band_env(x, lo, hi, HOP / SR)
        w = 10                                              # 58 ms: a jump must beat the recent maximum
        recent = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.zeros(w), e[:-1]]), w).max(1)
        d = np.maximum(0.0, e - recent)
        d = np.concatenate([d, np.zeros(max(0, len(total) - len(d)))])
        return d[: len(total)]
    return total, jumps(*LOW), jumps(*HIGH)

def pick_onsets(f, hop_s):
    """Causal peak picking: local max in +-25 ms, above a running mean + delta."""
    w = max(1, int(0.025 / hop_s)); m = int(0.4 / hop_s)
    cs = np.concatenate([[0.0], np.cumsum(f)])
    out, last = [], -10
    for i in range(w, len(f) - w):
        if f[i] < 0.10 or f[i] < f[i - w:i + w + 1].max():
            continue
        mean = (cs[i] - cs[max(0, i - m)]) / min(i, m)
        if f[i] < mean + 0.06 or (i - last) * hop_s < 0.04:
            continue
        out.append(i); last = i
    return np.array(out, int)

def acf(x, max_lag):
    x = x - x.mean()
    n = len(x)
    if n < 8 or np.abs(x).sum() < 1e-9:
        return np.zeros(max_lag + 1)
    N = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, N)
    r = np.fft.irfft(X * np.conj(X), N)[: max_lag + 1]
    r = r / np.maximum(n - np.arange(max_lag + 1), 1)
    return r / (r[0] + 1e-12)

def refine_peak(r, i):
    if 0 < i < len(r) - 1:
        a, b, c = r[i - 1], r[i], r[i + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            return i + 0.5 * (a - c) / den
    return float(i)

def estimate_tatum(rt, hop_s, max_lag):
    lo, hi = int(MIN_TATUM / hop_s), int(MAX_TATUM / hop_s)
    if hi >= len(rt):
        return 0.0
    i = lo + int(np.argmax(rt[lo:hi]))
    if rt[i] < 0.15:
        return 0.0
    th = refine_peak(rt, i)
    for mult in (8, 24):                      # sharpen from longer lags
        c = th * mult; w = int(0.3 * th)
        a, b = int(c - w), int(c + w) + 1
        if b >= max_lag or a < 1:
            break
        j = a + int(np.argmax(rt[a:b]))
        if rt[j] < 0.1:
            break
        th = refine_peak(rt, j) / mult
    return th * hop_s

def quantise(train, n0, n, th):
    """Put train[n0:n] (hop grid) onto a tatum grid of spacing th hops; returns (g, phase)."""
    seg = train[n0:n]
    M = int(len(seg) / th) - 1
    if M < 4:
        return None, 0.0
    best, bestg = -1.0, None
    half = max(1, int(0.25 * th))
    for ph in np.linspace(0, th, 8, endpoint=False):
        centres = (ph + np.arange(M) * th).astype(int)
        idx = centres[:, None] + np.arange(-half, half + 1)[None, :]
        idx = np.clip(idx, 0, len(seg) - 1)
        g = seg[idx].max(1)
        e = g.sum()
        if e > best:
            best, bestg, bestph = e, g, ph
    return bestg, bestph

def fundamental(acf_t, kmin=2, ratio=0.9, floor=0.3):
    """Comb-scored period in tatums: (period, strength) or (0, 0)."""
    n = len(acf_t)
    if n <= kmin:
        return 0, 0.0
    score = np.full(n, np.nan)
    for k in range(kmin, n):
        score[k] = 0.5 * (acf_t[k] + acf_t[2 * k]) if 2 * k < n else acf_t[k]
    P = kmin + int(np.nanargmax(score[kmin:]))
    s = score[P]
    if s < floor:
        return 0, float(s)
    for k in range(P // kmin, 1, -1):
        if P % k == 0 and score[P // k] >= ratio * s:
            return P // k, float(score[P // k])
    return P, float(s)

def main(mix_path, out_path):
    x = load(mix_path); dur = len(x) / SR; hop_s = HOP / SR
    mag = stft_mag(x)
    total, L, H = fluxes(mag, x)
    onsets = pick_onsets(total, hop_s)
    thumpness = np.zeros(len(onsets)); strength = np.zeros(len(onsets))
    for j, i in enumerate(onsets):
        l = L[max(0, i - 3):i + 4].max(); h = H[max(0, i - 3):i + 4].max()
        thumpness[j] = l / (l + THUMP_C * h + 1e-12); strength[j] = total[i]
    names = ["thump", "click"]
    trains = np.zeros((2, len(total)))
    for j, i in enumerate(onsets):
        trains[0 if thumpness[j] >= THUMP_T else 1, i] = strength[j]
    n_frames = int(np.ceil(dur * FPS))
    acf_tat = np.full((n_frames, 2, KMAX + 1), np.nan)
    acf_hop = np.zeros((n_frames, 400))                       # log-flux ACF, first 400 hops (2.3 s): hook graphic
    grid = np.full((n_frames, 2, 220), np.nan)                 # quantised trains (last 20 s), for the fold plot
    grid_t0 = np.zeros(n_frames)                               # time of grid[0], seconds
    tatum = np.zeros(n_frames)
    fund = np.zeros((n_frames, 2), int); fstr = np.zeros((n_frames, 2))
    active = np.zeros((n_frames, 2), bool)
    cycle = np.zeros(n_frames, int); beat = np.zeros(n_frames, int)
    for k in range(n_frames):
        t = k / FPS; n = int(t / hop_s)
        if n < int(0.5 / hop_s):
            continue
        n0 = max(0, n - int(WINDOW_S / hop_s))
        max_lag = (n - n0) // 2
        rt = acf(total[n0:n], max_lag)
        acf_hop[k, :min(400, len(rt))] = rt[:400]
        tatum[k] = estimate_tatum(rt, hop_s, max_lag)
        if tatum[k] <= 0:
            continue
        th = tatum[k] / hop_s
        for j in range(2):
            g, ph = quantise(trains[j], n0, n, th)
            if g is None:
                continue
            m = min(len(g), 220)
            grid[k, j, :m] = g[-m:]
            grid_t0[k] = (n0 + ph + (len(g) - m) * th) * hop_s
            active[k, j] = (g > 0).sum() >= 6                  # at least six onsets of this type in the window
            r = acf(g, len(g) // 2)
            acf_tat[k, j, :len(r)] = r
            if active[k, j]:
                fund[k, j], fstr[k, j] = fundamental(r)
        fs = [f for f in fund[k] if f > 0]
        if fs:
            cycle[k] = max(fs); beat[k] = min(fs) if len(fs) > 1 and min(fs) != max(fs) else 0
    np.savez_compressed(out_path, hop_s=hop_s, fps=FPS, dur=dur, names=np.array(names),
                        total_flux=total, L=L, H=H, onsets=onsets, thumpness=thumpness, strength=strength,
                        trains=trains, acf_tat=acf_tat, acf_hop=acf_hop, grid=grid, grid_t0=grid_t0,
                        tatum=tatum, fund=fund, fstr=fstr, active=active, cycle=cycle, beat=beat)
    lines = []
    for k in range(0, n_frames, FPS):
        lines.append(f"t={k/FPS:5.1f}s tatum={tatum[k]*1000:6.1f}ms  thump{'*' if active[k,0] else ' '}: P={fund[k,0]:3d} ({fstr[k,0]:.2f})"
                     f"  click{'*' if active[k,1] else ' '}: P={fund[k,1]:3d} ({fstr[k,1]:.2f})  -> cycle={cycle[k]:3d} beat={beat[k]:2d}")
    log = "\n".join(lines)
    open(os.path.splitext(out_path)[0] + "_log.txt", "w").write(log + "\n")
    print(log)
    print(f"onsets: {len(onsets)}  thump {(thumpness >= THUMP_T).sum()}  click {(thumpness < THUMP_T).sum()}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/mix.wav",
         sys.argv[2] if len(sys.argv) > 2 else "build/analysis.npz")

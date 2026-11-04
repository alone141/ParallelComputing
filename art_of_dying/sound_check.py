"""Checks for a rendered soundtrack WAV (analysis only).

    python3 sound_check.py out/soundtrack.wav                     # the film
    python3 sound_check.py out/excerpt.wav --start 17             # an excerpt (give its start time in the film)
    python3 sound_check.py out/soundtrack.wav --focus             # + every layer's loudness in its focus lap
    python3 sound_check.py out/soundtrack.wav --excerpt out/excerpt.wav --excerpt-start 17   # + excerpt == film?

Reports: length, NaN/Inf, DC, sample and true peak, clipping, loudness (own BS.1770 meter; ffmpeg's
ebur128 is the reference), the loop seam (film only), and onsets found in the audio against the scheduled
hit times. --focus renders the six layers again as dry stems (a few seconds) and measures them lap by lap.
"""

from __future__ import annotations

import argparse

import numpy as np

from rhythm import LAP, LAYERS, N_LAPS, entry_lap, events
from sound_dsp import SR, filt, highpass, loudness, read_wav, true_peak_env
from soundtrack import GRACE, N


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def basic(x, film):
    print(f"length {len(x)} samples = {len(x) / SR:.4f} s"
          + (f" (expected {N}: {'ok' if len(x) == N else 'MISMATCH'})" if film else ""))
    print(f"finite {bool(np.isfinite(x).all())} · DC L {x[:, 0].mean():+.1e} R {x[:, 1].mean():+.1e}"
          f" · sample peak {db(np.abs(x).max()):.2f} dBFS · samples at |x| >= 0.999: {int(np.sum(np.abs(x) >= 0.999))}")
    print(f"true peak {db(true_peak_env(x, periodic=film).max()):.2f} dBTP (4x) · loudness {loudness(x):.2f} LUFS")


def seam(x):
    """The loop point (last sample -> first sample) against ordinary sample steps, film-wide and nearby."""
    steps = np.abs(np.diff(x, axis=0))
    jump = np.abs(x[0] - x[-1])
    w = int(0.05 * SR)
    near = np.abs(np.diff(np.concatenate([x[-w:], x[:w]]), axis=0))      # 50 ms either side of the seam
    rank = max(np.mean(steps[:, c] < jump[c]) for c in range(2)) * 100
    rank_near = max(np.mean(near[:, c] < jump[c]) for c in range(2)) * 100
    rms = lambda y: np.sqrt(np.mean(y ** 2))
    print(f"loop seam: step {jump.max():.4f}, larger than {rank:.1f}% of the film's sample steps and "
          f"{rank_near:.1f}% of those within 50 ms · RMS of the last/first 50 ms {db(rms(x[-w:])):.1f}/{db(rms(x[:w])):.1f} dBFS"
          " (the first holds frame 0's rim click)")


def detect_onsets(mono, rise_db=8.0, floor_db=-35.0, gap=0.02):
    """Onset times (s), found blind in the audio. The 1.5 kHz+ band's fast envelope (0.67 ms trailing RMS)
    is compared with its steady level just before (RMS over the 9 ms ending 1 ms earlier). A candidate
    rises by rise_db over that level and reaches within floor_db of the file's loud transients; its onset is
    the first sample 6 dB above the level before. Of candidates closer than `gap`, the strongest is kept."""
    hf = filt(mono, lambda f: highpass(f, 1500, 4))
    c = np.concatenate([[0.0], np.cumsum(hf ** 2)])
    i = np.arange(len(hf))
    rms = lambda end, w: np.sqrt((c[np.clip(end, 0, len(hf))] - c[np.clip(end - w, 0, len(hf))]) / w) + 1e-9
    fast = 20 * np.log10(rms(i + 1, 32))
    before = 20 * np.log10(rms(i + 1 - int(0.001 * SR), int(0.009 * SR)))
    rise = fast - before
    loud = np.percentile(fast, 99.9)
    edges = np.diff(np.concatenate([[0], ((rise > rise_db) & (fast > loud + floor_db)).astype(np.int8), [0]]))
    cands = []
    for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        a0 = max(a - int(0.002 * SR), 0)
        o = a0 + int(np.argmax(fast[a0:b] > before[a] + 6.0))
        cands.append((rise[a:b].max(), o))
    kept = []
    for strength, o in sorted(cands, reverse=True):
        if all(abs(o - k) >= gap * SR for k in kept):
            kept.append(o)
    return np.array(sorted(kept)) / SR


def onsets(x, start):
    """Detected onsets against every scheduled hit (and pulse grace note) inside the file."""
    dur = len(x) / SR
    times = {}
    for h in events():
        times.setdefault(round(h.time, 6), "hit")
        if h.kind == "pulse":
            times.setdefault(round(h.time - GRACE, 6), "grace")
    sched = sorted((t - start, kind) for t, kind in times.items() if 0.02 <= t - start <= dur - 0.02)
    found = detect_onsets(x.mean(axis=1))
    errors, missed = {"hit": [], "grace": []}, {"hit": [], "grace": []}
    for t, kind in sched:
        j = np.searchsorted(found, t)
        near = found[max(j - 1, 0) : j + 1]
        e = near[np.argmin(np.abs(near - t))] - t if len(near) else np.inf
        if abs(e) <= 0.02:
            errors[kind].append(e)
        else:
            missed[kind].append(round(t + start, 3))
    sched_t = np.array([t for t, _ in sched])
    extra = int(sum(np.min(np.abs(sched_t - f)) > 0.02 for f in found)) if len(sched_t) else len(found)
    for kind in ("hit", "grace"):
        e = np.abs(np.array(errors[kind])) * 1000
        total = len(e) + len(missed[kind])
        if not total:
            continue
        print(f"onsets, {kind} times: {len(e)}/{total} found within 20 ms"
              + (f" · |error| median {np.median(e):.2f} ms, 99th pct {np.percentile(e, 99):.2f} ms, "
                 f"max {e.max():.2f} ms, within 1 ms {np.mean(e <= 1) * 100:.1f}%" if len(e) else "")
              + (f" · missed at {missed[kind][:8]}{' ...' if len(missed[kind]) > 8 else ''} s" if missed[kind] else ""))
    print(f"onsets detected with no scheduled hit within 20 ms: {extra}")


def focus():
    """Each layer alone (dry) against the rest of the mix, in its focus lap and in the lap after."""
    import soundtrack as st
    kit, choir = st.Kit(), st.Choir(N)
    levels = st.calibrate(kit, choir)
    hits = events()
    stems = {}
    for layer in LAYERS:
        bus = st.Bus(0, N, period=N)
        for h in hits:
            if h.key == layer.key:
                st.play(bus, kit, levels, h)
        stems[layer.key] = bus.dry
    lap = lambda k: slice(round(k * LAP * SR), round((k + 1) * LAP * SR))
    print("focus: layer loudness vs the rest of the mix (LUFS, dry stems), focus lap -> the lap after")
    for layer in LAYERS:
        cells = []
        for k in (entry_lap(layer.key), entry_lap(layer.key) + 1):
            me = loudness(stems[layer.key][lap(k)])
            rest = loudness(sum(stems[o.key][lap(k)] for o in LAYERS if o.key != layer.key))
            cells.append(f"{me:6.1f} vs {rest:6.1f} ({me - rest:+5.1f} LU)" if np.isfinite(rest) else f"{me:6.1f} alone")
        print(f"  {layer.name:<13} {cells[0]}  ->  {cells[1]}")
    k = slice(round((N_LAPS - 1) * LAP * SR), round(((N_LAPS - 1) * LAP + 3.5) * SR))   # full mix, before the peel
    print("balance in the full-mix lap (LUFS): " + "  ".join(f"{o.key} {loudness(stems[o.key][k]):.1f}" for o in LAYERS)
          + f"  all {loudness(sum(stems[o.key][k] for o in LAYERS)):.1f}")
    print("spectrum in the full-mix lap (share of each stem's energy): below 100 Hz / 100 Hz-5 kHz / above 5 kHz")
    for o in LAYERS:
        p = np.abs(np.fft.rfft(stems[o.key][k].sum(axis=1))) ** 2
        f = np.fft.rfftfreq(k.stop - k.start, 1 / SR)
        share = [p[sel].sum() / p.sum() * 100 for sel in (f < 100, (f >= 100) & (f < 5000), f >= 5000)]
        print(f"  {o.name:<13} " + " / ".join(f"{v:5.1f}%" for v in share))


def excerpt_match(film, path, start):
    _, e = read_wav(path)
    s = round(start * SR)
    f = film[s : s + len(e)]
    g = np.sum(e * f) / np.sum(f * f)
    resid = 10 * np.log10(np.sum((e - g * f) ** 2) / np.sum(e ** 2))
    print(f"excerpt {path} vs the film from {start} s: level {db(g):+.2f} dB, residual after level match {resid:.1f} dB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wav")
    ap.add_argument("--start", type=float, help="the file is an excerpt starting here in the film (s)")
    ap.add_argument("--focus", action="store_true")
    ap.add_argument("--excerpt")
    ap.add_argument("--excerpt-start", type=float, default=0.0)
    a = ap.parse_args()
    sr, x = read_wav(a.wav)
    assert sr == SR, sr
    film = a.start is None
    basic(x, film)
    if film:
        seam(x)
    onsets(x, a.start or 0.0)
    if a.excerpt:
        excerpt_match(x, a.excerpt, a.excerpt_start)
    if a.focus:
        focus()

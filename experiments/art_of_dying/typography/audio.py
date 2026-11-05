"""Synthesised soundtrack: numpy drums and two piper counting voices.

Nothing here is sampled from the record.  Per lap (see timeline.py):
  rim   : rim clicks on the 34 hits + choir pad + group voice on the rests
  beat  : + tribal stacked hit (with a 32nd-note stick grace) on every beat
          + beat voice counting 1 2 3 4
  band  : kick / snare / ride / bass on C1 / palm-muted dead-note guitars,
          ride bell + beat voice on the beats
Output: a 48 kHz stereo WAV, later loudness-normalised by ffmpeg.
"""
import os
import sys
import wave

import numpy as np

import rhythm as R
import timeline as T

SR = 48000
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(45)


def env(n, attack, decay, curve=1.0):
    """Attack-decay envelope of n samples (seconds)."""
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    d = np.exp(-np.maximum(t - attack, 0) / decay) ** curve
    return a * d


def sine_sweep(dur, f0, f1, tau):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / tau)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph)


def lowpass(x, cutoff, taps=101):
    m = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff / SR * m) * np.hamming(taps)
    h /= h.sum()
    return np.convolve(x, h, mode="same")


def highpass(x, cutoff, taps=101):
    return x - lowpass(x, cutoff, taps)


def bandpass(x, lo, hi):
    return highpass(lowpass(x, hi), lo)


# ---- instruments (each returns a mono float array) ---------------------------
def rim_click():
    n = int(0.045 * SR)
    noise = rng.standard_normal(n)
    body = bandpass(noise, 2200, 7000) * env(n, 0.0005, 0.006)
    tick = np.sin(2 * np.pi * 3100 * np.arange(n) / SR) * env(n, 0.0003, 0.012)
    wood = np.sin(2 * np.pi * 1450 * np.arange(n) / SR) * env(n, 0.0003, 0.02)
    return 0.9 * body + 0.5 * tick + 0.35 * wood


def stick_grace():
    n = int(0.03 * SR)
    noise = rng.standard_normal(n)
    return 0.6 * bandpass(noise, 3000, 9000) * env(n, 0.0003, 0.004)


def tribal_hit():
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    bass = sine_sweep(0.6, 110, 48, 0.04) * env(n, 0.001, 0.18)
    tom = sine_sweep(0.6, 170, 95, 0.06) * env(n, 0.001, 0.22)
    surdo = np.sin(2 * np.pi * 62 * t) * env(n, 0.002, 0.35)
    block = np.sin(2 * np.pi * 980 * t) * env(n, 0.0003, 0.018)
    cast = bandpass(rng.standard_normal(n), 2500, 8000) * env(n, 0.0005, 0.02)
    return 1.0 * bass + 0.5 * tom + 0.5 * surdo + 0.4 * block + 0.45 * cast


def kick():
    n = int(0.35 * SR)
    body = sine_sweep(0.35, 140, 52, 0.03) * env(n, 0.0008, 0.12)
    click = bandpass(rng.standard_normal(n), 1500, 5000) * env(n, 0.0003, 0.004)
    return 1.1 * body + 0.35 * click


def snare():
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    wires = bandpass(rng.standard_normal(n), 900, 9000) * env(n, 0.0005, 0.07)
    shell = (np.sin(2 * np.pi * 195 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)) * env(n, 0.0005, 0.05)
    return 0.9 * wires + 0.6 * shell


def ride(bell=False):
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    fs = [3150, 4420, 5870, 7300] if bell else [2400, 3700, 5200, 6800, 8900]
    tone = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) for f in fs) / len(fs)
    wash = bandpass(rng.standard_normal(n), 4000, 12000)
    e = env(n, 0.0005, 0.25 if bell else 0.12)
    return (0.8 * tone + 0.3 * wash) * e * (1.0 if bell else 0.45)


def bass_note(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 32.70
    x = (np.sin(2 * np.pi * f * t) + 0.55 * np.sin(2 * np.pi * 2 * f * t)
         + 0.3 * np.sin(2 * np.pi * 3 * f * t) + 0.15 * np.sin(2 * np.pi * 4 * f * t))
    x = np.tanh(2.2 * x)
    e = env(n, 0.004, max(dur, 0.1) * 1.5)
    e[-int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))
    return x * e


def dead_note():
    n = int(0.08 * SR)
    t = np.arange(n) / SR
    thump = np.sin(2 * np.pi * 65.4 * t) * env(n, 0.0005, 0.02)
    scratch = lowpass(rng.standard_normal(n), 2600) * env(n, 0.0005, 0.012)
    return 0.6 * np.tanh(3 * (thump + 0.7 * scratch))


def choir(dur, fade_in=1.5, fade_out=1.0):
    """A held, slightly moving C minor-ish pad: the 'held choir note' of bars 1-33."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f, a in [(65.4, 0.5), (130.8, 1.0), (155.6, 0.5), (196.0, 0.7), (261.6, 0.4), (392.0, 0.2)]:
        for det in (-0.7, 0.0, 0.8):
            ff = f * (1 + det / 300)
            vib = 1 + 0.0025 * np.sin(2 * np.pi * (0.15 + 0.07 * det) * t + det)
            out += a * np.sin(2 * np.pi * ff * vib * t + rng.uniform(0, 6.28))
    out = lowpass(out, 1800)
    e = np.ones(n)
    fi = int(fade_in * SR)
    fo = int(fade_out * SR)
    e[:fi] = np.linspace(0, 1, fi) ** 2
    e[-fo:] *= np.linspace(1, 0, fo)
    return out / np.abs(out).max() * e


# ---- voices -------------------------------------------------------------------
def load_word(name):
    path = os.path.join(HERE, "words", name + ".wav")
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    # trim silence
    thr = 0.02 * np.abs(x).max()
    idx = np.where(np.abs(x) > thr)[0]
    x = x[max(idx[0] - int(0.005 * sr), 0): idx[-1] + int(0.02 * sr)]
    return x, sr


def resample(x, sr_in, sr_out):
    n_out = int(len(x) * sr_out / sr_in)
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)


def word(name, pitch=1.0, max_dur=None):
    """A word at 48 kHz; pitch > 1 raises it (simple resampling, also shorter)."""
    x, sr = load_word(name)
    y = resample(x, sr * pitch, SR)
    if max_dur and len(y) > int(max_dur * SR):
        # time-compress what still does not fit by crude resampling (rare)
        y = resample(y, SR * len(y) / (max_dur * SR), SR)
    fo = int(0.012 * SR)
    y[-fo:] *= np.linspace(1, 0, fo)
    return y


# ---- the arrangement ---------------------------------------------------------
def place(buf, x, t_sec, gain=1.0):
    i = int(round(t_sec * SR))
    j = min(i + len(x), len(buf))
    if i < len(buf):
        buf[i:j] += gain * x[: j - i]


def build():
    total = T.total_seconds() + 0.5
    n = int(total * SR)
    drums = np.zeros(n)
    low = np.zeros(n)
    voice = np.zeros(n)
    pad = np.zeros(n)

    RIM, GRACE, TRIBAL = rim_click(), stick_grace(), tribal_hit()
    KICK_S, SNARE_S, RIDE_S, BELL_S, DEAD = kick(), snare(), ride(), ride(bell=True), dead_note()
    group_words = {5: word("five", pitch=0.78), 3: word("three", pitch=0.78, max_dur=0.3)}
    beat_words = {k: word(w, pitch=1.0) for k, w in enumerate(["one", "two", "three", "four"], 1)}

    # pads: laps 0-4 continuous, lap 6 again
    pad_spans = []
    cur = None
    for lap, mode in enumerate(T.LAPS):
        if mode in ("rim", "beat"):
            if cur is None:
                cur = [lap, lap]
            else:
                cur[1] = lap
        else:
            if cur:
                pad_spans.append(cur)
                cur = None
    if cur:
        pad_spans.append(cur)
    for a, b in pad_spans:
        dur = (b - a + 1) * R.CYCLE * R.SIX
        place(pad, choir(dur, fade_in=0.9 if a == 0 else 0.6, fade_out=0.9), a * R.CYCLE * R.SIX)

    for lap, mode in enumerate(T.LAPS):
        base = lap * R.CYCLE
        for t in range(R.CYCLE):
            g = base + t
            sec = g * R.SIX
            beat = T.beat(g)
            if mode in ("rim", "beat"):
                if t in R.RIM:
                    # alternate accents slightly, like two sticks
                    place(drums, RIM, sec, 1.15 if t % 2 else 1.3)
                if t in R.RESTS:
                    size = R.slot_of(t)[2]
                    place(voice, group_words[size], sec, 1.0)
            if mode == "beat" and beat:
                place(drums, GRACE, sec - R.SIX / 2, 0.5)
                place(low, TRIBAL, sec, 1.0)
                place(voice, beat_words[R.beat_count(g, T.FIRST_BEAT)], sec, 0.9)
            if mode == "band":
                if t in R.KICK:
                    place(low, KICK_S, sec, 1.0 if t % 2 == 0 else 0.92)
                if t in R.SNARE:
                    place(drums, SNARE_S, sec, 1.0)
                if (g - T.FIRST_BEAT) % 4 == 2:
                    place(drums, RIDE_S, sec, 0.5)
                if beat:
                    place(drums, BELL_S, sec, 0.7)
                    place(voice, beat_words[R.beat_count(g, T.FIRST_BEAT)], sec, 0.7)
                place(drums, DEAD, sec, 0.75)
                if t in R.RIM:
                    # bass follows the rim pattern; the group's last note holds through the rest
                    hold = 2 if (t + 1) in R.RESTS else 1
                    place(low, bass_note(hold * R.SIX * 0.98), sec, 0.9)

    def norm(x, peak):
        m = np.abs(x).max()
        return x / m * peak if m > 0 else x

    mono = (norm(drums, 0.5) + norm(low, 0.7) + norm(voice, 0.55) + norm(pad, 0.26))
    # a touch of stereo: voice slightly left/right by lap kind is overkill; keep centre,
    # widen the pad with a short delay
    left = mono + 0.08 * np.roll(norm(pad, 0.26), int(0.011 * SR))
    right = mono + 0.08 * np.roll(norm(pad, 0.26), -int(0.013 * SR))
    st = np.stack([left, right], axis=1)
    st = np.tanh(st * 1.2) / np.tanh(1.2)
    st = st / np.abs(st).max() * 0.95
    return st


def write_wav(path, st):
    pcm = (np.clip(st, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "soundtrack.wav"
    write_wav(out, build())
    print("wrote", out)

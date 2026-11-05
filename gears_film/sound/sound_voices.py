"""Borrowed unchanged from the owner's own Art of Dying study (branch ccr-03f861e3-540mn1,
art_of_dying/sound_voices.py), with permission, for round 2 of the gears experiment.

The instruments, all synthesised: one function per stroke, plus the choir drone.

A stroke function returns one hit as a mono array whose sample 0 is the onset, peak-normalised to 1 (the
mix sets the levels). The mix makes a few takes of each with different random seeds, so repeats are never
identical, the way no two strokes on a real drum are. The modes of an instrument (its pitches and decays)
stay the same from take to take: it is the same drum, struck again.

Design notes, since none of this could be auditioned while it was written:
* every transient starts at full strength on sample 0 (after a 0.2 ms ramp), so onsets sit on the grid;
* all saturation runs 4x oversampled (sound_dsp.shape), so nothing aliases into fizz;
* the guitars end in a steep 4.4-4.8 kHz cabinet roll-off: a "chk" with a thump, never a hiss.
"""

from __future__ import annotations

import numpy as np

from .sound_dsp import SR, convolve, eq, filt, fir, highpass, lowpass, resonance, shape

C1, C2, G2, C3 = 32.703, 65.406, 97.999, 130.813


# ---- building blocks ------------------------------------------------------------------------------
def _t(seconds):
    return np.arange(int(round(seconds * SR))) / SR


def _ring(t, f, tau, amp=1.0, phase=0.0):
    """A damped sinusoidal mode: a shell, a block or a head ringing."""
    return amp * np.sin(2 * np.pi * f * t + phase) * np.exp(-t / tau)


def _glide(t, f_start, f_end, tau):
    """Phase of a tone sliding exponentially from f_start to f_end: a struck membrane relaxing."""
    return 2 * np.pi * (f_end * t + (f_start - f_end) * tau * (1 - np.exp(-t / tau)))


def _burst(rng, t, tau, response):
    """Noise shaped by `response`, with an instant onset and an exponential decay."""
    return filt(rng.standard_normal(len(t)), response) * np.exp(-t / tau)


def _gate(t, start, end):
    """1 until `start`, then a cosine fade to 0 at `end`."""
    u = np.clip((t - start) / (end - start), 0, 1)
    return 0.5 * (1 + np.cos(np.pi * u))


def _pad(x, n):
    out = np.zeros(n)
    out[: min(n, len(x))] = x[:n]
    return out


def _ends(x):
    """A 0.2 ms ramp in (no wide-band step on sample 0) and a 4 ms fade out."""
    x = np.array(x, float)
    k, m = int(0.0002 * SR), int(0.004 * SR)
    x[:k] *= np.arange(k) / k
    x[-m:] *= np.linspace(1, 0, m)
    return x


def _norm(x):
    return x / np.max(np.abs(x))


def _drive(x, amount, bias=0.0):
    """tanh saturation, 4x oversampled. amount ~1 warms, ~15 is a high-gain amp; bias makes it asymmetric."""
    x = _norm(x)
    return shape(x, lambda v: (np.tanh(amount * v + bias) - np.tanh(bias)) / np.tanh(amount))


# ---- rim clicks -----------------------------------------------------------------------------------
def rim_click(rng):
    """Side stick: the stick laid across the snare, its shaft striking the rim. A hollow wooden "tock"
    (rim and shell near 465 Hz and 1.1 kHz), the snare head answering briefly, a sharp contact click."""
    t = _t(0.15)
    j = 1 + 0.006 * rng.standard_normal()
    x = (_ring(t, 465 * j, 0.032)
         + _ring(t, 1130 * j, 0.011, 0.45)
         + _ring(t, 196, 0.026, 0.30)
         + 1.1 * _burst(rng, t, 0.0011, lambda f: highpass(f, 1500) * lowpass(f, 9000))
         + 0.10 * _burst(rng, t, 0.018, lambda f: resonance(f, 4200, 1.1)))      # the snare wires, a whisper
    return _norm(_ends(x))


def stick_click(rng):
    """Two wooden sticks struck together: a dry, high knock (inharmonic wood modes, 2.4-5.2 kHz)."""
    t = _t(0.07)
    j = 1 + 0.006 * rng.standard_normal()
    x = (_ring(t, 2380 * j, 0.010)
         + _ring(t, 3650 * j, 0.0065, 0.55)
         + _ring(t, 5230 * j, 0.004, 0.30)
         + 0.9 * _burst(rng, t, 0.0007, lambda f: highpass(f, 2200) * lowpass(f, 12000)))
    return _norm(_ends(x))


# ---- tribal pulse ---------------------------------------------------------------------------------
STICKS_IN_PULSE = 0.42        # the sticks' share of the stacked hit
GRACE_DB = -6.0               # the grace note, relative to the sticks inside the hit


def _castanets(rng, n):
    """Castanets: two quick clacks of hardwood shells, 9 ms apart."""
    out = np.zeros(n)
    t = _t(0.05)
    for delay, amp in ((0.0, 1.0), (0.009, 0.6)):
        i = int(delay * SR)
        clack = (_ring(t, 2950, 0.005, 0.8) + _ring(t, 4600, 0.0035, 0.45)
                 + _burst(rng, t, 0.0012, lambda f: highpass(f, 2500) * lowpass(f, 11000)))
        out[i : i + len(t)] += amp * clack[: n - i]
    return out


def pulse_hit(rng):
    """The tribal stack on every beat: bass drum and mute surdo (a membrane gliding 90 -> 60 Hz), a low floor
    tom (118 -> 82 Hz, with its upper membrane modes), the mallet slap, a low wood block, castanets and sticks
    for the woody transient. Big and roomy (the mix sends it to the reverb generously).

    Returns (hit, grace): the grace is the softer stick note played a 1/32 before the hit, scaled in the
    hit's units (GRACE_DB below the sticks inside the hit)."""
    t = _t(1.2)
    v = 1 + 0.012 * rng.standard_normal()
    surdo = np.sin(_glide(t, 90 * v, 60, 0.045)) * np.exp(-t / 0.22)
    tom = np.sin(_glide(t, 118 * v, 82, 0.030)) * np.exp(-t / 0.32)
    tom_modes = _ring(t, 82 * 1.59, 0.10, 0.35) + _ring(t, 82 * 2.14, 0.06, 0.18) + _ring(t, 82 * 2.65, 0.04, 0.10)
    skins = _drive(surdo + 0.6 * tom + 0.5 * tom_modes, 1.3)    # a little warmth: harmonics phones can play
    slap = 0.45 * _burst(rng, t, 0.007, lambda f: highpass(f, 150, 1) * lowpass(f, 2200))
    block = _ring(t, 540 * v, 0.045, 0.35) + _ring(t, 1490 * v, 0.016, 0.12)
    cast = 0.20 * _castanets(rng, len(t))
    sticks = STICKS_IN_PULSE * _pad(stick_click(rng), len(t))
    x = _ends(skins + slap + block + cast + sticks)
    peak = np.max(np.abs(x))
    grace = STICKS_IN_PULSE * 10 ** (GRACE_DB / 20) * stick_click(rng) / peak
    return x / peak, grace


# ---- kick and snare -------------------------------------------------------------------------------
def kick(rng):
    """Tight metal kick: a hard beater click (3-4 kHz) over a short body that drops from 130 to 55 Hz,
    gated by 180 ms so sixteenth-note runs stay distinct."""
    t = _t(0.20)
    v = 1 + 0.01 * rng.standard_normal()
    body = _drive(np.sin(_glide(t, 130 * v, 55, 0.016)) * np.exp(-t / 0.055), 1.6)
    click = (0.5 * _burst(rng, t, 0.0015, lambda f: highpass(f, 1500) * resonance(f, 4000, 0.7))
             + _ring(t, 3300, 0.0028, 0.22))
    return _norm(_ends((body + click) * _gate(t, 0.11, 0.18)))


def snare(rng):
    """Snare: a rimshot crack (a noise burst peaking near 2.6 kHz and the rim's ring), the shell near
    200 Hz and its upper modes, and a bright burst of wire noise."""
    t = _t(0.38)
    v = 1 + 0.01 * rng.standard_normal()
    shell = (0.65 * np.sin(_glide(t, 245 * v, 200, 0.010)) * np.exp(-t / 0.07)
             + _ring(t, 200 * 1.59, 0.040, 0.32) + _ring(t, 200 * 2.30, 0.022, 0.15))
    crack = _burst(rng, t, 0.0045, lambda f: highpass(f, 800) * lowpass(f, 7500) * eq(f, 2600, 4, 0.6))
    rim = _ring(t, 1180, 0.009, 0.22)
    wires = (0.5 * _burst(rng, t, 0.09, lambda f: highpass(f, 1800) * lowpass(f, 8500, 3))
             * (1 - np.exp(-t / 0.0015)))
    x = _drive(shell + crack + rim + wires, 1.4)
    return _norm(_ends(x * _gate(t, 0.24, 0.37)))


# ---- ride -----------------------------------------------------------------------------------------
def _partials(seed, count, lo, hi):
    """A cymbal's modes: dense and inharmonic, the higher ones dying faster. Fixed per cymbal."""
    rng = np.random.default_rng(seed)
    f = np.exp(rng.uniform(np.log(lo), np.log(hi), count))
    amp = rng.uniform(0.4, 1.0, count) * (f / 1000) ** -0.4
    tau = np.clip(1.4 * (f / 1000) ** -0.6, 0.12, 2.2)
    return f, amp, tau


RIDE_MODES = _partials(1990, 70, 420.0, 11000.0)
BELL_MODES = (784.0 * np.array([1.0, 1.51, 2.24, 2.76, 3.47, 4.29, 5.11, 6.42]),    # the cup, near G5
              np.array([1.0, 0.35, 0.60, 0.45, 0.30, 0.22, 0.15, 0.10]),
              np.array([1.8, 1.0, 1.1, 0.8, 0.6, 0.5, 0.4, 0.3]))


def _modes(rng, t, modes, tilt=0.0, spread=0.3):
    """Sum of a cymbal's modes, each excited a little differently this stroke."""
    f, amp, tau = modes
    a = amp * (f / 1000) ** tilt * rng.uniform(1 - spread, 1.0, len(f))
    phase = rng.uniform(0, 2 * np.pi, len(f))
    out = np.zeros(len(t))
    for fi, ai, ti, ph in zip(f, a, tau, phase):
        out += ai * np.sin(2 * np.pi * fi * t + ph) * np.exp(-t / ti)
    return out


def ride_ping(rng):
    """A soft, ghosted ride stroke on the bow: a dark wash with a slow 1.5 ms bloom and a faint tick."""
    t = _t(1.6)
    wash = _norm(_modes(rng, t, RIDE_MODES, tilt=-0.25) * np.clip(t / 0.0015, 0, 1))
    tick = 0.25 * _burst(rng, t, 0.0010, lambda f: highpass(f, 3000) * lowpass(f, 12000))
    x = filt(wash + tick, lambda f: highpass(f, 350) * lowpass(f, 12000))
    return _norm(_ends(x * _gate(t, 1.3, 1.6)))


def ride_bell(rng):
    """The ride bell on the beat: a clear, bright ping (the cup's modes, near G5) over the cymbal's wash."""
    t = _t(2.0)
    tone = _norm(_modes(rng, t, BELL_MODES, spread=0.15))
    wash = _norm(_modes(rng, t, RIDE_MODES, tilt=0.1))
    tick = _burst(rng, t, 0.0008, lambda f: highpass(f, 2500) * lowpass(f, 13000))
    x = (tone + 0.35 * wash) * np.clip(t / 0.0004, 0, 1) + 0.35 * tick
    x = filt(x, lambda f: highpass(f, 400) * lowpass(f, 13000))
    return _norm(_ends(x * _gate(t, 1.7, 2.0)))


# ---- bass -----------------------------------------------------------------------------------------
def bass_note(rng, length):
    """A picked electric bass on C1 (32.7 Hz): string harmonics (slightly stretched, notched by the pick
    position, the high ones dying first), the pick's click and a brief sharp pitch from the attack. The
    fretting hand stops it after `length` s; mild drive and a DI-style EQ (a growl near 800 Hz) let small
    speakers hear the note through its harmonics."""
    t = _t(length + 0.02)
    k = np.arange(1, 90)
    f = C1 * k * np.sqrt(1 + 1e-4 * k * k)
    a = np.abs(np.sin(np.pi * k * 0.12)) / k ** 0.85 * rng.uniform(0.85, 1.0, len(k))
    tau = 0.8 / (1 + 0.03 * k ** 1.4)
    warp = t + 0.006 * 0.02 * (1 - np.exp(-t / 0.02))             # +0.6% pitch, relaxing in 20 ms
    x = np.zeros(len(t))
    for fk, ak, tk in zip(f, a, tau):
        x += ak * np.sin(2 * np.pi * fk * warp) * np.exp(-t / tk)
    x = _norm(x) + 0.10 * _burst(rng, t, 0.002, lambda f: resonance(f, 2000, 0.9))
    x = _drive(x * _gate(t, length, length + 0.015), 2.0)
    x = filt(x, lambda f: highpass(f, 28) * lowpass(f, 2800) * eq(f, 800, 3.0, 0.7))
    return _norm(_ends(x))


# ---- guitars --------------------------------------------------------------------------------------
GUITARS = {    # two players, two rigs: slightly different pick spot, gain, tuning and cabinet
    "L": dict(pick=0.14, drive=13.0, cents=0.0, chk=1800.0, presence=2300.0, top=4800.0),
    "R": dict(pick=0.19, drive=16.0, cents=4.0, chk=2300.0, presence=1700.0, top=4400.0),
}


def chug(rng, side):
    """A palm-muted dead-note chug on C2 (65.4 Hz): the palm kills the string within ~30 ms (the highs
    sooner), the pick adds a "chk". Then a mid boost into a high-gain stage (keeps the lows tight), and a
    4x12-style cabinet: a thump near 105 Hz, scooped mids, a little presence, a steep roll-off."""
    g = GUITARS[side]
    t = _t(0.15)
    f0 = C2 * 2 ** (g["cents"] / 1200)
    k = np.arange(1, 80)
    f = f0 * k * np.sqrt(1 + 5e-5 * k * k)
    keep = f < 6000
    k, f = k[keep], f[keep]
    a = np.abs(np.sin(np.pi * k * g["pick"])) / k ** 0.8 * rng.uniform(0.8, 1.0, len(k))
    tau = 0.028 / (1 + 0.02 * k ** 1.5)
    x = np.zeros(len(t))
    for fk, ak, tk in zip(f, a, tau):
        x += ak * np.sin(2 * np.pi * fk * t) * np.exp(-t / tk)
    x = _norm(x) + 0.3 * _burst(rng, t, 0.0035, lambda f: resonance(f, g["chk"], 0.8))
    x = filt(x, lambda f: highpass(f, 100, 1) * eq(f, 750, 6.0, 0.9))
    x = _drive(x, g["drive"], bias=0.2)
    x = filt(x, lambda f: (highpass(f, 70) * eq(f, 105, 4.0, 0.45) * eq(f, 480, -4.0, 0.8)
                           * eq(f, g["presence"], 2.0, 0.5) * lowpass(f, g["top"], 6)))
    return _norm(_ends(x * _gate(t, 0.10, 0.145)))


# ---- the choir ------------------------------------------------------------------------------------
FORMANTS_AH = ((600, 60), (1040, 70), (2250, 110), (2450, 120), (2750, 130))    # a bass voice singing "a"


def vowel_ah(f):
    """Spectral envelope of a low sung "aah": the glottal source (with lip radiation) through five formants."""
    f = np.maximum(np.asarray(f, float), 1e-3)
    env = (f / 100) / (1 + (f / 100) ** 2)
    for fc, bw in FORMANTS_AH:
        env = env / np.sqrt((1 - (f / fc) ** 2) ** 2 + (f * bw / fc ** 2) ** 2)
    return env


class Choir:
    """A low choir holding C on "aah" (C2, G2, C3) for the whole film, breathing slowly.

    Each singer is a wavetable of formant-weighted harmonics with their own detune, vibrato, slow drift and
    breathing (staggered, so the chord never drops out), plus a little formant-shaped breath noise. Every
    oscillator and modulation makes a whole number of cycles in `period` samples, so the drone is exactly
    periodic and the loop has no seam. render(idx) is a pure function of the sample index.
    """

    VOICES = ((C2, 3, 0.75, 7), (G2, 3, 0.50, 5), (C3, 4, 1.00, 6))   # (Hz, singers, level, breaths per loop)
    TABLE = 4096
    AIR = 0.10

    def __init__(self, period, seed=8):
        rng = np.random.default_rng(seed)
        self.period = period
        cycles = lambda hz: max(1, int(round(hz * period / SR)))
        self.singers = []
        for f0, count, level, breaths in self.VOICES:
            for _ in range(count):
                k = np.arange(1, int(4500 / f0) + 1)
                spec = np.zeros(self.TABLE // 2 + 1, complex)
                spec[k] = vowel_ah(k * f0) * np.exp(2j * np.pi * rng.uniform(0, 1, len(k)))
                table = np.fft.irfft(spec, self.TABLE)
                table /= np.sqrt(np.mean(table ** 2))
                pan = rng.uniform(-0.5, 0.5)
                self.singers.append(dict(
                    table=np.append(table, table[0]),
                    carrier=cycles(f0 * 2 ** (rng.uniform(-9, 9) / 1200)),
                    vib=cycles(rng.uniform(4.6, 5.6)), vib_depth=rng.uniform(0.0025, 0.004),
                    vib_phase=rng.uniform(0, 2 * np.pi),
                    drift=int(rng.integers(2, 6)), drift_depth=rng.uniform(0.0008, 0.0018),
                    drift_phase=rng.uniform(0, 2 * np.pi),
                    breaths=breaths, breath_phase=rng.uniform(0, 2 * np.pi),
                    level=level, gains=(np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4))))
        # breath noise: one loop of white noise through the vowel, filtered circularly (seamless)
        h = fir(lambda f: vowel_ah(f) * highpass(f, 300) * lowpass(f, 5000), taps=4096)
        air = convolve(rng.standard_normal(period), h, period=period)
        self.air = air / np.sqrt(np.mean(air ** 2))

    def _turn(self, cycles, i):
        """Phase (in turns, 0..1) of something making `cycles` whole cycles per loop, at sample i (exact)."""
        return ((cycles * i) % self.period) / self.period

    def render(self, idx):
        """The drone at absolute sample indices idx (any integers; taken modulo the loop). Stereo (n, 2)."""
        i = np.asarray(idx, np.int64) % self.period
        out = np.zeros((len(i), 2))
        breath = np.zeros(len(i))
        hz = lambda cycles: cycles * SR / self.period
        for s in self.singers:
            f0 = hz(s["carrier"])
            vib = 2 * np.pi * self._turn(s["vib"], i) + s["vib_phase"]
            drift = 2 * np.pi * self._turn(s["drift"], i) + s["drift_phase"]
            # frequency f0 (1 + dv sin(vib) + dd sin(drift)), integrated in closed form
            pos = (self._turn(s["carrier"], i)
                   - f0 * s["vib_depth"] / (2 * np.pi * hz(s["vib"])) * np.cos(vib)
                   - f0 * s["drift_depth"] / (2 * np.pi * hz(s["drift"])) * np.cos(drift)) % 1.0
            x = pos * self.TABLE
            j = np.minimum(x.astype(np.int64), self.TABLE - 1)
            fr = x - j
            v = s["table"][j] * (1 - fr) + s["table"][j + 1] * fr
            env = s["level"] * (0.72 + 0.28 * np.sin(2 * np.pi * self._turn(s["breaths"], i) + s["breath_phase"]))
            v *= env
            out[:, 0] += v * s["gains"][0]
            out[:, 1] += v * s["gains"][1]
            breath += env
        air = self.AIR * breath / len(self.singers)
        out[:, 0] += air * self.air[i]
        out[:, 1] += air * self.air[(i + self.period // 3) % self.period]
        return out

"""The round-2 soundtrack: every part of the intro, synthesised and placed by rhythm.events().

    python3 soundtrack.py out/soundtrack.wav

Adapted from the repository owner's art_of_dying/soundtrack.py (branch ccr-03f861e3-540mn1): the Kit /
Bus / calibrate structure and the mix numbers are theirs; the timeline (rhythm.py) and the loop length are
this film's. The instruments are in sound_voices.py, the DSP in sound_dsp.py (both copied unchanged).

Layers, in the order they enter: rim clicks, tribal pulse (with a stick grace note a 1/32 early), kick +
snare, ride (ghost pings off the beat, bell on it), bass on C1 (each group's last note held through its
rest), two palm-muted guitars on C2. A low choir holds C under everything and is never ducked.
Each layer is levelled by measurement (alone over the full-mix lap) to BALANCE_LU relative to kick+snare.

The film loops: tails fold back onto the start, the choir's oscillators make whole cycles in the loop.
Master: -14 LUFS integrated, true peak <= -2.0 dBTP before AAC.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from rhythm import (CYCLE, FULL_MIX_LAP, LAP, LAYERS, RIM_RESTS, SIXTEENTH, TOTAL, GROUP_STARTS, Hit,
                    events, hits_at)
from sound_dsp import (SR, convolve, fir, fold, highpass, limit, loudness, reverb_ir, true_peak_env,
                       write_wav)
from sound_voices import (Choir, bass_note, chug, kick, pulse_hit, ride_bell, ride_ping, rim_click, snare,
                          stick_click)

N = round(TOTAL * SR)                 # one loop, the film's exact length
GRACE = SIXTEENTH / 2
BASS_NOTE = 0.88 * SIXTEENTH
BASS_HOLD = 1.88 * SIXTEENTH
SEED = 2008

BALANCE_LU = {"rim": -7.0, "pulse": -1.0, "ks": 0.0, "ride": -8.0, "bass": -6.0, "gtr": -4.5, "choir": -17.0}
REF_LUFS = -20.0

PARTS = {
    "click": (("rim", 0.0), ("stick", 0.0)),
    "pulse": (("pulse", 0.0), ("grace", -GRACE)),
    "kick": (("kick", 0.0),),
    "snare": (("snare", 0.0),),
    "ride": (("ride", 0.0),),
    "bell": (("bell", 0.0),),
    "note": (("bass", 0.0),),
    "chug": (("gtr_L", 0.0), ("gtr_R", None)),
}
KINDS = tuple(PARTS)
SOUNDS = {   # part: (dB within its layer, pan, reverb send, velocity spread dB)
    "rim": (0.0, -0.15, 0.20, 0.35),
    "stick": (-3.0, 0.12, 0.15, 0.35),
    "pulse": (0.0, 0.00, 0.50, 0.30),
    "grace": (0.0, 0.10, 0.30, 0.50),
    "kick": (0.0, 0.00, 0.06, 0.25),
    "snare": (-3.0, 0.05, 0.30, 0.30),
    "ride": (-8.0, 0.35, 0.20, 0.80),
    "bell": (0.0, 0.30, 0.25, 0.40),
    "bass": (0.0, 0.00, 0.00, 0.40),
    "gtr_L": (0.0, -0.60, 0.08, 0.50),
    "gtr_R": (0.0, 0.60, 0.08, 0.50),
}
ACCENT_DB = 2.0
GTR_LAG = (0.001, 0.004)
CHOIR_SEND = 0.8
WET = 0.5
TARGET_LUFS = -14.0
CEILING_DBTP = -2.0


class Kit:
    def __init__(self, seed=SEED):
        rng = np.random.default_rng(seed)
        take = lambda make, count: [make(rng) for _ in range(count)]
        pulses = take(pulse_hit, 4)
        self.parts = {
            "rim": take(rim_click, 8), "stick": take(stick_click, 8),
            "pulse": [h for h, _ in pulses], "grace": [g for _, g in pulses],
            "kick": take(kick, 8), "snare": take(snare, 6),
            "ride": take(ride_ping, 6), "bell": take(ride_bell, 4),
            "bass": take(lambda r: bass_note(r, BASS_NOTE), 6),
            "gtr_L": take(lambda r: chug(r, "L"), 6), "gtr_R": take(lambda r: chug(r, "R"), 6),
            "bass_long": take(lambda r: bass_note(r, BASS_HOLD), 6),
        }


class Bus:
    """Stereo dry and reverb-send buffers for one loop of `n` samples starting at absolute sample i0."""
    def __init__(self, i0, n):
        self.i0, self.n = i0, n
        self.dry = np.zeros((n, 2)); self.send = np.zeros((n, 2))

    def _mix(self, a, seg, left, right, send):
        self.dry[a:a + len(seg), 0] += seg * left; self.dry[a:a + len(seg), 1] += seg * right
        if send:
            self.send[a:a + len(seg), 0] += seg * left * send; self.send[a:a + len(seg), 1] += seg * right * send

    def add(self, i, mono, gain, pan, send):
        left, right = gain * np.cos((pan + 1) * np.pi / 4), gain * np.sin((pan + 1) * np.pi / 4)
        a = (i - self.i0) % self.n
        head = min(len(mono), self.n - a)
        self._mix(a, mono[:head], left, right, send)
        if head < len(mono):
            self._mix(0, mono[head:], left, right, send)


def play(bus, kit, levels, hit):
    rng = np.random.default_rng((SEED, hit.n, KINDS.index(hit.kind)))
    for part, delay in PARTS[hit.kind]:
        db, pan, send, spread = SOUNDS[part]
        if delay is None:
            delay = rng.uniform(*GTR_LAG)
        db += levels[hit.key] + spread * float(np.clip(rng.standard_normal(), -2, 2))
        if hit.accent and hit.key == "rim":
            db += ACCENT_DB
        takes = kit.parts[part]
        if part == "bass" and (hit.n + 1) % CYCLE in RIM_RESTS:
            takes = kit.parts["bass_long"]
        mono = takes[rng.integers(len(takes))]
        bus.add(round((hit.time + delay) * SR), mono, hit.gain * 10 ** (db / 20), pan, send)


def calibrate(kit, choir, lap=FULL_MIX_LAP):
    i0, n = round(lap * LAP * SR), round(LAP * SR)
    levels = {}
    for layer in LAYERS:
        bus = Bus(i0, n)
        for m in range(lap * CYCLE, (lap + 1) * CYCLE):
            for kind in hits_at(layer.key, m):
                play(bus, kit, {layer.key: 0.0}, Hit(m, m * SIXTEENTH, layer.key, kind, m % CYCLE in GROUP_STARTS, 1.0))
        levels[layer.key] = REF_LUFS + BALANCE_LU[layer.key] - loudness(bus.dry)
    levels["choir"] = REF_LUFS + BALANCE_LU["choir"] - loudness(choir.render(i0 + np.arange(n)))
    return levels


def render():
    clock = time.time()
    kit, choir = Kit(), Choir(N)
    levels = calibrate(kit, choir)
    bus = Bus(0, N)
    for hit in events():
        play(bus, kit, levels, hit)
    drone = choir.render(np.arange(N)) * 10 ** (levels["choir"] / 20)
    bus.dry += drone; bus.send += drone * CHOIR_SEND
    wet = fold(convolve(bus.send, reverb_ir(), tail=True), N)
    mix = bus.dry + WET * wet
    mix = convolve(mix, fir(lambda f: highpass(f, 24)), period=N)
    env = true_peak_env(mix, periodic=True)
    gain_db = 6.0
    for attempt in range(8):
        g = 10 ** (gain_db / 20)
        out, gr = limit(mix * g, CEILING_DBTP, env * g, periodic=True)
        error = TARGET_LUFS - loudness(out)
        if abs(error) < 0.01:
            break
        gain_db += error
    info = dict(levels=levels, gain_db=gain_db, gr=gr, lufs=loudness(out),
                true_peak=20 * np.log10(true_peak_env(out, periodic=True).max()), seconds=time.time() - clock)
    assert np.isfinite(out).all()
    return out, info


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out/soundtrack.wav"
    audio, info = render()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_wav(out_path, audio)
    gr = info["gr"]
    print(f"wrote {out_path}: {len(audio)} samples ({len(audio)/SR:.4f} s), {info['seconds']:.0f} s to render")
    print("  levels (dB): " + "  ".join(f"{k} {v:+.1f}" for k, v in info["levels"].items()))
    print(f"  master gain {info['gain_db']:+.2f} dB · limiter max {gr.max():.1f} dB, >1 dB on {100*np.mean(gr>1):.1f}% of samples")
    print(f"  loudness {info['lufs']:.2f} LUFS · true peak {info['true_peak']:.2f} dBTP")

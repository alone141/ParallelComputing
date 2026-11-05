"""The soundtrack: every part of the intro, synthesised and placed by rhythm.events().

Adapted from the owner's own Art of Dying study (branch ccr-03f861e3-540mn1, art_of_dying/soundtrack.py):
the mixer, balance and master are theirs; the timeline (rhythm.py) and the film tempo are this experiment's.

    python3 soundtrack.py build/soundtrack.wav                      # the film: 48 kHz stereo, round(TOTAL * 48000) samples
    python3 soundtrack.py build/excerpt.wav --start 17 --dur 10     # an excerpt, for quick checks

Layers, in the order they enter (rhythm.LAYERS). Every hit is scaled by Hit.gain (presence x focus duck):
    rim    a side-stick rim click layered with a wooden stick click, +2 dB on group starts (Hit.accent)
    pulse  a big tribal stacked drum on every beat, with a softer stick grace note a 1/32 before it
    ks     a tight metal kick and a cracking snare
    ride   soft, ghosted ride pings on the off-beats; the ride bell on the beats
    bass   a picked electric bass on C1: short notes, mild drive; each group's last note is held
           through its rest, as the tab writes it (an eighth note)
    gtr    two palm-muted guitars chugging C2, panned left and right, never quite together
Under everything a low choir holds C on "aah". It is not one of the six layers and is never ducked.

The balance is set by measurement, not by hand: each layer is rendered alone over one lap of the full mix
and levelled to its BALANCE_LU (K-weighted, relative to the kick + snare).

The film loops, so the sound is circular: tails of the last hits and of the reverb fold back onto the start,
and the choir's oscillators make whole cycles in the loop. The file's end runs seamlessly into its start.
Master: -14 LUFS integrated, true peak <= -2.0 dBTP (room for AAC to stay under -1 dBTP).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from rhythm import (CYCLE, LAP, LAYERS, N_LAPS, RIM_RESTS, SIXTEENTH, TOTAL, Hit, events, hits_at,
                    is_group_start, lap_start)
from sound_dsp import (SR, convolve, fir, fold, highpass, limit, loudness, reverb_ir, true_peak_env,
                       write_wav)
from sound_voices import (Choir, bass_note, chug, kick, pulse_hit, ride_bell, ride_ping, rim_click, snare,
                          stick_click)

N = round(TOTAL * SR)                 # samples in the film: one loop
GRACE = SIXTEENTH / 2                 # the pulse's grace note: a 1/32 early (53.6 ms)
BASS_NOTE = 0.88 * SIXTEENTH          # the bass is stopped just before the next sixteenth
BASS_HOLD = 1.88 * SIXTEENTH          # ...except a group's last note, held through its rest (an
                                      # eighth note in the tab), stopped before the next group
SEED = 2008

# ---- mix --------------------------------------------------------------------------------------------
# Each layer's loudness alone and dry over one lap of the full mix, in LU relative to the kick + snare.
BALANCE_LU = {"rim": -7.0, "pulse": -1.0, "ks": 0.0, "ride": -8.0, "bass": -6.0, "gtr": -4.5, "choir": -17.0}
REF_LUFS = -20.0                      # where the kick + snare sit alone, before the master gain

# The sounds that make up each kind of hit: (part, delay in s; None = the second guitar's own timing).
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
# part: (dB within its layer, pan -1..1, reverb send, velocity spread in dB (one standard deviation))
SOUNDS = {
    "rim": (0.0, -0.15, 0.20, 0.35),
    "stick": (-3.0, 0.12, 0.15, 0.35),
    "pulse": (0.0, 0.00, 0.50, 0.30),
    "grace": (0.0, 0.10, 0.30, 0.50),      # its -6 dB (re the sticks in the hit) is built into the voice
    "kick": (0.0, 0.00, 0.06, 0.25),
    "snare": (-3.0, 0.05, 0.30, 0.30),
    "ride": (-8.0, 0.35, 0.20, 0.80),
    "bell": (0.0, 0.30, 0.25, 0.40),
    "bass": (0.0, 0.00, 0.00, 0.40),
    "gtr_L": (0.0, -0.60, 0.08, 0.50),
    "gtr_R": (0.0, 0.60, 0.08, 0.50),
}
ACCENT_DB = 2.0                       # rim clicks on a group start (an interpretive accent, kept subtle)
GTR_LAG = (0.001, 0.004)              # the right guitar trails the left by 1-4 ms, hit to hit
CHOIR_SEND = 0.8
WET = 0.5                             # reverb return (the impulse response has unit energy)

# ---- master -----------------------------------------------------------------------------------------
TARGET_LUFS = -14.0
CEILING_DBTP = -2.0                   # AAC 256k adds about 0.5 dB of overshoot: it then stays under -1 dBTP
MASTER_GAIN_DB = 6.34                 # the full render solves for this and prints it; excerpts reuse it
PRE, POST = int(3.0 * SR), int(0.05 * SR)    # an excerpt's pre-roll (reverb, limiter) and post-roll


class Kit:
    """A few takes of every sound, made once per run (deterministic); the mix picks one take per hit."""

    def __init__(self, seed=SEED):
        rng = np.random.default_rng(seed)
        take = lambda make, count: [make(rng) for _ in range(count)]
        pulses = take(pulse_hit, 4)
        self.parts = {
            "rim": take(rim_click, 8),
            "stick": take(stick_click, 8),
            "pulse": [hit for hit, _ in pulses],
            "grace": [grace for _, grace in pulses],
            "kick": take(kick, 8),
            "snare": take(snare, 6),
            "ride": take(ride_ping, 6),
            "bell": take(ride_bell, 4),
            "bass": take(lambda r: bass_note(r, BASS_NOTE), 6),
            "gtr_L": take(lambda r: chug(r, "L"), 6),
            "gtr_R": take(lambda r: chug(r, "R"), 6),
            "bass_long": take(lambda r: bass_note(r, BASS_HOLD), 6),   # last: keeps the other takes
        }


class Bus:
    """Stereo dry and reverb-send buffers for absolute samples [i0, i0 + n).
    With period=n the bus is one loop: whatever runs off one end comes back in at the other."""

    def __init__(self, i0, n, period=None):
        self.i0, self.n, self.period = i0, n, period
        self.dry = np.zeros((n, 2))
        self.send = np.zeros((n, 2))
        self.wrapped = 0.0            # energy that folded round the loop (a check that tails wrap)

    def _mix(self, a, seg, left, right, send):
        self.dry[a : a + len(seg), 0] += seg * left
        self.dry[a : a + len(seg), 1] += seg * right
        if send:
            self.send[a : a + len(seg), 0] += seg * (left * send)
            self.send[a : a + len(seg), 1] += seg * (right * send)

    def add(self, i, mono, gain, pan, send):
        """Add `mono` x gain starting at absolute sample i, equal-power panned."""
        left, right = gain * np.cos((pan + 1) * np.pi / 4), gain * np.sin((pan + 1) * np.pi / 4)
        a = i - self.i0
        if self.period:
            a %= self.period
            head = min(len(mono), self.n - a)
            self._mix(a, mono[:head], left, right, send)
            if head < len(mono):
                self._mix(0, mono[head:], left, right, send)
                self.wrapped += float(np.sum(mono[head:] ** 2)) * gain * gain
        else:
            lo, hi = max(a, 0), min(a + len(mono), self.n)
            if lo < hi:
                self._mix(lo, mono[lo - a : hi - a], left, right, send)


def play(bus, kit, levels, hit, offsets=(0,)):
    """Place one Hit: a take of each of its parts at its level (layer level x Hit.gain, the accent and a
    little velocity spread), pan and send. `offsets` (samples) also places it a loop earlier and later, so
    an excerpt hears tails across the loop point."""
    rng = np.random.default_rng((SEED, hit.n, KINDS.index(hit.kind)))
    for part, delay in PARTS[hit.kind]:
        db, pan, send, spread = SOUNDS[part]
        if delay is None:
            delay = rng.uniform(*GTR_LAG)
        db += levels[hit.key] + spread * float(np.clip(rng.standard_normal(), -2, 2))
        if hit.accent and hit.key == "rim":
            db += ACCENT_DB
        takes = kit.parts[part]
        if part == "bass" and (hit.n + 1) % CYCLE in RIM_RESTS:   # the group's last note: held
            takes = kit.parts["bass_long"]
        mono = takes[rng.integers(len(takes))]
        i = round((hit.time + delay) * SR)
        for off in offsets:
            bus.add(i + off, mono, hit.gain * 10 ** (db / 20), pan, send)


def calibrate(kit, choir, lap=N_LAPS - 1):
    """Layer levels (dB): each layer alone, dry, at full gain over one lap of the full mix (a loop of its
    own) measures REF_LUFS + BALANCE_LU. The choir is levelled the same way."""
    i0, n = round(lap_start(lap) * SR), round(LAP * SR)
    levels = {}
    for layer in LAYERS:
        bus = Bus(i0, n, period=n)
        for m in range(lap * CYCLE, (lap + 1) * CYCLE):
            for kind in hits_at(layer.key, m):
                play(bus, kit, {layer.key: 0.0}, Hit(m, m * SIXTEENTH, layer.key, kind, is_group_start(m), 1.0))
        levels[layer.key] = REF_LUFS + BALANCE_LU[layer.key] - loudness(bus.dry)
    levels["choir"] = REF_LUFS + BALANCE_LU["choir"] - loudness(choir.render(i0 + np.arange(n)))
    return levels


def render(start=None, dur=None):
    """The mastered soundtrack, (n, 2) float. With dur=None, the whole film as one loop; otherwise the
    excerpt [start, start + dur) s, identical to that stretch of the film apart from the master gain, which
    is MASTER_GAIN_DB rather than solved. Returns (audio, info)."""
    clock = time.time()
    kit, choir = Kit(), Choir(N)
    levels = calibrate(kit, choir)
    full = dur is None
    if full:
        bus, offsets = Bus(0, N, period=N), (0,)
    else:
        s, d = round(start * SR) % N, round(dur * SR)
        bus, offsets = Bus(s - PRE, PRE + d + POST), (-N, 0, N)
    for hit in events():
        play(bus, kit, levels, hit, offsets)
    drone = choir.render(bus.i0 + np.arange(bus.n)) * 10 ** (levels["choir"] / 20)
    bus.dry += drone
    bus.send += drone * CHOIR_SEND
    info = dict(levels=levels)

    # reverb: a loop for the film (its tail folds onto the start), linear for an excerpt
    ir = reverb_ir()
    if full:
        wet = convolve(bus.send, ir, tail=True)
        info["tail_db"] = 10 * np.log10(np.sum(wet[N:] ** 2) / np.sum(wet ** 2))
        info["dry_wrap_db"] = 10 * np.log10(bus.wrapped / np.sum(bus.dry ** 2) + 1e-30)
        wet = fold(wet, N)
    else:
        wet = convolve(bus.send, ir)
    mix = bus.dry + WET * wet
    mix = convolve(mix, fir(lambda f: highpass(f, 24)), period=N if full else None)   # sub-sonic clean-up

    # master: gain into the true-peak limiter; the film solves the gain for TARGET_LUFS
    env = true_peak_env(mix, periodic=full)
    gain_db = MASTER_GAIN_DB
    for attempt in range(8 if full else 1):
        g = 10 ** (gain_db / 20)
        out, gr = limit(mix * g, CEILING_DBTP, env * g, periodic=full)
        error = TARGET_LUFS - loudness(out) if full else 0.0
        if abs(error) < 0.01 or attempt == 7:
            break
        gain_db += error          # the limiter takes a little back, so this converges in a few steps
    if not full:
        out, gr = out[PRE : PRE + d], gr[PRE : PRE + d]
    info.update(gain_db=gain_db, gr=gr, lufs=loudness(out),
                true_peak=20 * np.log10(true_peak_env(out, periodic=full).max()),
                seconds=time.time() - clock)
    assert np.isfinite(out).all()
    return out, info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render the soundtrack (48 kHz, 24-bit stereo WAV).")
    ap.add_argument("out", nargs="?", default="build/soundtrack.wav")
    ap.add_argument("--start", type=float, help="excerpt start, s")
    ap.add_argument("--dur", type=float, help="excerpt length, s")
    args = ap.parse_args(argv)
    if args.start is None and args.dur is None:
        start = dur = None
    else:
        start = args.start or 0.0
        dur = args.dur if args.dur is not None else TOTAL - start
    audio, info = render(start, dur)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_wav(args.out, audio)

    what = "film" if dur is None else f"excerpt {start:.3f}-{start + dur:.3f} s"
    print(f"wrote {args.out}: {what}, {len(audio)} samples ({len(audio) / SR:.4f} s), {info['seconds']:.1f} s to render")
    print("  levels (dB): " + "  ".join(f"{k} {v:+.1f}" for k, v in info["levels"].items()))
    gr = info["gr"]
    print(f"  master gain {info['gain_db']:+.2f} dB · limiter: max {gr.max():.1f} dB, "
          f">1 dB on {100 * np.mean(gr > 1):.1f}% · >3 dB on {100 * np.mean(gr > 3):.2f}% of samples")
    print(f"  loudness {info['lufs']:.2f} LUFS (own BS.1770 meter) · true peak {info['true_peak']:.2f} dBTP")
    if dur is None:
        print(f"  loop: reverb tail folded onto the start {info['tail_db']:.1f} dB of the reverb, "
              f"dry tails {info['dry_wrap_db']:.1f} dB of the dry mix")
        if abs(info["gain_db"] - MASTER_GAIN_DB) > 0.05:
            print(f"  note: set MASTER_GAIN_DB = {info['gain_db']:.2f} so excerpts match the film's level")


if __name__ == "__main__":
    main()

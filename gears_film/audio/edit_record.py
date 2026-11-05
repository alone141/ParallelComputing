"""Build the film's audio from the real recording, and verify a muxed film against the picture's grid.

    python3 edit_record.py build  <record_48k.wav> <record_map.json> <fit.json> <film_audio.wav>
    python3 edit_record.py verify <decoded_film_audio.wav> <record_map.json>

build: takes each film segment from the record at the start time fit_record.py measured, at the
measured tempo; joins segments with a short linear crossfade centred on the segment boundary (a bar
line, so a cut is masked by the downbeat); fades the end; trims the gain so the true peak stays
under -1 dBTP after AAC (the recording is left otherwise untouched: no limiting, no EQ). The film's
step clock is the engine's: each segment lasts steps × 60 / (bpm × steps_per_beat) with the bpm the
engine used, so the picture must have been rendered with the same tempo map (paste fit.json's bpm per
segment into the song data). Settings in record_map.json: crossfade_s (default 0.04), fade_out_s
(1.2), gain_db (-2.4), and per-segment "bpm" (the engine's).

verify: give it the audio decoded from the finished MP4
(`ffmpeg -i film.mp4 -vn -ar 48000 -c:a pcm_s24le decoded.wav`) and it reports, for each anchor and
segment, the onset energy on the film grid versus half a step off and the best shift within ±40 ms.
Anything beyond a few ms means the picture and the sound disagree.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from fit_record import Envelopes, read_wav


def film_clock(rm):
    """(film start s, step s) per segment, from the engine's tempo map."""
    spb = int(rm["steps_per_beat"])
    out, t = [], 0.0
    for seg in rm["segments"]:
        step_s = 60.0 / (float(seg.get("bpm", rm["bpm"])) * spb)
        out.append((t, step_s))
        t += int(seg["steps"]) * step_s
    return out, t


def write_wav(path, y, sr):
    import wave
    q = np.clip(np.round(y * (1 << 23)), -(1 << 23), (1 << 23) - 1).astype(np.int32)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(np.stack([q & 0xFF, (q >> 8) & 0xFF, (q >> 16) & 0xFF], -1).astype(np.uint8).tobytes())


def build(wav, map_path, fit_path, out):
    rm, fit = json.load(open(map_path)), json.load(open(fit_path))
    sr, x = read_wav(wav)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    clock, total_s = film_clock(rm)
    nX = int(float(rm.get("crossfade_s", 0.04)) / 2 * sr)
    total = int(round(total_s * sr))
    y = np.zeros((total, 2))
    cursor, prev_a, prev_n = 0, 0, 0
    for i, (seg, fseg) in enumerate(zip(rm["segments"], fit["segments"])):
        if fseg.get("start_s") is None:
            raise SystemExit(f"segment {i} has no fit; give it an anchor")
        n = int(round(int(seg["steps"]) * clock[i][1] * sr))
        a = int(round(fseg["start_s"] * sr))
        if i == 0:
            y[:n] = x[a:a + n]
        elif fseg.get("contiguous_with_previous") and abs(a - (prev_a + prev_n)) <= int(0.002 * sr):
            # the record plays straight through here and the picture's clock was set from the
            # measured boundary: copy on, no crossfade, from exactly where the last segment ended
            a = prev_a + prev_n
            y[cursor: cursor + n] = x[a: a + n]
        else:                                           # crossfade centred on the boundary
            ramp = np.linspace(0, 1, 2 * nX)[:, None]
            prev_tail = x[prev_a + prev_n - nX: prev_a + prev_n + nX]
            new_head = x[a - nX: a + nX]
            y[cursor - nX: cursor + nX] = prev_tail * (1 - ramp) + new_head * ramp
            y[cursor + nX: cursor + n] = x[a + nX: a + n]
        prev_a, prev_n = a, n
        cursor += n
        print(f"segment {i}: record {fseg['start_s']:.4f} s → film {clock[i][0]:.4f} s, "
              f"{n / sr:.4f} s at {seg.get('bpm', rm['bpm'])} bpm")
    fade = int(float(rm.get("fade_out_s", 1.2)) * sr)
    if fade:
        y[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.5
    y *= 10 ** (float(rm.get("gain_db", -2.4)) / 20)
    print(f"film audio: {total / sr:.4f} s · peak {20 * np.log10(np.abs(y).max() + 1e-12):.2f} dBFS")
    write_wav(out, y, sr)
    print("wrote", out)
    verify_array(y.mean(1), sr, rm)


def film_times_of_record_steps(rm, record_steps):
    clock, _ = film_clock(rm)
    times = []
    for s in record_steps:
        for seg, (t0, step_s) in zip(rm["segments"], clock):
            r0, n = int(seg["record_step"]), int(seg["steps"])
            if r0 <= s < r0 + n:
                times.append(t0 + (s - r0) * step_s)
                break
    return np.array(times)


def verify_array(mono, sr, rm):
    envs = Envelopes(mono, sr)
    clock, _ = film_clock(rm)
    print("verify (onset energy on the film grid vs half a step off; best shift within ±40 ms):")
    for a in rm["anchors"]:
        env = envs.band(*a["band_hz"])
        for i, seg in enumerate(rm["segments"]):
            r0, n = int(seg["record_step"]), int(seg["steps"])
            inside = [s for s in a["record_steps"] if r0 <= s < r0 + n]
            if len(inside) < 8:
                continue
            t = film_times_of_record_steps(rm, inside)
            on, off = envs.score(env, t), envs.score(env, t + clock[i][1] / 2)
            best = max((envs.score(env, t + d), d) for d in np.arange(-0.04, 0.0401, 0.001))
            # Sync is inaudible below about 20 ms; a flag means a wrong grid, not a weak anchor
            # (an anchor whose on/off ratio is near 1 says little either way: say so).
            flag = ("   <-- check" if abs(best[1]) > 0.020 or on <= off else
                    "   (weak anchor)" if on < off * 1.2 else "")
            print(f"  {a['name']:<14} segment {i}: on-grid {on:.2f} · half off {off:.2f} · "
                  f"best at {best[1] * 1000:+.0f} ms{flag}")


def main(argv):
    if argv[:1] == ["build"] and len(argv) == 5:
        return build(*argv[1:])
    if argv[:1] == ["verify"] and len(argv) == 3:
        sr, x = read_wav(argv[1])
        return verify_array(x.mean(1), sr, json.load(open(argv[2])))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

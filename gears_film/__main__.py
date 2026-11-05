"""Command line: python3 -m gears_film <command> songs/<slug>.py [...]

  check songs/x.py                      validate the song data and print the timeline
  still songs/x.py <seconds> out.png    render one frame at a film time
  render songs/x.py [--workers 4] [--crf 28]
                                        render the silent MP4 (4 parallel slices), check the frame count,
                                        and write record_map.json next to it (gears_film/out/<slug>/)
  export-record-map songs/x.py [out.json]
Environment overrides: GEARS_BPM (nominal bpm), GEARS_OFFSET (seconds before the first step).
"""
import json
import os
import sys

from .song import FPS, load_song


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "check":
        song = load_song(argv[1])
        problems = song.check()
        print(f"{song.artist} - {song.title}: {song.bpm:g} bpm, {song.steps_per_beat} steps/beat, "
              f"{song.total_steps} steps, {song.total_s:.4f} s, {song.n_frames} frames @ {FPS} fps")
        for i, s in enumerate(song.segments):
            print(f"  segment {i}: film {song.seg_film_start[i]}.. (+{s.steps}) = record {s.record_start}.., "
                  f"{(s.bpm or song.bpm):g} bpm, from {song.seg_time_start[i]:.3f} s {s.label!r}")
        for g in song.gears:
            n_tall = sum(1 for v in g.strikes.values() if v != "stub")
            print(f"  gear {g.key:7s} {g.teeth:3d} teeth, {n_tall:2d} striking, side {g.side:+d}, x {g.x:6.0f}, ranges {g.ranges}")
        for sl in song.slips:
            cyc, per_lap, laps, k_ref = song.slip_params(sl.k_from)
            print(f"  slip: cycle {cyc} vs beat {sl.beat_steps}: {per_lap} per lap, home every {laps} laps from step {k_ref}"
                  + (f" (steps {sl.k_from}..{sl.k_to})" if sl.k_to is not None else ""))
        if song.slips:
            print(f"  homes {song.homes}")
        for m in song.moments:
            print(f"  moment {m.step:4d} {m.kind:8s} {m.title}")
        if problems:
            print("PROBLEMS:\n  " + "\n  ".join(problems))
            return 1
        print("ok")
        return 0
    if cmd == "still":
        from .render import Assets, render_frame
        song = load_song(argv[1])
        seconds = float(argv[2])
        out = argv[3]
        f = int(round(seconds * FPS))
        A = Assets(song)
        render_frame(A, f).save(out)
        print(f"wrote {out}: frame {f}, film step {song.k_at(f):.2f}")
        return 0
    if cmd == "render":
        from .build import render_film
        workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 4
        crf = int(argv[argv.index("--crf") + 1]) if "--crf" in argv else 28
        render_film(argv[1], workers, crf)
        return 0
    if cmd == "export-record-map":
        from .build import out_dir, write_record_map
        song = load_song(argv[1])
        out = argv[2] if len(argv) > 2 else os.path.join(out_dir(song), "record_map.json")
        write_record_map(song, out)
        print(f"wrote {out}")
        print(json.dumps({k: v for k, v in song.record_map().items() if k != "anchors"}, indent=1))
        return 0
    print(f"unknown command {cmd!r}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

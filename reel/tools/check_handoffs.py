#!/usr/bin/env python3
"""Lossless, dust-free exactness check of every beat boundary (and the loop seam).

    /home/user/.venvs/manim/bin/python reel/tools/check_handoffs.py                 # 30 + 60 fps, all 9 seams
    /home/user/.venvs/manim/bin/python reel/tools/check_handoffs.py --fps 60
    /home/user/.venvs/manim/bin/python reel/tools/check_handoffs.py --beats b03_cpu b04_gpu   # seams touching these
    /home/user/.venvs/manim/bin/python reel/tools/check_handoffs.py --reuse          # compare the last capture again
    ... check_handoffs.py --save-baseline DIR      # copy this run's captures to DIR
    ... check_handoffs.py --against DIR            # also report which boundary frames CHANGED vs DIR

For every beat (and fps) it runs the beat's FULL logic headlessly (every updater on every frame,
exactly as a render would) but rasterises only frame 0 and the last 4 frames, straight from the
Cairo frame buffer (no video encode, no grade). The dust field is kept (its time-based updater is
what makes waits render frame by frame) but drawn invisible, so frames compare bit for bit.

Checks (all lossless, tolerance 0 unless --tol):
  seam    last frame of beat N == frame 0 of beat N+1, for the 8 cuts and b09 -> b01 (the loop)
  canon   both sides == handoff_state(<name>) rendered on its own (says WHICH side drifted)
  frames  the beat renders exactly round(duration * fps) frames
  hold    the last >= 3 frames at 60 fps are static (b09 excluded: its last frame IS the hop
          onto idx 32 == b01 frame 0); at 30 fps the static count is reported only

Captures go to reel/build/handoffs/<fps>/ (first.png, last0.png = final frame, last1..3.png);
for every failing comparison a sheet A | B | |A-B| x8 is written next to them.
Exit status 1 if any check fails. Beats run in parallel subprocesses (--jobs), each with its
own media dir (manim's text SVG cache is not safe to share between processes).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REEL))
PYTHON = os.environ.get("REEL_PYTHON", "/home/user/.venvs/manim/bin/python")
OUT_ROOT = REEL / "build" / "handoffs"
N_LAST = 4

import timeline  # noqa: E402

BEATS = [(cls, Path(mod).stem, dur) for cls, mod, dur in timeline.BEATS]
STEMS = [s for _, s, _ in BEATS]


def _entry_exit():
    """(BEAT_ENTRY, BEAT_EXIT) without importing manim in the parent process."""
    out = subprocess.run([PYTHON, "-c", "import json, motif; print(json.dumps([motif.BEAT_ENTRY, motif.BEAT_EXIT]))"],
                         capture_output=True, text=True, cwd=REEL, env=dict(os.environ, PYTHONPATH=str(REEL)))
    if out.returncode != 0:
        raise SystemExit(f"cannot import motif:\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------
# worker: one beat (or one canonical handoff state) at one fps
# --------------------------------------------------------------------------
def _worker(kind: str, name: str, fps: int, w: int, h: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    media = Path(tempfile.mkdtemp(prefix=f"hc_{name}_{fps}_"))
    from manim import config, tempconfig

    with tempconfig({"pixel_width": w, "pixel_height": h, "frame_rate": fps, "write_to_movie": False,
                     "disable_caching": True, "media_dir": str(media), "verbosity": "WARNING",
                     "progress_bar": "none", "frame_height": 16.0, "frame_width": 9.0}):
        from manim.renderer.cairo_renderer import CairoRenderer
        from manim.scene.scene_file_writer import SceneFileWriter
        import numpy as np
        from PIL import Image

        SceneFileWriter.finish = lambda self: None
        SceneFileWriter.combine_to_movie = lambda self: None
        SceneFileWriter.combine_to_section_videos = lambda self: None
        SceneFileWriter.save_image = lambda self, image: None
        import style
        style.CUE_DIR = media / "cues"          # never overwrite reel/build/cues

        if kind == "canon":
            import motif
            from manim import Scene

            class Canon(Scene):
                def construct(self):
                    self.add(motif.handoff_state(name))

            sc = Canon()
            sc.render()
            frame = sc.renderer.get_frame()
            Image.fromarray(frame[..., :3].copy()).save(out_dir / f"canon_{name}.png")
            return

        cls_name, dur = next((c, d) for c, s, d in BEATS if s == name)
        n_exp = int(round(dur * fps))
        state = {"i": 0}
        frames: dict[int, np.ndarray] = {}

        def render(self, scene, time, moving_mobjects=None):
            i = state["i"]
            state["i"] += 1
            if i == 0 or i >= n_exp - N_LAST:
                self.update_frame(scene, moving_mobjects)
                frames[i] = self.get_frame()[..., :3].copy()

        CairoRenderer.render = render
        mod = importlib.import_module(name)
        base = getattr(mod, cls_name)

        def setup(self):
            base.setup(self)
            dust = getattr(self, "dust", None)
            if dust is not None:
                for d in dust.submobjects:
                    d.set_fill(opacity=0)
                    d.set_stroke(opacity=0)

        sub = type(cls_name, (base,), {"setup": setup})
        sub().render()
        total = state["i"]
        for i, arr in frames.items():
            tag = "first" if i == 0 else f"last{total - 1 - i}"
            Image.fromarray(arr).save(out_dir / f"{name}_{tag}.png")
        (out_dir / f"{name}.json").write_text(json.dumps({"frames": total, "expected": n_exp}))


# --------------------------------------------------------------------------
# parent: run workers, compare
# --------------------------------------------------------------------------
def _run_worker(kind, name, fps, w, h, out_dir, log_dir) -> tuple[str, int, str]:
    cmd = [PYTHON, str(Path(__file__).resolve()), "--worker", kind, name, str(fps), str(w), str(h), str(out_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REEL, env=dict(os.environ, PYTHONPATH=str(REEL)))
    log = log_dir / f"{kind}_{name}_{fps}.log"
    log.write_text(res.stdout + res.stderr)
    return name, res.returncode, str(log)


def _load(p: Path):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB")).astype(np.int16)


def _cmp(a, b, tol: int):
    import numpy as np
    d = np.abs(a - b).max(axis=2)
    bad = d > tol
    n = int(bad.sum())
    bbox = None
    if n:
        ys, xs = np.nonzero(bad)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return int(d.max()), n, bbox


def _sheet(a, b, path: Path) -> None:
    import numpy as np
    from PIL import Image
    d = np.clip(np.abs(a - b) * 8, 0, 255)
    sep = np.full((a.shape[0], 4, 3), 128)
    Image.fromarray(np.concatenate([a, sep, b, sep, d], axis=1).astype(np.uint8)).save(path)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        kind, name, fps, w, h, out = sys.argv[2:8]
        _worker(kind, name, int(fps), int(w), int(h), Path(out))
        return 0

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, nargs="+", default=[30, 60])
    ap.add_argument("--res", default="540x960", help="raster size WxH (logic is resolution independent)")
    ap.add_argument("--beats", nargs="+", default=None, help="only seams touching these beats (stems or classes)")
    ap.add_argument("--jobs", type=int, default=max(1, min(9, (os.cpu_count() or 2))))
    ap.add_argument("--tol", type=int, default=0, help="per-channel tolerance (0 = bit exact)")
    ap.add_argument("--reuse", action="store_true", help="re-compare existing captures without rendering")
    ap.add_argument("--no-canon", action="store_true", help="skip the canonical handoff_state comparison")
    ap.add_argument("--against", type=Path, default=None,
                    help="a previous capture root (e.g. a --save-baseline copy): report which boundary "
                         "frames changed (informational, never a failure)")
    ap.add_argument("--save-baseline", type=Path, default=None, help="copy this run's captures to DIR")
    a = ap.parse_args()
    w, h = (int(v) for v in a.res.lower().split("x"))

    # seams: (A stem, B stem, handoff name)
    entry, exit_ = _entry_exit()
    seams = []
    for k, (cls, stem, _) in enumerate(BEATS):
        ncls, nstem, _ = BEATS[(k + 1) % len(BEATS)]
        assert exit_[cls] == entry[ncls] or {exit_[cls], entry[ncls]} == {"LOOP"}, (cls, ncls)
        seams.append((stem, nstem, exit_[cls]))
    if a.beats:
        want = {s for s in STEMS for b in a.beats if b in (s, s.split("_", 1)[0])}
        want |= {s for c, s, _ in BEATS if c in a.beats}
        seams = [s for s in seams if s[0] in want or s[1] in want]
    stems = sorted({s for sm in seams for s in sm[:2]}, key=STEMS.index)
    names = sorted({sm[2] for sm in seams})

    ok = True
    for fps in a.fps:
        out_dir = OUT_ROOT / str(fps)
        log_dir = out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if not a.reuse:
            jobs = [("beat", s) for s in stems] + ([] if a.no_canon else [("canon", n) for n in names])
            # slowest first (b04 / b08 / b06 dominate)
            order = {"b04_gpu": 0, "b08_amdahl": 1, "b06_datacenter": 2, "b07_planet": 3}
            jobs.sort(key=lambda j: order.get(j[1], 9))
            print(f"[{fps} fps] capturing {len(jobs)} jobs ({a.jobs} parallel) -> {out_dir}", flush=True)
            with ThreadPoolExecutor(a.jobs) as ex:
                for name, rc, log in ex.map(lambda j: _run_worker(j[0], j[1], fps, w, h, out_dir, log_dir), jobs):
                    if rc != 0:
                        ok = False
                        print(f"  FAIL  {name}: worker exited {rc}, see {log}")
                        print("        " + "\n        ".join(Path(log).read_text().strip().splitlines()[-6:]))

        print(f"[{fps} fps] results")
        for stem in stems:
            info_p = out_dir / f"{stem}.json"
            if not info_p.exists():
                ok = False
                print(f"  FAIL  frames  {stem}: no capture")
                continue
            info = json.loads(info_p.read_text())
            if info["frames"] != info["expected"]:
                ok = False
                print(f"  FAIL  frames  {stem}: {info['frames']} frames, expected {info['expected']}")
            last = [_load(out_dir / f"{stem}_last{k}.png") for k in range(N_LAST)
                    if (out_dir / f"{stem}_last{k}.png").exists()]
            static = 1
            for k in range(1, len(last)):
                if _cmp(last[k], last[0], a.tol)[1]:
                    break
                static += 1
            exempt = stem == STEMS[-1]
            need = 3 if fps >= 60 and not exempt else 0
            flag = "ok  " if static >= need else "FAIL"
            if static < need:
                ok = False
            print(f"  {flag}  hold    {stem}: {static}{'+' if static == len(last) else ''} static exit frame(s)"
                  f"{' (exempt: the loop hop)' if exempt else '' if need else ' (reported only)'}")

        for sa, sb, name in seams:
            pa, pb = out_dir / f"{sa}_last0.png", out_dir / f"{sb}_first.png"
            if not (pa.exists() and pb.exists()):
                ok = False
                print(f"  FAIL  seam    {sa} -> {sb} ({name}): missing capture")
                continue
            A, B = _load(pa), _load(pb)
            mx, n, bbox = _cmp(A, B, a.tol)
            label = f"{sa} -> {sb} ({name})"
            if n:
                ok = False
                _sheet(A, B, out_dir / f"seam_{sa}__{sb}.png")
                print(f"  FAIL  seam    {label}: max diff {mx}, {n} px > tol, bbox {bbox} "
                      f"-> {out_dir / f'seam_{sa}__{sb}.png'}")
            else:
                print(f"  ok    seam    {label}: bit exact" + (f" (max {mx} <= tol)" if mx else ""))
            pc = out_dir / f"canon_{name}.png"
            if a.no_canon or not pc.exists():
                continue
            C = _load(pc)
            for side, X, tag in (("exit ", A, sa), ("entry", B, sb)):
                mx, n, bbox = _cmp(X, C, a.tol)
                if n:
                    ok = False
                    _sheet(X, C, out_dir / f"canon_{tag}_{name}.png")
                    print(f"  FAIL  canon   {side} {tag} vs handoff_state('{name}'): max diff {mx}, {n} px, bbox {bbox}")
                else:
                    print(f"  ok    canon   {side} {tag} == handoff_state('{name}')")
        if a.against is not None:
            ref_dir = a.against / str(fps)
            print(f"[{fps} fps] boundary frames vs {ref_dir} (informational)")
            for sa, sb, name in seams:
                for p in (out_dir / f"{sa}_last0.png", out_dir / f"{sb}_first.png"):
                    q = ref_dir / p.name
                    if not (p.exists() and q.exists()):
                        print(f"  ?     {p.name}: no reference")
                        continue
                    mx, n, bbox = _cmp(_load(p), _load(q), 0)
                    print(f"  {'same ' if not n else 'CHANGED'} {p.name}" + (f": {n} px, max {mx}, bbox {bbox}" if n else ""))
        if a.save_baseline is not None:
            import shutil
            dst = a.save_baseline / str(fps)
            dst.mkdir(parents=True, exist_ok=True)
            for f in out_dir.glob("*.png"):
                if not f.name.startswith(("seam_", "canon_b")):
                    shutil.copy2(f, dst / f.name)
            for f in out_dir.glob("*.json"):
                shutil.copy2(f, dst / f.name)
            print(f"[{fps} fps] captures saved to {dst}")
    print("ALL OK" if ok else "FAILURES (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

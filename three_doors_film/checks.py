#!/usr/bin/env python3
"""Three Doors: the definition-of-done checks on the rendered MP4.

    python checks.py [out/three_doors_final.mp4] [--preview] [--story build/story.json]
                     [--wav out/soundtrack.wav] [--post POST.md] [--report out/checks_report.json]

Checks (every one re-measured on the file, nothing taken from the render's logs):
  1. picture: H.264, 1080 x 1920, 60 fps, exactly 2100 frames (counted with ffprobe -count_frames), yuv420p;
     with --preview: 540 x 960, 30 fps, 1050 frames;
  2. sound: AAC stereo at 48 kHz; integrated loudness within -16 ... -14 LUFS and true peak <= -1.0 dBTP,
     measured by ffmpeg's ebur128 on the MP4's own audio (soundtrack.measure);
  3. the loop: PSNR >= 45 dB between the MP4's last and first decoded frames (layout(2099) == layout(0)
     before the encode; the CRF 18 encode with temporal grain keeps them within a few dB of each other);
  4. the phone check: the MP4's audio decoded to PCM, high-passed at 300 Hz, every sound cue of
     build/story.json peaking above -30 dBFS within 50 ms of its frame (soundtrack.phone_check); the mastered
     out/soundtrack.wav is checked the same way when it exists;
  5. the answer stays off the post texts: POST.md (when it exists) is searched for the answer ("Switch." as a
     sentence, "switching wins", "2 in 3", "two in three", "two thirds", "twice as often", "always switch",
     "should switch") in every section except the alt text; "Switch or stay?" is the question and is allowed;
     a missing POST.md is reported and skipped;
  6. in final mode: the silent copy has the same frame count and no audio stream, and the cover is a
     1080 x 1920 JPEG.

Standard library plus numpy; imports measure() and phone_check() from soundtrack.py beside this file.
Prints a PASS / FAIL line per check, writes out/checks_report.json, exits 1 if anything failed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from soundtrack import measure, phone_check  # noqa: E402  (numpy only; ffmpeg ebur128 for the loudness)

LUFS_RANGE = (-16.0, -14.0)
TRUE_PEAK_MAX = -1.0
LOOP_PSNR_MIN = 45.0
ANSWER_PATTERNS = [re.compile(r"(?<!\w)switch\.", re.IGNORECASE), re.compile(r"switching wins", re.IGNORECASE),
                   re.compile(r"(?<!\w)2 in 3(?!\d)"), re.compile(r"two in three", re.IGNORECASE), re.compile(r"two.thirds", re.IGNORECASE),
                   re.compile(r"twice as often", re.IGNORECASE), re.compile(r"always switch", re.IGNORECASE), re.compile(r"should switch", re.IGNORECASE)]
ALT_TEXT_HEADING = re.compile(r"alt[\s-]*text", re.IGNORECASE)


class Report:
    def __init__(self):
        self.results = []
        self.notes = []

    def check(self, name: str, ok: bool, detail: str, **data):
        self.results.append({"name": name, "ok": bool(ok), "detail": detail, **data})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    def note(self, text: str):
        self.notes.append(text)
        print(f"note  {text}")

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.results)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, check=False, **kw)


def ffprobe(path: str, count_frames: bool = True) -> dict:
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,"
           "sample_rate,channels:format=duration,format_name", "-of", "json"]
    if count_frames:
        cmd.insert(3, "-count_frames")
    cmd.append(path)
    p = run(cmd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {p.stderr.strip()}")
    return json.loads(p.stdout)


def video_stream(info: dict) -> dict | None:
    return next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)


def audio_stream(info: dict) -> dict | None:
    return next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)


def decode_frames(path: str, indices: list[int], width: int, height: int) -> list[np.ndarray]:
    """Decode the given frames (by index) of a video to RGB arrays, in index order."""
    expr = "+".join(f"eq(n\\,{i})" for i in sorted(indices))
    p = run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='{expr}'", "-fps_mode", "passthrough",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode frames {indices} of {path}: {p.stderr.decode(errors='replace')[-800:]}")
    n = width * height * 3
    if len(p.stdout) != n * len(indices):
        raise RuntimeError(f"expected {len(indices)} frames of {n} bytes, got {len(p.stdout)} bytes")
    return [np.frombuffer(p.stdout[i * n:(i + 1) * n], dtype=np.uint8).reshape(height, width, 3) for i in range(len(indices))]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def extract_audio(path: str, wav_path: str) -> None:
    p = run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-map", "0:a:0", "-c:a", "pcm_s24le", "-ar", "48000", wav_path])
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg could not extract the audio of {path}: {p.stderr.decode(errors='replace')[-800:]}")


def post_text_hits(post_path: str) -> tuple[list[dict], int, int]:
    """Search POST.md for the answer. Sections whose heading names the alt text are skipped, as are lines that
    start with an 'Alt text' label. Returns (hits, lines searched, lines skipped)."""
    hits, searched, skipped = [], 0, 0
    in_alt = False
    with open(post_path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            heading = re.match(r"^\s{0,3}#{1,6}\s+(.*)", line)
            if heading:
                in_alt = bool(ALT_TEXT_HEADING.search(heading.group(1)))
            if in_alt or re.match(r"^\s*(\*\*|_)?alt[\s-]*text", line, re.IGNORECASE):
                skipped += 1
                continue
            searched += 1
            for pat in ANSWER_PATTERNS:
                for m in pat.finditer(line):
                    hits.append({"line": ln, "match": m.group(0), "text": line.rstrip()[:160]})
    return hits, searched, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mp4", nargs="?", default=os.path.join(HERE, "out", "three_doors_final.mp4"))
    ap.add_argument("--preview", action="store_true", help="expect the preview's size and rate (540 x 960, 30 fps, 1050 frames)")
    ap.add_argument("--story", default=os.path.join(HERE, "build", "story.json"))
    ap.add_argument("--wav", default=os.path.join(HERE, "out", "soundtrack.wav"), help="the mastered soundtrack (checked too when present)")
    ap.add_argument("--post", default=os.path.join(HERE, "POST.md"))
    ap.add_argument("--report", default=os.path.join(HERE, "out", "checks_report.json"))
    args = ap.parse_args(argv)

    rep = Report()
    with open(args.story, encoding="utf-8") as fh:
        story = json.load(fh)
    meta = story["meta"]
    if args.preview:
        want_w, want_h, want_fps = meta["width"] // 2, meta["height"] // 2, Fraction(meta["fps"], 2)
        want_frames = meta["frames"] // 2
    else:
        want_w, want_h, want_fps, want_frames = meta["width"], meta["height"], Fraction(meta["fps"]), meta["frames"]
    mp4 = args.mp4
    print(f"checks: {mp4} ({'preview' if args.preview else 'final'}; story {args.story})")
    if not os.path.isfile(mp4):
        rep.check("file", False, f"{mp4} does not exist")
        return finish(rep, args)

    # 1. picture
    info = ffprobe(mp4)
    v = video_stream(info)
    if v is None:
        rep.check("video stream", False, "no video stream")
    else:
        w, h = int(v.get("width", 0)), int(v.get("height", 0))
        rate = Fraction(v.get("r_frame_rate", "0/1"))
        n = int(v.get("nb_read_frames", 0))
        rep.check("size", (w, h) == (want_w, want_h), f"{w} x {h} (want {want_w} x {want_h})", width=w, height=h)
        rep.check("frame rate", rate == want_fps, f"{rate} fps (want {want_fps})", r_frame_rate=str(rate), avg_frame_rate=v.get("avg_frame_rate"))
        rep.check("frame count", n == want_frames, f"{n} frames counted (want exactly {want_frames})", frames=n)
        rep.check("codec", v.get("codec_name") == "h264" and v.get("pix_fmt") == "yuv420p",
                  f"{v.get('codec_name')} {v.get('pix_fmt')} (want h264 yuv420p)", codec=v.get("codec_name"), pix_fmt=v.get("pix_fmt"))
        dur = float(info.get("format", {}).get("duration", 0))
        rep.note(f"container duration {dur:.3f} s; the picture is {n / float(want_fps):.3f} s")

    # 2. sound
    a = audio_stream(info)
    if a is None:
        rep.check("audio stream", False, "no audio stream")
    else:
        rep.check("audio format", a.get("codec_name") == "aac" and int(a.get("sample_rate", 0)) == 48000 and int(a.get("channels", 0)) == 2,
                  f"{a.get('codec_name')} {a.get('sample_rate')} Hz {a.get('channels')} ch (want aac 48000 Hz 2 ch)")
        lufs, tp = measure(mp4)
        rep.check("loudness", LUFS_RANGE[0] <= lufs <= LUFS_RANGE[1], f"{lufs:.2f} LUFS integrated (want {LUFS_RANGE[0]} ... {LUFS_RANGE[1]})", lufs=lufs)
        rep.check("true peak", tp <= TRUE_PEAK_MAX, f"{tp:.2f} dBTP (want <= {TRUE_PEAK_MAX})", true_peak_dbtp=tp)

    # 3. the loop
    if v is not None and n >= 2 and (w, h) == (want_w, want_h):
        first, last = decode_frames(mp4, [0, n - 1], w, h)
        p = psnr(last, first)
        rep.check("loop PSNR", p >= LOOP_PSNR_MIN, f"{p:.2f} dB between frame {n - 1} and frame 0 (want >= {LOOP_PSNR_MIN})", loop_psnr_db=None if p == math.inf else p)
        bg = tuple(int(story["meta"]["colours"]["bg"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        corner = first[0:8, 0:8].reshape(-1, 3).mean(axis=0)
        rep.note(f"frame 0 top-left corner mean RGB {tuple(round(float(c), 1) for c in corner)} (background {bg})")

    # 4. the phone check
    if a is not None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "mp4_audio.wav")
            extract_audio(mp4, wav)
            ok, misses = phone_check(wav, args.story)
            worst = ", ".join(f"frame {m['frame']} {m['cue']} {m['id']}: {m['hp300_peak_db']} dBFS" for m in misses[:6])
            rep.check("phone check (MP4 audio)", ok, f"{len(misses)} cues below -30 dBFS after a 300 Hz high-pass" + (f" ({worst})" if misses else ""),
                      misses=misses)
    if os.path.isfile(args.wav):
        ok, misses = phone_check(args.wav, args.story)
        rep.check("phone check (soundtrack.wav)", ok, f"{os.path.relpath(args.wav, HERE)}: {len(misses)} misses", misses=misses)
    else:
        rep.note(f"{args.wav} not found; the mastered WAV was not checked separately")

    # 5. the answer stays off the post texts
    if os.path.isfile(args.post):
        hits, searched, skipped = post_text_hits(args.post)
        where = "; ".join(f"line {h['line']} '{h['match']}'" for h in hits[:8])
        rep.check("post texts", not hits, f"{os.path.relpath(args.post, HERE)}: {len(hits)} occurrences of the answer outside the alt text "
                  f"({searched} lines searched, {skipped} alt-text lines skipped)" + (f": {where}" if hits else ""), hits=hits)
    else:
        rep.note(f"{os.path.relpath(args.post, HERE)} does not exist yet: the answer search over the post texts was skipped")

    # 6. the silent copy and the cover (final mode)
    if not args.preview:
        silent = os.path.splitext(mp4)[0] + "_silent.mp4"
        if os.path.isfile(silent):
            si = ffprobe(silent)
            sv, sa = video_stream(si), audio_stream(si)
            sn = int(sv.get("nb_read_frames", 0)) if sv else 0
            rep.check("silent copy", sv is not None and sa is None and sn == want_frames,
                      f"{os.path.relpath(silent, HERE)}: {sn} frames, {'no audio' if sa is None else 'HAS AUDIO'}")
        else:
            rep.check("silent copy", False, f"{os.path.relpath(silent, HERE)} does not exist")
        cover = os.path.join(os.path.dirname(mp4), "three_doors_cover_hook.jpg")
        if os.path.isfile(cover):
            ci = ffprobe(cover, count_frames=False)
            cv = video_stream(ci)
            cw, ch = (int(cv.get("width", 0)), int(cv.get("height", 0))) if cv else (0, 0)
            rep.check("cover", cv is not None and (cw, ch) == (want_w, want_h) and cv.get("codec_name") == "mjpeg",
                      f"{os.path.relpath(cover, HERE)}: {cv.get('codec_name') if cv else 'none'} {cw} x {ch}")
        else:
            rep.check("cover", False, f"{os.path.relpath(cover, HERE)} does not exist")

    return finish(rep, args)


def finish(rep: Report, args) -> int:
    failed = [r["name"] for r in rep.results if not r["ok"]]
    report = {"file": os.path.abspath(args.mp4), "mode": "preview" if args.preview else "final", "ok": rep.ok,
              "results": rep.results, "notes": rep.notes}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    if rep.ok:
        print(f"checks: all {len(rep.results)} passed -> {args.report}")
        return 0
    print(f"checks: {len(failed)} FAILED ({', '.join(failed)}) -> {args.report}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

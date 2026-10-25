"""Procedural textures for The Shuffle, in the style of ``doubling_film/assets.py``.

Each function returns ``(rgb, alpha)``: premultiplied colour HxWx3 and coverage HxW, both
floats in [0, 1], white, so a sprite tints them.  They share The Doubling's disk cache
machinery but write to this film's own ``media/texture_cache``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from doubling_film import assets  # noqa: E402

assets.CACHE = Path(__file__).with_name("media") / "texture_cache"
_cached, _grid, _splat, _window, _white, blur = (
    assets._cached, assets._grid, assets._splat, assets._window, assets._white, assets.blur)


def _norm(img, gain=1.0, pct=99.5):
    hi = np.percentile(img, pct)
    return np.clip(img / max(hi, 1e-9) * gain, 0, 1)


@_cached
def point_disc(n=640, count=1500, seed=9, clumps=16, version=1):
    """Everyone alive: a disc of points of light, clustered like settlements seen at night."""
    rng = np.random.default_rng(seed)
    k_cl = int(count * 0.62)
    centres = rng.uniform(-0.6, 0.6, (clumps, 2))
    centres = centres[np.hypot(centres[:, 0], centres[:, 1]) < 0.7]
    which = rng.integers(0, len(centres), k_cl)
    cl = centres[which] + rng.normal(0, 0.11, (k_cl, 2)) * rng.uniform(0.5, 1.4, (k_cl, 1))
    r = 0.9 * np.sqrt(rng.uniform(0, 1, count - k_cl))
    th = rng.uniform(0, 2 * np.pi, count - k_cl)
    un = np.c_[r * np.cos(th), r * np.sin(th)]
    xy = np.vstack([cl, un])
    xy = xy[np.hypot(xy[:, 0], xy[:, 1]) < 0.92]
    w = rng.uniform(0.35, 1.0, (len(xy), 1))
    img = _splat(xy, w, n, 1.0)[..., 0]
    a = _norm(blur(img, 0.9), 1.6, 99.0)
    x, y = _grid(n)
    rr = np.hypot(x, y)
    a = a * np.clip((0.97 - rr) / 0.06, 0, 1)
    a = np.clip(a + 0.05 * np.exp(-((rr / 0.9) ** 2)) * (rr < 0.93), 0, 1)
    return _white(a)


@_cached
def spiral(n=768, turns=2.4, count=14000, seed=4, version=1):
    """Time since the Big Bang: a bright point with time spiralling out from it to now."""
    rng = np.random.default_rng(seed)
    u = np.sqrt(rng.uniform(0, 1, count))  # denser near the outer turns, where the arm is wide
    th = u * 2 * np.pi * turns
    r = 0.92 * (th / (2 * np.pi * turns)) ** 0.85
    width = 0.012 + 0.045 * (r / 0.92)
    xy = np.c_[r * np.cos(th + np.pi / 2), r * np.sin(th + np.pi / 2)] + rng.normal(0, 1, (count, 2)) * width[:, None]
    w = (0.25 + 0.75 * (th / th.max()) ** 1.5)[:, None]
    img = _splat(xy, w, n, 1.0)[..., 0]
    a = _norm(blur(img, 1.3), 1.25, 99.3)
    x, y = _grid(n)
    rr = np.hypot(x, y)
    a = a + 0.9 * np.exp(-((rr / 0.045) ** 2)) + 0.35 * np.exp(-((rr / 0.14) ** 2))
    a = np.clip(a * _window(rr), 0, 1)
    return _white(a)


@_cached
def dust(n=640, count=2600, seed=6, version=1):
    """The orders: a faint dust of points, denser toward the centre."""
    rng = np.random.default_rng(seed)
    r = 0.95 * rng.uniform(0, 1, count) ** 0.7
    th = rng.uniform(0, 2 * np.pi, count)
    xy = np.c_[r * np.cos(th), r * np.sin(th)]
    w = rng.uniform(0.15, 1.0, (count, 1)) ** 2
    img = _splat(xy, w, n, 1.0)[..., 0]
    a = _norm(blur(img, 0.7), 1.3, 99.7)
    x, y = _grid(n)
    a = np.clip(a * _window(np.hypot(x, y)), 0, 1)
    return _white(a)


if __name__ == "__main__":
    for fn in (point_disc, spiral, dust):
        rgb, a = fn()
        print(f"{fn.__name__}: {a.shape}, coverage mean {a.mean():.3f}, max {a.max():.2f}")

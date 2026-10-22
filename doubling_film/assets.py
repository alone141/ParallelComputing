"""Procedural textures (numpy only).  Each function returns ``(rgb, alpha)``
float arrays: premultiplied colour HxWx3 and coverage HxW, both in [0, 1].
Results are cached on disk next to this file so re-renders start fast."""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
from pathlib import Path

import numpy as np

CACHE = Path(__file__).with_name("media") / "texture_cache"


def _cached(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        ident = repr((inspect.getsource(fn), sorted(bound.arguments.items())))
        key = hashlib.sha1(ident.encode()).hexdigest()[:16]
        path = CACHE / f"{fn.__name__}_{key}.npz"
        if path.exists():
            d = np.load(path)
            return d["rgb"].astype(np.float32), d["alpha"].astype(np.float32)
        rgb, alpha = fn(*args, **kwargs)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, rgb=rgb.astype(np.float16), alpha=alpha.astype(np.float16))
        os.replace(tmp, path)  # atomic, so parallel renders can share the cache
        return rgb.astype(np.float32), alpha.astype(np.float32)

    return wrapper


def _grid(n, extent=1.0):
    c = (np.arange(n) + 0.5) / n * 2 - 1
    x, y = np.meshgrid(c * extent, -c * extent)
    return x, y


def blur(img, sigma):
    if sigma <= 0:
        return img
    h, w = img.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    g = np.exp(-2 * (np.pi * sigma) ** 2 * (fx**2 + fy**2))
    if img.ndim == 2:
        return np.fft.irfft2(np.fft.rfft2(img) * g, s=(h, w))
    return np.stack([np.fft.irfft2(np.fft.rfft2(img[..., c]) * g, s=(h, w)) for c in range(img.shape[2])], -1)


def _splat(xy, weights, n, extent):
    """Histogram points (in [-extent, extent]^2) into an n x n image."""
    px = ((xy[:, 0] / extent + 1) / 2 * n).astype(np.int64)
    py = ((1 - (xy[:, 1] / extent + 1) / 2) * n).astype(np.int64)
    ok = (px >= 0) & (px < n) & (py >= 0) & (py < n)
    idx = py[ok] * n + px[ok]
    out = []
    for c in range(weights.shape[1]):
        out.append(np.bincount(idx, weights=weights[ok, c], minlength=n * n).reshape(n, n))
    return np.stack(out, -1)


def _window(r):
    return np.clip(1 - r**2, 0, 1) ** 2


def _white(a):
    return np.repeat(a[..., None], 3, axis=2), a


@_cached
def glow(n=256, sigma=0.32, core=0.0, core_sigma=0.05):
    x, y = _grid(n)
    r = np.hypot(x, y)
    a = np.exp(-((r / sigma) ** 2)) * 0.85 + 0.15 * np.exp(-r / (sigma * 0.6))
    if core:
        a = a + core * np.exp(-((r / core_sigma) ** 2))
    a = np.clip(a * _window(r), 0, 1)
    return _white(a)


@_cached
def star(n=256, spikes=0.35):
    """Bright core, halo and faint diffraction spikes."""
    x, y = _grid(n)
    r = np.hypot(x, y)
    a = np.exp(-((r / 0.035) ** 2)) + 0.45 * np.exp(-((r / 0.12) ** 2)) + 0.18 * np.exp(-r / 0.22)
    sp = np.exp(-np.abs(x) / 0.008 - np.abs(y) / 0.3) + np.exp(-np.abs(y) / 0.008 - np.abs(x) / 0.3)
    a = np.clip((a + spikes * sp) * _window(r), 0, 1)
    return _white(a)


@_cached
def disc(n=256, edge=0.02):
    x, y = _grid(n)
    r = np.hypot(x, y)
    a = np.clip((1 - r) / edge, 0, 1)
    return _white(a)


@_cached
def ring(n=512, r0=0.8, inner=0.03, outer=0.12):
    x, y = _grid(n)
    r = np.hypot(x, y)
    d = r - r0
    a = np.where(d < 0, np.exp(-((d / inner) ** 2)), np.exp(-((d / outer) ** 2)))
    a = np.clip(a * _window(r), 0, 1)
    return _white(a)


@_cached
def sphere_shade(n=512, lx=-0.55, ly=0.55, strength=0.88, color=(0.004, 0.012, 0.035)):
    """Dark overlay for the night side of a sphere ("over" blend)."""
    x, y = _grid(n)
    r2 = x**2 + y**2
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    lz = np.sqrt(max(1 - lx**2 - ly**2, 0))
    lam = x * lx + y * ly + z * lz
    a = np.clip((0.25 - lam) / 0.75, 0, 1) ** 1.15 * strength
    a = a * np.clip((1 - np.sqrt(r2)) / 0.006, 0, 1)
    rgb = np.ones((n, n, 3)) * np.array(color)[None, None, :] * a[..., None]
    return rgb, a


@_cached
def rim_light(n=512, lx=-0.55, ly=0.55, width=0.08):
    """Thin bright crescent on the lit edge of a sphere ("screen" blend)."""
    x, y = _grid(n)
    r = np.hypot(x, y)
    edge = np.exp(-(((r - 0.985) / width) ** 2)) * (r < 1.02)
    facing = np.clip((x * lx + y * ly) / (r + 1e-6), 0, 1)
    a = np.clip(edge * facing**1.5, 0, 1)
    return _white(a)


@_cached
def electron_cloud(n=900, a0=0.21, seed=3):
    rng = np.random.default_rng(seed)
    N = 260_000
    r = rng.gamma(3.0, a0 / 2, N)
    v = rng.normal(size=(N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    xy = v[:, :2] * r[:, None]
    hist = _splat(xy, np.ones((N, 1)), n, 1.0)[..., 0]
    speck = blur(hist, 1.1)
    smoothd = blur(hist, 14)
    speck /= np.percentile(speck, 99.9)
    smoothd /= smoothd.max()
    x, y = _grid(n)
    rr = np.hypot(x, y)
    inten = np.clip(0.75 * smoothd**0.8 + 0.35 * speck * smoothd**0.3, 0, None)
    inten = 1 - np.exp(-2.4 * inten)
    inten *= _window(rr)
    inner = np.array([0.85, 0.96, 1.0])
    outer = np.array([0.18, 0.55, 1.0])
    t = np.clip(inten * 1.4, 0, 1)[..., None]
    col = outer + (inner - outer) * t
    return col * inten[..., None], inten


@_cached
def sun_disc(n=1024, seed=5):
    rng = np.random.default_rng(seed)
    x, y = _grid(n)
    r = np.hypot(x, y)
    mu = np.sqrt(np.clip(1 - r**2, 0, 1))
    limb = 0.38 + 0.62 * mu**0.55
    noise = rng.normal(size=(n, n))
    gran = blur(noise, 1.4) - blur(noise, 4.5)
    gran /= gran.std() + 1e-9
    centre = np.array([1.0, 0.96, 0.82])
    edge = np.array([1.0, 0.52, 0.12])
    t = np.clip((1 - mu) ** 0.7, 0, 1)[..., None]
    col = centre + (edge - centre) * t
    inten = limb * (1 + 0.07 * gran)
    for sx, sy, s in [(0.25, 0.18, 0.025), (0.31, 0.12, 0.016), (-0.35, -0.22, 0.02), (-0.05, 0.42, 0.012)]:
        d2 = (x - sx) ** 2 + (y - sy) ** 2
        inten *= 1 - 0.75 * np.exp(-d2 / (s * s)) - 0.25 * np.exp(-d2 / (2.6 * s) ** 2)
    a = np.clip((1 - r) / 0.004, 0, 1)
    rgb = np.clip(col * inten[..., None], 0, 1) * a[..., None]
    return rgb, a


def _arm(rng, m, phase, k, r0, width0, width1, rscale=0.26, rmax=1.0):
    """Points scattered around a logarithmic spiral arm (normal offsets)."""
    r = r0 + rng.gamma(2.0, rscale, m * 3)
    r = r[r < rmax][:m]
    th = phase + k * np.log(r / r0)
    tx, ty = np.cos(th) - k * np.sin(th), np.sin(th) + k * np.cos(th)
    tn = np.hypot(tx, ty)
    nx, ny = -ty / tn, tx / tn
    off = rng.normal(0, 1, len(r)) * (width0 + width1 * r)
    xy = np.stack([r * np.cos(th) + nx * off, r * np.sin(th) + ny * off], 1)
    return xy, r, np.stack([nx, ny], 1)


def _clumps(rng, centres, n_min, n_max, s_min, s_max):
    counts = rng.integers(n_min, n_max, len(centres))
    sig = rng.uniform(s_min, s_max, len(centres))
    c = np.repeat(centres, counts, axis=0)
    s = np.repeat(sig, counts)[:, None]
    return c + rng.normal(0, 1, (len(c), 2)) * s


@_cached
def galaxy(n=1600, extent=1.15, seed=11, sun_angle_deg=-118.0, sun_r=0.534, version=2):
    """A barred spiral seen face-on, radius 1 = 50,000 light-years."""
    rng = np.random.default_rng(seed)
    k = 1 / np.tan(np.radians(13.5))
    r0 = 0.12
    bar_angle = np.radians(25)
    pts, cols = [], []

    def add(xy, colour, w):
        pts.append(xy)
        cols.append(np.tile(np.asarray(colour, float) * w, (len(xy), 1)))

    add(rng.normal(0, 0.06, (70_000, 2)), (1.0, 0.8, 0.52), 1.0)
    bar = rng.normal(0, 1, (50_000, 2)) * [0.12, 0.035]
    c, s = np.cos(bar_angle), np.sin(bar_angle)
    add(bar @ np.array([[c, s], [-s, c]]), (1.0, 0.84, 0.6), 0.9)
    rd = rng.exponential(0.3, 160_000)
    rd = rd[rd < 1.05]
    td = rng.uniform(0, 2 * np.pi, len(rd))
    add(np.stack([rd * np.cos(td), rd * np.sin(td)], 1), (0.95, 0.86, 0.75), 0.32)
    dust = []
    for ph, w in [(0, 1.0), (np.pi, 1.0), (np.pi / 2, 0.5), (3 * np.pi / 2, 0.5)]:
        phase = bar_angle + ph
        xy, r, nrm = _arm(rng, int(160_000 * w), phase, k, r0, 0.018, 0.05)
        add(xy, (0.62, 0.76, 1.0), 0.8 * w)
        cen, _, _ = _arm(rng, int(1300 * w), phase, k, r0, 0.01, 0.035)
        add(_clumps(rng, cen, 6, 50, 0.004, 0.014), (0.72, 0.84, 1.0), 1.5)
        hii, _, _ = _arm(rng, int(260 * w), phase, k, r0, 0.008, 0.03)
        add(_clumps(rng, hii, 6, 30, 0.003, 0.008), (1.0, 0.38, 0.62), 2.6)
        if w == 1.0:
            dxy, _, dn = _arm(rng, 70_000, phase, k, r0 * 1.15, 0.006, 0.012)
            dust.append(dxy - dn * 0.03)
    # the Orion spur, where the Sun lives
    sa = np.radians(sun_angle_deg)
    t = rng.normal(0, 0.4, 20_000)
    rs = sun_r * np.exp(t / k)
    ths = sa + t
    spur = np.stack([rs * np.cos(ths), rs * np.sin(ths)], 1) + rng.normal(0, 0.02, (len(t), 2))
    add(spur, (0.7, 0.82, 1.0), 0.7)
    add(_clumps(rng, spur[::120], 5, 25, 0.004, 0.01), (0.75, 0.86, 1.0), 1.4)

    xy = np.concatenate(pts)
    w = np.concatenate(cols)
    img = _splat(xy, w, n, extent)
    lum = blur(img, 0.8) + blur(img, 4.0) * 0.7 + blur(img, 30.0) * 0.55
    d = _splat(np.concatenate(dust), np.ones((sum(len(x) for x in dust), 1)), n, extent)[..., 0]
    d = blur(d, 2.5)
    d /= np.percentile(d, 99.5) + 1e-9
    lum *= np.exp(-0.7 * np.clip(d, 0, 2))[..., None]
    ref = np.percentile(lum.max(axis=2), 99.7)
    lum = 1 - np.exp(-1.9 * lum / ref)
    x, y = _grid(n, extent)
    lum *= _window(np.hypot(x, y) / extent)[..., None]
    return np.clip(lum, 0, 1), np.clip(lum.max(axis=2), 0, 1)


@_cached
def cosmic_web(n=1600, extent=1.1, seed=21, version=2):
    rng = np.random.default_rng(seed)
    nodes = []
    while len(nodes) < 330:
        p = rng.uniform(-1, 1, 2)
        if np.hypot(*p) < 0.99 and all(np.hypot(*(p - q)) > 0.055 for q in nodes):
            nodes.append(p)
    nodes = np.array(nodes)
    mass = rng.lognormal(0, 0.8, len(nodes))
    pts, cols = [], []
    edges = set()
    for i, p in enumerate(nodes):
        d = np.hypot(*(nodes - p).T)
        for j in np.argsort(d)[1:4]:
            edges.add((min(i, j), max(i, j)))
    for i, j in edges:
        a, b = nodes[i], nodes[j]
        L = np.hypot(*(b - a))
        bright = np.sqrt(mass[i] * mass[j])
        m = int(5000 * L * min(bright, 3))
        if m < 10:
            continue
        t = rng.uniform(0, 1, m)
        perp = np.array([-(b - a)[1], (b - a)[0]]) / (L + 1e-9)
        bend = rng.normal(0, 0.03)
        width = rng.uniform(0.003, 0.009)
        xy = a + np.outer(t, b - a) + np.outer(np.sin(np.pi * t) * bend + rng.normal(0, width, m), perp)
        pts.append(xy)
        cols.append(np.tile([0.6, 0.42, 1.0], (m, 1)) * min(bright, 2.5) ** 0.5)
        g = xy[rng.integers(0, m, max(m // 40, 1))]
        pts.append(_clumps(rng, g, 3, 14, 0.001, 0.003))
        cols.append(np.tile([0.95, 0.8, 1.0], (len(pts[-1]), 1)))
    for p, ms in zip(nodes, mass):
        m = int(700 * ms)
        pts.append(p + rng.normal(0, 0.005 * np.sqrt(ms), (m, 2)))
        cols.append(np.tile([1.0, 0.88, 0.92], (m, 1)))
    xy = np.concatenate(pts)
    w = np.concatenate(cols)
    img = _splat(xy, w, n, extent)
    lum = blur(img, 0.8) * 0.9 + blur(img, 4) * 1.0 + blur(img, 22) * 1.6
    ref = np.percentile(lum.max(axis=2), 99.7)
    lum = 1 - np.exp(-1.4 * lum / ref)
    x, y = _grid(n, extent)
    r = np.hypot(x, y)
    walls = np.clip(blur(rng.normal(size=(n, n)), 40), 0, None)
    walls /= walls.max() + 1e-9
    lum += (walls * 0.10)[..., None] * np.array([0.45, 0.3, 0.9])
    lum *= np.clip((1.0 - r) / 0.06, 0, 1)[..., None]
    # the cosmic microwave background: the edge of what we can see
    edge = np.exp(-(((r - 1.0) / 0.012) ** 2))[..., None] * np.array([1.0, 0.58, 0.48]) * 0.5
    edge += np.exp(-(((r - 1.0) / 0.06) ** 2))[..., None] * np.array([0.85, 0.3, 0.6]) * 0.16
    lum = np.clip(lum + edge, 0, 1)
    return lum, np.clip(lum.max(axis=2), 0, 1)


@_cached
def belts(n=1024, tilt=0.42, seed=8):
    """Asteroid and Kuiper belts for the solar system (radius 1 = Neptune)."""
    rng = np.random.default_rng(seed)
    pts, cols = [], []
    for (r0, r1, m, colr) in [(2.2 / 30.07, 3.3 / 30.07, 2500, (0.55, 0.5, 0.45)),
                              (30.5 / 30.07, 48 / 30.07, 16000, (0.6, 0.75, 1.0))]:
        r = rng.uniform(r0, r1, m)
        th = rng.uniform(0, 2 * np.pi, m)
        pts.append(np.stack([r * np.cos(th), r * np.sin(th) * tilt], 1))
        cols.append(np.tile(colr, (m, 1)))
    img = _splat(np.concatenate(pts), np.concatenate(cols), n, 1.6)
    lum = blur(img, 0.8) * 1.4 + blur(img, 6) * 1.5
    lum = 1 - np.exp(-lum / (np.percentile(lum.max(axis=2), 99.8) + 1e-9))
    return np.clip(lum, 0, 1), np.clip(lum.max(axis=2), 0, 1)

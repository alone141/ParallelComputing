"""Build every Lumiera brand asset into ./assets.

    python build_brand.py            # uses copy.json for the tagline and highlight names
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from brandlib import (
    PALETTE,
    Light,
    _aa,
    draw_mark,
    draw_tracked,
    font,
    rgb,
    space_background,
    to_image,
    tracked_text_size,
    wordmark_size,
)

HERE = Path(__file__).parent
OUT = HERE / "assets"
FILM = HERE.parent / "doubling_film" / "out" / "the_doubling_final.mp4"
COPY = json.loads((HERE / "copy.json").read_text()) if (HERE / "copy.json").exists() else {}
TAGLINE = COPY.get("tagline", "Science, visualized.")
HIGHLIGHTS = COPY.get("highlights", ["Start here", "Space", "Physics", "Life", "Math"])


def save(img: Image.Image, rel: str, **kw):
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, **kw)
    return path


def paste_light(base: Image.Image, light: Light, xy=(0, 0)):
    im = light.image()
    base.paste(im, xy, im)
    return base


# --------------------------------------------------------------------------- logo
def mark_image(size: int, on_dark: bool, mono: str | None = None) -> Image.Image:
    L = Light(size, size)
    draw_mark(L, size / 2, size / 2, size * 0.34, mono=mono, glow=0.0 if mono else 1.0)
    L.fade_edges(size / 2, size / 2, size * 0.40, size * 0.5)
    if not on_dark:
        return L.image()
    base = to_image(space_background(size, size, stars=int(60 * size / 1080)))
    return paste_light(base, L)


def wordmark_image(cap_px: int, color: str, pad: int = 0) -> Image.Image:
    f, tracking = wordmark_size(cap_px)
    w, top, h = tracked_text_size("LUMIERA", f, tracking)
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    draw_tracked(img, (img.width / 2, img.height / 2), "LUMIERA", f, color, tracking)
    return img


def lockup(height: int, stacked: bool, on_dark: bool) -> Image.Image:
    """Mark + wordmark.  ``height`` is the mark's outer diameter in px."""
    mark_px = int(height / 0.68)  # mark_image draws the outer ring at 0.68 of its canvas
    mark = mark_image(mark_px, on_dark=False)
    word = wordmark_image(int(height * 0.3), PALETTE["starlight"])
    gap = int(height * 0.28)
    crop = (mark_px - height) // 2 - int(height * 0.2)  # keep the faded glow around the rings
    mark = mark.crop((crop, crop, mark_px - crop, mark_px - crop))
    if stacked:
        w = max(mark.width, word.width) + 2 * gap
        h = mark.height + gap + word.height + 2 * gap
    else:
        w = mark.width + gap + word.width + 2 * gap
        h = max(mark.height, word.height) + 2 * gap
    if on_dark:
        canvas = to_image(space_background(w, h, stars=0, glow_radius=max(w, h))).convert("RGBA")
    else:
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if stacked:
        canvas.alpha_composite(mark, ((w - mark.width) // 2, gap))
        canvas.alpha_composite(word, ((w - word.width) // 2, gap + mark.height + gap // 2))
    else:
        canvas.alpha_composite(mark, (gap, (h - mark.height) // 2))
        canvas.alpha_composite(word, (gap + mark.width + gap // 2, (h - word.height) // 2))
    return canvas


# --------------------------------------------------------------------------- icons
def _seg_dist(x, y, pts):
    """Distance from every pixel to a polyline (N, 2)."""
    d = np.full(x.shape, np.inf, np.float32)
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy + 1e-12
        t = np.clip(((x - ax) * vx + (y - ay) * vy) / L2, 0, 1)
        d = np.minimum(d, np.hypot(x - ax - t * vx, y - ay - t * vy))
    return d


def _ellipse_pts(a, b, rot, n=160, t0=0.0, t1=2 * math.pi):
    t = np.linspace(t0, t1, n)
    c, s = math.cos(rot), math.sin(rot)
    ex, ey = a * np.cos(t), b * np.sin(t)
    return np.stack([ex * c - ey * s, ex * s + ey * c], 1)


def icon(name: str, size: int, color: str) -> Image.Image:
    """Line icons in the same light style as the mark (transparent PNG)."""
    L = Light(size, size)
    x, y = L.grid(size / 2, size / 2, size / 2)
    px = 2.0 / size
    col = rgb(color)
    w = 0.055

    def stroke(dist, width=w, strength=1.0):
        cov = _aa(dist - width / 2, px)
        halo = np.exp(-(dist / (width * 2.4)) ** 2) * 0.45
        L.rgb += (cov * 1.2 * strength)[..., None] * col + (halo * strength)[..., None] * col * 0.6
        L.a = np.maximum(L.a, cov)

    def dot(cx, cy, r, c=col, strength=1.4):
        d = np.hypot(x - cx, y - cy)
        cov = _aa(d - r, px)
        L.rgb += (cov * strength)[..., None] * c + (np.exp(-(d / (r * 2.5)) ** 2) * 0.6)[..., None] * c
        L.a = np.maximum(L.a, cov)

    key = name.lower()
    if "start" in key or "lumiera" in key:
        draw_mark(L, size / 2, size / 2, size * 0.42, glow=0.8)
    elif "scale" in key:  # four circles, each double the last, on one baseline
        x0 = -0.9
        for k, rr in enumerate((0.06, 0.12, 0.24, 0.48)):
            cx, cy = x0 + rr, -0.42 + rr
            x0 += 2 * rr
            if k == 0:
                dot(cx, cy, rr)
            else:
                stroke(np.abs(np.hypot(x - cx, y - cy) - rr), width=0.05)
    elif any(k in key for k in ("space", "cosmos", "universe")):
        r = np.hypot(x, y)
        stroke(np.abs(r - 0.42))  # planet
        ring = _ellipse_pts(0.86, 0.24, math.radians(-18))
        d = _seg_dist(x, y, ring)
        hidden = (r < 0.42 + w) & (y > 0.08 * (x - 0.0))  # the back of the ring hides behind the planet
        stroke(np.where(hidden, 9.0, d))
        dot(0.62, 0.55, 0.045)
    elif any(k in key for k in ("physics", "matter", "atom", "quantum")):
        for rot in (0, 60, 120):
            stroke(_seg_dist(x, y, _ellipse_pts(0.86, 0.3, math.radians(rot))), width=0.045)
        dot(0.0, 0.0, 0.085)
    elif any(k in key for k in ("life", "bio", "body", "cell")):
        t = np.linspace(-0.82, 0.82, 120)
        a = np.stack([0.32 * np.sin(t * 4.2), t], 1)
        b = np.stack([-0.32 * np.sin(t * 4.2), t], 1)
        stroke(_seg_dist(x, y, a))
        stroke(_seg_dist(x, y, b))
        for ty in np.linspace(-0.66, 0.66, 7):
            xx = 0.32 * math.sin(ty * 4.2)
            if abs(xx) > 0.08:
                stroke(_seg_dist(x, y, np.array([[-xx * 0.82, ty], [xx * 0.82, ty]])), width=0.035, strength=0.7)
    elif any(k in key for k in ("math", "number", "pattern")):
        stroke(_seg_dist(x, y, np.array([[-0.75, -0.75], [-0.75, 0.8]])), width=0.04, strength=0.6)
        stroke(_seg_dist(x, y, np.array([[-0.75, -0.75], [0.8, -0.75]])), width=0.04, strength=0.6)
        xs = np.linspace(-0.75, 0.72, 120)
        curve = np.stack([xs, -0.75 + 0.06 * (2 ** ((xs + 0.75) / 0.36) - 1)], 1)
        curve = curve[curve[:, 1] < 0.85]
        stroke(_seg_dist(x, y, curve))
        for k in range(5):  # the doubling points on y = 2^x
            xx = -0.75 + 0.36 * k
            yy = -0.75 + 0.06 * (2**k - 1)
            if yy < 0.85:
                dot(xx, yy, 0.045)
    else:  # fallback: a light point
        dot(0, 0, 0.12)
    L.fade_edges(size / 2, size / 2, size * 0.42, size * 0.5)
    return L.image()


# --------------------------------------------------------------------------- compositions
def highlight_cover(name: str) -> Image.Image:
    W, H = 1080, 1920
    base = to_image(space_background(W, H, cy=H / 2, stars=90, glow_radius=700))
    L = Light(W, H)
    xg, yg = L.grid(W / 2, H / 2, 1)
    r = np.hypot(xg, yg)
    ring_col = rgb(PALETTE["lavender"])
    L.rgb += (np.exp(-((r - 330) / 6) ** 2) * 0.9)[..., None] * ring_col
    L.rgb += (np.exp(-((r - 330) / 40) ** 2) * 0.18)[..., None] * ring_col
    paste_light(base, L)
    ic = icon(name, 360, PALETTE["starlight"])
    base.paste(ic, ((W - 360) // 2, (H - 360) // 2), ic)
    return base


def youtube_banner() -> Image.Image:
    W, H = 2560, 1440
    base = to_image(space_background(W, H, stars=420, glow_radius=1100, seed=11))
    # huge, faint doubling rings centred on the safe area (visible on TV/desktop crops)
    L = Light(W, H)
    x, y = L.grid(W / 2, H / 2, 1)
    r = np.hypot(x, y)
    for k, R in enumerate((180, 360, 720, 1440)):
        col = rgb(PALETTE["lavender"]) * (1 - k / 4) + rgb(PALETTE["cyan"]) * (k / 4)
        L.rgb += (np.exp(-((r - R) / 2.2) ** 2) * 0.35)[..., None] * col
        L.rgb += (np.exp(-((r - R) / 28) ** 2) * 0.05)[..., None] * col
    L.rgb *= (1 - 0.9 * np.exp(-((x / 820) ** 2 + ((y + 20) / 210) ** 2)))[..., None]  # keep the text area clean
    paste_light(base, L)
    # everything important stays inside YouTube's 1546x423 all-device safe area
    lock = lockup(270, stacked=False, on_dark=False)
    base.paste(lock, ((W - lock.width) // 2, H // 2 - lock.height // 2 - 52), lock)
    draw_tracked(base, (W / 2, H // 2 + 158), TAGLINE, font("body", 70), PALETTE["mist"], 0.01)
    return base


def palette_sheet() -> Image.Image:
    names = list(PALETTE.items())
    W, H = 1600, 420
    img = Image.new("RGB", (W, H), PALETTE["deep_space"])
    d = ImageDraw.Draw(img)
    cw = W // len(names)
    for i, (name, hx) in enumerate(names):
        x0 = i * cw
        d.rounded_rectangle((x0 + 16, 24, x0 + cw - 16, 270), radius=22, fill=hx,
                            outline="#2A2550" if name == "deep_space" else None, width=3)
        d.text((x0 + cw / 2, 310), name.replace("_", " ").title(), font=font("body", 30), fill=PALETTE["starlight"], anchor="mm")
        d.text((x0 + cw / 2, 356), hx.upper(), font=font("mono", 28), fill=PALETTE["mist"], anchor="mm")
    return img


def brand_sheet() -> Image.Image:
    W, H = 1600, 2160
    base = to_image(space_background(W, H, cy=520, stars=160, glow_radius=900))
    lock = lockup(300, stacked=True, on_dark=False)
    base.paste(lock, ((W - lock.width) // 2, 120), lock)
    draw_tracked(base, (W / 2, 120 + lock.height + 30), TAGLINE, font("body", 44), PALETTE["mist"], 0.02)
    pal = palette_sheet()
    base.paste(pal, (0, 1000))
    d = ImageDraw.Draw(base)
    y = 1500
    d.text((90, y), "Display", font=font("mono", 26), fill=PALETTE["mist"])
    d.text((90, y + 40), "Inter Display", font=font("display", 84), fill=PALETTE["starlight"])
    d.text((90, y + 170), "Body", font=font("mono", 26), fill=PALETTE["mist"])
    d.text((90, y + 210), "Inter — clear, calm, precise.", font=font("body", 48), fill=PALETTE["starlight"])
    d.text((90, y + 310), "Numbers", font=font("mono", 26), fill=PALETTE["mist"])
    d.text((90, y + 350), "STEP 185 · 1.0 × 10^21 m", font=font("mono", 46), fill=PALETTE["lavender"])
    for i, name in enumerate(HIGHLIGHTS[:5]):
        ic = icon(name, 150, PALETTE["starlight"])
        base.paste(ic, (1010 + (i % 3) * 190, y + 20 + (i // 3) * 210), ic)
    return base


def film_frame(t: float) -> Image.Image | None:
    if not FILM.exists():
        return None
    tmp = OUT / f".frame_{t:.2f}.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", str(FILM), "-frames:v", "1", str(tmp)], check=True)
    img = Image.open(tmp).convert("RGB")
    tmp.unlink()
    return img


def main():
    OUT.mkdir(exist_ok=True)
    save(mark_image(2048, on_dark=False), "logo/lumiera_mark_transparent.png")
    save(mark_image(2048, on_dark=True), "logo/lumiera_mark_on_dark.png")
    save(mark_image(2048, on_dark=False, mono=PALETTE["starlight"]), "logo/lumiera_mark_white_mono.png")
    save(wordmark_image(160, PALETTE["starlight"], pad=40), "logo/lumiera_wordmark_white.png")
    save(wordmark_image(160, PALETTE["deep_space"], pad=40), "logo/lumiera_wordmark_dark.png")
    save(lockup(420, stacked=False, on_dark=False), "logo/lumiera_lockup_horizontal_transparent.png")
    save(lockup(420, stacked=False, on_dark=True).convert("RGB"), "logo/lumiera_lockup_horizontal_on_dark.png")
    save(lockup(420, stacked=True, on_dark=False), "logo/lumiera_lockup_stacked_transparent.png")
    save(lockup(420, stacked=True, on_dark=True).convert("RGB"), "logo/lumiera_lockup_stacked_on_dark.png")

    pfp = mark_image(1080, on_dark=True)
    save(pfp, "profile/profile_picture_1080.png")  # TikTok + Instagram
    save(pfp.resize((800, 800), Image.LANCZOS), "profile/profile_picture_800_youtube.png")

    save(youtube_banner(), "youtube/youtube_banner_2560x1440.png")
    wm = mark_image(600, on_dark=False, mono=PALETTE["starlight"]).resize((150, 150), Image.LANCZOS)
    save(wm, "youtube/youtube_watermark_150.png")

    for name in HIGHLIGHTS:
        slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
        save(highlight_cover(name), f"instagram/highlight_{slug}.png")

    save(palette_sheet(), "brand/palette.png")
    save(brand_sheet(), "brand/brand_sheet.png")

    hook = film_frame(1.9)
    if hook:
        save(hook, "posts/the_doubling_cover_hook.jpg", quality=94)
    galaxy = film_frame(56.6)
    if galaxy:
        save(galaxy, "posts/the_doubling_cover_milky_way.jpg", quality=94)
    print("assets written to", OUT)


if __name__ == "__main__":
    main()

# Lumiera: account kit

Everything needed to open **Lumiera** on TikTok, Instagram and YouTube Shorts:

- `assets/`: logo, profile pictures, YouTube banner and watermark, Instagram highlight covers, Reels covers.
- `PROFILE.md`: handles, names and bios, ready to paste.
- `FIRST_POST.md`: the launch post for *The Doubling* (captions, hashtags, pinned comment, posting checklist).
- `BRAND.md`: how the brand looks and sounds (palette, type, logo rules, voice, content pillars).

All images are generated from code (`build_brand.py` + `brandlib.py`), so any of them can be
rebuilt or resized. The text-dependent pieces (tagline, highlight names) are read from `copy.json`.

```bash
pip install numpy pillow
python build_brand.py        # writes assets/ (the Reels covers need ../doubling_film/out/the_doubling_final.mp4 + ffmpeg)
```

## Where each file goes

| File | Use it for | Platform spec |
|---|---|---|
| `assets/profile/profile_picture_1080.png` | TikTok and Instagram profile photo | Square, shown as a circle (TikTok ≥ 200×200, Instagram shows 110×110) |
| `assets/profile/profile_picture_800_youtube.png` | YouTube profile picture | 800×800 recommended, ≤ 4 MB, shown as a 98 px circle |
| `assets/youtube/youtube_banner_2560x1440.png` | YouTube channel banner | 2560×1440, ≤ 6 MB. Logo and tagline sit inside the 1546×423 area every device shows |
| `assets/youtube/youtube_watermark_150.png` | YouTube Studio → Customization → Branding → Video watermark | 150×150, transparent, single colour, ≤ 1 MB |
| `assets/instagram/highlight_*.png` | Instagram story highlight covers | 1080×1920; the icon sits inside the circle Instagram crops |
| `assets/posts/the_doubling_cover_hook.jpg` | Cover for the first video on TikTok and Reels | 1080×1920; the question stays inside Instagram's 3:4 grid crop |
| `assets/posts/the_doubling_cover_milky_way.jpg` | Alternative cover for a later re-post | Shows "185", the hook's answer, so don't use it for the first post |
| `assets/logo/lumiera_mark_*.png` | The mark on its own (transparent, on dark, white mono) | 2048×2048 |
| `assets/logo/lumiera_lockup_*` | Mark + wordmark, horizontal or stacked | For end cards, thumbnails, collaborations |
| `assets/logo/lumiera_wordmark_*.png` | "LUMIERA" wordmark, white or dark | Transparent |
| `assets/brand/brand_sheet.png`, `palette.png` | One-page overview of the brand | |

"""Loop seam check: compares the film's last frame with its first and measures the audio across the seam.
python3 loop_check.py film.mp4"""
import subprocess, sys, numpy as np, tempfile, os
from PIL import Image
src = sys.argv[1]
with tempfile.TemporaryDirectory() as d:
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', src, '-vf', 'select=eq(n\\,0)', '-vframes', '1', os.path.join(d, 'first.png')], check=True)
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-sseof', '-0.05', '-i', src, '-update', '1', os.path.join(d, 'last.png')], check=True)
    a = np.asarray(Image.open(os.path.join(d, 'first.png')).convert('L'), float)
    b = np.asarray(Image.open(os.path.join(d, 'last.png')).convert('L'), float)
    # the play area only (the HUD legitimately changes at the lap line; the hook card is on frame 0)
    lanes = (slice(560, 1470), slice(100, 880))
    diff = np.abs(a[lanes] - b[lanes]).mean()
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', src, '-vn', '-ac', '1', '-ar', '48000', '-f', 'f32le', os.path.join(d, 'a.raw')], check=True)
    x = np.fromfile(os.path.join(d, 'a.raw'), np.float32)
    rms = lambda s: float(np.sqrt((s ** 2).mean()))
    print(f'picture: mean |last - first| over the lanes = {diff:.1f} (0-255); expect a few units (notes 1/60 s apart)')
    print(f'audio: rms last 150 ms {rms(x[-7200:]):.3f} vs first 150 ms {rms(x[:7200]):.3f}; last 5 ms peak {np.abs(x[-240:]).max():.3f}')

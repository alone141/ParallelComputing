"""Contact sheet of a video: one frame every `every` seconds, tiled. python3 contact_sheet.py in.mp4 out.png"""
import subprocess, sys, tempfile, os, glob
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
every = float(sys.argv[3]) if len(sys.argv) > 3 else 1.6
with tempfile.TemporaryDirectory() as d:
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', src, '-vf', f'fps=1/{every},scale=270:-1', os.path.join(d, 'f%03d.png')], check=True)
    files = sorted(glob.glob(os.path.join(d, 'f*.png')))
    cols = 8; rows = (len(files) + cols - 1) // cols
    w, h = Image.open(files[0]).size
    sheet = Image.new('RGB', (cols * w, rows * h), (0, 0, 0))
    for i, f in enumerate(files):
        sheet.paste(Image.open(f), ((i % cols) * w, (i // cols) * h))
    sheet.save(dst)
print(f'wrote {dst}: {len(files)} frames, one every {every} s')

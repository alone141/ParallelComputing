"""Make the counting words with piper-tts (en_US-lessac-medium).

Two voices, both from the same model:
  group voice: "five" / "three", said on each group's rest (the drummer's silent count)
  beat voice:  "one two three four", said on the quarter-note beat
The beat voice is left at natural pitch; the group voice is pitched down by
resampling in audio.py.  Output: words/<word>.wav (22050 Hz mono 16-bit).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "words")
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "piper_data")
WORDS = {"one": 0.75, "two": 0.75, "three": 0.62, "four": 0.75, "five": 0.72}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(os.path.join(DATA, "en_US-lessac-medium.onnx")):
        subprocess.run([sys.executable, "-m", "piper.download_voices", "en_US-lessac-medium",
                        "--data-dir", DATA], check=True)
    for w, ls in WORDS.items():
        subprocess.run([sys.executable, "-m", "piper", "--model", "en_US-lessac-medium",
                        "--data-dir", DATA, "--length-scale", str(ls),
                        "--sentence-silence", "0", "--noise-scale", "0.4",
                        "--output_file", os.path.join(OUT, f"{w}.wav")],
                       input=(w + ".").encode(), check=True)
        print("made", w)


if __name__ == "__main__":
    main()

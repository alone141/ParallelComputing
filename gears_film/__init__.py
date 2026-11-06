"""gears_film: a song-agnostic engine for gears-style rhythm-breakdown films.

Every instrument of a song's section is a gear on a common rack; teeth are hits, stub teeth are
rests; a gear is hidden until its part plays. The engine reads one data module per song
(songs/<slug>.py exporting SONG) and renders a silent 1080x1920 60 fps picture on the record's own
clock, plus a record_map.json for whoever aligns and muxes the recording.

    python3 -m gears_film check songs/x.py
    python3 -m gears_film still songs/x.py <seconds> out.png
    python3 -m gears_film render songs/x.py
    python3 -m gears_film export-record-map songs/x.py
"""
__version__ = "1.0"

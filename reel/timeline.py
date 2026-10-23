"""Single source of truth for beat order, durations (seconds) and music energy.

Every cut lands on the 0.5 s grid of the 120 BPM music bed.
Script v2 (reel/design/script_v2.json): Hook -> CPU -> GPU -> Data center ->
Amdahl twist -> loop. The SIMD, server and planet beats were cut.
"""

BPM = 120

BEATS = [
    ("B01Hook", "b01_hook.py", 5.0),
    ("B03Cpu", "b03_cpu.py", 9.5),
    ("B04Gpu", "b04_gpu.py", 12.0),
    ("B06DataCenter", "b06_datacenter.py", 12.5),
    ("B08Amdahl", "b08_amdahl.py", 12.5),
    ("B09Loop", "b09_loop.py", 3.5),
]

FILM_SECONDS = sum(d for _, _, d in BEATS)  # 55.0 s = 3,300 frames at 60 fps

# Music bed intensity per beat (0..1): arp >= 0.3, hats >= 0.5, kick >= 0.72.
ENERGY = {
    "B01Hook": 0.55,
    "B03Cpu": 0.6,
    "B04Gpu": 0.72,
    "B06DataCenter": 0.9,
    "B08Amdahl": 0.3,
    "B09Loop": 0.5,
}

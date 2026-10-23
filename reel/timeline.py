"""Single source of truth for beat order, durations (seconds) and music energy.

Every cut lands on the 0.5 s grid of the 120 BPM music bed.
"""

BPM = 120

BEATS = [
    ("B01Hook", "b01_hook.py", 4.5),
    ("B02Simd", "b02_simd.py", 5.0),
    ("B03Cpu", "b03_cpu.py", 4.5),
    ("B04Gpu", "b04_gpu.py", 6.5),
    ("B05Node", "b05_node.py", 4.5),
    ("B06DataCenter", "b06_datacenter.py", 6.0),
    ("B07Planet", "b07_planet.py", 5.0),
    ("B08Amdahl", "b08_amdahl.py", 8.0),
    ("B09Loop", "b09_loop.py", 3.5),
]

FILM_SECONDS = sum(d for _, _, d in BEATS)  # 47.5 s = 2,850 frames at 60 fps

# Music bed intensity per beat (0..1): arp >= 0.3, hats >= 0.5, kick >= 0.72.
ENERGY = {
    "B01Hook": 0.55,
    "B02Simd": 0.5,
    "B03Cpu": 0.6,
    "B04Gpu": 0.7,
    "B05Node": 0.72,
    "B06DataCenter": 0.9,
    "B07Planet": 0.8,
    "B08Amdahl": 0.25,
    "B09Loop": 0.5,
}

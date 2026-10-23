"""Single source of truth for beat order and durations (seconds)."""
BEATS = [
    ("FoundationTest", "_foundation_test.py", 3.0),
]
FILM_SECONDS = sum(d for _, _, d in BEATS)

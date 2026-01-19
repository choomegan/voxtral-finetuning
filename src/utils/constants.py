"""
Script for language mapping constants
"""

# Mapping for language class weighting.
SRCLANG2ID = {"zsm": 0, "ind": 1}
ID2SRCLANG = {v: k for k, v in SRCLANG2ID.items()}

TASKTYPE2ID = {"asr": 0, "s2tt": 1, "t2tt": 2, "lid": 3}

# Mapping for 3-letter ISO code to 2-letter code. Used for ASR language specification.
LANGCODE_MAP = {
    "en": "en",
    "zsm": "ms",
    "ind": "id",
}

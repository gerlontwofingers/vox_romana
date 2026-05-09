"""voice_prep — utilities for preparing and registering cloned voice profiles.

Two-step workflow:
    1. clean_audio()       → demucs vocal isolation + normalize + resample
    2. register_voice()    → write profile dir (reference.wav + metadata.json)
"""

from .cleaner import clean_audio
from .profiler import register_voice

__all__ = ["clean_audio", "register_voice"]

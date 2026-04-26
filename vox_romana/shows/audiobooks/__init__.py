"""vox_romana.shows.audiobooks — audiobook generation.

Generates structured audiobooks from a plain-text brief. The output is a
sequence of AudioEvents organized into chapters, ready for TTS rendering.

The pipeline mirrors the debate and news patterns:

    BOOTSTRAP:
        bootstrap.py  → Reads the brief, plans the chapters, profiles narrator.

    CONDUCTOR:
        conductor.py  → Generates the narration chapter by chapter. The narrator
                        is an LLM agent. The conductor manages director check-ins
                        and context compression.

    POST-PRODUCTION:
        serializer.py → Saves to JSON.

USAGE:
    from vox_romana.shows.audiobooks import AudiobookBootstrap, AudiobookConductor
    from vox_romana.core.client import DeepSeekClient

    client = DeepSeekClient()
    config = AudiobookBootstrap(client).from_brief("A short sci-fi story about first contact")
    result = AudiobookConductor(config, client=client).run()
"""

from .bootstrap import AudiobookBootstrap
from .conductor import AudiobookConductor, AudiobookConductorResult
from .config import (
    AudiobookConfig,
    AudiobookDirectorBrief,
    ChapterConfig,
    DirectorBrief,  # backwards-compat alias for AudiobookDirectorBrief
    NarratorConfig,
)
from .models import AudiobookChapter, AudiobookEvent, MergedAudiobook
from .serializer import AudiobookSerializer

__all__ = [
    "AudiobookBootstrap",
    "AudiobookChapter",
    "AudiobookConductor",
    "AudiobookConductorResult",
    "AudiobookConfig",
    "AudiobookDirectorBrief",
    "AudiobookEvent",
    "AudiobookSerializer",
    "ChapterConfig",
    "DirectorBrief",  # backwards-compat alias for AudiobookDirectorBrief
    "MergedAudiobook",
    "NarratorConfig",
]

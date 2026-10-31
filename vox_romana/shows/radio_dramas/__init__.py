"""vox_romana.shows.radio_dramas — scripted multi-voice narrative fiction.

Generates radio-drama episodes (golden-age serials, noir, sci-fi, etc.)
from a plain-text brief. The output is a sequence of AudioEvents organized
into scenes, ready for TTS rendering with era-styled music cues.

The pipeline mirrors the audiobook pattern:

    BOOTSTRAP:
        bootstrap.py  → Reads the brief, plans plot/scenes/cast, profiles
                        characters, and decides pilot vs serial. (shot 5.1-c)

    CONDUCTOR:
        conductor.py  → Generates the drama scene by scene, with an
                        announcer cold-open/sign-off and director check-ins.
                        (shot 5.1-d)

    POST-PRODUCTION:
        serializer.py → Saves to JSON.

NOTE (work in progress): bootstrap.py, conductor.py, and prompts.py land
in later shots (5.1-b/5.1-c/5.1-d). This __init__ currently exports only
the config, models, and serializer that exist today.

USAGE:
    from vox_romana.shows.radio_dramas import (
        RadioDramaConfig, MergedRadioDrama, RadioDramaSerializer,
    )
"""

from .config import (
    CharacterConfig,
    ERAS,
    RadioDramaConfig,
    RadioDramaDirectorBrief,
    SceneConfig,
    SERIES_MODES,
)
from .models import MergedRadioDrama, RadioDramaEvent, RadioDramaScene
from .serializer import RadioDramaSerializer

__all__ = [
    "CharacterConfig",
    "ERAS",
    "MergedRadioDrama",
    "RadioDramaConfig",
    "RadioDramaDirectorBrief",
    "RadioDramaEvent",
    "RadioDramaScene",
    "RadioDramaSerializer",
    "SceneConfig",
    "SERIES_MODES",
]

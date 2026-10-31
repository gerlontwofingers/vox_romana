"""Radio drama data models — event aliases and output structures.

Follows the same pattern as debates, news, and audiobooks:
    RadioDramaEvent = AudioEvent (alias)
    RadioDramaScene — groups events for one scene
    MergedRadioDrama — final serializable output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.models import AudioEvent

# ── Event alias ──────────────────────────────────────────────────────────
# Intentional re-export: type alias for readability within the radio_dramas
# package. AudioEvent is the canonical unified type in vox_romana.core.models.
# Its per-line delivery fields (tone, pace, pause_before, pause_after,
# nonverbal) are exactly what a scripted drama line needs.
RadioDramaEvent = AudioEvent


# ── Scene grouping ────────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class RadioDramaScene:
    """A scene's worth of drama events."""

    scene_id: str
    scene_order: int
    title: str
    location: str | None = None
    characters: tuple[str, ...] = ()
    events: tuple[RadioDramaEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_order": self.scene_order,
            "title": self.title,
            "location": self.location,
            "characters": list(self.characters),
            "events": [e.to_dict() for e in self.events],
        }


# ── Final output ─────────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class MergedRadioDrama:
    """The complete radio drama, ready for serialization and TTS rendering."""

    drama_id: str
    title: str
    series_mode: str = "pilot"
    series_id: str | None = None
    episode_number: int | None = None
    genre: str | None = None
    era: str = "1940s"
    scenes: tuple[RadioDramaScene, ...] = ()
    events: tuple[RadioDramaEvent, ...] = ()
    announcer: str = ""
    characters: tuple[str, ...] = ()
    voice_casting: dict[str, str] = field(default_factory=dict)
    voice_hints: dict[str, str] = field(default_factory=dict)
    tts_styles: dict[str, str] = field(default_factory=dict)
    pronunciation_overrides: dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    music_style: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "drama_id": self.drama_id,
            "title": self.title,
            "series_mode": self.series_mode,
            "series_id": self.series_id,
            "episode_number": self.episode_number,
            "genre": self.genre,
            "era": self.era,
            "scenes": [s.to_dict() for s in self.scenes],
            "events": [e.to_dict() for e in self.events],
            "announcer": self.announcer,
            "characters": list(self.characters),
            "voice_casting": dict(self.voice_casting),
            "voice_hints": dict(self.voice_hints),
            "tts_styles": dict(self.tts_styles),
            "pronunciation_overrides": dict(self.pronunciation_overrides),
            "language": self.language,
            "music_style": self.music_style,
        }

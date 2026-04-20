"""Audiobook data models — event aliases and output structures.

Follows the same pattern as debates and news:
    AudiobookEvent = AudioEvent (alias)
    AudiobookChapter — groups events for one chapter
    MergedAudiobook — final serializable output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.models import AudioEvent

# ── Event alias ──────────────────────────────────────────────────────────
AudiobookEvent = AudioEvent


# ── Chapter grouping ─────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class AudiobookChapter:
    """A chapter's worth of narration events."""

    chapter_id: str
    chapter_order: int
    title: str
    tone: str | None = None
    narrator_name: str = ""
    events: tuple[AudiobookEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "chapter_order": self.chapter_order,
            "title": self.title,
            "tone": self.tone,
            "narrator_name": self.narrator_name,
            "events": [e.to_dict() for e in self.events],
        }


# ── Final output ─────────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class MergedAudiobook:
    """The complete audiobook, ready for serialization and TTS rendering."""

    audiobook_id: str
    title: str
    chapters: tuple[AudiobookChapter, ...] = ()
    events: tuple[AudiobookEvent, ...] = ()
    narrator: str = ""
    characters: tuple[str, ...] = ()
    genre: str | None = None
    tone: str | None = None
    chapter_count: int = 0
    voice_casting: dict[str, str] = field(default_factory=dict)
    voice_hints: dict[str, str] = field(default_factory=dict)
    tts_styles: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audiobook_id": self.audiobook_id,
            "title": self.title,
            "chapters": [c.to_dict() for c in self.chapters],
            "events": [e.to_dict() for e in self.events],
            "narrator": self.narrator,
            "characters": list(self.characters),
            "genre": self.genre,
            "tone": self.tone,
            "chapter_count": self.chapter_count,
            "voice_casting": dict(self.voice_casting),
            "voice_hints": dict(self.voice_hints),
            "tts_styles": dict(self.tts_styles),
        }

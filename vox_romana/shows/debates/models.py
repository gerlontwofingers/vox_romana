from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ...core.models import AudioEvent

# DebateEvent is now an alias for the universal AudioEvent in core.
# All existing code that references DebateEvent continues to work unchanged.
DebateEvent = AudioEvent


@dataclass(slots=True)
class DebateChunk:
    template_id: str
    debate_id: str
    title: str
    chunk_order: int
    is_final_chunk: bool
    participants: tuple[str, ...]
    moderators: tuple[str, ...] = ()
    hosts: tuple[str, ...] = ()
    audience_present: bool = False
    audience_participation: bool | None = None
    topic: str | None = None
    debate_format: str | None = None
    setting: str | None = None
    moderator_mode: str | None = None
    continuity_notes: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    events: tuple[DebateEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["participants"] = list(self.participants)
        data["moderators"] = list(self.moderators)
        data["hosts"] = list(self.hosts)
        data["events"] = [event.to_dict() for event in self.events]
        return data


@dataclass(slots=True)
class ParsedDebateDocument:
    template_id: str
    parser_name: str
    raw_text: str
    chunks: tuple[DebateChunk, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "parser_name": self.parser_name,
            "raw_text": self.raw_text,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


@dataclass(slots=True)
class MergedDebate:
    template_id: str
    debate_id: str
    title: str
    chunks: tuple[DebateChunk, ...]
    events: tuple[DebateEvent, ...]
    parser_names: tuple[str, ...] = ()
    participants: tuple[str, ...] = ()
    moderators: tuple[str, ...] = ()
    hosts: tuple[str, ...] = ()
    audience_present: bool = False
    audience_participation: bool | None = None
    topic: str | None = None
    debate_format: str | None = None
    setting: str | None = None
    moderator_mode: str | None = None
    continuity_notes: tuple[str, ...] = ()
    source_count: int = 0
    language: str = "en"
    voice_casting: dict[str, str] = field(default_factory=dict)
    voice_hints: dict[str, str] = field(default_factory=dict)
    tts_styles: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "debate_id": self.debate_id,
            "title": self.title,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "events": [event.to_dict() for event in self.events],
            "parser_names": list(self.parser_names),
            "participants": list(self.participants),
            "moderators": list(self.moderators),
            "hosts": list(self.hosts),
            "audience_present": self.audience_present,
            "audience_participation": self.audience_participation,
            "topic": self.topic,
            "debate_format": self.debate_format,
            "setting": self.setting,
            "moderator_mode": self.moderator_mode,
            "continuity_notes": list(self.continuity_notes),
            "source_count": self.source_count,
            "language": self.language,
            "voice_casting": self.voice_casting,
            "voice_hints": self.voice_hints,
            "tts_styles": self.tts_styles,
        }

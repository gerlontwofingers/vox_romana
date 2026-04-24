"""News show data models.

NewsEvent is an alias for AudioEvent (same as DebateEvent — one universal
event schema for all content types). The TTS renderer consumes AudioEvents
generically regardless of show type.

NewsSegment and MergedNewsShow mirror DebateChunk and MergedDebate but
are shaped for a broadcast: ordered segments, each with a headline, events,
and metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ...core.models import AudioEvent

# NewsEvent is the same universal AudioEvent used by debates.
NewsEvent = AudioEvent


@dataclass(slots=True)
class NewsSegment:
    """One completed segment of the news broadcast.

    Attributes:
        segment_id:     Unique ID for this segment within the show.
        segment_order:  Position in the rundown (0-indexed).
        headline:       The story headline.
        segment_type:   "anchor_read", "package", "panel", "breaking".
        anchor_name:    Who delivered this segment.
        tone:           The achieved tone of this segment.
        events:         The generated audio events for this segment.
    """

    segment_id: str
    segment_order: int
    headline: str
    segment_type: str = "anchor_read"
    anchor_name: str = ""
    tone: str | None = None
    events: tuple[NewsEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["events"] = [event.to_dict() for event in self.events]
        return data


@dataclass(slots=True)
class MergedNewsShow:
    """The complete, serializable output of a news show generation.

    This is what gets saved to JSON and fed to the TTS renderer.

    Attributes:
        show_id:        Unique ID for this broadcast.
        title:          Show title (e.g., "The Evening Brief").
        segments:       The ordered segments with their events.
        events:         All events in global order (for the renderer).
        anchors:        Names of anchors who appeared.
        hosts:          Names of show hosts (narrator/emcee).
        topic:          The broadcast theme/topic.
        tone:           The overall tone achieved.
        segment_count:  Number of segments in the show.
    """

    show_id: str
    title: str
    segments: tuple[NewsSegment, ...] = ()
    events: tuple[NewsEvent, ...] = ()
    anchors: tuple[str, ...] = ()
    hosts: tuple[str, ...] = ()
    guests: tuple[str, ...] = ()
    topic: str | None = None
    tone: str | None = None
    segment_count: int = 0
    voice_casting: dict[str, str] = field(default_factory=dict)
    voice_hints: dict[str, str] = field(default_factory=dict)
    tts_styles: dict[str, str] = field(default_factory=dict)
    language: str = "en"
    series_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "show_id": self.show_id,
            "title": self.title,
            "language": self.language,
            "segments": [s.to_dict() for s in self.segments],
            "events": [e.to_dict() for e in self.events],
            "anchors": list(self.anchors),
            "hosts": list(self.hosts),
            "guests": list(self.guests),
            "topic": self.topic,
            "tone": self.tone,
            "segment_count": self.segment_count,
            "voice_casting": self.voice_casting,
            "voice_hints": self.voice_hints,
            "tts_styles": self.tts_styles,
            "series_id": self.series_id,
        }

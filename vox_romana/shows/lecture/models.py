"""Lecture data models — event aliases and output structures.

Mirrors the audiobooks/news/podcast pattern:
    LectureEvent  = AudioEvent (alias)
    LectureSection — groups events for one section
    MergedLecture — final serializable output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.models import AudioEvent

# ── Event alias ──────────────────────────────────────────────────────────
LectureEvent = AudioEvent


# ── Section grouping ─────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class LectureSection:
    """A section's worth of professor events."""

    section_id: str
    section_order: int
    title: str
    section_type: str = "concept"
    learning_objective: str | None = None
    professor_name: str = ""
    events: tuple[LectureEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_order": self.section_order,
            "title": self.title,
            "section_type": self.section_type,
            "learning_objective": self.learning_objective,
            "professor_name": self.professor_name,
            "events": [e.to_dict() for e in self.events],
        }


# ── Final output ─────────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class MergedLecture:
    """The complete lecture, ready for serialization and TTS rendering."""

    lecture_id: str
    title: str
    sections: tuple[LectureSection, ...] = ()
    events: tuple[LectureEvent, ...] = ()
    professor: str = ""
    professor_title: str | None = None
    subject_area: str | None = None
    tone: str | None = None
    section_count: int = 0
    voice_casting: dict[str, str] = field(default_factory=dict)
    voice_hints: dict[str, str] = field(default_factory=dict)
    tts_styles: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lecture_id": self.lecture_id,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "events": [e.to_dict() for e in self.events],
            "professor": self.professor,
            "professor_title": self.professor_title,
            "subject_area": self.subject_area,
            "tone": self.tone,
            "section_count": self.section_count,
            "voice_casting": dict(self.voice_casting),
            "voice_hints": dict(self.voice_hints),
            "tts_styles": dict(self.tts_styles),
        }

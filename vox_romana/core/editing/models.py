"""Data models for the editing pipeline.

These are the building blocks for the Director outline, Editor verdicts,
and chunk results. All are plain dataclasses — no behavior, just data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class OutlineSection:
    """A single section in the Director's show outline."""

    id: str
    title: str
    expected_duration: str = ""
    key_beats: list[str] = field(default_factory=list)
    speaker_assignments: list[str] = field(default_factory=list)
    acceptance_criteria: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutlineSection:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class OutlineSnapshot:
    """A versioned snapshot of the Director's outline.

    Every time the outline changes, a new snapshot is created. This is
    what gets broadcast to the cast and persisted for audit.
    """

    outline_id: str
    version: int
    timestamp: str
    summary: str
    sections: list[OutlineSection]
    changed_sections: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sections"] = [s.to_dict() for s in self.sections]
        return data

    def to_broadcast_text(self) -> str:
        """Format for injection into agent context windows."""
        lines = [
            f"OUTLINE UPDATE (v{self.version}): {self.summary}",
        ]
        if self.changed_sections:
            lines.append(
                f"Changed sections: {', '.join(self.changed_sections)}"
            )
        lines.append("Current sections:")
        for s in self.sections:
            beats = ", ".join(s.key_beats) if s.key_beats else "no specific beats"
            speakers = (
                ", ".join(s.speaker_assignments)
                if s.speaker_assignments
                else "TBD"
            )
            lines.append(
                f"  [{s.id}] {s.title} ({s.expected_duration}) — "
                f"beats: {beats} | speakers: {speakers}"
            )
        return "\n".join(lines)


@dataclass
class EditorVerdict:
    """The Editor's judgment on a single chunk."""

    status: str  # "pass" or "fail"
    chunk_id: str
    outline_version: int
    summary: str
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditorVerdict:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class ChunkResult:
    """The result of generating (and optionally retrying) a single chunk."""

    chunk_id: str
    content: list[dict[str, Any]]
    outline_version: int
    attempt_number: int
    editor_verdict: EditorVerdict | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.editor_verdict:
            data["editor_verdict"] = self.editor_verdict.to_dict()
        return data

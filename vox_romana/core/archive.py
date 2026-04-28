"""Persistent storage for Editor notes, agent notes, and outline snapshots.

Notes are stored as JSONL (one JSON object per line) for efficient
append-only writes and streaming reads. This is intentionally simple —
no database, no vector store, just files.

THREE ARCHIVES:

    NoteArchive      — Editor review notes (pass/fail/warn per chunk).
    AgentNoteStore   — General-purpose inter-agent notes. Any agent can
                       write a note (key facts, character states, plot threads,
                       arc decisions) and any downstream agent can read them.
                       Used by the multi-agent literature pipeline and by
                       conductors for long-range coherence.
    OutlineArchive   — Persisted outline snapshots for audit trails.

COMPACTION POLICY (from the design spec):
  - Keep the last 10 broadcasts verbatim.
  - Summarize everything older into a rolling "patterns" summary.
  - Session-start note injection must never exceed ~600 tokens.

USAGE:
    from vox_romana.core.archive import EditorNote, NoteArchive

    archive = NoteArchive("output/studio_memory/editor_notes.jsonl")
    archive.append(EditorNote(
        broadcast_id="abc123",
        content_id="chunk_01",
        outline_version=1,
        status="fail",
        attempt_number=1,
        summary="Round 2 repeated round 1 positions almost verbatim.",
        issues=["repetition"],
        recommendations=["inject a wildcard pivot topic"],
    ))

    # Agent-to-agent notes:
    from vox_romana.core.archive import AgentNote, AgentNoteStore

    store = AgentNoteStore("output/studio_memory/agent_notes.jsonl")
    store.append(AgentNote(
        run_id="run_abc",
        agent="genre_classifier",
        note_type="genre",
        key="genre",
        value="fantasy",
    ))
    store.append(AgentNote(
        run_id="run_abc",
        agent="arc_planner",
        note_type="arc",
        key="act_1_summary",
        value="The hero departs the village...",
    ))
    # Downstream agents read all notes for this run:
    notes = store.get_by_run("run_abc")
    genre_notes = store.get_by_run("run_abc", note_type="genre")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class EditorNote:
    """A single editor review note — the atomic unit of the archive.

    Matches the schema from the design spec:
        broadcast_id, chunk_id, outline_version, status (pass|fail|warn),
        attempt_number, summary, issues[], recommendations[]
    """

    broadcast_id: str
    chunk_id: str
    outline_version: int
    status: str  # "pass", "fail", "warn"
    attempt_number: int
    summary: str
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    timestamp: str = ""
    show_type: str = ""
    chunk_type: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditorNote:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class OutlineRecord:
    """A persisted outline snapshot for audit trails."""

    broadcast_id: str
    outline_id: str
    version: int
    timestamp: str
    summary: str
    sections_json: str  # JSON-serialized list of OutlineSection dicts
    changed_sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutlineRecord:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class NoteArchive:
    """Append-only JSONL store for editor notes.

    Each line is a JSON object representing one EditorNote. Reads are
    sequential scans — fine for the expected volume (dozens to hundreds
    of notes, not millions).
    """

    def __init__(self, archive_path: str | Path) -> None:
        self._path = Path(archive_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, note: EditorNote) -> None:
        """Append a single note to the archive."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(note.to_dict()) + "\n")

    def load_all(self) -> list[EditorNote]:
        """Load all notes from the archive. Returns empty list if missing."""
        if not self._path.exists():
            return []
        notes: list[EditorNote] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                notes.append(EditorNote.from_dict(data))
        return notes

    def query(
        self,
        *,
        broadcast_id: str | None = None,
        status: str | None = None,
        show_type: str | None = None,
        chunk_type: str | None = None,
        limit: int | None = None,
    ) -> list[EditorNote]:
        """Query notes with optional filters. Returns newest first."""
        notes = self.load_all()
        if broadcast_id:
            notes = [n for n in notes if n.broadcast_id == broadcast_id]
        if status:
            notes = [n for n in notes if n.status == status]
        if show_type:
            notes = [n for n in notes if n.show_type == show_type]
        if chunk_type:
            notes = [n for n in notes if n.chunk_type == chunk_type]
        notes.reverse()  # newest first
        if limit:
            notes = notes[:limit]
        return notes

    def get_broadcast_ids(self) -> list[str]:
        """Get unique broadcast IDs in chronological order."""
        notes = self.load_all()
        seen: set[str] = set()
        ids: list[str] = []
        for note in notes:
            if note.broadcast_id not in seen:
                seen.add(note.broadcast_id)
                ids.append(note.broadcast_id)
        return ids

    def get_recent_broadcasts(self, n: int = 10) -> list[EditorNote]:
        """Get notes from the last n broadcasts (verbatim, newest first)."""
        broadcast_ids = self.get_broadcast_ids()
        recent_ids = set(broadcast_ids[-n:]) if broadcast_ids else set()
        notes = [note for note in self.load_all() if note.broadcast_id in recent_ids]
        notes.reverse()
        return notes

    def format_notes_compact(
        self, notes: list[EditorNote], token_budget: int = 600
    ) -> str:
        """Format notes into a compact text summary within a token budget.

        Uses a rough 4-chars-per-token estimate. Returns a string suitable
        for injection into an agent's context.
        """
        char_budget = token_budget * 4
        lines: list[str] = []
        total_chars = 0

        for note in notes:
            issues_str = ", ".join(note.issues) if note.issues else "none"
            line = (
                f"[{note.status.upper()}] {note.broadcast_id}/{note.chunk_id} "
                f"(attempt {note.attempt_number}): {note.summary} "
                f"Issues: {issues_str}"
            )
            if total_chars + len(line) > char_budget:
                lines.append("... (older notes truncated for token budget)")
                break
            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines) if lines else "(no historical notes)"


class OutlineArchive:
    """Append-only JSONL store for outline snapshots."""

    def __init__(self, archive_path: str | Path) -> None:
        self._path = Path(archive_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: OutlineRecord) -> None:
        """Append a single outline snapshot to the archive."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def load_all(self) -> list[OutlineRecord]:
        """Load all outline records. Returns empty list if missing."""
        if not self._path.exists():
            return []
        records: list[OutlineRecord] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                records.append(OutlineRecord.from_dict(data))
        return records


# ---------------------------------------------------------------------------
# Agent-to-agent notes — general-purpose inter-agent communication
# ---------------------------------------------------------------------------


@dataclass
class AgentNote:
    """A single inter-agent note.

    Any pipeline agent can write a note (key facts, character states,
    arc decisions, running tallies) and any downstream agent can read it.

    Fields:
        run_id     — groups notes for one pipeline run / episode.
        agent      — which agent wrote this note.
        note_type  — category tag (e.g. "genre", "arc", "character",
                     "fact", "continuity", "style"). Free-form string.
        key        — short label within the type (e.g. "protagonist_name").
        value      — the payload (string, number, short JSON blob, etc.).
        timestamp  — auto-set to UTC ISO timestamp on creation.
    """

    run_id: str
    agent: str
    note_type: str
    key: str
    value: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentNote:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class AgentNoteStore:
    """Append-only JSONL store for inter-agent notes.

    Same pattern as NoteArchive but stores AgentNote objects and
    supports filtering by run_id, agent, and note_type.
    """

    def __init__(self, store_path: str | Path) -> None:
        self._path = Path(store_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, note: AgentNote) -> None:
        """Append a single note."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(note.to_dict()) + "\n")

    def load_all(self) -> list[AgentNote]:
        """Load all notes. Returns empty list if file missing."""
        if not self._path.exists():
            return []
        notes: list[AgentNote] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                notes.append(AgentNote.from_dict(data))
        return notes

    def get_by_run(
        self,
        run_id: str,
        *,
        note_type: str | None = None,
        agent: str | None = None,
    ) -> list[AgentNote]:
        """Get all notes for a run, optionally filtered by type / agent."""
        notes = [n for n in self.load_all() if n.run_id == run_id]
        if note_type:
            notes = [n for n in notes if n.note_type == note_type]
        if agent:
            notes = [n for n in notes if n.agent == agent]
        return notes

    def format_notes_compact(
        self, notes: list[AgentNote], token_budget: int = 600
    ) -> str:
        """Format notes into compact text for context injection."""
        char_budget = token_budget * 4
        lines: list[str] = []
        total_chars = 0

        for note in notes:
            line = f"[{note.note_type}:{note.key}] ({note.agent}) {note.value}"
            if total_chars + len(line) > char_budget:
                lines.append("... (older notes truncated for token budget)")
                break
            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines) if lines else "(no agent notes)"

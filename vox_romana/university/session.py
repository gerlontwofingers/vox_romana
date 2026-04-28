"""Scholar session model — persistent state for a University research run.

This module defines two public classes:

:class:`ScholarNote`
    A frozen, citable record of what the Scholar learned from one article.
    Mirrors :class:`vox_romana.core.models.FactCard` in structure and intent.

:class:`ScholarSession`
    Mutable state for an entire research run: all notes, a running synthesis,
    article queue, timing info, and status.  Mirrors
    :class:`vox_romana.core.config.ContentConfig` in mutability and
    ``validate()`` convention.

Persistence
-----------
Sessions are checkpointed after every article to
``{UNIVERSITY_ROOT}/{session_id}/session.json``.  Notes are appended
incrementally to ``{UNIVERSITY_ROOT}/{session_id}/notes.jsonl`` in
:class:`vox_romana.memory.ingest.DocumentChunk` JSONL format so the
existing ``ChromaVectorIndex.migrate_documents()`` pipeline indexes them
without any new wiring.

A lightweight session registry at ``{UNIVERSITY_ROOT}/index.json`` keeps
one record per session so ``run.py --mode university list`` is O(1) rather than
scanning every folder.

IDs
---
``session_id = uuid.uuid4().hex[:12]`` — identical to the pattern used by
:meth:`vox_romana.core.catalog.Catalog.log_run` and ``log_study``.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vox_romana.core.paths import UNIVERSITY_ROOT

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Current UTC time in ISO 8601 format with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_session_id() -> str:
    """Generate a short session ID. Mirrors Catalog.log_run() pattern."""
    return uuid.uuid4().hex[:12]


# ─────────────────────────────────────────────────────────────────────────────
# ScholarNote — frozen, mirrors FactCard
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScholarNote:
    """What the Scholar learned from one article.

    Frozen so notes can be safely hashed, cached, and shared.  Uses
    ``tuple`` for collection fields (frozen-dataclass compat) — exactly
    the same pattern as :class:`vox_romana.core.models.FactCard`.

    Attributes:
        article_title:   Title of the article this note was taken from.
        source_url:      Canonical URL (or pseudo-URL like ``memory://…``).
        source_name:     Name of the originating KnowledgeSource
                         (``"wikipedia"``, ``"local"``, etc.).
        key_facts:       Bullet-point facts extracted by the LLM.
        summary:         One-paragraph summary of the article.
        links_found:     Outbound article identifiers that look worth following.
        relevance_score: How relevant was this article to the session topic?
                         0–10 scale set by the Scholar's note-extraction prompt.
        timestamp:       ISO 8601 UTC time this note was created.
    """

    article_title: str
    source_url: str = ""
    source_name: str = ""
    key_facts: tuple[str, ...] = ()
    summary: str = ""
    links_found: tuple[str, ...] = ()
    relevance_score: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (tuples → lists). Mirrors FactCard."""
        d = asdict(self)
        d["key_facts"] = list(self.key_facts)
        d["links_found"] = list(self.links_found)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScholarNote":
        """Inverse of :meth:`to_dict`. Tolerant of missing keys."""
        return cls(
            article_title=d.get("article_title", ""),
            source_url=d.get("source_url", ""),
            source_name=d.get("source_name", ""),
            key_facts=tuple(d.get("key_facts", ())),
            summary=d.get("summary", ""),
            links_found=tuple(d.get("links_found", ())),
            relevance_score=float(d.get("relevance_score", 0.0)),
            timestamp=d.get("timestamp", ""),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ScholarSession — mutable session state, mirrors ContentConfig
# ─────────────────────────────────────────────────────────────────────────────

# Valid session status values.
_VALID_STATUSES = frozenset({"active", "complete", "paused", "interrupted"})


@dataclass
class ScholarSession:
    """Full state for one Scholar research run.

    Mutable so the Scholar can update it incrementally as articles are read.
    Mirrors the mutability convention from
    :class:`vox_romana.core.config.ContentConfig`.

    Attributes:
        session_id:       Short UUID (12 hex chars). Auto-generated.
        topic:            The research directive given by the user.
        started_at:       ISO 8601 UTC timestamp when the session started.
        updated_at:       ISO 8601 UTC timestamp of the last checkpoint.
        budget_hours:     Time budget in hours. ``None`` = self-determined by
                          the Scholar's complexity estimator.
        estimated_hours:  The Scholar's own estimate of required time (set
                          once by ``_determine_duration``; shown in progress).
        notes:            All notes collected so far, in arrival order.
        synthesis:        Running synthesis text; updated every N articles.
        articles_read:    Titles of articles already processed (deduped).
        articles_queued:  Titles yet to read, in priority order.
        status:           One of ``"active"``, ``"complete"``, ``"paused"``,
                          ``"interrupted"``.
        sources_used:     Names of KnowledgeSources that contributed articles.
        catalog_study_id: ID returned by ``catalog.log_study()`` on completion.
                          ``None`` until cataloged.
    """

    session_id: str = field(default_factory=_new_session_id)
    topic: str = ""
    started_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    budget_hours: float | None = None
    estimated_hours: float | None = None
    notes: list[ScholarNote] = field(default_factory=list)
    synthesis: str = ""
    articles_read: list[str] = field(default_factory=list)
    articles_queued: list[str] = field(default_factory=list)
    status: str = "active"
    sources_used: list[str] = field(default_factory=list)
    catalog_study_id: str | None = None

    # ── validation (mirrors ContentConfig.validate) ─────────────────────

    def validate(self) -> list[str]:
        """Return a list of problems. Empty list = all good.

        Mirrors the soft-check pattern from ContentConfig.validate().
        """
        problems: list[str] = []

        if not self.topic.strip():
            problems.append("topic is empty — the Scholar needs a research directive.")

        if self.status not in _VALID_STATUSES:
            problems.append(
                f"status must be one of {sorted(_VALID_STATUSES)}, "
                f"got '{self.status}'."
            )

        if self.budget_hours is not None and self.budget_hours <= 0:
            problems.append(
                f"budget_hours must be positive, got {self.budget_hours}."
            )

        if self.estimated_hours is not None and self.estimated_hours <= 0:
            problems.append(
                f"estimated_hours must be positive, got {self.estimated_hours}."
            )

        return problems

    # ── serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "budget_hours": self.budget_hours,
            "estimated_hours": self.estimated_hours,
            "notes": [n.to_dict() for n in self.notes],
            "synthesis": self.synthesis,
            "articles_read": list(self.articles_read),
            "articles_queued": list(self.articles_queued),
            "status": self.status,
            "sources_used": list(self.sources_used),
            "catalog_study_id": self.catalog_study_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScholarSession":
        """Inverse of :meth:`to_dict`. Tolerant of missing keys."""
        session = cls(
            session_id=d.get("session_id") or _new_session_id(),
            topic=d.get("topic", ""),
            started_at=d.get("started_at", _now_iso()),
            updated_at=d.get("updated_at", _now_iso()),
            budget_hours=d.get("budget_hours"),
            estimated_hours=d.get("estimated_hours"),
            synthesis=d.get("synthesis", ""),
            status=d.get("status", "active"),
            sources_used=list(d.get("sources_used", [])),
            catalog_study_id=d.get("catalog_study_id"),
        )
        session.notes = [ScholarNote.from_dict(n) for n in d.get("notes", [])]
        session.articles_read = list(d.get("articles_read", []))
        session.articles_queued = list(d.get("articles_queued", []))
        return session

    # ── persistence ─────────────────────────────────────────────────────

    def session_dir(self, base_dir: Path | None = None) -> Path:
        """Absolute path to this session's folder."""
        root = Path(base_dir) if base_dir else UNIVERSITY_ROOT
        return root / self.session_id

    def save(self, base_dir: Path | None = None) -> Path:
        """Checkpoint the full session state to disk.

        Writes ``{session_dir}/session.json`` and updates the registry at
        ``{base_dir}/index.json``.

        Args:
            base_dir: Root directory for all sessions. Defaults to
                      :data:`vox_romana.core.paths.UNIVERSITY_ROOT`.

        Returns:
            Path to the written ``session.json``.
        """
        self.updated_at = _now_iso()
        sdir = self.session_dir(base_dir)
        sdir.mkdir(parents=True, exist_ok=True)
        path = sdir / "session.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        self._update_index(base_dir)
        return path

    def save_notes_jsonl(self, base_dir: Path | None = None) -> Path:
        """Write all notes as a JSONL file in DocumentChunk format.

        The file is **rewritten from scratch** on every call — this is
        intentional (safe, idempotent, cheap given typical note counts).
        The Scholar writes this after each batch so the Chroma migration
        pipeline always has an up-to-date snapshot.

        The JSONL format matches :class:`vox_romana.memory.ingest.DocumentChunk`
        exactly so ``ChromaVectorIndex.migrate_documents()`` indexes these
        notes without any new code.

        Args:
            base_dir: Root directory for all sessions. Defaults to
                      :data:`vox_romana.core.paths.UNIVERSITY_ROOT`.

        Returns:
            Path to the written ``notes.jsonl``.
        """
        sdir = self.session_dir(base_dir)
        sdir.mkdir(parents=True, exist_ok=True)
        path = sdir / "notes.jsonl"
        chunks = self.to_jsonl_chunks()
        with path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, session_dir: Path) -> "ScholarSession":
        """Load a session from its directory.

        Args:
            session_dir: Path to the session folder (contains
                         ``session.json``).  May also be just the
                         session_id string — if so, it is resolved under
                         :data:`vox_romana.core.paths.UNIVERSITY_ROOT`.

        Returns:
            The loaded ScholarSession.

        Raises:
            FileNotFoundError: If ``session.json`` does not exist.
            ValueError: If the JSON is malformed.
        """
        p = Path(session_dir)
        # Allow passing just a session_id string
        if not p.is_absolute() and not (p / "session.json").exists():
            p = UNIVERSITY_ROOT / p
        json_path = p / "session.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Session file not found: {json_path}. "
                f"Check the session ID or run `run.py --mode university list`."
            )
        try:
            with json_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed session.json at {json_path}: {exc}"
            ) from exc
        return cls.from_dict(data)

    # ── DocumentChunk-compatible export ────────────────────────────────

    def to_jsonl_chunks(self) -> list[dict[str, Any]]:
        """Export all notes as a list of DocumentChunk-compatible dicts.

        Each :class:`ScholarNote` becomes one DocumentChunk record.
        The ``text`` field contains the note's summary + key_facts so it
        is fully searchable via the existing ChromaVectorIndex and MemoryRecall.

        The ``source`` field uses ``university://{session_id}`` so
        MemorySearchProvider can distinguish university notes from other
        ingested documents.

        Format is identical to :meth:`vox_romana.memory.ingest.DocumentChunk.to_dict`:

        .. code-block:: json

            {
                "text": "...",
                "title": "...",
                "source": "university://abc123/Toyota Corolla",
                "chunk_index": 0,
                "total_chunks": 1,
                "format": ".university_note",
                "ingested_at": "2026-04-26T12:00:00+00:00",
                "tag": "university_research"
            }
        """
        chunks: list[dict[str, Any]] = []
        total = len(self.notes)
        for i, note in enumerate(self.notes):
            # Build a rich text block: summary + bulleted key facts.
            facts_block = "\n".join(f"- {f}" for f in note.key_facts)
            text = note.summary
            if facts_block:
                text = f"{text}\n\nKey facts:\n{facts_block}" if text else facts_block

            chunks.append({
                "text": text,
                "title": note.article_title,
                "source": f"university://{self.session_id}/{note.article_title}",
                "chunk_index": i,
                "total_chunks": total,
                "format": ".university_note",
                "ingested_at": note.timestamp or self.updated_at,
                "tag": "university_research",
            })
        return chunks

    # ── index registry ──────────────────────────────────────────────────

    def _update_index(self, base_dir: Path | None = None) -> None:
        """Update the session registry at ``{base_dir}/index.json``.

        The index is a JSON array of lightweight records — one per session.
        It is read, patched (upsert by session_id), and written atomically.
        """
        root = Path(base_dir) if base_dir else UNIVERSITY_ROOT
        root.mkdir(parents=True, exist_ok=True)
        index_path = root / "index.json"

        # Read existing index.
        existing: list[dict[str, Any]] = []
        if index_path.exists():
            try:
                with index_path.open(encoding="utf-8") as fh:
                    existing = json.load(fh)
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, OSError):
                existing = []

        # Build updated record for this session.
        record: dict[str, Any] = {
            "session_id": self.session_id,
            "topic": self.topic,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "articles_read": len(self.articles_read),
            "articles_queued": len(self.articles_queued),
            "notes_count": len(self.notes),
            "budget_hours": self.budget_hours,
            "estimated_hours": self.estimated_hours,
        }

        # Upsert — replace existing record with same session_id if found.
        updated = [r for r in existing if r.get("session_id") != self.session_id]
        updated.append(record)
        # Most recent first.
        updated.sort(key=lambda r: r.get("updated_at", ""), reverse=True)

        try:
            with index_path.open("w", encoding="utf-8") as fh:
                json.dump(updated, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("Failed to update session index: %s", exc)

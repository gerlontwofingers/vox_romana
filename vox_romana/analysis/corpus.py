"""Corpus loading — converts various input formats to SpeakerCorpus objects.

Supported inputs:
    vox_romana debate/podcast JSON  — parse AudioEvent.text per speaker_name.
    plain text / transcript file    — each qualifying line becomes one utterance.
    any format in memory/ingest.py  — PDF, DOCX, HTML, EPUB, ZIP, etc.

USAGE::

    from vox_romana.analysis.corpus import CorpusLoader

    # From a vox_romana debate JSON (one SpeakerCorpus per speaker):
    corpora = CorpusLoader.from_vox_romana_json(
        "output/text/debates/debate_01.json"
    )
    # → {"Teal'c": SpeakerCorpus(...), "Ronon Dex": SpeakerCorpus(...)}

    # From a plain text transcript (single speaker):
    corpus = CorpusLoader.from_text_file(
        "speeches/candidate_a.txt", speaker_name="Candidate A"
    )

    # From any ingest-supported format (PDF, DOCX, HTML, …):
    corpus = CorpusLoader.from_ingest_path(
        "papers/manifesto.pdf", speaker_name="Author X"
    )
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from .models import SpeakerCorpus

logger = logging.getLogger(__name__)

# Speaker roles whose output we treat as substantive discourse.
_DEFAULT_ROLES: frozenset[str] = frozenset({
    "participant",
    "host",
    "anchor",
    "reporter",
    "guest",
    "moderator",
    "narrator",
})

# event_kind values that carry actual spoken content.
_SPEECH_KINDS: frozenset[str] = frozenset({
    "utterance",
    "narration",
    "speech",
})

# Extensions handled natively without DocumentIngester.
_TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".log", "",
})


class CorpusLoader:
    """Factory for creating SpeakerCorpus objects from various sources."""

    # ─────────────────────────────────────────────────────────────────────────
    # vox_romana JSON
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def from_vox_romana_json(
        path: str | Path,
        *,
        roles: set[str] | None = None,
        min_words: int = 3,
        exclude_speakers: set[str] | None = None,
    ) -> dict[str, SpeakerCorpus]:
        """Load per-speaker corpora from a vox_romana debate/podcast JSON.

        Handles all three event layouts produced by vox_romana:
            chunks[].events[]   — standard debate output
            segments[].events[] — news / podcast output
            events[]            — flat layout (top-level)

        Args:
            path:              Path to the JSON file.
            roles:             Speaker roles to include.  Defaults to
                               ``_DEFAULT_ROLES``.
            min_words:         Minimum words per utterance.  Filters stage
                               directions and very short interjections.
            exclude_speakers:  Speaker names to skip entirely.

        Returns:
            Dict mapping speaker_name → SpeakerCorpus.  Only speakers with
            at least one qualifying utterance are included.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError:        If no qualifying utterances are found.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"vox_romana JSON not found: {path}")

        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        active_roles = roles if roles is not None else _DEFAULT_ROLES
        skip_speakers = exclude_speakers or set()

        base_metadata: dict[str, Any] = {
            "source_file": str(path),
            "title": data.get("title", ""),
            "debate_id": data.get("debate_id", ""),
            "show_id": data.get("show_id", ""),
        }

        utterances_by_speaker: dict[str, list[str]] = {}

        for event in _extract_events(data):
            role = event.get("speaker_role", "")
            kind = event.get("event_kind", "utterance")
            speaker = (event.get("speaker_name") or "").strip()
            text = (event.get("text") or "").strip()

            if not speaker or speaker in skip_speakers:
                continue
            if role not in active_roles:
                continue
            if kind not in _SPEECH_KINDS:
                continue
            if not text:
                continue
            if len(text.split()) < min_words:
                continue

            utterances_by_speaker.setdefault(speaker, []).append(text)

        if not utterances_by_speaker:
            raise ValueError(
                f"No qualifying utterances found in {path}.  "
                f"Tried roles={set(active_roles)}, min_words={min_words}."
            )

        corpora = {
            speaker: SpeakerCorpus(
                name=speaker,
                utterances=utterances,
                metadata={**base_metadata, "speaker": speaker},
            )
            for speaker, utterances in utterances_by_speaker.items()
        }

        logger.info(
            "Loaded %d speakers from %s: %s",
            len(corpora),
            path.name,
            list(corpora.keys()),
        )
        return corpora

    # ─────────────────────────────────────────────────────────────────────────
    # Plain text
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def from_text_file(
        path: str | Path,
        *,
        speaker_name: str | None = None,
        min_words: int = 3,
    ) -> SpeakerCorpus:
        """Load a plain text file as a single-speaker corpus.

        Each non-empty line with at least ``min_words`` words becomes one
        utterance.  Non-text formats (.pdf, .docx, etc.) are automatically
        delegated to :meth:`from_ingest_path`.

        Args:
            path:          Path to the file.
            speaker_name:  Display name.  Defaults to the filename stem.
            min_words:     Minimum words per line.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError:        If no qualifying lines are found.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")

        speaker_name = speaker_name or path.stem

        if path.suffix.lower() not in _TEXT_EXTENSIONS:
            return CorpusLoader.from_ingest_path(
                path, speaker_name=speaker_name, min_words=min_words
            )

        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = path.read_text(encoding="latin-1", errors="replace")

        utterances = [
            line.strip()
            for line in raw.splitlines()
            if len(line.strip().split()) >= min_words
        ]

        if not utterances:
            raise ValueError(
                f"No qualifying lines found in {path} with min_words={min_words}."
            )

        logger.info("Loaded %d utterances from %s", len(utterances), path.name)
        return SpeakerCorpus(
            name=speaker_name,
            utterances=utterances,
            metadata={"source_file": str(path)},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Any ingest-supported format
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def from_ingest_path(
        path: str | Path,
        *,
        speaker_name: str | None = None,
        min_words: int = 3,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> SpeakerCorpus:
        """Load any DocumentIngester-supported file as a single-speaker corpus.

        Uses a temporary directory for JSONL output so ``data/memory/documents/``
        is never polluted by analysis runs.

        Args:
            path:          Path to a file or directory.
            speaker_name:  Display name.  Defaults to path stem.
            min_words:     Minimum words per chunk.
            chunk_size:    Characters per chunk (passed to DocumentIngester).
            chunk_overlap: Overlap between chunks.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError:        If no qualifying chunks are produced.
        """
        from vox_romana.memory.ingest import DocumentIngester

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Ingest path not found: {path}")

        speaker_name = speaker_name or path.stem
        utterances: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            ingester = DocumentIngester(
                output_dir=tmpdir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if path.is_dir():
                ingester.ingest_directory(str(path))
            else:
                ingester.ingest(str(path))

            for jsonl_path in sorted(Path(tmpdir).glob("*.jsonl")):
                with jsonl_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk_text = (chunk.get("text") or "").strip()
                        if len(chunk_text.split()) >= min_words:
                            utterances.append(chunk_text)

        if not utterances:
            raise ValueError(
                f"No qualifying chunks extracted from {path}."
            )

        logger.info(
            "Loaded %d chunks from %s via ingest", len(utterances), path.name
        )
        return SpeakerCorpus(
            name=speaker_name,
            utterances=utterances,
            metadata={"source_file": str(path)},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all AudioEvents from a vox_romana JSON.

    Handles three layouts:
        chunks[].events[]   — debate output
        segments[].events[] — news / podcast output
        events[]            — flat (top-level)

    Events are deduplicated by global_order when present and returned in
    ascending order.  Insertion order is preserved for events without
    a global_order field.
    """
    seen_orders: set[int] = set()
    ordered: list[tuple[int, dict[str, Any]]] = []
    unordered: list[dict[str, Any]] = []

    def _add(event: dict[str, Any]) -> None:
        go = event.get("global_order")
        if go is not None:
            if go not in seen_orders:
                seen_orders.add(go)
                ordered.append((go, event))
        else:
            unordered.append(event)

    for chunk in data.get("chunks", []):
        for ev in chunk.get("events", []):
            _add(ev)

    for segment in data.get("segments", []):
        for ev in segment.get("events", []):
            _add(ev)

    for ev in data.get("events", []):
        _add(ev)

    ordered.sort(key=lambda x: x[0])
    return [ev for _, ev in ordered] + unordered

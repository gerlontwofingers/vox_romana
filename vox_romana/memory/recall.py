"""Recall — unified memory query interface.

The single entry point for all memory retrieval.  Combines three tiers:

    1. **LM synthesis** — The knowledge LM generates what it "remembers"
       about the query.  This is associative, lossy, generative — like
       real memory.  Runs first to provide a hypothesis.

    2. **Vector search** — Persistent Chroma index (preferred) or an
       in-memory cosine index (fallback).  Indexes catalog runs/studies
       AND all ingested documents (from ``data/memory/documents/``).

    3. **FTS5 fallback** — If no vector index is available, falls back to
       the catalog's built-in FTS5 full-text search.  Always available.

USAGE:
    from vox_romana.memory.recall import MemoryRecall
    from vox_romana.core.catalog import Catalog

    recall = MemoryRecall(catalog=Catalog())
    result = recall.query("UBI employment effects")

    print(result.synthesis)       # LM-generated memory (or "")
    print(result.passages)        # Ranked (text, score, source_id) tuples
    print(result.catalog_studies) # StudyRecord list from catalog
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.catalog import Catalog, StudyRecord
from .vector_index import ChromaVectorIndex, _CosineFallbackIndex

logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """Combined output from all memory tiers."""

    synthesis: str = ""
    """LM-generated memory synthesis (empty if LM not available)."""

    passages: list[tuple[str, float, str]] = field(default_factory=list)
    """Ranked (text, similarity_score, source_id) from vector search."""

    catalog_studies: list[StudyRecord] = field(default_factory=list)
    """StudyRecords from catalog (via vector or FTS5 search)."""

    catalog_runs: list[Any] = field(default_factory=list)
    """RunRecords from catalog search."""

    method: str = "fts5"
    """Search method used: ``"chroma"``, ``"semantic"``, ``"fts5"``, or ``"none"``."""


# ─────────────────────────────────────────────────────────────────────────────
# MemoryRecall
# ─────────────────────────────────────────────────────────────────────────────

class MemoryRecall:
    """Unified memory query interface.

    Priority:
        1. LM synthesis (if lm provided)
        2. Chroma persistent vector search (if chromadb installed and use_chroma=True)
        3. In-memory cosine search (if encoder provided and Chroma unavailable)
        4. FTS5 fallback (always available when catalog provided)

    Args:
        catalog:       The vox_romana Catalog for FTS5 and study lookup.
        encoder:       Optional KnowledgeEncoder (used for in-memory fallback).
        lm:            Optional KnowledgeLM for synthesis.
        use_chroma:    If True (default), use Chroma when available.
        _chroma_client: Injected Chroma client for testing (e.g.
                       ``chromadb.EphemeralClient()``).  When provided, the
                       Chroma index runs in-memory without touching disk.
    """

    def __init__(
        self,
        catalog: Catalog | None = None,
        *,
        encoder: Any | None = None,
        lm: Any | None = None,
        use_chroma: bool = True,
        _chroma_client: Any | None = None,
    ) -> None:
        self._catalog = catalog
        self._encoder = encoder
        self._lm = lm
        self._use_chroma = use_chroma
        self._chroma_client = _chroma_client
        self._index: ChromaVectorIndex | _CosineFallbackIndex | None = None
        self._indexed = False

    def _ensure_index(self) -> None:
        """Initialise the vector index (lazy, once per MemoryRecall instance).

        Tries Chroma first; falls back to the in-memory cosine index.
        When Chroma is used, also migrates any ingested JSONL documents.
        """
        if self._indexed:
            return

        # ── Option 1: Chroma persistent index ────────────────────────────
        if self._use_chroma and ChromaVectorIndex.is_available():
            index = ChromaVectorIndex(_client=self._chroma_client)

            # Migrate ingested documents (fast after first run — marker file).
            try:
                from .ingest import DEFAULT_DOCUMENT_DIR
                index.migrate_documents(DEFAULT_DOCUMENT_DIR)
            except Exception as exc:
                logger.warning("Document migration into Chroma failed: %s", exc)

            # Add catalog entries (only new ones — batch existence check).
            if self._catalog is not None:
                try:
                    entries: list[tuple[str, str, str]] = []
                    for run in self._catalog.recent(limit=10_000):
                        if run.brief:
                            entries.append((run.brief, run.run_id, "catalog_run"))
                        for study in self._catalog.get_studies(run.run_id):
                            if study.findings:
                                entries.append(
                                    (study.findings, study.study_id, "catalog_study")
                                )
                    added = index.add_catalog_entries(entries)
                    if added:
                        logger.info(
                            "Added %d new catalog entries to Chroma index", added
                        )
                except Exception as exc:
                    logger.warning("Failed to index catalog into Chroma: %s", exc)

            self._index = index
            self._indexed = True
            logger.info("Using Chroma vector index (%d entries)", index.count())
            return

        # ── Option 2: In-memory cosine index (encoder required) ──────────
        if self._encoder is None or self._catalog is None:
            self._indexed = True  # nothing to build — skip on future calls
            return

        index_fb = _CosineFallbackIndex(self._encoder)
        for run in self._catalog.recent(limit=10_000):
            if run.brief:
                index_fb.add(run.brief, run.run_id, "catalog_run")
            for study in self._catalog.get_studies(run.run_id):
                if study.findings:
                    index_fb.add(study.findings, study.study_id, "catalog_study")
        index_fb.build()
        self._index = index_fb
        self._indexed = True
        logger.info(
            "Built in-memory fallback index with %d entries", index_fb.size
        )

    def query(
        self,
        text: str,
        *,
        top_k: int = 10,
        synthesize: bool = True,
    ) -> RecallResult:
        """Query memory for relevant knowledge.

        Args:
            text:       The query / research brief.
            top_k:      Max results from vector or FTS5 search.
            synthesize: If True and LM is available, generate a synthesis.

        Returns:
            RecallResult with synthesis, passages, and catalog records.
        """
        result = RecallResult()

        # ── Tier 1: LM synthesis ──────────────────────────────────────
        if synthesize and self._lm is not None:
            try:
                result.synthesis = self._lm.generate(
                    f"What do you know about: {text}"
                )
            except Exception as e:
                logger.warning("LM synthesis failed: %s", e)

        # ── Tier 2: Vector search (Chroma or in-memory fallback) ──────
        try:
            self._ensure_index()
        except Exception as e:
            logger.warning("Failed to build search index: %s", e)

        if self._index is not None:
            try:
                passages = self._index.search(text, top_k=top_k)
                if passages:
                    result.passages = passages
                    result.method = (
                        "chroma"
                        if isinstance(self._index, ChromaVectorIndex)
                        else "semantic"
                    )
                    # Look up catalog records for catalog source_types.
                    study_ids: set[str] = set()
                    run_ids: set[str] = set()
                    for _, _, source in passages:
                        study_ids.add(source)
                        run_ids.add(source)

                    if self._catalog:
                        for run in self._catalog.recent(limit=10_000):
                            if run.run_id in run_ids:
                                result.catalog_runs.append(run)
                                for study in self._catalog.get_studies(run.run_id):
                                    if study.study_id in study_ids:
                                        result.catalog_studies.append(study)
            except Exception as e:
                logger.warning(
                    "Vector search failed, falling back to FTS5: %s", e
                )

        # ── Tier 3: FTS5 fallback ─────────────────────────────────────
        if not result.passages and self._catalog is not None:
            try:
                fts_runs = self._catalog.search(text, limit=top_k)
                result.catalog_runs = fts_runs
                result.method = "fts5"
                for run in fts_runs:
                    studies = self._catalog.get_studies(run.run_id)
                    result.catalog_studies.extend(studies)
            except Exception as e:
                logger.warning("FTS5 search failed: %s", e)
                result.method = "none"

        return result

    def format_context(self, result: RecallResult, *, max_chars: int = 3000) -> str:
        """Format a RecallResult into a text context block for LLM prompts.

        Args:
            result:    The RecallResult to format.
            max_chars: Maximum characters in the output.

        Returns:
            Formatted text suitable for insertion into a system prompt.
        """
        parts: list[str] = []

        if result.synthesis:
            parts.append(f"MEMORY SYNTHESIS:\n{result.synthesis}")

        if result.catalog_studies:
            lines = ["RELATED STUDIES:"]
            for study in result.catalog_studies[:10]:
                lines.append(
                    f"- [{study.content_type}] {study.method}: "
                    f"{study.findings[:200]} "
                    f"(dataset: {study.dataset}, {study.created_at[:10]})"
                )
            parts.append("\n".join(lines))

        if result.passages and not result.catalog_studies:
            lines = ["SEMANTIC MATCHES:"]
            for text, score, source_id in result.passages[:5]:
                lines.append(f"- [{score:.3f}] {text[:200]}")
            parts.append("\n".join(lines))

        combined = "\n\n".join(parts)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n... (truncated)"
        return combined or "(no prior knowledge found)"



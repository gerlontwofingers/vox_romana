"""Recall — unified memory query interface.

The single entry point for all memory retrieval.  Combines three tiers:

    1. **LM synthesis** — The knowledge LM generates what it "remembers"
       about the query.  This is associative, lossy, generative — like
       real memory.  Runs first to provide a hypothesis.

    2. **Semantic search** — The encoder finds the most similar passages
       from the catalog via embedding cosine similarity.  Returns ranked
       results with scores.

    3. **FTS5 fallback** — If no encoder is trained, falls back to the
       catalog's built-in FTS5 full-text search.  Always available.

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

logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """Combined output from all memory tiers."""

    synthesis: str = ""
    """LM-generated memory synthesis (empty if LM not available)."""

    passages: list[tuple[str, float, str]] = field(default_factory=list)
    """Ranked (text, similarity_score, source_id) from encoder search."""

    catalog_studies: list[StudyRecord] = field(default_factory=list)
    """StudyRecords from catalog (via FTS5 or semantic search)."""

    catalog_runs: list[Any] = field(default_factory=list)
    """RunRecords from catalog search."""

    method: str = "fts5"
    """Which search method was used: "semantic", "fts5", or "none"."""


# ─────────────────────────────────────────────────────────────────────────────
# Embedding index — in-memory for now, SQLite-backed later if needed.
# ─────────────────────────────────────────────────────────────────────────────

class _EmbeddingIndex:
    """Simple in-memory cosine similarity index over catalog content.

    Indexes study findings + run briefs into a numpy matrix.  On query,
    encodes the query and returns top-k by cosine similarity.
    """

    def __init__(self, encoder) -> None:
        self._encoder = encoder
        self._texts: list[str] = []
        self._source_ids: list[str] = []
        self._source_types: list[str] = []
        self._embeddings = None  # numpy (n, dim) matrix

    @property
    def size(self) -> int:
        return len(self._texts)

    def add(self, text: str, source_id: str, source_type: str = "") -> None:
        self._texts.append(text)
        self._source_ids.append(source_id)
        self._source_types.append(source_type)

    def build(self) -> None:
        """Encode all texts into the embedding matrix."""
        if not self._texts:
            return
        import numpy as np
        self._embeddings = self._encoder.encode(self._texts)
        # Normalize for cosine similarity via dot product.
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._embeddings = self._embeddings / norms

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float, str]]:
        """Find top-k most similar passages.

        Returns:
            List of (text, cosine_similarity, source_id) sorted by score desc.
        """
        if self._embeddings is None or len(self._texts) == 0:
            return []

        import numpy as np
        q_emb = self._encoder.encode(query)
        q_emb = q_emb / (np.linalg.norm(q_emb) or 1.0)
        scores = self._embeddings @ q_emb
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((
                self._texts[idx],
                float(scores[idx]),
                self._source_ids[idx],
            ))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# MemoryRecall
# ─────────────────────────────────────────────────────────────────────────────

class MemoryRecall:
    """Unified memory query interface.

    Combines LM synthesis, semantic encoder search, and FTS5 fallback.
    Gracefully degrades: if no encoder/LM is trained, uses FTS5 only.

    Args:
        catalog: The vox_romana catalog for FTS5 and study lookup.
        encoder: Optional trained KnowledgeEncoder.
        lm: Optional trained KnowledgeLM.
    """

    def __init__(
        self,
        catalog: Catalog | None = None,
        *,
        encoder: Any | None = None,
        lm: Any | None = None,
    ) -> None:
        self._catalog = catalog
        self._encoder = encoder
        self._lm = lm
        self._index: _EmbeddingIndex | None = None
        self._indexed = False

    def _ensure_index(self) -> None:
        """Build the embedding index from catalog content (lazy, once)."""
        if self._indexed or self._encoder is None or self._catalog is None:
            return

        self._index = _EmbeddingIndex(self._encoder)

        # Index all studies.
        all_runs = self._catalog.recent(limit=10000)
        for run in all_runs:
            if run.brief:
                self._index.add(run.brief, run.run_id, "run")
            studies = self._catalog.get_studies(run.run_id)
            for study in studies:
                if study.findings:
                    self._index.add(study.findings, study.study_id, "study")

        self._index.build()
        self._indexed = True
        logger.info("Built embedding index with %d entries", self._index.size)

    def query(
        self,
        text: str,
        *,
        top_k: int = 10,
        synthesize: bool = True,
    ) -> RecallResult:
        """Query memory for relevant knowledge.

        Args:
            text: The query / research brief.
            top_k: Max results from semantic or FTS5 search.
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

        # ── Tier 2: Semantic search ───────────────────────────────────
        if self._encoder is not None:
            try:
                self._ensure_index()
                if self._index is not None:
                    result.passages = self._index.search(text, top_k=top_k)
                    result.method = "semantic"

                    # Look up actual study/run records for the top hits.
                    study_ids = set()
                    run_ids = set()
                    for _, _, source_id in result.passages:
                        study_ids.add(source_id)
                        run_ids.add(source_id)

                    if self._catalog:
                        for run in self._catalog.recent(limit=10000):
                            if run.run_id in run_ids:
                                result.catalog_runs.append(run)
                                for study in self._catalog.get_studies(run.run_id):
                                    if study.study_id in study_ids:
                                        result.catalog_studies.append(study)
            except Exception as e:
                logger.warning("Semantic search failed, falling back to FTS5: %s", e)

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
            result: The RecallResult to format.
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

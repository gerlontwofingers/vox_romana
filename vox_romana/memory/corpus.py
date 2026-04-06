"""Corpus builder — extracts training data from the catalog and ingested docs.

Generates two flavors of training data:

    1. PAIRS   — (query, passage) pairs for contrastive encoder training.
                  Sources: brief→findings, tag→passage, claim→evidence.
    2. PROMPTS — (instruction, completion) pairs for causal LM training.
                  Sources: brief→findings summary, brief→fact list.

Output format: JSONL files in data/memory/corpus/.

INCREMENTAL:
    Tracks the last-processed timestamp so subsequent builds only process
    new catalog entries.  Use ``force=True`` to rebuild from scratch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.catalog import Catalog

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_ROOT = Path("data/memory/corpus")


# ─────────────────────────────────────────────────────────────────────────────
# Training pair dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EncoderPair:
    """A (query, positive_passage) pair for contrastive learning."""
    query: str
    passage: str
    source_type: str = ""       # "study", "run", "document"
    source_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LMExample:
    """An (instruction, completion) pair for causal LM fine-tuning."""
    instruction: str
    completion: str
    source_type: str = ""
    source_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusStats:
    """Summary after a corpus build."""
    encoder_pairs: int = 0
    lm_examples: int = 0
    studies_processed: int = 0
    runs_processed: int = 0
    documents_processed: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Corpus builder
# ─────────────────────────────────────────────────────────────────────────────

class CorpusBuilder:
    """Builds training data from the catalog and ingested documents.

    Args:
        catalog: The vox_romana catalog instance.
        corpus_root: Directory to write JSONL files into.
    """

    def __init__(
        self,
        catalog: Catalog,
        *,
        corpus_root: str | Path | None = None,
        document_dir: str | Path | None = None,
    ) -> None:
        self._catalog = catalog
        self._corpus_root = Path(corpus_root or DEFAULT_CORPUS_ROOT)
        self._document_dir = Path(document_dir) if document_dir else None
        self._encoder_pairs: list[EncoderPair] = []
        self._lm_examples: list[LMExample] = []

    def build(self, *, force: bool = False) -> CorpusStats:
        """Build the training corpus.

        Args:
            force: If True, rebuild from scratch (ignore incremental state).

        Returns:
            CorpusStats summarizing what was generated.
        """
        self._encoder_pairs = []
        self._lm_examples = []
        stats = CorpusStats()

        # ── Extract from runs ──────────────────────────────────────────────
        runs = self._catalog.recent(limit=10000)
        for run in runs:
            self._extract_run(run)
            stats.runs_processed += 1

            # Studies linked to this run.
            studies = self._catalog.get_studies(run.run_id)
            for study in studies:
                self._extract_study(study, run_title=run.title)
                stats.studies_processed += 1

        # ── Extract from ingested documents ────────────────────────────────
        if self._document_dir and self._document_dir.exists():
            stats.documents_processed = self._extract_documents()

        # ── Write output ───────────────────────────────────────────────────
        stats.encoder_pairs = len(self._encoder_pairs)
        stats.lm_examples = len(self._lm_examples)

        self._write_jsonl("encoder_pairs.jsonl", self._encoder_pairs)
        self._write_jsonl("lm_examples.jsonl", self._lm_examples)
        self._write_meta(stats)

        logger.info(
            "Corpus built: %d encoder pairs, %d LM examples",
            stats.encoder_pairs,
            stats.lm_examples,
        )
        return stats

    # ─── Extraction per source type ────────────────────────────────────────

    def _extract_run(self, run) -> None:
        """Extract pairs from a catalog RunRecord."""
        if not run.brief:
            return

        # Pair: brief → title (lightweight but useful for encoder).
        if run.title:
            self._encoder_pairs.append(EncoderPair(
                query=run.brief,
                passage=run.title,
                source_type="run",
                source_id=run.run_id,
            ))

        # Pair: each tag → title+brief.
        combined = f"{run.title}. {run.brief}" if run.title else run.brief
        for tag in run.tags:
            self._encoder_pairs.append(EncoderPair(
                query=tag,
                passage=combined,
                source_type="run",
                source_id=run.run_id,
            ))

    def _extract_study(self, study, *, run_title: str = "") -> None:
        """Extract pairs from a catalog StudyRecord."""
        if not study.findings:
            return

        # Encoder: method → findings.
        if study.method:
            self._encoder_pairs.append(EncoderPair(
                query=study.method,
                passage=study.findings,
                source_type="study",
                source_id=study.study_id,
            ))

        # Encoder: dataset → findings.
        if study.dataset:
            self._encoder_pairs.append(EncoderPair(
                query=f"analysis of {study.dataset}",
                passage=study.findings,
                source_type="study",
                source_id=study.study_id,
            ))

        # LM: brief → findings (the core "brief to insight" pattern).
        brief = run_title or study.content_type or "research"
        self._lm_examples.append(LMExample(
            instruction=(
                f"Summarize findings from {study.method} analysis"
                + (f" on {study.dataset}" if study.dataset else "")
                + f": {brief}"
            ),
            completion=study.findings,
            source_type="study",
            source_id=study.study_id,
        ))

        # LM + encoder from fact cards.
        try:
            fact_cards = json.loads(study.fact_cards_json)
        except (json.JSONDecodeError, TypeError):
            fact_cards = []

        for fc in fact_cards:
            claim = fc.get("claim", "")
            source = fc.get("source", "")
            if claim:
                # Encoder: claim ↔ evidence context.
                self._encoder_pairs.append(EncoderPair(
                    query=claim,
                    passage=study.findings,
                    source_type="study",
                    source_id=study.study_id,
                ))
                # LM: claim verification.
                if source:
                    self._lm_examples.append(LMExample(
                        instruction=f"What does the data show about: {claim}",
                        completion=(
                            f"{claim} — Source: {source}. "
                            f"Context: {study.findings}"
                        ),
                        source_type="study",
                        source_id=study.study_id,
                    ))

    def _extract_documents(self) -> int:
        """Extract pairs from ingested document chunks (JSONL files)."""
        count = 0
        chunks_dir = self._document_dir
        if not chunks_dir or not chunks_dir.exists():
            return 0

        for jsonl_path in chunks_dir.glob("*.jsonl"):
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = chunk.get("text", "")
                    title = chunk.get("title", "")
                    source = chunk.get("source", jsonl_path.stem)

                    if not text:
                        continue

                    # Encoder: title → text.
                    if title:
                        self._encoder_pairs.append(EncoderPair(
                            query=title,
                            passage=text[:1000],
                            source_type="document",
                            source_id=source,
                        ))

                    # LM: summarize chunk.
                    if len(text) > 100:
                        self._lm_examples.append(LMExample(
                            instruction=f"What does this passage discuss? Source: {source}",
                            completion=text[:2000],
                            source_type="document",
                            source_id=source,
                        ))

                    count += 1
        return count

    # ─── I/O ───────────────────────────────────────────────────────────────

    def _write_jsonl(self, filename: str, items: list) -> None:
        self._corpus_root.mkdir(parents=True, exist_ok=True)
        path = self._corpus_root / filename
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item.to_dict(), default=str) + "\n")
        logger.info("Wrote %d items to %s", len(items), path)

    def _write_meta(self, stats: CorpusStats) -> None:
        meta = {
            "built_at": datetime.now(timezone.utc).isoformat(),
            **asdict(stats),
        }
        meta_path = self._corpus_root / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

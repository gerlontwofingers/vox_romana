"""Trainer — orchestrates corpus building and model training.

Provides a single ``train()`` entry point that runs the full pipeline:
    1. Build corpus from catalog + ingested documents
    2. Train encoder (sentence-transformer, contrastive)
    3. Train knowledge LM (LoRA adapter, causal)

Each step can be run independently or skipped.  Training is idempotent —
re-running overwrites previous checkpoints.

USAGE:
    from vox_romana.core.catalog import Catalog
    from vox_romana.memory.trainer import MemoryTrainer

    trainer = MemoryTrainer(catalog=Catalog())
    result = trainer.train()  # full pipeline
    result = trainer.train(skip_lm=True)  # encoder only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.catalog import Catalog
from .corpus import CorpusBuilder, CorpusStats, DEFAULT_CORPUS_ROOT
from .encoder import KnowledgeEncoder, DEFAULT_ENCODER_DIR
from .ingest import DEFAULT_DOCUMENT_DIR
from .lm import KnowledgeLM, DEFAULT_LM_DIR

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Summary of a full training run."""

    corpus: CorpusStats | None = None
    encoder: dict[str, Any] = field(default_factory=dict)
    lm: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


class MemoryTrainer:
    """Orchestrates the full memory training pipeline.

    Args:
        catalog: The vox_romana catalog to extract knowledge from.
        memory_root: Root directory for all memory data (corpus, models).
        encoder_base: HuggingFace model name for the encoder base.
        lm_base: HuggingFace model name for the LM base.
    """

    def __init__(
        self,
        catalog: Catalog,
        *,
        memory_root: str | Path | None = None,
        encoder_base: str = "all-MiniLM-L6-v2",
        lm_base: str = "Qwen/Qwen2-1.5B",
    ) -> None:
        self._catalog = catalog
        root = Path(memory_root) if memory_root else Path("data/memory")
        self._corpus_root = root / "corpus"
        self._document_dir = root / "documents"
        self._encoder_dir = root / "encoder"
        self._lm_dir = root / "lm"
        self._encoder_base = encoder_base
        self._lm_base = lm_base

    def train(
        self,
        *,
        skip_corpus: bool = False,
        skip_encoder: bool = False,
        skip_lm: bool = False,
        encoder_epochs: int = 3,
        encoder_batch_size: int = 32,
        lm_epochs: int = 3,
        lm_batch_size: int = 4,
    ) -> TrainingResult:
        """Run the full training pipeline.

        Args:
            skip_corpus: Skip corpus building (use existing corpus files).
            skip_encoder: Skip encoder training.
            skip_lm: Skip LM training.
            encoder_epochs: Epochs for encoder training.
            encoder_batch_size: Batch size for encoder training.
            lm_epochs: Epochs for LM training.
            lm_batch_size: Batch size for LM training.

        Returns:
            TrainingResult with stats from each step.
        """
        result = TrainingResult()

        # ── Step 1: Build corpus ──────────────────────────────────────
        if not skip_corpus:
            try:
                builder = CorpusBuilder(
                    self._catalog,
                    corpus_root=self._corpus_root,
                    document_dir=self._document_dir,
                )
                result.corpus = builder.build(force=True)
                logger.info(
                    "Corpus: %d encoder pairs, %d LM examples",
                    result.corpus.encoder_pairs,
                    result.corpus.lm_examples,
                )
            except Exception as e:
                msg = f"Corpus build failed: {e}"
                logger.error(msg)
                result.errors.append(msg)
                return result

        # ── Step 2: Train encoder ─────────────────────────────────────
        pairs_path = self._corpus_root / "encoder_pairs.jsonl"
        if not skip_encoder and pairs_path.exists():
            try:
                encoder = KnowledgeEncoder(
                    model_dir=self._encoder_dir,
                    base_model_name=self._encoder_base,
                )
                result.encoder = encoder.train(
                    pairs_path,
                    epochs=encoder_epochs,
                    batch_size=encoder_batch_size,
                )
            except ImportError:
                msg = (
                    "Encoder training skipped: sentence-transformers not installed. "
                    "Install with: pip install -r requirements-data-deep.txt"
                )
                logger.warning(msg)
                result.encoder = {"status": "skipped", "reason": msg}
            except Exception as e:
                msg = f"Encoder training failed: {e}"
                logger.error(msg)
                result.errors.append(msg)
        elif skip_encoder:
            result.encoder = {"status": "skipped"}

        # ── Step 3: Train LM ─────────────────────────────────────────
        examples_path = self._corpus_root / "lm_examples.jsonl"
        if not skip_lm and examples_path.exists():
            try:
                lm = KnowledgeLM(
                    adapter_dir=self._lm_dir,
                    base_model_name=self._lm_base,
                )
                result.lm = lm.train(
                    examples_path,
                    epochs=lm_epochs,
                    batch_size=lm_batch_size,
                )
            except ImportError:
                msg = (
                    "LM training skipped: transformers/peft not installed. "
                    "Install with: pip install -r requirements-data-deep.txt"
                )
                logger.warning(msg)
                result.lm = {"status": "skipped", "reason": msg}
            except Exception as e:
                msg = f"LM training failed: {e}"
                logger.error(msg)
                result.errors.append(msg)
        elif skip_lm:
            result.lm = {"status": "skipped"}

        return result

    def build_corpus_only(self) -> CorpusStats:
        """Just build the corpus without training any models."""
        builder = CorpusBuilder(
            self._catalog,
            corpus_root=self._corpus_root,
            document_dir=self._document_dir,
        )
        return builder.build(force=True)

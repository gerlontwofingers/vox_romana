"""Encoder — fine-tunes a sentence-transformer for semantic recall.

Takes encoder pairs (query, passage) from the corpus builder and trains
a contrastive model that understands YOUR semantic space.  After training,
``encode(text)`` returns an embedding that can be compared via cosine
similarity for recall.

BASE MODEL:
    Default: ``all-MiniLM-L6-v2`` (~22 MB, 384-dim).  Good tradeoff between
    speed and quality for a personal knowledge base.  Override via
    ``base_model_name``.

TRAINING:
    Uses MultipleNegativesRankingLoss — each (query, passage) pair is a
    positive, and other passages in the batch serve as negatives.  This
    works well with small corpora because it doesn't need explicit negatives.

DEPENDENCY SPECS:
    sentence-transformers>=5.4,<5.5  (in requirements.txt)
    torch>=2.4,<2.6                  (in requirements.txt)

STORAGE:
    Trained model saved to ``data/memory/encoder/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from vox_romana.core.paths import ENCODER_DIR as DEFAULT_ENCODER_DIR
DEFAULT_BASE_MODEL = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports — all heavy deps are optional.
# ─────────────────────────────────────────────────────────────────────────────

def _load_deps():
    """Returns (SentenceTransformer, MultipleNegativesRankingLoss, Trainer, TrainingArgs) or raises."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    return SentenceTransformer, MultipleNegativesRankingLoss, SentenceTransformerTrainer, SentenceTransformerTrainingArguments


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeEncoder
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeEncoder:
    """Sentence-transformer fine-tuned on your knowledge corpus.

    Args:
        model_dir: Directory containing a trained model (or where to save).
        base_model_name: HuggingFace model to start from if no trained model.
    """

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        base_model_name: str = DEFAULT_BASE_MODEL,
    ) -> None:
        self._model_dir = Path(model_dir or DEFAULT_ENCODER_DIR)
        self._base_model_name = base_model_name
        self._model = None

    @property
    def is_trained(self) -> bool:
        """True if a trained model exists on disk."""
        return (self._model_dir / "config.json").exists()

    def load(self) -> None:
        """Load the trained model (or base model if not trained yet)."""
        SentenceTransformer, *_ = _load_deps()

        if self.is_trained:
            self._model = SentenceTransformer(str(self._model_dir))
            logger.info("Loaded trained encoder from %s", self._model_dir)
        else:
            self._model = SentenceTransformer(self._base_model_name)
            logger.info("Loaded base encoder: %s", self._base_model_name)

    def encode(self, texts: str | list[str]) -> Any:
        """Encode text(s) into embeddings.

        Args:
            texts: A single string or list of strings.

        Returns:
            numpy array of shape (n, dim) or (dim,) for a single string.
        """
        if self._model is None:
            self.load()
        return self._model.encode(texts, convert_to_numpy=True)

    def train(
        self,
        pairs_path: str | Path,
        *,
        epochs: int = 3,
        batch_size: int = 32,
        warmup_fraction: float = 0.1,
        learning_rate: float = 2e-5,
    ) -> dict[str, Any]:
        """Fine-tune the encoder on (query, passage) pairs.

        Args:
            pairs_path: Path to encoder_pairs.jsonl from corpus builder.
            epochs: Training epochs.
            batch_size: Batch size.
            warmup_fraction: Fraction of steps for linear warmup.
            learning_rate: Peak learning rate.

        Returns:
            Dict with training stats (pairs_loaded, epochs, etc.).
        """
        from datasets import Dataset
        SentenceTransformer, MultipleNegativesRankingLoss, SentenceTransformerTrainer, SentenceTransformerTrainingArguments = _load_deps()

        # Load base model.
        model = SentenceTransformer(self._base_model_name)

        # Load pairs from JSONL.
        pairs_path = Path(pairs_path)
        rows = []
        with open(pairs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                rows.append({"anchor": pair["query"], "positive": pair["passage"]})

        if not rows:
            logger.warning("No encoder pairs found in %s", pairs_path)
            return {"pairs_loaded": 0, "status": "skipped"}

        logger.info("Training encoder on %d pairs for %d epochs", len(rows), epochs)

        dataset = Dataset.from_list(rows)
        loss = MultipleNegativesRankingLoss(model)

        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(self._model_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            warmup_ratio=warmup_fraction,
            learning_rate=learning_rate,
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            loss=loss,
        )
        trainer.train()
        model.save_pretrained(str(self._model_dir))

        self._model = model
        logger.info("Encoder saved to %s", self._model_dir)

        return {
            "pairs_loaded": len(rows),
            "epochs": epochs,
            "batch_size": batch_size,
            "output_dir": str(self._model_dir),
            "status": "trained",
        }

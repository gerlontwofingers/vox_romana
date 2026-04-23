"""SemanticFlowAgent — Pearson r between topic drift and emotion drift.

For each pair of consecutive segments computes:
    - topic_shift[i]:   cosine distance between segment embeddings i and i+1
    - emotion_shift[i]: cosine distance between 7-dim emotion score vectors

Then correlates the two shift sequences via Pearson r (scipy.stats.pearsonr).

Uses CorpusEmbedder.embed_texts() for semantic embeddings; inject _embed_fn
for testing without loading any model.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from .models import EMOTION_LABELS, EmotionSequence, SemanticFlowResult

logger = logging.getLogger(__name__)


class SemanticFlowAgent:
    """Compute semantic-emotion flow correlation for an EmotionSequence.

    Args:
        cache_dir:  Passed to CorpusEmbedder for embedding cache.
        _embed_fn:  Optional callable(texts: list[str]) -> np.ndarray (N, dim)
                    that replaces the real embedder. Inject for testing.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        *,
        _embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._embed_fn = _embed_fn

    def compute(self, sequence: EmotionSequence) -> SemanticFlowResult:
        """Compute topic/emotion shifts and their Pearson correlation."""
        from scipy.stats import pearsonr

        texts = sequence.texts
        tags = sequence.tags

        if len(texts) < 2:
            return SemanticFlowResult(
                pearson_r=0.0,
                p_value=1.0,
                topic_shifts=[],
                emotion_shifts=[],
                n_pairs=0,
            )

        embeddings = self._embed(texts)            # (N, dim)
        topic_shifts = _cosine_distances(embeddings)     # length N-1
        emotion_shifts = _emotion_cosine_distances(tags) # length N-1

        n_pairs = len(topic_shifts)
        if n_pairs < 2:
            return SemanticFlowResult(
                pearson_r=0.0,
                p_value=1.0,
                topic_shifts=topic_shifts,
                emotion_shifts=emotion_shifts,
                n_pairs=n_pairs,
            )

        r, p = pearsonr(topic_shifts, emotion_shifts)
        # Handle zero-variance case (scipy returns NaN)
        if np.isnan(r):
            r, p = 0.0, 1.0

        return SemanticFlowResult(
            pearson_r=float(r),
            p_value=float(p),
            topic_shifts=topic_shifts,
            emotion_shifts=emotion_shifts,
            n_pairs=n_pairs,
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the injected function or CorpusEmbedder."""
        if self._embed_fn is not None:
            return np.asarray(self._embed_fn(texts), dtype=np.float32)
        from vox_romana.analysis.embedder import CorpusEmbedder
        embedder = CorpusEmbedder(cache_dir=self._cache_dir)
        return embedder.embed_texts(texts)


def _cosine_distances(embeddings: np.ndarray) -> list[float]:
    """Cosine distance between consecutive rows.

    Returns a list of length len(embeddings) - 1. Values in [0, 2].
    """
    distances: list[float] = []
    for i in range(len(embeddings) - 1):
        a = embeddings[i]
        b = embeddings[i + 1]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            distances.append(1.0)
        else:
            sim = float(np.dot(a, b) / (na * nb))
            distances.append(float(1.0 - float(np.clip(sim, -1.0, 1.0))))
    return distances


def _emotion_cosine_distances(tags: list) -> list[float]:
    """Cosine distance between consecutive 7-dim emotion score vectors."""
    distances: list[float] = []
    for i in range(len(tags) - 1):
        a = np.array(
            [tags[i].scores.get(l, 0.0) for l in EMOTION_LABELS], dtype=np.float32
        )
        b = np.array(
            [tags[i + 1].scores.get(l, 0.0) for l in EMOTION_LABELS], dtype=np.float32
        )
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            distances.append(1.0)
        else:
            sim = float(np.dot(a, b) / (na * nb))
            distances.append(float(1.0 - float(np.clip(sim, -1.0, 1.0))))
    return distances

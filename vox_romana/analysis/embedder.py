"""Corpus embedding — dense vector representations for SpeakerCorpus objects.

Backend priority:
    1. Chroma DefaultEmbeddingFunction (all-MiniLM-L6-v2 via onnxruntime,
       384-dim) — semantic, no sentence-transformers required.
    2. TF-IDF vectorizer (sklearn, always available) — sparse but functional.

For hypothesis testing, always use ``embed_pair()`` rather than ``embed()``
twice.  The TF-IDF backend fits its vocabulary on the combined corpus so both
speakers share the same feature space.  The Chroma backend is unaffected (each
utterance is embedded independently into the same 384-dim space).

All output is L2-normalised — cosine similarity equals dot product.

Results are cached on disk by SHA-256 of the corpus content, keyed by backend.

USAGE::

    from vox_romana.analysis.embedder import CorpusEmbedder
    from vox_romana.analysis.corpus import CorpusLoader

    corpora = CorpusLoader.from_vox_romana_json("debate.json")
    a, b = list(corpora.values())[:2]

    embedder = CorpusEmbedder()
    emb_a, emb_b = embedder.embed_pair(a, b)
    # emb_a.shape == (len(a.utterances), embedder.embedding_dim)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .models import SpeakerCorpus

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_SUBDIR = "analysis/embed_cache"


def _corpus_hash(corpus: SpeakerCorpus) -> str:
    """Return a 16-char hex digest of corpus content."""
    return hashlib.sha256(
        corpus.full_text.encode("utf-8", errors="replace")
    ).hexdigest()[:16]


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise rows, guarding against zero vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return embeddings / norms


class CorpusEmbedder:
    """Embeds SpeakerCorpus utterances into dense vectors.

    Args:
        cache_dir:       Directory for cached ``.npy`` files.  Defaults to
                         ``DATA_ROOT/analysis/embed_cache/``.
        _force_backend:  Force a specific backend — ``"chroma"`` or
                         ``"tfidf"``.  Used in tests to bypass auto-detection.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        _force_backend: str | None = None,
    ) -> None:
        self._force_backend = _force_backend
        self._detected_backend: str | None = None
        self._chroma_fn: Any | None = None

        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
        else:
            try:
                from vox_romana.core.paths import DATA_ROOT
                self._cache_dir = DATA_ROOT / _DEFAULT_CACHE_SUBDIR
            except ImportError:
                self._cache_dir = Path.home() / ".vox_romana" / _DEFAULT_CACHE_SUBDIR

    # ─────────────────────────────────────────────────────────────────────────
    # Public properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Active backend: ``"chroma"`` or ``"tfidf"``."""
        return self._get_backend()

    @property
    def embedding_dim(self) -> int:
        """Dimension of output vectors (384 for Chroma, ≤2000 for TF-IDF)."""
        if self._get_backend() == "chroma":
            return 384
        return 2000

    # ─────────────────────────────────────────────────────────────────────────
    # Core embedding API
    # ─────────────────────────────────────────────────────────────────────────

    def embed(self, corpus: SpeakerCorpus) -> np.ndarray:
        """Embed all utterances → ``(N, dim)`` float32, L2-normalised.

        Caches result by content hash.  For comparison tasks, prefer
        :meth:`embed_pair` to ensure a shared feature space (TF-IDF).

        Raises:
            ValueError: If the corpus has no utterances.
        """
        if not corpus.utterances:
            raise ValueError(f"Cannot embed empty corpus '{corpus.name}'.")

        cache_key = f"{self._get_backend()}_{_corpus_hash(corpus)}"
        cache_path = self._cache_dir / f"{cache_key}.npy"

        if cache_path.exists():
            logger.debug("Embed cache hit for '%s'", corpus.name)
            return np.load(str(cache_path))

        embeddings = self._embed_texts_internal(corpus.utterances)
        self._save_cache(cache_path, embeddings)
        return embeddings

    def embed_pair(
        self, a: SpeakerCorpus, b: SpeakerCorpus
    ) -> tuple[np.ndarray, np.ndarray]:
        """Embed two corpora in a shared feature space.

        For TF-IDF the vectorizer is fit on the combined vocabulary so both
        matrices are directly comparable.  For Chroma, equivalent to calling
        :meth:`embed` twice (shared space guaranteed by the model).

        Returns:
            ``(emb_a, emb_b)`` — float32 arrays, each ``(N, dim)``,
            L2-normalised.
        """
        if self._get_backend() == "chroma":
            return self.embed(a), self.embed(b)

        # TF-IDF: fit on union of both corpora, transform separately.
        from sklearn.feature_extraction.text import TfidfVectorizer

        combined = a.utterances + b.utterances
        vec = TfidfVectorizer(max_features=2000, sublinear_tf=True)
        vec.fit(combined)

        emb_a = _normalize(
            vec.transform(a.utterances).toarray().astype(np.float32)
        )
        emb_b = _normalize(
            vec.transform(b.utterances).toarray().astype(np.float32)
        )
        logger.debug(
            "TF-IDF embed_pair: vocab=%d  A=%s  B=%s",
            len(vec.vocabulary_),
            emb_a.shape,
            emb_b.shape,
        )
        return emb_a, emb_b

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed an arbitrary list of strings.  No caching.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("Cannot embed empty text list.")
        return self._embed_texts_internal(texts)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_backend(self) -> str:
        """Detect and cache the active backend."""
        if self._force_backend is not None:
            return self._force_backend

        if self._detected_backend is not None:
            return self._detected_backend

        try:
            from chromadb.utils.embedding_functions import (  # type: ignore[import]
                DefaultEmbeddingFunction,
            )
            self._chroma_fn = DefaultEmbeddingFunction()
            self._detected_backend = "chroma"
            logger.debug("CorpusEmbedder: chroma (all-MiniLM-L6-v2) backend")
        except Exception:
            self._detected_backend = "tfidf"
            logger.debug("CorpusEmbedder: chromadb unavailable, using TF-IDF backend")

        return self._detected_backend

    def _embed_texts_internal(self, texts: list[str]) -> np.ndarray:
        """Embed texts with the active backend. Always returns L2-norm float32."""
        backend = self._get_backend()

        if backend == "chroma":
            if self._chroma_fn is None:
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction  # type: ignore[import]
                self._chroma_fn = DefaultEmbeddingFunction()
            raw = self._chroma_fn(texts)
            embeddings = np.array(raw, dtype=np.float32)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=2000, sublinear_tf=True)
            embeddings = vec.fit_transform(texts).toarray().astype(np.float32)

        return _normalize(embeddings)

    def _save_cache(self, cache_path: Path, embeddings: np.ndarray) -> None:
        """Write embeddings to disk cache. Silently skips on I/O error."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), embeddings)
            logger.debug("Embed cached → %s", cache_path.name)
        except OSError as exc:
            logger.warning("Failed to cache embeddings: %s", exc)

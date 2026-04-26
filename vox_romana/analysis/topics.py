"""LDA topic modeling for SpeakerCorpus objects.

Always fits on the combined vocabulary of both corpora so topic indices
are shared and per-corpus topic distributions are directly comparable.

USAGE::

    from vox_romana.analysis.topics import TopicModeler
    from vox_romana.analysis.corpus import CorpusLoader

    corpora = CorpusLoader.from_vox_romana_json("debate.json")
    a, b = list(corpora.values())[:2]

    tm = TopicModeler(n_topics=10)
    dist_a, dist_b = tm.fit_transform(a, b)
    # dist_a.shape == (len(a.utterances), 10)

    print(tm.topic_label(0))          # e.g. "economy/tax/reform/income/budget"
    weights = tm.corpus_weights(dist_a)  # (10,) sum ≈ 1.0
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .models import SpeakerCorpus

logger = logging.getLogger(__name__)


class TopicModeler:
    """LDA topic model fit on the combined vocabulary of two SpeakerCorpora.

    All topic distributions produced by this instance share the same feature
    space and topic indices, making them directly comparable.

    Args:
        n_topics:     Number of latent topics.  Automatically capped to
                      ``max(2, n_combined_utterances - 1)`` at fit time.
        max_features: Vocabulary cap for :class:`~sklearn.feature_extraction
                      .text.CountVectorizer`.
        random_state: RNG seed for reproducible LDA fits.
    """

    def __init__(
        self,
        n_topics: int = 10,
        max_features: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.n_topics = n_topics
        self.max_features = max_features
        self.random_state = random_state
        self._vectorizer: CountVectorizer | None = None
        self._lda: LatentDirichletAllocation | None = None
        self._fitted_n_topics: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        """True if :meth:`fit_transform` has been called."""
        return self._lda is not None

    def fit_transform(
        self, corpus_a: SpeakerCorpus, corpus_b: SpeakerCorpus
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit LDA on the combined corpora and transform each separately.

        Args:
            corpus_a: First speaker's corpus.
            corpus_b: Second speaker's corpus.

        Returns:
            ``(dist_a, dist_b)`` — float32 arrays of shape
            ``(N_a, n_topics)`` and ``(N_b, n_topics)``.  Each row is a
            probability distribution over topics (sums to approximately 1.0).
        """
        combined = corpus_a.utterances + corpus_b.utterances
        na = len(corpus_a.utterances)

        # Cap n_topics to avoid LDA n_components > n_samples error
        n_topics = min(self.n_topics, max(2, len(combined) - 1))
        self._fitted_n_topics = n_topics

        self._vectorizer = CountVectorizer(
            max_features=self.max_features,
            stop_words="english",
        )
        X = self._vectorizer.fit_transform(combined)

        self._lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=self.random_state,
            max_iter=20,
            learning_method="batch",
        )
        all_topics = self._lda.fit_transform(X)

        dist_a = all_topics[:na].astype(np.float32)
        dist_b = all_topics[na:].astype(np.float32)

        logger.debug(
            "LDA fitted: n_topics=%d  vocab=%d  A=%s  B=%s",
            n_topics,
            len(self._vectorizer.vocabulary_),
            dist_a.shape,
            dist_b.shape,
        )
        return dist_a, dist_b

    def top_terms(self, topic_id: int, n_terms: int = 8) -> list[str]:
        """Return the top ``n_terms`` terms for a topic by component weight.

        Args:
            topic_id: Zero-based topic index.
            n_terms:  Number of terms to return.

        Returns:
            List of term strings, highest-weight first.

        Raises:
            RuntimeError: If the model has not been fitted yet.
            IndexError:   If ``topic_id`` is out of range.
        """
        self._require_fitted()
        feature_names = self._vectorizer.get_feature_names_out()  # type: ignore[union-attr]
        component = self._lda.components_[topic_id]  # type: ignore[index]
        top_indices = np.argsort(component)[::-1][:n_terms]
        return [feature_names[i] for i in top_indices]

    def topic_label(self, topic_id: int, n_terms: int = 5) -> str:
        """Return a slash-joined label for a topic: ``"term1/term2/term3"``."""
        return "/".join(self.top_terms(topic_id, n_terms))

    def corpus_weights(self, topic_dist: np.ndarray) -> np.ndarray:
        """Compute mean per-topic weight vector for a corpus.

        Args:
            topic_dist: ``(N_utterances, n_topics)`` array from
                        :meth:`fit_transform`.

        Returns:
            ``(n_topics,)`` float32 array.  Values are non-negative and sum
            to approximately 1.0.  Returns a uniform distribution as fallback
            if the sum is zero (degenerate corpus).
        """
        weights = topic_dist.mean(axis=0).astype(np.float32)
        total = float(weights.sum())
        if total > 0.0:
            weights = weights / total
        else:
            n = len(weights)
            weights = np.ones(n, dtype=np.float32) / max(n, 1)
        return weights

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "TopicModeler is not fitted.  Call fit_transform() first."
            )

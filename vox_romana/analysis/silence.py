"""Silence / topic-avoidance detection between two SpeakerCorpus objects.

"Silence" in rhetorical analysis means a topic that is systematically absent
from a speaker's discourse — either because both speakers avoid it, or because
one speaker avoids it while the other engages with it.

The detector works in topic-weight space: a topic is considered "engaged" by a
corpus if its mean corpus weight exceeds ``engage_threshold``, and "silent"
(avoided) if it falls below ``silence_threshold``.

AVOIDANCE CATEGORIES
    avoided_by_both  — mean weight < silence_threshold for BOTH corpora
    avoided_by_a     — B engages (≥ engage_threshold) but A does not
    avoided_by_b     — A engages (≥ engage_threshold) but B does not
    reference_topics_missed — topics in a reference corpus that neither
                              speaker addresses (optional)

USAGE::

    from vox_romana.analysis.silence import SilenceDetector

    detector = SilenceDetector()
    report = detector.detect(corpus_a, corpus_b, topic_modeler, dist_a, dist_b)
    print(report.avoided_by_both)   # list of topic labels
    print(report.avoided_by_a)

    # With optional reference corpus (e.g., a policy agenda)
    report = detector.detect(
        corpus_a, corpus_b, topic_modeler, dist_a, dist_b,
        reference_corpus=reference_corpus
    )
    print(report.reference_topics_missed)
"""

from __future__ import annotations

import logging

import numpy as np

from .models import SilenceReport, SpeakerCorpus
from .topics import TopicModeler

logger = logging.getLogger(__name__)


class SilenceDetector:
    """Detects topics systematically avoided by one or both speakers.

    Args:
        silence_threshold: A corpus is considered "silent" on a topic if
                           its mean weight is below this value.  Default 0.05.
        engage_threshold:  A corpus is considered to "engage" with a topic
                           if its mean weight is at or above this value.
                           Default 0.15.  Must be >= silence_threshold.
    """

    def __init__(
        self,
        *,
        silence_threshold: float = 0.05,
        engage_threshold: float = 0.15,
    ) -> None:
        if engage_threshold < silence_threshold:
            raise ValueError(
                f"engage_threshold ({engage_threshold}) must be >= "
                f"silence_threshold ({silence_threshold})."
            )
        self.silence_threshold = silence_threshold
        self.engage_threshold = engage_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect(
        self,
        corpus_a: SpeakerCorpus,
        corpus_b: SpeakerCorpus,
        topic_modeler: TopicModeler,
        dist_a: np.ndarray,
        dist_b: np.ndarray,
        *,
        reference_corpus: SpeakerCorpus | None = None,
        ref_dist: np.ndarray | None = None,
    ) -> SilenceReport:
        """Detect topic silences between two corpora.

        Args:
            corpus_a, corpus_b: Speaker corpora.
            topic_modeler:      Fitted :class:`TopicModeler`.
            dist_a, dist_b:     Per-utterance topic distributions from
                                ``topic_modeler.fit_transform()``.
            reference_corpus:   Optional external reference corpus.
                                Its topic distribution must be provided
                                as ``ref_dist``.
            ref_dist:           Per-utterance topic distributions for the
                                reference corpus.  Required when
                                ``reference_corpus`` is provided.

        Returns:
            :class:`~vox_romana.analysis.models.SilenceReport`.
        """
        topic_modeler._require_fitted()
        n_topics = dist_a.shape[1]

        w_a = topic_modeler.corpus_weights(dist_a)
        w_b = topic_modeler.corpus_weights(dist_b)

        avoided_by_both: list[str] = []
        avoided_by_a: list[str] = []
        avoided_by_b: list[str] = []

        for t in range(n_topics):
            label = topic_modeler.topic_label(t)
            wa = float(w_a[t])
            wb = float(w_b[t])

            a_silent = wa < self.silence_threshold
            b_silent = wb < self.silence_threshold
            a_engage = wa >= self.engage_threshold
            b_engage = wb >= self.engage_threshold

            if a_silent and b_silent:
                avoided_by_both.append(label)
            elif b_engage and a_silent:
                avoided_by_a.append(label)
            elif a_engage and b_silent:
                avoided_by_b.append(label)

        # Reference topics
        reference_topics_missed: list[str] = []
        if reference_corpus is not None and ref_dist is not None:
            w_ref = topic_modeler.corpus_weights(ref_dist)
            for t in range(n_topics):
                ref_engaged = float(w_ref[t]) >= self.engage_threshold
                both_silent = (
                    float(w_a[t]) < self.silence_threshold
                    and float(w_b[t]) < self.silence_threshold
                )
                if ref_engaged and both_silent:
                    reference_topics_missed.append(topic_modeler.topic_label(t))

        logger.info(
            "SilenceDetector: %s vs %s — "
            "both=%d  a_avoids=%d  b_avoids=%d  ref_missed=%d",
            corpus_a.name,
            corpus_b.name,
            len(avoided_by_both),
            len(avoided_by_a),
            len(avoided_by_b),
            len(reference_topics_missed),
        )

        return SilenceReport(
            avoided_by_both=avoided_by_both,
            avoided_by_a=avoided_by_a,
            avoided_by_b=avoided_by_b,
            reference_topics_missed=reference_topics_missed,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience: detect without pre-fitted distributions
    # ─────────────────────────────────────────────────────────────────────────

    def detect_from_corpora(
        self,
        corpus_a: SpeakerCorpus,
        corpus_b: SpeakerCorpus,
        *,
        n_topics: int = 10,
        random_state: int = 42,
    ) -> tuple[SilenceReport, TopicModeler]:
        """Convenience wrapper: fit a topic model internally and detect silences.

        Returns:
            ``(SilenceReport, fitted_TopicModeler)`` — the modeler is returned
            so callers can inspect topic labels.
        """
        tm = TopicModeler(n_topics=n_topics, random_state=random_state)
        dist_a, dist_b = tm.fit_transform(corpus_a, corpus_b)
        report = self.detect(corpus_a, corpus_b, tm, dist_a, dist_b)
        return report, tm

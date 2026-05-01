"""Data models for the emotion_flow pipeline.

All dataclasses use __slots__ for memory efficiency and are intentionally
immutable-friendly (no setattr after construction in pipeline code).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EMOTION_LABELS: list[str] = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


@dataclass(slots=True)
class SpeechSegment:
    """One paragraph-level unit of speech, translated to English."""

    text: str
    original_text: str
    source_lang: str
    translation_confidence: float
    translation_flagged: bool
    segment_index: int
    speech_id: str


@dataclass(slots=True)
class EmotionTag:
    """Emotion classification result for a single SpeechSegment."""

    dominant: str
    dominant_score: float
    scores: dict[str, float]


@dataclass(slots=True)
class EmotionSequence:
    """Ordered sequence of tagged segments for a single leader."""

    leader_name: str
    segments: list[SpeechSegment]
    tags: list[EmotionTag]
    speech_ids: list[str]

    @classmethod
    def from_pooled(cls, sequences: list[EmotionSequence]) -> EmotionSequence:
        """Merge multiple sequences (same leader, different speeches) into one.

        Speech boundaries are preserved in segment.speech_id so that
        TransitionBuilderAgent can avoid counting cross-speech transitions.
        """
        if not sequences:
            raise ValueError("Cannot pool empty list of sequences.")
        leader = sequences[0].leader_name
        all_segments: list[SpeechSegment] = []
        all_tags: list[EmotionTag] = []
        all_ids: list[str] = []
        for seq in sequences:
            all_segments.extend(seq.segments)
            all_tags.extend(seq.tags)
            for sid in seq.speech_ids:
                if sid not in all_ids:
                    all_ids.append(sid)
        return cls(
            leader_name=leader,
            segments=all_segments,
            tags=all_tags,
            speech_ids=all_ids,
        )

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def texts(self) -> list[str]:
        return [s.text for s in self.segments]


@dataclass(slots=True)
class TransitionMatrix:
    """Markov transition matrix with Wilson-score confidence intervals.

    This is the same mathematical object as
    :class:`vox_romana.core.markov.CharBigramFingerprinter` but over a
    discrete emotion-label state space (N \u2248 7) rather than printable
    ASCII (N = 96).  Both are first-order row-stochastic Markov matrices;
    transformer attention is the learned, context-conditioned generalisation.

    See :meth:`to_fingerprint` for inter-leader similarity comparison.
    For char-level document fingerprinting see ``vox_romana.core.markov``.
    """

    leader_name: str
    matrix: np.ndarray        # (N, N) float32 row-normalised probabilities
    count_matrix: np.ndarray  # (N, N) int32 raw counts
    ci_low: np.ndarray        # (N, N) float32 Wilson CI lower bound
    ci_high: np.ndarray       # (N, N) float32 Wilson CI upper bound
    emotion_labels: list[str]
    smoothing_k: float
    n_transitions: int

    def to_fingerprint(self) -> np.ndarray:
        """Flatten the transition matrix to a 1-D float32 comparison vector.

        Returns ``self.matrix.ravel()`` \u2014 an ``(N*N,)`` float32 vector
        (e.g. shape ``(49,)`` for 7 emotion labels).  Two leaders' emotion
        styles can be compared via cosine similarity on these vectors.

        .. note::
            This vector's dimension (N\u00b2) differs from
            ``CharBigramFingerprinter.vector_dim`` (9,216), so it is
            **not** directly compatible with
            ``get_fingerprinter().pairwise_cosine``.  Compare directly::

                import numpy as np
                a = leader_a.to_fingerprint()
                b = leader_b.to_fingerprint()
                sim = float(
                    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                )

            Once a ``DiscreteStateFingerprinter`` is added to
            ``vox_romana.core.markov``, this method should delegate to
            ``get_fingerprinter().build(self.matrix)`` instead.
        """
        return self.matrix.ravel().astype(np.float32, copy=False)


@dataclass(slots=True)
class SemanticFlowResult:
    """Pearson correlation between topic shifts and emotion shifts."""

    pearson_r: float
    p_value: float
    topic_shifts: list[float]
    emotion_shifts: list[float]
    n_pairs: int


@dataclass(slots=True)
class NullTestResult:
    """Multi-test null-hypothesis rejection result."""

    permutation_p: float
    chi2_p: float
    chi2_statistic: float
    g_test_p: float
    g_statistic: float
    cramers_v: float
    entropy_observed: float
    entropy_random_mean: float
    entropy_ci: tuple[float, float]
    is_structured: bool   # chi2_p < 0.05 AND g_test_p < 0.05 AND permutation_p < 0.05
    n_permutations: int


@dataclass(slots=True)
class ReproducibilityManifest:
    """Full environment snapshot for exact result reproduction."""

    seed: int
    python_version: str
    package_versions: dict[str, str]
    model_versions: dict[str, str]
    hardware: str
    timestamp: str
    git_commit: str
    deterministic_mode: bool


@dataclass(slots=True)
class EmotionFlowReport:
    """Top-level result object for a single leader's emotion flow analysis."""

    leader_name: str
    n_speeches: int
    n_segments: int
    dominant_cycle: list[str]
    transition_matrix: TransitionMatrix | None = None
    null_test: NullTestResult | None = None
    semantic_flow: SemanticFlowResult | None = None
    manifest: ReproducibilityManifest | None = None
    back_translation_agreement: float = 1.0
    n_flagged_segments: int = 0
    narrative: str = ""
    report_path: str = ""

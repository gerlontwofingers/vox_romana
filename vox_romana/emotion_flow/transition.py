"""TransitionBuilderAgent — Markov transition matrix with add-k smoothing.

Builds a first-order transition probability matrix from an EmotionSequence.
Cross-speech boundaries are excluded by default (no spurious transitions when
one speech ends and another begins).

Wilson score 95% confidence intervals are computed per cell from raw counts.

Note on shared lineage
----------------------
The row-stochastic matrix built here is the same mathematical object as
:class:`vox_romana.core.markov.CharBigramFingerprinter`, just over a
discrete emotion-label state space rather than printable ASCII.  If the
math in ``core.markov`` is ever upgraded (e.g. to a learned encoder),
consider whether :class:`TransitionBuilderAgent` should delegate via a
``DiscreteStateFingerprinter``.  See also
:meth:`vox_romana.emotion_flow.models.TransitionMatrix.to_fingerprint`.
"""

from __future__ import annotations

import logging

import numpy as np

from .models import EMOTION_LABELS, EmotionSequence, TransitionMatrix

logger = logging.getLogger(__name__)

_Z95: float = 1.9599639845400536  # scipy.stats.norm.ppf(0.975)


class TransitionBuilderAgent:
    """Build a Markov transition matrix from an EmotionSequence.

    Args:
        smoothing_k:       Add-k smoothing constant. Default 0.1 (Laplace-like).
        skip_cross_speech: If True, ignore transitions between segments from
                           different speech IDs. Default True.
    """

    def __init__(
        self,
        smoothing_k: float = 0.1,
        skip_cross_speech: bool = True,
    ) -> None:
        self._k = smoothing_k
        self._skip_cross = skip_cross_speech

    def build(self, sequence: EmotionSequence) -> TransitionMatrix:
        """Compute the transition matrix for the given emotion sequence."""
        labels = EMOTION_LABELS
        n = len(labels)
        label_idx: dict[str, int] = {l: i for i, l in enumerate(labels)}

        count_matrix = np.zeros((n, n), dtype=np.int32)
        tags = sequence.tags
        segs = sequence.segments
        n_valid = min(len(tags), len(segs))

        if n_valid < 2:
            logger.warning(
                "EmotionSequence for '%s' has < 2 tagged segments; "
                "transition matrix will be near-uniform.",
                sequence.leader_name,
            )
        else:
            for i in range(n_valid - 1):
                if self._skip_cross and segs[i].speech_id != segs[i + 1].speech_id:
                    continue
                src = label_idx.get(tags[i].dominant)
                dst = label_idx.get(tags[i + 1].dominant)
                if src is not None and dst is not None:
                    count_matrix[src, dst] += 1

        n_transitions = int(count_matrix.sum())

        # Add-k smoothing → row-normalised probability matrix
        smoothed = count_matrix.astype(np.float64) + self._k
        row_sums = smoothed.sum(axis=1, keepdims=True)
        # Avoid divide-by-zero when k=0 and an emotion is never observed as a
        # source state.  Such rows stay all-zero after normalisation.
        safe_row_sums = np.where(row_sums == 0, 1.0, row_sums)
        matrix = (smoothed / safe_row_sums).astype(np.float32)

        # Wilson score 95% CIs computed from raw counts (before smoothing)
        raw_row_sums = count_matrix.sum(axis=1)
        ci_low = np.zeros((n, n), dtype=np.float32)
        ci_high = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            n_row = int(raw_row_sums[i])
            for j in range(n):
                lo, hi = _wilson_ci(int(count_matrix[i, j]), n_row)
                ci_low[i, j] = lo
                ci_high[i, j] = hi

        return TransitionMatrix(
            leader_name=sequence.leader_name,
            matrix=matrix,
            count_matrix=count_matrix,
            ci_low=ci_low,
            ci_high=ci_high,
            emotion_labels=list(labels),
            smoothing_k=self._k,
            n_transitions=n_transitions,
        )


def _wilson_ci(k: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score 95% CI for proportion k/n.

    Returns (0.0, 1.0) when n == 0 (no observations — maximum uncertainty).
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z2n = z * z / n
    center = (p + z2n / 2.0) / (1.0 + z2n)
    margin = z * np.sqrt(p * (1.0 - p) / n + z2n / (4.0 * n)) / (1.0 + z2n)
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))

"""Tests for vox_romana.emotion_flow.tester — NullHypothesisTesterAgent."""

from __future__ import annotations

import numpy as np
import pytest

from vox_romana.emotion_flow.tester import (
    NullHypothesisTesterAgent,
    _chi2_and_cramers,
    _chi2_statistic_only,
    _entropy_rate,
    _rebuild_count_matrix,
)
from vox_romana.emotion_flow.models import EMOTION_LABELS, TransitionMatrix


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_tm(count_matrix: np.ndarray) -> TransitionMatrix:
    """Build a TransitionMatrix from raw counts (add-0.1 smoothed for matrix)."""
    n = len(EMOTION_LABELS)
    smoothed = count_matrix.astype(np.float64) + 0.1
    row_sums = smoothed.sum(axis=1, keepdims=True)
    matrix = (smoothed / row_sums).astype(np.float32)
    return TransitionMatrix(
        leader_name="TestLeader",
        matrix=matrix,
        count_matrix=count_matrix.astype(np.int32),
        ci_low=np.zeros((n, n), dtype=np.float32),
        ci_high=np.ones((n, n), dtype=np.float32),
        emotion_labels=list(EMOTION_LABELS),
        smoothing_k=0.1,
        n_transitions=int(count_matrix.sum()),
    )


def _diagonal_counts(per_diagonal: int = 50) -> np.ndarray:
    """Strong self-loop matrix: each emotion always transitions to itself."""
    n = len(EMOTION_LABELS)
    return np.eye(n, dtype=np.int32) * per_diagonal


def _uniform_counts(per_cell: int = 10) -> np.ndarray:
    """Perfectly uniform transition matrix."""
    n = len(EMOTION_LABELS)
    return np.full((n, n), per_cell, dtype=np.int32)


def _diagonal_sequence(per_block: int = 51) -> list[str]:
    """Tag sequence matching a diagonal count matrix (each emotion in a block)."""
    seq: list[str] = []
    for label in EMOTION_LABELS:
        seq.extend([label] * per_block)
    return seq


def _uniform_sequence(n: int = 350) -> list[str]:
    """Pseudo-random-looking tag sequence cycling through all labels."""
    labels = EMOTION_LABELS
    return [labels[i % len(labels)] for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# NullHypothesisTesterAgent — integration
# ─────────────────────────────────────────────────────────────────────────────

class TestNullHypothesisTesterAgent:
    def test_structured_matrix_is_structured(self):
        """A strong diagonal matrix should be flagged as structured."""
        tm = _make_tm(_diagonal_counts(50))
        seq = _diagonal_sequence(51)
        agent = NullHypothesisTesterAgent(seed=42)
        result = agent.test(tm, seq, n_permutations=200)
        assert result.is_structured

    def test_uniform_matrix_not_structured(self):
        """A perfectly uniform matrix should not be flagged as structured."""
        tm = _make_tm(_uniform_counts(10))
        seq = _uniform_sequence(350)
        agent = NullHypothesisTesterAgent(seed=42)
        result = agent.test(tm, seq, n_permutations=200)
        assert not result.is_structured

    def test_cramers_v_nonnegative(self):
        tm = _make_tm(_diagonal_counts(20))
        result = NullHypothesisTesterAgent().test(tm, _diagonal_sequence(21), n_permutations=50)
        assert result.cramers_v >= 0.0

    def test_cramers_v_high_for_structured(self):
        tm = _make_tm(_diagonal_counts(100))
        result = NullHypothesisTesterAgent().test(tm, _diagonal_sequence(101), n_permutations=50)
        assert result.cramers_v > 0.5

    def test_cramers_v_low_for_uniform(self):
        tm = _make_tm(_uniform_counts(10))
        result = NullHypothesisTesterAgent().test(tm, _uniform_sequence(), n_permutations=50)
        assert result.cramers_v < 0.3

    def test_entropy_ci_is_ordered(self):
        tm = _make_tm(_diagonal_counts(30))
        result = NullHypothesisTesterAgent().test(tm, _diagonal_sequence(31), n_permutations=100)
        lo, hi = result.entropy_ci
        assert lo <= hi

    def test_entropy_observed_nonnegative(self):
        tm = _make_tm(_diagonal_counts(30))
        result = NullHypothesisTesterAgent().test(tm, _diagonal_sequence(31), n_permutations=50)
        assert result.entropy_observed >= 0.0

    def test_entropy_random_mean_nonnegative(self):
        tm = _make_tm(_uniform_counts(10))
        result = NullHypothesisTesterAgent().test(tm, _uniform_sequence(), n_permutations=50)
        assert result.entropy_random_mean >= 0.0

    def test_n_permutations_stored(self):
        tm = _make_tm(_uniform_counts(5))
        result = NullHypothesisTesterAgent().test(tm, _uniform_sequence(100), n_permutations=77)
        assert result.n_permutations == 77

    def test_chi2_statistic_nonnegative(self):
        tm = _make_tm(_diagonal_counts(20))
        result = NullHypothesisTesterAgent().test(tm, _diagonal_sequence(21), n_permutations=50)
        assert result.chi2_statistic >= 0.0

    def test_g_statistic_nonnegative(self):
        tm = _make_tm(_diagonal_counts(20))
        result = NullHypothesisTesterAgent().test(tm, _diagonal_sequence(21), n_permutations=50)
        assert result.g_statistic >= 0.0

    def test_p_values_in_range(self):
        tm = _make_tm(_diagonal_counts(20))
        result = NullHypothesisTesterAgent().test(tm, _diagonal_sequence(21), n_permutations=50)
        assert 0.0 <= result.chi2_p <= 1.0
        assert 0.0 <= result.g_test_p <= 1.0
        assert 0.0 <= result.permutation_p <= 1.0

    def test_is_structured_requires_all_three(self):
        """is_structured must be False if any of the three p-values >= 0.05."""
        n = len(EMOTION_LABELS)
        # Near-uniform matrix: chi2_p will be high → not structured
        tm = _make_tm(_uniform_counts(10))
        result = NullHypothesisTesterAgent().test(tm, _uniform_sequence(), n_permutations=50)
        # is_structured must match the conjunction
        expected = (result.chi2_p < 0.05) and (result.g_test_p < 0.05) and (result.permutation_p < 0.05)
        assert result.is_structured == expected

    def test_zero_permutations_returns_p_one(self):
        tm = _make_tm(_diagonal_counts(20))
        result = NullHypothesisTesterAgent().test(
            tm, _diagonal_sequence(21), n_permutations=0
        )
        assert result.permutation_p == 1.0

    def test_empty_sequence(self):
        tm = _make_tm(np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)), dtype=np.int32))
        result = NullHypothesisTesterAgent().test(tm, [], n_permutations=10)
        assert result.permutation_p == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _chi2_and_cramers unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChiAndCramers:
    def test_returns_three_values(self):
        counts = np.eye(7) * 10
        result = _chi2_and_cramers(counts, lambda_=None)
        assert len(result) == 3

    def test_cramers_v_zero_for_uniform(self):
        counts = np.ones((7, 7)) * 10
        _, _, v = _chi2_and_cramers(counts, lambda_=None)
        assert v == pytest.approx(0.0, abs=0.05)

    def test_cramers_v_high_for_identity(self):
        counts = np.eye(7) * 100
        _, _, v = _chi2_and_cramers(counts, lambda_=None)
        assert v > 0.5

    def test_all_zero_matrix_handled(self):
        counts = np.zeros((7, 7))
        stat, p, v = _chi2_and_cramers(counts, lambda_=None)
        assert stat >= 0.0
        assert 0.0 <= p <= 1.0

    def test_g_test_lambda(self):
        counts = np.eye(7) * 20
        g_stat, g_p, _ = _chi2_and_cramers(counts, lambda_="log-likelihood")
        assert g_stat > 0.0
        assert g_p < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# _chi2_statistic_only unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChi2StatisticOnly:
    def test_uniform_matrix_gives_zero(self):
        counts = np.ones((7, 7)) * 10
        stat = _chi2_statistic_only(counts)
        assert stat == pytest.approx(0.0, abs=1e-6)

    def test_identity_matrix_gives_high_stat(self):
        counts = np.eye(7) * 100
        stat = _chi2_statistic_only(counts)
        assert stat > 100.0

    def test_zero_matrix_gives_zero(self):
        counts = np.zeros((7, 7))
        stat = _chi2_statistic_only(counts)
        assert stat == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# _entropy_rate unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEntropyRate:
    def test_uniform_matrix_max_entropy(self):
        n = 7
        matrix = np.full((n, n), 1.0 / n)
        row_totals = np.ones(n)
        h = _entropy_rate(matrix, row_totals)
        assert h == pytest.approx(np.log(n), abs=0.01)

    def test_identity_matrix_zero_entropy(self):
        n = 7
        matrix = np.eye(n, dtype=np.float64)
        row_totals = np.ones(n)
        h = _entropy_rate(matrix, row_totals)
        assert h == pytest.approx(0.0, abs=1e-6)

    def test_zero_row_totals_returns_zero(self):
        n = 7
        matrix = np.full((n, n), 1.0 / n)
        row_totals = np.zeros(n)
        h = _entropy_rate(matrix, row_totals)
        assert h == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# _rebuild_count_matrix unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRebuildCountMatrix:
    def test_shape(self):
        seq = ["joy", "anger", "fear"]
        counts = _rebuild_count_matrix(seq, EMOTION_LABELS)
        assert counts.shape == (len(EMOTION_LABELS), len(EMOTION_LABELS))

    def test_known_transition(self):
        seq = ["joy", "anger", "joy"]
        counts = _rebuild_count_matrix(seq, EMOTION_LABELS)
        joy_idx = EMOTION_LABELS.index("joy")
        anger_idx = EMOTION_LABELS.index("anger")
        assert counts[joy_idx, anger_idx] == 1.0
        assert counts[anger_idx, joy_idx] == 1.0

    def test_unknown_labels_ignored(self):
        seq = ["joy", "happiness", "anger"]  # "happiness" is unknown
        counts = _rebuild_count_matrix(seq, EMOTION_LABELS)
        joy_idx = EMOTION_LABELS.index("joy")
        # joy→happiness skipped, happiness→anger skipped
        assert counts[joy_idx].sum() == 0.0

    def test_empty_sequence_returns_zeros(self):
        counts = _rebuild_count_matrix([], EMOTION_LABELS)
        assert counts.sum() == 0.0

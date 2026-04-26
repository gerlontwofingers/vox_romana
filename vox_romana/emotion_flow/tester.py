"""NullHypothesisTesterAgent — tests whether observed emotion transitions are
structured (non-random) using chi-square, G-test, permutation test, and
entropy rate analysis.

is_structured = True when chi2_p < 0.05 AND g_test_p < 0.05 AND permutation_p < 0.05.

Cramér's V measures effect size (0 = no association, 1 = perfect association).
"""

from __future__ import annotations

import logging

import numpy as np

from .models import NullTestResult, TransitionMatrix

logger = logging.getLogger(__name__)


class NullHypothesisTesterAgent:
    """Test whether emotion transition patterns are non-random.

    Args:
        seed: Random seed for permutation test reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def test(
        self,
        tm: TransitionMatrix,
        tag_sequence: list[str],
        n_permutations: int = 1000,
    ) -> NullTestResult:
        """Run all null-hypothesis tests and return a NullTestResult.

        Args:
            tm:             Transition matrix from TransitionBuilderAgent.
            tag_sequence:   Flat list of dominant emotion labels (one per segment),
                            used to generate the permutation null distribution.
            n_permutations: Number of permutation draws. Default 1000.
        """
        counts = tm.count_matrix.astype(np.float64)

        # Chi-square and G-test on observed counts
        chi2_stat, chi2_p, cramers_v = _chi2_and_cramers(counts, lambda_=None)
        g_stat, g_p, _ = _chi2_and_cramers(counts, lambda_="log-likelihood")

        # Entropy rate of the smoothed observed matrix
        row_totals = counts.sum(axis=1)
        entropy_obs = _entropy_rate(tm.matrix.astype(np.float64), row_totals)

        # Permutation test: null distribution of chi2 statistics
        perm_p, perm_entropies = _permutation_test(
            observed_chi2=chi2_stat,
            tag_sequence=tag_sequence,
            labels=tm.emotion_labels,
            n_permutations=n_permutations,
            seed=self._seed,
        )

        if perm_entropies:
            entropy_random_mean = float(np.mean(perm_entropies))
            entropy_ci: tuple[float, float] = (
                float(np.percentile(perm_entropies, 2.5)),
                float(np.percentile(perm_entropies, 97.5)),
            )
        else:
            entropy_random_mean = entropy_obs
            entropy_ci = (entropy_obs, entropy_obs)

        is_structured = (chi2_p < 0.05) and (g_p < 0.05) and (perm_p < 0.05)

        return NullTestResult(
            permutation_p=perm_p,
            chi2_p=chi2_p,
            chi2_statistic=chi2_stat,
            g_test_p=g_p,
            g_statistic=g_stat,
            cramers_v=cramers_v,
            entropy_observed=entropy_obs,
            entropy_random_mean=entropy_random_mean,
            entropy_ci=entropy_ci,
            is_structured=is_structured,
            n_permutations=n_permutations,
        )


def _chi2_and_cramers(
    counts: np.ndarray,
    lambda_: str | None,
) -> tuple[float, float, float]:
    """Run chi2_contingency and compute Cramér's V.

    Returns (statistic, p_value, cramers_v).
    Handles all-zero rows by adding a small epsilon to avoid NaN.
    """
    from scipy.stats import chi2_contingency

    safe = counts.copy()
    # Replace all-zero rows with tiny value to prevent NaN expected frequencies
    zero_rows = safe.sum(axis=1) == 0.0
    if zero_rows.any():
        safe[zero_rows] = 1e-10

    kwargs: dict = {}
    if lambda_ is not None:
        kwargs["lambda_"] = lambda_

    try:
        result = chi2_contingency(safe, **kwargs)
        chi2_stat = float(result.statistic)
        p_value = float(result.pvalue)
    except Exception as exc:
        logger.warning("chi2_contingency failed: %s", exc)
        chi2_stat, p_value = 0.0, 1.0

    n = float(counts.sum())
    k = min(counts.shape)  # min(n_rows, n_cols)
    cramers_v = float(np.sqrt(chi2_stat / (n * (k - 1)))) if (n > 0 and k > 1) else 0.0

    return chi2_stat, p_value, cramers_v


def _entropy_rate(matrix: np.ndarray, row_totals: np.ndarray) -> float:
    """Weighted entropy rate of a row-normalised transition matrix.

    H = Σ_i w_i · (−Σ_j p_ij · log(p_ij))
    where w_i = row_total_i / Σ row_totals.
    """
    total = float(row_totals.sum())
    if total == 0.0:
        return 0.0
    weights = row_totals / total
    h = 0.0
    for i, row in enumerate(matrix):
        row_h = float(-np.sum(row * np.log(row + 1e-300)))
        h += float(weights[i]) * row_h
    return h


def _chi2_statistic_only(counts: np.ndarray) -> float:
    """Compute chi-square statistic without p-value (fast path for permutation loop)."""
    counts = counts.astype(np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)
    col_sums = counts.sum(axis=0, keepdims=True)
    n = float(counts.sum())
    if n == 0.0:
        return 0.0
    expected = row_sums * col_sums / n
    expected = np.where(expected < 1e-10, 1e-10, expected)
    return float(np.sum((counts - expected) ** 2 / expected))


def _rebuild_count_matrix(tag_sequence: list[str], labels: list[str]) -> np.ndarray:
    """Build a transition count matrix from a flat tag sequence."""
    label_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    counts = np.zeros((n, n), dtype=np.float64)
    for a, b in zip(tag_sequence, tag_sequence[1:]):
        i = label_idx.get(a, -1)
        j = label_idx.get(b, -1)
        if i >= 0 and j >= 0:
            counts[i, j] += 1.0
    return counts


def _permutation_test(
    observed_chi2: float,
    tag_sequence: list[str],
    labels: list[str],
    n_permutations: int,
    seed: int,
) -> tuple[float, list[float]]:
    """Permutation test: fraction of shuffled chi2 statistics >= observed_chi2.

    Returns (p_value, list of permuted entropy rates for CI computation).
    """
    if n_permutations == 0 or len(tag_sequence) < 2:
        return 1.0, []

    rng = np.random.default_rng(seed)
    arr = np.array(tag_sequence, dtype=object)

    perm_chi2s: list[float] = []
    perm_entropies: list[float] = []

    for _ in range(n_permutations):
        perm = rng.permutation(arr).tolist()
        counts_p = _rebuild_count_matrix(perm, labels)

        perm_chi2s.append(_chi2_statistic_only(counts_p))

        # Entropy rate of permuted matrix (add-1 smoothing for log stability)
        row_totals_p = counts_p.sum(axis=1)
        safe_p = counts_p + 1.0
        norm_p = safe_p / safe_p.sum(axis=1, keepdims=True)
        perm_entropies.append(_entropy_rate(norm_p, row_totals_p))

    perm_p = float(np.mean(np.array(perm_chi2s) >= observed_chi2))
    return perm_p, perm_entropies

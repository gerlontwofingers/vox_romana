"""Six-test distribution-free hypothesis battery for rhetorical corpus analysis.

Tests whether two speakers occupy statistically distinct ideological/semantic
spaces.  All six tests are non-parametric (distribution-free).  P-values are
corrected for multiple comparisons using Benjamini-Hochberg FDR.

VERDICT RULE
    H0 is rejected (speakers ARE statistically distinct) if **≥ 3 of 6 tests**
    pass at FDR-corrected p < 0.05.

THE SIX TESTS
    1. Permutation centroid distance — cosine distance between embedding
       centroids against a 10 k-shuffle null.
    2. Mann-Whitney U on cosine similarities — within-speaker vs cross-speaker
       pairwise similarities (H1: within > cross).
    3. Kolmogorov-Smirnov on dominant topic distributions — KS test on the
       per-utterance dominant topic index.
    4. Jensen-Shannon divergence — JSD between mean topic-weight vectors, with
       permutation p-value and bootstrap 95 % CI.
    5. Fisher's exact on stance contingency — 2×2 table of high/low weight on
       the most discriminating topic, split by speaker.
    6. Burrows' Delta stylometry — mean absolute Z-score difference on top-100
       word frequencies, permutation p-value.

USAGE::

    from vox_romana.analysis import CorpusLoader, CorpusEmbedder
    from vox_romana.analysis.hypothesis import HypothesisTester

    corpora = CorpusLoader.from_vox_romana_json("debate.json")
    a, b = list(corpora.values())[:2]

    embedder = CorpusEmbedder()
    emb_a, emb_b = embedder.embed_pair(a, b)

    result = HypothesisTester().run(a, b, emb_a, emb_b)
    print(result.verdict)          # "REJECTED" or "NOT_REJECTED"
    print(result.tests_passed)     # int, e.g. 5
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections import Counter

import numpy as np

from .models import HypothesisResult, SpeakerCorpus, TestResult
from .topics import TopicModeler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 − similarity) between two possibly-unnormalised vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return float(1.0 - float(np.dot(a, b)) / (na * nb))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: 1-D array of raw p-values.

    Returns:
        Adjusted p-values in the same order as the input, clipped to [0, 1].
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(p_values)               # ascending
    sorted_p = p_values[order]

    # BH adjustment: p_adj[i] = p[i] * n / (i+1)  (for i = 0..n-1 sorted)
    adjusted = np.minimum(1.0, sorted_p * n / np.arange(1, n + 1, dtype=np.float64))

    # Enforce isotonic monotonicity from largest rank downward
    for i in range(n - 2, -1, -1):
        if adjusted[i] > adjusted[i + 1]:
            adjusted[i] = adjusted[i + 1]

    result = np.empty(n, dtype=np.float64)
    result[order] = adjusted
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HypothesisTester
# ─────────────────────────────────────────────────────────────────────────────


class HypothesisTester:
    """Runs the six-test distribution-free rhetorical hypothesis battery.

    All individual test methods are public and callable independently (e.g.
    for custom pipelines or inspection).  When called via :meth:`run`, raw
    p-values are corrected together with Benjamini-Hochberg before the
    ``passed`` flag is set on each :class:`~vox_romana.analysis.models.TestResult`.

    Args:
        n_permutations: Permutation iterations for tests 1, 4 (null), and 6.
                        Default 10 000.
        n_bootstrap:    Bootstrap iterations for test 4 CI.  Default 2 000.
        n_topics:       LDA topics for tests 3–5.  Capped to corpus size
                        automatically.  Default 10.
        random_state:   RNG seed.  Default 42.
    """

    def __init__(
        self,
        n_permutations: int = 10_000,
        n_bootstrap: int = 2_000,
        n_topics: int = 10,
        random_state: int = 42,
    ) -> None:
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap
        self.n_topics = n_topics
        self.random_state = random_state
        self._last_topic_modeler: TopicModeler | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Full battery
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        corpus_a: SpeakerCorpus,
        corpus_b: SpeakerCorpus,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        *,
        topic_modeler: TopicModeler | None = None,
    ) -> HypothesisResult:
        """Run all six tests, apply FDR correction, and return a verdict.

        Args:
            corpus_a, corpus_b: Speaker corpora to compare.
            emb_a, emb_b:       L2-normalised embeddings from
                                :class:`~vox_romana.analysis.embedder.CorpusEmbedder`.
            topic_modeler:      Optional pre-fitted :class:`TopicModeler`.
                                A new one is created and fitted if ``None``.

        Returns:
            :class:`~vox_romana.analysis.models.HypothesisResult` with
            FDR-corrected ``TestResult`` objects and a final verdict.
        """
        # ── Fit (or refit) topic model ────────────────────────────────────────
        if topic_modeler is None:
            n_topics = min(
                self.n_topics,
                max(2, len(corpus_a.utterances) // 2),
                max(2, len(corpus_b.utterances) // 2),
            )
            topic_modeler = TopicModeler(
                n_topics=n_topics, random_state=self.random_state
            )

        dist_a, dist_b = topic_modeler.fit_transform(corpus_a, corpus_b)
        self._last_topic_modeler = topic_modeler

        # ── Six tests ─────────────────────────────────────────────────────────
        raw: list[TestResult] = [
            self.permutation_centroid(emb_a, emb_b),
            self.mann_whitney_cosine(emb_a, emb_b),
            self.ks_topic_distributions(dist_a, dist_b),
            self.jsd_bootstrap(dist_a, dist_b, topic_modeler),
            self.fisher_stance_contingency(dist_a, dist_b, topic_modeler),
            self.burrows_delta(corpus_a, corpus_b),
        ]

        # ── Benjamini-Hochberg FDR correction ─────────────────────────────────
        p_raw = np.array([r.p_value for r in raw], dtype=np.float64)
        p_corrected = self.benjamini_hochberg(p_raw)

        final = [
            dataclasses.replace(r, p_value=float(cp), passed=float(cp) < 0.05)
            for r, cp in zip(raw, p_corrected)
        ]

        n_passed = sum(1 for r in final if r.passed)
        verdict = "REJECTED" if n_passed >= 3 else "NOT_REJECTED"

        logger.info(
            "HypothesisTester: %s vs %s → %s (%d/6 tests passed)",
            corpus_a.name,
            corpus_b.name,
            verdict,
            n_passed,
        )

        return HypothesisResult(
            speaker_a=corpus_a.name,
            speaker_b=corpus_b.name,
            test_results=final,
            verdict=verdict,
            fdr_corrected=True,
            notes=f"{n_passed}/{len(final)} tests rejected H0 at FDR-corrected p < 0.05",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1 — Permutation centroid distance
    # ─────────────────────────────────────────────────────────────────────────

    def permutation_centroid(
        self, emb_a: np.ndarray, emb_b: np.ndarray
    ) -> TestResult:
        """Test 1 — Permutation test on centroid cosine distance.

        Observed statistic: cosine distance between the mean embedding of each
        corpus.  Null distribution: 10 k random splits of the pooled rows.

        ``passed=False`` until FDR correction is applied by :meth:`run`.
        """
        centroid_a = emb_a.mean(axis=0)
        centroid_b = emb_b.mean(axis=0)
        observed = _cosine_distance(centroid_a, centroid_b)

        rng = np.random.default_rng(self.random_state)
        pooled = np.vstack([emb_a, emb_b])
        na = len(emb_a)
        n_total = len(pooled)

        null = np.empty(self.n_permutations, dtype=np.float64)
        for i in range(self.n_permutations):
            idx = rng.permutation(n_total)
            ca = pooled[idx[:na]].mean(axis=0)
            cb = pooled[idx[na:]].mean(axis=0)
            null[i] = _cosine_distance(ca, cb)

        p_value = float((null >= observed).mean())
        null_std = float(null.std()) + 1e-10
        effect_size = float((observed - float(null.mean())) / null_std)
        ci_low = float(np.percentile(null, 2.5))
        ci_high = float(np.percentile(null, 97.5))

        return TestResult(
            name="Permutation centroid distance",
            statistic=float(observed),
            p_value=p_value,
            effect_size=effect_size,
            ci_low=ci_low,
            ci_high=ci_high,
            passed=False,
            notes=f"{self.n_permutations} permutations",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2 — Mann-Whitney U on cosine similarities
    # ─────────────────────────────────────────────────────────────────────────

    def mann_whitney_cosine(
        self, emb_a: np.ndarray, emb_b: np.ndarray
    ) -> TestResult:
        """Test 2 — Mann-Whitney U: within-corpus vs cross-corpus cosine sims.

        H1 (alternative='greater'): within-speaker similarities are higher
        than cross-speaker similarities (speakers occupy distinct regions).

        Capped at 200 utterances per corpus for compute efficiency.
        Effect size: rank-biserial r = 1 − 2U / (n₁ · n₂).
        """
        from scipy.stats import mannwhitneyu

        MAX = 200
        a = emb_a[:MAX]
        b = emb_b[:MAX]
        na, nb = len(a), len(b)

        # Cross-corpus: embeddings are L2-normalised → dot product = cosine sim
        cross = (a @ b.T).ravel().astype(np.float64)

        # Within-corpus: upper triangle (excluding self-similarity diagonal)
        wi_a = (a @ a.T)[np.triu_indices(na, k=1)].astype(np.float64)
        wi_b = (b @ b.T)[np.triu_indices(nb, k=1)].astype(np.float64)
        within = np.concatenate([wi_a, wi_b])

        if len(within) == 0 or len(cross) == 0:
            return TestResult(
                name="Mann-Whitney U cosine similarity",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                ci_low=0.0,
                ci_high=0.0,
                passed=False,
                notes="Insufficient data for MWU (single-utterance corpus)",
            )

        U, p_value = mannwhitneyu(within, cross, alternative="greater")
        n1, n2 = len(within), len(cross)
        r = float(1.0 - (2.0 * float(U)) / (n1 * n2))

        return TestResult(
            name="Mann-Whitney U cosine similarity",
            statistic=float(U),
            p_value=float(p_value),
            effect_size=r,
            ci_low=float(np.percentile(cross, 2.5)),
            ci_high=float(np.percentile(within, 97.5)),
            passed=False,
            notes=f"|A|={na}  |B|={nb}  within={len(within)}  cross={len(cross)}",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3 — KS on dominant topic distributions
    # ─────────────────────────────────────────────────────────────────────────

    def ks_topic_distributions(
        self, dist_a: np.ndarray, dist_b: np.ndarray
    ) -> TestResult:
        """Test 3 — Kolmogorov-Smirnov on per-utterance dominant topic index.

        Each utterance is reduced to its argmax topic index.  The KS statistic
        D is the maximum CDF difference between the two per-corpus distributions.
        Effect size = D.
        """
        from scipy.stats import ks_2samp

        dom_a = dist_a.argmax(axis=1).astype(np.float64)
        dom_b = dist_b.argmax(axis=1).astype(np.float64)

        D, p_value = ks_2samp(dom_a, dom_b)

        return TestResult(
            name="KS dominant topic distribution",
            statistic=float(D),
            p_value=float(p_value),
            effect_size=float(D),
            ci_low=0.0,
            ci_high=0.0,
            passed=False,
            notes="KS on per-utterance dominant topic index",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Test 4 — Jensen-Shannon divergence
    # ─────────────────────────────────────────────────────────────────────────

    def jsd_bootstrap(
        self,
        dist_a: np.ndarray,
        dist_b: np.ndarray,
        topic_modeler: TopicModeler,
    ) -> TestResult:
        """Test 4 — Jensen-Shannon divergence between corpus topic distributions.

        Statistic: JS distance (√JSD, ∈ [0, 1]) between mean topic-weight
        vectors.  P-value from permutation null (pool utterances, split
        randomly).  95 % CI from per-corpus bootstrap resampling.

        Effect size = observed JS distance.
        """
        from scipy.spatial.distance import jensenshannon

        def _safe_jsd(p: np.ndarray, q: np.ndarray) -> float:
            p = np.maximum(p, 1e-10).astype(np.float64)
            q = np.maximum(q, 1e-10).astype(np.float64)
            p = p / p.sum()
            q = q / q.sum()
            return float(jensenshannon(p, q))

        w_a = topic_modeler.corpus_weights(dist_a)
        w_b = topic_modeler.corpus_weights(dist_b)
        observed_jsd = _safe_jsd(w_a, w_b)

        rng = np.random.default_rng(self.random_state)
        na = len(dist_a)
        nb = len(dist_b)
        pooled = np.vstack([dist_a, dist_b])
        n_total = len(pooled)

        # Bootstrap CI — resample each corpus independently
        boot = np.empty(self.n_bootstrap, dtype=np.float64)
        for i in range(self.n_bootstrap):
            sa = dist_a[rng.integers(0, na, na)]
            sb = dist_b[rng.integers(0, nb, nb)]
            boot[i] = _safe_jsd(
                topic_modeler.corpus_weights(sa),
                topic_modeler.corpus_weights(sb),
            )

        # Permutation p-value — pool then split
        n_perm = max(50, self.n_permutations // 5)
        null = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            idx = rng.permutation(n_total)
            null[i] = _safe_jsd(
                topic_modeler.corpus_weights(pooled[idx[:na]]),
                topic_modeler.corpus_weights(pooled[idx[na:]]),
            )

        p_value = float((null >= observed_jsd).mean())
        ci_low = float(np.percentile(boot, 2.5))
        ci_high = float(np.percentile(boot, 97.5))

        return TestResult(
            name="Jensen-Shannon divergence",
            statistic=observed_jsd,
            p_value=p_value,
            effect_size=observed_jsd,
            ci_low=ci_low,
            ci_high=ci_high,
            passed=False,
            notes=(
                f"JS distance on corpus topic weights; "
                f"{self.n_bootstrap} bootstrap samples; "
                f"{n_perm} permutation null samples"
            ),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Test 5 — Fisher's exact on stance contingency
    # ─────────────────────────────────────────────────────────────────────────

    def fisher_stance_contingency(
        self,
        dist_a: np.ndarray,
        dist_b: np.ndarray,
        topic_modeler: TopicModeler,
    ) -> TestResult:
        """Test 5 — Fisher's exact on a 2×2 stance contingency table.

        Selects the topic with the largest absolute mean-weight difference
        between the two corpora.  Classifies each utterance as high (≥ median)
        or low (< median) weight on that topic.  Builds a 2×2 table:

            +---------+-------+-------+
            |         |   A   |   B   |
            +---------+-------+-------+
            | high    | high_a| high_b|
            | low     | low_a | low_b |
            +---------+-------+-------+

        Statistic: odds ratio.  Effect size: phi coefficient.
        """
        from scipy.stats import fisher_exact

        w_a = topic_modeler.corpus_weights(dist_a)
        w_b = topic_modeler.corpus_weights(dist_b)
        best_topic = int(np.argmax(np.abs(w_a - w_b)))

        col_a = dist_a[:, best_topic].astype(np.float64)
        col_b = dist_b[:, best_topic].astype(np.float64)
        threshold = float(np.median(np.concatenate([col_a, col_b])))

        high_a = int((col_a >= threshold).sum())
        low_a = int((col_a < threshold).sum())
        high_b = int((col_b >= threshold).sum())
        low_b = int((col_b < threshold).sum())
        n = high_a + high_b + low_a + low_b

        if n == 0:
            return TestResult(
                name="Fisher stance contingency",
                statistic=1.0,
                p_value=1.0,
                effect_size=0.0,
                ci_low=0.0,
                ci_high=0.0,
                passed=False,
                notes="Zero-count contingency table",
            )

        contingency = [[high_a, high_b], [low_a, low_b]]
        odds_ratio, p_value = fisher_exact(contingency, alternative="two-sided")

        # Phi coefficient
        denom = (
            (high_a + high_b)
            * (low_a + low_b)
            * (high_a + low_a)
            * (high_b + low_b)
        )
        if denom > 0:
            chi2 = n * (high_a * low_b - high_b * low_a) ** 2 / denom
            phi = float(np.sqrt(chi2 / n))
        else:
            phi = 0.0

        label = topic_modeler.topic_label(best_topic)
        return TestResult(
            name="Fisher stance contingency",
            statistic=float(odds_ratio),
            p_value=float(p_value),
            effect_size=phi,
            ci_low=0.0,
            ci_high=0.0,
            passed=False,
            notes=(
                f"Best topic {best_topic}: {label}; "
                f"table [[{high_a},{high_b}],[{low_a},{low_b}]]"
            ),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Test 6 — Burrows' Delta stylometry
    # ─────────────────────────────────────────────────────────────────────────

    def burrows_delta(
        self, corpus_a: SpeakerCorpus, corpus_b: SpeakerCorpus
    ) -> TestResult:
        """Test 6 — Burrows' Delta stylometric distance.

        Uses the top-100 most frequent words from the combined corpus.
        Computes per-utterance frequency vectors, Z-score-normalises using the
        combined corpus statistics, then measures the mean absolute Z-score
        difference between the two speakers.

        P-value: permutation test — utterances are randomly reassigned to
        speakers ``n_permutations`` times; p = fraction of null deltas ≥
        observed delta.

        Effect size: (observed − mean_null) / std_null (standardised distance).
        """
        _TOKENIZE = re.compile(r"\b[a-z]+\b")

        def _tokenize(text: str) -> list[str]:
            return _TOKENIZE.findall(text.lower())

        all_utts = corpus_a.utterances + corpus_b.utterances
        na = len(corpus_a.utterances)
        n_total = len(all_utts)

        all_tokens = _tokenize(" ".join(all_utts))
        n_words = min(100, max(1, len(set(all_tokens))))
        common_words = [w for w, _ in Counter(all_tokens).most_common(n_words)]
        word_idx = {w: i for i, w in enumerate(common_words)}

        # Per-segment frequency matrix  (n_total, n_words)
        seg_freqs = np.zeros((n_total, n_words), dtype=np.float32)
        for s, utt in enumerate(all_utts):
            toks = _tokenize(utt)
            total = max(len(toks), 1)
            for tok in toks:
                if tok in word_idx:
                    seg_freqs[s, word_idx[tok]] += 1.0 / total

        mean_f = seg_freqs.mean(axis=0)
        std_f = seg_freqs.std(axis=0) + 1e-10

        # Observed Delta
        mean_a = seg_freqs[:na].mean(axis=0)
        mean_b = seg_freqs[na:].mean(axis=0)
        z_a = (mean_a - mean_f) / std_f
        z_b = (mean_b - mean_f) / std_f
        observed_delta = float(np.abs(z_a - z_b).mean())

        # Permutation null
        rng = np.random.default_rng(self.random_state)
        null = np.empty(self.n_permutations, dtype=np.float64)
        for i in range(self.n_permutations):
            idx = rng.permutation(n_total)
            ga = seg_freqs[idx[:na]].mean(axis=0)
            gb = seg_freqs[idx[na:]].mean(axis=0)
            pz_a = (ga - mean_f) / std_f
            pz_b = (gb - mean_f) / std_f
            null[i] = float(np.abs(pz_a - pz_b).mean())

        p_value = float((null >= observed_delta).mean())
        null_std = float(null.std()) + 1e-10
        effect_size = float((observed_delta - float(null.mean())) / null_std)
        ci_low = float(np.percentile(null, 2.5))
        ci_high = float(np.percentile(null, 97.5))

        return TestResult(
            name="Burrows Delta stylometry",
            statistic=float(observed_delta),
            p_value=p_value,
            effect_size=effect_size,
            ci_low=ci_low,
            ci_high=ci_high,
            passed=False,
            notes=f"Top {n_words} words; {self.n_permutations} permutations",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FDR correction (also a module-level function)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
        """Alias for the module-level :func:`benjamini_hochberg`."""
        return benjamini_hochberg(p_values)

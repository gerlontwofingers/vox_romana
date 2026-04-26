"""Tests for vox_romana.analysis.hypothesis — HypothesisTester."""

from __future__ import annotations

import numpy as np
import pytest

from vox_romana.analysis.embedder import CorpusEmbedder
from vox_romana.analysis.hypothesis import HypothesisTester, benjamini_hochberg
from vox_romana.analysis.models import HypothesisResult, SpeakerCorpus, TestResult
from vox_romana.analysis.topics import TopicModeler


# ─────────────────────────────────────────────────────────────────────────────
# Test corpora  (maximally distinct vocabularies for reliable rejection)
# ─────────────────────────────────────────────────────────────────────────────

_MILITARY = [
    "National defense requires increased military spending and weapon procurement",
    "Armed forces must maintain readiness with modern equipment and training",
    "Pentagon funding ensures soldiers have necessary weapons for combat operations",
    "Strategic nuclear deterrence protects sovereignty from foreign aggression",
    "Military bases require substantial infrastructure investment for personnel",
    "Intelligence agencies monitor terrorist threats to national security interests",
    "Naval warships patrol international waters to enforce maritime agreements",
    "Air force fighter jets defend territorial airspace against hostile incursion",
    "Missile defense systems intercept ballistic threats protecting civilian population",
    "Veterans deserve healthcare benefits earned through honorable military service",
    "Cybersecurity forces protect military networks from foreign adversary hacking",
    "Special operations units conduct classified missions in hostile territories abroad",
]

_ENVIRONMENTAL = [
    "Climate change threatens ecosystems through rising temperatures and flooding",
    "Carbon emissions from fossil fuels must be reduced to prevent catastrophe",
    "Renewable solar and wind energy creates sustainable economic opportunities",
    "Ocean acidification destroys coral reefs and decimates marine biodiversity",
    "Deforestation accelerates greenhouse warming and destroys wildlife habitat",
    "Environmental regulations protect clean drinking water and breathable air",
    "Green technology innovation reduces industrial pollution and carbon footprint",
    "Sustainable agriculture practices protect soil health and water quality",
    "Electric vehicles reduce transportation emissions and improve urban air quality",
    "Conservation efforts preserve endangered species and natural wilderness areas",
    "International climate agreements require binding commitments to emissions reductions",
    "Plastic pollution in oceans kills marine life and enters human food chains",
]


def _make_corpus(name: str, utterances: list[str]) -> SpeakerCorpus:
    return SpeakerCorpus(name=name, utterances=utterances)


def _make_distinct_pair() -> tuple[SpeakerCorpus, SpeakerCorpus]:
    return (
        _make_corpus("General", _MILITARY),
        _make_corpus("Ecologist", _ENVIRONMENTAL),
    )


def _make_identical_pair() -> tuple[SpeakerCorpus, SpeakerCorpus]:
    return (
        _make_corpus("Speaker A", _MILITARY),
        _make_corpus("Speaker B", _MILITARY),
    )


def _tfidf_embeddings(
    corpus_a: SpeakerCorpus,
    corpus_b: SpeakerCorpus,
    tmp_path,
) -> tuple[np.ndarray, np.ndarray]:
    emb = CorpusEmbedder(cache_dir=tmp_path, _force_backend="tfidf")
    return emb.embed_pair(corpus_a, corpus_b)


def _fast_tester(**kwargs) -> HypothesisTester:
    """HypothesisTester with small iteration counts for speed in tests."""
    defaults = dict(n_permutations=300, n_bootstrap=100, n_topics=4, random_state=42)
    defaults.update(kwargs)
    return HypothesisTester(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level benjamini_hochberg
# ─────────────────────────────────────────────────────────────────────────────

class TestBenjaminiHochberg:
    def test_empty_input(self):
        result = benjamini_hochberg(np.array([]))
        assert len(result) == 0

    def test_single_pvalue_unchanged(self):
        result = benjamini_hochberg(np.array([0.03]))
        np.testing.assert_allclose(result, [0.03], atol=1e-10)

    def test_simple_correction(self):
        # p = [0.01, 0.04, 0.03, 0.20], n=4
        # sorted: 0.01 (r1), 0.03 (r2), 0.04 (r3), 0.20 (r4)
        # raw adj: [4/1*0.01, 4/2*0.03, 4/3*0.04, 4/4*0.20] = [0.04, 0.06, 0.0533, 0.20]
        # mono: [0.04, 0.0533, 0.0533, 0.20]
        p = np.array([0.01, 0.04, 0.03, 0.20])
        adj = benjamini_hochberg(p)
        assert adj[0] <= adj[1]   # adj is not required to be sorted in input order
        assert (adj <= 1.0).all()
        assert (adj >= 0.0).all()
        # Only p=0.01 → adj=0.04 < 0.05 should pass
        assert adj[0] < 0.05
        assert adj[1] >= 0.05
        assert adj[2] >= 0.05

    def test_all_small_pvalues_all_pass(self):
        p = np.array([0.001, 0.002, 0.001, 0.001, 0.002, 0.001])
        adj = benjamini_hochberg(p)
        assert (adj < 0.05).all()

    def test_all_large_pvalues_none_pass(self):
        p = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        adj = benjamini_hochberg(p)
        assert (adj >= 0.05).all()

    def test_adjusted_values_clipped_to_one(self):
        p = np.array([1.0, 1.0, 1.0])
        adj = benjamini_hochberg(p)
        assert (adj <= 1.0).all()

    def test_monotonicity_preserved(self):
        """Adjusted p-values in sorted order should be non-decreasing."""
        p = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.40])
        adj = benjamini_hochberg(p)
        order = np.argsort(p)
        sorted_adj = adj[order]
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-12

    def test_output_length_matches_input(self):
        p = np.array([0.1, 0.2, 0.3])
        adj = benjamini_hochberg(p)
        assert len(adj) == 3

    def test_static_method_alias(self):
        p = np.array([0.01, 0.04, 0.03])
        np.testing.assert_allclose(
            HypothesisTester.benjamini_hochberg(p),
            benjamini_hochberg(p),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Permutation centroid distance
# ─────────────────────────────────────────────────────────────────────────────

class TestPermutationCentroid:
    def test_returns_test_result_instance(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().permutation_centroid(emb_a, emb_b)
        assert isinstance(result, TestResult)
        assert result.name == "Permutation centroid distance"

    def test_statistic_is_between_zero_and_two(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().permutation_centroid(emb_a, emb_b)
        assert 0.0 <= result.statistic <= 2.0

    def test_identical_corpora_high_pvalue(self, tmp_path):
        a, b = _make_identical_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().permutation_centroid(emb_a, emb_b)
        # Centroids should be identical → observed ≈ 0 → p ≈ 1
        assert result.p_value > 0.05

    def test_distinct_corpora_low_pvalue(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().permutation_centroid(emb_a, emb_b)
        assert result.p_value < 0.05

    def test_passed_is_false_before_fdr(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().permutation_centroid(emb_a, emb_b)
        assert result.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Mann-Whitney U cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestMannWhitneyCosine:
    def test_returns_test_result_instance(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().mann_whitney_cosine(emb_a, emb_b)
        assert isinstance(result, TestResult)
        assert result.name == "Mann-Whitney U cosine similarity"

    def test_statistic_is_nonnegative(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().mann_whitney_cosine(emb_a, emb_b)
        assert result.statistic >= 0.0

    def test_identical_corpora_high_pvalue(self, tmp_path):
        a, b = _make_identical_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().mann_whitney_cosine(emb_a, emb_b)
        assert result.p_value > 0.05

    def test_distinct_corpora_low_pvalue(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().mann_whitney_cosine(emb_a, emb_b)
        assert result.p_value < 0.05

    def test_single_utterance_corpus_returns_graceful_result(self, tmp_path):
        tiny_a = SpeakerCorpus(name="A", utterances=["just one sentence here"])
        tiny_b = SpeakerCorpus(name="B", utterances=["another single sentence"])
        emb_a, emb_b = _tfidf_embeddings(tiny_a, tiny_b, tmp_path)
        result = _fast_tester().mann_whitney_cosine(emb_a, emb_b)
        # Single utterances → within is empty → fallback
        assert result.p_value == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — KS dominant topic distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestKSTopicDistributions:
    def _get_dists(self):
        a, b = _make_distinct_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        return tm.fit_transform(a, b), tm

    def test_returns_test_result_instance(self):
        (dist_a, dist_b), _ = self._get_dists()
        result = _fast_tester().ks_topic_distributions(dist_a, dist_b)
        assert isinstance(result, TestResult)
        assert result.name == "KS dominant topic distribution"

    def test_statistic_between_zero_and_one(self):
        (dist_a, dist_b), _ = self._get_dists()
        result = _fast_tester().ks_topic_distributions(dist_a, dist_b)
        assert 0.0 <= result.statistic <= 1.0

    def test_identical_distributions_high_pvalue(self):
        """Same distribution → KS D should be 0 → p should be high."""
        a, b = _make_identical_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)
        result = _fast_tester().ks_topic_distributions(dist_a, dist_b)
        # Identical corpora → same topic assignments → D=0 → p=1
        assert result.p_value > 0.05

    def test_effect_size_equals_statistic(self):
        (dist_a, dist_b), _ = self._get_dists()
        result = _fast_tester().ks_topic_distributions(dist_a, dist_b)
        assert result.effect_size == result.statistic


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Jensen-Shannon divergence
# ─────────────────────────────────────────────────────────────────────────────

class TestJSDBootstrap:
    def _setup(self, n_topics: int = 4):
        a, b = _make_distinct_pair()
        tm = TopicModeler(n_topics=n_topics, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)
        return dist_a, dist_b, tm

    def test_returns_test_result_instance(self):
        dist_a, dist_b, tm = self._setup()
        result = _fast_tester().jsd_bootstrap(dist_a, dist_b, tm)
        assert isinstance(result, TestResult)
        assert result.name == "Jensen-Shannon divergence"

    def test_statistic_between_zero_and_one(self):
        dist_a, dist_b, tm = self._setup()
        result = _fast_tester().jsd_bootstrap(dist_a, dist_b, tm)
        assert 0.0 <= result.statistic <= 1.0

    def test_ci_bounds_are_valid(self):
        dist_a, dist_b, tm = self._setup()
        result = _fast_tester().jsd_bootstrap(dist_a, dist_b, tm)
        assert result.ci_low <= result.ci_high
        assert result.ci_low >= 0.0
        assert result.ci_high <= 1.0

    def test_identical_corpora_near_zero_statistic(self):
        a, b = _make_identical_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)
        result = _fast_tester().jsd_bootstrap(dist_a, dist_b, tm)
        # Identical corpora → JSD near 0 → high p-value
        assert result.p_value > 0.05

    def test_effect_size_equals_statistic(self):
        dist_a, dist_b, tm = self._setup()
        result = _fast_tester().jsd_bootstrap(dist_a, dist_b, tm)
        assert result.effect_size == result.statistic


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Fisher's exact stance contingency
# ─────────────────────────────────────────────────────────────────────────────

class TestFisherStanceContingency:
    def _setup(self):
        a, b = _make_distinct_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)
        return dist_a, dist_b, tm

    def test_returns_test_result_instance(self):
        dist_a, dist_b, tm = self._setup()
        result = _fast_tester().fisher_stance_contingency(dist_a, dist_b, tm)
        assert isinstance(result, TestResult)
        assert result.name == "Fisher stance contingency"

    def test_pvalue_between_zero_and_one(self):
        dist_a, dist_b, tm = self._setup()
        result = _fast_tester().fisher_stance_contingency(dist_a, dist_b, tm)
        assert 0.0 <= result.p_value <= 1.0

    def test_odds_ratio_is_nonnegative(self):
        dist_a, dist_b, tm = self._setup()
        result = _fast_tester().fisher_stance_contingency(dist_a, dist_b, tm)
        assert result.statistic >= 0.0

    def test_effect_size_between_zero_and_one(self):
        dist_a, dist_b, tm = self._setup()
        result = _fast_tester().fisher_stance_contingency(dist_a, dist_b, tm)
        # Phi coefficient ∈ [0, 1]
        assert 0.0 <= result.effect_size <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Burrows' Delta stylometry
# ─────────────────────────────────────────────────────────────────────────────

class TestBurrowsDelta:
    def test_returns_test_result_instance(self):
        a, b = _make_distinct_pair()
        result = _fast_tester().burrows_delta(a, b)
        assert isinstance(result, TestResult)
        assert result.name == "Burrows Delta stylometry"

    def test_identical_corpora_zero_delta(self):
        a = _make_corpus("A", _MILITARY)
        b = _make_corpus("B", _MILITARY)
        result = _fast_tester().burrows_delta(a, b)
        assert result.statistic == pytest.approx(0.0, abs=1e-6)

    def test_identical_corpora_high_pvalue(self):
        a = _make_corpus("A", _MILITARY)
        b = _make_corpus("B", _MILITARY)
        result = _fast_tester().burrows_delta(a, b)
        assert result.p_value > 0.05

    def test_distinct_corpora_nonzero_delta(self):
        a, b = _make_distinct_pair()
        result = _fast_tester().burrows_delta(a, b)
        assert result.statistic > 0.0

    def test_distinct_corpora_low_pvalue(self):
        a, b = _make_distinct_pair()
        result = _fast_tester().burrows_delta(a, b)
        assert result.p_value < 0.05

    def test_statistic_is_nonnegative(self):
        a, b = _make_distinct_pair()
        result = _fast_tester().burrows_delta(a, b)
        assert result.statistic >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Full run() integration
# ─────────────────────────────────────────────────────────────────────────────

class TestFullRun:
    def test_returns_hypothesis_result(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        assert isinstance(result, HypothesisResult)

    def test_has_six_test_results(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        assert len(result.test_results) == 6

    def test_test_result_names_are_unique(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        names = [r.name for r in result.test_results]
        assert len(set(names)) == 6

    def test_expected_test_names_present(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        names = [r.name for r in result.test_results]
        assert "Permutation centroid distance" in names
        assert "Mann-Whitney U cosine similarity" in names
        assert "KS dominant topic distribution" in names
        assert "Jensen-Shannon divergence" in names
        assert "Fisher stance contingency" in names
        assert "Burrows Delta stylometry" in names

    def test_fdr_corrected_flag_is_true(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        assert result.fdr_corrected is True

    def test_speaker_names_recorded(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        assert result.speaker_a == "General"
        assert result.speaker_b == "Ecologist"

    def test_verdict_is_valid_string(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        assert result.verdict in {"REJECTED", "NOT_REJECTED"}

    def test_distinct_corpora_verdict_rejected(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        assert result.verdict == "REJECTED", (
            f"Expected REJECTED for clearly distinct corpora. "
            f"tests_passed={result.tests_passed}/6  "
            f"p-values={[round(r.p_value,4) for r in result.test_results]}"
        )

    def test_identical_corpora_verdict_not_rejected(self, tmp_path):
        a, b = _make_identical_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        assert result.verdict == "NOT_REJECTED", (
            f"Expected NOT_REJECTED for identical corpora. "
            f"tests_passed={result.tests_passed}/6  "
            f"p-values={[round(r.p_value,4) for r in result.test_results]}"
        )

    def test_tests_passed_property(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        assert result.tests_passed == sum(r.passed for r in result.test_results)

    def test_total_tests_property(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        assert result.total_tests == 6

    def test_rejection_rate_in_range(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        assert 0.0 <= result.rejection_rate <= 1.0

    def test_last_topic_modeler_stored(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        tester = _fast_tester()
        tester.run(a, b, emb_a, emb_b)
        assert tester._last_topic_modeler is not None
        assert tester._last_topic_modeler.is_fitted

    def test_pvalue_after_fdr_correction_in_range(self, tmp_path):
        a, b = _make_distinct_pair()
        emb_a, emb_b = _tfidf_embeddings(a, b, tmp_path)
        result = _fast_tester().run(a, b, emb_a, emb_b)
        for tr in result.test_results:
            assert 0.0 <= tr.p_value <= 1.0

"""Tests for vox_romana.analysis.similarity — SimilarityAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from vox_romana.analysis.embedder import CorpusEmbedder
from vox_romana.analysis.models import SimilarityReport, SpeakerCorpus
from vox_romana.analysis.similarity import SimilarityAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
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
]

_CLIMATE = [
    "Climate change threatens ecosystems through rising temperatures and flooding",
    "Carbon emissions from fossil fuels must be reduced to prevent catastrophe",
    "Renewable solar and wind energy creates sustainable economic opportunities",
    "Ocean acidification destroys coral reefs and decimates marine biodiversity",
    "Deforestation accelerates greenhouse warming and destroys wildlife habitat",
    "Environmental regulations protect clean drinking water and breathable air",
    "Green technology innovation reduces industrial pollution and carbon footprint",
    "Sustainable agriculture practices protect soil health and water quality",
]

_NEAR_DUPLICATE = [
    u + " indeed" for u in _MILITARY
]


def _embeddings(a, b, tmp_path):
    emb = CorpusEmbedder(cache_dir=tmp_path, _force_backend="tfidf")
    return emb.embed_pair(a, b)


def _corpus(name, utts):
    return SpeakerCorpus(name=name, utterances=utts)


# ─────────────────────────────────────────────────────────────────────────────
# SimilarityReport model
# ─────────────────────────────────────────────────────────────────────────────

class TestSimilarityReport:
    def test_default_verdict_empty(self):
        r = SimilarityReport(
            ngram_overlap=0.1, semantic_similarity=0.2,
            burrows_delta=1.5, stylometric={}, verdict=""
        )
        assert r.verdict == ""

    def test_fields_accessible(self):
        r = SimilarityReport(
            ngram_overlap=0.5, semantic_similarity=0.7,
            burrows_delta=0.3, stylometric={"ttr_a": 0.8}, verdict="distinct"
        )
        assert r.ngram_overlap == 0.5
        assert r.semantic_similarity == 0.7
        assert r.stylometric["ttr_a"] == 0.8


# ─────────────────────────────────────────────────────────────────────────────
# analyze() — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyze:
    def test_returns_similarity_report(self, tmp_path):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        emb_a, emb_b = _embeddings(a, b, tmp_path)
        report = SimilarityAnalyzer().analyze(a, b, emb_a, emb_b)
        assert isinstance(report, SimilarityReport)

    def test_all_fields_populated(self, tmp_path):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        emb_a, emb_b = _embeddings(a, b, tmp_path)
        report = SimilarityAnalyzer().analyze(a, b, emb_a, emb_b)
        assert 0.0 <= report.ngram_overlap <= 1.0
        assert -1.0 <= report.semantic_similarity <= 1.0
        assert report.burrows_delta >= 0.0
        assert isinstance(report.stylometric, dict)
        assert report.verdict in {"distinct", "moderately similar", "highly similar"}

    def test_distinct_corpora_verdict_distinct(self, tmp_path):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        emb_a, emb_b = _embeddings(a, b, tmp_path)
        report = SimilarityAnalyzer().analyze(a, b, emb_a, emb_b)
        assert report.verdict == "distinct"

    def test_identical_corpora_high_ngram_overlap(self, tmp_path):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _MILITARY)
        emb_a, emb_b = _embeddings(a, b, tmp_path)
        report = SimilarityAnalyzer().analyze(a, b, emb_a, emb_b)
        assert report.ngram_overlap > 0.8

    def test_identical_corpora_verdict_highly_similar(self, tmp_path):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _MILITARY)
        emb_a, emb_b = _embeddings(a, b, tmp_path)
        report = SimilarityAnalyzer().analyze(a, b, emb_a, emb_b)
        assert report.verdict == "highly similar"

    def test_identical_corpora_zero_delta(self, tmp_path):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _MILITARY)
        emb_a, emb_b = _embeddings(a, b, tmp_path)
        report = SimilarityAnalyzer().analyze(a, b, emb_a, emb_b)
        assert report.burrows_delta == pytest.approx(0.0, abs=1e-5)

    def test_near_duplicate_high_ngram_overlap(self, tmp_path):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _NEAR_DUPLICATE)
        emb_a, emb_b = _embeddings(a, b, tmp_path)
        report = SimilarityAnalyzer().analyze(a, b, emb_a, emb_b)
        assert report.ngram_overlap > 0.60

    def test_stylometric_keys_present(self, tmp_path):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        emb_a, emb_b = _embeddings(a, b, tmp_path)
        report = SimilarityAnalyzer().analyze(a, b, emb_a, emb_b)
        for key in ("ttr_a", "ttr_b", "vocab_a", "vocab_b",
                    "mean_len_a", "mean_len_b",
                    "lexical_density_a", "lexical_density_b"):
            assert key in report.stylometric, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# ngram_overlap
# ─────────────────────────────────────────────────────────────────────────────

class TestNgramOverlap:
    def test_identical_text_returns_one(self):
        a = _corpus("A", ["hello world foo bar baz"])
        b = _corpus("B", ["hello world foo bar baz"])
        score = SimilarityAnalyzer().ngram_overlap(a, b)
        assert score == pytest.approx(1.0)

    def test_disjoint_text_returns_low_score(self):
        a = _corpus("A", ["aaa bbb ccc ddd eee fff"])
        b = _corpus("B", ["zzz yyy xxx www vvv uuu"])
        score = SimilarityAnalyzer().ngram_overlap(a, b)
        assert score < 0.3

    def test_score_in_range(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        score = SimilarityAnalyzer().ngram_overlap(a, b)
        assert 0.0 <= score <= 1.0

    def test_symmetric(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        analyzer = SimilarityAnalyzer()
        assert analyzer.ngram_overlap(a, b) == pytest.approx(
            analyzer.ngram_overlap(b, a), rel=1e-6
        )

    def test_empty_text_returns_zero(self):
        # Single-word utterance → minimal ngrams
        a = SpeakerCorpus(name="A", utterances=["abc"])
        b = SpeakerCorpus(name="B", utterances=["xyz"])
        score = SimilarityAnalyzer().ngram_overlap(a, b)
        assert score == 0.0  # no overlap on disjoint chars


# ─────────────────────────────────────────────────────────────────────────────
# semantic_similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticSimilarity:
    def test_identical_embeddings_returns_one(self):
        # All rows are the same unit vector → every dot product = 1.0
        row = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        emb = np.tile(row, (5, 1))
        sim = SimilarityAnalyzer().semantic_similarity(emb, emb)
        assert sim == pytest.approx(1.0)

    def test_orthogonal_embeddings_returns_zero(self):
        # 2D basis vectors are orthogonal — mean cosine sim = 0
        a = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        sim = SimilarityAnalyzer().semantic_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_result_in_range(self, tmp_path):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        emb_a, emb_b = _embeddings(a, b, tmp_path)
        sim = SimilarityAnalyzer().semantic_similarity(emb_a, emb_b)
        assert -1.0 <= sim <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# burrows_delta
# ─────────────────────────────────────────────────────────────────────────────

class TestBurrowsDelta:
    def test_identical_corpora_zero_delta(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _MILITARY)
        delta = SimilarityAnalyzer().burrows_delta(a, b)
        assert delta == pytest.approx(0.0, abs=1e-5)

    def test_distinct_corpora_nonzero_delta(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        delta = SimilarityAnalyzer().burrows_delta(a, b)
        assert delta > 0.0

    def test_nonnegative(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        delta = SimilarityAnalyzer().burrows_delta(a, b)
        assert delta >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# stylometric_features
# ─────────────────────────────────────────────────────────────────────────────

class TestStylometricFeatures:
    def test_returns_dict_with_expected_keys(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        features = SimilarityAnalyzer().stylometric_features(a, b)
        for key in ("ttr_a", "ttr_b", "vocab_a", "vocab_b",
                    "mean_len_a", "mean_len_b",
                    "lexical_density_a", "lexical_density_b"):
            assert key in features

    def test_ttr_in_range(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        features = SimilarityAnalyzer().stylometric_features(a, b)
        assert 0.0 <= features["ttr_a"] <= 1.0
        assert 0.0 <= features["ttr_b"] <= 1.0

    def test_lexical_density_in_range(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        features = SimilarityAnalyzer().stylometric_features(a, b)
        assert 0.0 <= features["lexical_density_a"] <= 1.0
        assert 0.0 <= features["lexical_density_b"] <= 1.0

    def test_vocab_is_positive_integer(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        features = SimilarityAnalyzer().stylometric_features(a, b)
        assert isinstance(features["vocab_a"], int)
        assert features["vocab_a"] > 0

    def test_mean_len_is_positive(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        features = SimilarityAnalyzer().stylometric_features(a, b)
        assert features["mean_len_a"] > 0.0
        assert features["mean_len_b"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Verdict logic
# ─────────────────────────────────────────────────────────────────────────────

class TestVerdict:
    def _analyzer(self):
        return SimilarityAnalyzer(
            highly_similar_ngram=0.70,
            highly_similar_semantic=0.85,
            moderately_similar_ngram=0.40,
            moderately_similar_semantic=0.65,
        )

    def test_highly_similar_via_ngram(self):
        a = self._analyzer()
        assert a._verdict(ngram=0.80, semantic=0.30) == "highly similar"

    def test_highly_similar_via_semantic(self):
        a = self._analyzer()
        assert a._verdict(ngram=0.20, semantic=0.90) == "highly similar"

    def test_moderately_similar_via_ngram(self):
        a = self._analyzer()
        assert a._verdict(ngram=0.50, semantic=0.30) == "moderately similar"

    def test_moderately_similar_via_semantic(self):
        a = self._analyzer()
        assert a._verdict(ngram=0.20, semantic=0.70) == "moderately similar"

    def test_distinct(self):
        a = self._analyzer()
        assert a._verdict(ngram=0.10, semantic=0.20) == "distinct"

    def test_boundary_highly_similar_ngram(self):
        # Threshold is strict (>), so exactly 0.70 is NOT highly similar
        a = self._analyzer()
        assert a._verdict(ngram=0.701, semantic=0.10) == "highly similar"
        assert a._verdict(ngram=0.70, semantic=0.10) == "moderately similar"

    def test_just_below_highly_ngram(self):
        a = self._analyzer()
        assert a._verdict(ngram=0.699, semantic=0.10) == "moderately similar"

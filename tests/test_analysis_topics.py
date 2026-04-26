"""Tests for vox_romana.analysis.topics — TopicModeler."""

from __future__ import annotations

import numpy as np
import pytest

from vox_romana.analysis.models import SpeakerCorpus
from vox_romana.analysis.topics import TopicModeler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEFENSE = [
    "National defense requires increased military spending and weapon procurement",
    "Armed forces must maintain readiness with modern equipment and training",
    "Pentagon funding ensures soldiers have necessary weapons for combat",
    "Strategic nuclear deterrence protects sovereignty from foreign aggression",
    "Military bases require substantial infrastructure investment for personnel",
    "Intelligence agencies monitor terrorist threats to national security",
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


def _make_pair(
    utts_a: list[str] = _DEFENSE,
    utts_b: list[str] = _CLIMATE,
) -> tuple[SpeakerCorpus, SpeakerCorpus]:
    return (
        SpeakerCorpus(name="A", utterances=utts_a),
        SpeakerCorpus(name="B", utterances=utts_b),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestTopicModeler
# ─────────────────────────────────────────────────────────────────────────────

class TestTopicModeler:
    def test_fit_transform_returns_correct_shapes(self):
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)

        assert dist_a.shape == (len(_DEFENSE), 4)
        assert dist_b.shape == (len(_CLIMATE), 4)

    def test_dist_rows_sum_to_approximately_one(self):
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)

        np.testing.assert_allclose(dist_a.sum(axis=1), np.ones(len(_DEFENSE)), atol=1e-4)
        np.testing.assert_allclose(dist_b.sum(axis=1), np.ones(len(_CLIMATE)), atol=1e-4)

    def test_dist_values_are_nonnegative(self):
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)

        assert (dist_a >= 0.0).all()
        assert (dist_b >= 0.0).all()

    def test_dist_dtype_is_float32(self):
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)
        assert dist_a.dtype == np.float32
        assert dist_b.dtype == np.float32

    def test_top_terms_returns_requested_count(self):
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        tm.fit_transform(a, b)
        terms = tm.top_terms(0, n_terms=6)
        assert len(terms) == 6
        assert all(isinstance(t, str) for t in terms)

    def test_top_terms_are_strings(self):
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        tm.fit_transform(a, b)
        for topic_id in range(4):
            terms = tm.top_terms(topic_id, n_terms=5)
            assert all(isinstance(t, str) and len(t) > 0 for t in terms)

    def test_topic_label_format(self):
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        tm.fit_transform(a, b)
        label = tm.topic_label(0, n_terms=4)
        parts = label.split("/")
        assert len(parts) == 4
        assert all(len(p) > 0 for p in parts)

    def test_corpus_weights_sum_to_one(self):
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)
        w_a = tm.corpus_weights(dist_a)
        w_b = tm.corpus_weights(dist_b)
        assert abs(float(w_a.sum()) - 1.0) < 1e-4
        assert abs(float(w_b.sum()) - 1.0) < 1e-4

    def test_corpus_weights_nonnegative(self):
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, _ = tm.fit_transform(a, b)
        w = tm.corpus_weights(dist_a)
        assert (w >= 0.0).all()

    def test_corpus_weights_dtype_is_float32(self):
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, _ = tm.fit_transform(a, b)
        w = tm.corpus_weights(dist_a)
        assert w.dtype == np.float32

    def test_corpus_weights_zero_input_returns_uniform(self):
        tm = TopicModeler(n_topics=4, random_state=42)
        # Manually set is_fitted state with a zero dist
        a, b = _make_pair()
        tm.fit_transform(a, b)
        zero_dist = np.zeros((5, 4), dtype=np.float32)
        w = tm.corpus_weights(zero_dist)
        np.testing.assert_allclose(w, np.full(4, 0.25, dtype=np.float32), atol=1e-5)

    def test_not_fitted_raises_on_top_terms(self):
        tm = TopicModeler(n_topics=4)
        with pytest.raises(RuntimeError, match="not fitted"):
            tm.top_terms(0)

    def test_is_fitted_flag(self):
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        assert not tm.is_fitted
        tm.fit_transform(a, b)
        assert tm.is_fitted

    def test_n_topics_capped_for_small_corpus(self):
        """n_topics must not exceed combined utterance count."""
        tiny_a = SpeakerCorpus(name="A", utterances=[
            "Alpha beta gamma delta epsilon zeta eta",
            "Theta iota kappa lambda mu nu xi omicron",
        ])
        tiny_b = SpeakerCorpus(name="B", utterances=[
            "Pi rho sigma tau upsilon phi chi psi",
            "Omega alpha beta gamma delta epsilon zeta",
        ])
        tm = TopicModeler(n_topics=50, random_state=42)
        dist_a, dist_b = tm.fit_transform(tiny_a, tiny_b)
        # Should not raise; n_topics auto-capped to max(2, 4-1) = 3
        assert dist_a.shape[1] >= 2
        assert dist_b.shape[1] >= 2

    def test_topics_shared_between_corpora(self):
        """Both dist arrays must have the same number of topics."""
        a, b = _make_pair()
        tm = TopicModeler(n_topics=5, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)
        assert dist_a.shape[1] == dist_b.shape[1]

    def test_distinct_corpora_have_different_weights(self):
        """Defense and climate corpora should produce different mean topic weights."""
        a, b = _make_pair()
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)
        w_a = tm.corpus_weights(dist_a)
        w_b = tm.corpus_weights(dist_b)
        diff = float(np.abs(w_a - w_b).sum())
        assert diff > 0.01, "Distinct corpora should have non-identical topic weights"

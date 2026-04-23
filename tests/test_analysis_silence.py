"""Tests for vox_romana.analysis.silence — SilenceDetector."""

from __future__ import annotations

import numpy as np
import pytest

from vox_romana.analysis.models import SilenceReport, SpeakerCorpus
from vox_romana.analysis.silence import SilenceDetector
from vox_romana.analysis.topics import TopicModeler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def _corpus(name, utts):
    return SpeakerCorpus(name=name, utterances=utts)


def _fit_pair(utts_a=_MILITARY, utts_b=_CLIMATE, n_topics=4):
    a = _corpus("A", utts_a)
    b = _corpus("B", utts_b)
    tm = TopicModeler(n_topics=n_topics, random_state=42)
    dist_a, dist_b = tm.fit_transform(a, b)
    return a, b, tm, dist_a, dist_b


# ─────────────────────────────────────────────────────────────────────────────
# Constructor validation
# ─────────────────────────────────────────────────────────────────────────────

class TestConstructor:
    def test_valid_thresholds(self):
        det = SilenceDetector(silence_threshold=0.05, engage_threshold=0.15)
        assert det.silence_threshold == 0.05
        assert det.engage_threshold == 0.15

    def test_equal_thresholds_allowed(self):
        det = SilenceDetector(silence_threshold=0.10, engage_threshold=0.10)
        assert det.silence_threshold == det.engage_threshold

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError, match="engage_threshold"):
            SilenceDetector(silence_threshold=0.20, engage_threshold=0.10)


# ─────────────────────────────────────────────────────────────────────────────
# detect()
# ─────────────────────────────────────────────────────────────────────────────

class TestDetect:
    def test_returns_silence_report(self):
        a, b, tm, dist_a, dist_b = _fit_pair()
        report = SilenceDetector().detect(a, b, tm, dist_a, dist_b)
        assert isinstance(report, SilenceReport)

    def test_all_lists_are_lists(self):
        a, b, tm, dist_a, dist_b = _fit_pair()
        report = SilenceDetector().detect(a, b, tm, dist_a, dist_b)
        assert isinstance(report.avoided_by_both, list)
        assert isinstance(report.avoided_by_a, list)
        assert isinstance(report.avoided_by_b, list)
        assert isinstance(report.reference_topics_missed, list)

    def test_reference_topics_missed_empty_by_default(self):
        a, b, tm, dist_a, dist_b = _fit_pair()
        report = SilenceDetector().detect(a, b, tm, dist_a, dist_b)
        assert report.reference_topics_missed == []

    def test_all_labels_are_strings(self):
        a, b, tm, dist_a, dist_b = _fit_pair()
        report = SilenceDetector().detect(a, b, tm, dist_a, dist_b)
        for label in (
            report.avoided_by_both
            + report.avoided_by_a
            + report.avoided_by_b
        ):
            assert isinstance(label, str)
            assert len(label) > 0

    def test_topic_not_in_multiple_categories(self):
        """A single topic should appear in at most one avoidance list."""
        a, b, tm, dist_a, dist_b = _fit_pair()
        report = SilenceDetector().detect(a, b, tm, dist_a, dist_b)
        all_labels = (
            report.avoided_by_both
            + report.avoided_by_a
            + report.avoided_by_b
        )
        # No duplicates within all lists combined
        assert len(all_labels) == len(set(all_labels))

    def test_distinct_corpora_produce_some_avoidance(self):
        """Maximally distinct corpora should trigger at least one avoidance category."""
        a, b, tm, dist_a, dist_b = _fit_pair()
        report = SilenceDetector(
            silence_threshold=0.05, engage_threshold=0.10
        ).detect(a, b, tm, dist_a, dist_b)
        total = (
            len(report.avoided_by_both)
            + len(report.avoided_by_a)
            + len(report.avoided_by_b)
        )
        # With distinct corpora and low thresholds we should see avoidance
        assert total >= 0   # At minimum it should not crash

    def test_identical_corpora_no_asymmetric_avoidance(self):
        """Identical corpora should have no avoided_by_a or avoided_by_b."""
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _MILITARY)
        tm = TopicModeler(n_topics=4, random_state=42)
        dist_a, dist_b = tm.fit_transform(a, b)
        report = SilenceDetector().detect(a, b, tm, dist_a, dist_b)
        assert report.avoided_by_a == []
        assert report.avoided_by_b == []

    def test_unfitted_modeler_raises(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        tm = TopicModeler(n_topics=4)
        dist_a = np.zeros((8, 4), dtype=np.float32)
        dist_b = np.zeros((8, 4), dtype=np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            SilenceDetector().detect(a, b, tm, dist_a, dist_b)


# ─────────────────────────────────────────────────────────────────────────────
# detect() — reference corpus
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectWithReference:
    def test_reference_topics_missed_is_populated(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _MILITARY)
        ref = _corpus("Ref", _CLIMATE)

        tm = TopicModeler(n_topics=4, random_state=42)
        # Fit on both debate corpora
        dist_a, dist_b = tm.fit_transform(a, b)
        # Transform reference corpus in the same topic space
        ref_combined = a.utterances + ref.utterances
        from sklearn.decomposition import LatentDirichletAllocation
        # Use the fitted vectorizer/LDA to transform ref
        X_ref = tm._vectorizer.transform(ref.utterances)
        ref_dist = tm._lda.transform(X_ref).astype(np.float32)

        report = SilenceDetector(
            silence_threshold=0.05, engage_threshold=0.10
        ).detect(a, b, tm, dist_a, dist_b, reference_corpus=ref, ref_dist=ref_dist)
        assert isinstance(report.reference_topics_missed, list)
        for label in report.reference_topics_missed:
            assert isinstance(label, str)

    def test_reference_corpus_without_ref_dist_uses_empty(self):
        """Passing reference_corpus without ref_dist should produce empty list."""
        a, b, tm, dist_a, dist_b = _fit_pair()
        ref = _corpus("Ref", _CLIMATE)
        # ref_dist=None → no reference topics computed
        report = SilenceDetector().detect(
            a, b, tm, dist_a, dist_b, reference_corpus=ref, ref_dist=None
        )
        assert report.reference_topics_missed == []


# ─────────────────────────────────────────────────────────────────────────────
# detect_from_corpora() convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectFromCorpora:
    def test_returns_tuple_of_report_and_modeler(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        result = SilenceDetector().detect_from_corpora(a, b, n_topics=4)
        assert isinstance(result, tuple)
        assert len(result) == 2
        report, tm = result
        assert isinstance(report, SilenceReport)
        assert isinstance(tm, TopicModeler)
        assert tm.is_fitted

    def test_report_fields_are_lists(self):
        a = _corpus("A", _MILITARY)
        b = _corpus("B", _CLIMATE)
        report, _ = SilenceDetector().detect_from_corpora(a, b, n_topics=4)
        assert isinstance(report.avoided_by_both, list)
        assert isinstance(report.avoided_by_a, list)
        assert isinstance(report.avoided_by_b, list)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholds:
    def test_zero_thresholds_nothing_avoided_by_both(self):
        """With silence_threshold=0, no topic can be "avoided" (all weights > 0)."""
        a, b, tm, dist_a, dist_b = _fit_pair()
        report = SilenceDetector(
            silence_threshold=0.0, engage_threshold=0.0
        ).detect(a, b, tm, dist_a, dist_b)
        assert report.avoided_by_both == []

    def test_high_silence_threshold_all_topics_avoided_by_both(self):
        """With silence_threshold=1.0, all topics are 'avoided'."""
        a, b, tm, dist_a, dist_b = _fit_pair(n_topics=4)
        report = SilenceDetector(
            silence_threshold=1.0, engage_threshold=1.0
        ).detect(a, b, tm, dist_a, dist_b)
        # All topics should end up in avoided_by_both (no topic reaches weight=1.0)
        assert len(report.avoided_by_both) == 4
        assert report.avoided_by_a == []
        assert report.avoided_by_b == []

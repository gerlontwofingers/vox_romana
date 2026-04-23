"""Tests for vox_romana.emotion_flow.models — all dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from vox_romana.emotion_flow.models import (
    EMOTION_LABELS,
    EmotionFlowReport,
    EmotionSequence,
    EmotionTag,
    NullTestResult,
    ReproducibilityManifest,
    SemanticFlowResult,
    SpeechSegment,
    TransitionMatrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# EMOTION_LABELS
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionLabels:
    def test_has_seven_classes(self):
        assert len(EMOTION_LABELS) == 7

    def test_contains_expected_labels(self):
        for label in ("anger", "fear", "joy", "sadness", "neutral"):
            assert label in EMOTION_LABELS

    def test_no_duplicates(self):
        assert len(EMOTION_LABELS) == len(set(EMOTION_LABELS))


# ─────────────────────────────────────────────────────────────────────────────
# SpeechSegment
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeechSegment:
    def _make(self, **kwargs) -> SpeechSegment:
        defaults = dict(
            text="The economy needs reform.",
            original_text="Die Wirtschaft braucht Reformen.",
            source_lang="de",
            translation_confidence=0.91,
            translation_flagged=False,
            segment_index=0,
            speech_id="speech_01",
        )
        defaults.update(kwargs)
        return SpeechSegment(**defaults)

    def test_creation(self):
        seg = self._make()
        assert seg.source_lang == "de"
        assert seg.translation_confidence == 0.91
        assert not seg.translation_flagged

    def test_flagged_below_threshold(self):
        seg = self._make(translation_confidence=0.72, translation_flagged=True)
        assert seg.translation_flagged

    def test_passthrough_english(self):
        seg = self._make(
            text="Hello world.",
            original_text="Hello world.",
            source_lang="en",
            translation_confidence=1.0,
            translation_flagged=False,
        )
        assert seg.translation_confidence == 1.0
        assert not seg.translation_flagged


# ─────────────────────────────────────────────────────────────────────────────
# EmotionTag
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionTag:
    def _make(self) -> EmotionTag:
        scores = {label: 1.0 / len(EMOTION_LABELS) for label in EMOTION_LABELS}
        scores["fear"] = 0.55
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}
        return EmotionTag(dominant="fear", dominant_score=scores["fear"], scores=scores)

    def test_dominant_matches_scores(self):
        tag = self._make()
        assert tag.dominant == max(tag.scores, key=tag.scores.__getitem__)

    def test_scores_cover_all_labels(self):
        tag = self._make()
        for label in EMOTION_LABELS:
            assert label in tag.scores


# ─────────────────────────────────────────────────────────────────────────────
# EmotionSequence
# ─────────────────────────────────────────────────────────────────────────────

def _make_segment(idx: int, speech_id: str = "s1") -> SpeechSegment:
    return SpeechSegment(
        text=f"Segment {idx}",
        original_text=f"Segment {idx}",
        source_lang="en",
        translation_confidence=1.0,
        translation_flagged=False,
        segment_index=idx,
        speech_id=speech_id,
    )


def _make_tag(dominant: str = "neutral") -> EmotionTag:
    scores = {label: 0.0 for label in EMOTION_LABELS}
    scores[dominant] = 1.0
    return EmotionTag(dominant=dominant, dominant_score=1.0, scores=scores)


class TestEmotionSequence:
    def test_n_segments(self):
        segs = [_make_segment(i) for i in range(5)]
        tags = [_make_tag() for _ in range(5)]
        seq = EmotionSequence(leader_name="Alice", segments=segs, tags=tags, speech_ids=["s1"])
        assert seq.n_segments == 5

    def test_texts_property(self):
        segs = [_make_segment(i) for i in range(3)]
        seq = EmotionSequence(leader_name="Alice", segments=segs, tags=[], speech_ids=["s1"])
        assert seq.texts == ["Segment 0", "Segment 1", "Segment 2"]

    def test_from_pooled(self):
        seqs = [
            EmotionSequence(
                leader_name="Alice",
                segments=[_make_segment(i, speech_id=f"speech_{n}") for i in range(3)],
                tags=[_make_tag() for _ in range(3)],
                speech_ids=[f"speech_{n}"],
            )
            for n in range(2)
        ]
        pooled = EmotionSequence.from_pooled(seqs)
        assert pooled.n_segments == 6
        assert len(pooled.speech_ids) == 2

    def test_from_pooled_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            EmotionSequence.from_pooled([])

    def test_speech_ids_preserved_on_pool(self):
        seqs = [
            EmotionSequence(
                leader_name="L",
                segments=[_make_segment(0, speech_id="a")],
                tags=[_make_tag()],
                speech_ids=["a"],
            ),
            EmotionSequence(
                leader_name="L",
                segments=[_make_segment(0, speech_id="b")],
                tags=[_make_tag()],
                speech_ids=["b"],
            ),
        ]
        pooled = EmotionSequence.from_pooled(seqs)
        assert "a" in pooled.speech_ids
        assert "b" in pooled.speech_ids


# ─────────────────────────────────────────────────────────────────────────────
# TransitionMatrix
# ─────────────────────────────────────────────────────────────────────────────

class TestTransitionMatrix:
    def _make(self) -> TransitionMatrix:
        n = len(EMOTION_LABELS)
        matrix = np.eye(n, dtype=np.float32) / n
        counts = np.ones((n, n), dtype=np.int32)
        ci_low = np.zeros((n, n), dtype=np.float32)
        ci_high = np.ones((n, n), dtype=np.float32)
        return TransitionMatrix(
            leader_name="Alice",
            matrix=matrix,
            count_matrix=counts,
            ci_low=ci_low,
            ci_high=ci_high,
            emotion_labels=list(EMOTION_LABELS),
            smoothing_k=0.1,
            n_transitions=42,
        )

    def test_matrix_shape(self):
        tm = self._make()
        n = len(EMOTION_LABELS)
        assert tm.matrix.shape == (n, n)
        assert tm.ci_low.shape == (n, n)

    def test_count_matrix_dtype(self):
        tm = self._make()
        assert tm.count_matrix.dtype == np.int32

    def test_labels_match_matrix_dimension(self):
        tm = self._make()
        assert len(tm.emotion_labels) == tm.matrix.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# ReproducibilityManifest
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibilityManifest:
    def _make(self) -> ReproducibilityManifest:
        return ReproducibilityManifest(
            seed=42,
            python_version="3.12.3",
            package_versions={"numpy": "1.26.4", "torch": "2.4.0"},
            model_versions={"j-hartmann/emotion-english-distilroberta-base": "abc123"},
            hardware="Apple M3 Pro, CPU",
            timestamp="2026-04-23T10:00:00+00:00",
            git_commit="a1b2c3d",
            deterministic_mode=True,
        )

    def test_creation(self):
        m = self._make()
        assert m.seed == 42
        assert m.deterministic_mode is True

    def test_model_versions_stored(self):
        m = self._make()
        assert "j-hartmann/emotion-english-distilroberta-base" in m.model_versions


# ─────────────────────────────────────────────────────────────────────────────
# EmotionFlowReport
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionFlowReport:
    def test_defaults(self):
        report = EmotionFlowReport(
            leader_name="Test Leader",
            n_speeches=3,
            n_segments=45,
            dominant_cycle=["fear", "anger", "fear"],
        )
        assert report.narrative == ""
        assert report.n_flagged_segments == 0
        assert report.back_translation_agreement == 1.0
        assert report.transition_matrix is None

    def test_with_all_fields(self):
        n = len(EMOTION_LABELS)
        tm = TransitionMatrix(
            leader_name="L",
            matrix=np.eye(n, dtype=np.float32),
            count_matrix=np.ones((n, n), dtype=np.int32),
            ci_low=np.zeros((n, n), dtype=np.float32),
            ci_high=np.ones((n, n), dtype=np.float32),
            emotion_labels=list(EMOTION_LABELS),
            smoothing_k=0.1,
            n_transitions=20,
        )
        report = EmotionFlowReport(
            leader_name="Leader",
            n_speeches=1,
            n_segments=20,
            dominant_cycle=["fear", "anger"],
            transition_matrix=tm,
            n_flagged_segments=3,
            back_translation_agreement=0.93,
            narrative="Fear dominates early speeches.",
        )
        assert report.transition_matrix is not None
        assert report.narrative == "Fear dominates early speeches."

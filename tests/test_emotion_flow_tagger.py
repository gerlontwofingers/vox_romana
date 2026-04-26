"""Tests for vox_romana.emotion_flow.tagger — EmotionTaggerAgent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vox_romana.emotion_flow.tagger import (
    DEFAULT_BATCH_SIZE,
    MODEL_NAME,
    EmotionTaggerAgent,
    _forced_tag,
    _parse_output,
)
from vox_romana.emotion_flow.models import EMOTION_LABELS, EmotionTag, SpeechSegment


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_segment(text: str, idx: int = 0) -> SpeechSegment:
    return SpeechSegment(
        text=text,
        original_text=text,
        source_lang="en",
        translation_confidence=1.0,
        translation_flagged=False,
        segment_index=idx,
        speech_id="s1",
    )


def _make_segments(n: int) -> list[SpeechSegment]:
    texts = [
        "We shall fight on the beaches.",
        "Victory at all costs.",
        "Never surrender under any circumstances.",
        "This is our finest hour.",
        "We shall never fail.",
    ]
    return [_make_segment(texts[i % len(texts)], idx=i) for i in range(n)]


def _uniform_pipeline_output(texts, **kwargs):
    """Mock pipeline that returns uniform scores for every text."""
    base = 1.0 / len(EMOTION_LABELS)
    return [
        [{"label": l, "score": base} for l in EMOTION_LABELS]
        for _ in texts
    ]


def _make_mock_factory(dominant_label: str = "neutral"):
    """Factory returning a mock pipeline where dominant_label wins."""
    def factory(model, **kwargs):
        pipe = MagicMock()
        scores = {l: 0.01 for l in EMOTION_LABELS}
        scores[dominant_label] = 0.94
        def side_effect(texts, **kw):
            return [
                [{"label": l, "score": scores[l]} for l in EMOTION_LABELS]
                for _ in texts
            ]
        pipe.side_effect = side_effect
        return pipe
    return factory


# ─────────────────────────────────────────────────────────────────────────────
# _force_label mode
# ─────────────────────────────────────────────────────────────────────────────

class TestForcedLabel:
    def test_all_segments_get_forced_label(self):
        agent = EmotionTaggerAgent(_force_label="joy")
        result = agent.tag(_make_segments(5))
        assert all(t.dominant == "joy" for t in result)

    def test_forced_label_score_is_one(self):
        agent = EmotionTaggerAgent(_force_label="fear")
        tags = agent.tag(_make_segments(3))
        assert all(t.dominant_score == 1.0 for t in tags)

    def test_other_scores_are_zero(self):
        agent = EmotionTaggerAgent(_force_label="sadness")
        tags = agent.tag(_make_segments(2))
        for tag in tags:
            for label in EMOTION_LABELS:
                if label != "sadness":
                    assert tag.scores[label] == 0.0

    def test_empty_returns_empty(self):
        agent = EmotionTaggerAgent(_force_label="neutral")
        assert agent.tag([]) == []

    def test_result_parallel_to_input(self):
        agent = EmotionTaggerAgent(_force_label="anger")
        segs = _make_segments(7)
        tags = agent.tag(segs)
        assert len(tags) == len(segs)

    def test_invalid_force_label_raises(self):
        agent = EmotionTaggerAgent(_force_label="happiness")  # not a valid label
        with pytest.raises(ValueError, match="Unknown emotion label"):
            agent.tag(_make_segments(1))


# ─────────────────────────────────────────────────────────────────────────────
# Mock pipeline injection
# ─────────────────────────────────────────────────────────────────────────────

class TestMockPipeline:
    def test_dominant_label_selected_correctly(self):
        agent = EmotionTaggerAgent(_pipeline_factory=_make_mock_factory("fear"))
        tags = agent.tag(_make_segments(3))
        assert all(t.dominant == "fear" for t in tags)

    def test_scores_dict_contains_all_labels(self):
        agent = EmotionTaggerAgent(_pipeline_factory=_make_mock_factory("joy"))
        tags = agent.tag(_make_segments(2))
        for tag in tags:
            for label in EMOTION_LABELS:
                assert label in tag.scores

    def test_pipeline_called_with_correct_texts(self):
        mock_pipe = MagicMock()
        mock_pipe.side_effect = _uniform_pipeline_output

        def factory(model, **kwargs):
            return mock_pipe

        segs = _make_segments(3)
        agent = EmotionTaggerAgent(_pipeline_factory=factory)
        agent.tag(segs)

        mock_pipe.assert_called_once()
        call_texts = mock_pipe.call_args[0][0]
        assert call_texts == [s.text for s in segs]

    def test_batch_size_controls_pipeline_calls(self):
        call_count = [0]

        def factory(model, **kwargs):
            pipe = MagicMock()
            def side_effect(texts, **kw):
                call_count[0] += 1
                return _uniform_pipeline_output(texts)
            pipe.side_effect = side_effect
            return pipe

        # 5 segments with batch_size=2 → ceil(5/2) = 3 calls
        agent = EmotionTaggerAgent(batch_size=2, _pipeline_factory=factory)
        agent.tag(_make_segments(5))
        assert call_count[0] == 3

    def test_model_name_passed_to_factory(self):
        received_model = [None]

        def factory(model, **kwargs):
            received_model[0] = model
            pipe = MagicMock()
            pipe.side_effect = _uniform_pipeline_output
            return pipe

        agent = EmotionTaggerAgent(_pipeline_factory=factory)
        agent.tag(_make_segments(1))
        assert received_model[0] == MODEL_NAME

    def test_pipeline_created_once_and_reused(self):
        factory_calls = [0]

        def factory(model, **kwargs):
            factory_calls[0] += 1
            pipe = MagicMock()
            pipe.side_effect = _uniform_pipeline_output
            return pipe

        agent = EmotionTaggerAgent(_pipeline_factory=factory)
        agent.tag(_make_segments(2))
        agent.tag(_make_segments(2))
        assert factory_calls[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# _parse_output unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParseOutput:
    def _uniform_result(self) -> list[dict]:
        base = 1.0 / len(EMOTION_LABELS)
        return [{"label": l, "score": base} for l in EMOTION_LABELS]

    def test_selects_highest_score(self):
        result = [{"label": l, "score": 0.01} for l in EMOTION_LABELS]
        result[0] = {"label": EMOTION_LABELS[0], "score": 0.94}
        tag = _parse_output(result)
        assert tag.dominant == EMOTION_LABELS[0]
        assert tag.dominant_score == pytest.approx(0.94)

    def test_all_labels_present_in_scores(self):
        tag = _parse_output(self._uniform_result())
        for label in EMOTION_LABELS:
            assert label in tag.scores

    def test_handles_uppercase_label(self):
        result = [{"label": l.upper(), "score": 0.01} for l in EMOTION_LABELS]
        result[2] = {"label": EMOTION_LABELS[2].upper(), "score": 0.9}
        tag = _parse_output(result)
        assert tag.dominant == EMOTION_LABELS[2]

    def test_missing_label_defaults_to_zero(self):
        # Provide only 6 of the 7 labels
        result = [{"label": l, "score": 0.1} for l in EMOTION_LABELS[:-1]]
        tag = _parse_output(result)
        assert tag.scores[EMOTION_LABELS[-1]] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _forced_tag unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestForcedTag:
    def test_dominant_score_is_one(self):
        tag = _forced_tag("anger")
        assert tag.dominant_score == 1.0

    def test_others_are_zero(self):
        tag = _forced_tag("joy")
        assert tag.scores["anger"] == 0.0
        assert tag.scores["fear"] == 0.0

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Unknown emotion label"):
            _forced_tag("happiness")

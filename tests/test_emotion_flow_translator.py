"""Tests for vox_romana.emotion_flow.translator — TranslatorAgent.

All tests use _force_passthrough=True or inject a mock _pipeline_factory
so that no real HuggingFace models are loaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vox_romana.emotion_flow.translator import (
    BERT_FLAG_THRESHOLD,
    TranslatorAgent,
    _bert_score_f1,
    _detect_language,
)
from vox_romana.emotion_flow.models import SpeechSegment, EMOTION_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_segment(text: str, idx: int = 0, speech_id: str = "s1") -> SpeechSegment:
    return SpeechSegment(
        text=text,
        original_text=text,
        source_lang="",
        translation_confidence=0.0,
        translation_flagged=False,
        segment_index=idx,
        speech_id=speech_id,
    )


def _english_segments(n: int = 3) -> list[SpeechSegment]:
    texts = [
        "We shall fight on the beaches with great determination.",
        "We shall never surrender under any circumstances whatsoever.",
        "Victory at all costs, whatever the road may bring us.",
    ]
    return [_make_segment(texts[i % len(texts)], idx=i) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# _force_passthrough mode
# ─────────────────────────────────────────────────────────────────────────────

class TestForcePassthrough:
    def test_returns_same_count(self):
        agent = TranslatorAgent(_force_passthrough=True)
        segs = _english_segments(4)
        result = agent.translate(segs)
        assert len(result) == 4

    def test_sets_source_lang_to_en(self):
        agent = TranslatorAgent(_force_passthrough=True)
        result = agent.translate(_english_segments(2))
        assert all(s.source_lang == "en" for s in result)

    def test_confidence_is_one(self):
        agent = TranslatorAgent(_force_passthrough=True)
        result = agent.translate(_english_segments(3))
        assert all(s.translation_confidence == 1.0 for s in result)

    def test_not_flagged(self):
        agent = TranslatorAgent(_force_passthrough=True)
        result = agent.translate(_english_segments(3))
        assert all(not s.translation_flagged for s in result)

    def test_text_preserved(self):
        agent = TranslatorAgent(_force_passthrough=True)
        segs = _english_segments(2)
        result = agent.translate(segs)
        for orig, out in zip(segs, result):
            assert out.text == orig.original_text

    def test_empty_list_returns_empty(self):
        agent = TranslatorAgent(_force_passthrough=True)
        assert agent.translate([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Mock pipeline injection (non-English translation path)
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_factory(translated: str = "translated text", back: str = "original approx"):
    """Return a factory that produces a mock pipeline with fixed outputs."""
    def factory(task, model, **kwargs):
        pipe = MagicMock()
        pipe.return_value = [{"translation_text": translated}]
        return pipe
    return factory


class TestMockTranslation:
    def test_non_english_calls_pipeline(self):
        factory = _make_mock_factory(translated="The economy needs reform.")
        agent = TranslatorAgent(_pipeline_factory=factory)

        seg = SpeechSegment(
            text="Die Wirtschaft braucht Reformen.",
            original_text="Die Wirtschaft braucht Reformen.",
            source_lang="",
            translation_confidence=0.0,
            translation_flagged=False,
            segment_index=0,
            speech_id="s1",
        )
        with patch("vox_romana.emotion_flow.translator._detect_language", return_value="de"), \
             patch("vox_romana.emotion_flow.translator._bert_score_f1", return_value=0.91):
            result = agent.translate([seg])

        assert len(result) == 1
        assert result[0].source_lang == "de"
        assert result[0].translation_confidence == 0.91
        assert not result[0].translation_flagged

    def test_low_bert_score_flags_segment(self):
        factory = _make_mock_factory(translated="something translated")
        agent = TranslatorAgent(_pipeline_factory=factory, bert_threshold=0.85)
        seg = _make_segment("Etwas auf Deutsch das übersetzt werden muss.")
        with patch("vox_romana.emotion_flow.translator._detect_language", return_value="de"), \
             patch("vox_romana.emotion_flow.translator._bert_score_f1", return_value=0.72):
            result = agent.translate([seg])
        assert result[0].translation_flagged
        assert result[0].translation_confidence == 0.72

    def test_english_passthrough_no_pipeline_called(self):
        factory = MagicMock()
        agent = TranslatorAgent(_pipeline_factory=factory)
        seg = _make_segment("This is an English sentence here.")
        with patch("vox_romana.emotion_flow.translator._detect_language", return_value="en"):
            result = agent.translate([seg])
        factory.assert_not_called()
        assert result[0].translation_confidence == 1.0

    def test_segment_index_and_speech_id_preserved(self):
        factory = _make_mock_factory()
        agent = TranslatorAgent(_pipeline_factory=factory)
        seg = SpeechSegment(
            text="Deutsche Rede.",
            original_text="Deutsche Rede.",
            source_lang="",
            translation_confidence=0.0,
            translation_flagged=False,
            segment_index=7,
            speech_id="bundestag_1945",
        )
        with patch("vox_romana.emotion_flow.translator._detect_language", return_value="en"):
            result = agent.translate([seg])
        assert result[0].segment_index == 7
        assert result[0].speech_id == "bundestag_1945"


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_quality
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateQuality:
    def _make_translated_segments(self) -> list[SpeechSegment]:
        return [
            SpeechSegment(
                text="text", original_text="text", source_lang="de",
                translation_confidence=0.92, translation_flagged=False,
                segment_index=0, speech_id="s1",
            ),
            SpeechSegment(
                text="text", original_text="text", source_lang="de",
                translation_confidence=0.71, translation_flagged=True,
                segment_index=1, speech_id="s1",
            ),
            SpeechSegment(
                text="text", original_text="text", source_lang="en",
                translation_confidence=1.0, translation_flagged=False,
                segment_index=2, speech_id="s1",
            ),
        ]

    def test_counts_flagged(self):
        agent = TranslatorAgent(_force_passthrough=True)
        q = agent.aggregate_quality(self._make_translated_segments())
        assert q["n_flagged"] == 1

    def test_mean_confidence(self):
        agent = TranslatorAgent(_force_passthrough=True)
        q = agent.aggregate_quality(self._make_translated_segments())
        expected = (0.92 + 0.71 + 1.0) / 3
        assert abs(q["mean_confidence"] - expected) < 1e-6

    def test_languages_seen(self):
        agent = TranslatorAgent(_force_passthrough=True)
        q = agent.aggregate_quality(self._make_translated_segments())
        assert "de" in q["languages_seen"]
        assert "en" in q["languages_seen"]

    def test_empty_returns_defaults(self):
        agent = TranslatorAgent(_force_passthrough=True)
        q = agent.aggregate_quality([])
        assert q["n_total"] == 0
        assert q["mean_confidence"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _detect_language (unit — mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectLanguage:
    def test_returns_empty_string_on_failure(self):
        with patch("vox_romana.emotion_flow.translator.detect", side_effect=Exception("fail")):
            result = _detect_language("x")
        assert result == ""

    def test_returns_string(self):
        with patch("vox_romana.emotion_flow.translator.detect", return_value="fr"):
            result = _detect_language("Bonjour le monde")
        assert result == "fr"


# ─────────────────────────────────────────────────────────────────────────────
# _bert_score_f1 (unit — mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestBertScoreF1:
    def test_returns_float_on_success(self):
        mock_f = MagicMock()
        mock_f.mean.return_value.item.return_value = 0.88
        with patch("vox_romana.emotion_flow.translator.bs_score", return_value=(None, None, mock_f)):
            result = _bert_score_f1(["ref"], ["hyp"], lang="de")
        assert abs(result - 0.88) < 1e-6

    def test_returns_fallback_on_failure(self):
        with patch("vox_romana.emotion_flow.translator.bs_score", side_effect=RuntimeError("fail")):
            result = _bert_score_f1(["ref"], ["hyp"], lang="de")
        assert result == 0.8

"""Tests for vox_romana.analysis.stance — StanceExtractor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from vox_romana.analysis.models import SpeakerCorpus, StanceRecord
from vox_romana.analysis.stance import StanceExtractor
from vox_romana.analysis.topics import TopicModeler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_MILITARY = [
    "National defense requires increased military spending and procurement",
    "Armed forces must maintain readiness with modern equipment and training",
    "Pentagon funding ensures soldiers have necessary weapons for combat",
    "Strategic nuclear deterrence protects sovereignty from foreign aggression",
    "Military bases require substantial infrastructure investment and support",
    "Intelligence agencies monitor terrorist threats to national security",
    "Naval warships patrol international waters to enforce maritime agreements",
    "Air force jets defend territorial airspace against hostile incursion",
]

_CLIMATE = [
    "Climate change threatens ecosystems through rising temperatures and flooding",
    "Carbon emissions from fossil fuels must be reduced to prevent catastrophe",
    "Renewable solar and wind energy creates sustainable economic opportunities",
    "Ocean acidification destroys coral reefs and marine biodiversity completely",
    "Deforestation accelerates greenhouse warming and destroys wildlife habitat",
    "Environmental regulations protect clean drinking water and breathable air",
    "Green technology innovation reduces industrial pollution and carbon footprint",
    "Sustainable agriculture practices protect soil health and water quality",
]


def _fit_topics(utts_a=_MILITARY, utts_b=_CLIMATE, n_topics=4):
    a = SpeakerCorpus(name="General", utterances=utts_a)
    b = SpeakerCorpus(name="Ecologist", utterances=utts_b)
    tm = TopicModeler(n_topics=n_topics, random_state=42)
    dist_a, dist_b = tm.fit_transform(a, b)
    return a, b, tm, dist_a, dist_b


def _mock_client(position="pro", summary="Speaker supports this position.", confidence=0.8):
    """Return a mock DeepSeekClient whose .chat() returns valid JSON."""
    client = MagicMock()
    response = json.dumps({
        "position": position,
        "summary": summary,
        "confidence": confidence,
    })
    client.chat.return_value = response
    return client


# ─────────────────────────────────────────────────────────────────────────────
# StanceExtractor.extract — no-LLM path (DEEPSEEK_API_KEY absent)
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractNoLLM:
    def test_returns_list(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        a, b, tm, dist_a, dist_b = _fit_topics()
        extractor = StanceExtractor()
        records = extractor.extract(a, b, tm, dist_a, dist_b)
        assert isinstance(records, list)

    def test_all_records_are_stance_records(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        a, b, tm, dist_a, dist_b = _fit_topics()
        extractor = StanceExtractor()
        records = extractor.extract(a, b, tm, dist_a, dist_b)
        assert all(isinstance(r, StanceRecord) for r in records)

    def test_no_llm_produces_ambiguous_positions(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        a, b, tm, dist_a, dist_b = _fit_topics()
        extractor = StanceExtractor()
        records = extractor.extract(a, b, tm, dist_a, dist_b)
        for r in records:
            assert r.position == "ambiguous"
            assert r.confidence == 0.0

    def test_speaker_names_correct(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        a, b, tm, dist_a, dist_b = _fit_topics()
        extractor = StanceExtractor()
        records = extractor.extract(a, b, tm, dist_a, dist_b)
        speakers = {r.speaker for r in records}
        assert speakers <= {"General", "Ecologist"}

    def test_topic_ids_in_range(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        a, b, tm, dist_a, dist_b = _fit_topics(n_topics=4)
        extractor = StanceExtractor()
        records = extractor.extract(a, b, tm, dist_a, dist_b)
        for r in records:
            assert 0 <= r.topic_id < 4

    def test_min_weight_filters_topics(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        a, b, tm, dist_a, dist_b = _fit_topics(n_topics=4)
        # Very high min_weight should produce fewer or zero records
        extractor_low = StanceExtractor(min_weight=0.0)
        extractor_high = StanceExtractor(min_weight=0.99)
        records_low = extractor_low.extract(a, b, tm, dist_a, dist_b)
        records_high = extractor_high.extract(a, b, tm, dist_a, dist_b)
        assert len(records_high) <= len(records_low)

    def test_evidence_quote_is_string(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        a, b, tm, dist_a, dist_b = _fit_topics()
        records = StanceExtractor().extract(a, b, tm, dist_a, dist_b)
        for r in records:
            assert isinstance(r.evidence_quote, str)

    def test_topic_label_is_nonempty_string(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        a, b, tm, dist_a, dist_b = _fit_topics()
        records = StanceExtractor().extract(a, b, tm, dist_a, dist_b)
        for r in records:
            assert isinstance(r.topic_label, str)
            assert len(r.topic_label) > 0


# ─────────────────────────────────────────────────────────────────────────────
# StanceExtractor — with injected mock LLM client
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractWithMockLLM:
    def test_uses_llm_when_client_injected(self):
        a, b, tm, dist_a, dist_b = _fit_topics()
        client = _mock_client(position="pro", confidence=0.9)
        extractor = StanceExtractor(_llm_client=client)
        records = extractor.extract(a, b, tm, dist_a, dist_b)
        assert client.chat.called

    def test_positions_come_from_llm(self):
        a, b, tm, dist_a, dist_b = _fit_topics()
        client = _mock_client(position="anti", confidence=0.7)
        extractor = StanceExtractor(_llm_client=client)
        records = extractor.extract(a, b, tm, dist_a, dist_b)
        for r in records:
            assert r.position == "anti"

    def test_confidence_comes_from_llm(self):
        a, b, tm, dist_a, dist_b = _fit_topics()
        client = _mock_client(position="neutral", confidence=0.65)
        extractor = StanceExtractor(_llm_client=client)
        records = extractor.extract(a, b, tm, dist_a, dist_b)
        for r in records:
            assert r.confidence == pytest.approx(0.65, abs=1e-3)

    def test_summary_comes_from_llm(self):
        summary = "The speaker strongly advocates for this."
        a, b, tm, dist_a, dist_b = _fit_topics()
        client = _mock_client(position="pro", summary=summary)
        extractor = StanceExtractor(_llm_client=client)
        records = extractor.extract(a, b, tm, dist_a, dist_b)
        for r in records:
            assert r.summary == summary

    def test_llm_error_falls_back_to_ambiguous(self):
        a, b, tm, dist_a, dist_b = _fit_topics()
        client = MagicMock()
        client.chat.side_effect = RuntimeError("connection refused")
        extractor = StanceExtractor(_llm_client=client)
        records = extractor.extract(a, b, tm, dist_a, dist_b)
        for r in records:
            assert r.position == "ambiguous"
            assert r.confidence == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _parse_response
# ─────────────────────────────────────────────────────────────────────────────

class TestParseResponse:
    def _extractor(self):
        return StanceExtractor()

    def test_clean_json_pro(self):
        raw = '{"position": "pro", "summary": "Speaker supports this clearly.", "confidence": 0.9}'
        pos, summary, conf = self._extractor()._parse_response(raw)
        assert pos == "pro"
        assert conf == pytest.approx(0.9, abs=1e-3)

    def test_clean_json_anti(self):
        raw = '{"position": "anti", "summary": "Speaker opposes this firmly.", "confidence": 0.75}'
        pos, summary, conf = self._extractor()._parse_response(raw)
        assert pos == "anti"

    def test_clean_json_neutral(self):
        raw = '{"position": "neutral", "summary": "No clear lean detected here.", "confidence": 0.5}'
        pos, _, _ = self._extractor()._parse_response(raw)
        assert pos == "neutral"

    def test_json_with_markdown_fence(self):
        raw = '```json\n{"position": "pro", "summary": "Supports strongly.", "confidence": 0.8}\n```'
        pos, _, _ = self._extractor()._parse_response(raw)
        assert pos == "pro"

    def test_invalid_position_falls_back_to_ambiguous(self):
        raw = '{"position": "maybe", "summary": "Unknown stance.", "confidence": 0.5}'
        pos, _, _ = self._extractor()._parse_response(raw)
        assert pos == "ambiguous"

    def test_malformed_json_uses_regex_fallback(self):
        raw = 'Sorry I cannot parse. "position": "anti" "summary": "They oppose it clearly really." "confidence": 0.6'
        pos, summary, conf = self._extractor()._parse_response(raw)
        assert pos == "anti"
        assert conf == pytest.approx(0.6, abs=1e-3)

    def test_confidence_clipped_above_one(self):
        raw = '{"position": "pro", "summary": "Too confident.", "confidence": 1.5}'
        _, _, conf = self._extractor()._parse_response(raw)
        assert conf == pytest.approx(1.0)

    def test_confidence_clipped_below_zero(self):
        raw = '{"position": "pro", "summary": "Negative confidence.", "confidence": -0.3}'
        _, _, conf = self._extractor()._parse_response(raw)
        assert conf == pytest.approx(0.0)

    def test_completely_empty_response(self):
        pos, summary, conf = self._extractor()._parse_response("")
        assert pos == "ambiguous"
        assert isinstance(summary, str)
        assert 0.0 <= conf <= 1.0

    def test_uppercase_position_normalised(self):
        raw = '{"position": "PRO", "summary": "This is a strong endorsement here.", "confidence": 0.8}'
        pos, _, _ = self._extractor()._parse_response(raw)
        assert pos == "pro"

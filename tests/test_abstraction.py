"""Tests for vox_romana.analysis.abstraction (shot 5.3-a).

Covers sentence splitting, the LLM tagger (stubbed), parse/fallback behavior,
and graceful degradation without an API key.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

from vox_romana.analysis.abstraction import (
    MOVE_LEVELS,
    AbstractionMove,
    AbstractionTagger,
    split_sentences,
)
from vox_romana.analysis.models import SpeakerCorpus


def _corpus() -> SpeakerCorpus:
    return SpeakerCorpus(
        name="Candidate A",
        utterances=[
            "We must lower taxes. My family grew up poor.",
            "The policy costs 4 billion dollars a year.",
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Sentence splitting
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitSentences:
    def test_splits_on_period(self):
        assert split_sentences("One. Two.") == ["One.", "Two."]

    def test_splits_on_question_and_bang(self):
        assert split_sentences("Really? No!") == ["Really?", "No!"]

    def test_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_no_trailing_period_single(self):
        assert split_sentences("A sentence") == ["A sentence"]


# ═══════════════════════════════════════════════════════════════════════════
# Tagger — LLM path
# ═══════════════════════════════════════════════════════════════════════════

class TestAbstractionTagger:
    def test_tags_all_sentences_with_llm(self):
        client = Mock()
        client.chat.return_value = json.dumps({
            "0": {"level": "abstract", "confidence": 0.9},
            "1": {"level": "anecdote", "confidence": 0.8},
            "2": {"level": "concrete", "confidence": 0.7},
            "3": {"level": "mid", "confidence": 0.6},
        })
        tagger = AbstractionTagger(_llm_client=client)
        moves = tagger.tag_corpus(_corpus())

        # "We must lower taxes." / "My family grew up poor." /
        # "The policy costs 4 billion dollars a year." → 3 sentences total
        assert len(moves) == 3
        assert all(isinstance(m, AbstractionMove) for m in moves)
        assert [m.level for m in moves] == ["abstract", "anecdote", "concrete"]
        assert moves[0].speaker == "Candidate A"

    def test_batches_sentences_across_calls(self):
        client = Mock()
        client.chat.side_effect = [
            json.dumps({"0": {"level": "abstract", "confidence": 0.9}}),
            json.dumps({"1": {"level": "concrete", "confidence": 0.8}}),
        ]
        tagger = AbstractionTagger(_llm_client=client, batch_size=1)
        corpus = SpeakerCorpus(name="X", utterances=["One sentence.", "Two sentence."])
        moves = tagger.tag_corpus(corpus)
        assert client.chat.call_count == 2
        assert [m.level for m in moves] == ["abstract", "concrete"]

    def test_parse_fallback_handles_malformed(self):
        tagger = AbstractionTagger(_llm_client=Mock())
        result = tagger._parse_response("not json", {0, 1, 2})
        assert result == {0: ("mid", 0.0), 1: ("mid", 0.0), 2: ("mid", 0.0)}

    def test_parse_validates_level(self):
        tagger = AbstractionTagger(_llm_client=Mock())
        raw = json.dumps({
            "0": {"level": "gobbledygook", "confidence": 0.9},
            "1": {"level": "concrete", "confidence": 0.8},
        })
        result = tagger._parse_response(raw, {0, 1, 2})
        assert result[0] == ("mid", 0.9)  # invalid level → mid, keep confidence
        assert result[1] == ("concrete", 0.8)
        assert result[2] == ("mid", 0.0)  # unanswered index → fallback

    def test_parse_ignores_out_of_range_indices(self):
        tagger = AbstractionTagger(_llm_client=Mock())
        raw = json.dumps({"99": {"level": "abstract", "confidence": 0.9}})
        result = tagger._parse_response(raw, {0})
        assert result == {0: ("mid", 0.0)}

    # ── Degradation without LLM ───────────────────────────────────────────

    def test_degrades_to_mid_without_api_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        tagger = AbstractionTagger()
        moves = tagger.tag_corpus(_corpus())
        assert len(moves) == 3
        assert all(m.level == "mid" for m in moves)
        assert all(m.confidence == 0.0 for m in moves)

    def test_empty_corpus_returns_empty(self):
        tagger = AbstractionTagger(_llm_client=Mock())
        corpus = SpeakerCorpus(name="Empty", utterances=["", "   "])
        assert tagger.tag_corpus(corpus) == []


class TestMoveLevels:
    def test_levels_constant(self):
        assert MOVE_LEVELS == ("abstract", "mid", "concrete", "anecdote")

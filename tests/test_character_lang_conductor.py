"""Tests for Chunk 2 per-character language — LLM injection layer.

Covers:
  - _build_language_instruction() helper in conductor.py
  - {language_instruction} placeholder in PARTICIPANT_SYSTEM_PROMPT
  - _build_participant_prompt() injects the right instruction
  - character_langs collected in to_merged_debate()
  - character_langs serialized in MergedDebate.to_dict()
"""

from __future__ import annotations

import pytest

from vox_romana import (
    DebateConfig,
    MergedDebate,
    ParticipantProfile,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_config(participants):
    from vox_romana.shows.debates.config import DebateConfig
    from unittest.mock import Mock
    cfg = Mock(spec=DebateConfig)
    cfg.participants = participants
    cfg.debate_format = "debate"
    cfg.topic = "Test topic"
    cfg.brief = "Test brief"
    cfg.target_turns = 10
    cfg.tone = "neutral"
    cfg.moderator = None
    cfg.show_host = None
    cfg.audience = None
    cfg.setting = None
    cfg.title = "Test Debate"
    cfg.language = "en"
    return cfg


def _make_participant(name, language=None):
    return ParticipantProfile(
        name=name,
        side="pro",
        language=language,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _build_language_instruction
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildLanguageInstruction:
    def _call(self, language):
        from vox_romana.shows.debates.conductor import _build_language_instruction
        return _build_language_instruction(language)

    def test_none_returns_empty(self):
        assert self._call(None) == ""

    def test_en_returns_empty(self):
        assert self._call("en") == ""

    def test_zh_contains_chinese(self):
        result = self._call("zh")
        assert "Chinese" in result
        assert "zh" in result

    def test_es_contains_spanish(self):
        result = self._call("es")
        assert "Spanish" in result
        assert "es" in result

    def test_fr_contains_french(self):
        result = self._call("fr")
        assert "French" in result

    def test_contains_text_field_mention(self):
        # Instruction must tell the LLM to write the "text" field in target lang.
        result = self._call("zh")
        assert '"text"' in result or "text" in result.lower()

    def test_json_keys_stay_english_noted(self):
        result = self._call("zh")
        assert "English" in result

    def test_unknown_code_uses_code_as_fallback_name(self):
        result = self._call("xx")
        assert "xx" in result

    def test_it_returns_italian(self):
        result = self._call("it")
        assert "Italian" in result


# ─────────────────────────────────────────────────────────────────────────────
# PARTICIPANT_SYSTEM_PROMPT has {language_instruction} placeholder
# ─────────────────────────────────────────────────────────────────────────────


class TestParticipantSystemPromptPlaceholder:
    def test_language_instruction_placeholder_present(self):
        from vox_romana.shows.debates.prompts import PARTICIPANT_SYSTEM_PROMPT
        assert "{language_instruction}" in PARTICIPANT_SYSTEM_PROMPT

    def test_prompt_formats_without_error_when_empty(self):
        from vox_romana.shows.debates.prompts import PARTICIPANT_SYSTEM_PROMPT
        # Should not raise even with an empty string for language_instruction
        result = PARTICIPANT_SYSTEM_PROMPT.format(
            name="Alice",
            role="participant",
            debate_format="debate",
            topic="Test",
            side="pro",
            ground_rules="You are debating Bob.",
            voice_fingerprint="Assertive.",
            behavioral_triggers="None.",
            emotional_arc="Steady.",
            key_positions="  1. Point A",
            factual_claims="  (none)",
            tangent_topics="  (none)",
            reference_corpus_section="",
            language_instruction="",
        )
        assert "Alice" in result

    def test_prompt_formats_with_language_instruction(self):
        from vox_romana.shows.debates.conductor import _build_language_instruction
        from vox_romana.shows.debates.prompts import PARTICIPANT_SYSTEM_PROMPT
        instr = _build_language_instruction("zh")
        result = PARTICIPANT_SYSTEM_PROMPT.format(
            name="Marcus",
            role="participant",
            debate_format="debate",
            topic="Test",
            side="pro",
            ground_rules="You are debating Bob.",
            voice_fingerprint="Authoritative.",
            behavioral_triggers="None.",
            emotional_arc="Steady.",
            key_positions="  1. Point A",
            factual_claims="  (none)",
            tangent_topics="  (none)",
            reference_corpus_section="",
            language_instruction=instr,
        )
        assert "Chinese" in result


# ─────────────────────────────────────────────────────────────────────────────
# _build_participant_prompt — language injection
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildParticipantPromptLanguage:
    def _make_conductor(self, participants):
        from unittest.mock import Mock, MagicMock
        from vox_romana.shows.debates.conductor import DebateConductor
        from vox_romana.shows.debates.config import DebateConfig

        cfg = Mock(spec=DebateConfig)
        cfg.participants = participants
        cfg.debate_format = "debate"
        cfg.topic = "Test topic"
        cfg.brief = "Test brief"
        cfg.target_turns = 10
        cfg.tone = "neutral"
        cfg.moderator = None
        cfg.show_host = None
        cfg.audience = None
        cfg.setting = None
        cfg.title = "Test"
        cfg.language = "en"

        conductor = object.__new__(DebateConductor)
        conductor.config = cfg
        return conductor

    def test_no_language_constraint_in_prompt_for_none(self):
        p = _make_participant("Alice", language=None)
        conductor = self._make_conductor([p])
        prompt = conductor._build_participant_prompt(p)
        assert "LANGUAGE CONSTRAINT" not in prompt

    def test_no_language_constraint_in_prompt_for_en(self):
        p = _make_participant("Alice", language="en")
        conductor = self._make_conductor([p])
        prompt = conductor._build_participant_prompt(p)
        assert "LANGUAGE CONSTRAINT" not in prompt

    def test_zh_injects_language_constraint(self):
        p = _make_participant("Marcus", language="zh")
        conductor = self._make_conductor([p])
        prompt = conductor._build_participant_prompt(p)
        assert "LANGUAGE CONSTRAINT" in prompt
        assert "Chinese" in prompt

    def test_es_injects_language_constraint(self):
        p = _make_participant("Sofia", language="es")
        conductor = self._make_conductor([p])
        prompt = conductor._build_participant_prompt(p)
        assert "Spanish" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# MergedDebate.character_langs field
# ─────────────────────────────────────────────────────────────────────────────


class TestMergedDebateCharacterLangs:
    def _make_merged(self, character_langs=None):
        from vox_romana.shows.debates.models import MergedDebate, DebateSegment
        seg = DebateSegment(
            template_id="conductor",
            debate_id="d1",
            title="Test",
            chunk_order=1,
            is_final_chunk=True,
            participants=(),
        )
        return MergedDebate(
            template_id="conductor",
            debate_id="d1",
            title="Test",
            chunks=(seg,),
            events=(),
            character_langs=character_langs or {},
        )

    def test_default_is_empty_dict(self):
        merged = self._make_merged()
        assert merged.character_langs == {}

    def test_character_langs_stored(self):
        merged = self._make_merged({"Marcus": "zh"})
        assert merged.character_langs == {"Marcus": "zh"}

    def test_to_dict_includes_character_langs(self):
        merged = self._make_merged({"Marcus": "zh", "Giulia": "it"})
        d = merged.to_dict()
        assert "character_langs" in d
        assert d["character_langs"] == {"Marcus": "zh", "Giulia": "it"}

    def test_to_dict_empty_character_langs(self):
        merged = self._make_merged()
        d = merged.to_dict()
        assert d["character_langs"] == {}


# ─────────────────────────────────────────────────────────────────────────────
# to_merged_debate() builds character_langs from profile.language
# ─────────────────────────────────────────────────────────────────────────────


class TestToMergedDebateCharacterLangs:
    def _make_result(self, participants):
        from vox_romana.shows.debates.conductor import DebateConductorResult
        from unittest.mock import Mock
        from vox_romana.shows.debates.config import DebateConfig

        cfg = Mock(spec=DebateConfig)
        cfg.participants = participants
        cfg.debate_format = "debate"
        cfg.topic = "Test"
        cfg.brief = "Test"
        cfg.target_turns = 4
        cfg.tone = "neutral"
        cfg.moderator = None
        cfg.show_host = None
        cfg.audience = None
        cfg.setting = None
        cfg.title = "Test Debate"
        cfg.language = "en"
        cfg.continuity_notes = []

        result = DebateConductorResult(config=cfg)
        result.debate_events = []
        result.debate_id = "test-id"
        return result

    def test_no_languages_gives_empty_character_langs(self):
        p1 = _make_participant("Alice")
        p2 = _make_participant("Bob")
        result = self._make_result([p1, p2])
        merged = result.to_merged_debate()
        assert merged.character_langs == {}

    def test_one_language_included(self):
        p1 = _make_participant("Marcus", language="zh")
        p2 = _make_participant("Bob")
        result = self._make_result([p1, p2])
        merged = result.to_merged_debate()
        assert merged.character_langs == {"Marcus": "zh"}

    def test_multiple_languages_included(self):
        p1 = _make_participant("Marcus", language="zh")
        p2 = _make_participant("Giulia", language="it")
        result = self._make_result([p1, p2])
        merged = result.to_merged_debate()
        assert merged.character_langs == {"Marcus": "zh", "Giulia": "it"}

    def test_en_language_excluded(self):
        # profile.language="en" should not appear in character_langs
        # (the whole show is en; it's noise)
        p1 = _make_participant("Alice", language="en")
        result = self._make_result([p1])
        merged = result.to_merged_debate()
        assert "Alice" not in merged.character_langs

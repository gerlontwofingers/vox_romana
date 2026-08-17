"""Tests for Chunk 1 per-character language wiring:

  - parse_character_langs() in llm_config.py
  - ParticipantProfile.language field
  - --character-lang flag in run.py parse_args()
  - _apply_character_langs() in runners.py
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# parse_character_langs
# ─────────────────────────────────────────────────────────────────────────────


class TestParseCharacterLangs:
    def _parse(self, specs):
        from vox_romana.core.llm_config import parse_character_langs
        return parse_character_langs(specs)

    def test_none_returns_empty_dict(self):
        assert self._parse(None) == {}

    def test_empty_list_returns_empty_dict(self):
        assert self._parse([]) == {}

    def test_single_entry(self):
        result = self._parse(["Marcus=zh"])
        assert result == {"Marcus": "zh"}

    def test_multiple_entries(self):
        result = self._parse(["Marcus=zh", "Giulia=it", "Sofia=es"])
        assert result == {"Marcus": "zh", "Giulia": "it", "Sofia": "es"}

    def test_code_lowercased(self):
        result = self._parse(["Hero=ZH"])
        assert result["Hero"] == "zh"

    def test_name_with_spaces(self):
        result = self._parse(["Dr. Chen=zh"])
        assert result == {"Dr. Chen": "zh"}

    def test_whitespace_stripped(self):
        result = self._parse([" Marcus = zh "])
        assert result == {"Marcus": "zh"}

    def test_missing_equals_raises(self):
        from vox_romana.core.llm_config import parse_character_langs
        with pytest.raises(ValueError, match="expected 'NAME=CODE'"):
            parse_character_langs(["MarcusZH"])

    def test_empty_name_raises(self):
        from vox_romana.core.llm_config import parse_character_langs
        with pytest.raises(ValueError, match="empty name or language code"):
            parse_character_langs(["=zh"])

    def test_empty_code_raises(self):
        from vox_romana.core.llm_config import parse_character_langs
        with pytest.raises(ValueError, match="empty name or language code"):
            parse_character_langs(["Marcus="])

    def test_last_writer_wins_for_duplicate_names(self):
        result = self._parse(["Marcus=zh", "Marcus=es"])
        assert result["Marcus"] == "es"

    def test_en_code_accepted(self):
        result = self._parse(["Host=en"])
        assert result == {"Host": "en"}


# ─────────────────────────────────────────────────────────────────────────────
# ParticipantProfile.language field
# ─────────────────────────────────────────────────────────────────────────────


class TestParticipantProfileLanguage:
    def test_language_field_defaults_to_none(self):
        from vox_romana.core.config import ParticipantProfile
        p = ParticipantProfile(name="Test")
        assert p.language is None

    def test_language_field_can_be_set(self):
        from vox_romana.core.config import ParticipantProfile
        p = ParticipantProfile(name="Marcus", language="zh")
        assert p.language == "zh"

    def test_language_field_mutable(self):
        from vox_romana.core.config import ParticipantProfile
        p = ParticipantProfile(name="Marcus")
        p.language = "es"
        assert p.language == "es"

    def test_language_field_not_passed_to_voice_profile_parent(self):
        # VoiceProfile base should NOT have language — it lives on the subclass.
        from vox_romana.core.config import VoiceProfile
        assert not hasattr(VoiceProfile(), "language")


# ─────────────────────────────────────────────────────────────────────────────
# run.py parse_args — --character-lang flag
# ─────────────────────────────────────────────────────────────────────────────


class TestParseArgsCharacterLang:
    def test_character_lang_default_is_none(self):
        from run import parse_args
        args = parse_args(["--mode", "voice-prep", "--vp-source", "r.wav", "--vp-name", "v"])
        assert args.character_lang is None

    def test_single_character_lang(self):
        from run import parse_args
        args = parse_args([
            "--mode", "voice-prep", "--vp-source", "r.wav", "--vp-name", "v",
            "--character-lang", "Marcus=zh",
        ])
        assert args.character_lang == ["Marcus=zh"]

    def test_multiple_character_lang_repeatable(self):
        from run import parse_args
        args = parse_args([
            "--mode", "voice-prep", "--vp-source", "r.wav", "--vp-name", "v",
            "--character-lang", "Marcus=zh",
            "--character-lang", "Giulia=it",
        ])
        assert args.character_lang == ["Marcus=zh", "Giulia=it"]

    def test_character_lang_stored_on_dest(self):
        from run import parse_args
        args = parse_args([
            "--mode", "voice-prep", "--vp-source", "r.wav", "--vp-name", "v",
            "--character-lang", "A=fr",
        ])
        assert hasattr(args, "character_lang")
        assert args.character_lang[0] == "A=fr"


# ─────────────────────────────────────────────────────────────────────────────
# _apply_character_langs
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyCharacterLangs:
    def _make_profile(self, name, language=None):
        from vox_romana.core.config import ParticipantProfile
        p = ParticipantProfile(name=name)
        p.language = language
        return p

    def _make_args(self, character_langs: dict) -> types.SimpleNamespace:
        return types.SimpleNamespace(character_langs=character_langs)

    def test_no_overrides_leaves_language_unchanged(self):
        from vox_romana.cli.runners import _apply_character_langs
        p = self._make_profile("Marcus")
        _apply_character_langs([p], self._make_args({}))
        assert p.language is None

    def test_matching_name_sets_language(self):
        from vox_romana.cli.runners import _apply_character_langs
        p = self._make_profile("Marcus")
        _apply_character_langs([p], self._make_args({"Marcus": "zh"}))
        assert p.language == "zh"

    def test_non_matching_name_unaffected(self):
        from vox_romana.cli.runners import _apply_character_langs
        p = self._make_profile("Giulia")
        _apply_character_langs([p], self._make_args({"Marcus": "zh"}))
        assert p.language is None

    def test_multiple_participants_each_matched(self):
        from vox_romana.cli.runners import _apply_character_langs
        marcus = self._make_profile("Marcus")
        giulia = self._make_profile("Giulia")
        _apply_character_langs([marcus, giulia], self._make_args({"Marcus": "zh", "Giulia": "it"}))
        assert marcus.language == "zh"
        assert giulia.language == "it"

    def test_partial_match_leaves_unmatched_none(self):
        from vox_romana.cli.runners import _apply_character_langs
        marcus = self._make_profile("Marcus")
        sofia = self._make_profile("Sofia")
        _apply_character_langs([marcus, sofia], self._make_args({"Marcus": "zh"}))
        assert marcus.language == "zh"
        assert sofia.language is None

    def test_no_character_langs_attr_is_safe(self):
        from vox_romana.cli.runners import _apply_character_langs
        p = self._make_profile("Marcus")
        # args with no character_langs attribute at all
        _apply_character_langs([p], types.SimpleNamespace())
        assert p.language is None

    def test_empty_participants_list_is_safe(self):
        from vox_romana.cli.runners import _apply_character_langs
        _apply_character_langs([], self._make_args({"Marcus": "zh"}))  # no error

    def test_overrides_existing_language(self):
        from vox_romana.cli.runners import _apply_character_langs
        p = self._make_profile("Marcus", language="fr")
        _apply_character_langs([p], self._make_args({"Marcus": "zh"}))
        assert p.language == "zh"

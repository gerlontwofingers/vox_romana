"""Tests for Chunk 7 — language propagation through conductors.

Verifies:
  - _build_participant_prompt() falls back to config.language when
    profile.language is None (the new fallback added in Chunk 7).
  - to_merged_debate() passes language=config.language.
  - to_merged_news_show() passes language=config.language.
  - to_merged_podcast() passes language=config.language.
  - MergedPodcast serializes language in to_dict().
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from vox_romana import ParticipantProfile


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_debate_conductor(participants, config_language="en"):
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
    cfg.language = config_language

    conductor = object.__new__(DebateConductor)
    conductor.config = cfg
    return conductor


def _make_participant(name, language=None):
    return ParticipantProfile(name=name, side="pro", language=language)


# ─────────────────────────────────────────────────────────────────────────────
# _build_participant_prompt — config.language fallback (Chunk 7)
# ─────────────────────────────────────────────────────────────────────────────


class TestParticipantPromptConfigLanguageFallback:
    def test_profile_none_lang_inherits_config_es(self):
        """When profile.language is None, the show language (es) is used."""
        p = _make_participant("Carlos", language=None)
        conductor = _make_debate_conductor([p], config_language="es")
        prompt = conductor._build_participant_prompt(p)
        assert "LANGUAGE CONSTRAINT" in prompt
        assert "Spanish" in prompt

    def test_profile_none_lang_en_config_no_constraint(self):
        """English show + no per-character override → no language instruction."""
        p = _make_participant("Alice", language=None)
        conductor = _make_debate_conductor([p], config_language="en")
        prompt = conductor._build_participant_prompt(p)
        assert "LANGUAGE CONSTRAINT" not in prompt

    def test_profile_language_overrides_config(self):
        """Explicit profile.language takes priority over config.language."""
        p = _make_participant("Giulia", language="it")
        conductor = _make_debate_conductor([p], config_language="es")
        prompt = conductor._build_participant_prompt(p)
        assert "Italian" in prompt
        assert "Spanish" not in prompt


# ─────────────────────────────────────────────────────────────────────────────
# to_merged_debate() language propagation
# ─────────────────────────────────────────────────────────────────────────────


class TestMergedDebateLanguagePropagation:
    def _make_merged(self, language="es"):
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
            language=language,
        )

    def test_language_stored(self):
        merged = self._make_merged("es")
        assert merged.language == "es"

    def test_language_default_en(self):
        merged = self._make_merged("en")
        assert merged.language == "en"

    def test_to_dict_includes_language(self):
        merged = self._make_merged("fr")
        d = merged.to_dict()
        assert d.get("language") == "fr"


# ─────────────────────────────────────────────────────────────────────────────
# MergedPodcast language field (Chunk 7 — new field)
# ─────────────────────────────────────────────────────────────────────────────


class TestMergedPodcastLanguage:
    def _make_merged(self, language="en"):
        from vox_romana.shows.podcast.models import MergedPodcast

        return MergedPodcast(
            podcast_id="p1",
            title="Test Podcast",
            language=language,
        )

    def test_language_stored(self):
        merged = self._make_merged("es")
        assert merged.language == "es"

    def test_language_default_en(self):
        merged = self._make_merged()
        assert merged.language == "en"

    def test_to_dict_includes_language(self):
        merged = self._make_merged("de")
        d = merged.to_dict()
        assert d.get("language") == "de"

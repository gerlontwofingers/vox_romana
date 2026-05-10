"""Tests for Chunk 6 — bootstrap language inference.

Verifies that the ``"language"`` field added to the Architect JSON schema
for debates, news, and podcast shows is:
  - parsed and stored on the resulting config object.
  - defaulted to ``"en"`` when absent or null.
  - stored as a lowercase stripped string.
"""

from __future__ import annotations

import json


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _debate_canned(language=None) -> str:
    data = {
        "title": "Test Debate",
        "topic": "Water rights",
        "debate_format": "panel",
        "setting": "community hall",
        "tone": "heated",
        "target_turns": 20,
        "participants": [{"name": "Alice", "side": "pro", "role": "participant"}],
        "moderator": None,
        "audience": None,
        "show_host": None,
    }
    if language is not None:
        data["language"] = language
    return json.dumps(data)


def _news_canned(language=None) -> str:
    data = {
        "title": "Morning Brief",
        "topic": "Tech news",
        "tone": "authoritative",
        "setting": "studio",
        "target_segments": 3,
        "anchors": [{"name": "Jane", "style": "crisp"}],
        "segments": [],
        "show_host": {"name": "Host", "role": "narrator", "style": "warm"},
    }
    if language is not None:
        data["language"] = language
    return json.dumps(data)


def _podcast_canned(language=None) -> str:
    data = {
        "title": "El Podcast",
        "show_name": "Hablemos",
        "topic": "Cultura",
        "tone": "playful",
        "hosts": [
            {
                "name": "Rodrigo",
                "gender": "male",
                "role": "lead host",
                "speech_style": "energetic",
                "knowledge_domains": ["culture"],
                "forbidden_knowledge": [],
                "interruption_tendency": "medium",
            }
        ],
        "segments": [],
    }
    if language is not None:
        data["language"] = language
    return json.dumps(data)


# ─────────────────────────────────────────────────────────────────────────────
# Debate architect
# ─────────────────────────────────────────────────────────────────────────────


class TestDebateArchitectLanguage:
    def _parse(self, raw_json: str):
        from vox_romana.shows.debates.architect import DebateArchitect
        arch = DebateArchitect.__new__(DebateArchitect)
        return arch.parse_response(raw_json)

    def test_language_es_parsed(self):
        cfg = self._parse(_debate_canned("es"))
        assert cfg.language == "es"

    def test_language_fr_parsed(self):
        cfg = self._parse(_debate_canned("fr"))
        assert cfg.language == "fr"

    def test_language_absent_defaults_to_en(self):
        cfg = self._parse(_debate_canned())
        assert cfg.language == "en"

    def test_language_null_defaults_to_en(self):
        data = json.loads(_debate_canned())
        data["language"] = None
        cfg = self._parse(json.dumps(data))
        assert cfg.language == "en"

    def test_language_uppercased_lowercased(self):
        cfg = self._parse(_debate_canned("ES"))
        assert cfg.language == "es"

    def test_language_with_whitespace_stripped(self):
        cfg = self._parse(_debate_canned("  de  "))
        assert cfg.language == "de"


# ─────────────────────────────────────────────────────────────────────────────
# News bootstrap
# ─────────────────────────────────────────────────────────────────────────────


class TestNewsBootstrapLanguage:
    def _parse(self, raw_json: str):
        from vox_romana.shows.news.bootstrap import NewsBootstrap
        nb = NewsBootstrap.__new__(NewsBootstrap)
        return nb._parse_architect_response(raw_json)

    def test_language_es_parsed(self):
        cfg = self._parse(_news_canned("es"))
        assert cfg.language == "es"

    def test_language_absent_defaults_to_en(self):
        cfg = self._parse(_news_canned())
        assert cfg.language == "en"

    def test_language_null_defaults_to_en(self):
        data = json.loads(_news_canned())
        data["language"] = None
        cfg = self._parse(json.dumps(data))
        assert cfg.language == "en"

    def test_language_uppercased_lowercased(self):
        cfg = self._parse(_news_canned("FR"))
        assert cfg.language == "fr"


# ─────────────────────────────────────────────────────────────────────────────
# Podcast bootstrap
# ─────────────────────────────────────────────────────────────────────────────


class TestPodcastBootstrapLanguage:
    def _parse(self, raw_json: str):
        from vox_romana.shows.podcast.bootstrap import PodcastBootstrap
        pb = PodcastBootstrap.__new__(PodcastBootstrap)
        return pb._parse_architect_response(raw_json)

    def test_language_es_parsed(self):
        cfg = self._parse(_podcast_canned("es"))
        assert cfg.language == "es"

    def test_language_absent_defaults_to_en(self):
        cfg = self._parse(_podcast_canned())
        assert cfg.language == "en"

    def test_language_null_defaults_to_en(self):
        data = json.loads(_podcast_canned())
        data["language"] = None
        cfg = self._parse(json.dumps(data))
        assert cfg.language == "en"

    def test_language_uppercased_lowercased(self):
        cfg = self._parse(_podcast_canned("IT"))
        assert cfg.language == "it"

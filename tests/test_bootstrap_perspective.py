"""Tests for Chunk 4 — bootstrap architect perspective inference.

Verifies that the ``"perspective"`` field added to the Architect JSON schema
for news and audiobook shows is:
  - parsed and stored on the resulting config object,
  - defaulted to ``None`` when absent / null / empty string,
  - threaded through the prompt so the architect is told to infer it.
"""

from __future__ import annotations

import json


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _news_canned(perspective=...) -> str:
    data = {
        "title": "Noticias del Norte",
        "topic": "Frontera",
        "tone": "serious",
        "setting": "studio",
        "target_segments": 2,
        "anchors": [{"name": "Lucia Morales", "style": "warm"}],
        "segments": [],
        "show_host": {"name": "Host", "role": "narrator", "style": "warm"},
        "language": "es",
        "music_style": "latin news theme",
    }
    if perspective is not ...:
        data["perspective"] = perspective
    return json.dumps(data)


def _audiobook_canned(perspective=...) -> str:
    data = {
        "title": "Tales of the South",
        "genre": "history",
        "topic": "Migration",
        "tone": "reflective",
        "setting": "Central America",
        "narrator": {"name": "Aram Solano", "style": "warm"},
        "characters": [],
        "chapters": [],
        "language": "es",
        "music_style": "atmospheric acoustic",
    }
    if perspective is not ...:
        data["perspective"] = perspective
    return json.dumps(data)


# ─────────────────────────────────────────────────────────────────────────────
# News bootstrap perspective
# ─────────────────────────────────────────────────────────────────────────────


class TestNewsBootstrapPerspective:
    def _parse(self, raw_json: str):
        from vox_romana.shows.news.bootstrap import NewsBootstrap
        nb = NewsBootstrap.__new__(NewsBootstrap)
        return nb._parse_architect_response(raw_json)

    def test_perspective_string_parsed(self):
        cfg = self._parse(_news_canned("from Guatemala"))
        assert cfg.perspective == "from Guatemala"

    def test_perspective_absent_is_none(self):
        cfg = self._parse(_news_canned())
        assert cfg.perspective is None

    def test_perspective_null_is_none(self):
        cfg = self._parse(_news_canned(None))
        assert cfg.perspective is None

    def test_perspective_empty_string_is_none(self):
        cfg = self._parse(_news_canned(""))
        assert cfg.perspective is None

    def test_architect_prompt_documents_perspective(self):
        from vox_romana.shows.news.bootstrap import NEWS_ARCHITECT_PROMPT
        prompt_lower = NEWS_ARCHITECT_PROMPT.lower()
        assert "perspective" in prompt_lower
        assert "observer-relative" in prompt_lower or "vantage" in prompt_lower
        # Guides the LLM with a concrete geographic example so it doesn't
        # default to None on briefs with obvious location framing.
        assert "guatemala" in prompt_lower


# ─────────────────────────────────────────────────────────────────────────────
# Audiobook bootstrap perspective
# ─────────────────────────────────────────────────────────────────────────────


class TestAudiobookBootstrapPerspective:
    def _parse(self, raw_json: str):
        from vox_romana.shows.audiobooks.bootstrap import AudiobookBootstrap
        ab = AudiobookBootstrap.__new__(AudiobookBootstrap)
        return ab._parse_architect_response(raw_json)

    def test_perspective_string_parsed(self):
        cfg = self._parse(_audiobook_canned("from Guatemala"))
        assert cfg.perspective == "from Guatemala"

    def test_perspective_absent_is_none(self):
        cfg = self._parse(_audiobook_canned())
        assert cfg.perspective is None

    def test_perspective_null_is_none(self):
        cfg = self._parse(_audiobook_canned(None))
        assert cfg.perspective is None

    def test_perspective_empty_string_is_none(self):
        cfg = self._parse(_audiobook_canned(""))
        assert cfg.perspective is None

    def test_architect_prompt_documents_perspective(self):
        from vox_romana.shows.audiobooks.bootstrap import AUDIOBOOK_ARCHITECT_PROMPT
        prompt_lower = AUDIOBOOK_ARCHITECT_PROMPT.lower()
        assert "perspective" in prompt_lower
        assert "observer-relative" in prompt_lower or "vantage" in prompt_lower
        assert "guatemala" in prompt_lower

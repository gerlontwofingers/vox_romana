"""Tests for ContentModeClassifier — auto-classification of content mode from brief.

Tests cover:
  TestDefaultsToFactual  — news/debate default to "factual" via LLM; LLM is still called
  TestPodcastClassifier  — podcast/audiobook asks the LLM + parses response
  TestParseMode          — _parse_mode() edge cases (valid, invalid, partial)
  TestFallback           — any error defaults to "factual"

No real API calls are made. All LLM calls are mocked.

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_content_mode.py -v
"""

from __future__ import annotations

import logging
from unittest.mock import Mock, MagicMock

import pytest

from vox_romana.core.content_mode import ContentModeClassifier


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_client(response: str) -> Mock:
    client = Mock()
    client.chat = Mock(return_value=response)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# TEST: News/debate default to factual (LLM still called)
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultsToFactual:
    """News and debate default to 'factual' — the LLM is still consulted."""

    def test_news_always_factual(self):
        client = _make_client("factual")
        result = ContentModeClassifier(client).classify(
            brief="Breaking: inflation rises to 4%",
            show_type="news",
        )
        assert result == "factual"
        client.chat.assert_called_once()

    def test_debate_always_factual(self):
        client = _make_client("factual")
        result = ContentModeClassifier(client).classify(
            brief="A debate on universal basic income",
            show_type="debate",
        )
        assert result == "factual"
        client.chat.assert_called_once()

    def test_news_ignores_fiction_keywords_in_brief(self):
        """When LLM returns factual, result is factual even with fiction-sounding brief."""
        client = _make_client("factual")
        result = ContentModeClassifier(client).classify(
            brief="Star Wars-themed marketing news report",
            show_type="news",
        )
        assert result == "factual"
        client.chat.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# TEST: Podcast / audiobook — LLM call + parse
# ─────────────────────────────────────────────────────────────────────────────

class TestPodcastClassifier:
    """Podcast and audiobook ask the LLM and parse its response."""

    @pytest.mark.parametrize("show_type", ["podcast", "audiobook"])
    def test_returns_factual_from_llm(self, show_type):
        client = _make_client("factual")
        result = ContentModeClassifier(client).classify(
            brief="A podcast about the history of space exploration",
            show_type=show_type,
        )
        assert result == "factual"
        client.chat.assert_called_once()

    @pytest.mark.parametrize("show_type", ["podcast", "audiobook"])
    def test_returns_fictional_universe_from_llm(self, show_type):
        client = _make_client("fictional-universe")
        result = ContentModeClassifier(client).classify(
            brief="An audiobook chapter about Luke Skywalker's training on Tatooine",
            show_type=show_type,
        )
        assert result == "fictional-universe"

    @pytest.mark.parametrize("show_type", ["podcast", "audiobook"])
    def test_returns_pure_fiction_from_llm(self, show_type):
        client = _make_client("pure-fiction")
        result = ContentModeClassifier(client).classify(
            brief="An original short story about dragons in a made-up world",
            show_type=show_type,
        )
        assert result == "pure-fiction"

    def test_llm_called_with_brief_in_message(self):
        """The brief is included in the LLM user message."""
        client = _make_client("factual")
        brief = "A UNIQUE BRIEF STRING abc123"
        ContentModeClassifier(client).classify(brief=brief, show_type="podcast")
        call_args = client.chat.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        combined = " ".join(m["content"] for m in messages)
        assert brief in combined


# ─────────────────────────────────────────────────────────────────────────────
# TEST: _parse_mode edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestParseMode:
    """_parse_mode() handles all reasonable LLM response shapes."""

    def _classify_with_response(self, response: str) -> str:
        """Helper: run classify on a podcast with a mocked LLM returning response."""
        client = _make_client(response)
        return ContentModeClassifier(client).classify(
            brief="some brief", show_type="podcast"
        )

    def test_exact_factual(self):
        assert self._classify_with_response("factual") == "factual"

    def test_exact_fictional_universe(self):
        assert self._classify_with_response("fictional-universe") == "fictional-universe"

    def test_exact_pure_fiction(self):
        assert self._classify_with_response("pure-fiction") == "pure-fiction"

    def test_extra_whitespace_stripped(self):
        assert self._classify_with_response("  factual  ") == "factual"

    def test_uppercase_handled(self):
        assert self._classify_with_response("FACTUAL") == "factual"

    def test_response_with_prose_containing_factual(self):
        """LLM adds explanation — we still extract the mode."""
        assert self._classify_with_response(
            "Based on the brief, I classify this as factual."
        ) == "factual"

    def test_response_with_prose_containing_fictional_universe(self):
        assert self._classify_with_response(
            "This is clearly fictional-universe content."
        ) == "fictional-universe"

    def test_unknown_word_falls_back_to_factual(self):
        assert self._classify_with_response("comedy") == "factual"

    def test_empty_response_falls_back_to_factual(self):
        assert self._classify_with_response("") == "factual"


# ─────────────────────────────────────────────────────────────────────────────
# TEST: Error fallback behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestFallback:
    """Any LLM failure defaults to 'factual' — never produce unguarded content."""

    def test_llm_exception_returns_factual(self):
        client = Mock()
        client.chat = Mock(side_effect=RuntimeError("LLM is down"))
        result = ContentModeClassifier(client).classify(
            brief="some brief", show_type="podcast"
        )
        assert result == "factual"

    def test_llm_exception_logs_warning(self, caplog):
        client = Mock()
        client.chat = Mock(side_effect=ConnectionError("timeout"))
        with caplog.at_level(logging.WARNING, logger="vox_romana.core.content_mode"):
            ContentModeClassifier(client).classify(brief="x", show_type="podcast")
        assert caplog.records  # at least one warning logged

    def test_unknown_show_type_calls_llm(self):
        """Unknown show type: treat like podcast — consult the LLM."""
        client = _make_client("factual")
        result = ContentModeClassifier(client).classify(
            brief="some brief", show_type="unknown_type"
        )
        assert result == "factual"
        client.chat.assert_called_once()

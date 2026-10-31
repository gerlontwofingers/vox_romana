"""Tests for the show router — classifies briefs into show types."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vox_romana.core.router import (
    SHOW_TYPES,
    RouterResult,
    ShowRouter,
    parse_router_response,
)


# ═══════════════════════════════════════════════════════════════════════════
# RouterResult
# ═══════════════════════════════════════════════════════════════════════════


class TestRouterResult:
    def test_valid_debate(self):
        r = RouterResult("debate", 0.95, "mentions arguing")
        assert r.show_type == "debate"
        assert r.confidence == 0.95
        assert r.reasoning == "mentions arguing"
        assert r.needs_research is False  # default

    def test_valid_news(self):
        r = RouterResult("news", 0.8, "mentions headlines")
        assert r.show_type == "news"

    def test_needs_research_flag(self):
        r = RouterResult("debate", 0.9, "real data requested", needs_research=True)
        assert r.needs_research is True
        r2 = RouterResult("debate", 0.9, "fictional", needs_research=False)
        assert r2.needs_research is False

    def test_valid_podcast(self):
        r = RouterResult("podcast", 0.85, "multi-host conversation")
        assert r.show_type == "podcast"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown show type"):
            RouterResult("interview", 0.9, "nope")

    def test_confidence_clamped_high(self):
        r = RouterResult("debate", 1.5, "over")
        assert r.confidence == 1.0

    def test_confidence_clamped_low(self):
        r = RouterResult("news", -0.3, "under")
        assert r.confidence == 0.0

    def test_repr(self):
        r = RouterResult("debate", 0.92, "test")
        assert "debate" in repr(r)
        assert "0.92" in repr(r)
        assert "needs_research" in repr(r)


# ═══════════════════════════════════════════════════════════════════════════
# parse_router_response
# ═══════════════════════════════════════════════════════════════════════════


class TestParseRouterResponse:
    def test_clean_json(self):
        raw = json.dumps({
            "show_type": "debate",
            "confidence": 0.95,
            "reasoning": "The brief asks for a debate."
        })
        result = parse_router_response(raw)
        assert result.show_type == "debate"
        assert result.confidence == 0.95

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"show_type": "news", "confidence": 0.8, "reasoning": "headlines"}\n```'
        result = parse_router_response(raw)
        assert result.show_type == "news"
        assert result.confidence == 0.8

    def test_json_with_plain_fences(self):
        raw = '```\n{"show_type": "debate", "confidence": 0.7, "reasoning": "discussion"}\n```'
        result = parse_router_response(raw)
        assert result.show_type == "debate"

    def test_trailing_comma(self):
        raw = '{"show_type": "news", "confidence": 0.6, "reasoning": "breaking news",}'
        result = parse_router_response(raw)
        assert result.show_type == "news"

    def test_needs_research_parsed(self):
        raw = json.dumps({
            "show_type": "debate",
            "confidence": 0.9,
            "needs_research": True,
            "reasoning": "asks for statistics"
        })
        result = parse_router_response(raw)
        assert result.needs_research is True

    def test_needs_research_defaults_false(self):
        raw = json.dumps({
            "show_type": "debate",
            "confidence": 0.9,
            "reasoning": "fictional debate"
        })
        result = parse_router_response(raw)
        assert result.needs_research is False

    def test_missing_confidence_defaults(self):
        raw = '{"show_type": "debate", "reasoning": "looks like a debate"}'
        result = parse_router_response(raw)
        assert result.confidence == 0.5

    def test_missing_reasoning_defaults(self):
        raw = '{"show_type": "news", "confidence": 0.9}'
        result = parse_router_response(raw)
        assert result.reasoning == ""

    def test_uppercase_type_normalized(self):
        raw = '{"show_type": "DEBATE", "confidence": 0.9, "reasoning": "loud"}'
        result = parse_router_response(raw)
        assert result.show_type == "debate"

    def test_padded_type_stripped(self):
        raw = '{"show_type": "  news  ", "confidence": 0.8, "reasoning": "ok"}'
        result = parse_router_response(raw)
        assert result.show_type == "news"

    def test_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parse_router_response("not json at all")

    def test_podcast_type_valid(self):
        raw = '{"show_type": "podcast", "confidence": 0.9, "reasoning": "multi-host"}'
        result = parse_router_response(raw)
        assert result.show_type == "podcast"

    def test_unknown_type_raises(self):
        raw = '{"show_type": "interview", "confidence": 0.9, "reasoning": "?"}'
        with pytest.raises(ValueError, match="Unknown show type"):
            parse_router_response(raw)

    def test_whitespace_around_json(self):
        raw = '  \n  {"show_type": "debate", "confidence": 0.85, "reasoning": "ok"}  \n  '
        result = parse_router_response(raw)
        assert result.show_type == "debate"


# ═══════════════════════════════════════════════════════════════════════════
# ShowRouter (with mocked client)
# ═══════════════════════════════════════════════════════════════════════════


class FakeClient:
    """Fake DeepSeekClient that returns canned responses."""

    def __init__(self, response: str):
        self._response = response
        self.last_messages = None

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.last_messages = messages
        return self._response


class TestShowRouter:
    def test_classify_debate(self):
        canned = json.dumps({
            "show_type": "debate",
            "confidence": 0.95,
            "reasoning": "The brief mentions debating UBI."
        })
        client = FakeClient(canned)
        router = ShowRouter(client)
        result = router.classify("Four economists debate UBI")
        assert result.show_type == "debate"
        assert result.confidence == 0.95

    def test_classify_news(self):
        canned = json.dumps({
            "show_type": "news",
            "confidence": 0.88,
            "reasoning": "The brief asks for headlines."
        })
        client = FakeClient(canned)
        router = ShowRouter(client)
        result = router.classify("Top 5 AI news stories this week")
        assert result.show_type == "news"

    def test_system_prompt_sent(self):
        canned = json.dumps({
            "show_type": "debate",
            "confidence": 0.9,
            "reasoning": "test"
        })
        client = FakeClient(canned)
        router = ShowRouter(client)
        router.classify("test brief")

        assert client.last_messages is not None
        assert client.last_messages[0]["role"] == "system"
        assert "classifier" in client.last_messages[0]["content"].lower()

    def test_brief_in_user_message(self):
        canned = json.dumps({
            "show_type": "news",
            "confidence": 0.7,
            "reasoning": "test"
        })
        client = FakeClient(canned)
        router = ShowRouter(client)
        router.classify("Breaking: AI takes over the world")

        user_msg = client.last_messages[1]
        assert user_msg["role"] == "user"
        assert "Breaking: AI takes over the world" in user_msg["content"]

    def test_markdown_fenced_response(self):
        canned = '```json\n{"show_type": "debate", "confidence": 0.92, "reasoning": "panel discussion"}\n```'
        client = FakeClient(canned)
        router = ShowRouter(client)
        result = router.classify("Panel discussion on climate policy")
        assert result.show_type == "debate"


# ═══════════════════════════════════════════════════════════════════════════
# run_show.py CLI (argument parsing only — no LLM calls)
# ═══════════════════════════════════════════════════════════════════════════


class TestRunShowArgs:
    """Test that run_show.py parses arguments correctly."""

    def _parse(self, args: list[str]):
        # Import inline to avoid dotenv issues in test env
        from run import parse_args
        return parse_args(args)

    def test_prompt_only(self, tmp_path):
        from vox_romana.core.paths import OUTPUT_DIR
        p = tmp_path / "brief.txt"
        p.write_text("test")
        args = self._parse([str(p)])
        assert args.prompt == p
        assert args.show_type is None
        assert args.output == OUTPUT_DIR

    def test_force_type(self, tmp_path):
        p = tmp_path / "brief.txt"
        p.write_text("test")
        args = self._parse([str(p), "--type", "news"])
        assert args.show_type == "news"

    def test_debate_overrides(self, tmp_path):
        p = tmp_path / "brief.txt"
        p.write_text("test")
        args = self._parse([str(p), "--turns", "50", "--tone", "unhinged", "--debate-format", "oxford"])
        assert args.turns == 50
        assert args.tone == "unhinged"
        assert args.debate_format == "oxford"

    def test_news_overrides(self, tmp_path):
        p = tmp_path / "brief.txt"
        p.write_text("test")
        args = self._parse([str(p), "--segments", "6"])
        assert args.segments == 6

    def test_research_flag(self, tmp_path):
        p = tmp_path / "brief.txt"
        p.write_text("test")
        args = self._parse([str(p), "--research"])
        assert args.research is True

    def test_no_research_flag(self, tmp_path):
        p = tmp_path / "brief.txt"
        p.write_text("test")
        args = self._parse([str(p), "--no-research"])
        assert args.research is False

    def test_research_default_is_none(self, tmp_path):
        p = tmp_path / "brief.txt"
        p.write_text("test")
        args = self._parse([str(p)])
        assert args.research is None

    def test_tag_and_quiet(self, tmp_path):
        p = tmp_path / "brief.txt"
        p.write_text("test")
        args = self._parse([str(p), "--tag", "v2", "--quiet"])
        assert args.tag == "v2"
        assert args.quiet is True


# ═══════════════════════════════════════════════════════════════════════════
# SHOW_TYPES constant
# ═══════════════════════════════════════════════════════════════════════════


class TestShowTypes:
    def test_debate_in_types(self):
        assert "debate" in SHOW_TYPES

    def test_news_in_types(self):
        assert "news" in SHOW_TYPES

    def test_radio_drama_in_types(self):
        assert "radio_drama" in SHOW_TYPES

    def test_is_tuple(self):
        assert isinstance(SHOW_TYPES, tuple)


# ═══════════════════════════════════════════════════════════════════════════
# render_show.py CLI arg parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderShowArgs:
    """Tests for render_show.py's parse_args — voice-policy flag."""

    @staticmethod
    def _parse(argv: list[str]):
        from vox_romana.cli.runners import _render_parse_args as parse_args
        return parse_args(argv)

    def test_voice_policy_default_is_auto(self, tmp_path):
        p = tmp_path / "show.json"
        p.write_text("{}")
        args = self._parse([str(p)])
        assert args.voice_policy == "auto"

    def test_voice_policy_matched(self, tmp_path):
        p = tmp_path / "show.json"
        p.write_text("{}")
        args = self._parse([str(p), "--voice-policy", "matched"])
        assert args.voice_policy == "matched"

    def test_voice_policy_invalid_rejected(self, tmp_path):
        import sys
        p = tmp_path / "show.json"
        p.write_text("{}")
        with pytest.raises(SystemExit):
            self._parse([str(p), "--voice-policy", "bogus"])

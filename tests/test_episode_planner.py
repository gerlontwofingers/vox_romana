"""Tests for vox_romana.planning.episode_planner.

Tests both the planner unit (with a stub LLM) and the ``--mode series next``
CLI subcommand wiring (with the planner mocked).

THE TESTING PHILOSOPHY: we test that the LLM contract is built correctly
(system prompt, user prompt, JSON parsing, default fallbacks) and that
the CLI correctly threads catalog data through to the planner — NOT
that the LLM produces good briefs.

Run with:
    PYTHONPATH=. .venv/bin/pytest tests/test_episode_planner.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.cli import series as series_cli
from vox_romana.core.catalog import Catalog
from vox_romana.planning.episode_planner import (
    EpisodeBrief,
    EpisodePlanner,
    _build_user_prompt,
    _extract_json_blob,
    _format_recent_episodes,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _stub_llm(payload: dict) -> MagicMock:
    """Build a stub _ChatClient whose .chat() returns the given JSON payload."""
    client = MagicMock()
    client.chat.return_value = json.dumps(payload)
    return client


def _good_payload(**overrides) -> dict:
    """A minimally valid LLM response payload."""
    base = {
        "title": "The Algorithm Will See You Now",
        "angle": "Triage AIs are silently rationing emergency-room care.",
        "hooks": ["A patient flagged 'low priority' codes blue.", "ER nurse blows the whistle."],
        "guests": ["Dr. Emily Sato", "Mark Patel"],
        "estimated_duration": 18,
        "brief": (
            "Investigate how triage AIs deployed in U.S. emergency rooms in 2025 "
            "began silently re-ranking patients in ways that disadvantaged elderly "
            "and non-English-speaking visitors.  Build the segment around two "
            "documented misclassifications and one whistleblower account."
        ),
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════
# 1. Pure helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatRecentEpisodes:
    def test_empty_returns_placeholder(self):
        assert _format_recent_episodes([]) == "(none yet — this is episode 1)"

    def test_includes_episode_number_and_title(self):
        out = _format_recent_episodes([
            {"episode_number": 3, "title": "Solar Wars", "brief": "Tariffs hit panels."},
        ])
        assert "[3]" in out
        assert "Solar Wars" in out
        assert "Tariffs hit panels" in out

    def test_truncates_long_briefs(self):
        long_brief = "x" * 500
        out = _format_recent_episodes([
            {"episode_number": 1, "title": "T", "brief": long_brief},
        ])
        # 200-char cap + ellipsis.
        assert "…" in out
        assert out.count("x") <= 210


class TestBuildUserPrompt:
    def test_includes_all_key_fields(self):
        p = _build_user_prompt(
            series_title="Tech Files",
            series_show_type="news",
            series_tone="serious",
            bible="A weekly tech-news show.",
            recent_episodes=[],
            next_episode_number=4,
            topic="quantum networking",
        )
        assert "Tech Files" in p
        assert "news" in p
        assert "serious" in p
        assert "weekly tech-news" in p
        assert "#4" in p
        assert "quantum networking" in p

    def test_no_topic_uses_placeholder(self):
        p = _build_user_prompt(
            series_title="X",
            series_show_type=None,
            series_tone=None,
            bible="",
            recent_episodes=[],
            next_episode_number=1,
            topic=None,
        )
        assert "(none — pick the most compelling fresh angle)" in p


class TestExtractJsonBlob:
    def test_plain_object(self):
        assert _extract_json_blob('{"a": 1}') == '{"a": 1}'

    def test_fenced_block(self):
        wrapped = 'Sure, here is the brief:\n```json\n{"a": 2}\n```\nHope this helps.'
        assert _extract_json_blob(wrapped) == '{"a": 2}'

    def test_substring_fallback(self):
        wrapped = 'Some preamble {"a": 3, "b": [1,2]} and trailing chatter.'
        assert _extract_json_blob(wrapped) == '{"a": 3, "b": [1,2]}'

    def test_no_object_raises(self):
        with pytest.raises(ValueError):
            _extract_json_blob("no braces here")


# ═══════════════════════════════════════════════════════════════════════════
# 2. EpisodePlanner.plan_next
# ═══════════════════════════════════════════════════════════════════════════


class TestEpisodePlanner:
    def test_returns_episode_brief_with_all_fields(self):
        client = _stub_llm(_good_payload())
        planner = EpisodePlanner(client)
        brief = planner.plan_next(
            series_title="The Tech Files",
            series_show_type="news",
            series_tone="serious",
            bible="A weekly tech-news investigative show.",
            recent_episodes=[],
            next_episode_number=7,
            topic=None,
        )
        assert isinstance(brief, EpisodeBrief)
        assert brief.title == "The Algorithm Will See You Now"
        assert brief.episode_number == 7
        assert brief.estimated_duration == 18
        assert len(brief.hooks) == 2
        assert "Dr. Emily Sato" in brief.guests
        assert "Investigate how triage AIs" in brief.prompt_text

    def test_threads_topic_into_prompt(self):
        client = _stub_llm(_good_payload())
        planner = EpisodePlanner(client)
        planner.plan_next(
            series_title="X",
            series_show_type="news",
            series_tone=None,
            bible="b",
            recent_episodes=[],
            next_episode_number=1,
            topic="AI copyright",
        )
        # Verify the user prompt sent to the LLM contained our topic.
        msgs = client.chat.call_args.args[0]
        assert any("AI copyright" in m["content"] for m in msgs if m["role"] == "user")

    def test_threads_recent_episodes_into_prompt(self):
        client = _stub_llm(_good_payload())
        planner = EpisodePlanner(client)
        planner.plan_next(
            series_title="X",
            series_show_type="news",
            series_tone=None,
            bible="b",
            recent_episodes=[
                {"episode_number": 1, "title": "Solar Wars", "brief": "tariffs"},
                {"episode_number": 2, "title": "Chip Shortage", "brief": "TSMC"},
            ],
            next_episode_number=3,
            topic=None,
        )
        msgs = client.chat.call_args.args[0]
        user_content = next(m["content"] for m in msgs if m["role"] == "user")
        assert "Solar Wars" in user_content
        assert "Chip Shortage" in user_content
        assert "#3" in user_content

    def test_handles_fenced_json_response(self):
        raw = "Here you go:\n```json\n" + json.dumps(_good_payload()) + "\n```"
        client = MagicMock()
        client.chat.return_value = raw
        planner = EpisodePlanner(client)
        brief = planner.plan_next(
            series_title="X",
            series_show_type=None,
            series_tone=None,
            bible="",
            recent_episodes=[],
            next_episode_number=1,
            topic=None,
        )
        assert brief.title == "The Algorithm Will See You Now"

    def test_default_duration_falls_back_to_show_type(self):
        # Payload omits estimated_duration → falls back to news = 12.
        client = _stub_llm(_good_payload(estimated_duration=0))
        planner = EpisodePlanner(client)
        brief = planner.plan_next(
            series_title="X",
            series_show_type="news",
            series_tone=None,
            bible="",
            recent_episodes=[],
            next_episode_number=1,
            topic=None,
        )
        assert brief.estimated_duration == 12.0

    def test_missing_brief_field_raises(self):
        bad = _good_payload()
        bad["brief"] = ""
        client = _stub_llm(bad)
        planner = EpisodePlanner(client)
        with pytest.raises(RuntimeError, match="missing the required 'brief' field"):
            planner.plan_next(
                series_title="X",
                series_show_type=None,
                series_tone=None,
                bible="",
                recent_episodes=[],
                next_episode_number=1,
                topic=None,
            )

    def test_invalid_json_raises(self):
        client = MagicMock()
        client.chat.return_value = "totally not json"
        planner = EpisodePlanner(client)
        with pytest.raises(RuntimeError, match="did not return valid JSON"):
            planner.plan_next(
                series_title="X",
                series_show_type=None,
                series_tone=None,
                bible="",
                recent_episodes=[],
                next_episode_number=1,
                topic=None,
            )


# ═══════════════════════════════════════════════════════════════════════════
# 3. CLI wiring (--mode series next)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def in_memory_catalog(monkeypatch):
    """Redirect Catalog() in series.py to an in-memory SQLite."""
    cat = Catalog(db_path=":memory:")

    def _factory(*args, **kwargs):
        return cat

    monkeypatch.setattr(series_cli, "Catalog", _factory)
    monkeypatch.setattr(Catalog, "__exit__", lambda self, *a: None)
    return cat


@pytest.fixture
def planner_stub(monkeypatch):
    """Replace EpisodePlanner inside series.py with a stub returning a fixed brief."""

    def _make_brief(*, ep_no=1, prompt="Auto-drafted brief for episode."):
        return EpisodeBrief(
            title=f"Episode {ep_no} title",
            angle="A tight angle.",
            prompt_text=prompt,
            episode_number=ep_no,
            hooks=["Hook 1"],
            guests=[],
            estimated_duration=15.0,
            topic="",
        )

    instance = MagicMock()
    instance.plan_next.side_effect = lambda **kw: _make_brief(
        ep_no=kw["next_episode_number"]
    )

    # series._cmd_next imports inside the function — patch the module
    # the import resolves to.
    import vox_romana.planning.episode_planner as ep_mod

    monkeypatch.setattr(ep_mod, "EpisodePlanner", lambda *a, **kw: instance)
    # And short-circuit DeepSeekClient construction (no API key in tests).
    import vox_romana.core.client as client_mod

    monkeypatch.setattr(client_mod, "DeepSeekClient", lambda *a, **kw: MagicMock())
    return instance


class TestSeriesNextCli:
    def test_writes_brief_file_and_returns_zero(
        self, in_memory_catalog, planner_stub, tmp_path, capsys,
    ):
        sid = in_memory_catalog.create_series(
            title="The Tech Files", show_type="news", tone="serious", bible="bible body",
        )
        out = tmp_path / "ep_brief.txt"
        rc = series_cli.main(["next", sid, "--brief-output", str(out)])
        assert rc is None  # series_cli.main returns None on success
        assert out.exists()
        assert "Auto-drafted brief" in out.read_text()
        captured = capsys.readouterr()
        assert "Episode #1" in captured.out
        assert "Brief →" in captured.out

    def test_threads_topic_into_planner(
        self, in_memory_catalog, planner_stub, tmp_path,
    ):
        sid = in_memory_catalog.create_series(title="X", show_type="news", bible="b")
        out = tmp_path / "ep.txt"
        series_cli.main([
            "next", sid,
            "--topic", "AI copyright in creative industries",
            "--brief-output", str(out),
        ])
        # planner_stub.plan_next should have received our topic.
        kwargs = planner_stub.plan_next.call_args.kwargs
        assert kwargs["topic"] == "AI copyright in creative industries"
        assert kwargs["series_title"] == "X"
        assert kwargs["next_episode_number"] == 1

    def test_unknown_series_returns_one(
        self, in_memory_catalog, planner_stub, tmp_path, capsys,
    ):
        with pytest.raises(SystemExit) as exc:
            series_cli.main(["next", "deadbeef", "--brief-output", str(tmp_path / "x.txt")])
        assert exc.value.code == 1
        assert "unknown series" in capsys.readouterr().err

    def test_default_run_does_not_invoke_subprocess(
        self, in_memory_catalog, planner_stub, tmp_path, monkeypatch,
    ):
        """Without --run we must NOT shell out."""
        called = []
        monkeypatch.setattr(
            "subprocess.call",
            lambda *a, **kw: called.append((a, kw)) or 0,
        )
        sid = in_memory_catalog.create_series(title="X", show_type="news", bible="b")
        series_cli.main(["next", sid, "--brief-output", str(tmp_path / "x.txt")])
        assert called == []

    def test_run_flag_invokes_subprocess(
        self, in_memory_catalog, planner_stub, tmp_path, monkeypatch,
    ):
        called = []

        def fake_call(cmd, *a, **kw):
            called.append(cmd)
            return 0

        monkeypatch.setattr("subprocess.call", fake_call)
        sid = in_memory_catalog.create_series(title="X", show_type="news", bible="b")
        out = tmp_path / "x.txt"
        series_cli.main(["next", sid, "--run", "--brief-output", str(out)])
        assert len(called) == 1
        cmd = called[0]
        # The command should reference run.py with our brief and series id.
        assert "run.py" in cmd
        assert str(out) in cmd
        assert "--series" in cmd
        assert sid in cmd
        # And forward the show type captured from the series record.
        assert "--type" in cmd
        assert "news" in cmd

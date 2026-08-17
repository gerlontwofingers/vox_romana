"""Tests for the seminar telemetry layer added in the telemetry-hardening pass.

Covers:
    - _seminar_log_path: None when save_dir is None, correct path otherwise
    - _log_seminar_event: no-ops on None path, writes valid JSON on real path
    - _log_seminar_event: swallows write errors gracefully (no exception escapes)
    - University.seminar(): seminar_log.jsonl is written with expected events
    - _verify_seminar_result: passing case prints no FAIL rows
    - _verify_seminar_result: failing case prints FAIL rows for each broken check
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import vox_romana.core.paths as _paths
import vox_romana.university.session as _sm
from vox_romana.university.config import UniversityConfig
from vox_romana.university.session import ScholarNote, ScholarSession
from vox_romana.university.sources import Article, KnowledgeSource
from vox_romana.university.university import (
    University,
    _log_seminar_event,
    _seminar_log_path,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures & fakes
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def isolated_university_root(tmp_path, monkeypatch):
    monkeypatch.setattr(_paths, "UNIVERSITY_ROOT", tmp_path)
    monkeypatch.setattr(_sm, "UNIVERSITY_ROOT", tmp_path)
    return tmp_path


class _ScriptedClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def chat(self, messages, **kwargs) -> str:
        idx = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return self._responses[idx]


class _FakeSource(KnowledgeSource):
    name = "fake"

    def __init__(self, by_query: dict | None = None):
        self._by_query = dict(by_query or {})

    def search(self, query: str, n: int = 10) -> list[Article]:
        ql = query.lower()
        for key, hits in self._by_query.items():
            if key.lower() in ql:
                return list(hits)[:n]
        return []

    def fetch(self, identifier: str) -> Article | None:
        for hits in self._by_query.values():
            for art in hits:
                if art.title == identifier:
                    return art
        return None


def _note_resp(score: float = 8.0) -> str:
    return json.dumps({
        "key_facts": ["Fact"],
        "summary": "Summary.",
        "links_to_explore": [],
        "relevance_score": score,
    })


def _minimal_config(**overrides) -> UniversityConfig:
    base = dict(
        notes_per_synthesis=100,
        priority_interval=100,
        relevance_threshold=4.0,
        max_links_per_article=5,
    )
    base.update(overrides)
    return UniversityConfig(**base)


# ══════════════════════════════════════════════════════════════════════════════
# _seminar_log_path
# ══════════════════════════════════════════════════════════════════════════════


class TestSeminarLogPath:
    def test_none_save_dir_returns_none(self):
        assert _seminar_log_path(None) is None

    def test_real_dir_returns_correct_path(self, tmp_path):
        p = _seminar_log_path(tmp_path)
        assert p is not None
        assert p == tmp_path / "seminar_log.jsonl"

    def test_returns_path_object(self, tmp_path):
        p = _seminar_log_path(tmp_path)
        assert isinstance(p, Path)


# ══════════════════════════════════════════════════════════════════════════════
# _log_seminar_event
# ══════════════════════════════════════════════════════════════════════════════


class TestLogSeminarEvent:
    def test_noop_on_none_path(self):
        # Must not raise even when path is None.
        _log_seminar_event(None, "test_event", foo="bar")

    def test_writes_json_entry(self, tmp_path):
        p = tmp_path / "seminar_log.jsonl"
        _log_seminar_event(p, "test_event", foo="bar", n=42)
        lines = p.read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == "test_event"
        assert entry["foo"] == "bar"
        assert entry["n"] == 42
        assert "logged_at" in entry

    def test_appends_multiple_events(self, tmp_path):
        p = tmp_path / "seminar_log.jsonl"
        _log_seminar_event(p, "event_a")
        _log_seminar_event(p, "event_b")
        _log_seminar_event(p, "event_c")
        lines = p.read_text().splitlines()
        assert len(lines) == 3
        event_types = [json.loads(l)["event_type"] for l in lines]
        assert event_types == ["event_a", "event_b", "event_c"]

    def test_swallows_write_error_silently(self, tmp_path):
        # A path inside a non-existent deeply nested read-only tree
        # would normally raise; the helper must not propagate.
        p = tmp_path / "nonexistent_parent" / "sub" / "log.jsonl"
        # Create the grandparent as a FILE so mkdir will fail inside:
        bad_parent = tmp_path / "nonexistent_parent"
        bad_parent.write_text("I am a file, not a dir")
        # Should not raise.
        _log_seminar_event(p, "test_event")


# ══════════════════════════════════════════════════════════════════════════════
# University.seminar() — seminar_log.jsonl written with expected events
# ══════════════════════════════════════════════════════════════════════════════


class TestSeminarEventLog:
    def test_seminar_log_written_on_happy_path(self, tmp_path, isolated_university_root):
        """seminar() with two topics must write seminar_log.jsonl containing
        at least: seminar_started, seed_phase_complete, seminar_complete."""
        article = Article(title="Aqueduct", url="u1", text="text", source_name="fake")
        src = _FakeSource({"q": [article]})

        # Enough responses for: plan×2, note-extract×2, synthesis×2,
        # colloquium exchange×2 (A→B, B→A)
        synth_resp = "Short synthesis for this topic."
        exchange_resp = json.dumps({"comment": "connection", "queries": []})
        client = _ScriptedClient([
            json.dumps({"queries": ["q"]}),   # plan for topic A
            _note_resp(),                      # article A
            synth_resp,                        # synthesis A
            json.dumps({"queries": ["q"]}),   # plan for topic B
            _note_resp(),                      # article B
            synth_resp,                        # synthesis B
            exchange_resp,                     # colloquium A→B
            exchange_resp,                     # colloquium B→A
        ])

        uni = University(client, [src], _minimal_config())
        uni.seminar(
            ["Topic A", "Topic B"],
            budget_hours_each=10.0,
            save_dir=tmp_path,
        )

        log_path = tmp_path / "seminar_log.jsonl"
        assert log_path.exists(), "seminar_log.jsonl must be written"

        lines = log_path.read_text().splitlines()
        event_types = [json.loads(l)["event_type"] for l in lines if l.strip()]
        assert "seminar_started" in event_types
        assert "seed_phase_complete" in event_types
        assert "seminar_complete" in event_types

    def test_seminar_started_contains_topic_list(self, tmp_path, isolated_university_root):
        article = Article(title="X", url="u", text="t", source_name="fake")
        src = _FakeSource({"q": [article]})
        synth_resp = "synthesis"
        exchange_resp = json.dumps({"comment": "ok", "queries": []})
        client = _ScriptedClient([
            json.dumps({"queries": ["q"]}), _note_resp(), synth_resp,
            json.dumps({"queries": ["q"]}), _note_resp(), synth_resp,
            exchange_resp, exchange_resp,
        ])
        uni = University(client, [src], _minimal_config())
        uni.seminar(["Alpha", "Beta"], budget_hours_each=10.0, save_dir=tmp_path)

        log_path = tmp_path / "seminar_log.jsonl"
        started = next(
            json.loads(l) for l in log_path.read_text().splitlines()
            if json.loads(l).get("event_type") == "seminar_started"
        )
        assert started["topics"] == ["Alpha", "Beta"]
        assert started["n_scholars"] == 2

    def test_seminar_complete_has_per_session_summary(self, tmp_path, isolated_university_root):
        article = Article(title="X", url="u", text="t", source_name="fake")
        src = _FakeSource({"q": [article]})
        synth_resp = "synthesis text"
        exchange_resp = json.dumps({"comment": "ok", "queries": []})
        client = _ScriptedClient([
            json.dumps({"queries": ["q"]}), _note_resp(), synth_resp,
            json.dumps({"queries": ["q"]}), _note_resp(), synth_resp,
            exchange_resp, exchange_resp,
        ])
        uni = University(client, [src], _minimal_config())
        uni.seminar(["Topic A", "Topic B"], budget_hours_each=10.0, save_dir=tmp_path)

        log_path = tmp_path / "seminar_log.jsonl"
        complete = next(
            json.loads(l) for l in log_path.read_text().splitlines()
            if json.loads(l).get("event_type") == "seminar_complete"
        )
        sessions_info = complete["sessions"]
        assert len(sessions_info) == 2
        for info in sessions_info:
            assert "session_id" in info
            assert "status" in info
            assert "n_notes" in info
            assert "has_synthesis" in info


# ══════════════════════════════════════════════════════════════════════════════
# _verify_seminar_result
# ══════════════════════════════════════════════════════════════════════════════


class TestVerifySeminarResult:
    """Tests for the cli._verify_seminar_result() helper."""

    def _make_session(
        self,
        status: str = "completed",
        synthesis: str = "Some synthesis text.",
        source_url: str = "https://example.com/article",
        n_colloquium_rounds: int = 1,
    ) -> ScholarSession:
        from vox_romana.university.session import ColloquiumRound

        s = ScholarSession(topic="Test topic")
        s.status = status
        s.synthesis = synthesis
        note = ScholarNote(
            article_title="An article",
            source_url=source_url,
            source_name="wikipedia",
        )
        s.notes = [note]
        for i in range(n_colloquium_rounds):
            s.colloquium_rounds.append(ColloquiumRound(round_n=i + 1))
        return s

    def test_passing_case_prints_no_fail(self, tmp_path, capsys):
        from vox_romana.university.cli import _verify_seminar_result

        log_path = tmp_path / "seminar_log.jsonl"
        log_path.write_text("")  # file exists

        s1 = self._make_session()
        s2 = self._make_session()
        _verify_seminar_result([s1, s2], tmp_path)

        out = capsys.readouterr().out
        assert "FAIL" not in out
        assert "PASS" in out

    def test_interrupted_session_fails_check(self, tmp_path, capsys):
        from vox_romana.university.cli import _verify_seminar_result

        log_path = tmp_path / "seminar_log.jsonl"
        log_path.write_text("")

        s1 = self._make_session(status="interrupted")
        s2 = self._make_session()
        _verify_seminar_result([s1, s2], tmp_path)

        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "interrupted" in out

    def test_no_synthesis_fails_check(self, tmp_path, capsys):
        from vox_romana.university.cli import _verify_seminar_result

        log_path = tmp_path / "seminar_log.jsonl"
        log_path.write_text("")

        s1 = self._make_session(synthesis="")
        s2 = self._make_session()
        _verify_seminar_result([s1, s2], tmp_path)

        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "synthesis" in out.lower()

    def test_no_colloquium_rounds_fails_check(self, tmp_path, capsys):
        from vox_romana.university.cli import _verify_seminar_result

        log_path = tmp_path / "seminar_log.jsonl"
        log_path.write_text("")

        s1 = self._make_session(n_colloquium_rounds=0)
        s2 = self._make_session(n_colloquium_rounds=0)
        _verify_seminar_result([s1, s2], tmp_path)

        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "colloquium" in out.lower()

    def test_memory_only_urls_fail_source_url_check(self, tmp_path, capsys):
        from vox_romana.university.cli import _verify_seminar_result

        log_path = tmp_path / "seminar_log.jsonl"
        log_path.write_text("")

        s1 = self._make_session(source_url="memory://chunk/abc")
        s2 = self._make_session(source_url="memory://chunk/xyz")
        _verify_seminar_result([s1, s2], tmp_path)

        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "source" in out.lower()

    def test_missing_log_file_fails_log_check(self, tmp_path, capsys):
        from vox_romana.university.cli import _verify_seminar_result

        # Do NOT create the log file.
        s1 = self._make_session()
        s2 = self._make_session()
        _verify_seminar_result([s1, s2], tmp_path)

        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "seminar_log" in out.lower()

    def test_none_save_dir_fails_log_check(self, capsys):
        from vox_romana.university.cli import _verify_seminar_result

        s1 = self._make_session()
        _verify_seminar_result([s1], None)

        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "save_dir" in out.lower()

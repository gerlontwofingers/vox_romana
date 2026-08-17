"""Tests for vox_romana.university.scholar.

Covers:
    - Module-level helpers: strip_think, extract_json, _split_qid,
      _make_qid, _format_hms
    - Scholar construction (requires at least one source)
    - Scholar._fetch dispatch
    - Scholar.study: one-article loop, below-threshold note, disk checkpoint,
      invalid topic raises
    - Scholar.ask: returns string, does not mutate session
    - Scholar.interrupt: sets stop flag; loop exits with status "interrupted"
    - Scholar.extend: resumes paused session
"""

from __future__ import annotations

import json

import pytest

import vox_romana.core.paths as _paths
import vox_romana.university.session as _sm
from vox_romana.university.config import UniversityConfig
from vox_romana.university.scholar import (
    Scholar,
    _format_hms,
    _make_qid,
    _split_qid,
    extract_json,
    strip_think,
)
from vox_romana.university.session import ScholarNote, ScholarSession
from vox_romana.university.sources import Article, KnowledgeSource


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures and stubs
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def isolated_university_root(tmp_path, monkeypatch):
    monkeypatch.setattr(_paths, "UNIVERSITY_ROOT", tmp_path)
    monkeypatch.setattr(_sm, "UNIVERSITY_ROOT", tmp_path)
    return tmp_path


class _FakeClient:
    """Cycles through preset responses by call index; repeats last on overflow."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


class _FakeSource(KnowledgeSource):
    name = "fake"

    def __init__(self, articles: list[Article] | None = None):
        self._articles = {a.title: a for a in (articles or [])}

    def search(self, query: str, n: int = 10) -> list[Article]:
        return list(self._articles.values())[:n]

    def fetch(self, identifier: str) -> Article | None:
        return self._articles.get(identifier)


def _minimal_config(**overrides) -> UniversityConfig:
    """UniversityConfig tuned to avoid mid-loop LLM calls in small tests."""
    cfg = UniversityConfig(
        notes_per_synthesis=100,   # never trigger mid-loop synthesis update
        priority_interval=100,     # never trigger re-prioritisation
        relevance_threshold=4.0,
        max_links_per_article=5,
        max_queue_size=50,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _note_resp(key_facts=None, summary="Summary text.", score=8.0, links=None) -> str:
    return json.dumps({
        "key_facts": key_facts or ["Fact A", "Fact B"],
        "summary": summary,
        "links_to_explore": links or [],
        "relevance_score": score,
    })


ARTICLE_ONE = Article(
    title="Article One",
    url="https://example.com/article_one",
    text="This is the full article text about Roman roads.",
    snippet="Roman roads snippet.",
    links=("Linked Article",),
    source_name="fake",
)


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════


class TestStripThink:
    def test_strips_think_block(self):
        raw = "<think>internal reasoning</think>\nactual response"
        assert strip_think(raw) == "actual response"

    def test_no_think_block_unchanged(self):
        assert strip_think("hello world") == "hello world"

    def test_none_returns_empty_string(self):
        assert strip_think(None) == ""

    def test_empty_string_returns_empty(self):
        assert strip_think("") == ""

    def test_multiline_think_stripped(self):
        raw = "<think>\nline 1\nline 2\n</think>answer"
        assert strip_think(raw) == "answer"

    def test_case_insensitive_think_tag(self):
        raw = "<THINK>ignore this</THINK>real answer"
        assert strip_think(raw) == "real answer"


class TestExtractJson:
    def test_plain_json_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_plain_json_array(self):
        assert extract_json('[1, 2, 3]') == [1, 2, 3]

    def test_fenced_json(self):
        raw = "```json\n{\"x\": 42}\n```"
        assert extract_json(raw) == {"x": 42}

    def test_json_after_think_block(self):
        raw = '<think>thinking</think>\n{"key": "val"}'
        assert extract_json(raw) == {"key": "val"}

    def test_prose_wrapping_json(self):
        raw = 'Here is the result: {"a": 1} as requested.'
        assert extract_json(raw) == {"a": 1}

    def test_unparseable_returns_none(self):
        assert extract_json("not json at all") is None

    def test_empty_string_returns_none(self):
        assert extract_json("") is None

    def test_none_returns_none(self):
        assert extract_json(None) is None


class TestSplitMakeQid:
    def test_split_with_colon(self):
        src, ident = _split_qid("wikipedia:Roman roads")
        assert src == "wikipedia"
        assert ident == "Roman roads"

    def test_split_no_colon_returns_none_source(self):
        src, ident = _split_qid("Roman roads")
        assert src is None
        assert ident == "Roman roads"

    def test_split_preserves_colons_in_identifier(self):
        src, ident = _split_qid("local:study:abc123")
        assert src == "local"
        assert ident == "study:abc123"

    def test_make_qid_joins_with_colon(self):
        assert _make_qid("wikipedia", "Roman roads") == "wikipedia:Roman roads"

    def test_make_qid_roundtrips_split(self):
        qid = _make_qid("fake", "My Article")
        src, ident = _split_qid(qid)
        assert src == "fake"
        assert ident == "My Article"


class TestFormatHms:
    def test_zero(self):
        assert _format_hms(0) == "00:00:00"

    def test_one_hour(self):
        assert _format_hms(3600) == "01:00:00"

    def test_ninety_minutes(self):
        assert _format_hms(5400) == "01:30:00"

    def test_negative_clamps_to_zero(self):
        assert _format_hms(-10) == "00:00:00"

    def test_seconds_only(self):
        assert _format_hms(45) == "00:00:45"


# ══════════════════════════════════════════════════════════════════════════════
# Scholar construction
# ══════════════════════════════════════════════════════════════════════════════


class TestScholarConstruction:
    def test_requires_at_least_one_source(self):
        client = _FakeClient(["{}"])
        with pytest.raises(ValueError):
            Scholar(client, sources=[])

    def test_accepts_single_source(self):
        client = _FakeClient(["{}"])
        src = _FakeSource()
        scholar = Scholar(client, [src], _minimal_config())
        assert scholar.sources == [src]

    def test_accepts_multiple_sources(self):
        client = _FakeClient(["{}"])
        src1 = _FakeSource()
        src2 = _FakeSource()
        scholar = Scholar(client, [src1, src2], _minimal_config())
        assert len(scholar.sources) == 2

    def test_default_config_applied_when_none_given(self, monkeypatch):
        """Scholar falls back to from_project_yaml() which reads the real YAML."""
        client = _FakeClient(["{}"])
        src = _FakeSource()
        # Don't blow up — just ensure construction succeeds.
        # (from_project_yaml may return defaults if the block is absent.)
        scholar = Scholar(client, [src])
        assert scholar.config is not None


# ══════════════════════════════════════════════════════════════════════════════
# Scholar._fetch dispatch
# ══════════════════════════════════════════════════════════════════════════════


class TestScholarFetch:
    def test_fetch_finds_article_by_source_prefix(self, tmp_path):
        client = _FakeClient(["{}"])
        src = _FakeSource([ARTICLE_ONE])
        scholar = Scholar(client, [src], _minimal_config())
        article = scholar._fetch("fake:Article One")
        assert article is not None
        assert article.title == "Article One"

    def test_fetch_returns_none_for_unknown_article(self, tmp_path):
        client = _FakeClient(["{}"])
        src = _FakeSource()
        scholar = Scholar(client, [src], _minimal_config())
        assert scholar._fetch("fake:Nonexistent") is None

    def test_fetch_returns_none_for_empty_text_article(self, tmp_path):
        """An article with empty text is treated as not found."""
        no_text = Article(title="Empty", text="", source_name="fake")
        src = _FakeSource([no_text])
        client = _FakeClient(["{}"])
        scholar = Scholar(client, [src], _minimal_config())
        # fetch() returns None when article.text is empty
        result = scholar._fetch("fake:Empty")
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# Scholar.study — one-article study loop
# ══════════════════════════════════════════════════════════════════════════════


class TestScholarStudy:
    def test_completes_with_single_article(self, tmp_path):
        client = _FakeClient([
            _note_resp(),             # _read_and_note
            "Final synthesis text.",  # _finalize
        ])
        src = _FakeSource([ARTICLE_ONE])
        scholar = Scholar(client, [src], _minimal_config())
        session = scholar.study(
            "Roman roads",
            budget_hours=1.0,
            save_dir=tmp_path,
            seed_article_ids=["fake:Article One"],
        )
        assert session.status == "complete"
        assert len(session.notes) == 1
        assert session.notes[0].article_title == "Article One"
        assert session.synthesis == "Final synthesis text."

    def test_article_id_recorded_in_articles_read(self, tmp_path):
        client = _FakeClient([_note_resp(), "synthesis"])
        src = _FakeSource([ARTICLE_ONE])
        scholar = Scholar(client, [src], _minimal_config())
        session = scholar.study(
            "Roman roads",
            budget_hours=1.0,
            save_dir=tmp_path,
            seed_article_ids=["fake:Article One"],
        )
        assert len(session.articles_read) == 1
        assert "Article One" in session.articles_read[0]

    def test_note_below_threshold_not_added_to_notes(self, tmp_path):
        client = _FakeClient([
            _note_resp(score=1.0),  # below default threshold=4.0
            "synthesis",
        ])
        src = _FakeSource([ARTICLE_ONE])
        scholar = Scholar(client, [src], _minimal_config())
        session = scholar.study(
            "Roman roads",
            budget_hours=1.0,
            save_dir=tmp_path,
            seed_article_ids=["fake:Article One"],
        )
        # Article processed but note discarded
        assert len(session.notes) == 0
        assert len(session.articles_read) == 1

    def test_session_saved_to_disk(self, tmp_path):
        client = _FakeClient([_note_resp(), "synthesis"])
        src = _FakeSource([ARTICLE_ONE])
        scholar = Scholar(client, [src], _minimal_config())
        session = scholar.study(
            "Roman roads",
            budget_hours=1.0,
            save_dir=tmp_path,
            seed_article_ids=["fake:Article One"],
        )
        session_json = tmp_path / session.session_id / "session.json"
        assert session_json.exists()

    def test_invalid_topic_raises_value_error(self, tmp_path):
        client = _FakeClient(["{}"])
        src = _FakeSource([ARTICLE_ONE])
        scholar = Scholar(client, [src], _minimal_config())
        with pytest.raises(ValueError):
            scholar.study("", budget_hours=1.0, save_dir=tmp_path)

    def test_empty_queue_gives_complete_status(self, tmp_path):
        """No articles queued → loop doesn't run → status is 'complete'."""
        client = _FakeClient(["{}"])
        src = _FakeSource()  # returns nothing
        scholar = Scholar(client, [src], _minimal_config())
        session = scholar.study(
            "Topic",
            budget_hours=1.0,
            save_dir=tmp_path,
            seed_article_ids=[],  # explicitly empty queue
        )
        assert session.status == "complete"
        assert len(session.notes) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Scholar.ask
# ══════════════════════════════════════════════════════════════════════════════


class TestScholarAsk:
    def _session_with_notes(self) -> ScholarSession:
        s = ScholarSession(topic="Roman roads", status="complete")
        s.notes = [
            ScholarNote(
                article_title="Via Appia",
                summary="The Appian Way was Rome's first great road.",
                relevance_score=9.0,
            )
        ]
        s.synthesis = "Roman roads were a feat of engineering."
        return s

    def test_returns_string_answer(self):
        session = self._session_with_notes()
        client = _FakeClient(["Roman roads used stone and concrete."])
        src = _FakeSource()
        scholar = Scholar(client, [src], _minimal_config())
        answer = scholar.ask("What materials did Roman roads use?", session=session)
        assert isinstance(answer, str)
        assert "Roman roads" in answer

    def test_strips_think_blocks_from_answer(self):
        session = self._session_with_notes()
        client = _FakeClient(["<think>internal</think>The answer is concrete."])
        src = _FakeSource()
        scholar = Scholar(client, [src], _minimal_config())
        answer = scholar.ask("question", session=session)
        assert "<think>" not in answer
        assert "The answer is concrete." == answer

    def test_does_not_mutate_session_notes(self):
        session = self._session_with_notes()
        original_note_count = len(session.notes)
        client = _FakeClient(["Answer here."])
        src = _FakeSource()
        scholar = Scholar(client, [src], _minimal_config())
        scholar.ask("question", session=session)
        assert len(session.notes) == original_note_count

    def test_does_not_change_session_status(self):
        session = self._session_with_notes()
        client = _FakeClient(["Answer."])
        src = _FakeSource()
        scholar = Scholar(client, [src], _minimal_config())
        scholar.ask("question", session=session)
        assert session.status == "complete"


# ══════════════════════════════════════════════════════════════════════════════
# Scholar.interrupt
# ══════════════════════════════════════════════════════════════════════════════


class TestScholarInterrupt:
    def test_interrupt_sets_stop_flag(self):
        client = _FakeClient(["{}"])
        src = _FakeSource()
        scholar = Scholar(client, [src], _minimal_config())
        assert not scholar._stop.is_set()
        scholar.interrupt()
        assert scholar._stop.is_set()

    def test_stop_flag_is_cleared_on_study_start(self, tmp_path):
        """_run_loop calls _stop.clear() on entry, so a pre-set interrupt
        does not carry over — the loop starts fresh each time."""
        client = _FakeClient([_note_resp(), "synthesis"])
        src = _FakeSource([ARTICLE_ONE])
        scholar = Scholar(client, [src], _minimal_config())
        scholar.interrupt()  # set _stop before study
        assert scholar._stop.is_set()  # confirmed set
        session = scholar.study(
            "Roman roads",
            budget_hours=1.0,
            save_dir=tmp_path,
            seed_article_ids=["fake:Article One"],
        )
        # _run_loop.clear() reset the flag — loop ran normally
        assert session.status == "complete"


# ══════════════════════════════════════════════════════════════════════════════
# Scholar.extend
# ══════════════════════════════════════════════════════════════════════════════


class TestScholarExtend:
    def test_extend_resumes_paused_session(self, tmp_path):
        """extend() re-runs the loop; a paused session with no queued
        articles re-plans queries and, with an empty source, finishes
        immediately as 'complete'."""
        client = _FakeClient([
            '{"queries": ["q1"]}',  # _plan_initial_queries (queue was empty)
            "{}",                   # fallback
        ])
        src = _FakeSource()  # empty — no articles
        scholar = Scholar(client, [src], _minimal_config())

        session = ScholarSession(topic="Topic", budget_hours=1.0, status="paused")
        session.save(tmp_path)

        extended = scholar.extend(
            session,
            additional_hours=0.5,
            save_dir=tmp_path,
        )
        assert extended.status == "complete"
        assert extended.budget_hours == 1.5

    def test_extend_increments_budget(self, tmp_path):
        client = _FakeClient(['{"queries": []}'])
        src = _FakeSource()
        scholar = Scholar(client, [src], _minimal_config())

        session = ScholarSession(topic="Topic", budget_hours=2.0, status="paused")
        session.save(tmp_path)

        extended = scholar.extend(session, additional_hours=1.0, save_dir=tmp_path)
        assert extended.budget_hours == 3.0

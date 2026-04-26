"""End-to-end integration tests for the University research system.

Uses fake LLM client and fake knowledge source — no network, no Ollama.

Covers:
    - University construction (no sources raises)
    - University.spawn_scholar returns fresh instances
    - University.study: completes, persists, logs to catalog, no-catalog works
    - University.study_many: one session per topic, skips empty topics
    - University.cross_synthesize: merges sessions, skips empty, returns ""
      for no usable sessions
    - UniversitySearchProvider integration with a fully-studied session
"""

from __future__ import annotations

import json

import pytest

import vox_romana.core.paths as _paths
import vox_romana.university.session as _sm
from vox_romana.core.catalog import Catalog
from vox_romana.university.config import UniversityConfig
from vox_romana.university.search_provider import UniversitySearchProvider
from vox_romana.university.session import ScholarNote, ScholarSession
from vox_romana.university.sources import Article, KnowledgeSource
from vox_romana.university.university import University


# ══════════════════════════════════════════════════════════════════════════════
# Fakes / stubs
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
        self.call_count = 0

    def chat(self, messages, **kwargs) -> str:
        idx = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return self._responses[idx]


class _FakeSource(KnowledgeSource):
    name = "fake"

    def __init__(self, articles: list[Article] | None = None):
        self._articles = {a.title: a for a in (articles or [])}

    def search(self, query: str, n: int = 10) -> list[Article]:
        return list(self._articles.values())[:n]

    def fetch(self, identifier: str) -> Article | None:
        return self._articles.get(identifier)


class _EmptySource(KnowledgeSource):
    """Always returns empty results — useful for flow tests that don't
    need real articles."""
    name = "empty"

    def search(self, query: str, n: int = 10) -> list[Article]:
        return []

    def fetch(self, identifier: str) -> Article | None:
        return None


def _note_resp(score: float = 8.0) -> str:
    return json.dumps({
        "key_facts": ["Key fact A", "Key fact B"],
        "summary": "A well-written summary of the article.",
        "links_to_explore": [],
        "relevance_score": score,
    })


def _minimal_config() -> UniversityConfig:
    return UniversityConfig(
        notes_per_synthesis=100,
        priority_interval=100,
        relevance_threshold=4.0,
        max_links_per_article=5,
    )


FAKE_ARTICLE = Article(
    title="Research Article",
    url="https://example.com/research",
    text="Full article text about the research topic.",
    source_name="fake",
)

FAKE_ARTICLE_2 = Article(
    title="Second Article",
    url="https://example.com/second",
    text="Full text about the second research topic.",
    source_name="fake",
)


# ══════════════════════════════════════════════════════════════════════════════
# University construction
# ══════════════════════════════════════════════════════════════════════════════


class TestUniversityConstruction:
    def test_requires_at_least_one_source(self):
        with pytest.raises(ValueError):
            University(_FakeClient(["{}"]), sources=[])

    def test_spawn_scholar_returns_fresh_instances(self):
        uni = University(_FakeClient(["{}"]), [_EmptySource()], _minimal_config())
        s1 = uni.spawn_scholar()
        s2 = uni.spawn_scholar()
        assert s1 is not s2

    def test_stores_client_and_sources(self):
        client = _FakeClient(["{}"])
        src = _EmptySource()
        uni = University(client, [src], _minimal_config())
        assert uni.client is client
        assert uni.sources == [src]


# ══════════════════════════════════════════════════════════════════════════════
# University.study
# ══════════════════════════════════════════════════════════════════════════════


class TestUniversityStudy:
    def test_returns_complete_session_with_notes(self, isolated_university_root):
        client = _FakeClient([_note_resp(), "Final synthesis."])
        src = _FakeSource([FAKE_ARTICLE])
        uni = University(client, [src], _minimal_config())
        session = uni.study(
            "Research topic",
            budget_hours=1.0,
            save_dir=isolated_university_root,
            seed_article_ids=["fake:Research Article"],
        )
        assert session.status == "complete"
        assert len(session.notes) == 1
        assert session.synthesis == "Final synthesis."

    def test_session_persisted_to_disk(self, isolated_university_root):
        client = _FakeClient([_note_resp(), "synthesis"])
        src = _FakeSource([FAKE_ARTICLE])
        uni = University(client, [src], _minimal_config())
        session = uni.study(
            "Test topic",
            budget_hours=1.0,
            save_dir=isolated_university_root,
            seed_article_ids=["fake:Research Article"],
        )
        assert (isolated_university_root / session.session_id / "session.json").exists()

    def test_study_logs_to_catalog(self, isolated_university_root):
        cat = Catalog(db_path=":memory:")
        client = _FakeClient([_note_resp(), "synthesis"])
        src = _FakeSource([FAKE_ARTICLE])
        uni = University(client, [src], _minimal_config(), catalog=cat)
        session = uni.study(
            "Test topic",
            budget_hours=1.0,
            save_dir=isolated_university_root,
            seed_article_ids=["fake:Research Article"],
        )
        assert session.catalog_study_id is not None

    def test_study_without_catalog_leaves_study_id_none(self, isolated_university_root):
        client = _FakeClient([_note_resp(), "synthesis"])
        src = _FakeSource([FAKE_ARTICLE])
        uni = University(client, [src], _minimal_config())
        session = uni.study(
            "Test topic",
            budget_hours=1.0,
            save_dir=isolated_university_root,
            seed_article_ids=["fake:Research Article"],
        )
        assert session.catalog_study_id is None

    def test_empty_session_skips_catalog_log(self, isolated_university_root):
        """No notes + no synthesis → catalog not called even with catalog present."""
        cat = Catalog(db_path=":memory:")
        client = _FakeClient(['{"queries": []}'])  # no articles queued after plan
        src = _EmptySource()
        uni = University(client, [src], _minimal_config(), catalog=cat)
        session = uni.study(
            "Topic",
            budget_hours=1.0,
            save_dir=isolated_university_root,
            seed_article_ids=[],
        )
        # Empty study — catalog_study_id stays None
        assert session.catalog_study_id is None

    def test_study_with_run_id_passes_run_id_to_catalog(self, isolated_university_root):
        """If University has a run_id, log_study is called with that run_id."""
        cat = Catalog(db_path=":memory:")
        run_id = cat.log_run(content_type="podcast", title="Test show")
        client = _FakeClient([_note_resp(), "synthesis"])
        src = _FakeSource([FAKE_ARTICLE])
        uni = University(
            client,
            [src],
            _minimal_config(),
            catalog=cat,
            run_id=run_id,
        )
        session = uni.study(
            "topic",
            budget_hours=1.0,
            save_dir=isolated_university_root,
            seed_article_ids=["fake:Research Article"],
        )
        assert session.catalog_study_id is not None
        studies = cat.get_studies(run_id)
        assert any(s.study_id == session.catalog_study_id for s in studies)


# ══════════════════════════════════════════════════════════════════════════════
# University.study_many
# ══════════════════════════════════════════════════════════════════════════════


class TestUniversityStudyMany:
    def test_returns_one_session_per_non_empty_topic(self, isolated_university_root):
        # With _EmptySource, each study does: _plan_initial_queries (1 LLM call),
        # then queue is empty → status complete. No notes → no finalize call.
        client = _FakeClient([
            '{"queries": ["q1"]}',  # plan for Topic A
            '{"queries": ["q2"]}',  # plan for Topic B
        ])
        src = _EmptySource()
        uni = University(client, [src], _minimal_config())
        sessions = uni.study_many(
            ["Topic A", "Topic B"],
            budget_hours_each=1.0,
            save_dir=isolated_university_root,
        )
        assert len(sessions) == 2
        assert sessions[0].topic == "Topic A"
        assert sessions[1].topic == "Topic B"

    def test_skips_empty_topic_strings(self, isolated_university_root):
        client = _FakeClient(['{"queries": []}'])
        src = _EmptySource()
        uni = University(client, [src], _minimal_config())
        sessions = uni.study_many(
            ["", "Topic A", "  "],
            budget_hours_each=1.0,
            save_dir=isolated_university_root,
        )
        assert len(sessions) == 1
        assert sessions[0].topic == "Topic A"

    def test_all_empty_topics_returns_empty_list(self, isolated_university_root):
        client = _FakeClient(["{}"])
        src = _EmptySource()
        uni = University(client, [src], _minimal_config())
        sessions = uni.study_many(
            ["", "  "],
            budget_hours_each=1.0,
            save_dir=isolated_university_root,
        )
        assert sessions == []

    def test_each_session_has_complete_status(self, isolated_university_root):
        client = _FakeClient([
            '{"queries": []}',
            '{"queries": []}',
        ])
        src = _EmptySource()
        uni = University(client, [src], _minimal_config())
        sessions = uni.study_many(
            ["A", "B"],
            budget_hours_each=1.0,
            save_dir=isolated_university_root,
        )
        for s in sessions:
            assert s.status == "complete"


# ══════════════════════════════════════════════════════════════════════════════
# University.cross_synthesize
# ══════════════════════════════════════════════════════════════════════════════


def _session_with_synthesis(topic: str, synthesis: str) -> ScholarSession:
    s = ScholarSession(topic=topic, status="complete")
    s.synthesis = synthesis
    return s


class TestUniversityCrossSynthesize:
    def test_merges_two_sessions(self):
        client = _FakeClient(["Merged synthesis document."])
        uni = University(client, [_EmptySource()], _minimal_config())
        s1 = _session_with_synthesis("Topic A", "Synthesis A about foobar.")
        s2 = _session_with_synthesis("Topic B", "Synthesis B about bazqux.")
        result = uni.cross_synthesize([s1, s2])
        assert result == "Merged synthesis document."

    def test_empty_sessions_list_returns_empty_string(self):
        client = _FakeClient(["should not be called"])
        uni = University(client, [_EmptySource()], _minimal_config())
        assert uni.cross_synthesize([]) == ""

    def test_sessions_with_no_content_skipped(self):
        client = _FakeClient(["should not be called"])
        uni = University(client, [_EmptySource()], _minimal_config())
        s = ScholarSession(topic="Empty", status="complete")
        # No synthesis, no notes → usable list is empty → return ""
        assert uni.cross_synthesize([s]) == ""

    def test_custom_directive_passed_to_llm(self):
        """cross_synthesize sends the directive to the LLM — just verify
        it doesn't raise and returns the fake response."""
        client = _FakeClient(["Directive-influenced synthesis."])
        uni = University(client, [_EmptySource()], _minimal_config())
        s = _session_with_synthesis("Topic", "Some synthesis.")
        result = uni.cross_synthesize([s], directive="Frame as a debate brief.")
        assert result == "Directive-influenced synthesis."

    def test_strips_think_blocks_from_merged_output(self):
        client = _FakeClient(["<think>ignore</think>Clean output."])
        uni = University(client, [_EmptySource()], _minimal_config())
        s = _session_with_synthesis("Topic", "Synthesis content.")
        result = uni.cross_synthesize([s])
        assert "<think>" not in result
        assert result == "Clean output."


# ══════════════════════════════════════════════════════════════════════════════
# UniversitySearchProvider integration
# ══════════════════════════════════════════════════════════════════════════════


class TestSearchProviderIntegration:
    def test_search_provider_on_completed_session(self, isolated_university_root):
        """Run a full study → open the same session via SearchProvider → search."""
        client = _FakeClient([_note_resp(score=9.0), "Final synthesis."])
        src = _FakeSource([FAKE_ARTICLE])
        uni = University(client, [src], _minimal_config())
        session = uni.study(
            "Research topic",
            budget_hours=1.0,
            save_dir=isolated_university_root,
            seed_article_ids=["fake:Research Article"],
        )

        provider = UniversitySearchProvider(session.session_id)
        results = provider.search("research topic")
        assert len(results) >= 1
        titles = [r.title for r in results]
        assert any(t == "Research Article" or "[Synthesis]" in t for t in titles)

    def test_search_provider_num_results_respected(self, isolated_university_root):
        s = ScholarSession(topic="Many notes")
        s.notes = [
            ScholarNote(
                article_title=f"Note {i}",
                source_url=f"https://example.com/{i}",
                source_name="fake",
                summary=f"Summary {i}.",
                relevance_score=float(i),
            )
            for i in range(10)
        ]
        s.status = "complete"
        s.save(isolated_university_root)

        provider = UniversitySearchProvider(s.session_id)
        for n in (1, 3, 5):
            results = provider.search("note", num_results=n)
            assert len(results) <= n

    def test_search_provider_returns_highest_scored_note_first(
        self, isolated_university_root
    ):
        s = ScholarSession(topic="Ranking test")
        s.notes = [
            ScholarNote(
                article_title="Low relevance",
                source_url="https://example.com/low",
                source_name="fake",
                summary="Barely relevant.",
                relevance_score=2.0,
            ),
            ScholarNote(
                article_title="High relevance",
                source_url="https://example.com/high",
                source_name="fake",
                summary="Very relevant indeed.",
                relevance_score=9.0,
            ),
        ]
        s.status = "complete"
        s.save(isolated_university_root)

        provider = UniversitySearchProvider(s.session_id)
        results = provider.search("anything", num_results=5)
        assert results[0].title == "High relevance"

"""Tests for the inter-Scholar colloquium loop in University.seminar().

Covers:
    - ColloquiumExchange / ColloquiumRound serialisation round-trips
    - ScholarSession.colloquium_rounds persists through save/load
    - Scholar.study/extend max_articles parameter caps the loop
    - University.seminar() with colloquium disabled falls back to study_many
    - University.seminar() injects peer-suggested leads into queues
    - University._run_colloquium_round() skips when <2 syntheses available
    - LLM failure during a colloquium exchange is non-fatal
"""

from __future__ import annotations

import json

import pytest

import vox_romana.core.paths as _paths
import vox_romana.university.session as _sm
from vox_romana.university.config import UniversityConfig
from vox_romana.university.session import (
    ColloquiumExchange,
    ColloquiumRound,
    ScholarNote,
    ScholarSession,
)
from vox_romana.university.sources import Article, KnowledgeSource
from vox_romana.university.university import University


# ══════════════════════════════════════════════════════════════════════════════
# Fakes & fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def isolated_university_root(tmp_path, monkeypatch):
    monkeypatch.setattr(_paths, "UNIVERSITY_ROOT", tmp_path)
    monkeypatch.setattr(_sm, "UNIVERSITY_ROOT", tmp_path)
    return tmp_path


class _ScriptedClient:
    """Returns canned responses by call index; repeats last on overflow."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        idx = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return self._responses[idx]


class _ExplodingClient:
    """Always raises — used to verify graceful degradation."""

    def chat(self, messages, **kwargs) -> str:
        raise RuntimeError("LLM unavailable")


class _FakeSource(KnowledgeSource):
    name = "fake"

    def __init__(self, by_query: dict[str, list[Article]] | None = None):
        self._by_query = dict(by_query or {})
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, n: int = 10) -> list[Article]:
        self.search_calls.append((query, n))
        # Match a query if any registered key is a case-insensitive substring.
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
        "key_facts": ["Fact 1", "Fact 2"],
        "summary": "An article summary.",
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
# Dataclass serialisation
# ══════════════════════════════════════════════════════════════════════════════


class TestColloquiumDataclasses:
    def test_exchange_to_dict_round_trip(self):
        ex = ColloquiumExchange(
            from_topic="Roman engineering",
            to_topic="Greek philosophy",
            snippet="Romans built aqueducts using arches.",
            comment="The arch is a structural innovation worth tracing back.",
            suggested_queries=("Greek arch architecture", "Hellenistic engineering"),
            timestamp="2026-04-30T12:00:00+00:00",
        )
        roundtripped = ColloquiumExchange.from_dict(ex.to_dict())
        assert roundtripped == ex

    def test_round_to_dict_round_trip(self):
        ex = ColloquiumExchange(
            from_topic="A", to_topic="B",
            suggested_queries=("q1",),
        )
        r = ColloquiumRound(round_n=2, started_at="2026-04-30T12:00:00+00:00")
        r.exchanges.append(ex)
        roundtripped = ColloquiumRound.from_dict(r.to_dict())
        assert roundtripped.round_n == 2
        assert roundtripped.started_at == "2026-04-30T12:00:00+00:00"
        assert len(roundtripped.exchanges) == 1
        assert roundtripped.exchanges[0] == ex

    def test_session_persists_colloquium_rounds(self, isolated_university_root):
        ex = ColloquiumExchange(
            from_topic="A", to_topic="B",
            suggested_queries=("q1", "q2"),
        )
        r = ColloquiumRound(round_n=1)
        r.exchanges.append(ex)
        s = ScholarSession(topic="B")
        s.colloquium_rounds.append(r)
        s.save(isolated_university_root)

        loaded = ScholarSession.from_dict(
            json.loads((isolated_university_root / s.session_id / "session.json").read_text())
        )
        assert len(loaded.colloquium_rounds) == 1
        assert loaded.colloquium_rounds[0].round_n == 1
        assert loaded.colloquium_rounds[0].exchanges[0].suggested_queries == ("q1", "q2")

    def test_session_default_colloquium_rounds_empty(self):
        s = ScholarSession(topic="X")
        assert s.colloquium_rounds == []
        # to_dict / from_dict round-trips an empty list.
        assert ScholarSession.from_dict(s.to_dict()).colloquium_rounds == []


# ══════════════════════════════════════════════════════════════════════════════
# Scholar.max_articles bounded loop
# ══════════════════════════════════════════════════════════════════════════════


class TestScholarMaxArticles:
    def test_max_articles_caps_iteration_and_keeps_session_active(
        self, isolated_university_root
    ):
        from vox_romana.university.scholar import Scholar

        articles = [
            Article(title=f"A{i}", url=f"u{i}", text=f"text {i}", source_name="fake")
            for i in range(5)
        ]
        # Each call returns the full set; Scholar dedups by qid.
        src = _FakeSource({"plan": articles})
        client = _ScriptedClient(
            [
                # _plan_initial_queries → JSON queries
                json.dumps({"queries": ["plan query"]}),
                # 5 article reads, but max_articles=2 stops after 2 —
                # only need 2 note-extraction responses.
                _note_resp(),
                _note_resp(),
            ]
        )
        scholar = Scholar(client, [src], _minimal_config())
        session = scholar.study("topic", budget_hours=10.0, max_articles=2)
        assert len(session.articles_read) == 2
        assert session.status == "active"  # capped, not complete
        assert session.articles_queued  # still has work left


# ══════════════════════════════════════════════════════════════════════════════
# University.seminar — colloquium disabled fallback
# ══════════════════════════════════════════════════════════════════════════════


class TestSeminarFallback:
    def test_no_colloquium_falls_back_to_study_many(self, isolated_university_root):
        article = Article(
            title="X", url="u", text="t", source_name="fake",
        )
        src = _FakeSource({"q": [article]})
        client = _ScriptedClient(
            [
                json.dumps({"queries": ["q"]}),  # plan topic 1
                _note_resp(),                     # read article topic 1
                json.dumps({"queries": ["q"]}),  # plan topic 2
                _note_resp(),                     # read article topic 2
            ]
        )
        uni = University(client, [src], _minimal_config())
        sessions = uni.seminar(
            ["Topic A", "Topic B"],
            budget_hours_each=10.0,
            colloquium_enabled=False,
        )
        assert len(sessions) == 2
        # Fallback path = no colloquium rounds recorded.
        for s in sessions:
            assert s.colloquium_rounds == []

    def test_single_topic_falls_back_even_with_colloquium_enabled(
        self, isolated_university_root
    ):
        """A 1-Scholar 'seminar' has no peers — colloquium is a no-op."""
        article = Article(title="X", url="u", text="t", source_name="fake")
        src = _FakeSource({"q": [article]})
        client = _ScriptedClient(
            [json.dumps({"queries": ["q"]}), _note_resp()]
        )
        uni = University(client, [src], _minimal_config())
        sessions = uni.seminar(
            ["Solo topic"],
            budget_hours_each=10.0,
            colloquium_enabled=True,
        )
        assert len(sessions) == 1
        assert sessions[0].colloquium_rounds == []

    def test_empty_topics_returns_empty_list(self, isolated_university_root):
        uni = University(
            _ScriptedClient(["never called"]), [_FakeSource()], _minimal_config()
        )
        assert uni.seminar(["", "  "]) == []


# ══════════════════════════════════════════════════════════════════════════════
# University._run_colloquium_round
# ══════════════════════════════════════════════════════════════════════════════


class TestRunColloquiumRound:
    def test_skipped_when_fewer_than_two_syntheses(self):
        client = _ExplodingClient()  # would raise if called
        uni = University(client, [_FakeSource()], _minimal_config())
        # Only one session has synthesis; the other is empty.
        s1 = ScholarSession(topic="A", synthesis="some content")
        s2 = ScholarSession(topic="B", synthesis="")
        added = uni._run_colloquium_round([s1, s2], round_n=1)
        assert added == 0
        assert s1.colloquium_rounds == []
        assert s2.colloquium_rounds == []

    def test_injects_leads_into_peer_queue(self, isolated_university_root):
        """Two sessions, each with synthesis → LLM proposes 1 lead per pair."""
        peer_article = Article(
            title="PeerArticle",
            url="https://example.com/peer",
            text="text",
            source_name="fake",
        )
        # The fake source returns peer_article for any query containing 'peer'.
        src = _FakeSource({"peer": [peer_article]})

        # Each pair (A->B, B->A) gets one LLM call.  Both return the same
        # JSON for simplicity.
        exchange_resp = json.dumps({
            "comment": "These two topics share a common thread.",
            "queries": ["peer angle one"],
        })
        client = _ScriptedClient([exchange_resp, exchange_resp])

        uni = University(client, [src], _minimal_config())
        s1 = ScholarSession(topic="Topic A", synthesis="A's findings so far.")
        s2 = ScholarSession(topic="Topic B", synthesis="B's findings so far.")
        added = uni._run_colloquium_round([s1, s2], round_n=1)
        assert added == 2  # 1 qid added to each session
        # Each session received exactly one exchange from its peer.
        assert len(s1.colloquium_rounds) == 1
        assert len(s2.colloquium_rounds) == 1
        assert len(s1.colloquium_rounds[0].exchanges) == 1
        assert s1.colloquium_rounds[0].exchanges[0].from_topic == "Topic B"
        assert s2.colloquium_rounds[0].exchanges[0].from_topic == "Topic A"
        # The suggested queries become qids in the peer queue.
        assert any("PeerArticle" in q for q in s1.articles_queued)
        assert any("PeerArticle" in q for q in s2.articles_queued)

    def test_llm_failure_does_not_corrupt_session(self):
        """A LLM exception during exchange is logged and the round continues."""
        client = _ExplodingClient()
        uni = University(client, [_FakeSource()], _minimal_config())
        s1 = ScholarSession(topic="A", synthesis="AAA")
        s2 = ScholarSession(topic="B", synthesis="BBB")
        # Should not raise.
        added = uni._run_colloquium_round([s1, s2], round_n=1)
        assert added == 0
        # Sessions remain intact.
        assert s1.synthesis == "AAA"
        assert s2.synthesis == "BBB"
        assert s1.articles_queued == []
        assert s2.articles_queued == []

    def test_does_not_re_enqueue_already_seen_articles(self):
        """Peer-suggested leads that match articles_read or articles_queued are skipped."""
        peer_article = Article(
            title="Already", url="u", text="t", source_name="fake",
        )
        src = _FakeSource({"x": [peer_article]})
        exchange_resp = json.dumps({"comment": "c", "queries": ["x"]})
        client = _ScriptedClient([exchange_resp, exchange_resp])

        uni = University(client, [src], _minimal_config())
        s1 = ScholarSession(
            topic="A",
            synthesis="aa",
            articles_read=["fake:Already"],  # already read
        )
        s2 = ScholarSession(
            topic="B",
            synthesis="bb",
            articles_queued=["fake:Already"],  # already queued
        )
        added = uni._run_colloquium_round([s1, s2], round_n=1)
        # Both peers would have suggested the same qid, but it's already known.
        assert added == 0


# ══════════════════════════════════════════════════════════════════════════════
# University.seminar — full end-to-end with colloquium ON
# ══════════════════════════════════════════════════════════════════════════════


class TestSeminarEndToEnd:
    def test_seminar_records_colloquium_rounds_when_enabled(
        self, isolated_university_root
    ):
        """End-to-end: 2 Scholars + bounded rounds + colloquium exchanges.

        Uses a content-aware fake client that returns valid JSON for any
        prompt that needs it, plain prose otherwise — this is robust to
        the exact ordering of LLM calls inside the loop.
        """

        class _SmartClient:
            def __init__(self):
                self.exchange_calls = 0
                self.note_calls = 0

            def chat(self, messages, **kwargs) -> str:
                # Inspect the last user message to decide what to return.
                last_user = ""
                for m in messages:
                    if m.get("role") == "user":
                        last_user = m.get("content", "")
                # Colloquium exchange — system prompt mentions "research seminar".
                sys_text = next(
                    (m.get("content", "") for m in messages
                     if m.get("role") == "system"), ""
                )
                if "research seminar" in sys_text:
                    self.exchange_calls += 1
                    return json.dumps({
                        "comment": "shared theme",
                        "queries": ["peer"],
                    })
                # Note-extraction expects {"key_facts":..., "summary":..., "relevance_score":...}
                if "key_facts" in last_user or "ARTICLE TEXT" in last_user.upper():
                    self.note_calls += 1
                    return _note_resp()
                # Initial-queries planner expects {"queries": [...]}
                if "queries" in last_user.lower() and "json" in last_user.lower():
                    return json.dumps({"queries": ["q"]})
                # Anything else (synthesis, finalize, prioritise) — plain prose.
                return "Synthesis prose."

        articles = [
            Article(title=f"A{i}", url=f"u{i}", text="t", source_name="fake")
            for i in range(4)
        ]
        src = _FakeSource({"q": articles, "peer": articles})
        client = _SmartClient()
        cfg = _minimal_config(colloquium_interval=1, notes_per_synthesis=1)
        uni = University(client, [src], cfg)
        sessions = uni.seminar(
            ["Topic A", "Topic B"],
            budget_hours_each=10.0,
            colloquium_enabled=True,
            colloquium_interval=1,
        )
        assert len(sessions) == 2
        # At least one colloquium happened — exchange calls were made.
        assert client.exchange_calls >= 2  # one per (A,B) and (B,A)
        # Every session that received exchanges has rounds recorded.
        total_rounds = sum(len(s.colloquium_rounds) for s in sessions)
        assert total_rounds >= 2

    def test_seminar_with_no_synthesis_does_not_crash(
        self, isolated_university_root
    ):
        """If no Scholar has synthesis when the round triggers, skip cleanly."""
        # Source returns nothing → no notes → no synthesis.
        empty_src = _FakeSource({})
        client = _ScriptedClient(
            [
                json.dumps({"queries": ["x"]}),  # plan A — no hits
                json.dumps({"queries": ["x"]}),  # plan B — no hits
            ]
        )
        uni = University(
            client, [empty_src], _minimal_config(colloquium_interval=1),
        )
        sessions = uni.seminar(
            ["A", "B"],
            budget_hours_each=1.0,
            colloquium_enabled=True,
            colloquium_interval=1,
        )
        # Both sessions completed (empty queues) with no exchanges.
        assert len(sessions) == 2
        for s in sessions:
            assert s.colloquium_rounds == []

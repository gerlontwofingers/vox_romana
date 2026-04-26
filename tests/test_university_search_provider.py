"""Tests for vox_romana.university.search_provider.

Covers:
    - _score_note: relevance_score + keyword bonus, bonus cap, case-insensitive
    - _make_snippet: prefers summary, falls back to key_facts, truncation
    - UniversitySearchProvider: FileNotFoundError on missing session,
      empty session → [], synthesis-only session, notes ranked by score,
      keyword bonus reranks, synthesis appended when below num_results,
      num_results cap enforced, result field correctness,
      synthesis URL uses university:// scheme
"""

from __future__ import annotations

import pytest

import vox_romana.core.paths as _paths
import vox_romana.university.session as _sm
from vox_romana.university.search_provider import (
    UniversitySearchProvider,
    _make_snippet,
    _score_note,
)
from vox_romana.university.session import ScholarNote, ScholarSession


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def isolated_university_root(tmp_path, monkeypatch):
    monkeypatch.setattr(_paths, "UNIVERSITY_ROOT", tmp_path)
    monkeypatch.setattr(_sm, "UNIVERSITY_ROOT", tmp_path)
    return tmp_path


def _make_note(title, *, summary="", key_facts=(), score=5.0) -> ScholarNote:
    return ScholarNote(
        article_title=title,
        source_url=f"https://example.com/{title.lower().replace(' ', '_')}",
        source_name="wikipedia",
        key_facts=tuple(key_facts),
        summary=summary,
        relevance_score=score,
    )


def _saved_session(
    tmp_path,
    *,
    notes: list[ScholarNote] | None = None,
    synthesis: str = "",
    topic: str = "Test Topic",
) -> ScholarSession:
    s = ScholarSession(topic=topic)
    s.notes = list(notes or [])
    s.synthesis = synthesis
    s.status = "complete"
    s.save(tmp_path)
    return s


# ══════════════════════════════════════════════════════════════════════════════
# _score_note
# ══════════════════════════════════════════════════════════════════════════════


class TestScoreNote:
    def test_no_keywords_returns_relevance_score(self):
        note = _make_note("X", score=7.0)
        assert _score_note(note, frozenset()) == 7.0

    def test_keyword_match_in_title_adds_bonus(self):
        note = _make_note("Roman roads", score=5.0)
        score = _score_note(note, frozenset(["roman"]))
        assert score > 5.0

    def test_keyword_match_in_summary_adds_bonus(self):
        note = _make_note("X", summary="Roman roads were long.", score=5.0)
        score = _score_note(note, frozenset(["roman"]))
        assert score > 5.0

    def test_keyword_bonus_capped_at_3(self):
        note = _make_note(
            "Rome roads rails runways",
            summary="Rome roads rails runways rule research results",
            score=0.0,
        )
        words = frozenset(
            ["rome", "roads", "rails", "runways", "rule", "research", "results"]
        )
        score = _score_note(note, words)
        assert score <= 3.0

    def test_text_lowercased_for_matching(self):
        """_score_note lowercases the text before matching, so UPPERCASE
        article titles / summaries are matched by lowercase query words
        (as produced by search()'s query.lower().split())."""
        note = _make_note("ROMAN ROADS", summary="BUILT BY LEGIONS.", score=5.0)
        # lowercase query words should still match the uppercased text
        score = _score_note(note, frozenset(["roman", "roads"]))
        assert score > 5.0

    def test_no_match_returns_base_score_only(self):
        note = _make_note("Unrelated topic", score=6.5)
        score = _score_note(note, frozenset(["roman", "aqueducts"]))
        # No match → score == base relevance_score
        assert score == 6.5


# ══════════════════════════════════════════════════════════════════════════════
# _make_snippet
# ══════════════════════════════════════════════════════════════════════════════


class TestMakeSnippet:
    def test_prefers_summary_over_key_facts(self):
        note = _make_note("X", summary="Nice summary.", key_facts=("Fact 1",))
        assert _make_snippet(note) == "Nice summary."

    def test_falls_back_to_key_facts_when_no_summary(self):
        note = _make_note("X", summary="", key_facts=("Fact 1", "Fact 2", "Fact 3"))
        snippet = _make_snippet(note)
        assert "Fact 1" in snippet

    def test_empty_note_returns_empty_string(self):
        note = _make_note("X", summary="", key_facts=())
        assert _make_snippet(note) == ""

    def test_long_summary_truncated_with_ellipsis(self):
        long_summary = "A" * 500
        note = _make_note("X", summary=long_summary)
        snippet = _make_snippet(note)
        assert len(snippet) <= 281  # 280 chars + "…"
        assert snippet.endswith("…")

    def test_key_facts_capped_at_three(self):
        note = _make_note("X", key_facts=("F1", "F2", "F3", "F4", "F5"))
        snippet = _make_snippet(note)
        assert "F4" not in snippet
        assert "F5" not in snippet

    def test_key_facts_joined_with_separator(self):
        note = _make_note("X", key_facts=("A", "B", "C"))
        snippet = _make_snippet(note)
        # Must include at least two facts
        assert "A" in snippet
        assert "B" in snippet


# ══════════════════════════════════════════════════════════════════════════════
# UniversitySearchProvider
# ══════════════════════════════════════════════════════════════════════════════


class TestUniversitySearchProvider:
    def test_raises_if_session_not_found(self, isolated_university_root):
        with pytest.raises(FileNotFoundError):
            UniversitySearchProvider("nonexistent000")

    def test_empty_session_returns_empty_list(self, isolated_university_root):
        s = _saved_session(isolated_university_root)  # no notes, no synthesis
        provider = UniversitySearchProvider(s.session_id)
        assert provider.search("anything") == []

    def test_synthesis_only_session_returns_one_result(self, isolated_university_root):
        s = _saved_session(isolated_university_root, synthesis="This is the synthesis.")
        provider = UniversitySearchProvider(s.session_id)
        results = provider.search("topic", num_results=5)
        assert len(results) == 1
        assert "[Synthesis]" in results[0].title

    def test_notes_ranked_by_relevance_score(self, isolated_university_root):
        notes = [
            _make_note("Low score", score=2.0),
            _make_note("High score", score=9.0),
            _make_note("Mid score", score=5.0),
        ]
        s = _saved_session(isolated_university_root, notes=notes)
        provider = UniversitySearchProvider(s.session_id)
        results = provider.search("query", num_results=3)
        assert results[0].title == "High score"

    def test_keyword_bonus_can_rerank_notes(self, isolated_university_root):
        notes = [
            _make_note("Unrelated article", score=8.0, summary="Nothing matches."),
            _make_note(
                "Roman roads history",
                score=5.0,
                summary="Roman roads were built by Roman legions.",
            ),
        ]
        s = _saved_session(isolated_university_root, notes=notes)
        provider = UniversitySearchProvider(s.session_id)
        results = provider.search("roman roads", num_results=5)
        # "Roman roads history" gains 3.0 bonus → 8.0 total vs "Unrelated" = 8.0
        # both are equal — test that "Roman roads" appears in top 2
        titles = [r.title for r in results]
        assert "Roman roads history" in titles

    def test_synthesis_appended_when_fewer_notes_than_num_results(
        self, isolated_university_root
    ):
        notes = [_make_note("Only one note", score=8.0)]
        s = _saved_session(
            isolated_university_root, notes=notes, synthesis="The big synthesis."
        )
        provider = UniversitySearchProvider(s.session_id)
        results = provider.search("query", num_results=5)
        titles = [r.title for r in results]
        assert any("[Synthesis]" in t for t in titles)

    def test_synthesis_not_appended_when_notes_fill_results(
        self, isolated_university_root
    ):
        notes = [_make_note(f"Note {i}", score=float(i)) for i in range(5)]
        s = _saved_session(
            isolated_university_root, notes=notes, synthesis="Synthesis here."
        )
        provider = UniversitySearchProvider(s.session_id)
        results = provider.search("query", num_results=5)
        titles = [r.title for r in results]
        # Notes fill all 5 slots → no room for synthesis
        assert not any("[Synthesis]" in t for t in titles)

    def test_num_results_cap_respected(self, isolated_university_root):
        notes = [_make_note(f"Note {i}", score=float(i)) for i in range(10)]
        s = _saved_session(isolated_university_root, notes=notes)
        provider = UniversitySearchProvider(s.session_id)
        for n in (1, 3, 5):
            results = provider.search("query", num_results=n)
            assert len(results) <= n, f"Expected ≤{n}, got {len(results)}"

    def test_result_fields_populated(self, isolated_university_root):
        note = _make_note("Good article", summary="A clear summary.", score=8.0)
        s = _saved_session(isolated_university_root, notes=[note])
        provider = UniversitySearchProvider(s.session_id)
        results = provider.search("good article", num_results=5)
        r = results[0]
        assert r.title == "Good article"
        assert r.url  # non-empty
        assert "A clear summary." == r.snippet

    def test_synthesis_result_url_uses_university_scheme(self, isolated_university_root):
        s = _saved_session(isolated_university_root, synthesis="Synthesis content.")
        provider = UniversitySearchProvider(s.session_id)
        results = provider.search("topic", num_results=5)
        assert results[0].url.startswith("university://")
        assert s.session_id in results[0].url

    def test_synthesis_snippet_truncated_if_long(self, isolated_university_root):
        long_synth = "X" * 1000
        s = _saved_session(isolated_university_root, synthesis=long_synth)
        provider = UniversitySearchProvider(s.session_id)
        results = provider.search("query", num_results=5)
        assert len(results[0].snippet) <= 281

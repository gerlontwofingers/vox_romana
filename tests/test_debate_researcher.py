"""Tests for the search providers and debate researcher.

This file tests:
    1. SearchResult construction and defaults
    2. GoogleSearchProvider credential validation
    3. DebateResearcher query formulation parsing (canned JSON)
    4. DebateResearcher result compilation (canned search results)
    5. DebateResearcher full run() pipeline (mocked search + LLM)
    6. DebateResearcher edge cases (empty results, bad JSON, search failures)
    7. Bootstrap integration with auto_research=True

All tests use mocked search providers and LLM clients — no real API calls.
The research agent's parsing logic is deterministic and fast to test.

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_debate_researcher.py -v
"""

import json
from unittest.mock import Mock, patch, call

import pytest

from vox_romana.core.search import BaseSearchProvider, GoogleSearchProvider, SearchResult
from vox_romana.shows.debates.researcher import (
    DebateResearcher,
    DEFAULT_RESULTS_PER_QUERY,
    MAX_QUERIES,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

class MockSearchProvider(BaseSearchProvider):
    """A search provider that always returns canned results.

    Think of it like a librarian who already has the answer cards
    pre-sorted on their desk — no actual searching involved.
    """

    def __init__(self, results: list[SearchResult] | None = None):
        self._results = results or [
            SearchResult(
                title="Finland's UBI Pilot",
                url="https://example.com/finland-ubi",
                snippet="Finland's 2017-2018 UBI pilot gave 2,000 people €560/month.",
            ),
            SearchResult(
                title="Stockton SEED Program",
                url="https://example.com/stockton",
                snippet="The SEED program showed a 12% increase in full-time employment.",
            ),
            SearchResult(
                title="UBI Cost Estimates",
                url="https://example.com/ubi-cost",
                snippet="A nationwide UBI at $1,000/month would cost ~$3.1 trillion annually.",
            ),
        ]

    def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        return self._results[:num_results]


class FailingSearchProvider(BaseSearchProvider):
    """A search provider that always raises an exception.

    Used to verify that the researcher handles search failures gracefully
    (keeps going with partial results instead of crashing).
    """

    def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        raise ConnectionError("Network is down!")


# ═══════════════════════════════════════════════════════════════════════════
# SEARCH RESULT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchResult:
    """Test the SearchResult dataclass."""

    def test_defaults(self):
        """All fields default to empty string."""
        r = SearchResult()
        assert r.title == ""
        assert r.url == ""
        assert r.snippet == ""

    def test_construction(self):
        """Fields set correctly when provided."""
        r = SearchResult(title="Test", url="https://example.com", snippet="A snippet")
        assert r.title == "Test"
        assert r.url == "https://example.com"
        assert r.snippet == "A snippet"


# ═══════════════════════════════════════════════════════════════════════════
# GOOGLE SEARCH PROVIDER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestGoogleSearchProvider:
    """Test Google provider credential validation."""

    def test_missing_api_key_raises(self):
        """Must provide an API key via constructor or env var."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Google API key required"):
                GoogleSearchProvider()

    def test_missing_cse_id_raises(self):
        """Must provide a CSE ID via constructor or env var."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Google CSE ID required"):
                GoogleSearchProvider(api_key="fake-key")

    def test_accepts_constructor_args(self):
        """Constructor args bypass env vars."""
        provider = GoogleSearchProvider(api_key="test-key", cse_id="test-cse")
        assert provider._api_key == "test-key"
        assert provider._cse_id == "test-cse"

    def test_reads_env_vars(self):
        """Falls back to environment variables."""
        env = {"GOOGLE_API_KEY": "env-key", "GOOGLE_CSE_ID": "env-cse"}
        with patch.dict("os.environ", env, clear=True):
            provider = GoogleSearchProvider()
            assert provider._api_key == "env-key"
            assert provider._cse_id == "env-cse"


# ═══════════════════════════════════════════════════════════════════════════
# QUERY PARSING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestParseQueries:
    """Test the query formulation parsing in isolation.

    These tests exercise parse_queries() with canned JSON strings —
    no LLM calls involved. This is where we verify the JSON handling,
    markdown fence stripping, and max-query capping.
    """

    @pytest.fixture
    def researcher(self):
        """A researcher with a mock client and mock search provider."""
        return DebateResearcher(
            client=Mock(),
            search_provider=MockSearchProvider(),
        )

    def test_simple_json_array(self, researcher):
        """Happy path: clean JSON array of strings."""
        raw = '["query one", "query two", "query three"]'
        result = researcher.parse_queries(raw)
        assert result == ["query one", "query two", "query three"]

    def test_markdown_fenced_json(self, researcher):
        """LLMs often wrap JSON in ```json fences — we strip them."""
        raw = '```json\n["query one", "query two"]\n```'
        result = researcher.parse_queries(raw)
        assert result == ["query one", "query two"]

    def test_bare_markdown_fences(self, researcher):
        """Fences without a language tag should also be stripped."""
        raw = '```\n["alpha", "beta"]\n```'
        result = researcher.parse_queries(raw)
        assert result == ["alpha", "beta"]

    def test_caps_at_max_queries(self, researcher):
        """More than MAX_QUERIES queries get truncated."""
        queries = [f"query {i}" for i in range(10)]
        raw = json.dumps(queries)
        result = researcher.parse_queries(raw)
        assert len(result) == MAX_QUERIES

    def test_invalid_json_raises(self, researcher):
        """Bad JSON gives a clear error message."""
        with pytest.raises(ValueError, match="Query formulator returned invalid JSON"):
            researcher.parse_queries("not valid json at all")

    def test_non_array_raises(self, researcher):
        """If the LLM returns an object instead of array, report it."""
        with pytest.raises(ValueError, match="Expected a JSON array"):
            researcher.parse_queries('{"query": "oops"}')

    def test_coerces_non_strings(self, researcher):
        """Non-string items get str()-ified (handles LLM weirdness)."""
        raw = '["ok", 42, true]'
        result = researcher.parse_queries(raw)
        assert result == ["ok", "42", "True"]


# ═══════════════════════════════════════════════════════════════════════════
# FORMULATE QUERIES TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestFormulateQueries:
    """Test formulate_queries() — verifies it calls the LLM correctly."""

    def test_calls_client_chat(self):
        """formulate_queries() sends the brief to the LLM."""
        mock_client = Mock()
        mock_client.chat.return_value = '["q1", "q2", "q3"]'

        researcher = DebateResearcher(mock_client, MockSearchProvider())
        queries = researcher.formulate_queries("debate about cats vs dogs")

        # Verify the LLM was called.
        mock_client.chat.assert_called_once()

        # Verify the brief showed up in the messages.
        call_args = mock_client.chat.call_args
        messages = call_args[1].get("messages") or call_args[0][0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "cats vs dogs" in user_msg["content"]

        assert queries == ["q1", "q2", "q3"]


# ═══════════════════════════════════════════════════════════════════════════
# COMPILE RESULTS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCompileResults:
    """Test the result compilation step."""

    def test_compiles_results_into_reference(self):
        """compile_results() sends search results to LLM and returns its output."""
        mock_client = Mock()
        mock_client.chat.return_value = "## Key Statistics\n- Finland UBI: 2,000 people..."

        researcher = DebateResearcher(mock_client, MockSearchProvider())
        results = [
            SearchResult(title="Finland UBI", url="https://x.com", snippet="2,000 people got €560"),
            SearchResult(title="SEED Program", url="https://y.com", snippet="12% employment boost"),
        ]

        compiled = researcher.compile_results("UBI debate", results)

        assert compiled == "## Key Statistics\n- Finland UBI: 2,000 people..."
        mock_client.chat.assert_called_once()

        # Verify the search snippets appear in the prompt.
        call_args = mock_client.chat.call_args
        messages = call_args[1].get("messages") or call_args[0][0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "2,000 people got €560" in user_msg["content"]
        assert "12% employment boost" in user_msg["content"]

    def test_deduplicates_identical_snippets(self):
        """Duplicate snippets from overlapping queries should be removed."""
        mock_client = Mock()
        mock_client.chat.return_value = "## Facts\n- One fact"

        researcher = DebateResearcher(mock_client, MockSearchProvider())
        results = [
            SearchResult(title="A", url="https://a.com", snippet="Same snippet"),
            SearchResult(title="B", url="https://b.com", snippet="Same snippet"),
            SearchResult(title="C", url="https://c.com", snippet="Different snippet"),
        ]

        researcher.compile_results("test", results)

        # The duplicate "Same snippet" should only appear once in the prompt.
        call_args = mock_client.chat.call_args
        messages = call_args[1].get("messages") or call_args[0][0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert user_msg["content"].count("Same snippet") == 1

    def test_empty_snippets_are_skipped(self):
        """Results with blank snippets are useless — skip them."""
        mock_client = Mock()
        mock_client.chat.return_value = "## Facts"

        researcher = DebateResearcher(mock_client, MockSearchProvider())
        results = [
            SearchResult(title="A", url="https://a.com", snippet=""),
            SearchResult(title="B", url="https://b.com", snippet="   "),
        ]

        compiled = researcher.compile_results("test", results)
        # All snippets are empty → returns empty string without calling LLM.
        assert compiled == ""

    def test_all_empty_snippets_returns_empty_no_llm_call(self):
        """If every snippet is empty, don't waste an LLM call."""
        mock_client = Mock()
        researcher = DebateResearcher(mock_client, MockSearchProvider())

        result = researcher.compile_results("test", [
            SearchResult(title="A", url="u", snippet=""),
            SearchResult(title="B", url="u", snippet="   "),
        ])

        assert result == ""
        mock_client.chat.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestResearcherRun:
    """Test the full run() pipeline with mocked everything."""

    def test_full_pipeline(self):
        """run() chains query formulation → search → compilation."""
        mock_client = Mock()
        # First chat call: query formulation returns queries.
        # Second chat call: compilation returns reference material.
        mock_client.chat.side_effect = [
            '["query 1", "query 2"]',
            "## Key Statistics\n- Fact one\n- Fact two",
        ]

        mock_search = MockSearchProvider()
        researcher = DebateResearcher(mock_client, mock_search)

        result = researcher.run("Should AI be regulated?")

        # Two LLM calls: formulate + compile.
        assert mock_client.chat.call_count == 2

        # The final output is the compiled reference material.
        assert "Key Statistics" in result
        assert "Fact one" in result

    def test_returns_empty_when_all_searches_fail(self):
        """If every search fails, return empty string (no crash)."""
        mock_client = Mock()
        mock_client.chat.return_value = '["query 1", "query 2"]'

        researcher = DebateResearcher(mock_client, FailingSearchProvider())
        result = researcher.run("topic")

        # One LLM call for query formulation, none for compilation
        # (no results to compile).
        assert mock_client.chat.call_count == 1
        assert result == ""

    def test_partial_search_failure_still_works(self):
        """If some queries fail, we still compile what we got."""
        mock_client = Mock()
        mock_client.chat.side_effect = [
            '["q1", "q2", "q3"]',
            "## Compiled from partial results",
        ]

        call_count = 0

        class FlakySearchProvider(BaseSearchProvider):
            """Alternates between success and failure."""

            def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    raise ConnectionError("Flaky!")
                return [SearchResult(title="OK", url="u", snippet="data")]

        researcher = DebateResearcher(mock_client, FlakySearchProvider())
        result = researcher.run("topic")

        # Should still get compiled output from the successful queries.
        assert result == "## Compiled from partial results"
        assert mock_client.chat.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# BOOTSTRAP INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapAutoResearch:
    """Test that auto_research wires correctly into DebateBootstrap."""

    def test_auto_research_without_search_provider_uses_llm_knowledge(self):
        """auto_research=True without a search_provider uses LLM-only research."""
        from vox_romana.shows.debates.researcher import DebateResearcher

        mock_client = Mock()
        mock_client.chat.return_value = "## Key Facts\n- Some fact"

        researcher = DebateResearcher(mock_client)  # No search provider
        result = researcher.run("test debate topic")

        assert "Key Facts" in result
        # Should have made exactly one LLM call (knowledge compilation).
        assert mock_client.chat.call_count == 1

    def test_auto_research_skipped_when_reference_material_provided(self):
        """If user already provided reference_material, don't waste searches."""
        from vox_romana.shows.debates.bootstrap import DebateBootstrap

        mock_client = Mock()
        # The mock will handle Architect + Profile Builder + Director calls.
        mock_client.chat.side_effect = [
            # Architect response
            json.dumps({
                "title": "Test Debate",
                "topic": "Testing stuff",
                "debate_format": "public_forum",
                "setting": "Studio",
                "tone": "balanced",
                "target_turns": 10,
                "participants": [
                    {"name": "Alice", "title": "Expert", "side": "pro",
                     "position_summary": "For it"},
                ],
                "moderator": {"name": "Mod", "style": "neutral"},
                "audience": {"size": "small", "mood": "curious"},
            }),
            # Profile Builder response (one participant)
            json.dumps({
                "name": "Alice",
                "title": "Expert",
                "side": "pro",
                "position_summary": "For it",
                "voice_fingerprint": "clear",
                "behavioral_triggers": ["data"],
                "emotional_arc": "steady",
                "key_positions": ["Position one"],
                "factual_claims": ["Claim one"],
            }),
            # Director response
            json.dumps({
                "mood_progression": ["curious", "heated", "resolved"],
                "key_moments": [
                    {"turn_range": [1, 5], "event": "opening", "purpose": "set stage"},
                ],
                "closing_strategy": "find common ground",
            }),
        ]

        bootstrap = DebateBootstrap(client=mock_client)
        mock_search = MockSearchProvider()

        # Even though auto_research=True, we pass reference_material
        # so the researcher should NOT run.
        config = bootstrap.from_brief(
            "test debate",
            reference_material="Pre-existing facts here",
            auto_research=True,
            search_provider=mock_search,
        )

        # 3 LLM calls: Architect + Profile Builder + Director.
        # No extra calls for research (would be 5 if researcher ran).
        assert mock_client.chat.call_count == 3

    def test_auto_research_false_skips_researcher(self):
        """When auto_research=False (default), researcher never fires."""
        from vox_romana.shows.debates.bootstrap import DebateBootstrap

        mock_client = Mock()
        mock_client.chat.side_effect = [
            # Architect
            json.dumps({
                "title": "T", "topic": "T", "debate_format": "public_forum",
                "setting": "S", "tone": "balanced", "target_turns": 10,
                "participants": [
                    {"name": "A", "title": "E", "side": "pro",
                     "position_summary": "P"},
                ],
                "moderator": {"name": "M", "style": "neutral"},
                "audience": {"size": "small", "mood": "curious"},
            }),
            # Profile Builder
            json.dumps({
                "name": "A", "title": "E", "side": "pro",
                "position_summary": "P", "voice_fingerprint": "v",
                "behavioral_triggers": [], "emotional_arc": "e",
                "key_positions": [], "factual_claims": [],
            }),
            # Director
            json.dumps({
                "mood_progression": ["start"],
                "key_moments": [
                    {"turn_range": [1, 3], "event": "e", "purpose": "p"},
                ],
                "closing_strategy": "c",
            }),
        ]

        bootstrap = DebateBootstrap(client=mock_client)

        # auto_research defaults to False — no researcher, no search_provider needed.
        config = bootstrap.from_brief("test debate")
        assert mock_client.chat.call_count == 3

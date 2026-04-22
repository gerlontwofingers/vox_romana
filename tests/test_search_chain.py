"""Tests for the memory-first search chain.

Covers the two new providers added in Phase 3:

    * MemorySearchProvider — wraps MemoryRecall as a SearchProvider.
    * ChainSearchProvider  — composes providers into a tiered fallback.
    * build_research_provider() — convenience composer used by bootstraps.

These tests use lightweight stubs so they run in milliseconds without
hitting Chroma, sqlite, or the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from vox_romana.core.search import (
    BaseSearchProvider,
    ChainSearchProvider,
    MemorySearchProvider,
    SearchResult,
    build_research_provider,
)


# ═══════════════════════════════════════════════════════════════════════════
# Test doubles
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FakeRecallResult:
    """Stand-in for ``vox_romana.memory.recall.RecallResult``.

    We use a duck-typed dataclass instead of importing the real one so
    the tests don't require ``chromadb`` / ``sqlite`` to be available.
    """
    passages: list[tuple[str, float, str]] = field(default_factory=list)
    catalog_studies: list[Any] = field(default_factory=list)
    catalog_runs: list[Any] = field(default_factory=list)
    synthesis: str = ""
    method: str = "fts5"


class FakeRecall:
    """MemoryRecall stub — records calls and returns canned results."""

    def __init__(self, result: FakeRecallResult | None = None,
                 *, raises: Exception | None = None) -> None:
        self._result = result or FakeRecallResult()
        self._raises = raises
        self.calls: list[tuple[str, dict]] = []

    def query(self, text: str, *, top_k: int = 10,
              synthesize: bool = True) -> FakeRecallResult:
        self.calls.append((text, {"top_k": top_k, "synthesize": synthesize}))
        if self._raises:
            raise self._raises
        return self._result


class CountingProvider(BaseSearchProvider):
    """SearchProvider that records every call and returns canned results."""

    def __init__(self, results: list[SearchResult] | None = None,
                 *, raises: Exception | None = None) -> None:
        self.results = results or []
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        self.calls.append((query, num_results))
        if self.raises:
            raise self.raises
        return list(self.results)


# ═══════════════════════════════════════════════════════════════════════════
# MemorySearchProvider
# ═══════════════════════════════════════════════════════════════════════════


class TestMemorySearchProvider:
    def test_passes_synthesize_false_to_recall(self) -> None:
        """Search-style use should never trigger LM synthesis (saves a call)."""
        recall = FakeRecall()
        provider = MemorySearchProvider(recall)

        provider.search("anything", num_results=3)

        assert recall.calls == [("anything", {"top_k": 3, "synthesize": False})]

    def test_maps_passages_to_search_results(self) -> None:
        """Each high-scoring passage becomes one SearchResult."""
        recall = FakeRecall(FakeRecallResult(passages=[
            ("Finland UBI 2017 employment effects unclear.", 0.9, "study_42"),
            ("Negative income tax pilots in the 1970s.", 0.7, "study_91"),
        ]))
        provider = MemorySearchProvider(recall, min_score=0.0)

        results = provider.search("UBI evidence")

        assert len(results) == 2
        assert results[0].snippet.startswith("Finland UBI")
        assert results[0].url == "memory://study_42"
        assert results[1].url == "memory://study_91"

    def test_filters_by_min_score(self) -> None:
        """Low-scoring passages are dropped to avoid noise in the reference material."""
        recall = FakeRecall(FakeRecallResult(passages=[
            ("Strong match", 0.8, "a"),
            ("Weak match", 0.1, "b"),
        ]))
        provider = MemorySearchProvider(recall, min_score=0.3)

        results = provider.search("q")

        assert len(results) == 1
        assert results[0].snippet == "Strong match"

    def test_caps_at_num_results(self) -> None:
        """Even if Recall over-returns, we honor the caller's num_results cap."""
        recall = FakeRecall(FakeRecallResult(passages=[
            (f"text {i}", 0.9, f"src_{i}") for i in range(10)
        ]))
        provider = MemorySearchProvider(recall, min_score=0.0)

        results = provider.search("q", num_results=3)

        assert len(results) == 3

    def test_uses_run_title_when_available(self) -> None:
        """If the catalog knows a title for the source, prefer it over a synthetic one."""

        class FakeRun:
            run_id = "src_42"
            title = "Finland UBI 2017: A Critical Review"

        recall = FakeRecall(FakeRecallResult(
            passages=[("snippet", 0.9, "src_42")],
            catalog_runs=[FakeRun()],
        ))
        provider = MemorySearchProvider(recall, min_score=0.0)

        results = provider.search("q")

        assert results[0].title == "Finland UBI 2017: A Critical Review"

    def test_falls_back_to_synthetic_title_when_unknown(self) -> None:
        """Source without a known catalog entry gets a 'Memory: {id}' title."""
        recall = FakeRecall(FakeRecallResult(
            passages=[("snippet", 0.9, "unknown_source")],
        ))
        provider = MemorySearchProvider(recall, min_score=0.0)

        results = provider.search("q")

        assert results[0].title == "Memory: unknown_source"

    def test_swallows_recall_failures(self) -> None:
        """A broken memory backend must NEVER abort the research run."""
        recall = FakeRecall(raises=RuntimeError("chroma exploded"))
        provider = MemorySearchProvider(recall)

        results = provider.search("q")

        assert results == []

    def test_custom_source_label(self) -> None:
        """The URL scheme prefix is configurable for multi-backend setups."""
        recall = FakeRecall(FakeRecallResult(passages=[("t", 0.9, "abc")]))
        provider = MemorySearchProvider(
            recall, min_score=0.0, source_label="chroma://"
        )

        results = provider.search("q")

        assert results[0].url == "chroma://abc"


# ═══════════════════════════════════════════════════════════════════════════
# ChainSearchProvider
# ═══════════════════════════════════════════════════════════════════════════


class TestChainSearchProvider:
    def test_rejects_empty_provider_list(self) -> None:
        with pytest.raises(ValueError):
            ChainSearchProvider([])

    def test_first_provider_satisfies_min_results_short_circuits(self) -> None:
        """If memory has enough hits, Google should never be called."""
        memory = CountingProvider([
            SearchResult(title="m1", url="memory://1", snippet="..."),
            SearchResult(title="m2", url="memory://2", snippet="..."),
            SearchResult(title="m3", url="memory://3", snippet="..."),
        ])
        google = CountingProvider([
            SearchResult(title="g1", url="https://x", snippet="..."),
        ])
        chain = ChainSearchProvider([memory, google], min_results=3)

        results = chain.search("q", num_results=5)

        assert len(memory.calls) == 1
        assert len(google.calls) == 0  # short-circuit
        assert len(results) == 3

    def test_falls_through_when_first_provider_returns_too_few(self) -> None:
        """When memory misses, the chain consults the next provider."""
        memory = CountingProvider([])  # no hits
        google = CountingProvider([
            SearchResult(title="g1", url="https://x", snippet="..."),
            SearchResult(title="g2", url="https://y", snippet="..."),
        ])
        chain = ChainSearchProvider([memory, google], min_results=3)

        results = chain.search("q", num_results=5)

        assert len(memory.calls) == 1
        assert len(google.calls) == 1
        assert len(results) == 2
        assert results[0].title == "g1"

    def test_accumulates_across_providers(self) -> None:
        """Partial hits in tier 1 + tier 2 results combine in order."""
        memory = CountingProvider([
            SearchResult(title="m1", url="memory://1", snippet="..."),
        ])
        google = CountingProvider([
            SearchResult(title="g1", url="https://x", snippet="..."),
            SearchResult(title="g2", url="https://y", snippet="..."),
        ])
        chain = ChainSearchProvider([memory, google], min_results=3)

        results = chain.search("q", num_results=5)

        assert [r.title for r in results] == ["m1", "g1", "g2"]

    def test_truncates_to_caller_num_results(self) -> None:
        """Combined results never exceed the caller's cap."""
        memory = CountingProvider([
            SearchResult(title=f"m{i}", url=f"memory://{i}", snippet="...")
            for i in range(5)
        ])
        chain = ChainSearchProvider([memory], min_results=3)

        results = chain.search("q", num_results=2)

        assert len(results) == 2

    def test_provider_failure_is_skipped_not_propagated(self) -> None:
        """A broken provider must not abort the chain."""
        broken = CountingProvider(raises=RuntimeError("boom"))
        google = CountingProvider([
            SearchResult(title="g1", url="https://x", snippet="..."),
        ])
        chain = ChainSearchProvider([broken, google], min_results=1)

        results = chain.search("q")

        assert results == [SearchResult(title="g1", url="https://x", snippet="...")]


# ═══════════════════════════════════════════════════════════════════════════
# build_research_provider()
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildResearchProvider:
    def test_both_none_returns_none(self) -> None:
        """Caller without memory or web search gets None (LLM-only fallback)."""
        assert build_research_provider(None, None) is None

    def test_only_search_provider_returns_it_unchanged(self) -> None:
        """Web-only setup is the legacy path — pass straight through."""
        google = CountingProvider()
        result = build_research_provider(google, None)
        assert result is google

    def test_only_memory_returns_memory_provider(self) -> None:
        """Memory-only setup wraps the recall in a MemorySearchProvider."""
        recall = FakeRecall()
        result = build_research_provider(None, recall)
        assert isinstance(result, MemorySearchProvider)

    def test_both_returns_chain_with_memory_first(self) -> None:
        """Both present → ChainSearchProvider with memory FIRST in the order."""
        recall = FakeRecall()
        google = CountingProvider()
        result = build_research_provider(google, recall)
        assert isinstance(result, ChainSearchProvider)
        assert isinstance(result._providers[0], MemorySearchProvider)
        assert result._providers[1] is google

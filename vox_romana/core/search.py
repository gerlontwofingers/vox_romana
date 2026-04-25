"""Search provider interface — the plug-in point for web search backends.

The research agent needs to search the web, but we don't want to marry it
to a single search API. This module defines a simple interface that any
search provider can implement, and ships Google Custom Search as the default.

PLUG-IN ARCHITECTURE:

    Think of it like a power outlet. The research agent is an appliance —
    it doesn't care whether the electricity comes from solar, coal, or
    hamster wheels. It just needs a standard socket. The BaseSearchProvider
    is that socket.

    ┌──────────────┐     ┌────────────────────┐
    │ Research Agent│────▶│ BaseSearchProvider  │  ← the interface
    └──────────────┘     └────────┬───────────┘
                                  │
              ┌──────────┬────────┼────────┬──────────┐
              │          │        │        │          │
        ┌─────┴───┐ ┌────┴───┐ ┌──┴──┐ ┌───┴───┐ ┌────┴────┐
        │ Memory  │ │ Chain  │ │Google│ │Tavily │ │  Brave  │
        │ (local) │ │(layered)│ │(built)│ │(future)│ │ (future)│
        └─────────┘ └────────┘ └─────┘ └───────┘ └─────────┘

    The "Chain" provider composes other providers into a tiered ladder
    (e.g. memory-first → Google-fallback) so callers can pay the cheap
    cost first and only escalate when needed. See ChainSearchProvider
    at the bottom of this file.

TO ADD A NEW PROVIDER (future):
    1. Subclass BaseSearchProvider
    2. Implement search() — takes a query string, returns list of SearchResult
    3. Pass your provider to the ResearchAgent or DebateBootstrap
    That's it. No changes to any other code.

EXAMPLE:
    from vox_romana.core.search import GoogleSearchProvider

    provider = GoogleSearchProvider()  # uses env vars for credentials
    results = provider.search("Finland UBI pilot 2017 results")
    for r in results:
        print(f"{r.title}: {r.snippet}")
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """One result from a web search.

    This is the universal "search hit" object. Every provider returns
    these, regardless of how their underlying API formats results.

    Attributes:
        title:    The page title (or headline).
        url:      The full URL.
        snippet:  A brief excerpt or description from the page.
                  This is the most useful field for the research agent —
                  it's often enough to extract key facts without fetching
                  the full page.
    """

    title: str = ""
    url: str = ""
    snippet: str = ""


class BaseSearchProvider(ABC):
    """Interface that all search providers must implement.

    Subclasses need to implement exactly one method: search().
    Everything else (retries, rate limiting, caching) is the
    provider's own business.
    """

    @abstractmethod
    def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        """Search the web and return results.

        Args:
            query:       The search query string.
            num_results: Maximum number of results to return (default: 5).

        Returns:
            A list of SearchResult objects, ordered by relevance.
            May return fewer than num_results if the search engine
            doesn't have enough matches.
        """


class GoogleSearchProvider(BaseSearchProvider):
    """Google Custom Search JSON API provider.

    SETUP (one-time):
        1. Go to https://programmablesearchengine.google.com/ and create a
           search engine. Set it to search the entire web.
        2. Copy the Search Engine ID (cx).
        3. Go to https://console.cloud.google.com/, enable the
           "Custom Search API", and create an API key.
        4. Set environment variables:
               export GOOGLE_API_KEY="your-api-key"
               export GOOGLE_CSE_ID="your-search-engine-id"

    PRICING:
        - 100 queries/day: FREE
        - After that: $5 per 1,000 queries
        - For dev/testing, 100/day is plenty (~20-30 debates)

    USAGE:
        provider = GoogleSearchProvider()
        results = provider.search("Finland UBI pilot 2017 employment effects")
        # Returns up to 5 SearchResult objects with title, url, snippet.

    TESTING:
        Pass api_key and cse_id to the constructor to avoid reading env vars.
        Or mock the whole provider — the research agent accepts any
        BaseSearchProvider.
    """

    # Google Custom Search API endpoint.
    _BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cse_id: str | None = None,
    ) -> None:
        """Initialize the Google search provider.

        Args:
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
            cse_id:  Custom Search Engine ID. Falls back to GOOGLE_CSE_ID env var.
        """
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._cse_id = cse_id or os.environ.get("GOOGLE_CSE_ID", "")

        if not self._api_key:
            raise ValueError(
                "Google API key required. Pass api_key= or set GOOGLE_API_KEY env var. "
                "Get one at https://console.cloud.google.com/ (enable Custom Search API)."
            )
        if not self._cse_id:
            raise ValueError(
                "Google CSE ID required. Pass cse_id= or set GOOGLE_CSE_ID env var. "
                "Create one at https://programmablesearchengine.google.com/"
            )

    def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        """Search via Google Custom Search JSON API.

        Makes one HTTP GET request to Google's API. No external dependencies
        beyond the standard library — we use urllib, not requests.

        Args:
            query:       The search query.
            num_results: Max results (1-10, Google's API limit per request).

        Returns:
            List of SearchResult objects.

        Raises:
            RuntimeError: If the API returns an error (quota exceeded, bad key, etc.).
        """
        import json
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError

        # Clamp to Google's per-request limit.
        num_results = max(1, min(num_results, 10))

        # Build the request URL.
        # Using urllib instead of requests to avoid adding a dependency.
        params = (
            f"key={self._api_key}"
            f"&cx={self._cse_id}"
            f"&q={quote_plus(query)}"
            f"&num={num_results}"
        )
        url = f"{self._BASE_URL}?{params}"

        try:
            req = Request(url)
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except HTTPError as e:
            body = e.read().decode() if e.fp else ""
            raise RuntimeError(
                f"Google Custom Search API error (HTTP {e.code}): {body[:300]}"
            ) from e

        # Parse Google's response into our universal SearchResult format.
        results = []
        for item in data.get("items", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            ))

        return results


# ═════════════════════════════════════════════════════════════════════════════
# MEMORY + CHAIN PROVIDERS — the local-first / paid-fallback combo
# ═════════════════════════════════════════════════════════════════════════════
#
# Once Phase 1 (ingest) and Phase 2 (Chroma vector index) put domain documents
# and prior catalog runs into our local store, the cheapest, fastest, and most
# domain-relevant search target is *the stuff we already know*. These two
# providers turn that store into a first-class SearchProvider so the research
# pipeline can consult it BEFORE — or INSTEAD OF — a paid web API.
#
# Topology (typical production wiring):
#
#     ChainSearchProvider
#       ├─ MemorySearchProvider(MemoryRecall)   # free, sub-ms, domain-tuned
#       └─ GoogleSearchProvider()              # paid, slow, broad coverage
#
# The chain stops as soon as it has accumulated `min_results` hits, so a
# successful memory lookup means **zero Google calls** for that query.
# ═════════════════════════════════════════════════════════════════════════════


class MemorySearchProvider(BaseSearchProvider):
    """SearchProvider backed by ``MemoryRecall`` (Chroma + FTS5 + LM).

    This is the "do we already know about this?" tier. It wraps a
    ``MemoryRecall`` instance so the existing research pipeline — which
    only knows about ``BaseSearchProvider`` — can transparently query the
    local knowledge base built up by ``--mode ingest`` and prior runs.

    Memory is populated by ``python run.py --mode ingest`` —
    see ``vox_romana/memory/ingest.py`` for the ingestion pipeline and
    ``vox_romana/memory/vector_index.py`` for the embedding/persistence
    layer that this provider queries.

    HOW HITS BECOME RESULTS:

        MemoryRecall returns ``passages`` (text chunks ranked by cosine
        similarity) and ``catalog_studies`` (prior research records). We
        map each passage to a ``SearchResult`` whose:

            - ``title``    is the catalog title or "Memory: {source}"
            - ``url``      is ``memory://{source_id}`` so downstream
                           consumers can tell at-a-glance that a hit came
                           from local memory rather than the open web.
            - ``snippet``  is the passage text (the most useful field).

    SCORE FILTERING:

        Vector indexes always return *something*, even for irrelevant
        queries — they just have low scores. ``min_score`` (default 0.3)
        filters out weak matches so the downstream researcher doesn't
        waste tokens compiling noise.

    SYNTHESIS IS DISABLED:

        ``MemoryRecall.query()`` can also call the LM to synthesize a
        prose answer. We turn that off here (``synthesize=False``) — for
        a search-style call we only want passages, not a paragraph of
        LLM-generated text. Saves one LLM call per query.

    USAGE:
        from vox_romana.core.catalog import Catalog
        from vox_romana.memory.recall import MemoryRecall
        from vox_romana.core.search import MemorySearchProvider

        recall = MemoryRecall(catalog=Catalog())
        provider = MemorySearchProvider(recall)
        results = provider.search("UBI Finland 2017 employment")
        # → list[SearchResult] from local Chroma + FTS5

    Args:
        recall:        A ``MemoryRecall`` instance (already configured
                       with catalog / encoder / lm as desired).
        min_score:     Minimum similarity score to include in results.
                       Vector cosine similarity ∈ [-1, 1]; 0.3 is a
                       conservative "definitely related" threshold.
        source_label:  URL scheme for the synthetic ``url`` field.
                       Default ``"memory://"``. Override if you have
                       multiple memory backends and want to distinguish
                       them (e.g. ``"chroma://"``, ``"fts5://"``).
    """

    def __init__(
        self,
        recall: Any,
        *,
        min_score: float = 0.3,
        source_label: str = "memory://",
    ) -> None:
        self._recall = recall
        self._min_score = min_score
        self._source_label = source_label

    def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        """Search local memory and return ranked SearchResults.

        Never raises — a memory backend failure logs a warning and
        returns ``[]`` so the chain can fall through to the next
        provider. (A failing local lookup should never break a debate
        that could otherwise be researched on the open web.)

        Args:
            query:       The search query (same as any other provider).
            num_results: Max results to return.

        Returns:
            List of SearchResult objects. Empty if memory has nothing
            relevant or if the recall call fails.
        """
        try:
            recall_result = self._recall.query(
                query, top_k=num_results, synthesize=False
            )
        except Exception as exc:
            # Defensive: any catalog/index/IO error here should NEVER
            # cascade up and break the research pipeline.
            import logging
            logging.getLogger(__name__).warning(
                "MemorySearchProvider lookup failed: %s", exc
            )
            return []

        # Build a quick {source_id: catalog_title} lookup for nicer titles.
        title_by_source: dict[str, str] = {}
        for run in getattr(recall_result, "catalog_runs", []) or []:
            run_id = getattr(run, "run_id", None)
            title = getattr(run, "title", None) or getattr(run, "brief", "")
            if run_id and title:
                title_by_source[run_id] = title[:120]
        for study in getattr(recall_result, "catalog_studies", []) or []:
            sid = getattr(study, "study_id", None)
            method = getattr(study, "method", "") or ""
            dataset = getattr(study, "dataset", "") or ""
            if sid:
                # Studies don't have proper "titles" — synthesize one.
                title_by_source[sid] = f"{method} on {dataset}".strip(" /")

        results: list[SearchResult] = []
        for text, score, source in recall_result.passages or []:
            if score < self._min_score:
                continue
            results.append(SearchResult(
                title=title_by_source.get(source, f"Memory: {source}"),
                url=f"{self._source_label}{source}",
                snippet=text,
            ))
            if len(results) >= num_results:
                break
        return results


class ChainSearchProvider(BaseSearchProvider):
    """SearchProvider that tries multiple providers in order.

    The "ladder of escalation": cheap/fast providers come first; we only
    climb to slower/paid providers if the cheaper ones don't return
    enough material. This is the standard wiring for a production
    research pipeline that wants to minimize cost AND latency without
    sacrificing recall.

    BEHAVIOR:

        * Walks ``providers`` in the order given.
        * Accumulates SearchResults across providers.
        * Stops as soon as the running total reaches ``min_results``.
        * If a provider raises, the error is logged and the chain
          continues with the next provider — one broken backend should
          never abort the whole research run.

    EXAMPLE — memory-first, Google-fallback:

        from vox_romana.core.search import (
            ChainSearchProvider,
            GoogleSearchProvider,
            MemorySearchProvider,
        )
        from vox_romana.memory.recall import MemoryRecall

        chain = ChainSearchProvider([
            MemorySearchProvider(MemoryRecall(catalog=cat)),
            GoogleSearchProvider(),
        ])
        # Topic seen before? Memory wins, Google never called.
        # Brand new topic? Memory misses, chain falls through to Google.
        results = chain.search("Estonia digital ID system 2024")

    Args:
        providers:   Ordered list of providers to try. Cheapest first.
        min_results: Once the running total reaches this many results,
                     skip remaining providers. Default 3 — enough to
                     ground a well-known topic without ever hitting the
                     paid fallback.
    """

    def __init__(
        self,
        providers: list[BaseSearchProvider],
        *,
        min_results: int = 3,
    ) -> None:
        if not providers:
            raise ValueError("ChainSearchProvider requires at least one provider")
        self._providers = list(providers)
        self._min_results = min_results

    def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        """Search providers in order until ``min_results`` are accumulated.

        Args:
            query:       The search query.
            num_results: Max total results to return across all providers.

        Returns:
            Combined list of SearchResults, never longer than ``num_results``.
        """
        import logging
        logger = logging.getLogger(__name__)

        accumulated: list[SearchResult] = []
        for provider in self._providers:
            if len(accumulated) >= self._min_results:
                # Earlier providers already gave us enough — don't even
                # invoke the (potentially expensive) later ones.
                break
            try:
                # Ask each provider for the FULL num_results. Easier to
                # de-duplicate-by-truncate at the end than to predict
                # how many each tier will actually return.
                provider_results = provider.search(query, num_results=num_results)
            except Exception as exc:
                logger.warning(
                    "ChainSearchProvider: %s.search() failed: %s",
                    provider.__class__.__name__,
                    exc,
                )
                continue
            accumulated.extend(provider_results)

        # Truncate to the caller's requested cap.
        return accumulated[:num_results]


# ═════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: build the right provider for a (search_provider, memory) pair
# ═════════════════════════════════════════════════════════════════════════════

def build_research_provider(
    search_provider: BaseSearchProvider | None,
    memory: Any | None,
    *,
    min_results: int = 3,
) -> BaseSearchProvider | None:
    """Compose ``search_provider`` and ``memory`` into a single provider.

    This is the helper used by every show bootstrap (debates / news /
    podcast) to honor the "memory-first, web-fallback" policy without
    duplicating the same wiring logic three times.

    Decision table:

        memory   search_provider   →   returned provider
        ------   ---------------   -   -----------------
        None     None              →   None  (caller uses LLM-only research)
        None     Google            →   Google
        Recall   None              →   MemorySearchProvider(Recall)
        Recall   Google            →   ChainSearchProvider([Memory, Google])

    Args:
        search_provider: Web search provider (or None).
        memory:          ``MemoryRecall`` instance (or None). Imported
                         lazily inside this function to avoid a hard
                         dependency from ``core.search`` on the memory
                         subsystem.
        min_results:     Forwarded to ChainSearchProvider when both
                         providers are present. See its docstring.

    Returns:
        A single ``BaseSearchProvider`` (or ``None`` when both inputs
        are ``None``) ready to hand to a researcher.
    """
    if memory is None:
        return search_provider
    mem_provider = MemorySearchProvider(memory)
    if search_provider is None:
        return mem_provider
    return ChainSearchProvider(
        [mem_provider, search_provider], min_results=min_results
    )


def build_fact_check_provider() -> BaseSearchProvider | None:
    """Return a web search provider for the FactChecker, or ``None``.

    The FactChecker uses a search provider to verify claims it cannot
    resolve from ``reference_material`` alone (the "thorough" mode).
    Because search credentials are optional — many users run locally
    without a Google API key — this helper swallows the ``ValueError``
    that :class:`GoogleSearchProvider` raises when env vars are absent
    and returns ``None`` instead.  A ``None`` return means the
    FactChecker falls back to LLM-only verification ("quick" behaviour)
    rather than crashing.

    This mirrors the ``_enable_research`` fallback ladder in
    ``run_show.py``: try Google, silently degrade to LLM-only on
    any configuration error.

    Returns:
        A :class:`GoogleSearchProvider` if ``GOOGLE_API_KEY`` and
        ``GOOGLE_CSE_ID`` are set; ``None`` otherwise.
    """
    try:
        return GoogleSearchProvider()
    except (ValueError, ImportError):
        logger.debug(
            "build_fact_check_provider: Google credentials not available; "
            "FactChecker will use LLM-only verification."
        )
        return None


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
                    ┌─────────────┼─────────────┐
                    │             │             │
              ┌─────┴──┐   ┌─────┴──┐   ┌──────┴───┐
              │ Google  │   │ Tavily │   │ Brave    │
              │ (built) │   │(future)│   │ (future) │
              └────────┘   └────────┘   └──────────┘

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

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.parse import quote_plus


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

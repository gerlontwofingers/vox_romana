"""Knowledge sources for the University deep-research subsystem.

A KnowledgeSource is the Scholar's window onto a body of information.
Each source exposes a uniform two-method interface — ``search()`` to find
candidate articles and ``fetch()`` to retrieve a full article — so the
Scholar's research loop is decoupled from where knowledge actually lives.

This module ships two concrete sources:

* :class:`WikipediaSource` — on-demand MediaWiki REST client with a JSON
  disk cache under ``WIKIPEDIA_CACHE_DIR``. Cache TTL is configurable.
  Requires ``pip install wikipedia-api`` (optional dependency — without
  it the source instantiates but logs a clear error on first use).

* :class:`LocalSource` — adapter over the existing
  :class:`vox_romana.memory.recall.MemoryRecall`, exposing previously
  ingested documents and catalog records to the Scholar without any new
  vector-store wiring.

Design constraints (binding):

* Article is a frozen dataclass with tuple fields so it is hashable and
  safe to share across threads — mirrors :class:`vox_romana.core.models.FactCard`.
* ``KnowledgeSource`` is purely synchronous; concurrency is the Scholar's
  concern (Phase 2 University may run multiple Scholars in sequence).
* Network and parse failures NEVER raise out of these methods — they log
  a warning and return an empty result. The Scholar's research loop must
  be resilient to transient failures; one bad article should not abort
  a 3-hour study.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vox_romana.core.paths import WIKIPEDIA_CACHE_DIR

if TYPE_CHECKING:
    from vox_romana.memory.recall import MemoryRecall

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Article (transport object)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Article:
    """A single piece of knowledge passed between Source and Scholar.

    Frozen + tuple collections so the dataclass is hashable, safely shared,
    and trivially serialisable.

    Attributes:
        title:       Human-readable identifier (Wikipedia article title,
                     ingested document name, etc.).
        url:         Canonical URL or pseudo-URL (``memory://...`` for
                     local sources, ``https://en.wikipedia.org/...`` for wiki).
        text:        Full readable text. May be empty if only a search-snippet
                     was returned (caller should ``fetch()`` for full text).
        snippet:     Short summary or excerpt (always present for search hits).
        links:       Outbound article identifiers (titles for Wikipedia,
                     source-ids for memory).
        categories:  Topic tags from the source (Wikipedia categories,
                     ingest tags).
        source_name: Name of the originating KnowledgeSource (matches
                     ``KnowledgeSource.name``).
        score:       Relevance score from the source's own search if available
                     (0.0–1.0). 0.0 when not provided.
        fetched_at:  ISO 8601 UTC timestamp of retrieval.
    """

    title: str
    url: str = ""
    text: str = ""
    snippet: str = ""
    links: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    source_name: str = ""
    score: float = 0.0
    fetched_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (tuples → lists). Mirrors FactCard."""
        d = asdict(self)
        d["links"] = list(self.links)
        d["categories"] = list(self.categories)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Article":
        """Inverse of :meth:`to_dict`. Tolerant of missing keys."""
        return cls(
            title=d.get("title", ""),
            url=d.get("url", ""),
            text=d.get("text", ""),
            snippet=d.get("snippet", ""),
            links=tuple(d.get("links", ())),
            categories=tuple(d.get("categories", ())),
            source_name=d.get("source_name", ""),
            score=float(d.get("score", 0.0)),
            fetched_at=d.get("fetched_at", ""),
        )


def _now_iso() -> str:
    """Current UTC time in ISO 8601 format with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeSource — abstract base
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeSource:
    """Abstract base for all knowledge sources.

    A KnowledgeSource is defined by a ``name`` (used for logging and for
    ``Article.source_name`` tagging) and two methods:

    * :meth:`search` — return up to ``n`` candidate articles for a free-text
      query. Returned ``Article``s typically have a ``snippet`` but may have
      empty ``text``; the Scholar will :meth:`fetch` if it needs full text.

    * :meth:`fetch` — return a single full ``Article`` for a known
      identifier (the format of the identifier is source-specific —
      Wikipedia uses titles, memory uses source-ids).

    Subclasses MUST NOT raise on transient failures. Return ``[]`` from
    ``search`` and ``None`` from ``fetch`` instead, after logging.
    """

    #: Short identifier for the source. Override in subclasses.
    name: str = "abstract"

    def search(self, query: str, n: int = 10) -> list[Article]:
        """Return up to ``n`` candidate articles for ``query``.

        Args:
            query: Free-text search query.
            n:     Maximum number of results to return.

        Returns:
            A list of Articles, possibly empty. Never raises.
        """
        raise NotImplementedError

    def fetch(self, identifier: str) -> Article | None:
        """Return the full article for a known identifier.

        Args:
            identifier: Source-specific article identifier (Wikipedia title,
                        memory source-id, etc.).

        Returns:
            The Article, or None if not found / fetch failed. Never raises.
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# WikipediaSource — on-demand MediaWiki client with disk cache
# ─────────────────────────────────────────────────────────────────────────────

# Filesystem-safe slug for cache filenames. Wikipedia titles can contain
# slashes, colons, asterisks, etc. — strip them for the on-disk cache.
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(title: str, max_len: int = 120) -> str:
    """Filesystem-safe slug for a Wikipedia title.

    Truncates to ``max_len`` characters but always preserves at least the
    first 8 characters of any tail to keep collisions rare.
    """
    slug = _SLUG_RE.sub("_", title).strip("_") or "untitled"
    if len(slug) > max_len:
        slug = slug[: max_len - 9] + "_" + slug[-8:]
    return slug


class WikipediaSource(KnowledgeSource):
    """On-demand Wikipedia source with persistent JSON cache.

    Uses the ``wikipedia-api`` package (a thin client over the MediaWiki
    REST API) for both search and full-text fetch. Cached articles are
    re-used until ``ttl_days`` have elapsed since ``fetched_at``.

    Args:
        language:    Wikipedia language code (``"en"`` for English).
        cache_dir:   Cache directory. Defaults to
                     :data:`vox_romana.core.paths.WIKIPEDIA_CACHE_DIR`.
        ttl_days:    Cache freshness in days. ``0`` disables the cache
                     (always re-fetch). Negative values mean "never expire".
        user_agent:  HTTP User-Agent header. Wikipedia REQUIRES a contact
                     UA for non-trivial scripts; the default identifies
                     vox_romana but you should override it for production.
        max_links:   Maximum outbound links to keep per article. Caps the
                     graph explosion in the Scholar's queue.
        request_pause_secs: Seconds to sleep between live requests to be
                            polite to Wikipedia. Cache hits skip this.
    """

    name = "wikipedia"

    DEFAULT_USER_AGENT = (
        "vox_romana/0.x (https://github.com/; local-research-bot)"
    )

    def __init__(
        self,
        *,
        language: str = "en",
        cache_dir: Path | None = None,
        ttl_days: float = 30.0,
        user_agent: str | None = None,
        max_links: int = 200,
        request_pause_secs: float = 0.1,
    ) -> None:
        self._language = language
        self._cache_dir = Path(cache_dir) if cache_dir else WIKIPEDIA_CACHE_DIR
        self._ttl_days = ttl_days
        self._user_agent = user_agent or self.DEFAULT_USER_AGENT
        self._max_links = max_links
        self._pause = max(0.0, float(request_pause_secs))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._wiki: Any | None = None  # lazy: wikipediaapi.Wikipedia
        self._import_error: ImportError | None = None

    # ── client init (lazy) ──────────────────────────────────────────────

    def _client(self) -> Any | None:
        """Return the wikipediaapi.Wikipedia client, or None if unavailable.

        The first failure is logged once; subsequent calls return None
        silently to avoid log spam during long research loops.
        """
        if self._wiki is not None:
            return self._wiki
        if self._import_error is not None:
            return None
        try:
            import wikipediaapi  # type: ignore[import-not-found]
        except ImportError as exc:
            self._import_error = exc
            logger.error(
                "WikipediaSource unavailable: install with `pip install wikipedia-api`"
            )
            return None
        self._wiki = wikipediaapi.Wikipedia(
            user_agent=self._user_agent,
            language=self._language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )
        return self._wiki

    # ── cache helpers ───────────────────────────────────────────────────

    def _cache_path(self, title: str) -> Path:
        return self._cache_dir / f"{_slugify(title)}.json"

    def _is_fresh(self, cached: dict[str, Any]) -> bool:
        """True if cached payload is within the TTL window."""
        if self._ttl_days < 0:
            return True
        if self._ttl_days == 0:
            return False
        ts = cached.get("fetched_at", "")
        if not ts:
            return False
        try:
            fetched = datetime.fromisoformat(ts)
        except ValueError:
            return False
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - fetched).total_seconds() / 86400.0
        return age_days < self._ttl_days

    def _read_cache(self, title: str) -> Article | None:
        path = self._cache_path(title)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Wikipedia cache read failed for %s: %s", title, exc)
            return None
        if not self._is_fresh(payload):
            return None
        return Article.from_dict(payload)

    def _write_cache(self, article: Article) -> None:
        path = self._cache_path(article.title)
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(article.to_dict(), fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning(
                "Wikipedia cache write failed for %s: %s", article.title, exc
            )

    # ── public API ──────────────────────────────────────────────────────

    def search(self, query: str, n: int = 10) -> list[Article]:
        """Search Wikipedia and return up to ``n`` snippet-only Articles.

        Hits Wikipedia's ``opensearch`` endpoint via the MediaWiki API.
        ``wikipedia-api`` does not expose this directly, so we fall back to
        the requests-based MediaWiki ``list=search`` query.

        Returns lightweight Articles with snippets only; call :meth:`fetch`
        for the full text.
        """
        if not query.strip():
            return []
        client = self._client()
        if client is None:
            return []

        # Use the underlying requests session that wikipediaapi already
        # configured (correct UA + connection pooling).
        try:
            import requests  # type: ignore[import-not-found]
        except ImportError:
            logger.error("WikipediaSource.search requires `requests`")
            return []

        api_url = f"https://{self._language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(max(1, min(n, 50))),
            "format": "json",
            "utf8": "1",
        }
        try:
            if self._pause:
                time.sleep(self._pause)
            resp = requests.get(
                api_url,
                params=params,
                headers={"User-Agent": self._user_agent},
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("Wikipedia search failed for %r: %s", query, exc)
            return []

        hits = (payload.get("query") or {}).get("search") or []
        articles: list[Article] = []
        for hit in hits[:n]:
            title = hit.get("title", "")
            if not title:
                continue
            snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
            articles.append(
                Article(
                    title=title,
                    url=f"https://{self._language}.wikipedia.org/wiki/"
                    + title.replace(" ", "_"),
                    snippet=snippet,
                    source_name=self.name,
                    score=float(hit.get("score") or 0.0),
                    fetched_at=_now_iso(),
                )
            )
        return articles

    def fetch(self, identifier: str) -> Article | None:
        """Return the full Wikipedia article for ``identifier`` (the title).

        Cache-first: returns a cached copy if fresh. Otherwise hits the
        MediaWiki API, caches the result, and returns it. Returns ``None``
        if the page does not exist or the fetch fails.
        """
        title = (identifier or "").strip()
        if not title:
            return None

        cached = self._read_cache(title)
        if cached is not None:
            return cached

        client = self._client()
        if client is None:
            return None

        try:
            if self._pause:
                time.sleep(self._pause)
            page = client.page(title)
            if not page.exists():
                logger.info("Wikipedia page does not exist: %s", title)
                return None
            text = page.text or ""
            # Outbound article links (capped). page.links is a dict
            # {title: Page}. Iteration gives titles.
            try:
                links = tuple(list(page.links.keys())[: self._max_links])
            except Exception:
                links = ()
            # Categories come prefixed "Category:" — strip for cleanliness.
            try:
                cats = tuple(
                    c.replace("Category:", "", 1) for c in page.categories.keys()
                )
            except Exception:
                cats = ()
            article = Article(
                title=page.title,
                url=page.fullurl or "",
                text=text,
                snippet=(page.summary or "")[:500],
                links=links,
                categories=cats,
                source_name=self.name,
                score=0.0,
                fetched_at=_now_iso(),
            )
        except Exception as exc:
            logger.warning("Wikipedia fetch failed for %r: %s", title, exc)
            return None

        self._write_cache(article)
        return article


# ─────────────────────────────────────────────────────────────────────────────
# LocalSource — adapter over MemoryRecall
# ─────────────────────────────────────────────────────────────────────────────

class LocalSource(KnowledgeSource):
    """Adapter that exposes the existing MemoryRecall as a KnowledgeSource.

    Maps the Scholar's source interface onto :meth:`MemoryRecall.query`.
    Each ranked passage becomes an ``Article`` whose ``url`` is the
    pseudo-URL ``memory://<source_id>``. ``synthesize=False`` so we never
    burn LM cycles inside a search loop.

    Args:
        recall:        A configured :class:`MemoryRecall` instance.
        passage_chars: When fetch is called by source_id, return this many
                       characters of the matched passage as the article text.
                       Memory passages are already chunked (~800 chars) so
                       the default of 4000 typically returns the whole chunk.
    """

    name = "local"

    def __init__(
        self,
        recall: "MemoryRecall",
        *,
        passage_chars: int = 4000,
    ) -> None:
        self._recall = recall
        self._passage_chars = passage_chars

    def search(self, query: str, n: int = 10) -> list[Article]:
        if not query.strip():
            return []
        try:
            result = self._recall.query(query, top_k=n, synthesize=False)
        except Exception as exc:
            logger.warning("LocalSource.search failed for %r: %s", query, exc)
            return []

        articles: list[Article] = []
        ts = _now_iso()
        for text, score, source_id in result.passages[:n]:
            snippet = text[:300]
            articles.append(
                Article(
                    title=source_id or "memory_passage",
                    url=f"memory://{source_id}" if source_id else "memory://",
                    text=text[: self._passage_chars],
                    snippet=snippet,
                    source_name=self.name,
                    score=float(score),
                    fetched_at=ts,
                )
            )
        return articles

    def fetch(self, identifier: str) -> Article | None:
        """Return the Article for a memory source-id.

        Memory does not expose a direct "give me passage by id" API, so we
        re-query using the identifier as the search text and pick the
        best-matching passage with that exact source-id.
        """
        ident = (identifier or "").strip()
        if not ident:
            return None
        # Strip pseudo-URL scheme if present
        if ident.startswith("memory://"):
            ident = ident[len("memory://") :]
        try:
            result = self._recall.query(ident, top_k=20, synthesize=False)
        except Exception as exc:
            logger.warning("LocalSource.fetch failed for %r: %s", identifier, exc)
            return None
        for text, score, source_id in result.passages:
            if source_id == ident:
                return Article(
                    title=source_id,
                    url=f"memory://{source_id}",
                    text=text[: self._passage_chars],
                    snippet=text[:300],
                    source_name=self.name,
                    score=float(score),
                    fetched_at=_now_iso(),
                )
        return None

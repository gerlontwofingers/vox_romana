"""Tests for vox_romana.university.sources.

Covers:
    - Article dataclass (frozen, hashable, roundtrip, tolerant from_dict)
    - _slugify helper
    - KnowledgeSource abstract base (NotImplementedError, name attribute)
    - WikipediaSource init, cache helpers (no network required)
    - WikipediaSource graceful degradation when wikipedia-api is absent
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from vox_romana.university.sources import (
    Article,
    KnowledgeSource,
    WikipediaSource,
    _slugify,
)


# ══════════════════════════════════════════════════════════════════════════════
# Article
# ══════════════════════════════════════════════════════════════════════════════


class TestArticle:
    def test_defaults(self):
        a = Article(title="Test")
        assert a.url == ""
        assert a.text == ""
        assert a.links == ()
        assert a.categories == ()
        assert a.score == 0.0
        assert a.fetched_at == ""

    def test_frozen(self):
        a = Article(title="Immutable")
        with pytest.raises(Exception):
            a.title = "changed"  # type: ignore[misc]

    def test_hashable(self):
        a1 = Article(title="X", url="http://a")
        a2 = Article(title="X", url="http://a")
        assert hash(a1) == hash(a2)
        assert len({a1, a2}) == 1

    def test_different_articles_not_equal(self):
        a1 = Article(title="A")
        a2 = Article(title="B")
        assert a1 != a2

    def test_to_dict_converts_tuples_to_lists(self):
        a = Article(title="T", links=("A", "B"), categories=("Cat",))
        d = a.to_dict()
        assert isinstance(d["links"], list)
        assert d["links"] == ["A", "B"]
        assert isinstance(d["categories"], list)
        assert d["categories"] == ["Cat"]

    def test_from_dict_roundtrip(self):
        original = Article(
            title="Roman roads",
            url="https://en.wikipedia.org/wiki/Roman_roads",
            text="Long text here.",
            snippet="Short snippet.",
            links=("Via Appia", "Aqueduct"),
            categories=("Roman infrastructure",),
            source_name="wikipedia",
            score=0.9,
        )
        d = original.to_dict()
        restored = Article.from_dict(d)
        assert restored == original

    def test_from_dict_tolerant_of_missing_keys(self):
        a = Article.from_dict({"title": "Minimal"})
        assert a.title == "Minimal"
        assert a.score == 0.0
        assert a.links == ()
        assert a.categories == ()

    def test_from_dict_empty_dict(self):
        a = Article.from_dict({})
        assert a.title == ""

    def test_score_coerced_to_float(self):
        a = Article.from_dict({"title": "X", "score": "0.75"})
        assert a.score == 0.75


# ══════════════════════════════════════════════════════════════════════════════
# _slugify
# ══════════════════════════════════════════════════════════════════════════════


class TestSlugify:
    def test_simple_title(self):
        assert _slugify("Roman roads") == "Roman_roads"

    def test_special_chars_replaced(self):
        slug = _slugify("Apollo 11: A/B test (2001)")
        assert "/" not in slug
        assert ":" not in slug
        assert "(" not in slug

    def test_truncation_respects_max_len(self):
        long_title = "A" * 200
        slug = _slugify(long_title, max_len=120)
        assert len(slug) <= 120

    def test_empty_string_returns_untitled(self):
        assert _slugify("") == "untitled"

    def test_spaces_to_underscores(self):
        assert "_" in _slugify("hello world")


# ══════════════════════════════════════════════════════════════════════════════
# KnowledgeSource (abstract base)
# ══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeSource:
    def test_search_raises_not_implemented(self):
        src = KnowledgeSource()
        with pytest.raises(NotImplementedError):
            src.search("query")

    def test_fetch_raises_not_implemented(self):
        src = KnowledgeSource()
        with pytest.raises(NotImplementedError):
            src.fetch("id")

    def test_name_attribute(self):
        assert KnowledgeSource.name == "abstract"


# ══════════════════════════════════════════════════════════════════════════════
# WikipediaSource — offline (no network)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def wiki_cache_dir(tmp_path):
    return tmp_path / "wiki_cache"


@pytest.fixture
def wiki_source(wiki_cache_dir):
    return WikipediaSource(cache_dir=wiki_cache_dir, request_pause_secs=0.0)


def _fresh_article(title: str) -> Article:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return Article(
        title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        text=f"Full text of {title}.",
        source_name="wikipedia",
        fetched_at=ts,
    )


class TestWikipediaSourceInit:
    def test_default_name(self, wiki_source):
        assert wiki_source.name == "wikipedia"

    def test_cache_dir_created(self, wiki_cache_dir):
        WikipediaSource(cache_dir=wiki_cache_dir)
        assert wiki_cache_dir.exists()

    def test_custom_language(self, tmp_path):
        src = WikipediaSource(language="de", cache_dir=tmp_path)
        assert src._language == "de"

    def test_custom_ttl(self, tmp_path):
        src = WikipediaSource(cache_dir=tmp_path, ttl_days=7.0)
        assert src._ttl_days == 7.0


class TestWikipediaSourceCache:
    def test_cache_miss_returns_none(self, wiki_source):
        assert wiki_source._read_cache("Nonexistent Article") is None

    def test_write_then_read_fresh_cache(self, wiki_source):
        article = _fresh_article("Roman aqueducts")
        wiki_source._write_cache(article)
        cached = wiki_source._read_cache("Roman aqueducts")
        assert cached is not None
        assert cached.title == "Roman aqueducts"
        assert cached.text == "Full text of Roman aqueducts."

    def test_stale_cache_returns_none(self, tmp_path):
        src = WikipediaSource(cache_dir=tmp_path, ttl_days=1.0)
        stale_ts = (
            datetime.now(timezone.utc) - timedelta(days=2)
        ).replace(microsecond=0).isoformat()
        article = Article(
            title="Stale",
            fetched_at=stale_ts,
            source_name="wikipedia",
        )
        src._write_cache(article)
        assert src._read_cache("Stale") is None

    def test_negative_ttl_never_expires(self, tmp_path):
        src = WikipediaSource(cache_dir=tmp_path, ttl_days=-1.0)
        stale_ts = (
            datetime.now(timezone.utc) - timedelta(days=365)
        ).replace(microsecond=0).isoformat()
        article = Article(title="Old", fetched_at=stale_ts, source_name="wikipedia")
        src._write_cache(article)
        assert src._read_cache("Old") is not None

    def test_zero_ttl_always_stale(self, tmp_path):
        src = WikipediaSource(cache_dir=tmp_path, ttl_days=0.0)
        ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        article = Article(title="Fresh", fetched_at=ts, source_name="wikipedia")
        src._write_cache(article)
        assert src._read_cache("Fresh") is None

    def test_corrupted_cache_file_returns_none(self, wiki_source):
        path = wiki_source._cache_path("Corrupt")
        path.write_text("NOT VALID JSON", encoding="utf-8")
        assert wiki_source._read_cache("Corrupt") is None

    def test_cache_path_uses_slugified_name(self, wiki_source):
        path = wiki_source._cache_path("Title with spaces")
        assert " " not in path.name


class TestWikipediaSourceWithoutLibrary:
    """Source gracefully degrades when wikipedia-api is not installed."""

    def test_search_returns_empty_when_no_client(self, wiki_source, monkeypatch):
        monkeypatch.setattr(wiki_source, "_import_error", ImportError("no lib"))
        assert wiki_source.search("Roman roads") == []

    def test_fetch_returns_none_when_no_client(self, wiki_source, monkeypatch):
        monkeypatch.setattr(wiki_source, "_import_error", ImportError("no lib"))
        assert wiki_source.fetch("Roman roads") is None

    def test_search_empty_query_returns_empty(self, wiki_source):
        assert wiki_source.search("") == []
        assert wiki_source.search("   ") == []

    def test_fetch_empty_id_returns_none(self, wiki_source):
        assert wiki_source.fetch("") is None
        assert wiki_source.fetch("  ") is None

    def test_fetch_cache_hit_bypasses_client(self, wiki_source, monkeypatch):
        """A warm cache hit returns the article even when the library is absent."""
        article = _fresh_article("Concrete")
        wiki_source._write_cache(article)
        # Simulate missing library — cache hit should not need the client
        monkeypatch.setattr(wiki_source, "_import_error", ImportError("no lib"))
        result = wiki_source.fetch("Concrete")
        assert result is not None
        assert result.title == "Concrete"

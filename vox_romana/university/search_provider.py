"""UniversitySearchProvider — expose a ScholarSession as a SearchProvider.

This bridges the University deep-research system and the existing
show-generation pipeline.  A completed (or paused) ``ScholarSession``
holds structured :class:`~vox_romana.university.session.ScholarNote`
objects; this provider surfaces them as
:class:`~vox_romana.core.search.SearchResult` objects so any show
bootstrap can treat a University session exactly like Google or memory.

PRIORITY IN THE RESEARCH LADDER::

    University (session notes)
        ↓  (if not enough material)
    Memory (Chroma + FTS5, local)
        ↓  (if still not enough)
    Google Custom Search (paid, web)
        ↓  (last resort)
    LLM-only (no search at all)

USAGE::

    from vox_romana.university.search_provider import UniversitySearchProvider

    provider = UniversitySearchProvider("a1b2c3d4e5f6")   # session_id
    results  = provider.search("universal basic income Finland")
    for r in results:
        print(r.title, r.url)
        print(r.snippet)

The constructor loads the session from ``UNIVERSITY_ROOT / session_id``.
If the session directory is missing or corrupt it raises ``FileNotFoundError``
or ``ValueError`` so the caller can log a warning and fall back.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import vox_romana.core.paths as _paths
from vox_romana.core.search import BaseSearchProvider, SearchResult

if TYPE_CHECKING:
    from vox_romana.university.session import ScholarNote, ScholarSession

logger = logging.getLogger(__name__)

# How many characters of a note's summary to include in the snippet.
_SNIPPET_MAX = 280

# Maximum keyword-match bonus added on top of relevance_score.
# Keeps the scoring spread tight so highly-relevant notes aren't buried
# by coincidental keyword hits.
_KEYWORD_BONUS_MAX = 3.0


def _score_note(note: "ScholarNote", query_words: frozenset[str]) -> float:
    """Compute a ranking score for *note* against *query_words*.

    Score = ``relevance_score`` (0–10) + keyword_bonus (0–3).

    Keyword bonus: +1 for each query word found in the note's title or
    summary, capped at ``_KEYWORD_BONUS_MAX``.  Case-insensitive.
    """
    text = (note.article_title + " " + note.summary).lower()
    hits = sum(1 for w in query_words if w in text)
    bonus = min(float(hits), _KEYWORD_BONUS_MAX)
    return note.relevance_score + bonus


def _make_snippet(note: "ScholarNote") -> str:
    """Build a concise snippet from a ScholarNote.

    Prefers the note's ``summary`` field (trimmed to ``_SNIPPET_MAX``
    chars).  Falls back to the first three ``key_facts`` joined by '; '
    when the summary is empty.
    """
    if note.summary:
        text = note.summary.strip()
        if len(text) > _SNIPPET_MAX:
            text = text[: _SNIPPET_MAX - 1] + "…"
        return text
    if note.key_facts:
        return "; ".join(note.key_facts[:3])
    return ""


class UniversitySearchProvider(BaseSearchProvider):
    """A :class:`~vox_romana.core.search.BaseSearchProvider` backed by a
    persisted :class:`~vox_romana.university.session.ScholarSession`.

    Each ``ScholarNote`` in the session maps to one ``SearchResult``::

        SearchResult(
            title   = note.article_title,
            url     = note.source_url,          # e.g. Wikipedia URL
            snippet = note.summary or key_facts,
        )

    Notes are ranked by a weighted combination of ``relevance_score``
    (set by the Scholar's LLM) and keyword overlap with the caller's
    query.  The session synthesis is appended as an extra result (titled
    "[Synthesis]") when there are fewer notes than ``num_results``; this
    ensures callers always get at least one result as long as the Scholar
    produced any output.

    Args:
        session_id:   The 12-hex-char session ID, used to locate
                      ``{UNIVERSITY_ROOT}/{session_id}/session.json``.

    Raises:
        FileNotFoundError: If the session directory or ``session.json``
            does not exist.
        ValueError: If the session file is present but fails validation.
    """

    def __init__(self, session_id: str) -> None:
        from vox_romana.university.session import ScholarSession

        session_dir = _paths.UNIVERSITY_ROOT / session_id
        self._session: ScholarSession = ScholarSession.load(session_dir)
        self._session_id = session_id

    # ── BaseSearchProvider interface ──────────────────────────────────

    def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        """Return up to *num_results* ``SearchResult`` objects from the session.

        Notes are ranked by relevance_score + keyword overlap with *query*.
        The session synthesis is appended as a catch-all result when the
        note count is below *num_results*.

        Args:
            query:       The search query string.
            num_results: Maximum results to return (default: 5).

        Returns:
            A list of up to *num_results* ``SearchResult`` objects, ordered
            by descending relevance.  Returns an empty list only when the
            session has no notes AND no synthesis.
        """
        notes = list(self._session.notes)
        if not notes and not self._session.synthesis:
            logger.debug(
                "UniversitySearchProvider: session %s has no notes or synthesis",
                self._session_id,
            )
            return []

        query_words = frozenset(query.lower().split())

        # Rank notes
        ranked = sorted(
            notes,
            key=lambda n: _score_note(n, query_words),
            reverse=True,
        )

        results: list[SearchResult] = []

        for note in ranked[:num_results]:
            results.append(
                SearchResult(
                    title=note.article_title or self._session.topic,
                    url=note.source_url,
                    snippet=_make_snippet(note),
                )
            )

        # If we have room AND the session has a synthesis, add it as a
        # catch-all summary result so callers always get useful material.
        if len(results) < num_results and self._session.synthesis:
            synth_snippet = self._session.synthesis.strip()
            if len(synth_snippet) > _SNIPPET_MAX:
                synth_snippet = synth_snippet[: _SNIPPET_MAX - 1] + "…"
            results.append(
                SearchResult(
                    title=f"[Synthesis] {self._session.topic}",
                    url=f"university://{self._session_id}",
                    snippet=synth_snippet,
                )
            )

        return results[:num_results]

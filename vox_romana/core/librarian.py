"""The Librarian — ephemeral agent that searches the historical notes archive.

The Librarian never persists. It spins up, runs a query, returns results,
and is discarded. Any agent can invoke it at any time.

SEARCH STRATEGY:
  1. Keyword + metadata filter (fast, zero LLM cost).
  2. If keyword search returns nothing useful AND an LLM client is provided,
     fall back to LLM-assisted semantic matching.

USAGE:
    from vox_romana.core.archive import NoteArchive
    from vox_romana.core.librarian import Librarian

    archive = NoteArchive("output/studio_memory/editor_notes.jsonl")
    librarian = Librarian(archive)
    result = librarian.search(
        "repetition issues in debate rounds",
        token_budget=400,
    )
    print(result.text)       # Formatted summary for agent context
    print(result.note_count) # How many notes matched
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .archive import EditorNote, NoteArchive
from .client import DeepSeekClient


@dataclass
class LibrarianResult:
    """What the Librarian returns to the calling agent."""

    text: str
    note_count: int
    query: str
    matched_notes: list[EditorNote] = field(default_factory=list)


_SEMANTIC_PROMPT = (
    "You are a librarian searching through a notes archive. Given the query "
    "and the notes below, select and summarize ONLY the notes that are "
    "relevant to the query. Return a concise summary suitable for injection "
    "into an LLM agent's context. Stay within {token_budget} tokens.\n\n"
    "QUERY: {query}\n\n"
    "NOTES:\n{notes_text}\n\n"
    "Return ONLY the relevant summary. If nothing is relevant, say "
    "'No relevant historical notes found.'"
)


class Librarian:
    """Ephemeral archive search agent.

    Instantiate, call search(), read the result, discard the instance.
    The Librarian has no memory and no persistent session.
    """

    def __init__(
        self,
        archive: NoteArchive,
        client: DeepSeekClient | None = None,
    ) -> None:
        self._archive = archive
        self._client = client

    def search(
        self,
        query: str,
        *,
        token_budget: int = 600,
        filters: dict[str, str] | None = None,
    ) -> LibrarianResult:
        """Search the archive. Returns formatted results within token budget.

        Args:
            query: Natural language search query.
            token_budget: Max tokens for the returned text.
            filters: Optional metadata filters (broadcast_id, status,
                     show_type, chunk_type).
        """
        matched = self._keyword_search(query, filters or {})

        if matched:
            text = self._archive.format_notes_compact(matched, token_budget)
            return LibrarianResult(
                text=text,
                note_count=len(matched),
                query=query,
                matched_notes=matched,
            )

        # Semantic fallback if keyword search found nothing
        if self._client:
            return self._semantic_search(query, token_budget, filters or {})

        return LibrarianResult(
            text="No relevant historical notes found.",
            note_count=0,
            query=query,
        )

    def _keyword_search(
        self,
        query: str,
        filters: dict[str, str],
    ) -> list[EditorNote]:
        """Fast keyword + metadata filter search."""
        notes = self._archive.query(
            broadcast_id=filters.get("broadcast_id"),
            status=filters.get("status"),
            show_type=filters.get("show_type"),
            chunk_type=filters.get("chunk_type"),
        )

        if not query.strip():
            return notes

        keywords = query.lower().split()
        scored: list[tuple[int, EditorNote]] = []

        for note in notes:
            searchable = " ".join([
                note.summary,
                " ".join(note.issues),
                " ".join(note.recommendations),
                note.chunk_type,
                note.show_type,
            ]).lower()

            score = sum(1 for kw in keywords if kw in searchable)
            if score > 0:
                scored.append((score, note))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [note for _, note in scored]

    def _semantic_search(
        self,
        query: str,
        token_budget: int,
        filters: dict[str, str],
    ) -> LibrarianResult:
        """LLM-assisted semantic search fallback."""
        notes = self._archive.query(
            broadcast_id=filters.get("broadcast_id"),
            status=filters.get("status"),
            show_type=filters.get("show_type"),
            chunk_type=filters.get("chunk_type"),
        )

        if not notes:
            return LibrarianResult(
                text="No historical notes in archive.",
                note_count=0,
                query=query,
            )

        notes_text = self._archive.format_notes_compact(notes, token_budget * 2)
        prompt = _SEMANTIC_PROMPT.format(
            query=query,
            token_budget=token_budget,
            notes_text=notes_text,
        )

        result_text = self._client.chat([
            {"role": "system", "content": "You are a precise librarian."},
            {"role": "user", "content": prompt},
        ])

        return LibrarianResult(
            text=result_text,
            note_count=len(notes),
            query=query,
            matched_notes=notes,
        )

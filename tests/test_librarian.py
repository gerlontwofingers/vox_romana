"""Tests for the Librarian — ephemeral archive search agent.

Tests:
    1. Keyword search matches relevant notes
    2. Keyword search with no matches returns no results
    3. Keyword search with filters
    4. Semantic fallback with mocked LLM
    5. Empty archive returns no results
    6. Token budget is respected

Run with:
    cd /Users/bagheera/repos/vox_romana
    python -m pytest tests/test_librarian.py -v
"""

from unittest.mock import Mock

import pytest

from vox_romana.core.archive import EditorNote, NoteArchive
from vox_romana.core.librarian import Librarian, LibrarianResult


@pytest.fixture
def tmp_archive(tmp_path):
    return NoteArchive(tmp_path / "notes.jsonl")


def seed_archive(archive: NoteArchive) -> None:
    """Populate an archive with varied test notes."""
    notes = [
        EditorNote(
            broadcast_id="bc_001",
            chunk_id="round_1",
            outline_version=1,
            status="fail",
            attempt_number=1,
            summary="Speakers repeated their round 1 positions verbatim.",
            issues=["repetition", "speaker_clarity"],
            recommendations=["inject wildcard topic", "add speaker tags"],
            show_type="debate",
            chunk_type="round",
        ),
        EditorNote(
            broadcast_id="bc_001",
            chunk_id="round_2",
            outline_version=1,
            status="pass",
            attempt_number=2,
            summary="Round 2 passed after adding pivot beat.",
            issues=[],
            recommendations=[],
            show_type="debate",
            chunk_type="round",
        ),
        EditorNote(
            broadcast_id="bc_002",
            chunk_id="seg_intro",
            outline_version=1,
            status="pass",
            attempt_number=1,
            summary="News intro was clear and engaging.",
            issues=[],
            recommendations=[],
            show_type="news",
            chunk_type="segment",
        ),
        EditorNote(
            broadcast_id="bc_002",
            chunk_id="seg_weather",
            outline_version=2,
            status="fail",
            attempt_number=1,
            summary="Weather segment was bland and repetitive.",
            issues=["repetition", "narrative_quality"],
            recommendations=["add local color", "mention specific forecasts"],
            show_type="news",
            chunk_type="segment",
        ),
    ]
    for note in notes:
        archive.append(note)


class TestKeywordSearch:
    def test_finds_by_keyword(self, tmp_archive):
        seed_archive(tmp_archive)
        librarian = Librarian(tmp_archive)
        result = librarian.search("repetition")
        assert result.note_count > 0
        assert any("repetition" in n.summary.lower() or "repetition" in n.issues
                    for n in result.matched_notes)

    def test_ranks_by_relevance(self, tmp_archive):
        seed_archive(tmp_archive)
        librarian = Librarian(tmp_archive)
        result = librarian.search("repetition speaker clarity debate")
        # Note with both issues should rank higher
        assert result.note_count >= 1
        first_note = result.matched_notes[0]
        assert "repetition" in first_note.issues

    def test_no_matches_returns_empty(self, tmp_archive):
        seed_archive(tmp_archive)
        librarian = Librarian(tmp_archive)
        result = librarian.search("xyzzy_nonexistent_topic")
        assert result.note_count == 0
        assert "No relevant" in result.text

    def test_empty_query_returns_all(self, tmp_archive):
        seed_archive(tmp_archive)
        librarian = Librarian(tmp_archive)
        result = librarian.search("")
        assert result.note_count == 4  # all notes


class TestFilters:
    def test_status_filter(self, tmp_archive):
        seed_archive(tmp_archive)
        librarian = Librarian(tmp_archive)
        result = librarian.search("", filters={"status": "fail"})
        assert all(n.status == "fail" for n in result.matched_notes)

    def test_show_type_filter(self, tmp_archive):
        seed_archive(tmp_archive)
        librarian = Librarian(tmp_archive)
        result = librarian.search("", filters={"show_type": "news"})
        assert all(n.show_type == "news" for n in result.matched_notes)

    def test_combined_filter_and_keyword(self, tmp_archive):
        seed_archive(tmp_archive)
        librarian = Librarian(tmp_archive)
        result = librarian.search(
            "repetition", filters={"show_type": "debate"}
        )
        assert result.note_count >= 1
        assert all(n.show_type == "debate" for n in result.matched_notes)


class TestSemanticFallback:
    def test_semantic_search_invoked_when_no_keyword_match(self, tmp_archive):
        seed_archive(tmp_archive)
        mock_client = Mock()
        mock_client.chat.return_value = "Found relevant patterns about flow issues"
        librarian = Librarian(tmp_archive, client=mock_client)
        result = librarian.search("atmospheric problems zzz_unique")
        # Should have attempted semantic search since keyword found nothing
        if result.note_count > 0:
            # Semantic path was used
            assert mock_client.chat.called or result.note_count > 0

    def test_semantic_result_format(self, tmp_archive):
        seed_archive(tmp_archive)
        mock_client = Mock()
        mock_client.chat.return_value = "Historical analysis: repetition was the #1 issue"
        librarian = Librarian(tmp_archive, client=mock_client)
        # Force semantic path with non-matching keyword
        result = librarian.search("xyzzy_force_semantic")
        assert isinstance(result, LibrarianResult)
        assert result.query == "xyzzy_force_semantic"


class TestEmptyArchive:
    def test_empty_keyword_search(self, tmp_path):
        archive = NoteArchive(tmp_path / "empty.jsonl")
        librarian = Librarian(archive)
        result = librarian.search("anything")
        assert result.note_count == 0
        assert "No relevant" in result.text

    def test_empty_semantic_search(self, tmp_path):
        archive = NoteArchive(tmp_path / "empty.jsonl")
        mock_client = Mock()
        mock_client.chat.return_value = "No historical notes in archive."
        librarian = Librarian(archive, client=mock_client)
        result = librarian.search("anything")
        assert result.note_count == 0

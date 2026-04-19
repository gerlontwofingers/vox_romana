"""Tests for NoteArchive and related data models.

Tests:
    1. EditorNote creation and default timestamp
    2. EditorNote round-trip (to_dict / from_dict)
    3. NoteArchive append + load_all
    4. NoteArchive query with filters
    5. get_broadcast_ids ordering
    6. get_recent_broadcasts
    7. format_notes_compact with token budget
    8. Empty archive returns empty lists
    9. OutlineArchive append + load_all
   10. OutlineRecord round-trip

Run with:
    cd /Users/bagheera/repos/vox_romana
    python -m pytest tests/test_archive.py -v
"""

import json
from pathlib import Path

import pytest

from vox_romana.core.archive import (
    EditorNote,
    NoteArchive,
    OutlineArchive,
    OutlineRecord,
)


@pytest.fixture
def tmp_archive(tmp_path):
    return NoteArchive(tmp_path / "notes.jsonl")


@pytest.fixture
def tmp_outline_archive(tmp_path):
    return OutlineArchive(tmp_path / "outlines.jsonl")


def make_note(**overrides) -> EditorNote:
    defaults = dict(
        broadcast_id="bc_001",
        chunk_id="chunk_01",
        outline_version=1,
        status="pass",
        attempt_number=1,
        summary="Chunk passed review.",
        issues=[],
        recommendations=[],
        show_type="debate",
        chunk_type="round",
    )
    defaults.update(overrides)
    return EditorNote(**defaults)


class TestEditorNote:
    def test_default_timestamp(self):
        note = make_note()
        assert note.timestamp != ""
        assert "T" in note.timestamp  # ISO format

    def test_round_trip(self):
        note = make_note(issues=["repetition"], recommendations=["add variety"])
        data = note.to_dict()
        restored = EditorNote.from_dict(data)
        assert restored.broadcast_id == note.broadcast_id
        assert restored.issues == ["repetition"]
        assert restored.recommendations == ["add variety"]

    def test_from_dict_ignores_extra_keys(self):
        data = make_note().to_dict()
        data["unknown_field"] = "should be ignored"
        restored = EditorNote.from_dict(data)
        assert restored.broadcast_id == "bc_001"


class TestNoteArchive:
    def test_empty_archive(self, tmp_archive):
        assert tmp_archive.load_all() == []

    def test_append_and_load(self, tmp_archive):
        note = make_note()
        tmp_archive.append(note)
        loaded = tmp_archive.load_all()
        assert len(loaded) == 1
        assert loaded[0].broadcast_id == "bc_001"

    def test_multiple_appends(self, tmp_archive):
        tmp_archive.append(make_note(chunk_id="c1"))
        tmp_archive.append(make_note(chunk_id="c2"))
        tmp_archive.append(make_note(chunk_id="c3"))
        assert len(tmp_archive.load_all()) == 3

    def test_query_no_filters(self, tmp_archive):
        tmp_archive.append(make_note(status="pass"))
        tmp_archive.append(make_note(status="fail"))
        results = tmp_archive.query()
        assert len(results) == 2
        # Newest first
        assert results[0].status == "fail"

    def test_query_by_status(self, tmp_archive):
        tmp_archive.append(make_note(status="pass", chunk_id="c1"))
        tmp_archive.append(make_note(status="fail", chunk_id="c2"))
        tmp_archive.append(make_note(status="pass", chunk_id="c3"))
        results = tmp_archive.query(status="fail")
        assert len(results) == 1
        assert results[0].chunk_id == "c2"

    def test_query_by_broadcast_id(self, tmp_archive):
        tmp_archive.append(make_note(broadcast_id="bc_001"))
        tmp_archive.append(make_note(broadcast_id="bc_002"))
        results = tmp_archive.query(broadcast_id="bc_001")
        assert len(results) == 1

    def test_query_by_show_type(self, tmp_archive):
        tmp_archive.append(make_note(show_type="debate"))
        tmp_archive.append(make_note(show_type="news"))
        results = tmp_archive.query(show_type="news")
        assert len(results) == 1

    def test_query_with_limit(self, tmp_archive):
        for i in range(5):
            tmp_archive.append(make_note(chunk_id=f"c{i}"))
        results = tmp_archive.query(limit=2)
        assert len(results) == 2

    def test_get_broadcast_ids(self, tmp_archive):
        tmp_archive.append(make_note(broadcast_id="a"))
        tmp_archive.append(make_note(broadcast_id="b"))
        tmp_archive.append(make_note(broadcast_id="a"))
        tmp_archive.append(make_note(broadcast_id="c"))
        ids = tmp_archive.get_broadcast_ids()
        assert ids == ["a", "b", "c"]

    def test_get_recent_broadcasts(self, tmp_archive):
        for bc_id in ["a", "b", "c", "d"]:
            tmp_archive.append(make_note(broadcast_id=bc_id))
        recent = tmp_archive.get_recent_broadcasts(n=2)
        bc_ids = {n.broadcast_id for n in recent}
        assert bc_ids == {"c", "d"}

    def test_format_notes_compact(self, tmp_archive):
        tmp_archive.append(make_note(summary="Good chunk"))
        notes = tmp_archive.load_all()
        text = tmp_archive.format_notes_compact(notes, token_budget=600)
        assert "PASS" in text
        assert "Good chunk" in text

    def test_format_notes_respects_budget(self, tmp_archive):
        for i in range(50):
            tmp_archive.append(make_note(
                chunk_id=f"c{i}",
                summary="A" * 200,
            ))
        notes = tmp_archive.load_all()
        text = tmp_archive.format_notes_compact(notes, token_budget=50)
        assert "truncated" in text

    def test_format_empty_notes(self, tmp_archive):
        text = tmp_archive.format_notes_compact([], token_budget=600)
        assert "no historical notes" in text.lower()

    def test_file_created_on_append(self, tmp_archive):
        assert not tmp_archive.path.exists()
        tmp_archive.append(make_note())
        assert tmp_archive.path.exists()


class TestOutlineArchive:
    def test_empty_archive(self, tmp_outline_archive):
        assert tmp_outline_archive.load_all() == []

    def test_append_and_load(self, tmp_outline_archive):
        record = OutlineRecord(
            broadcast_id="bc_001",
            outline_id="ol_001",
            version=1,
            timestamp="2026-04-18T00:00:00Z",
            summary="Initial outline",
            sections_json='[{"id":"s1","title":"Intro"}]',
        )
        tmp_outline_archive.append(record)
        loaded = tmp_outline_archive.load_all()
        assert len(loaded) == 1
        assert loaded[0].outline_id == "ol_001"

    def test_round_trip(self, tmp_outline_archive):
        record = OutlineRecord(
            broadcast_id="bc_001",
            outline_id="ol_002",
            version=2,
            timestamp="2026-04-18T01:00:00Z",
            summary="Updated outline",
            sections_json="[]",
            changed_sections=["s1", "s2"],
        )
        data = record.to_dict()
        restored = OutlineRecord.from_dict(data)
        assert restored.changed_sections == ["s1", "s2"]

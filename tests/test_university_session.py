"""Tests for vox_romana.university.session.

Covers:
    - ScholarNote (frozen, roundtrip, tolerant from_dict)
    - ScholarSession.validate (empty topic, invalid status, bad budget)
    - ScholarSession save / load roundtrip
    - ScholarSession.save_notes_jsonl (DocumentChunk format)
    - ScholarSession.to_jsonl_chunks
    - ScholarSession._update_index (creates index.json, upserts, multi-session)
"""

from __future__ import annotations

import json

import pytest

import vox_romana.core.paths as _paths
import vox_romana.university.session as _sm
from vox_romana.university.session import ScholarNote, ScholarSession


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def isolated_university_root(tmp_path, monkeypatch):
    """Redirect UNIVERSITY_ROOT to a temp dir for every test in this module."""
    monkeypatch.setattr(_paths, "UNIVERSITY_ROOT", tmp_path)
    monkeypatch.setattr(_sm, "UNIVERSITY_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def simple_note():
    return ScholarNote(
        article_title="Roman engineering",
        source_url="https://en.wikipedia.org/wiki/Roman_engineering",
        source_name="wikipedia",
        key_facts=("Used concrete", "Built aqueducts"),
        summary="Romans were great engineers.",
        links_found=("Aqueduct", "Concrete"),
        relevance_score=9.0,
    )


@pytest.fixture
def simple_session(simple_note):
    s = ScholarSession(topic="Roman aqueducts")
    s.notes = [simple_note]
    s.synthesis = "Roman aqueducts were hydraulic marvels."
    s.status = "complete"
    return s


# ══════════════════════════════════════════════════════════════════════════════
# ScholarNote
# ══════════════════════════════════════════════════════════════════════════════


class TestScholarNote:
    def test_frozen(self, simple_note):
        with pytest.raises(Exception):
            simple_note.article_title = "changed"  # type: ignore[misc]

    def test_defaults(self):
        n = ScholarNote(article_title="Minimal")
        assert n.key_facts == ()
        assert n.links_found == ()
        assert n.relevance_score == 0.0
        assert n.timestamp == ""
        assert n.source_url == ""
        assert n.source_name == ""

    def test_to_dict_converts_tuples_to_lists(self, simple_note):
        d = simple_note.to_dict()
        assert isinstance(d["key_facts"], list)
        assert isinstance(d["links_found"], list)
        assert d["key_facts"] == ["Used concrete", "Built aqueducts"]
        assert d["links_found"] == ["Aqueduct", "Concrete"]

    def test_from_dict_roundtrip(self, simple_note):
        d = simple_note.to_dict()
        restored = ScholarNote.from_dict(d)
        assert restored == simple_note

    def test_from_dict_tolerant_of_missing_keys(self):
        n = ScholarNote.from_dict({"article_title": "Minimal"})
        assert n.key_facts == ()
        assert n.links_found == ()
        assert n.relevance_score == 0.0

    def test_from_dict_empty_dict(self):
        n = ScholarNote.from_dict({})
        assert n.article_title == ""

    def test_score_coerced_to_float(self):
        n = ScholarNote.from_dict({"article_title": "X", "relevance_score": "7"})
        assert n.relevance_score == 7.0


# ══════════════════════════════════════════════════════════════════════════════
# ScholarSession.validate
# ══════════════════════════════════════════════════════════════════════════════


class TestScholarSessionValidate:
    def test_valid_session_no_problems(self, simple_session):
        assert simple_session.validate() == []

    def test_empty_topic_flagged(self):
        s = ScholarSession(topic="")
        problems = s.validate()
        assert any("topic" in p for p in problems)

    def test_whitespace_only_topic_flagged(self):
        s = ScholarSession(topic="   ")
        assert s.validate()

    def test_invalid_status_flagged(self):
        s = ScholarSession(topic="valid topic", status="unknown_status")
        problems = s.validate()
        assert any("status" in p for p in problems)

    def test_negative_budget_flagged(self):
        s = ScholarSession(topic="valid topic", budget_hours=-1.0)
        problems = s.validate()
        assert any("budget" in p for p in problems)

    def test_zero_budget_flagged(self):
        s = ScholarSession(topic="valid topic", budget_hours=0.0)
        assert s.validate()

    def test_none_budget_ok(self):
        s = ScholarSession(topic="valid topic", budget_hours=None)
        assert s.validate() == []

    def test_all_valid_statuses_accepted(self):
        for status in ("active", "complete", "paused", "interrupted"):
            s = ScholarSession(topic="topic", status=status)
            problems = s.validate()
            assert not any("status" in p for p in problems), (
                f"status={status!r} should be valid but got: {problems}"
            )

    def test_multiple_problems_returned_together(self):
        s = ScholarSession(topic="", status="bad", budget_hours=-1.0)
        problems = s.validate()
        assert len(problems) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# ScholarSession persistence (save / load)
# ══════════════════════════════════════════════════════════════════════════════


class TestScholarSessionPersistence:
    def test_save_creates_session_json(self, simple_session, tmp_path):
        path = simple_session.save(tmp_path)
        assert path.exists()
        assert path.name == "session.json"

    def test_save_returns_path_to_json(self, simple_session, tmp_path):
        path = simple_session.save(tmp_path)
        assert path.suffix == ".json"
        data = json.loads(path.read_text())
        assert data["session_id"] == simple_session.session_id

    def test_load_roundtrip(self, simple_session, tmp_path):
        simple_session.save(tmp_path)
        loaded = ScholarSession.load(simple_session.session_dir(tmp_path))
        assert loaded.session_id == simple_session.session_id
        assert loaded.topic == simple_session.topic
        assert loaded.synthesis == simple_session.synthesis
        assert loaded.status == simple_session.status
        assert len(loaded.notes) == 1
        assert loaded.notes[0] == simple_session.notes[0]

    def test_load_missing_session_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ScholarSession.load(tmp_path / "nonexistent_session_id")

    def test_load_malformed_json_raises_value_error(self, tmp_path):
        sdir = tmp_path / "bad_session"
        sdir.mkdir()
        (sdir / "session.json").write_text("NOT JSON", encoding="utf-8")
        with pytest.raises(ValueError):
            ScholarSession.load(sdir)

    def test_save_updates_updated_at(self, simple_session, tmp_path):
        simple_session.save(tmp_path)
        assert simple_session.updated_at  # non-empty ISO timestamp

    def test_load_preserves_notes_count(self, simple_note, tmp_path):
        s = ScholarSession(topic="Multi-note")
        s.notes = [simple_note, simple_note]
        s.save(tmp_path)
        loaded = ScholarSession.load(s.session_dir(tmp_path))
        assert len(loaded.notes) == 2

    def test_load_preserves_optional_fields(self, tmp_path):
        s = ScholarSession(
            topic="topic",
            budget_hours=2.5,
            estimated_hours=1.5,
            catalog_study_id="abc123",
        )
        s.save(tmp_path)
        loaded = ScholarSession.load(s.session_dir(tmp_path))
        assert loaded.budget_hours == 2.5
        assert loaded.estimated_hours == 1.5
        assert loaded.catalog_study_id == "abc123"


# ══════════════════════════════════════════════════════════════════════════════
# ScholarSession.save_notes_jsonl
# ══════════════════════════════════════════════════════════════════════════════


class TestSaveNotesJsonl:
    def test_creates_notes_jsonl(self, simple_session, tmp_path):
        simple_session.save(tmp_path)  # creates the session dir
        path = simple_session.save_notes_jsonl(tmp_path)
        assert path.exists()
        assert path.name == "notes.jsonl"

    def test_one_line_per_note(self, simple_session, tmp_path):
        simple_session.save(tmp_path)
        path = simple_session.save_notes_jsonl(tmp_path)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_document_chunk_format(self, simple_session, tmp_path):
        simple_session.save(tmp_path)
        path = simple_session.save_notes_jsonl(tmp_path)
        chunk = json.loads(path.read_text().splitlines()[0])
        assert "text" in chunk
        assert "title" in chunk
        assert chunk["format"] == ".university_note"
        assert chunk["tag"] == "university_research"

    def test_empty_notes_creates_empty_file(self, tmp_path):
        s = ScholarSession(topic="Empty")
        s.save(tmp_path)
        path = s.save_notes_jsonl(tmp_path)
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip() == ""


# ══════════════════════════════════════════════════════════════════════════════
# ScholarSession.to_jsonl_chunks
# ══════════════════════════════════════════════════════════════════════════════


class TestToJsonlChunks:
    def test_returns_one_chunk_per_note(self, simple_session):
        chunks = simple_session.to_jsonl_chunks()
        assert len(chunks) == 1

    def test_chunk_title_matches_note_article_title(self, simple_session):
        chunk = simple_session.to_jsonl_chunks()[0]
        assert chunk["title"] == "Roman engineering"

    def test_chunk_source_preserves_note_source_url(self, simple_session):
        chunk = simple_session.to_jsonl_chunks()[0]
        assert chunk["source"] == simple_session.notes[0].source_url

    def test_chunk_source_falls_back_to_university_scheme_without_provenance(self):
        s = ScholarSession(topic="Roman aqueducts")
        s.notes = [ScholarNote(article_title="Roman engineering")]
        chunk = s.to_jsonl_chunks()[0]
        assert chunk["source"].startswith("university://")
        assert s.session_id in chunk["source"]

    def test_chunk_index_and_total(self, simple_session):
        chunk = simple_session.to_jsonl_chunks()[0]
        assert chunk["chunk_index"] == 0
        assert chunk["total_chunks"] == 1

    def test_text_includes_summary(self, simple_session):
        chunk = simple_session.to_jsonl_chunks()[0]
        assert "Romans were great engineers" in chunk["text"]

    def test_text_includes_key_facts(self, simple_session):
        chunk = simple_session.to_jsonl_chunks()[0]
        assert "Used concrete" in chunk["text"]

    def test_empty_session_returns_empty_list(self):
        s = ScholarSession(topic="empty")
        assert s.to_jsonl_chunks() == []

    def test_multiple_notes_all_indexed(self, simple_note, tmp_path):
        note2 = ScholarNote(article_title="Second article", relevance_score=5.0)
        s = ScholarSession(topic="multi")
        s.notes = [simple_note, note2]
        chunks = s.to_jsonl_chunks()
        assert len(chunks) == 2
        assert chunks[0]["chunk_index"] == 0
        assert chunks[1]["chunk_index"] == 1
        assert chunks[0]["total_chunks"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# Index registry (_update_index)
# ══════════════════════════════════════════════════════════════════════════════


class TestIndexRegistry:
    def test_save_creates_index_json(self, simple_session, tmp_path):
        simple_session.save(tmp_path)
        index_path = tmp_path / "index.json"
        assert index_path.exists()

    def test_index_contains_session_record(self, simple_session, tmp_path):
        simple_session.save(tmp_path)
        entries = json.loads((tmp_path / "index.json").read_text())
        assert isinstance(entries, list)
        ids = [e["session_id"] for e in entries]
        assert simple_session.session_id in ids

    def test_index_upserts_on_resave(self, simple_session, tmp_path):
        simple_session.save(tmp_path)
        simple_session.status = "paused"
        simple_session.save(tmp_path)
        entries = json.loads((tmp_path / "index.json").read_text())
        matching = [
            e for e in entries if e["session_id"] == simple_session.session_id
        ]
        assert len(matching) == 1
        assert matching[0]["status"] == "paused"

    def test_multiple_sessions_all_in_index(self, tmp_path):
        s1 = ScholarSession(topic="Topic A")
        s2 = ScholarSession(topic="Topic B")
        s1.save(tmp_path)
        s2.save(tmp_path)
        entries = json.loads((tmp_path / "index.json").read_text())
        ids = {e["session_id"] for e in entries}
        assert s1.session_id in ids
        assert s2.session_id in ids
        assert len(ids) == 2

    def test_index_record_has_expected_keys(self, simple_session, tmp_path):
        simple_session.save(tmp_path)
        entries = json.loads((tmp_path / "index.json").read_text())
        record = next(
            e for e in entries if e["session_id"] == simple_session.session_id
        )
        for key in ("session_id", "topic", "status", "updated_at"):
            assert key in record, f"missing key: {key}"

"""Tests for the Catalog — local SQLite content log.

Tests cover:
    1. Schema creation (in-memory DB)
    2. log_run + get_run round-trip
    3. Tags (add at creation, add later, dedup)
    4. FTS5 search (title, brief, ranking)
    5. recent() with filters
    6. log_study + get_studies
    7. Episode tracking (next_episode_number)
    8. count()
    9. Context manager
   10. Auditions + series asset pool (A1)
"""

import json

import pytest

from vox_romana.core.catalog import Catalog, RunRecord, StudyRecord, AuditionRecord, AssetRecord


@pytest.fixture
def catalog():
    """In-memory catalog for testing."""
    cat = Catalog(db_path=":memory:")
    yield cat
    cat.close()


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA
# ═══════════════════════════════════════════════════════════════════════════


class TestSchema:
    """Verify the database schema initializes correctly."""

    def test_tables_created(self, catalog):
        """All expected tables should exist."""
        rows = catalog._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {row["name"] for row in rows}
        assert "runs" in names
        assert "tags" in names
        assert "studies" in names

    def test_fts_table_created(self, catalog):
        """FTS5 virtual table should exist."""
        rows = catalog._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs_fts'"
        ).fetchall()
        assert len(rows) == 1

    def test_idempotent_init(self, catalog):
        """Calling _init_schema again should not fail."""
        catalog._init_schema()


# ═══════════════════════════════════════════════════════════════════════════
# LOG + GET RUN
# ═══════════════════════════════════════════════════════════════════════════


class TestLogRun:
    """Tests for logging and retrieving runs."""

    def test_log_run_returns_id(self, catalog):
        run_id = catalog.log_run(content_type="debate", title="Test")
        assert isinstance(run_id, str)
        assert len(run_id) == 12

    def test_get_run_round_trip(self, catalog):
        run_id = catalog.log_run(
            content_type="debate",
            title="UBI Debate",
            brief="Two economists debate UBI",
            config={"target_turns": 40},
            output_paths=["output/text/debates/ubi.json"],
            duration_seconds=925,
            cost_estimate=0.15,
        )

        run = catalog.get_run(run_id)
        assert run is not None
        assert run.run_id == run_id
        assert run.content_type == "debate"
        assert run.title == "UBI Debate"
        assert run.brief == "Two economists debate UBI"
        assert run.output_paths == ["output/text/debates/ubi.json"]
        assert run.duration_seconds == 925
        assert run.cost_estimate == 0.15
        assert json.loads(run.config_json) == {"target_turns": 40}

    def test_get_run_not_found(self, catalog):
        assert catalog.get_run("nonexistent") is None

    def test_log_run_minimal(self, catalog):
        """Should work with just content_type."""
        run_id = catalog.log_run(content_type="audiobook")
        run = catalog.get_run(run_id)
        assert run is not None
        assert run.content_type == "audiobook"
        assert run.title == ""
        assert run.output_paths == []

    def test_log_run_with_series(self, catalog):
        run_id = catalog.log_run(
            content_type="news",
            title="Morning Brief Ep. 1",
            series_id="morning-brief",
            episode_number=1,
        )
        run = catalog.get_run(run_id)
        assert run.series_id == "morning-brief"
        assert run.episode_number == 1

    def test_log_run_with_pipeline(self, catalog):
        tools = ["bootstrap", "conductor", "tts_renderer", "mixer"]
        run_id = catalog.log_run(
            content_type="debate",
            title="Pipeline Test",
            pipeline=tools,
        )
        run = catalog.get_run(run_id)
        assert run.pipeline == tools

    def test_log_run_pipeline_defaults_empty(self, catalog):
        run_id = catalog.log_run(content_type="debate")
        run = catalog.get_run(run_id)
        assert run.pipeline == []


# ═══════════════════════════════════════════════════════════════════════════
# TAGS
# ═══════════════════════════════════════════════════════════════════════════


class TestTags:
    """Tests for tagging runs."""

    def test_tags_at_creation(self, catalog):
        run_id = catalog.log_run(
            content_type="debate",
            title="Test",
            tags=["economics", "UBI"],
        )
        run = catalog.get_run(run_id)
        assert set(run.tags) == {"economics", "UBI"}

    def test_add_tags_later(self, catalog):
        run_id = catalog.log_run(content_type="debate", title="Test")
        catalog.add_tags(run_id, ["policy", "welfare"])
        run = catalog.get_run(run_id)
        assert set(run.tags) == {"policy", "welfare"}

    def test_duplicate_tags_ignored(self, catalog):
        run_id = catalog.log_run(
            content_type="debate",
            title="Test",
            tags=["econ"],
        )
        catalog.add_tags(run_id, ["econ", "new"])
        run = catalog.get_run(run_id)
        assert "econ" in run.tags
        assert "new" in run.tags


# ═══════════════════════════════════════════════════════════════════════════
# FTS5 SEARCH
# ═══════════════════════════════════════════════════════════════════════════


class TestSearch:
    """Tests for full-text search."""

    def test_search_by_title(self, catalog):
        catalog.log_run(content_type="debate", title="UBI Debate")
        catalog.log_run(content_type="debate", title="Climate Policy")
        results = catalog.search("UBI")
        assert len(results) == 1
        assert results[0].title == "UBI Debate"

    def test_search_by_brief(self, catalog):
        catalog.log_run(
            content_type="debate",
            title="Test",
            brief="Universal basic income and welfare reform",
        )
        results = catalog.search("welfare reform")
        assert len(results) == 1

    def test_search_no_results(self, catalog):
        catalog.log_run(content_type="debate", title="Climate")
        results = catalog.search("cryptocurrency")
        assert len(results) == 0

    def test_search_limit(self, catalog):
        for i in range(10):
            catalog.log_run(content_type="debate", title=f"Debate about AI #{i}")
        results = catalog.search("AI", limit=3)
        assert len(results) == 3

    def test_search_by_content_type(self, catalog):
        catalog.log_run(content_type="debate", title="A debate")
        catalog.log_run(content_type="news", title="A news show")
        results = catalog.search("debate")
        assert len(results) >= 1
        assert any(r.content_type == "debate" for r in results)


# ═══════════════════════════════════════════════════════════════════════════
# RECENT
# ═══════════════════════════════════════════════════════════════════════════


class TestRecent:
    """Tests for recent() retrieval."""

    def test_recent_all(self, catalog):
        catalog.log_run(content_type="debate", title="First")
        catalog.log_run(content_type="news", title="Second")
        results = catalog.recent()
        assert len(results) == 2
        # Newest first
        assert results[0].title == "Second"

    def test_recent_by_type(self, catalog):
        catalog.log_run(content_type="debate", title="D1")
        catalog.log_run(content_type="news", title="N1")
        catalog.log_run(content_type="debate", title="D2")

        debates = catalog.recent(content_type="debate")
        assert len(debates) == 2
        assert all(r.content_type == "debate" for r in debates)

    def test_recent_by_series(self, catalog):
        catalog.log_run(content_type="news", title="Ep1", series_id="s1")
        catalog.log_run(content_type="news", title="Ep2", series_id="s1")
        catalog.log_run(content_type="news", title="Other", series_id="s2")

        results = catalog.recent(series_id="s1")
        assert len(results) == 2

    def test_recent_limit(self, catalog):
        for i in range(5):
            catalog.log_run(content_type="debate", title=f"D{i}")
        results = catalog.recent(limit=2)
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════════════
# STUDIES
# ═══════════════════════════════════════════════════════════════════════════


class TestStudies:
    """Tests for data study logging."""

    def test_log_study(self, catalog):
        run_id = catalog.log_run(content_type="debate", title="UBI")
        study_id = catalog.log_study(
            run_id=run_id,
            content_type="debate",
            dataset="finland_ubi.csv",
            method="t-test",
            findings="Employment rose 4.8%",
            fact_cards=[{"claim": "Employment rose 4.8%", "source": "KELA"}],
        )
        assert isinstance(study_id, str)
        assert len(study_id) == 12

    def test_get_studies(self, catalog):
        run_id = catalog.log_run(content_type="debate", title="UBI")
        catalog.log_study(
            run_id=run_id, content_type="debate",
            dataset="a.csv", method="m1", findings="f1",
        )
        catalog.log_study(
            run_id=run_id, content_type="debate",
            dataset="b.csv", method="m2", findings="f2",
        )

        studies = catalog.get_studies(run_id)
        assert len(studies) == 2
        assert studies[0].dataset == "a.csv"
        assert studies[1].dataset == "b.csv"

    def test_study_without_run(self, catalog):
        """Studies can exist independently (no linked run)."""
        study_id = catalog.log_study(
            content_type="news",
            dataset="standalone.csv",
            method="regression",
            findings="Strong correlation",
        )
        assert isinstance(study_id, str)

    def test_study_content_type_and_kind(self, catalog):
        """content_type and kind are stored and retrievable."""
        run_id = catalog.log_run(content_type="news", title="Morning Brief")
        catalog.log_study(
            run_id=run_id,
            content_type="news",
            kind="research",
            dataset="employment.csv",
            method="trend",
            findings="Unemployment fell",
        )
        studies = catalog.get_studies(run_id)
        assert studies[0].content_type == "news"
        assert studies[0].kind == "research"

    def test_study_kind_defaults_to_research(self, catalog):
        """kind defaults to 'research' when not specified."""
        study_id = catalog.log_study(dataset="x.csv", method="m", findings="f")
        # Read it back directly
        row = catalog._conn.execute(
            "SELECT kind FROM studies WHERE study_id = ?", (study_id,)
        ).fetchone()
        assert row["kind"] == "research"

    def test_study_custom_kind(self, catalog):
        """kind can be set to a custom value."""
        study_id = catalog.log_study(
            content_type="debate",
            kind="fact-check",
            dataset="claims.csv",
            method="verification",
            findings="3 of 5 claims verified",
        )
        row = catalog._conn.execute(
            "SELECT kind FROM studies WHERE study_id = ?", (study_id,)
        ).fetchone()
        assert row["kind"] == "fact-check"


# ═══════════════════════════════════════════════════════════════════════════
# EPISODE TRACKING
# ═══════════════════════════════════════════════════════════════════════════


class TestEpisodeTracking:
    """Tests for series/episode number management."""

    def test_next_episode_fresh_series(self, catalog):
        assert catalog.next_episode_number("new-series") == 1

    def test_next_episode_increments(self, catalog):
        catalog.log_run(
            content_type="news",
            series_id="daily",
            episode_number=1,
        )
        catalog.log_run(
            content_type="news",
            series_id="daily",
            episode_number=2,
        )
        assert catalog.next_episode_number("daily") == 3

    def test_next_episode_ignores_other_series(self, catalog):
        catalog.log_run(content_type="news", series_id="s1", episode_number=5)
        catalog.log_run(content_type="news", series_id="s2", episode_number=10)
        assert catalog.next_episode_number("s1") == 6


# ═══════════════════════════════════════════════════════════════════════════
# COUNT
# ═══════════════════════════════════════════════════════════════════════════


class TestCount:
    """Tests for counting runs."""

    def test_count_empty(self, catalog):
        assert catalog.count() == 0

    def test_count_all(self, catalog):
        catalog.log_run(content_type="debate")
        catalog.log_run(content_type="news")
        assert catalog.count() == 2

    def test_count_by_type(self, catalog):
        catalog.log_run(content_type="debate")
        catalog.log_run(content_type="debate")
        catalog.log_run(content_type="news")
        assert catalog.count(content_type="debate") == 2
        assert catalog.count(content_type="news") == 1


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════════


class TestContextManager:
    """Tests for the context manager interface."""

    def test_with_statement(self):
        with Catalog(db_path=":memory:") as cat:
            run_id = cat.log_run(content_type="debate", title="Test")
            assert cat.get_run(run_id) is not None


# ═══════════════════════════════════════════════════════════════════════════
# AUDITIONS + SERIES ASSET POOL  (A1)
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditions:
    """Tests for audition session logging."""

    def test_log_audition_returns_pk(self, catalog):
        pk = catalog.log_audition(
            session_id="sess001",
            audition_type="voice",
        )
        assert isinstance(pk, int) and pk > 0

    def test_get_audition_round_trip(self, catalog):
        results = [{"label": "af_heart", "path": "/tmp/af_heart.wav", "score": 0.9}]
        catalog.log_audition(
            session_id="sess002",
            series_id="series_abc",
            audition_type="voice",
            brief="Warm female anchor",
            results=results,
        )
        rec = catalog.get_audition("sess002")
        assert rec is not None
        assert rec.session_id == "sess002"
        assert rec.series_id == "series_abc"
        assert rec.audition_type == "voice"
        assert rec.brief == "Warm female anchor"
        assert json.loads(rec.results_json)[0]["label"] == "af_heart"
        assert rec.chosen is None

    def test_get_audition_not_found(self, catalog):
        assert catalog.get_audition("nonexistent") is None

    def test_audition_tables_exist(self, catalog):
        rows = catalog._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {row["name"] for row in rows}
        assert "auditions" in names
        assert "series_assets" in names

    def test_duplicate_session_id_raises(self, catalog):
        catalog.log_audition(session_id="dup001", audition_type="music")
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            catalog.log_audition(session_id="dup001", audition_type="music")


class TestSeriesAssets:
    """Tests for the series asset pool."""

    def _make_series(self, catalog) -> str:
        return catalog.create_series(title="Test Series")

    def test_approve_asset_returns_pk(self, catalog):
        sid = self._make_series(catalog)
        pk = catalog.approve_asset(
            series_id=sid,
            asset_type="voice",
            asset_id="af_heart",
            character_name="Anchor A",
        )
        assert isinstance(pk, int) and pk > 0

    def test_list_assets_empty(self, catalog):
        sid = self._make_series(catalog)
        assert catalog.list_assets(sid) == []

    def test_list_assets_round_trip(self, catalog):
        sid = self._make_series(catalog)
        catalog.approve_asset(series_id=sid, asset_type="voice",
                               asset_id="af_heart", character_name="Alice")
        catalog.approve_asset(series_id=sid, asset_type="music",
                               asset_id="lo_fi_jazz")
        assets = catalog.list_assets(sid)
        assert len(assets) == 2
        types = {a.asset_type for a in assets}
        assert types == {"voice", "music"}

    def test_list_assets_filter_by_type(self, catalog):
        sid = self._make_series(catalog)
        catalog.approve_asset(series_id=sid, asset_type="voice", asset_id="v1")
        catalog.approve_asset(series_id=sid, asset_type="music", asset_id="m1")
        voices = catalog.list_assets(sid, asset_type="voice")
        assert len(voices) == 1
        assert voices[0].asset_id == "v1"

    def test_remove_asset(self, catalog):
        sid = self._make_series(catalog)
        pk = catalog.approve_asset(series_id=sid, asset_type="visual",
                                    asset_id="dark_studio")
        assert len(catalog.list_assets(sid)) == 1
        catalog.remove_asset(pk)
        assert catalog.list_assets(sid) == []

    def test_approve_all_from_audition(self, catalog):
        sid = self._make_series(catalog)
        results = [
            {"asset_id": "af_heart", "character_name": "Alice", "path": "/a.wav"},
            {"asset_id": "am_adam", "character_name": "Bob",   "path": "/b.wav"},
        ]
        catalog.log_audition(
            session_id="sess_bulk",
            series_id=sid,
            audition_type="voice",
            results=results,
        )
        pks = catalog.approve_all_from_audition("sess_bulk", series_id=sid)
        assert len(pks) == 2
        assets = catalog.list_assets(sid)
        assert len(assets) == 2
        asset_ids = {a.asset_id for a in assets}
        assert asset_ids == {"af_heart", "am_adam"}

    def test_approve_all_marks_chosen(self, catalog):
        sid = self._make_series(catalog)
        catalog.log_audition(
            session_id="sess_chosen",
            series_id=sid,
            audition_type="music",
            results=[{"asset_id": "lo_fi", "path": "/lo.wav"}],
        )
        catalog.approve_all_from_audition("sess_chosen", series_id=sid)
        rec = catalog.get_audition("sess_chosen")
        assert rec.chosen == "all"

    def test_approve_all_missing_session_raises(self, catalog):
        with pytest.raises(ValueError, match="not found"):
            catalog.approve_all_from_audition("ghost_session", series_id="any")

    def test_asset_record_fields(self, catalog):
        sid = self._make_series(catalog)
        pk = catalog.approve_asset(
            series_id=sid,
            asset_type="voice",
            asset_id="af_sky",
            character_name="Narrator",
            source_path="/clips/narrator.wav",
            notes="soft delivery",
        )
        assets = catalog.list_assets(sid)
        assert len(assets) == 1
        a = assets[0]
        assert a.id == pk
        assert a.series_id == sid
        assert a.asset_type == "voice"
        assert a.character_name == "Narrator"
        assert a.asset_id == "af_sky"
        assert a.source_path == "/clips/narrator.wav"
        assert a.notes == "soft delivery"

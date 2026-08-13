"""Tests for vox_romana.analysis.longitudinal."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from vox_romana.analysis.longitudinal import (
    _compute_piece_metrics,
    _extract_x_params,
    _safe_mean,
    init_db,
    ingest_run,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_debate_piece() -> dict:
    return {
        "template_id": "conductor",
        "debate_id": "abc123",
        "title": "Test Debate",
        "chunks": [],
        "events": [
            {
                "global_order": 1,
                "speaker_name": "Alice",
                "text": "Hello world testing this out for a long debate speech.",
                "pace": "steady",
                "nonverbal": [],
            },
            {
                "global_order": 2,
                "speaker_name": "Bob",
                "text": "I disagree with that argument entirely.",
                "pace": "steady",
                "nonverbal": [],
            },
            {
                "global_order": 3,
                "speaker_name": "Alice",
                "text": "Let me explain my position more clearly.",
                "pace": "fast",
                "nonverbal": [],
            },
        ],
        "participants": ["Alice", "Bob"],
        "moderators": ["Charlie"],
        "hosts": [],
    }


def _make_news_piece() -> dict:
    return {
        "show_id": "n1",
        "title": "Test News",
        "events": [
            {"global_order": 1, "speaker_name": "Anchor1", "text": "Breaking news here.", "pace": "fast", "nonverbal": []},
            {"global_order": 2, "speaker_name": "Anchor1", "text": "More details follow.", "pace": "fast", "nonverbal": []},
        ],
        "hosts": ["Host"],
        "anchors": ["Anchor1"],
        "segments": [
            {"segment_id": "s1", "segment_order": 0, "events": []},
            {"segment_id": "s2", "segment_order": 1, "events": []},
        ],
        "segment_count": 2,
    }


def _make_snapshot() -> dict:
    return {
        "generated_at": "2026-05-24T12:00:00+00:00",
        "git_sha": "deadbeef",
        "universal": {
            "overshoot_compensation": 0.92,
            "wrap_up_threshold": 0.70,
        },
        "by_show": {
            "debate": {
                "wpm_default": 150,
                "lead_in_ms": 1000,
                "max_beat_words": None,
                "avg_words_per_turn": 40,
                "host_overhead_words": 280,
                "closing_window_extra": 3,
            },
            "news": {
                "wpm_default": 165,
                "lead_in_ms": 800,
                "max_beat_words": 80,
                "avg_words_per_segment": 180,
                "host_overhead_words": 120,
            },
        },
    }


def _make_study_dir(
    tmp_path: Path,
    *,
    show_type: str = "debate",
    target_duration_mins: float = 3.0,
    exit_code: int = 0,
    word_count: int = 280,
    event_count: int = 3,
) -> Path:
    """Build a minimal fake study directory for use in tests."""
    study_dir = tmp_path / "20260524_fake"
    study_dir.mkdir()

    piece_data = _make_debate_piece() if show_type == "debate" else _make_news_piece()

    piece_dir = study_dir / show_type / "test-topic"
    piece_dir.mkdir(parents=True)
    piece_filename = f"{show_type}_brief_20260524_fake.json"
    (piece_dir / piece_filename).write_text(
        json.dumps(piece_data), encoding="utf-8"
    )

    snapshot = _make_snapshot()
    (study_dir / "params_snapshot.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )

    index = {
        "aggregate": {
            "generated_at": "2026-05-24T12:00:00+00:00",
            "target_duration_mins": target_duration_mins,
            "total_pieces": 1,
            "total_ok": 1 if exit_code == 0 else 0,
            "total_failed": 0 if exit_code == 0 else 1,
            "total_elapsed_secs": 45.0,
            "by_show": {
                show_type: {
                    "n": 1,
                    "ok": 1 if exit_code == 0 else 0,
                    "failed": 0 if exit_code == 0 else 1,
                    "elapsed_secs": 45.0,
                    "avg_word_count": word_count,
                    "avg_event_count": event_count,
                },
            },
        },
        "results": [
            {
                "show_type": show_type,
                "topic": "Test topic",
                "slug": "test-topic",
                "started_at": "2026-05-24T12:00:00+00:00",
                "elapsed_secs": 45.0,
                "exit_code": exit_code,
                "timed_out": False,
                "ok": exit_code == 0,
                "json_path": f"{show_type}/test-topic/{piece_filename}",
                "stderr_tail": [],
                "summary": {
                    "title": "Test Debate" if show_type == "debate" else "Test News",
                    "show_type": show_type,
                    "event_count": event_count,
                    "word_count": word_count,
                    "speakers": ["Alice", "Bob"],
                    "nonverbal_cue_count": 0,
                    "cast": [{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}],
                    "cast_size": 3,
                    "voice_hint_count": 0,
                },
            }
        ],
    }
    (study_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")

    return study_dir


# ─────────────────────────────────────────────────────────────────────────────
# _safe_mean
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeMean:
    def test_normal(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_with_none(self):
        assert _safe_mean([1.0, None, 3.0]) == pytest.approx(2.0)

    def test_all_none(self):
        assert _safe_mean([None, None]) is None

    def test_empty(self):
        assert _safe_mean([]) is None


# ─────────────────────────────────────────────────────────────────────────────
# _extract_x_params
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractXParams:
    def test_debate_params(self):
        snapshot = _make_snapshot()
        params = _extract_x_params(snapshot, "debate")
        assert params["overshoot_compensation"] == pytest.approx(0.92)
        assert params["wrap_up_threshold"] == pytest.approx(0.70)
        assert params["wpm_default"] == 150
        assert params["lead_in_ms"] == 1000
        assert params["avg_words_per_turn"] == 40
        assert params["closing_window_extra"] == 3
        assert params["max_beat_words"] is None

    def test_news_params(self):
        snapshot = _make_snapshot()
        params = _extract_x_params(snapshot, "news")
        assert params["max_beat_words"] == 80
        assert params["avg_words_per_segment"] == 180
        assert params["wpm_default"] == 165

    def test_missing_show_type_returns_nones(self):
        snapshot = _make_snapshot()
        params = _extract_x_params(snapshot, "lecture")
        assert params["overshoot_compensation"] == pytest.approx(0.92)  # universal still set
        assert params["wpm_default"] is None  # not in snapshot
        assert params["max_beat_words"] is None

    def test_empty_snapshot(self):
        params = _extract_x_params({}, "debate")
        for v in params.values():
            assert v is None


# ─────────────────────────────────────────────────────────────────────────────
# _compute_piece_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePieceMetrics:
    def _make_telemetry(self, word_count=280, event_count=3, cast_size=3, exit_code=0):
        return {
            "show_type": "debate",
            "slug": "test-topic",
            "exit_code": exit_code,
            "elapsed_secs": 45.0,
            "ok": exit_code == 0,
            "summary": {
                "word_count": word_count,
                "event_count": event_count,
                "nonverbal_cue_count": 0,
                "cast_size": cast_size,
                "speakers": ["Alice", "Bob"],
            },
        }

    def test_word_ratio(self):
        m = _compute_piece_metrics(
            piece_data=None,
            telemetry=self._make_telemetry(word_count=450),
            show_type="debate",
            target_words=450,
            target_duration_mins=3.0,
        )
        assert m["word_ratio"] == pytest.approx(1.0)

    def test_word_ratio_with_no_target(self):
        m = _compute_piece_metrics(
            piece_data=None,
            telemetry=self._make_telemetry(word_count=280),
            show_type="debate",
            target_words=None,
            target_duration_mins=3.0,
        )
        assert m["word_ratio"] is None

    def test_words_per_event(self):
        m = _compute_piece_metrics(
            piece_data=None,
            telemetry=self._make_telemetry(word_count=300, event_count=10),
            show_type="debate",
            target_words=450,
            target_duration_mins=3.0,
        )
        assert m["words_per_event"] == pytest.approx(30.0)

    def test_nonverbal_rate_zero(self):
        m = _compute_piece_metrics(
            piece_data=None,
            telemetry=self._make_telemetry(event_count=5),
            show_type="debate",
            target_words=450,
            target_duration_mins=3.0,
        )
        assert m["nonverbal_rate"] == pytest.approx(0.0)

    def test_name_diversity(self):
        m = _compute_piece_metrics(
            piece_data=None,
            telemetry=self._make_telemetry(cast_size=4),
            show_type="debate",
            target_words=450,
            target_duration_mins=3.0,
        )
        # 2 unique speakers / 4 cast slots = 0.5
        assert m["name_diversity"] == pytest.approx(0.5)

    def test_pace_monotony_from_piece_data(self):
        piece = _make_debate_piece()
        # events: steady, steady, fast → 2/3 ≈ 0.667
        m = _compute_piece_metrics(
            piece_data=piece,
            telemetry=self._make_telemetry(),
            show_type="debate",
            target_words=450,
            target_duration_mins=3.0,
        )
        assert m["pace_monotony"] == pytest.approx(2 / 3)

    def test_dominant_speaker_share_from_piece_data(self):
        piece = _make_debate_piece()
        # Alice: 2 events, Bob: 1 event → dominant = 2/3
        m = _compute_piece_metrics(
            piece_data=piece,
            telemetry=self._make_telemetry(),
            show_type="debate",
            target_words=450,
            target_duration_mins=3.0,
        )
        assert m["dominant_speaker_share"] == pytest.approx(2 / 3)

    def test_debate_architect_fields(self):
        piece = _make_debate_piece()
        m = _compute_piece_metrics(
            piece_data=piece,
            telemetry=self._make_telemetry(),
            show_type="debate",
            target_words=450,
            target_duration_mins=3.0,
        )
        assert m["architect_host_count"] == 1   # 1 moderator + 0 hosts
        assert m["architect_section_count"] == 2  # 2 participants
        assert m["architect_target_turns"] is None  # not available in output JSON

    def test_news_architect_fields(self):
        piece = _make_news_piece()
        telemetry = {
            "show_type": "news",
            "slug": "news-topic",
            "exit_code": 0,
            "elapsed_secs": 30.0,
            "ok": True,
            "summary": {
                "word_count": 200,
                "event_count": 2,
                "nonverbal_cue_count": 0,
                "cast_size": 2,
                "speakers": ["Anchor1"],
            },
        }
        m = _compute_piece_metrics(
            piece_data=piece,
            telemetry=telemetry,
            show_type="news",
            target_words=495,
            target_duration_mins=3.0,
        )
        assert m["architect_section_count"] == 2  # 2 segments
        assert m["architect_host_count"] == 2  # 1 host + 1 anchor

    def test_no_piece_data_returns_none_for_deep_metrics(self):
        m = _compute_piece_metrics(
            piece_data=None,
            telemetry=self._make_telemetry(),
            show_type="debate",
            target_words=450,
            target_duration_mins=3.0,
        )
        assert m["pace_monotony"] is None
        assert m["dominant_speaker_share"] is None
        assert m["architect_host_count"] is None


# ─────────────────────────────────────────────────────────────────────────────
# init_db
# ─────────────────────────────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_all_tables(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert {"runs", "pieces", "decisions", "calibration_state"} <= tables

    def test_idempotent(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path)
        init_db(db_path)  # second call must not raise

    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "nested" / "dir" / "test.db"
        init_db(db_path)
        assert db_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# ingest_run
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestRun:
    def test_missing_index_raises(self, tmp_path):
        empty_dir = tmp_path / "20260524_empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="index.json"):
            ingest_run(tmp_path / "test.db", empty_dir)

    def test_basic_ingest(self, tmp_path):
        study_dir = _make_study_dir(tmp_path)
        db_path = tmp_path / "longitudinal.db"
        inserted = ingest_run(db_path, study_dir)
        assert inserted == {"debate": 1}

    def test_runs_row_inserted(self, tmp_path):
        study_dir = _make_study_dir(tmp_path)
        db_path = tmp_path / "longitudinal.db"
        ingest_run(db_path, study_dir, production_id="test_studio")

        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT production_id, show_type, n_pieces, n_ok, target_duration_mins, "
                "wpm_default, overshoot_compensation FROM runs"
            ).fetchone()

        assert row[0] == "test_studio"
        assert row[1] == "debate"
        assert row[2] == 1  # n_pieces
        assert row[3] == 1  # n_ok
        assert row[4] == pytest.approx(3.0)
        assert row[5] == 150  # wpm_default from snapshot
        assert row[6] == pytest.approx(0.92)

    def test_pieces_row_inserted(self, tmp_path):
        study_dir = _make_study_dir(tmp_path, word_count=280)
        db_path = tmp_path / "longitudinal.db"
        ingest_run(db_path, study_dir)

        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT show_type, topic_slug, word_count, architect_section_count "
                "FROM pieces"
            ).fetchone()

        assert row[0] == "debate"
        assert row[1] == "test-topic"
        assert row[2] == 280
        assert row[3] == 2  # 2 participants

    def test_word_ratio_computed(self, tmp_path):
        # 3.0 min × 150 wpm = 450 target_words; 450 words → ratio 1.0
        study_dir = _make_study_dir(tmp_path, word_count=450)
        db_path = tmp_path / "longitudinal.db"
        ingest_run(db_path, study_dir)

        with sqlite3.connect(db_path) as conn:
            ratio = conn.execute("SELECT word_ratio FROM pieces").fetchone()[0]
            avg_ratio = conn.execute("SELECT avg_word_ratio FROM runs").fetchone()[0]

        assert ratio == pytest.approx(1.0)
        assert avg_ratio == pytest.approx(1.0)

    def test_duplicate_ingest_raises(self, tmp_path):
        study_dir = _make_study_dir(tmp_path)
        db_path = tmp_path / "longitudinal.db"
        ingest_run(db_path, study_dir)
        with pytest.raises(ValueError, match="already in the DB"):
            ingest_run(db_path, study_dir)

    def test_no_params_snapshot_still_ingests(self, tmp_path):
        study_dir = _make_study_dir(tmp_path)
        (study_dir / "params_snapshot.json").unlink()
        db_path = tmp_path / "longitudinal.db"
        inserted = ingest_run(db_path, study_dir)
        assert "debate" in inserted

        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT overshoot_compensation, wpm_default FROM runs"
            ).fetchone()
        # All X params should be NULL when snapshot is absent.
        assert row[0] is None
        assert row[1] is None

    def test_default_production_id(self, tmp_path):
        study_dir = _make_study_dir(tmp_path)
        db_path = tmp_path / "longitudinal.db"
        ingest_run(db_path, study_dir)

        with sqlite3.connect(db_path) as conn:
            pid = conn.execute("SELECT production_id FROM runs").fetchone()[0]
        assert pid == "default"

    def test_failed_piece_excluded_from_aggregates(self, tmp_path):
        study_dir = _make_study_dir(tmp_path, exit_code=1, word_count=0)
        db_path = tmp_path / "longitudinal.db"
        ingest_run(db_path, study_dir)

        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT avg_word_ratio, n_ok FROM runs").fetchone()
        # avg_word_ratio is over ok pieces only → no ok pieces → None
        assert row[0] is None
        assert row[1] == 0

    def test_news_show_ingest(self, tmp_path):
        study_dir = _make_study_dir(tmp_path, show_type="news", word_count=330)
        db_path = tmp_path / "longitudinal.db"
        inserted = ingest_run(db_path, study_dir)
        assert "news" in inserted

        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT show_type, architect_section_count, architect_host_count FROM pieces"
            ).fetchone()
        assert row[0] == "news"
        assert row[1] == 2  # 2 segments
        assert row[2] == 2  # 1 host + 1 anchor

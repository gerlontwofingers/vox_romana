"""vox_romana.analysis.longitudinal — SQLite longitudinal data store.

Captures per-run tunable parameter snapshots (X) and observed show outcomes (Y)
for data-driven calibration of creative parameters.

Public API::

    from vox_romana.analysis.longitudinal import init_db, ingest_run

    init_db("study_results/longitudinal.db")

    inserted = ingest_run(
        db_path="study_results/longitudinal.db",
        study_dir=Path("study_results/20260524_123833"),
        production_id="default",
    )
    # {"debate": 5, "news": 5, "podcast": 5, "audiobook": 5, "lecture": 5}

Schema
------
- ``runs``            — one row per (run_id × show_type); holds X params + Y aggregates.
- ``pieces``          — one row per piece; holds Y outcomes + architect decisions.
- ``decisions``       — human-authored log of deliberate parameter changes.
- ``calibration_state`` — current calibrated param values + model quality.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

__all__ = ["init_db", "ingest_run"]

# ─────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                  TEXT    NOT NULL,
    show_type               TEXT    NOT NULL,
    production_id           TEXT    NOT NULL DEFAULT 'default',
    timestamp               TEXT,
    study_dir               TEXT,
    git_sha                 TEXT,
    target_duration_mins    REAL,
    target_words            INTEGER,
    overshoot_compensation  REAL,
    wrap_up_threshold       REAL,
    wpm_default             INTEGER,
    lead_in_ms              INTEGER,
    max_beat_words          INTEGER,
    avg_words_per_turn      INTEGER,
    avg_words_per_segment   INTEGER,
    closing_window_extra    INTEGER,
    avg_word_ratio          REAL,
    avg_event_count         REAL,
    avg_words_per_event     REAL,
    avg_pace_monotony       REAL,
    avg_name_diversity      REAL,
    avg_dominant_share      REAL,
    avg_nonverbal_rate      REAL,
    total_elapsed_secs      REAL,
    n_pieces                INTEGER,
    n_ok                    INTEGER,
    PRIMARY KEY (run_id, show_type)
);

CREATE TABLE IF NOT EXISTS pieces (
    piece_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT    NOT NULL,
    show_type               TEXT    NOT NULL,
    production_id           TEXT    NOT NULL DEFAULT 'default',
    topic_slug              TEXT,
    target_duration_mins    REAL,
    target_words            INTEGER,
    architect_host_count    INTEGER,
    architect_section_count INTEGER,
    architect_target_turns  INTEGER,
    word_count              INTEGER,
    word_ratio              REAL,
    event_count             INTEGER,
    words_per_event         REAL,
    pace_monotony           REAL,
    name_diversity          REAL,
    dominant_speaker_share  REAL,
    nonverbal_rate          REAL,
    exit_code               INTEGER,
    elapsed_secs            REAL,
    FOREIGN KEY (run_id, show_type) REFERENCES runs (run_id, show_type)
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    production_id       TEXT    NOT NULL DEFAULT 'default',
    show_type           TEXT,
    timestamp           TEXT,
    run_id_before       TEXT,
    run_id_after        TEXT,
    hypothesis          TEXT,
    parameters_changed  TEXT,
    rationale           TEXT,
    outcome             TEXT
);

CREATE TABLE IF NOT EXISTS calibration_state (
    production_id           TEXT    NOT NULL,
    show_type               TEXT    NOT NULL,
    overshoot_compensation  REAL,
    wrap_up_threshold       REAL,
    wpm_default             INTEGER,
    max_beat_words          INTEGER,
    avg_words_per_turn      INTEGER,
    avg_words_per_segment   INTEGER,
    r_squared               REAL,
    n_runs_used             INTEGER,
    updated_at              TEXT,
    PRIMARY KEY (production_id, show_type)
);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Public: init_db
# ─────────────────────────────────────────────────────────────────────────────

def init_db(db_path: str | Path) -> None:
    """Create the longitudinal DB and all four tables if they don't exist.

    Safe to call on an existing DB — uses ``CREATE TABLE IF NOT EXISTS``.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(values: list[float | None]) -> float | None:
    """Return the mean of non-None values, or None if the list is empty."""
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def _extract_x_params(snapshot: dict[str, Any], show_type: str) -> dict[str, Any]:
    """Extract X-param values for *show_type* from a params_snapshot dict.

    Returns a flat dict with keys matching the ``runs`` table columns.
    Values are ``None`` when not present in the snapshot.
    """
    universal = snapshot.get("universal", {})
    by_show = snapshot.get("by_show", {}).get(show_type, {})
    return {
        "overshoot_compensation": universal.get("overshoot_compensation"),
        "wrap_up_threshold": universal.get("wrap_up_threshold"),
        "wpm_default": by_show.get("wpm_default"),
        "lead_in_ms": by_show.get("lead_in_ms"),
        "max_beat_words": by_show.get("max_beat_words"),
        "avg_words_per_turn": by_show.get("avg_words_per_turn"),
        "avg_words_per_segment": by_show.get("avg_words_per_segment"),
        "closing_window_extra": by_show.get("closing_window_extra"),
    }


def _compute_piece_metrics(
    piece_data: dict[str, Any] | None,
    telemetry: dict[str, Any],
    show_type: str,
    target_words: int | None,
    target_duration_mins: float | None,
) -> dict[str, Any]:
    """Compute per-piece Y metrics from a piece JSON dict + telemetry entry.

    *telemetry* is the result entry from ``index.json`` (same structure as
    ``telemetry.json``).  *piece_data* is the actual show output JSON; it may
    be ``None`` if the file is missing or the piece failed.

    Returns a flat dict ready for insertion into the ``pieces`` table.
    """
    summary = telemetry.get("summary") or {}
    word_count: int | None = summary.get("word_count")
    event_count: int | None = summary.get("event_count")
    nonverbal_cue_count: int | None = summary.get("nonverbal_cue_count")
    cast_size: int | None = summary.get("cast_size")
    speakers: list[str] = summary.get("speakers") or []

    word_ratio = (word_count / target_words) if (word_count is not None and target_words) else None
    words_per_event = (word_count / event_count) if (word_count and event_count) else None
    nonverbal_rate = (nonverbal_cue_count / event_count) if (nonverbal_cue_count is not None and event_count) else None
    # Unique speaker names / total cast slots — higher means all cast got airtime.
    name_diversity = (len(speakers) / cast_size) if (speakers and cast_size) else None

    pace_monotony: float | None = None
    dominant_speaker_share: float | None = None
    architect_host_count: int | None = None
    architect_section_count: int | None = None
    architect_target_turns: int | None = None  # not available in output JSON

    if piece_data:
        events: list[dict[str, Any]] = piece_data.get("events") or []

        # pace_monotony: fraction of paced events that share the most common pace.
        paces = [e.get("pace") for e in events if e.get("pace")]
        if paces:
            most_common_count = Counter(paces).most_common(1)[0][1]
            pace_monotony = most_common_count / len(paces)

        # dominant_speaker_share: fraction of events owned by the most talkative speaker.
        if events:
            speaker_counts = Counter(
                e.get("speaker_name") for e in events if e.get("speaker_name")
            )
            if speaker_counts:
                dominant_speaker_share = speaker_counts.most_common(1)[0][1] / len(events)

        # architect fields — which conductor decisions shaped this piece.
        if show_type == "debate":
            architect_host_count = len(piece_data.get("hosts") or []) + len(
                piece_data.get("moderators") or []
            )
            architect_section_count = len(piece_data.get("participants") or [])
        elif show_type == "news":
            architect_section_count = len(piece_data.get("segments") or [])
            architect_host_count = len(piece_data.get("hosts") or []) + len(
                piece_data.get("anchors") or []
            )
        elif show_type == "podcast":
            architect_host_count = len(piece_data.get("hosts") or [])
            architect_section_count = len(piece_data.get("segments") or [])
        elif show_type == "audiobook":
            architect_section_count = piece_data.get("chapter_count") or len(
                piece_data.get("chapters") or []
            )
            architect_host_count = 1 if piece_data.get("narrator") else None
        elif show_type == "lecture":
            architect_section_count = piece_data.get("section_count") or len(
                piece_data.get("sections") or []
            )
            architect_host_count = 1 if piece_data.get("professor") else None

    return {
        "topic_slug": telemetry.get("slug"),
        "target_duration_mins": target_duration_mins,
        "target_words": target_words,
        "architect_host_count": architect_host_count,
        "architect_section_count": architect_section_count,
        "architect_target_turns": architect_target_turns,
        "word_count": word_count,
        "word_ratio": word_ratio,
        "event_count": event_count,
        "words_per_event": words_per_event,
        "pace_monotony": pace_monotony,
        "name_diversity": name_diversity,
        "dominant_speaker_share": dominant_speaker_share,
        "nonverbal_rate": nonverbal_rate,
        "exit_code": telemetry.get("exit_code"),
        "elapsed_secs": telemetry.get("elapsed_secs"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public: ingest_run
# ─────────────────────────────────────────────────────────────────────────────

def ingest_run(
    db_path: str | Path,
    study_dir: str | Path,
    production_id: str = "default",
) -> dict[str, int]:
    """Read a battery study directory and insert its results into the DB.

    The ``run_id`` is taken from the leaf directory name (e.g.
    ``"20260524_123833"``).  Params are read from
    ``study_dir/params_snapshot.json``; ``target_duration_mins`` is read from
    ``study_dir/index.json``'s ``aggregate.target_duration_mins`` field.

    Returns a dict mapping ``show_type → n_pieces_inserted``.

    Raises
    ------
    FileNotFoundError
        If ``study_dir/index.json`` is missing.
    ValueError
        If the (run_id, show_type) pair already exists in the DB — prevents
        accidental double-ingestion.
    """
    db_path = Path(db_path)
    study_dir = Path(study_dir)

    index_path = study_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found in {study_dir}")

    init_db(db_path)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    run_id = study_dir.name  # leaf name is the timestamp

    snapshot: dict[str, Any] = {}
    snapshot_path = study_dir / "params_snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    aggregate = index.get("aggregate", {})
    target_duration_mins: float | None = aggregate.get("target_duration_mins")
    git_sha: str | None = snapshot.get("git_sha")
    run_timestamp: str | None = aggregate.get("generated_at")

    # Group result entries by show_type.
    by_show: dict[str, list[dict[str, Any]]] = {}
    for entry in index.get("results", []):
        by_show.setdefault(entry["show_type"], []).append(entry)

    inserted: dict[str, int] = {}

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        for show_type, entries in by_show.items():
            # Guard against re-ingestion.
            cur.execute(
                "SELECT 1 FROM runs WHERE run_id = ? AND show_type = ?",
                (run_id, show_type),
            )
            if cur.fetchone():
                raise ValueError(
                    f"run_id={run_id!r} show_type={show_type!r} is already "
                    "in the DB. Delete the row first to re-ingest."
                )

            x_params = _extract_x_params(snapshot, show_type)
            wpm: int = x_params.get("wpm_default") or 150
            target_words: int | None = (
                int(target_duration_mins * wpm) if target_duration_mins else None
            )

            piece_metrics_list: list[dict[str, Any]] = []
            for entry in entries:
                json_rel = entry.get("json_path")
                piece_data: dict[str, Any] | None = None
                if json_rel:
                    piece_path = study_dir / json_rel
                    if piece_path.exists():
                        try:
                            piece_data = json.loads(
                                piece_path.read_text(encoding="utf-8")
                            )
                        except Exception:
                            pass

                metrics = _compute_piece_metrics(
                    piece_data=piece_data,
                    telemetry=entry,
                    show_type=show_type,
                    target_words=target_words,
                    target_duration_mins=target_duration_mins,
                )
                piece_metrics_list.append(metrics)

            # Aggregate Y values over ok pieces only.
            ok_metrics = [
                m
                for m, e in zip(piece_metrics_list, entries)
                if e.get("ok")
            ]
            avg_word_ratio = _safe_mean([m["word_ratio"] for m in ok_metrics])
            avg_event_count = _safe_mean([m["event_count"] for m in ok_metrics])
            avg_words_per_event = _safe_mean([m["words_per_event"] for m in ok_metrics])
            avg_pace_monotony = _safe_mean([m["pace_monotony"] for m in ok_metrics])
            avg_name_diversity = _safe_mean([m["name_diversity"] for m in ok_metrics])
            avg_dominant_share = _safe_mean([m["dominant_speaker_share"] for m in ok_metrics])
            avg_nonverbal_rate = _safe_mean([m["nonverbal_rate"] for m in ok_metrics])
            total_elapsed = sum(
                e.get("elapsed_secs") or 0.0 for e in entries
            )

            cur.execute(
                """
                INSERT INTO runs (
                    run_id, show_type, production_id, timestamp, study_dir,
                    git_sha, target_duration_mins, target_words,
                    overshoot_compensation, wrap_up_threshold, wpm_default,
                    lead_in_ms, max_beat_words, avg_words_per_turn,
                    avg_words_per_segment, closing_window_extra,
                    avg_word_ratio, avg_event_count, avg_words_per_event,
                    avg_pace_monotony, avg_name_diversity, avg_dominant_share,
                    avg_nonverbal_rate, total_elapsed_secs, n_pieces, n_ok
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    run_id, show_type, production_id, run_timestamp, str(study_dir),
                    git_sha, target_duration_mins, target_words,
                    x_params.get("overshoot_compensation"),
                    x_params.get("wrap_up_threshold"),
                    x_params.get("wpm_default"),
                    x_params.get("lead_in_ms"),
                    x_params.get("max_beat_words"),
                    x_params.get("avg_words_per_turn"),
                    x_params.get("avg_words_per_segment"),
                    x_params.get("closing_window_extra"),
                    avg_word_ratio, avg_event_count, avg_words_per_event,
                    avg_pace_monotony, avg_name_diversity, avg_dominant_share,
                    avg_nonverbal_rate, total_elapsed,
                    len(entries),
                    sum(1 for e in entries if e.get("ok")),
                ),
            )

            for metrics, entry in zip(piece_metrics_list, entries):
                cur.execute(
                    """
                    INSERT INTO pieces (
                        run_id, show_type, production_id, topic_slug,
                        target_duration_mins, target_words,
                        architect_host_count, architect_section_count,
                        architect_target_turns,
                        word_count, word_ratio, event_count, words_per_event,
                        pace_monotony, name_diversity, dominant_speaker_share,
                        nonverbal_rate, exit_code, elapsed_secs
                    ) VALUES (
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?
                    )
                    """,
                    (
                        run_id, show_type, production_id, metrics["topic_slug"],
                        metrics["target_duration_mins"], metrics["target_words"],
                        metrics["architect_host_count"],
                        metrics["architect_section_count"],
                        metrics["architect_target_turns"],
                        metrics["word_count"], metrics["word_ratio"],
                        metrics["event_count"], metrics["words_per_event"],
                        metrics["pace_monotony"], metrics["name_diversity"],
                        metrics["dominant_speaker_share"],
                        metrics["nonverbal_rate"],
                        metrics["exit_code"], metrics["elapsed_secs"],
                    ),
                )

            inserted[show_type] = len(entries)

        conn.commit()

    return inserted

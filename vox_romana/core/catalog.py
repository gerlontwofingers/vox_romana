"""Catalog — local SQLite content log for recall and routing.

A lightweight, zero-dependency content catalog that tracks everything
vox_romana has ever produced. The VoxRouter will use this to understand
what's been done before ("have we covered UBI?", "what was the last
news episode number?").

Uses SQLite with FTS5 full-text search so the router can do natural
language lookups without a vector database.

USAGE:
    from vox_romana.core.catalog import Catalog

    catalog = Catalog()  # opens/creates data/vox_romana.db

    # Log a completed run
    run_id = catalog.log_run(
        content_type="debate",
        title="UBI Debate",
        brief="Two economists debate universal basic income",
        config={"participants": [...], "target_turns": 40},
        output_paths=["output/text/debates/ubi.json", "output/audio/debates/ubi.mp3"],
        duration_seconds=925,
    )

    # Tag it
    catalog.add_tags(run_id, ["economics", "UBI", "policy"])

    # Search past content
    results = catalog.search("universal basic income")

    # Recent runs by type
    recent = catalog.recent(content_type="debate", limit=5)

    # Log a data study (from future DS module)
    catalog.log_study(
        run_id=run_id,
        dataset="finland_ubi_pilot.csv",
        method="two-sample t-test",
        findings="Employment rose 4.8% (p=0.003, n=2000)",
        fact_cards=[{"claim": "Employment rose 4.8%", "source": "KELA"}],
    )

STORAGE:
    Default location: data/vox_romana.db (relative to workspace root).
    The data/ directory is gitignored. The schema lives here in code.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Default DB path — resolved via VOX_ROMANA_DATA_DIR or ~/.vox_romana/config.yaml.
from vox_romana.core.paths import CATALOG_DB_PATH as DEFAULT_DB_PATH

# Schema version — bump this when adding migrations.
SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    """A single content generation run."""

    run_id: str
    content_type: str
    title: str
    brief: str
    config_json: str
    output_paths: list[str]
    pipeline: list[str]
    duration_seconds: int | None
    cost_estimate: float | None
    tags: list[str]
    created_at: str
    series_id: str | None = None
    episode_number: int | None = None


@dataclass
class StudyRecord:
    """A data study result."""

    study_id: str
    run_id: str | None
    content_type: str
    kind: str
    dataset: str
    method: str
    findings: str
    fact_cards_json: str
    report_path: str
    created_at: str


# ─────────────────────────────────────────────────────────────────────────────
# Catalog
# ─────────────────────────────────────────────────────────────────────────────

class Catalog:
    """Local SQLite content catalog with full-text search.

    Thread-safe for reads. Writes should come from one process at a time
    (which is fine — vox_romana runs single-process).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Open or create the catalog database.

        Args:
            db_path: Path to the SQLite file. Defaults to data/vox_romana.db.
                     Use ":memory:" for testing.
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self._db_path = Path(db_path) if db_path != ":memory:" else db_path

        if self._db_path != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id       TEXT PRIMARY KEY,
                    content_type TEXT NOT NULL,
                    title        TEXT NOT NULL DEFAULT '',
                    brief        TEXT NOT NULL DEFAULT '',
                    config_json  TEXT NOT NULL DEFAULT '{}',
                    output_paths TEXT NOT NULL DEFAULT '[]',
                    pipeline     TEXT NOT NULL DEFAULT '[]',
                    duration_seconds INTEGER,
                    cost_estimate    REAL,
                    series_id    TEXT,
                    episode_number INTEGER,
                    created_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tags (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    tag    TEXT NOT NULL,
                    PRIMARY KEY (run_id, tag)
                );

                CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
                CREATE INDEX IF NOT EXISTS idx_runs_type ON runs(content_type);
                CREATE INDEX IF NOT EXISTS idx_runs_series ON runs(series_id);
                CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);

                CREATE TABLE IF NOT EXISTS studies (
                    study_id    TEXT PRIMARY KEY,
                    run_id      TEXT REFERENCES runs(run_id) ON DELETE SET NULL,
                    content_type TEXT NOT NULL DEFAULT '',
                    kind        TEXT NOT NULL DEFAULT 'research',
                    dataset     TEXT NOT NULL DEFAULT '',
                    method      TEXT NOT NULL DEFAULT '',
                    findings    TEXT NOT NULL DEFAULT '',
                    fact_cards_json TEXT NOT NULL DEFAULT '[]',
                    report_path TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_studies_type ON studies(content_type);
                CREATE INDEX IF NOT EXISTS idx_studies_kind ON studies(kind);

                CREATE TABLE IF NOT EXISTS wpm_stats (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_type TEXT NOT NULL,
                    word_count   INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    wpm          REAL NOT NULL,
                    created_at   TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_wpm_type ON wpm_stats(content_type);

                -- Voice registry: lock a character name to a specific voice_id
                -- within a series/show so the same character always sounds the same.
                -- series_key is: series_id if set, else show_name, else "global".
                CREATE TABLE IF NOT EXISTS character_voices (
                    character_name TEXT NOT NULL,
                    series_key     TEXT NOT NULL,
                    voice_id       TEXT NOT NULL,
                    first_seen     TEXT NOT NULL,
                    last_seen      TEXT NOT NULL,
                    PRIMARY KEY (character_name, series_key)
                );

                CREATE INDEX IF NOT EXISTS idx_charvoices_series
                    ON character_voices(series_key);

                -- Global usage counter per voice_id so the fallback picker
                -- can prefer less-used voices and avoid over-repeating any one.
                CREATE TABLE IF NOT EXISTS voice_usage (
                    voice_id    TEXT PRIMARY KEY,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    last_used   TEXT NOT NULL
                );
            """)

            # FTS5 virtual table for full-text search across runs.
            # We use a content-sync approach: the FTS table mirrors runs data.
            # Triggers keep it in sync automatically.
            self._conn.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS runs_fts USING fts5(
                    title,
                    brief,
                    content_type,
                    content='runs',
                    content_rowid='rowid'
                );

                -- Keep FTS in sync with INSERT/UPDATE/DELETE on runs.
                CREATE TRIGGER IF NOT EXISTS runs_ai AFTER INSERT ON runs BEGIN
                    INSERT INTO runs_fts(rowid, title, brief, content_type)
                    VALUES (NEW.rowid, NEW.title, NEW.brief, NEW.content_type);
                END;

                CREATE TRIGGER IF NOT EXISTS runs_ad AFTER DELETE ON runs BEGIN
                    INSERT INTO runs_fts(runs_fts, rowid, title, brief, content_type)
                    VALUES ('delete', OLD.rowid, OLD.title, OLD.brief, OLD.content_type);
                END;

                CREATE TRIGGER IF NOT EXISTS runs_au AFTER UPDATE ON runs BEGIN
                    INSERT INTO runs_fts(runs_fts, rowid, title, brief, content_type)
                    VALUES ('delete', OLD.rowid, OLD.title, OLD.brief, OLD.content_type);
                    INSERT INTO runs_fts(rowid, title, brief, content_type)
                    VALUES (NEW.rowid, NEW.title, NEW.brief, NEW.content_type);
                END;
            """)

    # ═══════════════════════════════════════════════════════════════════
    # WRITE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    def log_run(
        self,
        content_type: str,
        *,
        title: str = "",
        brief: str = "",
        config: dict[str, Any] | None = None,
        output_paths: list[str] | None = None,
        pipeline: list[str] | None = None,
        duration_seconds: int | None = None,
        cost_estimate: float | None = None,
        series_id: str | None = None,
        episode_number: int | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Log a completed content generation run.

        Args:
            content_type: "debate", "news", "audiobook", etc.
            title:        Human-readable title.
            brief:        The original brief/prompt.
            config:       The full config dict (stored as JSON).
            output_paths: List of output file paths produced.
            pipeline:     List of tool/module identifiers used
                          (e.g. ["bootstrap", "conductor", "tts", "mixer"]).
            duration_seconds: Audio duration in seconds.
            cost_estimate: Estimated LLM cost in USD.
            series_id:    For episodic content.
            episode_number: For episodic content.
            tags:         Optional tags for this run.

        Returns:
            The generated run_id (UUID).
        """
        run_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()

        with self._conn:
            self._conn.execute(
                """INSERT INTO runs
                   (run_id, content_type, title, brief, config_json,
                    output_paths, pipeline, duration_seconds, cost_estimate,
                    series_id, episode_number, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    content_type,
                    title,
                    brief,
                    json.dumps(config or {}, default=str),
                    json.dumps(output_paths or []),
                    json.dumps(pipeline or []),
                    duration_seconds,
                    cost_estimate,
                    series_id,
                    episode_number,
                    now,
                ),
            )

            if tags:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO tags (run_id, tag) VALUES (?, ?)",
                    [(run_id, tag) for tag in tags],
                )

        return run_id

    def add_tags(self, run_id: str, tags: list[str]) -> None:
        """Add tags to an existing run."""
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO tags (run_id, tag) VALUES (?, ?)",
                [(run_id, tag) for tag in tags],
            )

    def log_study(
        self,
        *,
        run_id: str | None = None,
        content_type: str = "",
        kind: str = "research",
        dataset: str = "",
        method: str = "",
        findings: str = "",
        fact_cards: list[dict[str, Any]] | None = None,
        report_path: str = "",
    ) -> str:
        """Log a data study result.

        Args:
            run_id:       Optional link to the content run that used this study.
            content_type: Show type this research supports ("debate", "news", etc.).
            kind:         Study category for fast filtering (default "research").
            dataset:      Name/path of the dataset analyzed.
            method:       Statistical method used.
            findings:     Plain English summary of findings.
            fact_cards:   List of fact card dicts (stored as JSON).
            report_path:  Path to the generated HTML report directory.

        Returns:
            The generated study_id (UUID).
        """
        study_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()

        with self._conn:
            self._conn.execute(
                """INSERT INTO studies
                   (study_id, run_id, content_type, kind, dataset, method,
                    findings, fact_cards_json, report_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    study_id,
                    run_id,
                    content_type,
                    kind,
                    dataset,
                    method,
                    findings,
                    json.dumps(fact_cards or [], default=str),
                    report_path,
                    now,
                ),
            )

        return study_id

    # ═══════════════════════════════════════════════════════════════════
    # READ OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    def search(self, query: str, *, limit: int = 20) -> list[RunRecord]:
        """Full-text search across run titles and briefs.

        Uses SQLite FTS5 — supports natural language queries and
        boolean operators (AND, OR, NOT, prefix*).

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of matching RunRecords, ranked by relevance.
        """
        rows = self._conn.execute(
            """SELECT r.*, GROUP_CONCAT(t.tag, ',') as tag_str
               FROM runs_fts fts
               JOIN runs r ON r.rowid = fts.rowid
               LEFT JOIN tags t ON t.run_id = r.run_id
               WHERE runs_fts MATCH ?
               GROUP BY r.run_id
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()

        return [self._row_to_run(row) for row in rows]

    def recent(
        self,
        *,
        content_type: str | None = None,
        series_id: str | None = None,
        limit: int = 10,
    ) -> list[RunRecord]:
        """Get recent runs, optionally filtered.

        Args:
            content_type: Filter by type ("debate", "news", etc.).
            series_id:    Filter by series.
            limit:        Maximum results.

        Returns:
            List of RunRecords, newest first.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if content_type:
            conditions.append("r.content_type = ?")
            params.append(content_type)
        if series_id:
            conditions.append("r.series_id = ?")
            params.append(series_id)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = self._conn.execute(
            f"""SELECT r.*, GROUP_CONCAT(t.tag, ',') as tag_str
                FROM runs r
                LEFT JOIN tags t ON t.run_id = r.run_id
                {where}
                GROUP BY r.run_id
                ORDER BY r.created_at DESC
                LIMIT ?""",
            params,
        ).fetchall()

        return [self._row_to_run(row) for row in rows]

    def get_run(self, run_id: str) -> RunRecord | None:
        """Get a single run by ID."""
        row = self._conn.execute(
            """SELECT r.*, GROUP_CONCAT(t.tag, ',') as tag_str
               FROM runs r
               LEFT JOIN tags t ON t.run_id = r.run_id
               WHERE r.run_id = ?
               GROUP BY r.run_id""",
            (run_id,),
        ).fetchone()

        return self._row_to_run(row) if row else None

    def get_studies(self, run_id: str) -> list[StudyRecord]:
        """Get all studies linked to a run."""
        rows = self._conn.execute(
            "SELECT * FROM studies WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()

        return [
            StudyRecord(
                study_id=row["study_id"],
                run_id=row["run_id"],
                content_type=row["content_type"],
                kind=row["kind"],
                dataset=row["dataset"],
                method=row["method"],
                findings=row["findings"],
                fact_cards_json=row["fact_cards_json"],
                report_path=row["report_path"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def count(self, *, content_type: str | None = None) -> int:
        """Count total runs, optionally by type."""
        if content_type:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM runs WHERE content_type = ?",
                (content_type,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return row[0] if row else 0

    def next_episode_number(self, series_id: str) -> int:
        """Get the next episode number for a series."""
        row = self._conn.execute(
            "SELECT MAX(episode_number) FROM runs WHERE series_id = ?",
            (series_id,),
        ).fetchone()
        current = row[0] if row and row[0] is not None else 0
        return current + 1

    # ═══════════════════════════════════════════════════════════════════
    # WPM STATS
    # ═══════════════════════════════════════════════════════════════════

    def log_wpm_stat(
        self,
        *,
        content_type: str,
        word_count: int,
        duration_seconds: float,
        wpm: float,
    ) -> None:
        """Record a WPM observation for adaptive duration learning."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """INSERT INTO wpm_stats
                   (content_type, word_count, duration_seconds, wpm, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (content_type, word_count, duration_seconds, wpm, now),
            )

    def get_wpm_stats(self, content_type: str) -> dict[str, Any] | None:
        """Get aggregate WPM statistics for a content type.

        Returns:
            Dict with keys ``count``, ``avg_wpm``, ``min_wpm``, ``max_wpm``,
            or None if no data exists for this type.
        """
        row = self._conn.execute(
            """SELECT COUNT(*) as cnt,
                      AVG(wpm) as avg_wpm,
                      MIN(wpm) as min_wpm,
                      MAX(wpm) as max_wpm
               FROM wpm_stats
               WHERE content_type = ?""",
            (content_type,),
        ).fetchone()

        if not row or row["cnt"] == 0:
            return None

        return {
            "count": row["cnt"],
            "avg_wpm": row["avg_wpm"],
            "min_wpm": row["min_wpm"],
            "max_wpm": row["max_wpm"],
        }

    # ═══════════════════════════════════════════════════════════════════
    # INTERNALS
    # ═══════════════════════════════════════════════════════════════════

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        """Convert a database row to a RunRecord."""
        tag_str = row["tag_str"] or ""
        tags = [t.strip() for t in tag_str.split(",") if t.strip()]

        return RunRecord(
            run_id=row["run_id"],
            content_type=row["content_type"],
            title=row["title"],
            brief=row["brief"],
            config_json=row["config_json"],
            output_paths=json.loads(row["output_paths"]),
            pipeline=json.loads(row["pipeline"]),
            duration_seconds=row["duration_seconds"],
            cost_estimate=row["cost_estimate"],
            tags=tags,
            created_at=row["created_at"],
            series_id=row["series_id"],
            episode_number=row["episode_number"],
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

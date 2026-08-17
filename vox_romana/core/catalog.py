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
from vox_romana.core.audience import AudienceProfile
from vox_romana.core.paths import CATALOG_DB_PATH as DEFAULT_DB_PATH

# Schema version — bump this when adding migrations.
SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SeriesRecord:
    """A series — the studio identity that ties episodes together.

    A series carries the writers' bible (tone, recurring rules, world
    facts, character canon) plus an optional audience profile. When a
    show is generated with ``--series ID``, this record's ``bible`` and
    ``audience_profile_json`` are formatted into a SERIES CONTEXT block
    that is prepended to the brief before the Architect runs.
    """

    series_id: str
    title: str
    show_type: str | None
    tone: str | None
    bible: str
    created_at: str
    audience_profile: AudienceProfile | None = None


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


@dataclass
class AuditionRecord:
    """A raw audition session — a batch of generated candidate assets."""

    id: int
    session_id: str
    series_id: str | None
    audition_type: str          # "voice" | "music" | "visual"
    created_at: str
    brief: str
    results_json: str           # JSON list of {label, path, score}
    chosen: str | None          # label of the chosen asset, if any


@dataclass
class AssetRecord:
    """An approved, greenlit asset in a series asset pool."""

    id: int
    series_id: str
    asset_type: str             # "voice" | "music" | "visual"
    character_name: str | None  # e.g. "Dr. Chen" for voice assets
    asset_id: str               # voice ID, music slug, etc.
    source_path: str            # path to the audition clip / reference file
    approved_at: str
    notes: str


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

                -- Series: the writers'-bible record that ties episodes
                -- together. ``bible`` is the free-form recurring context
                -- (tone, world rules, character canon) prepended to every
                -- brief generated under this series. ``audience_profile_json``
                -- is reserved for a future AudienceProfile feature; included
                -- now so we never have to ALTER TABLE.
                CREATE TABLE IF NOT EXISTS series (
                    series_id            TEXT PRIMARY KEY,
                    title                TEXT NOT NULL,
                    show_type            TEXT,
                    tone                 TEXT,
                    bible                TEXT NOT NULL DEFAULT '',
                    audience_profile_json TEXT,
                    created_at           TEXT NOT NULL
                );

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

                -- Audition sessions: raw batches of candidate assets generated
                -- by --mode audition (voice/music/visual).  Each session has a
                -- stable session_id; results_json stores the ranked candidates.
                CREATE TABLE IF NOT EXISTS auditions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id    TEXT NOT NULL UNIQUE,
                    series_id     TEXT,
                    audition_type TEXT NOT NULL,
                    created_at    TEXT NOT NULL,
                    brief         TEXT NOT NULL DEFAULT '',
                    results_json  TEXT NOT NULL DEFAULT '[]',
                    chosen        TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_auditions_session
                    ON auditions(session_id);
                CREATE INDEX IF NOT EXISTS idx_auditions_series
                    ON auditions(series_id);

                -- Series asset pool: the greenlit set of approved voices,
                -- music styles, and visual styles for a specific series.
                -- Assets are approved from audition sessions or added manually.
                CREATE TABLE IF NOT EXISTS series_assets (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    series_id      TEXT NOT NULL,
                    asset_type     TEXT NOT NULL,
                    character_name TEXT,
                    asset_id       TEXT NOT NULL,
                    source_path    TEXT NOT NULL DEFAULT '',
                    approved_at    TEXT NOT NULL,
                    notes          TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_assets_series
                    ON series_assets(series_id);
                CREATE INDEX IF NOT EXISTS idx_assets_type
                    ON series_assets(asset_type);
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
    # SERIES (writers' bible)
    # ═══════════════════════════════════════════════════════════════════

    def create_series(
        self,
        *,
        title: str,
        show_type: str | None = None,
        tone: str | None = None,
        bible: str = "",
        audience_profile: AudienceProfile | None = None,
    ) -> str:
        """Create a new series and return its generated UUID."""
        series_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        ap_json = audience_profile.to_json() if audience_profile else None
        with self._conn:
            self._conn.execute(
                """INSERT INTO series
                   (series_id, title, show_type, tone, bible,
                    audience_profile_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (series_id, title, show_type, tone, bible, ap_json, now),
            )
        return series_id

    def set_audience_profile(
        self, series_id: str, profile: AudienceProfile
    ) -> None:
        """Persist (or replace) the audience profile on an existing series."""
        with self._conn:
            self._conn.execute(
                "UPDATE series SET audience_profile_json = ? WHERE series_id = ?",
                (profile.to_json(), series_id),
            )

    def get_series(self, series_id: str) -> SeriesRecord | None:
        """Get a single series by ID, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM series WHERE series_id = ?",
            (series_id,),
        ).fetchone()
        return self._row_to_series(row) if row else None

    def list_series(self) -> list[SeriesRecord]:
        """Return all series, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM series ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_series(row) for row in rows]

    def update_series_bible(self, series_id: str, bible: str) -> None:
        """Replace the bible text for an existing series."""
        with self._conn:
            self._conn.execute(
                "UPDATE series SET bible = ? WHERE series_id = ?",
                (bible, series_id),
            )

    def get_recent_episode_summaries(
        self, series_id: str, *, n: int = 5
    ) -> list[dict[str, Any]]:
        """Return the last N episodes in a series as compact summaries.

        Used by the series-context builder so the Architect knows what
        has already been said. Each dict has: ``run_id``, ``title``,
        ``episode_number``, ``brief``, ``created_at``.
        """
        rows = self._conn.execute(
            """SELECT run_id, title, episode_number, brief, created_at
               FROM runs
               WHERE series_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (series_id, n),
        ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "title": row["title"],
                "episode_number": row["episode_number"],
                "brief": row["brief"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _row_to_series(self, row: sqlite3.Row) -> SeriesRecord:
        """Convert a database row to a SeriesRecord."""
        ap_json = row["audience_profile_json"]
        audience_profile = AudienceProfile.from_json(ap_json) if ap_json else None
        return SeriesRecord(
            series_id=row["series_id"],
            title=row["title"],
            show_type=row["show_type"],
            tone=row["tone"],
            bible=row["bible"],
            created_at=row["created_at"],
            audience_profile=audience_profile,
        )

    # ═══════════════════════════════════════════════════════════════════
    # AUDITIONS + SERIES ASSET POOL
    # ═══════════════════════════════════════════════════════════════════

    def log_audition(
        self,
        *,
        session_id: str,
        series_id: str | None = None,
        audition_type: str,
        brief: str = "",
        results: list[dict[str, Any]] | None = None,
    ) -> int:
        """Log a new audition session.

        Args:
            session_id:     Stable identifier for this audition batch
                            (e.g. a UUID hex).
            series_id:      Optional series this audition belongs to.
            audition_type:  "voice", "music", or "visual".
            brief:          The prompt / description used for the session.
            results:        Ranked list of candidate dicts
                            (each has at minimum: ``label``, ``path``).

        Returns:
            The ``id`` (auto-increment PK) of the inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO auditions
                   (session_id, series_id, audition_type, created_at,
                    brief, results_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    series_id,
                    audition_type,
                    now,
                    brief,
                    json.dumps(results or [], default=str),
                ),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def get_audition(self, session_id: str) -> AuditionRecord | None:
        """Retrieve a single audition session by its session_id."""
        row = self._conn.execute(
            "SELECT * FROM auditions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return AuditionRecord(
            id=row["id"],
            session_id=row["session_id"],
            series_id=row["series_id"],
            audition_type=row["audition_type"],
            created_at=row["created_at"],
            brief=row["brief"],
            results_json=row["results_json"],
            chosen=row["chosen"],
        )

    def approve_asset(
        self,
        *,
        series_id: str,
        asset_type: str,
        asset_id: str,
        character_name: str | None = None,
        source_path: str = "",
        notes: str = "",
    ) -> int:
        """Add one asset to the series pool.

        Args:
            series_id:      Series this asset belongs to.
            asset_type:     "voice", "music", or "visual".
            asset_id:       The canonical identifier (voice ID, music slug, …).
            character_name: For voice assets — the character this voice plays.
            source_path:    Filesystem path to the reference clip or file.
            notes:          Optional free-form notes (e.g. "preferred warm style").

        Returns:
            The ``id`` (auto-increment PK) of the inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO series_assets
                   (series_id, asset_type, character_name, asset_id,
                    source_path, approved_at, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (series_id, asset_type, character_name, asset_id,
                 source_path, now, notes),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def approve_all_from_audition(
        self,
        session_id: str,
        *,
        series_id: str,
    ) -> list[int]:
        """Approve every candidate in an audition session into the pool.

        Reads the ``results_json`` from the audition row and calls
        :meth:`approve_asset` for each entry.  Each result dict must have
        at minimum an ``asset_id`` key; ``character_name``, ``source_path``,
        and ``notes`` are used when present.

        Args:
            session_id: The audition session to approve from.
            series_id:  Series to add the assets to (may differ from the
                        audition's own ``series_id`` if it was ungrouped).

        Returns:
            List of auto-increment PKs for the approved asset rows.

        Raises:
            ValueError: If the session_id is not found.
        """
        rec = self.get_audition(session_id)
        if rec is None:
            raise ValueError(f"Audition session not found: {session_id}")

        results: list[dict] = json.loads(rec.results_json)
        ids: list[int] = []
        for item in results:
            asset_id = item.get("asset_id") or item.get("label") or ""
            if not asset_id:
                continue
            pk = self.approve_asset(
                series_id=series_id,
                asset_type=rec.audition_type,
                asset_id=asset_id,
                character_name=item.get("character_name"),
                source_path=item.get("path") or item.get("source_path") or "",
                notes=item.get("notes") or "",
            )
            ids.append(pk)

        # Mark the audition session as fully chosen.
        with self._conn:
            self._conn.execute(
                "UPDATE auditions SET chosen = 'all' WHERE session_id = ?",
                (session_id,),
            )
        return ids

    def list_assets(
        self,
        series_id: str,
        *,
        asset_type: str | None = None,
    ) -> list[AssetRecord]:
        """Return the approved asset pool for a series.

        Args:
            series_id:  Series to query.
            asset_type: Optional filter ("voice", "music", "visual").

        Returns:
            List of :class:`AssetRecord`, oldest-approved first.
        """
        if asset_type:
            rows = self._conn.execute(
                """SELECT * FROM series_assets
                   WHERE series_id = ? AND asset_type = ?
                   ORDER BY approved_at""",
                (series_id, asset_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM series_assets
                   WHERE series_id = ?
                   ORDER BY approved_at""",
                (series_id,),
            ).fetchall()
        return [self._row_to_asset(row) for row in rows]

    def remove_asset(self, asset_pk: int) -> None:
        """Remove an asset from the pool by its integer PK.

        Args:
            asset_pk: The ``id`` column value of the row to delete.
        """
        with self._conn:
            self._conn.execute(
                "DELETE FROM series_assets WHERE id = ?",
                (asset_pk,),
            )

    # ═══════════════════════════════════════════════════════════════════
    # VOICE CLONE SHORTCUTS
    # ═══════════════════════════════════════════════════════════════════

    def register_cloned_asset(
        self,
        *,
        series_id: str,
        voice_id: str,
        character_name: str | None = None,
        profile_dir: str = "",
        notes: str = "",
    ) -> int:
        """Record a cloned voice profile as a series asset.

        Thin wrapper around :meth:`approve_asset` that sets
        ``asset_type="voice_clone"`` and uses ``profile_dir`` as
        ``source_path`` for discoverability.

        Args:
            series_id:      Series this voice belongs to.
            voice_id:       The ``vc_*`` voice ID (e.g. ``"vc_marcus_deep"``).
            character_name: Optional — the character this voice portrays.
            profile_dir:    Filesystem path to the ``vc_*`` profile directory.
            notes:          Free-form notes (e.g. "cloned from interview.mp3").

        Returns:
            The ``id`` (auto-increment PK) of the inserted ``series_assets`` row.
        """
        return self.approve_asset(
            series_id=series_id,
            asset_type="voice_clone",
            asset_id=voice_id,
            character_name=character_name,
            source_path=profile_dir,
            notes=notes,
        )

    def list_voice_profiles(
        self,
        series_id: str,
    ) -> list[AssetRecord]:
        """Return all cloned voice assets registered for a series.

        Equivalent to ``list_assets(series_id, asset_type="voice_clone")``.

        Args:
            series_id: Series to query.

        Returns:
            List of :class:`AssetRecord` with ``asset_type == "voice_clone"``,
            oldest-approved first.
        """
        return self.list_assets(series_id, asset_type="voice_clone")

    def _row_to_asset(self, row: sqlite3.Row) -> AssetRecord:
        """Convert a database row to an AssetRecord."""
        return AssetRecord(
            id=row["id"],
            series_id=row["series_id"],
            asset_type=row["asset_type"],
            character_name=row["character_name"],
            asset_id=row["asset_id"],
            source_path=row["source_path"],
            approved_at=row["approved_at"],
            notes=row["notes"],
        )

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

    def get_wpm_stats_all(self) -> list[dict]:
        """Return all rows from wpm_stats as a list of dicts.

        Used by WPMModel.update_from_catalog() to retrain the ML predictor
        without poking the private _conn attribute.

        Returns:
            List of dicts with keys ``content_type``, ``word_count``, ``wpm``.
        """
        rows = self._conn.execute(
            "SELECT content_type, word_count, wpm FROM wpm_stats"
        ).fetchall()
        return [
            {
                "content_type": row["content_type"],
                "word_count": row["word_count"],
                "wpm": row["wpm"],
            }
            for row in rows
        ]

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

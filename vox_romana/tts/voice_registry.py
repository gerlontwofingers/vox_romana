"""Voice Registry — persistent character-to-voice locking and usage tracking.

Two concerns live here:

1. CHARACTER CONSISTENCY
   The same character (identified by name + series_key) always gets the same
   voice across episodes. "Dr. Chen" on "The Science Hour" podcast will always
   be af_jessica, whether this is episode 3 or episode 30.

   series_key priority (computed by the caller):
       series_id  — explicit series identifier from the show JSON / CLI
       show_name  — falls back to the show's name (groups episodes of the same show)
       "global"   — last resort; scopes to all content (rarely the right choice)

2. USAGE VARIETY
   A global counter tracks how many times each voice_id has been used across
   all content. When the voice selector has no tag match (i.e. the hint is
   empty or nothing in the catalog matches), it queries usage_counts() and
   picks the least-used voice first. This prevents am_adam (the first male
   voice alphabetically) from monopolising every uncast male character.

STORAGE:
   Two tables in the shared vox_romana.db SQLite database:

       character_voices (character_name, series_key, voice_id, first_seen, last_seen)
       voice_usage      (voice_id, usage_count, last_used)

   Both are created by Catalog._init_schema() — the registry merely reads and
   writes them; it does not own the schema.

USAGE:
   from vox_romana.tts.voice_registry import VoiceRegistry

   registry = VoiceRegistry()          # uses default db path
   registry = VoiceRegistry(":memory:") # for tests

   # Look up a locked voice (returns None if not yet assigned)
   voice_id = registry.resolve("Dr. Chen", "The Science Hour")

   # Lock a new assignment
   registry.register("Dr. Chen", "af_jessica", "The Science Hour")

   # Record that a voice was used (increments the counter)
   registry.record_usage("af_jessica")

   # Get all usage counts for the fallback picker
   counts = registry.usage_counts()  # {"af_jessica": 3, "am_adam": 1, ...}

THREAD SAFETY:
   The SQLite connection uses WAL mode (set by Catalog). Single-process use is
   safe. Multi-process writes are not expected — vox_romana runs one pipeline
   at a time.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


# Default DB path — same file the Catalog uses.
from vox_romana.core.paths import CATALOG_DB_PATH as _DEFAULT_DB_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VoiceRegistry:
    """Persistent voice assignment registry backed by SQLite.

    The registry is deliberately separate from Catalog to keep the TTS
    subsystem self-contained — it connects to the same database file but
    does not depend on the Catalog class itself.

    Attributes:
        _db_path: Path to the SQLite file (or ":memory:" for tests).
        _conn:    The SQLite connection.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Open the registry database.

        Args:
            db_path: Path to the SQLite file. Defaults to the shared
                     vox_romana.db. Pass ":memory:" for isolated tests.
                     The tables are created if they do not already exist
                     (idempotent — safe to call on an existing database).
        """
        if db_path is None:
            db_path = _DEFAULT_DB_PATH

        self._db_path: str | Path = db_path
        # In-process lock guards transactions when callers share a registry
        # across threads (sqlite3.connect was opened with check_same_thread=False,
        # which lets multiple threads use the connection but does NOT serialize
        # them — `with self._conn:` transactions can otherwise interleave).
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        with self._lock, self._conn:
            self._conn.executescript("""
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

                CREATE TABLE IF NOT EXISTS voice_usage (
                    voice_id    TEXT PRIMARY KEY,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    last_used   TEXT NOT NULL
                );
            """)

    # ─────────────────────────────────────────────────────────────────
    # Character locking
    # ─────────────────────────────────────────────────────────────────

    def resolve(self, character_name: str, series_key: str) -> str | None:
        """Look up the locked voice for a character in a series.

        Args:
            character_name: The speaker's display name (e.g. "Dr. Chen").
            series_key:     The series/show scope key. Use the helper
                            :func:`series_key_for` to compute this from
                            show metadata.

        Returns:
            The locked voice_id string, or None if this character has
            never been assigned a voice in this series.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT voice_id FROM character_voices "
                "WHERE character_name = ? AND series_key = ?",
                (character_name, series_key),
            ).fetchone()
        return row["voice_id"] if row else None

    def register(
        self,
        character_name: str,
        voice_id: str,
        series_key: str,
    ) -> None:
        """Lock a voice assignment for a character.

        If the character already has a voice in this series, the existing
        assignment is kept and only ``last_seen`` is updated — a character's
        voice never changes once set.

        Args:
            character_name: The speaker's display name.
            voice_id:       The voice_id to lock (e.g. "af_jessica").
            series_key:     The series/show scope key.
        """
        now = _now_iso()
        with self._lock, self._conn:
            # ON CONFLICT DO UPDATE preserves the existing voice_id and only
            # refreshes last_seen — a character's voice never changes once set.
            self._conn.execute(
                """
                INSERT INTO character_voices
                    (character_name, series_key, voice_id, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(character_name, series_key)
                    DO UPDATE SET last_seen = excluded.last_seen
                """,
                (character_name, series_key, voice_id, now, now),
            )

    # ─────────────────────────────────────────────────────────────────
    # Usage tracking
    # ─────────────────────────────────────────────────────────────────

    def record_usage(self, voice_id: str) -> None:
        """Increment the usage counter for a voice.

        Creates the row if it does not exist (first use).

        Args:
            voice_id: The voice_id that was just assigned (e.g. "am_echo").
        """
        now = _now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO voice_usage (voice_id, usage_count, last_used)
                VALUES (?, 1, ?)
                ON CONFLICT(voice_id)
                    DO UPDATE SET
                        usage_count = usage_count + 1,
                        last_used   = excluded.last_used
                """,
                (voice_id, now),
            )

    def usage_counts(self) -> dict[str, int]:
        """Return usage counts for every tracked voice.

        Voices that have never been used are not included — callers should
        treat a missing key as count 0.

        Returns:
            Dict mapping voice_id → total times used.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT voice_id, usage_count FROM voice_usage"
            ).fetchall()
        return {row["voice_id"]: row["usage_count"] for row in rows}

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "VoiceRegistry":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def series_key_for(
    series_id: str | None,
    show_name: str | None,
) -> str:
    """Compute the series_key for a given show's metadata.

    Priority: series_id > show_name > "global".

    Args:
        series_id: The explicit series identifier, if any.
        show_name: The show's name (e.g. "The Science Hour"), if any.

    Returns:
        A non-empty string to use as the series_key in registry calls.
    """
    return series_id or show_name or "global"

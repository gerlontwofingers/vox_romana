"""Tests for vox_romana.cli.audition_runner.

Tests the wiring of the audition runner without touching any real
TTS, MusicGen, or diffusion model.  Heavy engines are mocked at the
module level via unittest.mock.patch on the import path the runner
uses internally.

THE TESTING PHILOSOPHY (same as test_music_engine.py and
test_animation_runner.py): we test the WIRING — that the right helpers
are called with the right args, that catalog rows are created, that
approval policies are honoured — NOT that the audio itself sounds good.

Run with:
    PYTHONPATH=. .venv/bin/pytest tests/test_audition_runner.py -v
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.cli import audition_runner
from vox_romana.core.catalog import Catalog
from vox_romana.tts.engine import SynthesizedAudio


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_output_dir(tmp_path, monkeypatch):
    """Redirect OUTPUT_DIR (and the auditions/ tree it owns) into tmp_path.

    The runner writes WAVs under ``OUTPUT_DIR / 'auditions' / SESSION``;
    pointing OUTPUT_DIR at a tmp_path keeps tests hermetic.
    """
    monkeypatch.setattr("vox_romana.core.paths.OUTPUT_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def in_memory_catalog(monkeypatch):
    """Replace the Catalog() constructor with an in-memory one.

    The runner lazy-imports ``from vox_romana.core.catalog import Catalog``
    inside ``run_audition`` and instantiates it with ``with Catalog() as c``.
    We patch the class so:
      * Construction yields our shared in-memory instance.
      * ``__exit__`` is a no-op so the connection stays open for tests
        to inspect after ``run_audition()`` returns (otherwise SQLite
        closes the in-memory DB and we get "Cannot operate on a closed
        database" errors).
    """
    import vox_romana.core.catalog as catalog_module

    cat = Catalog(db_path=":memory:")

    def _factory(*args, **kwargs):
        return cat

    monkeypatch.setattr(catalog_module, "Catalog", _factory)
    # Dunder methods are looked up on the type, not the instance, so we
    # have to silence __exit__ at the class level.
    monkeypatch.setattr(Catalog, "__exit__", lambda self, *a: None)
    return cat


def _make_args(**overrides) -> argparse.Namespace:
    """Build a fully-populated audition Namespace with sensible defaults."""
    defaults = dict(
        audition_type="voice",
        speaker=None,
        music_brief=None,
        music_model="small",
        music_duration=4.0,
        audition_count=2,
        approve_all=False,
        no_approve=False,
        series=None,
        quiet=True,
        visual_brief=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fake_synthesized_audio() -> SynthesizedAudio:
    """Build a tiny but valid-ish SynthesizedAudio (just bytes, no real WAV)."""
    return SynthesizedAudio(audio=b"RIFF\x00\x00\x00\x00WAVE", sample_rate=24000)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Voice auditions
# ═══════════════════════════════════════════════════════════════════════════


class TestVoiceAuditions:
    """Verify --audition-type voice generates clips and logs the session."""

    def test_voice_generates_n_clips(self, tmp_output_dir, in_memory_catalog):
        """Each requested voice produces a WAV file on disk."""
        fake_engine = MagicMock()
        fake_engine.synthesize.return_value = _fake_synthesized_audio()

        args = _make_args(audition_type="voice", speaker="Anchor", audition_count=3)

        with patch(
            "vox_romana.tts.engines.kokoro_engine.KokoroEngine",
            return_value=fake_engine,
        ):
            output_dir = audition_runner.run_audition(args)

        assert output_dir is not None
        wavs = sorted((output_dir / "voice").glob("*.wav"))
        assert len(wavs) == 3, f"expected 3 WAVs, got {len(wavs)}"
        assert all(w.read_bytes().startswith(b"RIFF") for w in wavs)
        # Filenames should embed the character name.
        assert all("Anchor_" in w.name for w in wavs)
        # Engine was called once per voice.
        assert fake_engine.synthesize.call_count == 3

    def test_voice_logs_audition_to_catalog(
        self, tmp_output_dir, in_memory_catalog,
    ):
        """A row appears in the auditions table with results JSON."""
        fake_engine = MagicMock()
        fake_engine.synthesize.return_value = _fake_synthesized_audio()
        args = _make_args(audition_type="voice", speaker="Maya", audition_count=2)

        with patch(
            "vox_romana.tts.engines.kokoro_engine.KokoroEngine",
            return_value=fake_engine,
        ):
            audition_runner.run_audition(args)

        # Exactly one audition row should have been created.
        rows = in_memory_catalog._conn.execute(
            "SELECT session_id, audition_type, results_json FROM auditions"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["audition_type"] == "voice"
        results = json.loads(rows[0]["results_json"])
        assert len(results) == 2
        assert all("asset_id" in r and "path" in r for r in results)

    def test_voice_writes_manifest_json(
        self, tmp_output_dir, in_memory_catalog,
    ):
        """A manifest.json is written next to the clips."""
        fake_engine = MagicMock()
        fake_engine.synthesize.return_value = _fake_synthesized_audio()
        args = _make_args(audition_type="voice", speaker="X", audition_count=1)

        with patch(
            "vox_romana.tts.engines.kokoro_engine.KokoroEngine",
            return_value=fake_engine,
        ):
            output_dir = audition_runner.run_audition(args)

        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["audition_type"] == "voice"
        assert len(manifest["results"]) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. Music auditions
# ═══════════════════════════════════════════════════════════════════════════


class TestMusicAuditions:
    """Verify --audition-type music wires through MusicGenEngine."""

    def test_music_generates_n_clips(self, tmp_output_dir, in_memory_catalog):
        """N MusicGen.generate() calls produce N WAV files."""
        fake_engine = MagicMock()
        fake_engine.generate.return_value = _fake_synthesized_audio()

        args = _make_args(
            audition_type="music",
            music_brief="upbeat jazz intro",
            audition_count=3,
        )

        with patch(
            "vox_romana.tts.music_engine.MusicGenEngine",
            return_value=fake_engine,
        ):
            output_dir = audition_runner.run_audition(args)

        wavs = sorted((output_dir / "music").glob("*.wav"))
        assert len(wavs) == 3
        assert fake_engine.generate.call_count == 3
        # Brief was forwarded to every call.
        for c in fake_engine.generate.call_args_list:
            assert c.args[0] == "upbeat jazz intro"

    def test_music_requires_brief(
        self, tmp_output_dir, in_memory_catalog, capsys,
    ):
        """Missing --music-brief exits with an error."""
        args = _make_args(audition_type="music", music_brief=None)

        with pytest.raises(SystemExit) as exc:
            audition_runner.run_audition(args)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "--music-brief" in err


# ═══════════════════════════════════════════════════════════════════════════
# 3. Visual auditions (stub)
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualAuditions:
    """Visual auditions are a stub for v1 — they log an empty session."""

    def test_visual_logs_empty_session(self, tmp_output_dir, in_memory_catalog):
        args = _make_args(audition_type="visual")
        output_dir = audition_runner.run_audition(args)

        # Session is logged with zero results — runner does not crash.
        rows = in_memory_catalog._conn.execute(
            "SELECT audition_type, results_json FROM auditions"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["audition_type"] == "visual"
        assert json.loads(rows[0]["results_json"]) == []
        assert output_dir is not None


# ═══════════════════════════════════════════════════════════════════════════
# 4. Approval policy
# ═══════════════════════════════════════════════════════════════════════════


class TestApprovalPolicy:
    """Verify --approve-all, --no-approve, and the default behaviour."""

    def _setup_voice_run(self, tmp_output_dir, in_memory_catalog, **overrides):
        """Helper: run a voice audition that produces 2 clips."""
        fake_engine = MagicMock()
        fake_engine.synthesize.return_value = _fake_synthesized_audio()

        # Pre-create a series so approve_all can attach to it.
        in_memory_catalog._conn.execute(
            """INSERT INTO series (series_id, title, created_at)
               VALUES ('s1', 'Show', '2025-01-01')"""
        )
        in_memory_catalog._conn.commit()

        args = _make_args(
            audition_type="voice",
            speaker="Host",
            audition_count=2,
            **overrides,
        )
        with patch(
            "vox_romana.tts.engines.kokoro_engine.KokoroEngine",
            return_value=fake_engine,
        ):
            audition_runner.run_audition(args)

    def test_approve_all_writes_assets(self, tmp_output_dir, in_memory_catalog):
        """--approve-all + --series populates series_assets."""
        self._setup_voice_run(
            tmp_output_dir, in_memory_catalog,
            approve_all=True, series="s1",
        )

        rows = in_memory_catalog._conn.execute(
            "SELECT series_id, asset_type, asset_id FROM series_assets"
        ).fetchall()
        assert len(rows) == 2
        assert all(r["series_id"] == "s1" for r in rows)
        assert all(r["asset_type"] == "voice" for r in rows)

        # Audition row should be marked chosen='all'.
        chosen = in_memory_catalog._conn.execute(
            "SELECT chosen FROM auditions"
        ).fetchone()
        assert chosen["chosen"] == "all"

    def test_approve_all_without_series_errors(
        self, tmp_output_dir, in_memory_catalog, capsys,
    ):
        """--approve-all without --series exits with an error."""
        with pytest.raises(SystemExit) as exc:
            self._setup_voice_run(
                tmp_output_dir, in_memory_catalog,
                approve_all=True, series=None,
            )
        assert exc.value.code == 1
        assert "--series" in capsys.readouterr().err

    def test_no_approve_skips_assets(self, tmp_output_dir, in_memory_catalog):
        """--no-approve logs the audition but creates no series_assets rows."""
        self._setup_voice_run(
            tmp_output_dir, in_memory_catalog,
            no_approve=True, series="s1",
        )

        # Audition session exists but no assets were promoted.
        n_auditions = in_memory_catalog._conn.execute(
            "SELECT COUNT(*) AS n FROM auditions"
        ).fetchone()["n"]
        n_assets = in_memory_catalog._conn.execute(
            "SELECT COUNT(*) AS n FROM series_assets"
        ).fetchone()["n"]
        assert n_auditions == 1
        assert n_assets == 0

    def test_default_skips_approval(self, tmp_output_dir, in_memory_catalog):
        """No flags → audition logged, no assets approved (manual step)."""
        self._setup_voice_run(
            tmp_output_dir, in_memory_catalog,
            series="s1",
        )
        n_assets = in_memory_catalog._conn.execute(
            "SELECT COUNT(*) AS n FROM series_assets"
        ).fetchone()["n"]
        assert n_assets == 0

    def test_approve_all_and_no_approve_conflict(
        self, tmp_output_dir, in_memory_catalog, capsys,
    ):
        """Both flags simultaneously → exits with an error."""
        with pytest.raises(SystemExit) as exc:
            self._setup_voice_run(
                tmp_output_dir, in_memory_catalog,
                approve_all=True, no_approve=True, series="s1",
            )
        assert exc.value.code == 1
        assert "mutually exclusive" in capsys.readouterr().err


# ═══════════════════════════════════════════════════════════════════════════
# 5. Top-level guards
# ═══════════════════════════════════════════════════════════════════════════


class TestTopLevelGuards:
    """Sanity checks on run_audition()'s argument validation."""

    def test_unknown_audition_type_errors(
        self, tmp_output_dir, in_memory_catalog, capsys,
    ):
        args = _make_args(audition_type="bogus")
        with pytest.raises(SystemExit) as exc:
            audition_runner.run_audition(args)
        assert exc.value.code == 1
        assert "audition-type" in capsys.readouterr().err

    def test_missing_audition_type_errors(
        self, tmp_output_dir, in_memory_catalog, capsys,
    ):
        args = _make_args(audition_type=None)
        with pytest.raises(SystemExit) as exc:
            audition_runner.run_audition(args)
        assert exc.value.code == 1

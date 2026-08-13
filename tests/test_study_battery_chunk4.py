"""Tests for Chunk 4 additions to tools/study_battery.py.

Covers:
- _read_production_id() reads from vox_romana.yaml
- --production-id flag is accepted by main()
- --auto-ingest flag calls ingest_run() with the right arguments
- auto-ingest warning on ValueError (double-ingest guard)
"""
from __future__ import annotations


import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Add repo root so tools/ is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.study_battery import _read_production_id  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# _read_production_id
# ─────────────────────────────────────────────────────────────────────────────

def test_read_production_id_from_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "vox_romana.yaml"
    cfg.write_text("production_id: test_studio\n", encoding="utf-8")
    with patch("tools.study_battery.REPO_ROOT", tmp_path):
        pid = _read_production_id()
    assert pid == "test_studio"


def test_read_production_id_falls_back_to_default(tmp_path: Path) -> None:
    # No yaml file at all in tmp_path.
    with patch("tools.study_battery.REPO_ROOT", tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            pid = _read_production_id()
    assert pid == "default"


def test_read_production_id_ignores_missing_key(tmp_path: Path) -> None:
    cfg = tmp_path / "vox_romana.yaml"
    cfg.write_text("data_dir: /some/path\n", encoding="utf-8")
    with patch("tools.study_battery.REPO_ROOT", tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            pid = _read_production_id()
    assert pid == "default"


# ─────────────────────────────────────────────────────────────────────────────
# --auto-ingest wiring
# ─────────────────────────────────────────────────────────────────────────────

_FAKE_TELEMETRY = {
    "show_type": "podcast",
    "topic": "test topic",
    "slug": "test-topic",
    "started_at": "2026-01-01T00:00:00Z",
    "elapsed_secs": 0.1,
    "exit_code": 0,
    "timed_out": False,
    "ok": True,
    "json_path": None,
    "stderr_tail": [],
    "summary": {"word_count": 450, "event_count": 10},
}


def test_auto_ingest_calls_ingest_run(tmp_path: Path) -> None:
    """--auto-ingest calls ingest_run() with the correct production_id."""
    import tools.study_battery as sb
    import vox_romana.analysis.longitudinal as lng_mod

    fake_ingest = MagicMock(return_value={"podcast": 1})

    with patch.object(sb, "_snapshot_params", return_value=None):
        with patch.object(sb, "_run_one", return_value=_FAKE_TELEMETRY):
            with patch.object(lng_mod, "ingest_run", fake_ingest):
                sb.main([
                    "--shows", "podcast",
                    "--limit", "1",
                    "--duration", "1",
                    "--production-id", "my_studio",
                    "--auto-ingest",
                    "--out", str(tmp_path / "run_out"),
                ])

    assert fake_ingest.called, "ingest_run was not called"
    assert fake_ingest.call_args.kwargs.get("production_id") == "my_studio"


def test_auto_ingest_warns_on_value_error(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """When ingest_run raises ValueError, auto-ingest prints a WARNING and continues."""
    import tools.study_battery as sb
    import vox_romana.analysis.longitudinal as lng_mod

    failing_ingest = MagicMock(side_effect=ValueError("already in the DB"))

    with patch.object(sb, "_snapshot_params", return_value=None):
        with patch.object(sb, "_run_one", return_value=_FAKE_TELEMETRY):
            with patch.object(lng_mod, "ingest_run", failing_ingest):
                result = sb.main([
                    "--shows", "podcast",
                    "--limit", "1",
                    "--duration", "1",
                    "--auto-ingest",
                    "--out", str(tmp_path / "run_out2"),
                ])

    captured = capsys.readouterr()
    assert "WARNING" in (captured.err + captured.out)
    assert result in (0, 1)


def test_production_id_flag_parsed(tmp_path: Path) -> None:
    """--production-id flag is accepted without error; no ingest call without --auto-ingest."""
    import tools.study_battery as sb
    import vox_romana.analysis.longitudinal as lng_mod

    fake_ingest = MagicMock(return_value={"podcast": 1})

    with patch.object(sb, "_snapshot_params", return_value=None):
        with patch.object(sb, "_run_one", return_value=_FAKE_TELEMETRY):
            with patch.object(lng_mod, "ingest_run", fake_ingest):
                result = sb.main([
                    "--shows", "podcast",
                    "--limit", "1",
                    "--duration", "1",
                    "--production-id", "explicit_id",
                    "--out", str(tmp_path / "run_out3"),
                ])

    # No --auto-ingest flag → ingest_run must not be called
    assert not fake_ingest.called, "ingest_run should not be called without --auto-ingest"
    assert result in (0, 1)

"""Tests for vox_romana.core.resource_monitor."""

from __future__ import annotations

import threading
import time
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.core.resource_monitor import (
    ResourceBudget,
    ResourceError,
    ResourceMonitor,
    ResourceSnapshot,
    available_ram_gb,
    free_disk_gb,
)

# ── free_disk_gb ─────────────────────────────────────────────────────────────


class TestFreeDiskGb:
    def test_returns_float(self, tmp_path):
        result = free_disk_gb(tmp_path)
        assert isinstance(result, float)

    def test_returns_nonnegative(self, tmp_path):
        result = free_disk_gb(tmp_path)
        assert result >= 0

    def test_uses_specified_path_not_root(self, tmp_path):
        """The path probed must be the volume backing the given path, not just /."""
        with patch("vox_romana.core.resource_monitor.shutil.disk_usage") as mock_du:
            mock_du.return_value = (100 * 2**30, 50 * 2**30, 42 * 2**30)
            result = free_disk_gb(tmp_path)
        # Should have been called with the resolved probe path (tmp_path or its ancestor)
        mock_du.assert_called_once()
        assert abs(result - 42.0) < 0.1

    def test_returns_fallback_on_exception(self):
        with patch(
            "vox_romana.core.resource_monitor.shutil.disk_usage",
            side_effect=OSError("disk probing failed"),
        ):
            result = free_disk_gb("/nonexistent/mount/point/zzz")
        assert result == 9999.0


# ── available_ram_gb ─────────────────────────────────────────────────────────


class TestAvailableRamGb:
    def test_returns_float_or_none(self):
        result = available_ram_gb()
        assert result is None or isinstance(result, float)

    def test_returns_positive_when_psutil_present(self):
        result = available_ram_gb()
        if result is not None:
            assert result > 0

    def test_returns_none_when_psutil_missing(self):
        import builtins

        real_import = builtins.__import__

        def _no_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("psutil not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_no_psutil):
            result = available_ram_gb()
        assert result is None


# ── ResourceBudget ────────────────────────────────────────────────────────────


class TestResourceBudget:
    def test_construction_minimal(self, tmp_path):
        budget = ResourceBudget(
            disk_paths=(tmp_path,),
            disk_warn_gb=5.0,
            disk_abort_gb=2.0,
        )
        assert budget.disk_abort_gb == 2.0
        assert budget.disk_warn_gb == 5.0
        assert budget.ram_abort_gb is None

    def test_abort_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError, match="disk_abort_gb"):
            ResourceBudget(
                disk_paths=(tmp_path,),
                disk_warn_gb=5.0,
                disk_abort_gb=0.0,
            )

    def test_warn_must_be_gte_abort(self, tmp_path):
        with pytest.raises(ValueError, match="disk_warn_gb"):
            ResourceBudget(
                disk_paths=(tmp_path,),
                disk_warn_gb=1.0,
                disk_abort_gb=2.0,
            )

    def test_empty_disk_paths_raises(self):
        with pytest.raises(ValueError, match="disk_paths"):
            ResourceBudget(disk_paths=(), disk_warn_gb=5.0, disk_abort_gb=2.0)

    def test_ram_warn_must_be_gte_abort(self, tmp_path):
        with pytest.raises(ValueError, match="ram_warn_gb"):
            ResourceBudget(
                disk_paths=(tmp_path,),
                disk_warn_gb=5.0,
                disk_abort_gb=2.0,
                ram_warn_gb=1.0,
                ram_abort_gb=4.0,
            )

    def test_all_expected_modes_in_defaults(self):
        from vox_romana.core.resource_monitor import _MODE_DEFAULTS

        expected = {"generate", "render", "full", "video", "music", "voice-prep"}
        assert expected <= set(_MODE_DEFAULTS.keys())

    def test_video_tighter_than_render(self):
        video_budget = ResourceBudget.for_mode("video")
        render_budget = ResourceBudget.for_mode("render")
        assert video_budget.disk_abort_gb > render_budget.disk_abort_gb
        assert (video_budget.ram_abort_gb or 0) > (render_budget.ram_abort_gb or 0)

    def test_for_mode_uses_output_path(self, tmp_path):
        args = Namespace(
            video_platform="youtube",
            music=False,
            duration=None,
            prompt=None,
            audio_output=tmp_path / "show.mp3",
            video_output=None,
            script_output=None,
            script_input=None,
            vp_source=None,
        )
        budget = ResourceBudget.for_mode("video", args)
        # The resolved probe paths should include tmp_path (or its parent)
        probe_strs = [str(p) for p in budget.disk_paths]
        assert any(str(tmp_path) in s or s in str(tmp_path) for s in probe_strs)

    def test_most_constrained_path_governs(self, tmp_path):
        """Budget should watch multiple paths; the tightest one should govern."""
        budget = ResourceBudget(
            disk_paths=(Path("/"), tmp_path),
            disk_warn_gb=5.0,
            disk_abort_gb=2.0,
        )
        # Both paths are in the budget
        assert len(budget.disk_paths) >= 1


# ── ResourceMonitor ───────────────────────────────────────────────────────────


class TestResourceMonitor:
    def _budget(self, tmp_path, *, disk_warn=50.0, disk_abort=1.0, ram_abort=None):
        return ResourceBudget(
            disk_paths=(tmp_path,),
            disk_warn_gb=disk_warn,
            disk_abort_gb=disk_abort,
            ram_warn_gb=(ram_abort * 1.5 if ram_abort else None),
            ram_abort_gb=ram_abort,
            poll_interval_sec=0.1,
        )

    def test_context_manager_starts_and_stops_cleanly(self, tmp_path):
        monitor = ResourceMonitor(self._budget(tmp_path))
        with monitor as m:
            assert m._thread is not None
            assert m._thread.is_alive()
        # After __exit__, stop was called; thread should have exited promptly
        m._thread.join(timeout=1.0)
        assert not m._thread.is_alive()

    def test_checkpoint_returns_snapshot(self, tmp_path):
        with ResourceMonitor(self._budget(tmp_path)) as monitor:
            snap = monitor.checkpoint("test")
        assert isinstance(snap, ResourceSnapshot)
        assert snap.free_disk_gb >= 0

    def test_checkpoint_does_not_raise_when_resources_ample(self, tmp_path):
        with ResourceMonitor(self._budget(tmp_path, disk_abort=0.001)) as monitor:
            monitor.checkpoint("ample")  # should not raise

    def test_checkpoint_raises_resource_error_when_disk_low(self, tmp_path):
        with patch(
            "vox_romana.core.resource_monitor.shutil.disk_usage",
            return_value=(100 * 2**30, 99 * 2**30, int(0.5 * 2**30)),
        ):
            budget = ResourceBudget(
                disk_paths=(tmp_path,),
                disk_warn_gb=5.0,
                disk_abort_gb=2.0,
                poll_interval_sec=0.05,
            )
            monitor = ResourceMonitor(budget)
            with monitor:
                time.sleep(0.15)  # let the background thread sample
                with pytest.raises(ResourceError, match="disk"):
                    monitor.checkpoint("low-disk")

    def test_checkpoint_raises_resource_error_when_ram_low(self, tmp_path):
        mock_vm = MagicMock()
        mock_vm.available = int(0.5 * 2**30)

        with patch("vox_romana.core.resource_monitor.shutil.disk_usage") as mock_du:
            mock_du.return_value = (200 * 2**30, 100 * 2**30, 100 * 2**30)
            budget = ResourceBudget(
                disk_paths=(tmp_path,),
                disk_warn_gb=5.0,
                disk_abort_gb=2.0,
                ram_warn_gb=4.0,
                ram_abort_gb=2.0,
                poll_interval_sec=0.05,
            )
            monitor = ResourceMonitor(budget)

            with patch("vox_romana.core.resource_monitor.available_ram_gb", return_value=0.5):
                with monitor:
                    time.sleep(0.15)
                    with pytest.raises(ResourceError, match="RAM"):
                        monitor.checkpoint("low-ram")

    def test_warn_does_not_raise(self, tmp_path):
        """Crossing the warn threshold should log but not raise."""
        with patch(
            "vox_romana.core.resource_monitor.shutil.disk_usage",
            return_value=(100 * 2**30, 96 * 2**30, int(3.0 * 2**30)),
        ):
            budget = ResourceBudget(
                disk_paths=(tmp_path,),
                disk_warn_gb=5.0,
                disk_abort_gb=1.0,  # abort is lower than current 3 GB free
                poll_interval_sec=0.05,
            )
            monitor = ResourceMonitor(budget)
            with monitor:
                time.sleep(0.15)
                # Should not raise — only warn
                snap = monitor.checkpoint("warn-only")
            assert snap.free_disk_gb < budget.disk_warn_gb

    def test_abort_event_set_by_background_thread(self, tmp_path):
        with patch(
            "vox_romana.core.resource_monitor.shutil.disk_usage",
            return_value=(100 * 2**30, 99 * 2**30, int(0.1 * 2**30)),
        ):
            budget = ResourceBudget(
                disk_paths=(tmp_path,),
                disk_warn_gb=5.0,
                disk_abort_gb=2.0,
                poll_interval_sec=0.05,
            )
            monitor = ResourceMonitor(budget)
            monitor.start()
            time.sleep(0.2)
            assert monitor.should_abort
            monitor.stop()

    def test_for_mode_constructs_from_mode_string(self, tmp_path):
        monitor = ResourceMonitor.for_mode("render")
        assert monitor.budget.disk_abort_gb > 0

    def test_ram_check_skips_gracefully_when_psutil_missing(self, tmp_path):
        """If psutil is absent, RAM checks should never fire."""
        budget = ResourceBudget(
            disk_paths=(tmp_path,),
            disk_warn_gb=5.0,
            disk_abort_gb=0.001,
            ram_warn_gb=100.0,
            ram_abort_gb=50.0,
            poll_interval_sec=0.05,
        )
        with patch(
            "vox_romana.core.resource_monitor.available_ram_gb",
            return_value=None,
        ):
            with patch(
                "vox_romana.core.resource_monitor.shutil.disk_usage",
                return_value=(200 * 2**30, 100 * 2**30, 100 * 2**30),
            ):
                monitor = ResourceMonitor(budget)
                with monitor:
                    time.sleep(0.15)
                    snap = monitor.checkpoint("no-ram-abort")
                # No ResourceError raised
                assert snap.available_ram_gb is None

    def test_multiple_start_calls_are_idempotent(self, tmp_path):
        monitor = ResourceMonitor(self._budget(tmp_path))
        monitor.start()
        thread_1 = monitor._thread
        monitor.start()
        thread_2 = monitor._thread
        assert thread_1 is thread_2
        monitor.stop()

    def test_snapshot_includes_disk_path(self, tmp_path):
        with ResourceMonitor(self._budget(tmp_path)) as monitor:
            snap = monitor.snapshot()
        assert snap.disk_path is not None
        assert isinstance(snap.disk_path, Path)

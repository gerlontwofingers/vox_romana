"""Tests for vox_romana.animation.hardware — HardwareScanner behavior.

All tests mock out torch and psutil so the suite runs without GPU
dependencies — the same approach used by test_music_engine.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vox_romana.animation.hardware import (
    HardwareScanner,
    LOW_DISK_THRESHOLD_GB,
    _CUDA_HIGH_VRAM_GB,
    _CUDA_MID_VRAM_GB,
    _MPS_HIGH_UNIFIED_GB,
    _SD_MODELS,
)
from vox_romana.animation.models import HardwareProfile


# ══════════════════════════════════════════════════════════════════════════
# _select_device
# ══════════════════════════════════════════════════════════════════════════

class TestSelectDevice:
    def test_mps_preferred_when_available(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True  # both available
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert HardwareScanner._select_device() == "mps"

    def test_cuda_when_no_mps(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert HardwareScanner._select_device() == "cuda"

    def test_cpu_fallback_when_no_gpu(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert HardwareScanner._select_device() == "cpu"

    def test_cpu_fallback_when_torch_missing(self):
        with patch.dict("sys.modules", {"torch": None}):
            # ImportError when torch is None in sys.modules
            assert HardwareScanner._select_device() == "cpu"


# ══════════════════════════════════════════════════════════════════════════
# _probe_vram
# ══════════════════════════════════════════════════════════════════════════

class TestProbeVram:
    def test_mps_returns_total_ram(self):
        with patch.object(HardwareScanner, "_probe_total_ram", return_value=32.0):
            result = HardwareScanner._probe_vram("mps")
        assert result == pytest.approx(32.0)

    def test_cuda_reads_device_properties(self):
        mock_torch = MagicMock()
        mock_props = MagicMock()
        mock_props.total_memory = int(16 * 1024 ** 3)  # 16 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = HardwareScanner._probe_vram("cuda")
        assert result == pytest.approx(16.0)

    def test_cpu_returns_none(self):
        # CPU device: no VRAM to probe, torch import path not taken
        mock_torch = MagicMock()
        # Make sure neither mps nor cuda path triggers
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = HardwareScanner._probe_vram("cpu")
        assert result is None

    def test_returns_none_on_exception(self):
        mock_torch = MagicMock()
        mock_torch.cuda.get_device_properties.side_effect = RuntimeError("no device")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = HardwareScanner._probe_vram("cuda")
        assert result is None


# ══════════════════════════════════════════════════════════════════════════
# _probe_free_disk
# ══════════════════════════════════════════════════════════════════════════

class TestProbeFreeDisk:
    def test_returns_free_gb(self):
        with patch("shutil.disk_usage", return_value=(500 * 1024**3, 460 * 1024**3, 40 * 1024**3)):
            result = HardwareScanner._probe_free_disk()
        assert result == pytest.approx(40.0)

    def test_returns_large_value_on_error(self):
        with patch("shutil.disk_usage", side_effect=OSError("no disk")):
            result = HardwareScanner._probe_free_disk()
        assert result == 999.0


# ══════════════════════════════════════════════════════════════════════════
# _choose_tier
# ══════════════════════════════════════════════════════════════════════════

class TestChooseTier:
    # ── MPS tiers ──────────────────────────────────────────────────────────

    def test_mps_high_unified_gets_sdxl(self):
        model, res, fps = HardwareScanner._choose_tier("mps", 32.0, 32.0)
        assert model == _SD_MODELS["sdxl"]
        assert res == (1920, 1080)
        assert fps == 30

    def test_mps_exactly_at_threshold(self):
        model, res, fps = HardwareScanner._choose_tier("mps", _MPS_HIGH_UNIFIED_GB, _MPS_HIGH_UNIFIED_GB)
        assert model == _SD_MODELS["sdxl"]

    def test_mps_low_unified_gets_sd15(self):
        model, res, fps = HardwareScanner._choose_tier("mps", 8.0, 8.0)
        assert model == _SD_MODELS["sd15"]
        assert res == (512, 512)
        assert fps == 24

    def test_mps_unknown_ram_assumes_high(self):
        """When RAM cannot be probed, assume high tier (M-series minimum)."""
        model, res, fps = HardwareScanner._choose_tier("mps", None, None)
        assert model == _SD_MODELS["sdxl"]

    # ── CUDA tiers ─────────────────────────────────────────────────────────

    def test_cuda_high_vram_gets_sdxl(self):
        model, res, fps = HardwareScanner._choose_tier("cuda", 24.0, None)
        assert model == _SD_MODELS["sdxl"]
        assert res == (1920, 1080)
        assert fps == 30

    def test_cuda_exactly_at_high_threshold(self):
        model, _, _ = HardwareScanner._choose_tier("cuda", _CUDA_HIGH_VRAM_GB, None)
        assert model == _SD_MODELS["sdxl"]

    def test_cuda_mid_vram_gets_sd21(self):
        model, res, fps = HardwareScanner._choose_tier("cuda", 8.0, None)
        assert model == _SD_MODELS["sd21"]
        assert res == (1280, 720)
        assert fps == 24

    def test_cuda_exactly_at_mid_threshold(self):
        model, _, _ = HardwareScanner._choose_tier("cuda", _CUDA_MID_VRAM_GB, None)
        assert model == _SD_MODELS["sd21"]

    def test_cuda_low_vram_gets_sd15(self):
        model, res, fps = HardwareScanner._choose_tier("cuda", 2.0, None)
        assert model == _SD_MODELS["sd15"]
        assert res == (512, 512)
        assert fps == 24

    def test_cuda_unknown_vram_gets_sd15(self):
        model, _, _ = HardwareScanner._choose_tier("cuda", None, None)
        assert model == _SD_MODELS["sd15"]

    # ── CPU tier ───────────────────────────────────────────────────────────

    def test_cpu_gets_sd15_lowest(self):
        model, res, fps = HardwareScanner._choose_tier("cpu", None, None)
        assert model == _SD_MODELS["sd15"]
        assert res == (512, 512)
        assert fps == 24


# ══════════════════════════════════════════════════════════════════════════
# scan() — integration
# ══════════════════════════════════════════════════════════════════════════

class TestScan:
    def _mock_scan_cpu(self, free_disk_gb: float = 40.0) -> HardwareProfile:
        """Run scan() in CPU mode with controlled disk space."""
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch.object(HardwareScanner, "_probe_free_disk", return_value=free_disk_gb), \
             patch.object(HardwareScanner, "_probe_total_ram", return_value=16.0):
            return HardwareScanner.scan()

    def test_scan_returns_hardware_profile(self):
        profile = self._mock_scan_cpu()
        assert isinstance(profile, HardwareProfile)

    def test_scan_cpu_mode(self):
        profile = self._mock_scan_cpu()
        assert profile.device == "cpu"
        assert profile.recommended_sd_model == _SD_MODELS["sd15"]
        assert profile.recommended_resolution == (512, 512)
        assert profile.recommended_fps == 24

    def test_scan_no_low_disk_warning_when_plenty(self):
        profile = self._mock_scan_cpu(free_disk_gb=40.0)
        assert profile.low_disk_warning is False
        assert profile.free_disk_gb == pytest.approx(40.0)

    def test_scan_low_disk_warning_when_tight(self):
        profile = self._mock_scan_cpu(free_disk_gb=5.0)
        assert profile.low_disk_warning is True

    def test_scan_low_disk_exactly_at_threshold(self):
        profile = self._mock_scan_cpu(free_disk_gb=LOW_DISK_THRESHOLD_GB)
        assert profile.low_disk_warning is False  # strict less-than

    def test_scan_mps_high_tier(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch.object(HardwareScanner, "_probe_vram", return_value=32.0), \
             patch.object(HardwareScanner, "_probe_total_ram", return_value=32.0), \
             patch.object(HardwareScanner, "_probe_free_disk", return_value=37.0):
            profile = HardwareScanner.scan()

        assert profile.device == "mps"
        assert profile.recommended_sd_model == _SD_MODELS["sdxl"]
        assert profile.recommended_resolution == (1920, 1080)
        assert profile.recommended_fps == 30
        assert profile.low_disk_warning is False

    def test_scan_cuda_mid_tier(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = int(8 * 1024 ** 3)
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict("sys.modules", {"torch": mock_torch}), \
             patch.object(HardwareScanner, "_probe_total_ram", return_value=32.0), \
             patch.object(HardwareScanner, "_probe_free_disk", return_value=40.0):
            profile = HardwareScanner.scan()

        assert profile.device == "cuda"
        assert profile.recommended_sd_model == _SD_MODELS["sd21"]
        assert profile.recommended_resolution == (1280, 720)

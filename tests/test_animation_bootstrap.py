"""Tests for vox_romana.animation.bootstrap.AnimationBootstrap."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vox_romana.animation.bootstrap import AnimationBootstrap
from vox_romana.animation.models import AnimationConfig, HardwareProfile


@pytest.fixture
def hw_profile() -> HardwareProfile:
    return HardwareProfile(
        device="mps",
        vram_gb=32.0,
        total_ram_gb=32.0,
        recommended_sd_model="stabilityai/stable-diffusion-xl-base-1.0",
        recommended_resolution=(1920, 1080),
        recommended_fps=30,
        low_disk_warning=False,
        free_disk_gb=37.0,
    )


@pytest.fixture
def cpu_profile() -> HardwareProfile:
    return HardwareProfile(
        device="cpu",
        vram_gb=None,
        total_ram_gb=8.0,
        recommended_sd_model="runwayml/stable-diffusion-v1-5",
        recommended_resolution=(512, 512),
        recommended_fps=24,
        low_disk_warning=True,
        free_disk_gb=5.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruction:
    def test_init_with_explicit_client(self) -> None:
        client = MagicMock()
        bootstrap = AnimationBootstrap(client)
        assert bootstrap._client is client

    def test_init_creates_client_when_none(self) -> None:
        with patch("vox_romana.core.bootstrap.DeepSeekClient") as MockClient:
            MockClient.return_value = MagicMock()
            bootstrap = AnimationBootstrap(api_key="fake-key")
            MockClient.assert_called_once_with(api_key="fake-key")
            assert bootstrap._client is MockClient.return_value


# ─────────────────────────────────────────────────────────────────────────────
# from_brief() — hardware adaptation
# ─────────────────────────────────────────────────────────────────────────────

class TestFromBrief:
    def test_uses_provided_profile(self, hw_profile: HardwareProfile) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief("brief", hardware_profile=hw_profile)
        assert config.resolution == (1920, 1080)
        assert config.fps == 30
        assert config.image_model == "stabilityai/stable-diffusion-xl-base-1.0"

    def test_low_end_profile_yields_low_settings(
        self, cpu_profile: HardwareProfile,
    ) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief("brief", hardware_profile=cpu_profile)
        assert config.resolution == (512, 512)
        assert config.fps == 24
        assert config.image_model == "runwayml/stable-diffusion-v1-5"

    def test_scans_hardware_when_no_profile_given(self) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        fake_profile = HardwareProfile(
            device="cpu", vram_gb=None, total_ram_gb=4.0,
            recommended_sd_model="x", recommended_resolution=(256, 256),
            recommended_fps=24, low_disk_warning=False, free_disk_gb=100.0,
        )
        with patch(
            "vox_romana.animation.bootstrap.HardwareScanner.scan",
            return_value=fake_profile,
        ) as mock_scan:
            config = bootstrap.from_brief("brief")
            mock_scan.assert_called_once()
            assert config.resolution == (256, 256)

    def test_returns_animation_config(self, hw_profile: HardwareProfile) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief("brief", hardware_profile=hw_profile)
        assert isinstance(config, AnimationConfig)

    def test_default_enabled_true(self, hw_profile: HardwareProfile) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief("brief", hardware_profile=hw_profile)
        assert config.enabled is True

    def test_enabled_false_via_argument(self, hw_profile: HardwareProfile) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief(
            "brief", hardware_profile=hw_profile, enabled=False,
        )
        assert config.enabled is False

    def test_default_render_backend_is_moviepy(
        self, hw_profile: HardwareProfile,
    ) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief("brief", hardware_profile=hw_profile)
        assert config.render_backend == "moviepy"

    def test_visual_brief_stored_on_config(self, hw_profile: HardwareProfile) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief(
            "brief", hardware_profile=hw_profile,
            visual_brief="noir, high contrast",
        )
        assert config.visual_brief == "noir, high contrast"

    def test_visual_brief_defaults_to_none(self, hw_profile: HardwareProfile) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief("brief", hardware_profile=hw_profile)
        assert config.visual_brief is None


# ─────────────────────────────────────────────────────────────────────────────
# Overrides
# ─────────────────────────────────────────────────────────────────────────────

class TestOverrides:
    def test_override_resolution(self, hw_profile: HardwareProfile) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief(
            "brief", hardware_profile=hw_profile,
            overrides={"resolution": (1080, 1920)},
        )
        assert config.resolution == (1080, 1920)

    def test_override_fps(self, hw_profile: HardwareProfile) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief(
            "brief", hardware_profile=hw_profile, overrides={"fps": 60},
        )
        assert config.fps == 60

    def test_override_output_format(self, hw_profile: HardwareProfile) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief(
            "brief", hardware_profile=hw_profile,
            overrides={"output_format": "webm"},
        )
        assert config.output_format == "webm"

    def test_override_can_disable(self, hw_profile: HardwareProfile) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief(
            "brief", hardware_profile=hw_profile, overrides={"enabled": False},
        )
        assert config.enabled is False

    def test_unknown_override_keys_silently_ignored(
        self, hw_profile: HardwareProfile,
    ) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        # Should not raise
        config = bootstrap.from_brief(
            "brief", hardware_profile=hw_profile,
            overrides={"nonexistent_field": "x"},
        )
        assert isinstance(config, AnimationConfig)

    def test_no_overrides_dict_is_ok(self, hw_profile: HardwareProfile) -> None:
        bootstrap = AnimationBootstrap(MagicMock())
        config = bootstrap.from_brief(
            "brief", hardware_profile=hw_profile, overrides=None,
        )
        assert isinstance(config, AnimationConfig)

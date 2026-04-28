"""Tests for vox_romana.cli.animation_runner.

Verifies run_animation() wiring: correct pipeline call order,
early-exit when animation is disabled, output directory derivation,
visual_brief forwarding, brief reading from prompt file, and event
loading from show JSON.

All animation deps (AnimationBootstrap, AnimationDirector, etc.) are
mocked — no real LLM calls, torch, or diffusers needed.

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_animation_runner_cli.py -v
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.cli.animation_runner import run_animation


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_args(
    prompt: Path,
    *,
    visual_brief: str | None = None,
    video_output: Path | None = None,
    quiet: bool = True,
) -> argparse.Namespace:
    """Build a minimal args namespace for run_animation() tests."""
    ns = argparse.Namespace()
    ns.prompt = prompt
    ns.visual_brief = visual_brief
    ns.video_output = video_output
    ns.quiet = quiet
    return ns


def _write_show_json(
    tmp_path: Path,
    events: list[dict] | None = None,
) -> Path:
    """Write a minimal show JSON to tmp_path and return its Path."""
    if events is None:
        events = [
            {"global_order": 0, "speaker_name": "Alice", "text": "Hello world"},
            {"global_order": 1, "speaker_name": "Bob", "text": "Goodbye world"},
        ]
    data = {"debate_id": "test_001", "events": events}
    p = tmp_path / "show.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _enabled_config() -> MagicMock:
    """Return a mock AnimationConfig with enabled=True."""
    cfg = MagicMock()
    cfg.enabled = True
    cfg.render_backend = "moviepy"
    return cfg


def _disabled_config() -> MagicMock:
    """Return a mock AnimationConfig with enabled=False."""
    cfg = MagicMock()
    cfg.enabled = False
    cfg.render_backend = None
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline call order and delegation
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineCallOrder:
    """Verify all four stages are called in the correct sequence."""

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_calls_all_four_stages(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """All four stages should run: scan → bootstrap → director → render."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("Two economists debate UBI.", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"
        audio_path.touch()

        mock_profile = MagicMock()
        MockHardware.scan.return_value = mock_profile
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        mock_plan = MagicMock()
        MockDirector.return_value.run.return_value = mock_plan
        fake_video = tmp_path / "video" / "video.mp4"
        MockRenderer.return_value.render.return_value = fake_video

        args = _make_args(prompt)
        result = run_animation(args, json_path, audio_path)

        MockHardware.scan.assert_called_once()
        MockBootstrap.return_value.from_brief.assert_called_once()
        MockDirector.return_value.run.assert_called_once()
        MockRenderer.return_value.render.assert_called_once()
        assert result == fake_video

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_early_exit_when_disabled(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """When config.enabled is False, director and renderer are NOT called."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"

        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _disabled_config()

        args = _make_args(prompt)
        result = run_animation(args, json_path, audio_path)

        assert result is None
        MockDirector.return_value.run.assert_not_called()
        MockRenderer.return_value.render.assert_not_called()

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_returns_video_path_from_renderer(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """Return value should be whatever AnimationRenderer.render() returns."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"

        expected_path = Path("/some/output/video.mp4")
        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = expected_path

        args = _make_args(prompt)
        result = run_animation(args, json_path, audio_path)

        assert result == expected_path


# ─────────────────────────────────────────────────────────────────────────────
# Brief reading
# ─────────────────────────────────────────────────────────────────────────────

class TestBriefReading:
    """Verify the brief is read from args.prompt."""

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_reads_brief_from_prompt_file(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """from_brief() should receive the exact text from the prompt file."""
        brief_text = "Primatologists discuss migration in the Amazon."
        prompt = tmp_path / "brief.txt"
        prompt.write_text(brief_text, encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"

        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt)
        run_animation(args, json_path, audio_path)

        call_args = MockBootstrap.return_value.from_brief.call_args
        assert call_args[0][0] == brief_text

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_brief_also_forwarded_to_director(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """AnimationDirector.run() should receive the same brief text."""
        brief_text = "Climate scientists debate carbon capture."
        prompt = tmp_path / "brief.txt"
        prompt.write_text(brief_text, encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"

        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt)
        run_animation(args, json_path, audio_path)

        call_args = MockDirector.return_value.run.call_args
        assert call_args[0][0] == brief_text


# ─────────────────────────────────────────────────────────────────────────────
# Event loading
# ─────────────────────────────────────────────────────────────────────────────

class TestEventLoading:
    """Verify events are loaded from the show JSON and forwarded to the director."""

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_passes_events_from_json_to_director(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """AnimationDirector.run() should receive the events list from the JSON."""
        expected_events = [
            {"global_order": 0, "speaker_name": "Host", "text": "Welcome"},
            {"global_order": 1, "speaker_name": "Alice", "text": "Thank you"},
            {"global_order": 2, "speaker_name": "Bob", "text": "Indeed"},
        ]
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path, events=expected_events)
        audio_path = tmp_path / "audio.mp3"

        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt)
        run_animation(args, json_path, audio_path)

        call_args = MockDirector.return_value.run.call_args
        # Second positional arg is the events iterable.
        actual_events = list(call_args[0][1])
        assert actual_events == expected_events


# ─────────────────────────────────────────────────────────────────────────────
# visual_brief forwarding
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualBrief:
    """Verify visual_brief is forwarded to both bootstrap and director."""

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_visual_brief_forwarded_to_bootstrap(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """from_brief() should receive visual_brief from args."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"

        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt, visual_brief="documentary, warm lighting")
        run_animation(args, json_path, audio_path)

        call_kwargs = MockBootstrap.return_value.from_brief.call_args[1]
        assert call_kwargs.get("visual_brief") == "documentary, warm lighting"

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_visual_brief_forwarded_to_director(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """AnimationDirector.run() should receive visual_brief from args."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"

        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt, visual_brief="cinematic handheld")
        run_animation(args, json_path, audio_path)

        call_kwargs = MockDirector.return_value.run.call_args[1]
        assert call_kwargs.get("visual_brief") == "cinematic handheld"

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_visual_brief_none_when_not_set(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """When visual_brief is not set, None is forwarded (not omitted)."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"

        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt, visual_brief=None)
        run_animation(args, json_path, audio_path)

        bootstrap_kwargs = MockBootstrap.return_value.from_brief.call_args[1]
        assert bootstrap_kwargs.get("visual_brief") is None
        director_kwargs = MockDirector.return_value.run.call_args[1]
        assert director_kwargs.get("visual_brief") is None


# ─────────────────────────────────────────────────────────────────────────────
# Output directory derivation
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputDirectory:
    """Verify output_dir derivation from audio_path and args.video_output."""

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_derives_output_dir_from_audio_path(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """Without --video-output, output_dir should be audio_path.parent / 'video'."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"

        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt, video_output=None)
        run_animation(args, json_path, audio_path)

        render_call_kwargs = MockRenderer.return_value.render.call_args
        # Second positional arg is output_dir.
        actual_output_dir = render_call_kwargs[0][1]
        assert actual_output_dir == tmp_path / "video"

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_uses_explicit_video_output(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """When --video-output is set, output_dir should use that path."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"
        custom_output = tmp_path / "my_videos"

        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt, video_output=custom_output)
        run_animation(args, json_path, audio_path)

        render_call_kwargs = MockRenderer.return_value.render.call_args
        actual_output_dir = render_call_kwargs[0][1]
        assert actual_output_dir == custom_output

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_audio_path_forwarded_to_renderer(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """audio_path should be forwarded to AnimationRenderer.render()."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "soundtrack.mp3"
        audio_path.touch()

        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt)
        run_animation(args, json_path, audio_path)

        render_kwargs = MockRenderer.return_value.render.call_args[1]
        assert render_kwargs.get("audio_path") == audio_path

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_event_durations_is_none(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """event_durations=None should be passed to renderer (v1 behavior)."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"

        MockHardware.scan.return_value = MagicMock()
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt)
        run_animation(args, json_path, audio_path)

        render_kwargs = MockRenderer.return_value.render.call_args[1]
        assert render_kwargs.get("event_durations") is None


# ─────────────────────────────────────────────────────────────────────────────
# Hardware profile forwarding
# ─────────────────────────────────────────────────────────────────────────────

class TestHardwareProfileForwarding:
    """Verify the hardware profile flows from scan() into bootstrap and director."""

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_scan_result_forwarded_to_bootstrap(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """The HardwareProfile from scan() should be passed to from_brief()."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"

        sentinel_profile = MagicMock(name="hardware_profile")
        MockHardware.scan.return_value = sentinel_profile
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt)
        run_animation(args, json_path, audio_path)

        bootstrap_kwargs = MockBootstrap.return_value.from_brief.call_args[1]
        assert bootstrap_kwargs.get("hardware_profile") is sentinel_profile

    @patch("vox_romana.animation.AnimationRenderer")
    @patch("vox_romana.animation.AnimationDirector")
    @patch("vox_romana.animation.AnimationBootstrap")
    @patch("vox_romana.animation.HardwareScanner")
    @patch("vox_romana.core.client.DeepSeekClient")
    def test_scan_result_forwarded_to_director(
        self,
        MockClient,
        MockHardware,
        MockBootstrap,
        MockDirector,
        MockRenderer,
        tmp_path,
    ):
        """The HardwareProfile from scan() should be passed to director.run()."""
        prompt = tmp_path / "brief.txt"
        prompt.write_text("brief", encoding="utf-8")
        json_path = _write_show_json(tmp_path)
        audio_path = tmp_path / "audio.mp3"

        sentinel_profile = MagicMock(name="hardware_profile")
        MockHardware.scan.return_value = sentinel_profile
        MockBootstrap.return_value.from_brief.return_value = _enabled_config()
        MockDirector.return_value.run.return_value = MagicMock()
        MockRenderer.return_value.render.return_value = Path("/fake/video.mp4")

        args = _make_args(prompt)
        run_animation(args, json_path, audio_path)

        director_kwargs = MockDirector.return_value.run.call_args[1]
        assert director_kwargs.get("hardware_profile") is sentinel_profile

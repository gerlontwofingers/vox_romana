"""Tests for AnimationRenderer — pipeline orchestration.

Strategy:
    All external dependencies (ImageGenerator, TimelineSync, backends) are
    mocked so tests run without diffusers, moviepy, or torch installed.

    The renderer is tested through its public ``render()`` method only.
    We verify:
        - Correct dispatch to the right backend class
        - ImageGenerator.generate_for_plan() called (or skipped) correctly
        - TimelineSync.sync() called (or skipped) correctly
        - output_dir is created before backend.render() is called
        - ValueError raised for unknown backend names
        - Error propagation from backends (ImportError, NotImplementedError)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from vox_romana.animation.backends.base import BaseAnimationBackend
from vox_romana.animation.image_gen import ImageGenerator
from vox_romana.animation.models import AnimationPlan, SceneSpec
from vox_romana.animation.renderer import AnimationRenderer


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_plan(backend: str = "moviepy") -> AnimationPlan:
    """Minimal AnimationPlan for renderer tests."""
    return AnimationPlan(
        plan_id="test-plan",
        brief="test brief",
        style="cinematic",
        color_palette=["#000000"],
        scenes=[
            SceneSpec(
                scene_id="scene_00",
                event_global_orders=[0, 1],
                start_time_sec=0.0,
                end_time_sec=5.0,
            )
        ],
        render_backend=backend,
        resolution=(1920, 1080),
        fps=30,
        image_model="stabilityai/stable-diffusion-xl-base-1.0",
    )


def _make_mock_backend(name: str = "moviepy") -> MagicMock:
    """Create a mock backend that records calls and returns a fake path."""
    backend_instance = MagicMock(spec=BaseAnimationBackend)
    backend_instance.name = name
    backend_instance.render.return_value = Path("/tmp/video.mp4")
    return backend_instance


def _make_mock_backend_class(backend_instance: MagicMock) -> MagicMock:
    """Create a mock backend CLASS whose __call__() returns backend_instance."""
    cls = MagicMock()
    cls.return_value = backend_instance
    return cls


def _make_renderer(
    backend_name: str = "moviepy",
    backend_instance: MagicMock | None = None,
    image_generator: MagicMock | None = None,
) -> tuple[AnimationRenderer, MagicMock]:
    """Build a renderer with mocked backends and image_generator.

    Returns:
        (renderer, mock_backend_instance)
    """
    if backend_instance is None:
        backend_instance = _make_mock_backend(backend_name)
    backend_cls = _make_mock_backend_class(backend_instance)

    ig = image_generator or MagicMock(spec=ImageGenerator)
    ig.generate_for_plan.return_value = _make_plan(backend_name)

    renderer = AnimationRenderer(
        image_generator=ig,
        backends={backend_name: backend_cls},
    )
    return renderer, backend_instance


# ── Backend dispatch ───────────────────────────────────────────────────────

class TestBackendDispatch:
    def test_dispatches_to_moviepy_backend(self, tmp_path: Path):
        """plan.render_backend='moviepy' → MoviePyBackend is instantiated."""
        renderer, mock_backend = _make_renderer("moviepy")
        plan = _make_plan("moviepy")

        renderer.render(plan, output_dir=tmp_path)

        mock_backend.render.assert_called_once()

    def test_dispatches_to_manim_backend(self, tmp_path: Path):
        """plan.render_backend='manim' → ManifoldBackend is instantiated."""
        renderer, mock_backend = _make_renderer("manim")
        plan = _make_plan("manim")

        renderer.render(plan, output_dir=tmp_path)

        mock_backend.render.assert_called_once()

    def test_dispatches_to_ffmpeg_backend(self, tmp_path: Path):
        """plan.render_backend='ffmpeg' → FFmpegBackend is instantiated."""
        renderer, mock_backend = _make_renderer("ffmpeg")
        plan = _make_plan("ffmpeg")

        renderer.render(plan, output_dir=tmp_path)

        mock_backend.render.assert_called_once()

    def test_unknown_backend_raises_value_error(self, tmp_path: Path):
        """plan.render_backend='unknown' → ValueError with helpful message."""
        ig = MagicMock(spec=ImageGenerator)
        ig.generate_for_plan.return_value = _make_plan("unknown")
        renderer = AnimationRenderer(
            image_generator=ig,
            backends={"moviepy": MagicMock()},  # 'unknown' not registered
        )
        plan = _make_plan("unknown")

        with pytest.raises(ValueError, match="unknown"):
            renderer.render(plan, output_dir=tmp_path)

    def test_error_message_lists_available_backends(self, tmp_path: Path):
        """ValueError for unknown backend mentions what's available."""
        ig = MagicMock(spec=ImageGenerator)
        ig.generate_for_plan.return_value = _make_plan("unknown")
        renderer = AnimationRenderer(
            image_generator=ig,
            backends={"alpha": MagicMock(), "beta": MagicMock()},
        )
        plan = _make_plan("unknown")

        with pytest.raises(ValueError, match="alpha|beta"):
            renderer.render(plan, output_dir=tmp_path)

    def test_backend_render_called_with_correct_args(self, tmp_path: Path):
        """backend.render(plan, audio_path=..., output_path=...) shape is correct."""
        renderer, mock_backend = _make_renderer("moviepy")
        plan = _make_plan("moviepy")
        audio = Path("/fake/show.wav")

        renderer.render(plan, output_dir=tmp_path, audio_path=audio)

        args, kwargs = mock_backend.render.call_args
        assert args[0] is plan
        assert kwargs.get("audio_path") == audio or args[1] == audio
        # output_path should be inside output_dir
        output_path = kwargs.get("output_path") or args[2]
        assert str(tmp_path) in str(output_path)


# ── Image generation ───────────────────────────────────────────────────────

class TestImageGeneration:
    def test_generate_for_plan_called_with_frames_dir(self, tmp_path: Path):
        """image_generator.generate_for_plan() called with output_dir/frames."""
        ig = MagicMock(spec=ImageGenerator)
        ig.generate_for_plan.return_value = _make_plan()
        renderer, _ = _make_renderer(image_generator=ig)
        plan = _make_plan()

        renderer.render(plan, output_dir=tmp_path)

        ig.generate_for_plan.assert_called_once_with(plan, tmp_path / "frames")

    def test_generate_for_plan_skipped_when_flag_set(self, tmp_path: Path):
        """skip_image_gen=True → generate_for_plan not called."""
        ig = MagicMock(spec=ImageGenerator)
        renderer, _ = _make_renderer(image_generator=ig)
        plan = _make_plan()

        renderer.render(plan, output_dir=tmp_path, skip_image_gen=True)

        ig.generate_for_plan.assert_not_called()

    def test_default_image_generator_created_when_not_injected(self):
        """AnimationRenderer() without image_generator creates a default one."""
        # We just test construction doesn't crash; don't call render
        # because that would try to load diffusers.
        renderer = AnimationRenderer(backends={"moviepy": MagicMock()})
        assert renderer._image_generator is not None
        assert isinstance(renderer._image_generator, ImageGenerator)


# ── Timeline sync ──────────────────────────────────────────────────────────

class TestTimelineSync:
    def test_timeline_sync_called_when_event_durations_provided(
        self, tmp_path: Path
    ):
        """TimelineSync.sync() is called when event_durations is non-empty."""
        renderer, _ = _make_renderer()
        plan = _make_plan()
        event_durations = {0: 2.5, 1: 3.5}

        with patch(
            "vox_romana.animation.renderer.TimelineSync.sync",
            return_value=plan,
        ) as mock_sync:
            renderer.render(
                plan, output_dir=tmp_path, event_durations=event_durations
            )
            mock_sync.assert_called_once_with(plan, event_durations)

    def test_timeline_sync_skipped_when_no_event_durations(self, tmp_path: Path):
        """TimelineSync.sync() not called when event_durations is None."""
        renderer, _ = _make_renderer()
        plan = _make_plan()

        with patch(
            "vox_romana.animation.renderer.TimelineSync.sync",
            return_value=plan,
        ) as mock_sync:
            renderer.render(plan, output_dir=tmp_path)
            mock_sync.assert_not_called()

    def test_timeline_sync_skipped_with_skip_flag(self, tmp_path: Path):
        """skip_timeline_sync=True → TimelineSync.sync() not called."""
        renderer, _ = _make_renderer()
        plan = _make_plan()

        with patch(
            "vox_romana.animation.renderer.TimelineSync.sync",
            return_value=plan,
        ) as mock_sync:
            renderer.render(
                plan,
                output_dir=tmp_path,
                event_durations={0: 2.5},
                skip_timeline_sync=True,
            )
            mock_sync.assert_not_called()

    def test_timeline_sync_skipped_when_empty_event_durations(
        self, tmp_path: Path
    ):
        """TimelineSync.sync() not called when event_durations is empty dict."""
        renderer, _ = _make_renderer()
        plan = _make_plan()

        with patch(
            "vox_romana.animation.renderer.TimelineSync.sync",
            return_value=plan,
        ) as mock_sync:
            renderer.render(plan, output_dir=tmp_path, event_durations={})
            mock_sync.assert_not_called()


# ── Output directory creation ──────────────────────────────────────────────

class TestOutputDirectory:
    def test_output_dir_created_if_not_exists(self, tmp_path: Path):
        """output_dir is created (mkdir parents=True) before render."""
        renderer, _ = _make_renderer()
        plan = _make_plan()
        nested = tmp_path / "deeply" / "nested" / "dir"

        assert not nested.exists()
        renderer.render(plan, output_dir=nested)
        assert nested.exists()

    def test_output_dir_already_exists_no_error(self, tmp_path: Path):
        """render() doesn't crash if output_dir already exists."""
        renderer, _ = _make_renderer()
        plan = _make_plan()
        tmp_path.mkdir(exist_ok=True)  # already exists

        # Should not raise FileExistsError
        renderer.render(plan, output_dir=tmp_path)


# ── Return value ───────────────────────────────────────────────────────────

class TestReturnValue:
    def test_render_returns_path_from_backend(self, tmp_path: Path):
        """render() returns whatever the backend returns."""
        expected = Path("/tmp/video.mp4")
        backend_instance = _make_mock_backend()
        backend_instance.render.return_value = expected
        renderer, _ = _make_renderer(backend_instance=backend_instance)
        plan = _make_plan()

        result = renderer.render(plan, output_dir=tmp_path)

        assert result == expected


# ── Error propagation ──────────────────────────────────────────────────────

class TestErrorPropagation:
    def test_backend_import_error_propagated(self, tmp_path: Path):
        """ImportError from backend.render() propagates out of renderer."""
        backend_instance = _make_mock_backend()
        backend_instance.render.side_effect = ImportError("moviepy not installed")
        renderer, _ = _make_renderer(backend_instance=backend_instance)
        plan = _make_plan()

        with pytest.raises(ImportError, match="moviepy"):
            renderer.render(plan, output_dir=tmp_path)

    def test_backend_not_implemented_error_propagated(self, tmp_path: Path):
        """NotImplementedError from stub backend propagates out of renderer."""
        backend_instance = _make_mock_backend("manim")
        backend_instance.render.side_effect = NotImplementedError(
            "The manim backend is not yet implemented"
        )
        renderer, _ = _make_renderer(
            backend_name="manim", backend_instance=backend_instance
        )
        plan = _make_plan("manim")

        with pytest.raises(NotImplementedError):
            renderer.render(plan, output_dir=tmp_path)

    def test_image_gen_error_propagated(self, tmp_path: Path):
        """RuntimeError from image_generator propagates cleanly."""
        ig = MagicMock(spec=ImageGenerator)
        ig.generate_for_plan.side_effect = RuntimeError("SD model failed")
        renderer, _ = _make_renderer(image_generator=ig)
        plan = _make_plan()

        with pytest.raises(RuntimeError, match="SD model"):
            renderer.render(plan, output_dir=tmp_path)


# ── Audio path ────────────────────────────────────────────────────────────

class TestAudioPath:
    def test_audio_path_none_by_default(self, tmp_path: Path):
        """audio_path defaults to None → passed as None to backend."""
        renderer, mock_backend = _make_renderer()
        plan = _make_plan()

        renderer.render(plan, output_dir=tmp_path)

        args, kwargs = mock_backend.render.call_args
        audio_arg = kwargs.get("audio_path", args[1] if len(args) > 1 else None)
        assert audio_arg is None

    def test_audio_path_forwarded_to_backend(self, tmp_path: Path):
        """audio_path is forwarded to backend.render()."""
        renderer, mock_backend = _make_renderer()
        plan = _make_plan()
        audio = Path("/some/audio.wav")

        renderer.render(plan, output_dir=tmp_path, audio_path=audio)

        args, kwargs = mock_backend.render.call_args
        audio_arg = kwargs.get("audio_path", args[1] if len(args) > 1 else None)
        assert audio_arg == audio


# ── Constructor ───────────────────────────────────────────────────────────

class TestConstructor:
    def test_default_backends_populated(self):
        """Default renderer has moviepy, manim, ffmpeg backends registered."""
        ig = MagicMock(spec=ImageGenerator)
        renderer = AnimationRenderer(image_generator=ig)
        assert "moviepy" in renderer._backends
        assert "manim" in renderer._backends
        assert "ffmpeg" in renderer._backends

    def test_custom_backends_override_defaults(self):
        """backends= kwarg replaces the default registry entirely."""
        ig = MagicMock(spec=ImageGenerator)
        custom = {"custom_engine": MagicMock()}
        renderer = AnimationRenderer(image_generator=ig, backends=custom)
        assert renderer._backends is custom
        assert "moviepy" not in renderer._backends

    def test_injected_image_generator_used(self):
        """Injected image_generator is stored as-is."""
        ig = MagicMock(spec=ImageGenerator)
        renderer = AnimationRenderer(image_generator=ig)
        assert renderer._image_generator is ig

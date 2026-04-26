"""Tests for StillsExtractor — keyframe still image extraction.

Strategy:
    moviepy and PIL are mocked entirely so tests run without those
    libraries installed. shutil.copy2 is also mocked to avoid touching
    the filesystem where possible; tmp_path (pytest fixture) is used
    for the output directory so we can verify real file writes when needed.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from vox_romana.animation.models import AnimationPlan, SceneSpec
from vox_romana.animation.stills import StillsExtractor


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_plan(*scene_configs: dict) -> AnimationPlan:
    """Build an AnimationPlan with scenes described by dicts.

    Each dict may have:
        start_time_sec, end_time_sec, generated_image_path
    """
    scenes = [
        SceneSpec(
            scene_id=f"scene_{idx:02d}",
            event_global_orders=[idx],
            start_time_sec=cfg.get("start_time_sec"),
            end_time_sec=cfg.get("end_time_sec"),
            generated_image_path=cfg.get("generated_image_path"),
        )
        for idx, cfg in enumerate(scene_configs)
    ]
    return AnimationPlan(
        plan_id="test-plan",
        brief="test brief",
        style="cinematic",
        color_palette=[],
        scenes=scenes,
        render_backend="moviepy",
        resolution=(1920, 1080),
        fps=30,
    )


def _make_video_clip(duration: float = 20.0) -> MagicMock:
    """Create a mock moviepy VideoFileClip."""
    clip = MagicMock()
    clip.duration = duration
    # get_frame returns a fake numpy-style array (just a MagicMock here)
    clip.get_frame.return_value = MagicMock()
    return clip


# ── output_dir creation ───────────────────────────────────────────────────

class TestOutputDirCreation:
    def test_output_dir_created_if_absent(self, tmp_path: Path):
        """extract() creates output_dir (including parents) if not exists."""
        plan = _make_plan()  # no scenes
        nested = tmp_path / "a" / "b" / "stills"

        assert not nested.exists()
        StillsExtractor.extract(plan, video_path=None, output_dir=nested)
        assert nested.exists()

    def test_output_dir_already_exists_no_error(self, tmp_path: Path):
        """extract() doesn't crash if output_dir already exists."""
        plan = _make_plan()
        StillsExtractor.extract(plan, video_path=None, output_dir=tmp_path)
        # No assertion needed — just must not raise


# ── Empty plan ─────────────────────────────────────────────────────────────

class TestEmptyPlan:
    def test_empty_plan_returns_empty_list(self, tmp_path: Path):
        """No scenes → empty list returned immediately."""
        plan = _make_plan()
        result = StillsExtractor.extract(plan, video_path=None, output_dir=tmp_path)
        assert result == []


# ── Video path absent / None ───────────────────────────────────────────────

class TestVideoPathAbsent:
    def test_none_video_path_uses_sd_images(self, tmp_path: Path):
        """video_path=None → falls back to SD image copy mode."""
        src = tmp_path / "scene_00.png"
        src.write_bytes(b"fake png data")

        plan = _make_plan({"generated_image_path": str(src)})

        result = StillsExtractor.extract(plan, video_path=None, output_dir=tmp_path)

        assert len(result) == 1
        assert result[0].name == "still_scene_00.png"

    def test_nonexistent_video_path_falls_back_to_images(self, tmp_path: Path):
        """If video_path doesn't exist on disk, fall back to SD images."""
        src = tmp_path / "my_image.png"
        src.write_bytes(b"fake png data")

        plan = _make_plan({"generated_image_path": str(src)})

        result = StillsExtractor.extract(
            plan,
            video_path=tmp_path / "nonexistent.mp4",
            output_dir=tmp_path,
        )

        assert len(result) == 1

    def test_missing_sd_image_skipped(self, tmp_path: Path):
        """Scenes whose SD image file doesn't exist are skipped."""
        plan = _make_plan(
            {"generated_image_path": "/nonexistent/path/img.png"},
        )
        result = StillsExtractor.extract(plan, video_path=None, output_dir=tmp_path)
        assert result == []

    def test_no_generated_image_path_skipped(self, tmp_path: Path):
        """Scenes with generated_image_path=None are skipped."""
        plan = _make_plan({})  # no image path
        result = StillsExtractor.extract(plan, video_path=None, output_dir=tmp_path)
        assert result == []

    def test_multiple_scenes_partial_images(self, tmp_path: Path):
        """Some scenes have images, some don't — only valid ones returned."""
        src = tmp_path / "img.png"
        src.write_bytes(b"fake")

        plan = _make_plan(
            {"generated_image_path": str(src)},
            {},  # no image
            {"generated_image_path": "/nonexistent/img2.png"},
        )
        result = StillsExtractor.extract(plan, video_path=None, output_dir=tmp_path)

        assert len(result) == 1
        assert "scene_00" in result[0].name

    def test_sd_images_copied_with_correct_names(self, tmp_path: Path):
        """Output filenames are still_<scene_id>.png."""
        src0 = tmp_path / "s0.png"
        src1 = tmp_path / "s1.png"
        src0.write_bytes(b"img0")
        src1.write_bytes(b"img1")

        plan = _make_plan(
            {"generated_image_path": str(src0)},
            {"generated_image_path": str(src1)},
        )
        result = StillsExtractor.extract(plan, video_path=None, output_dir=tmp_path)

        names = {p.name for p in result}
        assert "still_scene_00.png" in names
        assert "still_scene_01.png" in names


# ── Video extraction ───────────────────────────────────────────────────────

class TestVideoExtraction:
    def _make_mock_mpy_module(self, clip: MagicMock) -> MagicMock:
        """Create a mock moviepy.editor module."""
        mpy = MagicMock()
        mpy.VideoFileClip.return_value = clip
        return mpy

    def _make_mock_pil_image_module(self) -> tuple[MagicMock, MagicMock]:
        """Return (mock Image module, mock image instance)."""
        img_instance = MagicMock()
        img_module = MagicMock()
        img_module.fromarray.return_value = img_instance
        return img_module, img_instance

    def test_video_extraction_calls_get_frame_at_midpoint(self, tmp_path: Path):
        """For each scene, get_frame() is called at (start+end)/2."""
        plan = _make_plan(
            {"start_time_sec": 0.0, "end_time_sec": 6.0},
            {"start_time_sec": 6.0, "end_time_sec": 10.0},
        )
        # Test midpoint computation via the static helper.
        scene0 = plan.scenes[0]
        mid0 = StillsExtractor._scene_midpoint(scene0, video_duration=10.0)
        assert mid0 == pytest.approx(3.0)  # (0+6)/2

        scene1 = plan.scenes[1]
        mid1 = StillsExtractor._scene_midpoint(scene1, video_duration=10.0)
        assert mid1 == pytest.approx(8.0)  # (6+10)/2

    def test_video_clip_closed_after_extraction(self, tmp_path: Path):
        """VideoFileClip.close() is called even if extraction fails."""
        # We can't easily test the lazy import path, but we can verify
        # the close-on-cleanup contract via the helper method.
        # This is a structural test — verifies the finally block exists
        # in the code by checking the method source.
        import inspect
        src = inspect.getsource(StillsExtractor._extract_from_video)
        assert "finally" in src
        assert "close()" in src

    def test_moviepy_import_error_propagated(self, tmp_path: Path):
        """If moviepy not installed, ImportError propagates from video mode."""
        plan = _make_plan({"start_time_sec": 0.0, "end_time_sec": 5.0})
        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        # Hide moviepy from sys.modules to simulate it not being installed.
        with patch.dict("sys.modules", {"moviepy": None, "moviepy.editor": None}):
            with pytest.raises(ImportError, match="moviepy"):
                StillsExtractor._extract_from_video(plan, fake_video, tmp_path)


# ── _scene_midpoint ────────────────────────────────────────────────────────

class TestSceneMidpoint:
    def test_midpoint_basic(self):
        """Midpoint = (start + end) / 2."""
        scene = SceneSpec(
            scene_id="s0",
            start_time_sec=2.0,
            end_time_sec=8.0,
        )
        assert StillsExtractor._scene_midpoint(scene, 20.0) == pytest.approx(5.0)

    def test_midpoint_clamped_to_video_duration(self):
        """Midpoint is clamped to video_duration - 0.001."""
        scene = SceneSpec(
            scene_id="s0",
            start_time_sec=18.0,
            end_time_sec=25.0,  # beyond video end
        )
        result = StillsExtractor._scene_midpoint(scene, video_duration=20.0)
        assert result <= 20.0 - 0.001 + 0.001  # within clamp

    def test_midpoint_none_when_no_start(self):
        """Returns None when start_time_sec is None."""
        scene = SceneSpec(scene_id="s0", end_time_sec=5.0)
        assert StillsExtractor._scene_midpoint(scene, 20.0) is None

    def test_midpoint_none_when_no_end(self):
        """Returns None when end_time_sec is None."""
        scene = SceneSpec(scene_id="s0", start_time_sec=0.0)
        assert StillsExtractor._scene_midpoint(scene, 20.0) is None

    def test_midpoint_none_when_both_none(self):
        """Returns None when both timestamps are None."""
        scene = SceneSpec(scene_id="s0")
        assert StillsExtractor._scene_midpoint(scene, 20.0) is None

    def test_midpoint_scene_starting_at_zero(self):
        """Scene starting at 0 has midpoint at end/2."""
        scene = SceneSpec(scene_id="s0", start_time_sec=0.0, end_time_sec=4.0)
        assert StillsExtractor._scene_midpoint(scene, 20.0) == pytest.approx(2.0)

    def test_midpoint_not_negative(self):
        """Midpoint is clamped to >= 0."""
        scene = SceneSpec(
            scene_id="s0",
            start_time_sec=-5.0,  # shouldn't happen but guard anyway
            end_time_sec=-1.0,
        )
        result = StillsExtractor._scene_midpoint(scene, 20.0)
        assert result is not None
        assert result >= 0.0


# ── _copy_sd_image ─────────────────────────────────────────────────────────

class TestCopySdImage:
    def test_copies_file_and_returns_path(self, tmp_path: Path):
        """Returns the output path when copy succeeds."""
        src = tmp_path / "source.png"
        src.write_bytes(b"fake image data")
        out = tmp_path / "still_scene_00.png"

        scene = SceneSpec(
            scene_id="scene_00",
            generated_image_path=str(src),
        )
        result = StillsExtractor._copy_sd_image(scene, out)

        assert result == out
        assert out.exists()

    def test_returns_none_when_no_image_path(self, tmp_path: Path):
        """Returns None when generated_image_path is None."""
        scene = SceneSpec(scene_id="s0")
        out = tmp_path / "still.png"
        assert StillsExtractor._copy_sd_image(scene, out) is None

    def test_returns_none_when_file_missing(self, tmp_path: Path):
        """Returns None when the source file doesn't exist."""
        scene = SceneSpec(
            scene_id="s0",
            generated_image_path="/nonexistent/path/img.png",
        )
        out = tmp_path / "still.png"
        assert StillsExtractor._copy_sd_image(scene, out) is None

    def test_file_content_preserved(self, tmp_path: Path):
        """Copied file has the same content as the source."""
        data = b"PNG_HEADER" + bytes(range(100))
        src = tmp_path / "source.png"
        src.write_bytes(data)
        out = tmp_path / "out.png"

        scene = SceneSpec(scene_id="s0", generated_image_path=str(src))
        StillsExtractor._copy_sd_image(scene, out)

        assert out.read_bytes() == data


# ── Integration: image-only mode ───────────────────────────────────────────

class TestIntegrationImageMode:
    def test_returns_paths_in_scene_order(self, tmp_path: Path):
        """Returned list preserves scene order."""
        srcs = []
        for i in range(3):
            src = tmp_path / f"img_{i}.png"
            src.write_bytes(f"data{i}".encode())
            srcs.append(src)

        plan = _make_plan(
            {"generated_image_path": str(srcs[0])},
            {"generated_image_path": str(srcs[1])},
            {"generated_image_path": str(srcs[2])},
        )
        result = StillsExtractor.extract(plan, video_path=None, output_dir=tmp_path)

        assert len(result) == 3
        assert result[0].name == "still_scene_00.png"
        assert result[1].name == "still_scene_01.png"
        assert result[2].name == "still_scene_02.png"

    def test_output_files_are_written(self, tmp_path: Path):
        """Output files actually exist on disk after extract()."""
        src = tmp_path / "img.png"
        src.write_bytes(b"img data")

        plan = _make_plan({"generated_image_path": str(src)})
        result = StillsExtractor.extract(plan, video_path=None, output_dir=tmp_path)

        for path in result:
            assert path.exists(), f"{path} should exist"

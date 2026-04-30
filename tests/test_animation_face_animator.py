"""Tests for vox_romana.animation.face_animator.

Verifies FaceAnimator lazy-loading behaviour, resume semantics in
``animate_for_plan``, and plan-level path population — all without
real torch or SadTalker installed.  Heavy imports are blocked via
``patch.dict("sys.modules", ...)``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.animation.face_animator import (
    FaceAnimationResult,
    FaceAnimator,
)
from vox_romana.animation.models import AnimationPlan, SceneSpec


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_plan(scenes: list[SceneSpec] | None = None) -> AnimationPlan:
    """Minimal AnimationPlan with optional scenes list."""
    return AnimationPlan(
        plan_id="test-plan",
        brief="test brief",
        scenes=scenes or [],
    )


def _make_talking_head_scene(
    scene_id: str = "s1",
    *,
    portrait_path: str | None = None,
    face_animation_path: str | None = None,
    start_time_sec: float | None = 0.0,
    end_time_sec: float | None = 3.0,
) -> SceneSpec:
    """Build a SceneSpec pre-configured for talking_head_sync."""
    scene = SceneSpec(scene_id=scene_id)
    scene.animation_hint = "talking_head_sync"
    scene.generated_image_path = portrait_path
    scene.face_animation_path = face_animation_path
    scene.start_time_sec = start_time_sec
    scene.end_time_sec = end_time_sec
    return scene


# ── Construction / lazy-loading ────────────────────────────────────────────

class TestFaceAnimatorConstruction:
    def test_no_import_at_construct_time(self) -> None:
        """FaceAnimator() must not import torch or sadtalker at construction."""
        with patch.dict("sys.modules", {"torch": None, "sadtalker": None}):
            # If torch/sadtalker were imported here this would raise ImportError.
            animator = FaceAnimator()
        assert animator._loaded is False
        assert animator._backend is None

    def test_default_model_is_sadtalker(self) -> None:
        animator = FaceAnimator()
        assert animator._model_name == "sadtalker"

    def test_custom_model_stored(self) -> None:
        animator = FaceAnimator(model="live_portrait")
        assert animator._model_name == "live_portrait"

    def test_pipeline_none_until_first_call(self) -> None:
        animator = FaceAnimator()
        assert not animator._loaded


# ── _ensure_loaded ImportError paths ──────────────────────────────────────

class TestEnsureLoaded:
    def test_raises_when_torch_missing(self) -> None:
        """Missing torch → ImportError with 'torch' in the message."""
        animator = FaceAnimator()
        with patch.dict("sys.modules", {"torch": None, "sadtalker": None}):
            with pytest.raises(ImportError, match="torch"):
                animator._ensure_loaded()

    def test_raises_when_sadtalker_missing(self) -> None:
        """torch present but sadtalker absent → ImportError with 'sadtalker'."""
        animator = FaceAnimator()
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch, "sadtalker": None}):
            with pytest.raises(ImportError, match="sadtalker"):
                animator._ensure_loaded()

    def test_noop_when_already_loaded(self) -> None:
        """_ensure_loaded() is a no-op when _loaded is True."""
        animator = FaceAnimator()
        animator._loaded = True
        # Blocked modules would raise if imported — proves early return.
        with patch.dict("sys.modules", {"torch": None, "sadtalker": None}):
            animator._ensure_loaded()  # Must not raise.


# ── animate_for_plan ────────────────────────────────────────────────────────

class TestAnimateForPlan:
    def test_skips_existing_output(self, tmp_path: Path) -> None:
        """Resume semantics: face_animation_path already on disk → animate() not called."""
        existing_video = tmp_path / "face_s1.mp4"
        existing_video.touch()

        scene = _make_talking_head_scene(
            scene_id="s1",
            portrait_path=str(tmp_path / "portrait.png"),
            face_animation_path=str(existing_video),
        )
        plan = _make_plan(scenes=[scene])
        animator = FaceAnimator()
        audio = tmp_path / "show.wav"
        audio.touch()

        with patch.object(animator, "animate") as mock_animate:
            animator.animate_for_plan(plan, audio, tmp_path / "faces")

        mock_animate.assert_not_called()

    def test_populates_scene_face_animation_path(self, tmp_path: Path) -> None:
        """animate_for_plan() sets scene.face_animation_path to result.video_path."""
        portrait = tmp_path / "portrait.png"
        portrait.touch()
        audio = tmp_path / "show.wav"
        audio.touch()

        scene = _make_talking_head_scene(
            scene_id="s1",
            portrait_path=str(portrait),
        )
        plan = _make_plan(scenes=[scene])

        faces_dir = tmp_path / "faces"
        expected_out = faces_dir / "face_s1.mp4"
        fake_result = FaceAnimationResult(
            scene_id="s1",
            video_path=expected_out,
            duration_sec=3.0,
        )

        animator = FaceAnimator()
        with (
            patch.object(animator, "animate", return_value=fake_result),
            patch.object(FaceAnimator, "_slice_audio"),
        ):
            animator.animate_for_plan(plan, audio, faces_dir)

        assert scene.face_animation_path == str(expected_out)

    def test_skips_non_talking_head_scenes(self, tmp_path: Path) -> None:
        """Scenes whose animation_hint != 'talking_head_sync' are ignored."""
        scene = SceneSpec(scene_id="s1")  # animation_hint defaults to "static"
        plan = _make_plan(scenes=[scene])
        animator = FaceAnimator()
        audio = tmp_path / "show.wav"
        audio.touch()

        with patch.object(animator, "animate") as mock_animate:
            animator.animate_for_plan(plan, audio, tmp_path / "faces")

        mock_animate.assert_not_called()

    def test_returns_same_plan_object(self, tmp_path: Path) -> None:
        """animate_for_plan() returns the same (mutated) plan."""
        plan = _make_plan()
        animator = FaceAnimator()
        audio = tmp_path / "show.wav"
        audio.touch()

        result = animator.animate_for_plan(plan, audio, tmp_path / "faces")

        assert result is plan

    def test_skips_scene_missing_portrait(self, tmp_path: Path) -> None:
        """Scene with missing generated_image_path is silently skipped."""
        scene = _make_talking_head_scene(
            scene_id="s1",
            portrait_path=str(tmp_path / "does_not_exist.png"),  # not created
        )
        plan = _make_plan(scenes=[scene])
        animator = FaceAnimator()
        audio = tmp_path / "show.wav"
        audio.touch()

        with patch.object(animator, "animate") as mock_animate:
            animator.animate_for_plan(plan, audio, tmp_path / "faces")

        mock_animate.assert_not_called()

    def test_skips_scene_missing_timestamps(self, tmp_path: Path) -> None:
        """Scene without start/end times is silently skipped."""
        portrait = tmp_path / "portrait.png"
        portrait.touch()
        scene = _make_talking_head_scene(
            scene_id="s1",
            portrait_path=str(portrait),
            start_time_sec=None,
            end_time_sec=None,
        )
        plan = _make_plan(scenes=[scene])
        animator = FaceAnimator()
        audio = tmp_path / "show.wav"
        audio.touch()

        with patch.object(animator, "animate") as mock_animate:
            animator.animate_for_plan(plan, audio, tmp_path / "faces")

        mock_animate.assert_not_called()

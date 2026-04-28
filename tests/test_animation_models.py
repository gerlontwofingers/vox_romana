"""Tests for vox_romana.animation.models — dataclass structure and behavior."""

from __future__ import annotations

import pytest

from vox_romana.animation.models import (
    AnimationConfig,
    AnimationPlan,
    HardwareProfile,
    SceneSpec,
)


# ══════════════════════════════════════════════════════════════════════════
# SceneSpec
# ══════════════════════════════════════════════════════════════════════════


class TestSceneSpec:
    def test_minimal_construction(self):
        scene = SceneSpec(scene_id="scene_01")
        assert scene.scene_id == "scene_01"
        assert scene.event_global_orders == []
        assert scene.start_time_sec is None
        assert scene.end_time_sec is None
        assert scene.image_prompt == ""
        assert scene.negative_prompt is None
        assert scene.text_overlay is None
        assert scene.speaker_label is None
        assert scene.transition_in == "cut"
        assert scene.transition_out == "cut"
        assert scene.animation_hint == "static"
        assert scene.generated_image_path is None

    def test_full_construction(self):
        scene = SceneSpec(
            scene_id="scene_02",
            event_global_orders=[0, 1, 2],
            start_time_sec=0.0,
            end_time_sec=12.5,
            image_prompt="dramatic noir hallway, film grain",
            negative_prompt="blurry, oversaturated",
            text_overlay="The economy is at a crossroads.",
            speaker_label="Dr. Elena Vasquez — Economist",
            transition_in="fade",
            transition_out="crossfade",
            animation_hint="ken_burns_zoom_in",
            generated_image_path="/tmp/scene_02.png",
        )
        assert scene.event_global_orders == [0, 1, 2]
        assert scene.start_time_sec == 0.0
        assert scene.end_time_sec == 12.5
        assert scene.image_prompt == "dramatic noir hallway, film grain"
        assert scene.transition_in == "fade"
        assert scene.animation_hint == "ken_burns_zoom_in"

    def test_duration_sec_computed(self):
        scene = SceneSpec(
            scene_id="s",
            start_time_sec=5.0,
            end_time_sec=17.5,
        )
        assert scene.duration_sec == pytest.approx(12.5)

    def test_duration_sec_none_when_not_synced(self):
        scene = SceneSpec(scene_id="s")
        assert scene.duration_sec is None

    def test_duration_sec_none_when_partial(self):
        scene = SceneSpec(scene_id="s", start_time_sec=0.0)
        assert scene.duration_sec is None

    def test_mutable(self):
        scene = SceneSpec(scene_id="s")
        scene.generated_image_path = "/tmp/out.png"
        scene.start_time_sec = 0.0
        assert scene.generated_image_path == "/tmp/out.png"
        assert scene.start_time_sec == 0.0

    def test_event_global_orders_independent(self):
        """Each SceneSpec has its own list, not a shared default."""
        s1 = SceneSpec(scene_id="s1")
        s2 = SceneSpec(scene_id="s2")
        s1.event_global_orders.append(99)
        assert s2.event_global_orders == []

    def test_scene_spec_face_animation_path_default_none(self):
        """face_animation_path defaults to None on a fresh SceneSpec."""
        scene = SceneSpec(scene_id="x")
        assert scene.face_animation_path is None


# ══════════════════════════════════════════════════════════════════════════
# HardwareProfile
# ══════════════════════════════════════════════════════════════════════════


class TestHardwareProfile:
    def test_defaults(self):
        p = HardwareProfile()
        assert p.device == "cpu"
        assert p.vram_gb is None
        assert p.total_ram_gb is None
        assert p.recommended_sd_model == "runwayml/stable-diffusion-v1-5"
        assert p.recommended_resolution == (512, 512)
        assert p.recommended_fps == 24
        assert p.low_disk_warning is False
        assert p.free_disk_gb == 0.0

    def test_mps_high_tier(self):
        p = HardwareProfile(
            device="mps",
            vram_gb=32.0,
            total_ram_gb=32.0,
            recommended_sd_model="stabilityai/stable-diffusion-xl-base-1.0",
            recommended_resolution=(1920, 1080),
            recommended_fps=30,
            free_disk_gb=37.0,
        )
        assert p.device == "mps"
        assert p.recommended_resolution == (1920, 1080)
        assert p.recommended_fps == 30
        assert "xl" in p.recommended_sd_model

    def test_low_disk_warning_flag(self):
        p = HardwareProfile(low_disk_warning=True, free_disk_gb=5.0)
        assert p.low_disk_warning is True
        assert p.free_disk_gb == 5.0

    def test_mutable(self):
        p = HardwareProfile()
        p.device = "mps"
        p.vram_gb = 32.0
        assert p.device == "mps"


# ══════════════════════════════════════════════════════════════════════════
# AnimationConfig
# ══════════════════════════════════════════════════════════════════════════


class TestAnimationConfig:
    def test_defaults(self):
        cfg = AnimationConfig()
        assert cfg.enabled is False
        assert cfg.output_format == "mp4"
        assert cfg.generate_stills is False
        assert cfg.stills_count is None
        assert cfg.resolution == (1920, 1080)
        assert cfg.fps == 24
        assert cfg.animation_style_hint is None
        assert cfg.image_model is None
        assert cfg.render_backend is None
        assert cfg.visual_brief is None

    def test_enabled_with_overrides(self):
        cfg = AnimationConfig(
            enabled=True,
            output_format="webm",
            generate_stills=True,
            stills_count=5,
            resolution=(1280, 720),
            fps=30,
            animation_style_hint="BBC documentary",
            image_model="stabilityai/stable-diffusion-xl-base-1.0",
            render_backend="moviepy",
            visual_brief="use split-screen for interruptions",
        )
        assert cfg.enabled is True
        assert cfg.output_format == "webm"
        assert cfg.generate_stills is True
        assert cfg.stills_count == 5
        assert cfg.resolution == (1280, 720)
        assert cfg.fps == 30
        assert cfg.animation_style_hint == "BBC documentary"
        assert cfg.render_backend == "moviepy"
        assert cfg.visual_brief == "use split-screen for interruptions"

    def test_disabled_by_default(self):
        """AnimationConfig is a no-op until enabled=True."""
        cfg = AnimationConfig()
        assert not cfg.enabled

    def test_mutable(self):
        cfg = AnimationConfig()
        cfg.enabled = True
        cfg.visual_brief = "noir cinematic"
        assert cfg.enabled is True
        assert cfg.visual_brief == "noir cinematic"


# ══════════════════════════════════════════════════════════════════════════
# AnimationPlan
# ══════════════════════════════════════════════════════════════════════════


class TestAnimationPlan:
    def _make_plan(self, n_scenes: int = 0) -> AnimationPlan:
        scenes = [SceneSpec(scene_id=f"scene_{i:02d}") for i in range(n_scenes)]
        return AnimationPlan(
            plan_id="test-plan-001",
            brief="Two economists debate UBI",
            scenes=scenes,
        )

    def test_minimal_construction(self):
        plan = AnimationPlan(plan_id="p1", brief="test brief")
        assert plan.plan_id == "p1"
        assert plan.brief == "test brief"
        assert plan.visual_instructions is None
        assert plan.style == "cinematic"
        assert plan.render_backend == "moviepy"
        assert plan.fps == 24
        assert plan.resolution == (1920, 1080)
        assert plan.image_model is None
        assert plan.scenes == []
        assert plan.created_at == ""
        assert isinstance(plan.color_palette, list)
        assert len(plan.color_palette) == 3
        assert isinstance(plan.hardware_profile, HardwareProfile)

    def test_total_scenes(self):
        plan = self._make_plan(n_scenes=4)
        assert plan.total_scenes == 4

    def test_total_scenes_empty(self):
        plan = self._make_plan(n_scenes=0)
        assert plan.total_scenes == 0

    def test_is_synced_true(self):
        plan = self._make_plan(n_scenes=2)
        for i, scene in enumerate(plan.scenes):
            scene.start_time_sec = float(i * 10)
            scene.end_time_sec = float(i * 10 + 10)
        assert plan.is_synced is True

    def test_is_synced_false_when_empty(self):
        plan = self._make_plan(n_scenes=0)
        # vacuously True (all() over empty)
        assert plan.is_synced is True

    def test_is_synced_false_partial(self):
        plan = self._make_plan(n_scenes=2)
        plan.scenes[0].start_time_sec = 0.0
        plan.scenes[0].end_time_sec = 10.0
        # scenes[1] has no timestamps
        assert plan.is_synced is False

    def test_is_rendered_false_by_default(self):
        plan = self._make_plan(n_scenes=2)
        assert plan.is_rendered is False

    def test_is_rendered_true(self):
        plan = self._make_plan(n_scenes=2)
        for scene in plan.scenes:
            scene.generated_image_path = "/tmp/out.png"
        assert plan.is_rendered is True

    def test_scenes_list_independent(self):
        """Each AnimationPlan has its own scenes list."""
        p1 = AnimationPlan(plan_id="p1", brief="b")
        p2 = AnimationPlan(plan_id="p2", brief="b")
        p1.scenes.append(SceneSpec(scene_id="s"))
        assert len(p2.scenes) == 0

    def test_color_palette_independent(self):
        """Each AnimationPlan has its own color_palette list."""
        p1 = AnimationPlan(plan_id="p1", brief="b")
        p2 = AnimationPlan(plan_id="p2", brief="b")
        p1.color_palette.append("#ffffff")
        assert len(p2.color_palette) == 3

    def test_mutable(self):
        plan = AnimationPlan(plan_id="p", brief="b")
        plan.style = "documentary"
        plan.render_backend = "moviepy"
        assert plan.style == "documentary"

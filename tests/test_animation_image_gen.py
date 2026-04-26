"""Tests for vox_romana.animation.image_gen.

Verifies ImageGenerator and _APIImageGenerator without real diffusers,
torch, or PIL installed. All HuggingFace/PyTorch imports are patched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.animation.image_gen import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_NEGATIVE_PROMPT,
    ImageGenerator,
    _APIImageGenerator,
    _select_device,
)
from vox_romana.animation.models import AnimationPlan, HardwareProfile, SceneSpec


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_profile() -> HardwareProfile:
    return HardwareProfile(
        device="cpu",
        vram_gb=None,
        total_ram_gb=16.0,
        recommended_sd_model="runwayml/stable-diffusion-v1-5",
        recommended_resolution=(512, 512),
        recommended_fps=24,
        low_disk_warning=False,
        free_disk_gb=100.0,
    )


def _make_plan(scenes: list[SceneSpec] | None = None) -> AnimationPlan:
    return AnimationPlan(
        plan_id="test-plan",
        brief="test brief",
        visual_instructions=None,
        style="documentary",
        render_backend="moviepy",
        color_palette=["#111", "#222"],
        font_family=None,
        fps=24,
        resolution=(512, 512),
        image_model="runwayml/stable-diffusion-v1-5",
        scenes=scenes or [],
        hardware_profile=_make_profile(),
        created_at="2026-01-01T00:00:00+00:00",
    )


def _make_scene(
    scene_id: str = "s1",
    image_prompt: str = "a test scene",
    negative_prompt: str | None = None,
    generated_image_path: str | None = None,
) -> SceneSpec:
    s = SceneSpec(scene_id=scene_id, image_prompt=image_prompt)
    s.negative_prompt = negative_prompt
    s.generated_image_path = generated_image_path
    return s


def _make_mock_pipeline(image: MagicMock | None = None) -> MagicMock:
    """Build a mock diffusers pipeline that returns a PIL Image mock."""
    pil_image = image or MagicMock()
    pipeline = MagicMock()
    pipeline.return_value.images = [pil_image]
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# _select_device
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectDevice:
    def test_explicit_device_returned_as_is(self) -> None:
        assert _select_device("cuda") == "cuda"
        assert _select_device("cpu") == "cpu"
        assert _select_device("mps") == "mps"

    def test_auto_selects_mps_when_available(self) -> None:
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _select_device(None) == "mps"

    def test_auto_selects_cuda_when_no_mps(self) -> None:
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _select_device(None) == "cuda"

    def test_cpu_fallback_when_no_gpu(self) -> None:
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _select_device(None) == "cpu"

    def test_cpu_fallback_when_torch_missing(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            result = _select_device(None)
        assert result == "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# ImageGenerator — construction
# ─────────────────────────────────────────────────────────────────────────────

class TestImageGeneratorConstruction:
    def test_default_steps_and_guidance(self) -> None:
        gen = ImageGenerator(device="cpu")
        assert gen._steps == DEFAULT_INFERENCE_STEPS
        assert gen._guidance_scale == DEFAULT_GUIDANCE_SCALE

    def test_custom_steps_and_guidance(self) -> None:
        gen = ImageGenerator(device="cpu", steps=10, guidance_scale=5.0)
        assert gen._steps == 10
        assert gen._guidance_scale == 5.0

    def test_pipeline_none_until_first_call(self) -> None:
        gen = ImageGenerator(device="cpu")
        assert gen._pipeline is None

    def test_explicit_device_stored(self) -> None:
        gen = ImageGenerator(device="cpu")
        assert gen._device == "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# ImageGenerator — _ensure_loaded: ImportError paths
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureLoaded:
    def test_raises_import_error_when_torch_missing(self) -> None:
        gen = ImageGenerator(device="cpu")
        with patch.dict("sys.modules", {"torch": None, "diffusers": None}):
            with pytest.raises(ImportError, match="torch"):
                gen._ensure_loaded("runwayml/stable-diffusion-v1-5")

    def test_raises_import_error_when_diffusers_missing(self) -> None:
        gen = ImageGenerator(device="cpu")
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch, "diffusers": None}):
            with pytest.raises(ImportError, match="diffusers"):
                gen._ensure_loaded("runwayml/stable-diffusion-v1-5")

    def test_noop_when_pipeline_already_loaded(self) -> None:
        gen = ImageGenerator(device="cpu")
        gen._pipeline = MagicMock()  # Pre-load a fake pipeline.
        # _ensure_loaded should return immediately without touching sys.modules.
        gen._ensure_loaded("any-model")  # Should not raise.

    def test_loads_pipeline_and_stores_it(self) -> None:
        gen = ImageGenerator(device="cpu")

        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.enable_attention_slicing = MagicMock()
        mock_diffusion_pipeline_cls = MagicMock(return_value=mock_pipeline_instance)
        mock_pipeline_instance.to.return_value = mock_pipeline_instance

        mock_diffusers = MagicMock()
        mock_diffusers.DiffusionPipeline = mock_diffusion_pipeline_cls
        mock_diffusers.StableDiffusionPipeline = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch, "diffusers": mock_diffusers}):
            gen._ensure_loaded("runwayml/stable-diffusion-v1-5")

        assert gen._pipeline is not None

    def test_attention_slicing_enabled_on_cpu(self) -> None:
        gen = ImageGenerator(device="cpu")

        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"

        # Code calls DiffusionPipeline.from_pretrained(...), not DiffusionPipeline(...)
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.to.return_value = mock_pipeline_instance
        mock_diffusion_cls = MagicMock()
        mock_diffusion_cls.from_pretrained.return_value = mock_pipeline_instance

        mock_diffusers = MagicMock()
        mock_diffusers.DiffusionPipeline = mock_diffusion_cls

        with patch.dict("sys.modules", {"torch": mock_torch, "diffusers": mock_diffusers}):
            gen._ensure_loaded("runwayml/stable-diffusion-v1-5")

        mock_pipeline_instance.enable_attention_slicing.assert_called_once()

    def test_falls_back_to_stable_diffusion_pipeline_on_failure(self) -> None:
        """If DiffusionPipeline.from_pretrained raises, tries StableDiffusionPipeline."""
        gen = ImageGenerator(device="cpu")

        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"

        mock_sd_instance = MagicMock()
        mock_sd_instance.to.return_value = mock_sd_instance

        mock_diffusers = MagicMock()
        mock_diffusers.DiffusionPipeline.from_pretrained.side_effect = RuntimeError("no sdxl")
        mock_diffusers.StableDiffusionPipeline.from_pretrained.return_value = mock_sd_instance

        with patch.dict("sys.modules", {"torch": mock_torch, "diffusers": mock_diffusers}):
            gen._ensure_loaded("runwayml/stable-diffusion-v1-5")

        assert gen._pipeline is not None


# ─────────────────────────────────────────────────────────────────────────────
# ImageGenerator — generate()
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerate:
    def _make_gen_with_mock_pipeline(self) -> tuple[ImageGenerator, MagicMock]:
        gen = ImageGenerator(device="cpu", steps=5, guidance_scale=7.5)
        mock_image = MagicMock()
        gen._pipeline = _make_mock_pipeline(mock_image)
        return gen, mock_image

    def test_returns_pil_image(self) -> None:
        gen, expected_image = self._make_gen_with_mock_pipeline()
        result = gen.generate("a test prompt", width=512, height=512)
        assert result is expected_image

    def test_passes_prompt_and_negative_prompt(self) -> None:
        gen, _ = self._make_gen_with_mock_pipeline()
        gen.generate(
            "sunny meadow",
            negative_prompt="blurry",
            width=512, height=512,
        )
        call_kwargs = gen._pipeline.call_args.kwargs
        assert call_kwargs["prompt"] == "sunny meadow"
        assert call_kwargs["negative_prompt"] == "blurry"

    def test_uses_default_negative_prompt_when_none(self) -> None:
        gen, _ = self._make_gen_with_mock_pipeline()
        gen.generate("sunny meadow", width=512, height=512)
        call_kwargs = gen._pipeline.call_args.kwargs
        assert call_kwargs["negative_prompt"] == DEFAULT_NEGATIVE_PROMPT

    def test_passes_width_and_height(self) -> None:
        gen, _ = self._make_gen_with_mock_pipeline()
        gen.generate("a prompt", width=800, height=600)
        call_kwargs = gen._pipeline.call_args.kwargs
        assert call_kwargs["width"] == 800
        assert call_kwargs["height"] == 600

    def test_passes_steps_and_guidance(self) -> None:
        gen, _ = self._make_gen_with_mock_pipeline()
        gen.generate("a prompt", width=512, height=512)
        call_kwargs = gen._pipeline.call_args.kwargs
        assert call_kwargs["num_inference_steps"] == 5
        assert call_kwargs["guidance_scale"] == 7.5

    def test_calls_ensure_loaded_before_generation(self) -> None:
        gen = ImageGenerator(device="cpu")
        gen._pipeline = _make_mock_pipeline()
        # _ensure_loaded is a no-op when pipeline is already set.
        # Verify it doesn't crash and pipeline is still called.
        gen.generate("test", width=64, height=64)
        gen._pipeline.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# ImageGenerator — generate_for_plan()
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateForPlan:
    def _make_gen(self) -> ImageGenerator:
        gen = ImageGenerator(device="cpu")
        mock_image = MagicMock()
        gen._pipeline = _make_mock_pipeline(mock_image)
        return gen

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "frames"
        plan = _make_plan([_make_scene("s1")])
        gen = self._make_gen()
        gen.generate_for_plan(plan, output_dir)
        assert output_dir.exists()

    def test_saves_one_png_per_scene(self, tmp_path: Path) -> None:
        scenes = [_make_scene("s1"), _make_scene("s2"), _make_scene("s3")]
        plan = _make_plan(scenes)
        gen = self._make_gen()
        # PIL Image.save() is mocked — check it's called per scene.
        gen.generate_for_plan(plan, tmp_path)
        # 3 save calls (one per scene).
        save_calls = gen._pipeline.return_value.images[0].save.call_count
        assert save_calls == 3

    def test_sets_generated_image_path_on_each_scene(self, tmp_path: Path) -> None:
        scenes = [_make_scene("scene_01"), _make_scene("scene_02")]
        plan = _make_plan(scenes)
        gen = self._make_gen()
        gen.generate_for_plan(plan, tmp_path)
        for scene in plan.scenes:
            assert scene.generated_image_path is not None
            assert "scene_" in scene.generated_image_path

    def test_path_contains_scene_id(self, tmp_path: Path) -> None:
        scene = _make_scene("my_special_scene")
        plan = _make_plan([scene])
        gen = self._make_gen()
        gen.generate_for_plan(plan, tmp_path)
        assert "my_special_scene" in scene.generated_image_path

    def test_returns_mutated_plan(self, tmp_path: Path) -> None:
        plan = _make_plan([_make_scene("s1")])
        gen = self._make_gen()
        result = gen.generate_for_plan(plan, tmp_path)
        assert result is plan

    def test_skips_already_generated_scenes(self, tmp_path: Path) -> None:
        # Pre-create the file that scene_01 would point to.
        existing = tmp_path / "already_there.png"
        existing.write_bytes(b"fake png data")
        scene_done = _make_scene("s1", generated_image_path=str(existing))
        scene_new = _make_scene("s2")
        plan = _make_plan([scene_done, scene_new])
        gen = self._make_gen()
        gen.generate_for_plan(plan, tmp_path)
        # Only one generate call (for s2, not s1).
        assert gen._pipeline.call_count == 1

    def test_does_not_skip_when_path_set_but_file_missing(
        self, tmp_path: Path,
    ) -> None:
        scene = _make_scene("s1", generated_image_path="/nonexistent/path.png")
        plan = _make_plan([scene])
        gen = self._make_gen()
        gen.generate_for_plan(plan, tmp_path)
        assert gen._pipeline.call_count == 1

    def test_uses_plan_resolution_for_all_scenes(self, tmp_path: Path) -> None:
        plan = _make_plan([_make_scene("s1"), _make_scene("s2")])
        plan.resolution = (800, 600)
        gen = self._make_gen()
        gen.generate_for_plan(plan, tmp_path)
        for call in gen._pipeline.call_args_list:
            assert call.kwargs["width"] == 800
            assert call.kwargs["height"] == 600

    def test_uses_plan_image_model(self, tmp_path: Path) -> None:
        plan = _make_plan([_make_scene("s1")])
        plan.image_model = "custom/model-id"
        gen = self._make_gen()
        # Verify _ensure_loaded is called with the plan's model_id.
        with patch.object(gen, "_ensure_loaded") as mock_ensure:
            gen.generate_for_plan(plan, tmp_path)
        mock_ensure.assert_called_with("custom/model-id")

    def test_falls_back_to_default_model_when_plan_image_model_is_none(
        self, tmp_path: Path,
    ) -> None:
        plan = _make_plan([_make_scene("s1")])
        plan.image_model = None
        gen = self._make_gen()
        with patch.object(gen, "_ensure_loaded") as mock_ensure:
            gen.generate_for_plan(plan, tmp_path)
        model_used = mock_ensure.call_args[0][0]
        assert "stable-diffusion" in model_used.lower()

    def test_uses_scene_negative_prompt(self, tmp_path: Path) -> None:
        scene = _make_scene("s1", negative_prompt="no cars, no text")
        plan = _make_plan([scene])
        gen = self._make_gen()
        gen.generate_for_plan(plan, tmp_path)
        call_kwargs = gen._pipeline.call_args.kwargs
        assert call_kwargs["negative_prompt"] == "no cars, no text"

    def test_empty_scenes_list_is_a_noop(self, tmp_path: Path) -> None:
        plan = _make_plan([])
        gen = self._make_gen()
        gen.generate_for_plan(plan, tmp_path)
        gen._pipeline.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# _APIImageGenerator — stub contract
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIImageGeneratorStub:
    def test_generate_raises_not_implemented(self) -> None:
        stub = _APIImageGenerator()
        with pytest.raises(NotImplementedError, match="No API key"):
            stub.generate("a prompt")

    def test_generate_for_plan_raises_not_implemented(self, tmp_path: Path) -> None:
        stub = _APIImageGenerator()
        plan = _make_plan()
        with pytest.raises(NotImplementedError, match="No API key"):
            stub.generate_for_plan(plan, tmp_path)

    def test_not_implemented_message_suggests_local(self) -> None:
        stub = _APIImageGenerator()
        with pytest.raises(NotImplementedError) as exc_info:
            stub.generate("test")
        assert "ImageGenerator" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_default_negative_prompt_is_string(self) -> None:
        assert isinstance(DEFAULT_NEGATIVE_PROMPT, str)
        assert len(DEFAULT_NEGATIVE_PROMPT) > 10

    def test_default_steps_positive(self) -> None:
        assert DEFAULT_INFERENCE_STEPS > 0

    def test_default_guidance_positive(self) -> None:
        assert DEFAULT_GUIDANCE_SCALE > 0

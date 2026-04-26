"""ImageGenerator — local Stable Diffusion image generation for animation scenes.

This module owns all image synthesis for the animation pipeline. Given an
AnimationPlan (produced by the AnimationDirector), it generates one image
per SceneSpec, writes them to disk, and fills in ``generated_image_path``
on each scene — mutating the plan in place so the renderer can pick them up.

LAZY LOADING:

    The diffusers pipeline is heavy (~2-6 GB download on first use, then
    from cache). Like MusicGenEngine, we load on the FIRST ``generate()``
    call, not at import time. Import-time is instant; tests that mock this
    class don't touch PyTorch or diffusers at all.

LOCAL-FIRST DESIGN:

    Production path: local Stable Diffusion via HuggingFace ``diffusers``.
    The pipeline auto-selects the SD model from the AnimationPlan's
    ``image_model`` field (set by AnimationBootstrap from HardwareProfile).

    API stub: ``_APIImageGenerator`` — raises ``NotImplementedError`` with
    a clear message. Wired in when future API support is needed.

DEVICE SELECTION:

    Mirrors MusicGenEngine's priority: MPS > CUDA > CPU.
    On the primary development machine (Apple M1 Max, 32 GB unified memory),
    MPS gives the best performance for SD inference.

    Attention slicing is enabled automatically on MPS and CPU to reduce
    peak memory pressure during inference — important on machines without
    dedicated VRAM.

PLAN MUTATION (generate_for_plan):

    The AnimationPlan is a mutable dataclass graph. ``generate_for_plan()``
    fills in ``scene.generated_image_path`` for every SceneSpec in the plan.
    Scenes that already have a path are skipped (resume semantics — safe
    to call again after a partial run or interruption).

    The output images are PNG files named ``scene_<scene_id>.png`` inside
    the caller-supplied ``output_dir``.

EXAMPLE:

    from pathlib import Path
    from vox_romana.animation.image_gen import ImageGenerator

    gen = ImageGenerator()  # lazy; no model loaded yet

    plan = animation_director.run(brief, events, hardware_profile=profile)

    # This is where the model loads (first call).
    gen.generate_for_plan(plan, output_dir=Path("out/frames"))

    # Inspect results
    for scene in plan.scenes:
        print(scene.scene_id, scene.generated_image_path)

REQUIREMENTS (OPTIONAL):

    diffusers>=0.30,<0.32
    Pillow>=10.4,<11.0
    torch (already in requirements.txt via TTS pipeline)

    These are listed in requirements.txt under the optional animation section.
    If diffusers is not installed, ``ImageGenerator`` raises ``ImportError``
    with a clear install message when ``generate()`` is first called.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .models import AnimationPlan

if TYPE_CHECKING:  # pragma: no cover
    # Only imported for type checking — not at runtime, so tests don't need
    # the real diffusers / PIL packages installed.
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

# Default negative prompt applied when a SceneSpec has negative_prompt=None.
# Keeps output clean without requiring every scene to repeat itself.
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, watermark, text, low quality, distorted, oversaturated, "
    "cartoon, anime, 3d render"
)

# Number of inference steps. More steps = higher quality, slower generation.
# 25 is a good balance for editorial stills at 1080p.
# Users can override via the ``steps`` constructor arg.
DEFAULT_INFERENCE_STEPS = 25

# Guidance scale (classifier-free guidance). 7.5 is the SD default.
DEFAULT_GUIDANCE_SCALE = 7.5


# ─────────────────────────────────────────────────────────────────────────────
# Device selection (mirrors MusicGenEngine._select_device)
# ─────────────────────────────────────────────────────────────────────────────

def _select_device(device: str | None) -> str:
    """Auto-select the best available compute device.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU.

    Args:
        device: Explicit device string, or None to auto-select.

    Returns:
        A torch device string: "mps", "cuda", or "cpu".
    """
    if device is not None:
        return device

    try:
        import torch  # noqa: PLC0415
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover
        pass

    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# ImageGenerator — local Stable Diffusion, lazy-loaded
# ─────────────────────────────────────────────────────────────────────────────

class ImageGenerator:
    """Generates still images from text prompts using local Stable Diffusion.

    Uses HuggingFace ``diffusers`` with lazy model loading — the pipeline
    is not loaded until the first ``generate()`` call.

    Attributes:
        _device:          Compute device ("mps", "cuda", "cpu").
        _steps:           SD inference steps per image.
        _guidance_scale:  Classifier-free guidance scale.
        _pipeline:        The loaded diffusers pipeline (None until first use).
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        steps: int = DEFAULT_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    ) -> None:
        """Initialize the image generator.

        The SD pipeline is NOT loaded here — it loads on the first
        ``generate()`` call.

        Args:
            device:
                Compute device. None = auto-select (MPS > CUDA > CPU).
                Pass "cpu" to force CPU (slow, but always works).
            steps:
                Number of SD inference steps per image. More = higher
                quality, slower. 25 is a good default for editorial use.
            guidance_scale:
                Classifier-free guidance scale. Higher values follow the
                prompt more strictly but can over-saturate. 7.5 is SD's
                default.
        """
        self._device = _select_device(device)
        self._steps = steps
        self._guidance_scale = guidance_scale
        self._pipeline = None   # Lazy-loaded on first generate() call.

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self, model_id: str) -> None:
        """Load the diffusers pipeline for ``model_id`` if not already loaded.

        Called at the start of every ``generate()`` call. If the pipeline
        is already loaded for the same model, this is a no-op. Switching
        models mid-run is not supported — pass the model you want at
        construction time via the ``model_id`` to ``generate_for_plan()``.

        Args:
            model_id: HuggingFace model identifier (e.g.
                "stabilityai/stable-diffusion-xl-base-1.0").

        Raises:
            ImportError: If ``diffusers`` or ``torch`` is not installed,
                with a clear message listing the install command.
        """
        if self._pipeline is not None:
            return

        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "torch is required for local image generation. "
                "Install with: pip install torch"
            ) from exc

        try:
            from diffusers import (  # noqa: PLC0415
                DiffusionPipeline,
                StableDiffusionPipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "diffusers is required for local image generation. "
                "Install with: pip install 'diffusers>=0.30,<0.32' Pillow"
            ) from exc

        logger.info(
            "ImageGenerator: loading %s on %s (first use — may download model weights)",
            model_id, self._device,
        )

        # SDXL models expose DiffusionPipeline; SD 1.x/2.x expose
        # StableDiffusionPipeline. We try DiffusionPipeline first (it works
        # for both), which is the recommended API for SDXL.
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
                use_safetensors=True,
            )
        except Exception:
            # Fallback to the older StableDiffusionPipeline for SD 1.x/2.x
            # models that don't support DiffusionPipeline.
            logger.debug(
                "ImageGenerator: DiffusionPipeline failed; trying StableDiffusionPipeline",
            )
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
            )

        pipeline = pipeline.to(self._device)

        # Reduce peak memory pressure — important on MPS (unified memory)
        # and CPU where there's no dedicated VRAM budget.
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        self._pipeline = pipeline
        logger.info("ImageGenerator: pipeline ready")

    # ── Core generation ───────────────────────────────────────────────────

    def generate(
        self,
        image_prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int = 1920,
        height: int = 1080,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    ) -> "PILImage":
        """Generate one image from a text prompt.

        Args:
            image_prompt:
                The SD image prompt. Best results with specific, evocative
                prompts: subject + setting + lighting + mood + style.
            negative_prompt:
                Things to exclude from the image. If None, uses
                ``DEFAULT_NEGATIVE_PROMPT``.
            width:
                Output image width in pixels. Should match the plan's
                resolution[0].
            height:
                Output image height in pixels. Should match the plan's
                resolution[1].
            model_id:
                HuggingFace SD model identifier. Defaults to SDXL.
                Change to match the ``AnimationPlan.image_model``.

        Returns:
            A PIL Image object. Call ``.save(path)`` or pass to
            ``generate_for_plan()`` which handles saving automatically.

        Raises:
            ImportError: If ``diffusers`` or ``torch`` is not installed.
        """
        self._ensure_loaded(model_id)

        neg = negative_prompt or DEFAULT_NEGATIVE_PROMPT

        logger.debug(
            "ImageGenerator: generating %dx%d — %r",
            width, height, image_prompt[:60],
        )

        result = self._pipeline(  # type: ignore[misc]
            prompt=image_prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_inference_steps=self._steps,
            guidance_scale=self._guidance_scale,
        )

        return result.images[0]

    # ── Plan-level generation ─────────────────────────────────────────────

    def generate_for_plan(
        self,
        plan: AnimationPlan,
        output_dir: Path,
    ) -> AnimationPlan:
        """Generate images for every scene in the plan.

        Iterates over ``plan.scenes``, calls ``generate()`` for each,
        saves to ``<output_dir>/scene_<scene_id>.png``, and fills in
        ``scene.generated_image_path``.

        RESUME SEMANTICS:
            Scenes that already have a ``generated_image_path`` pointing
            to an existing file are skipped. Safe to call again after a
            partial run.

        Args:
            plan:
                The AnimationPlan from AnimationDirector. Mutated in place:
                ``generated_image_path`` is set on each SceneSpec.
            output_dir:
                Directory to write PNG files into. Created if it doesn't
                exist.

        Returns:
            The same (mutated) AnimationPlan, for convenient chaining:
            ``plan = gen.generate_for_plan(plan, out_dir)``.

        Raises:
            ImportError: If ``diffusers`` or ``torch`` is not installed.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        width, height = plan.resolution
        model_id = plan.image_model or "stabilityai/stable-diffusion-xl-base-1.0"
        total = len(plan.scenes)

        for idx, scene in enumerate(plan.scenes):
            out_path = output_dir / f"scene_{scene.scene_id}.png"

            # Resume semantics: skip if already generated.
            if scene.generated_image_path and Path(scene.generated_image_path).exists():
                logger.debug(
                    "ImageGenerator: scene %s already generated — skipping (%d/%d)",
                    scene.scene_id, idx + 1, total,
                )
                continue

            logger.info(
                "ImageGenerator: generating scene %s (%d/%d)",
                scene.scene_id, idx + 1, total,
            )

            image = self.generate(
                scene.image_prompt,
                negative_prompt=scene.negative_prompt,
                width=width,
                height=height,
                model_id=model_id,
            )

            image.save(str(out_path))
            scene.generated_image_path = str(out_path)

            logger.debug(
                "ImageGenerator: saved %s", out_path,
            )

        return plan


# ─────────────────────────────────────────────────────────────────────────────
# _APIImageGenerator — stub for future API-based generation
# ─────────────────────────────────────────────────────────────────────────────

class _APIImageGenerator:
    """Stub for future remote API image generation.

    Not yet functional — raises ``NotImplementedError`` immediately.
    Wired in for future integration with DALL-E, Stability AI API,
    Replicate, or similar services.

    When API support is ready:
        - Add API key handling (env var + constructor arg, same pattern
          as DeepSeekClient)
        - Implement ``generate()`` with httpx/requests
        - Return a PIL Image (download bytes, PIL.Image.open(BytesIO(...)))
    """

    def generate(
        self,
        image_prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int = 1920,
        height: int = 1080,
        model_id: str | None = None,
    ) -> "PILImage":
        raise NotImplementedError(
            "No API key configured — use local generation (ImageGenerator) instead. "
            "Remote image API support is planned for a future release."
        )

    def generate_for_plan(
        self,
        plan: AnimationPlan,
        output_dir: Path,
    ) -> AnimationPlan:
        raise NotImplementedError(
            "No API key configured — use local generation (ImageGenerator) instead. "
            "Remote image API support is planned for a future release."
        )

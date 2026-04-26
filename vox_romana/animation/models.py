"""Data models for the vox_romana animation module.

These are the core data structures that flow through the animation pipeline:

    AnimationConfig  — user-facing configuration (what the user asks for)
    HardwareProfile  — hardware capabilities (what the machine can do)
    AnimationPlan    — LLM director's creative decisions (the visual script)
    SceneSpec        — one scene's worth of visual instructions

MUTABILITY:

    All four classes are MUTABLE dataclasses — the same convention as
    ContentConfig. The pipeline fills them in progressively:
        1. HardwareScanner fills HardwareProfile
        2. AnimationBootstrap produces AnimationConfig
        3. AnimationDirector produces AnimationPlan (with SceneSpecs)
        4. ImageGenerator fills SceneSpec.generated_image_path
        5. TimelineSync fills SceneSpec.start_time_sec / end_time_sec

USAGE:

    from vox_romana.animation.models import (
        AnimationConfig,
        HardwareProfile,
        AnimationPlan,
        SceneSpec,
    )

    config = AnimationConfig(
        enabled=True,
        visual_brief="noir aesthetic, high contrast, 1940s film look",
    )

    profile = HardwareProfile(
        device="mps",
        vram_gb=32.0,
        recommended_sd_model="stabilityai/stable-diffusion-xl-base-1.0",
        recommended_resolution=(1920, 1080),
        recommended_fps=30,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Valid animation style names ────────────────────────────────────────────
AnimationStyle = Literal[
    "cinematic",      # wide shots, dramatic lighting, film-like
    "documentary",    # talking-head cuts, title cards, factual visuals
    "kinetic_text",   # text-driven, typographic, minimal imagery
    "illustrated",    # painterly / illustrated art style
    "talking_heads",  # speaker cards, name plates, podcast-style
]

# ── Valid render backend names ─────────────────────────────────────────────
RenderBackend = Literal["moviepy", "manim", "ffmpeg"]

# ── Valid animation hint names (per-scene motion type) ─────────────────────
AnimationHint = Literal[
    "ken_burns_zoom_in",   # slow zoom toward center
    "ken_burns_zoom_out",  # slow zoom out from center
    "ken_burns_pan_left",  # slow pan left
    "ken_burns_pan_right", # slow pan right
    "parallax",            # subtle depth-shift on multi-layer composite
    "static",              # no motion — still frame
    "text_pulse",          # text overlay pulses / fades in rhythm
]

# ── Valid scene transition names ────────────────────────────────────────────
TransitionKind = Literal["cut", "fade", "crossfade", "wipe"]


# ═══════════════════════════════════════════════════════════════════════════
# SceneSpec — one scene in the visual script
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SceneSpec:
    """Visual instructions for a single scene in the animation.

    A "scene" maps to one or more consecutive AudioEvents. The AnimationDirector
    (LLM) decides how to group events into scenes and what each scene looks like.

    Lifecycle:
        1. AnimationDirector creates SceneSpecs with image_prompt and overlays.
        2. ImageGenerator fills ``generated_image_path``.
        3. TimelineSync fills ``start_time_sec`` and ``end_time_sec``.
        4. AnimationRenderer reads all fields to composite the final clip.

    Attributes:
        scene_id:
            Unique identifier within the plan (e.g., "scene_01").

        event_global_orders:
            List of AudioEvent.global_order values this scene covers.
            Used by TimelineSync to compute timestamps from TTS durations.

        start_time_sec:
            Start time in seconds within the final video. Filled by
            TimelineSync after TTS rendering. None until then.

        end_time_sec:
            End time in seconds. Filled by TimelineSync. None until then.

        image_prompt:
            Stable Diffusion generation prompt for this scene's background
            image. Written by the LLM director.
            Example: "dimly lit lecture hall, dramatic spotlight, noir,
            high contrast black and white, cinematic 35mm film grain"

        negative_prompt:
            SD negative prompt (things to avoid).
            Example: "blurry, cartoon, anime, oversaturated, watermark"

        text_overlay:
            Optional caption or subtitle text rendered on top of the scene.
            None = no overlay.

        speaker_label:
            Optional name card (e.g., "Dr. Elena Vasquez — Economist").
            Rendered as a lower-third style card. None = no label.

        transition_in:
            How this scene enters. "cut" is instantaneous; others are
            animated. Default: "cut".

        transition_out:
            How this scene exits. Default: "cut".

        animation_hint:
            Motion type for the background image. Ken Burns effects add
            life to still images. "static" = no motion. Default: "static".

        generated_image_path:
            Filesystem path to the SD-generated image for this scene.
            Filled by ImageGenerator. None until then.
    """

    scene_id: str
    event_global_orders: list[int] = field(default_factory=list)

    # Filled by TimelineSync
    start_time_sec: float | None = None
    end_time_sec: float | None = None

    # Filled by AnimationDirector (LLM)
    image_prompt: str = ""
    negative_prompt: str | None = None
    text_overlay: str | None = None
    speaker_label: str | None = None
    transition_in: TransitionKind = "cut"
    transition_out: TransitionKind = "cut"
    animation_hint: AnimationHint = "static"

    # Filled by ImageGenerator
    generated_image_path: str | None = None

    @property
    def duration_sec(self) -> float | None:
        """Scene duration in seconds, or None if timestamps not yet set."""
        if self.start_time_sec is None or self.end_time_sec is None:
            return None
        return self.end_time_sec - self.start_time_sec


# ═══════════════════════════════════════════════════════════════════════════
# HardwareProfile — what this machine can do
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareProfile:
    """Hardware capabilities and recommended animation settings.

    Produced by HardwareScanner.scan(). The AnimationDirector and
    ImageGenerator read this to make hardware-appropriate choices.

    Tier logic (set by HardwareScanner):
        MPS (unified ≥ 16 GB) or CUDA ≥ 10 GB  → SDXL + 1080p + 30fps
        CUDA 4–10 GB                             → SD 2.1 + 720p + 24fps
        MPS (unified < 16 GB) / CUDA < 4 GB     → SD 1.5 + 512p + 24fps
        CPU only                                 → SD 1.5 + 480p + 24fps (warn)

    devstation_01 (M1 Max, 32 GB unified, MPS): SDXL + 1920×1080 + 30fps

    Attributes:
        device:
            PyTorch device string: "mps", "cuda", or "cpu".

        vram_gb:
            GPU VRAM in GB. For MPS (unified memory), this is the full
            system RAM (since CPU and GPU share the pool). None if
            torch is not installed (graceful degradation).

        total_ram_gb:
            Total system RAM in GB. Used to size the unified-memory
            tier on Apple Silicon.

        recommended_sd_model:
            HuggingFace model identifier recommended for this hardware.
            Example: "stabilityai/stable-diffusion-xl-base-1.0"

        recommended_resolution:
            (width, height) tuple recommended for SD output.
            Example: (1920, 1080)

        recommended_fps:
            Frames per second recommended for the final video.

        low_disk_warning:
            True when free disk space is below LOW_DISK_THRESHOLD_GB.
            SD model caches are 4–7 GB; tight disk can cause failures.

        free_disk_gb:
            Free disk space in GB at scan time. Used to set
            low_disk_warning.
    """

    device: str = "cpu"
    vram_gb: float | None = None
    total_ram_gb: float | None = None
    recommended_sd_model: str = "runwayml/stable-diffusion-v1-5"
    recommended_resolution: tuple[int, int] = (512, 512)
    recommended_fps: int = 24
    low_disk_warning: bool = False
    free_disk_gb: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# AnimationConfig — user-facing configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnimationConfig:
    """User-facing configuration for animation generation.

    This lives as an optional field on ContentConfig:
        config.animation_config = AnimationConfig(enabled=True, ...)

    When animation_config is None (default), the pipeline skips all
    animation steps entirely — zero behavioral change to existing code.

    Attributes:
        enabled:
            Master switch. False = skip all animation. Default: False.

        output_format:
            Video container format. "mp4" (H.264, broadest compat) or
            "webm" (VP9, open format). Default: "mp4".

        generate_stills:
            Also extract still images from the video (or directly from
            generated SD frames if no video). Used by the document
            pipeline. Default: False.

        stills_count:
            How many stills to extract. None = one per scene.

        resolution:
            Output video (width, height). Overridden by hardware
            recommendations if not explicitly set. Default: (1920, 1080).

        fps:
            Output video frame rate. Default: 24.

        animation_style_hint:
            Optional style hint for the LLM director. If None, the
            director infers style from the content brief. Examples:
            "BBC documentary", "noir film", "kinetic typography".

        image_model:
            HuggingFace SD model identifier. None = hardware-adaptive
            (HardwareScanner picks the right tier).
            Example: "stabilityai/stable-diffusion-xl-base-1.0"

        render_backend:
            Which video compositing backend to use. None = LLM chooses.
            Currently only "moviepy" is functional; others are stubs.

        visual_brief:
            Dedicated channel for animation-specific user instructions.
            This is what a future ``--animation-brief "..."`` CLI flag
            will populate. Passed directly to the AnimationDirector
            alongside the content brief.
            Example: "use split-screen when speakers interrupt each other"
    """

    enabled: bool = False
    output_format: Literal["mp4", "webm"] = "mp4"
    generate_stills: bool = False
    stills_count: int | None = None
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 24
    animation_style_hint: str | None = None
    image_model: str | None = None
    render_backend: RenderBackend | None = None
    visual_brief: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# AnimationPlan — the LLM director's full creative brief
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnimationPlan:
    """The complete visual script produced by the AnimationDirector.

    This is the LLM's creative output: a structured plan that specifies
    everything the rendering pipeline needs to produce the final video.

    Produced by: AnimationDirector.plan()
    Consumed by: ImageGenerator, TimelineSync, AnimationRenderer

    Attributes:
        plan_id:
            Unique identifier for this plan (UUID or slug).

        brief:
            The original content brief used to generate this plan.

        visual_instructions:
            The animation-specific instructions extracted or derived
            from the brief and visual_brief. Kept for traceability.

        style:
            The visual style chosen by the LLM director.
            One of: "cinematic", "documentary", "kinetic_text",
            "illustrated", "talking_heads".

        render_backend:
            The backend the LLM chose: "moviepy", "manim", or "ffmpeg".
            Only "moviepy" is currently functional.

        color_palette:
            List of hex color strings the LLM chose for overlays,
            title cards, and text. Example: ["#1a1a2e", "#e94560", "#f5f5f5"]

        font_family:
            Font name for text overlays and title cards. None = system default.

        fps:
            Frame rate for this plan (may differ from AnimationConfig.fps
            if the LLM downgraded for performance reasons).

        resolution:
            Output (width, height) for this plan.

        image_model:
            SD model identifier used for frame generation.

        scenes:
            Ordered list of SceneSpec objects — one per visual scene.
            These are the atomic units the renderer works with.

        hardware_profile:
            The hardware profile used when this plan was created.
            Serialized into the plan for reproducibility.

        created_at:
            ISO 8601 timestamp when this plan was created.
    """

    plan_id: str
    brief: str
    visual_instructions: str | None = None
    style: AnimationStyle = "cinematic"
    render_backend: RenderBackend = "moviepy"
    color_palette: list[str] = field(default_factory=lambda: ["#1a1a2e", "#e94560", "#f5f5f5"])
    font_family: str | None = None
    fps: int = 24
    resolution: tuple[int, int] = (1920, 1080)
    image_model: str | None = None
    scenes: list[SceneSpec] = field(default_factory=list)
    hardware_profile: HardwareProfile = field(default_factory=HardwareProfile)
    created_at: str = ""

    @property
    def total_scenes(self) -> int:
        """Number of scenes in this plan."""
        return len(self.scenes)

    @property
    def is_synced(self) -> bool:
        """True when all scenes have timestamps filled in by TimelineSync."""
        return all(
            s.start_time_sec is not None and s.end_time_sec is not None
            for s in self.scenes
        )

    @property
    def is_rendered(self) -> bool:
        """True when all scenes have a generated image path."""
        return all(s.generated_image_path is not None for s in self.scenes)

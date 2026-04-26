"""vox_romana.animation — LLM-directed animation and video generation.

This package turns a vox_romana content brief + AudioEvents JSON into a
synced video (MP4/WebM) and/or still images, driven by an LLM visual
director that makes all creative decisions about style, pacing, and
scene composition.

PIPELINE OVERVIEW:

    User brief + ContentConfig
           │
           ▼
    HardwareScanner ──── scan device + unified memory
           │                → HardwareProfile (recommended SD model, res, fps)
           ▼
    AnimationBootstrap ── brief + hardware → AnimationConfig
           │
           ▼
    AnimationDirector ─── brief + events + hardware → AnimationPlan
           │                (LLM chooses style, color palette, per-scene
           │                 image prompts, transitions, animation hints)
           ▼
    ImageGenerator ────── AnimationPlan → per-scene SD frames
           │                (local Stable Diffusion via diffusers, lazy-loaded;
           │                 API backend stub ready for future key support)
           ▼
    TimelineSync ──────── AnimationPlan + TTS durations → timestamps
           │                (fills SceneSpec.start_time_sec / end_time_sec)
           ▼
    AnimationRenderer ─── AnimationPlan + audio path → video file
           │                dispatches to backend chosen by LLM director:
           │                  moviepy  ← FUNCTIONAL (Ken Burns, overlays, sync)
           │                  manim    ← STUB (NotImplementedError)
           │                  ffmpeg   ← STUB (NotImplementedError)
           ▼
    StillsExtractor ───── video or generated images → keyframe stills
                            (for document pipeline; correlates with video scenes)

USER INPUT:

    All user-visible animation instructions flow through two channels:

    1. The content ``brief`` passed to run.py is read directly by the
       AnimationDirector.  Style cues embedded there ("make it look like
       a BBC documentary", "noir aesthetic, high contrast") are picked up
       naturally by the LLM.

    2. ``AnimationConfig.visual_brief`` — a dedicated override field for
       animation-specific instructions.  This is the channel that a future
       ``--animation-brief "..."`` CLI flag will populate.

NO RUN.PY WIRING YET:

    This package is fully built and ready but not yet connected to the
    main run.py pipeline.  Integration is a separate task (Phase 6+).

HARDWARE DEFAULTS (devstation_01):

    Apple M1 Max, 32 GB unified memory, MPS available.
    Default tier: SDXL + 1920×1080 + 30 fps.
"""

from .bootstrap import AnimationBootstrap
from .director import ANIMATION_DIRECTOR_SYSTEM_PROMPT, AnimationDirector
from .hardware import HardwareScanner
from .image_gen import ImageGenerator
from .models import (
    AnimationConfig,
    AnimationPlan,
    HardwareProfile,
    SceneSpec,
)
from .renderer import AnimationRenderer
from .stills import StillsExtractor

__all__ = [
    "ANIMATION_DIRECTOR_SYSTEM_PROMPT",
    "AnimationBootstrap",
    "AnimationConfig",
    "AnimationDirector",
    "AnimationPlan",
    "AnimationRenderer",
    "HardwareProfile",
    "HardwareScanner",
    "ImageGenerator",
    "SceneSpec",
    "StillsExtractor",
]

"""AnimationBootstrap — turns a brief into a hardware-adapted AnimationConfig.

The animation pipeline has TWO LLM stages, separated in time:

    1. BOOTSTRAP TIME (this module): Pick the hardware-appropriate
       defaults, set the visual_brief channel, decide whether animation
       is enabled. NO LLM CALL — just hardware detection + sensible
       defaults + user overrides. Cheap, fast, deterministic.

    2. RENDER TIME (AnimationDirector, separate module): The director
       reads the brief + the actual AudioEvents and produces the
       AnimationPlan with scene-by-scene image prompts. ONE LLM call,
       happens after the show has been rendered as audio.

Why split them? Because at bootstrap time we don't have the AudioEvents
yet — those come from the conductor running the show. The bootstrap's
job is to produce a config that the conductor + render pipeline can
both consume. The director can't run until events exist.

This mirrors the rest of vox_romana's bootstrap pattern (DebateBootstrap
produces DebateConfig; the actual debate runs later via the conductor).

EXAMPLE:

    from vox_romana.animation.bootstrap import AnimationBootstrap

    bootstrap = AnimationBootstrap()
    config = bootstrap.from_brief(
        "Two economists debate UBI in a university lecture hall",
        visual_brief="academic documentary style, warm lighting",
        overrides={"output_format": "webm"},
    )
    # → AnimationConfig(enabled=True, resolution=(1920, 1080), ...)
"""

from __future__ import annotations

from typing import Any

from ..core.bootstrap import BaseShowBootstrap
from ..core.client import DeepSeekClient
from .hardware import HardwareScanner
from .models import AnimationConfig, HardwareProfile


class AnimationBootstrap(BaseShowBootstrap):
    """Produces a hardware-adapted AnimationConfig from a brief.

    Inherits :class:`BaseShowBootstrap` for the shared client/api_key
    plumbing, even though ``from_brief()`` here makes NO LLM calls. The
    inheritance keeps the API consistent with other show bootstraps
    (DebateBootstrap, NewsBootstrap, etc.) and gives us the client for
    free if a future stage needs it.
    """

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        """Initialize. The client is held for future stages but not used here.

        Args:
            client:  Shared DeepSeekClient. Optional — bootstrap doesn't
                     call the LLM, but the AnimationDirector that runs
                     later does, and it's cleanest to construct the
                     client once at this layer.
            api_key: Forwarded to DeepSeekClient when ``client`` is None.
        """
        super().__init__(client, api_key=api_key)

    # ── Public entry point ────────────────────────────────────────────────

    def from_brief(
        self,
        brief: str,
        *,
        hardware_profile: HardwareProfile | None = None,
        visual_brief: str | None = None,
        enabled: bool = True,
        overrides: dict[str, Any] | None = None,
    ) -> AnimationConfig:
        """Build an AnimationConfig adapted to this machine.

        Args:
            brief:
                The original content brief. Stored for downstream
                consumers (the AnimationDirector reads it later); we
                don't analyze it here.

            hardware_profile:
                Pre-scanned profile. If None, scans the host now via
                :meth:`HardwareScanner.scan`. Pass an explicit profile
                in tests, or when targeting a non-host environment.

            visual_brief:
                Dedicated visual-instruction channel populated by future
                ``--animation-brief`` CLI flag. Stored on the config
                verbatim; the AnimationDirector reads it at render time
                and treats it as authoritative over the main brief.

            enabled:
                Whether the animation pipeline should run at all.
                Defaults True because callers typically wouldn't be
                here unless they want video. ``overrides={"enabled":
                False}`` flips it back off without bypassing the rest
                of the config.

            overrides:
                Dict of AnimationConfig field names → values. Applied
                last so the user always wins. Unknown keys are silently
                ignored (might be for future fields).

        Returns:
            A populated AnimationConfig with hardware-appropriate
            defaults and user overrides applied.
        """
        profile = hardware_profile or HardwareScanner.scan()

        # Hardware drives the defaults. The user (via overrides) can
        # always upgrade or downgrade individual fields.
        config = AnimationConfig(
            enabled=enabled,
            output_format="mp4",
            generate_stills=True,
            stills_count=6,
            resolution=profile.recommended_resolution,
            fps=profile.recommended_fps,
            animation_style_hint=None,
            image_model=profile.recommended_sd_model,
            render_backend="moviepy",  # Only functional backend in Phase 4.
            visual_brief=visual_brief,
        )

        self._apply_overrides_to_animation(config, overrides)
        return config

    # ── Override application (AnimationConfig is not a ContentConfig) ────

    @staticmethod
    def _apply_overrides_to_animation(
        config: AnimationConfig,
        overrides: dict[str, Any] | None,
    ) -> None:
        """Apply overrides in place; silently skip unknown keys.

        Mirrors ``BaseShowBootstrap._apply_overrides`` but typed for
        AnimationConfig (which is a standalone dataclass, not a
        ContentConfig subclass — animation is a cross-cutting feature,
        not a show type).
        """
        if not overrides:
            return
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

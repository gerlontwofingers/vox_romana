"""AnimationRenderer — orchestrates the full animation pipeline.

The AnimationRenderer is the single entry point for turning a completed
show (brief → AudioEvents → AnimationPlan + audio) into a video file.
It chains:

    1. ImageGenerator.generate_for_plan()  — SD frames for every scene
    2. TimelineSync.sync()                 — scene timestamps from TTS durations
    3. Backend.render()                    — moviepy / manim / ffmpeg compositing

Any stage can be skipped by passing None / empty data (see ``render()``
docstring). This lets callers do partial runs during development:
e.g., render from pre-generated images without running SD again.

BACKEND DISPATCH:

    The backend is chosen from ``plan.render_backend`` (set by the
    AnimationDirector). Available backends:

        "moviepy"  — MoviePyBackend (FUNCTIONAL)
        "manim"    — ManifoldBackend (STUB — raises NotImplementedError)
        "ffmpeg"   — FFmpegBackend (STUB — raises NotImplementedError)

    Unrecognised backend names raise ValueError so failures are loud
    rather than silent.

EXAMPLE:

    from pathlib import Path
    from vox_romana.animation.renderer import AnimationRenderer

    renderer = AnimationRenderer()

    output_path = renderer.render(
        plan=plan,
        output_dir=Path("out/video"),
        audio_path=Path("out/show.wav"),
        event_durations={0: 2.5, 1: 4.1, 2: 3.8},
    )
    print("Video at:", output_path)
"""

from __future__ import annotations

import logging
from pathlib import Path

from .backends.base import BaseAnimationBackend
from .backends.ffmpeg_backend import FFmpegBackend
from .backends.manim_backend import ManifoldBackend
from .backends.moviepy_backend import MoviePyBackend
from .image_gen import ImageGenerator
from .models import AnimationPlan
from .sync import TimelineSync

logger = logging.getLogger(__name__)


# Registry of all available backends.
# When a new backend is implemented, add it here.
_BACKEND_REGISTRY: dict[str, type[BaseAnimationBackend]] = {
    "moviepy": MoviePyBackend,
    "manim": ManifoldBackend,
    "ffmpeg": FFmpegBackend,
}


class AnimationRenderer:
    """Orchestrates image generation → timeline sync → video compositing.

    Attributes:
        _image_generator:
            ImageGenerator instance used for SD frame generation.
            Inject a mock in tests.
        _backends:
            Registry mapping backend name → backend class.
            Populated from ``_BACKEND_REGISTRY`` by default.
    """

    def __init__(
        self,
        image_generator: ImageGenerator | None = None,
        *,
        backends: dict[str, type[BaseAnimationBackend]] | None = None,
    ) -> None:
        """Initialize with optional injected dependencies.

        Args:
            image_generator:
                ImageGenerator to use for SD frame generation. If None,
                creates a default one (auto-selects device). Inject a
                mock in tests to avoid loading diffusers.
            backends:
                Backend registry override. If None, uses the module-level
                ``_BACKEND_REGISTRY``. Inject mock backends in tests.
        """
        self._image_generator = image_generator or ImageGenerator()
        self._backends = backends or _BACKEND_REGISTRY

    # ── Main entry point ──────────────────────────────────────────────────

    def render(
        self,
        plan: AnimationPlan,
        output_dir: Path,
        *,
        audio_path: Path | None = None,
        event_durations: dict[int, float] | None = None,
        skip_image_gen: bool = False,
        skip_timeline_sync: bool = False,
        output_resolution: tuple[int, int] | None = None,
        output_filename: str | None = None,
    ) -> Path:
        """Run the full animation pipeline and return the output video path.

        Args:
            plan:
                The AnimationPlan from AnimationDirector. Mutated in place:
                SceneSpec.generated_image_path and timing fields are filled.

            output_dir:
                Directory for all output files (frames subdir + final video).
                Created if it doesn't exist.

            audio_path:
                Path to the show's audio (WAV/MP3). Attached as the video's
                audio track. None = silent video.

            event_durations:
                Mapping of AudioEvent.global_order → duration_sec from TTS.
                Required for timeline sync. If None or empty, timeline sync
                is skipped and scenes get fallback durations.

            skip_image_gen:
                If True, skip ImageGenerator (use whatever
                ``generated_image_path`` is already set on scenes). Useful
                for re-renders after images are already cached.

            skip_timeline_sync:
                If True, skip TimelineSync (use whatever start/end times
                are already set, or let the backend use fallback durations).
                Useful when timestamps were set externally.

            output_resolution:
                When set, overrides ``plan.resolution`` for the compositing
                pass only (SD image generation is unaffected). The original
                resolution is restored after the backend call. Use this for
                ``--video-platform both`` where SDXL generates at 1024×1024
                but the video is composited to 1920×1080 or 1080×1920.

            output_filename:
                Output filename (with or without extension). When set, the
                stem replaces the default ``"video"`` stem. E.g. passing
                ``"ubi_v4_youtube.mp4"`` writes ``ubi_v4_youtube.mp4`` to
                ``output_dir``.

        Returns:
            Path to the written video file.

        Raises:
            ValueError: If ``plan.render_backend`` is not in the registry,
                or if the plan has no scenes.
            ImportError: If the required backend library is not installed.
            NotImplementedError: If the chosen backend is a stub.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Stage 1: Image generation ─────────────────────────────────
        if not skip_image_gen:
            frames_dir = output_dir / "frames"
            logger.info("AnimationRenderer: generating SD frames → %s", frames_dir)
            self._image_generator.generate_for_plan(plan, frames_dir)

        # ── Stage 2: Timeline sync ────────────────────────────────────
        if not skip_timeline_sync and event_durations:
            logger.info(
                "AnimationRenderer: syncing timeline (%d event durations)",
                len(event_durations),
            )
            TimelineSync.sync(plan, event_durations)
        elif not skip_timeline_sync:
            logger.debug(
                "AnimationRenderer: no event_durations supplied — "
                "skipping timeline sync (scenes will use fallback durations)"
            )

        # ── Stage 3: Video rendering ──────────────────────────────────
        backend = self._get_backend(plan.render_backend)
        logger.info(
            "AnimationRenderer: rendering via %s backend", backend.name,
        )

        # Determine output file extension.
        fmt = getattr(plan, "output_format", "mp4")
        # AnimationPlan doesn't carry output_format directly — that's on
        # AnimationConfig. Default to mp4.
        output_ext = fmt if fmt in ("mp4", "webm") else "mp4"

        # Derive output stem: from explicit filename, else default "video".
        stem = Path(output_filename).stem if output_filename else "video"
        output_path = output_dir / f"{stem}.{output_ext}"

        # Override plan.resolution for the compositing pass only.
        # SD image generation (Stage 1) already completed with the original
        # resolution; this affects only the moviepy compositing dimensions.
        _saved_resolution = plan.resolution
        if output_resolution is not None:
            plan.resolution = output_resolution

        try:
            return backend.render(plan, audio_path=audio_path, output_path=output_path)
        finally:
            if output_resolution is not None:
                plan.resolution = _saved_resolution

    # ── Backend dispatch ──────────────────────────────────────────────────

    def _get_backend(self, backend_name: str) -> BaseAnimationBackend:
        """Look up and instantiate a backend by name.

        Args:
            backend_name: Key in the backend registry ("moviepy", etc.).

        Returns:
            An instantiated BaseAnimationBackend.

        Raises:
            ValueError: If the backend name is not in the registry.
        """
        backend_cls = self._backends.get(backend_name)
        if backend_cls is None:
            available = sorted(self._backends.keys())
            raise ValueError(
                f"Unknown render backend {backend_name!r}. "
                f"Available: {available}. "
                "Set plan.render_backend to one of these."
            )
        return backend_cls()

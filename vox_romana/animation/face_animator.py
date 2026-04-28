"""Talking-head face animation via SadTalker.

Lazy-loaded portrait → lip-synced video module. Mirrors
``ImageGenerator``'s lazy-load + MPS-aware pattern: heavy imports
(``torch``, ``sadtalker``, ``moviepy``) are deferred to the first
``animate()`` call so the module is import-safe in test environments
that don't have the ML stack installed.

Public surface:
    - ``FaceAnimationResult``    one rendered talking-head clip
    - ``FaceAnimator``           orchestrator with ``animate`` /
                                  ``animate_for_plan``

The ``animate_for_plan`` helper iterates an ``AnimationPlan``, finds
scenes whose ``animation_hint == "talking_head_sync"`` and have a
``generated_image_path`` (used as the portrait), slices the monolithic
audio file to each scene's window, runs SadTalker, and fills in
``scene.face_animation_path``.

Resume semantics: if the per-scene output mp4 already exists on disk it
is reused without re-rendering, so partial runs can be safely resumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vox_romana.animation.models import AnimationPlan

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Device selection (mirrors ImageGenerator._select_device)
# ─────────────────────────────────────────────────────────────────────────────

def _select_device(device: str | None) -> str:
    """Auto-select the best available compute device.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU.
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
# FaceAnimationResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceAnimationResult:
    """Outcome of a single SadTalker run for one scene."""

    scene_id: str
    video_path: Path
    duration_sec: float


# ─────────────────────────────────────────────────────────────────────────────
# FaceAnimator
# ─────────────────────────────────────────────────────────────────────────────

class FaceAnimator:
    """SadTalker-based talking-head animator with lazy backend loading.

    The SadTalker checkpoint and ``torch`` are *not* loaded at construction
    time — they're loaded on the first ``animate()`` call. This keeps the
    module importable in environments without the ML stack (e.g., CI
    running pure-Python tests).

    Attributes:
        _device:        Compute device ("mps", "cuda", "cpu").
        _model_name:    Backend identifier (currently only "sadtalker").
        _loaded:        True once ``_ensure_loaded`` has succeeded.
        _backend:       The loaded SadTalker engine handle (None until use).
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        model: str = "sadtalker",
    ) -> None:
        """Initialize the animator. SadTalker is NOT loaded here."""
        self._device = _select_device(device)
        self._model_name = model
        self._loaded = False
        self._backend = None

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the SadTalker backend if not already loaded.

        Raises:
            ImportError: If ``torch`` or the SadTalker package is not
                installed, with a clear install command in the message.
        """
        if self._loaded:
            return

        try:
            import torch  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError(
                "torch is required for talking-head animation. "
                "Install with: pip install torch"
            ) from exc

        try:
            # SadTalker is distributed as a git-installable package; the
            # exact import path is wrapped here so we can swap implementations
            # without touching the rest of the codebase.
            import sadtalker  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sadtalker is required for talking-head animation. "
                "Install with: pip install "
                "'git+https://github.com/OpenTalker/SadTalker.git'"
            ) from exc

        logger.info(
            "FaceAnimator: loading %s on %s (first use — may download model weights)",
            self._model_name, self._device,
        )

        self._backend = sadtalker  # placeholder handle; real wiring in B3
        self._loaded = True
        logger.info("FaceAnimator: backend ready")

    # ── Core animation ────────────────────────────────────────────────────

    def animate(
        self,
        portrait_path: Path,
        audio_path: Path,
        output_path: Path,
        *,
        scene_id: str = "scene",
    ) -> FaceAnimationResult:
        """Render a single talking-head clip.

        Args:
            portrait_path: Still image of the speaker (PNG/JPG).
            audio_path:    Audio clip to lip-sync to (WAV).
            output_path:   Destination mp4 path.
            scene_id:      Scene identifier for the returned result.

        Returns:
            A ``FaceAnimationResult`` with the output path and duration.

        Raises:
            ImportError:        If torch/SadTalker are not installed.
            FileNotFoundError:  If portrait or audio paths don't exist.
        """
        self._ensure_loaded()

        portrait = Path(portrait_path)
        audio = Path(audio_path)
        if not portrait.exists():
            raise FileNotFoundError(f"portrait not found: {portrait}")
        if not audio.exists():
            raise FileNotFoundError(f"audio not found: {audio}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(
            "FaceAnimator: animating %s + %s -> %s",
            portrait.name, audio.name, output_path.name,
        )

        # Real SadTalker invocation lands in B3 (MoviePyBackend wiring).
        # Here we provide a typed surface and the duration probe so the
        # plan-level loop and tests have stable behavior.
        duration_sec = self._probe_audio_duration(audio)

        return FaceAnimationResult(
            scene_id=scene_id,
            video_path=output_path,
            duration_sec=duration_sec,
        )

    # ── Plan-level animation ──────────────────────────────────────────────

    def animate_for_plan(
        self,
        plan: "AnimationPlan",
        audio_path: Path,
        output_dir: Path,
    ) -> "AnimationPlan":
        """Render a talking-head clip for every applicable scene in the plan.

        A scene is "applicable" when ALL of the following hold:
            * ``scene.animation_hint == "talking_head_sync"``
            * ``scene.generated_image_path`` points to an existing file
              (used as the portrait)
            * ``scene.start_time_sec`` and ``scene.end_time_sec`` are set
              (so the monolithic audio can be sliced)

        For each applicable scene, the monolithic ``audio_path`` is sliced
        to ``[start_time_sec, end_time_sec]`` (written to a per-scene WAV
        in ``output_dir``), then ``animate()`` is called and
        ``scene.face_animation_path`` is set to the resulting mp4 path.

        RESUME SEMANTICS:
            Scenes whose ``face_animation_path`` already points to an
            existing file are skipped.

        Args:
            plan:        AnimationPlan, mutated in place.
            audio_path:  Path to the monolithic rendered audio (WAV).
            output_dir:  Directory for sliced audio + per-scene mp4 output.

        Returns:
            The same (mutated) AnimationPlan.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = Path(audio_path)

        applicable = [
            s for s in plan.scenes
            if s.animation_hint == "talking_head_sync"
        ]
        total = len(applicable)

        if total == 0:
            logger.debug(
                "FaceAnimator: no talking_head_sync scenes in plan — skipping"
            )
            return plan

        for idx, scene in enumerate(applicable):
            out_path = output_dir / f"face_{scene.scene_id}.mp4"

            # Resume semantics.
            if scene.face_animation_path and Path(scene.face_animation_path).exists():
                logger.debug(
                    "FaceAnimator: scene %s already animated — skipping (%d/%d)",
                    scene.scene_id, idx + 1, total,
                )
                continue

            if not scene.generated_image_path or not Path(scene.generated_image_path).exists():
                logger.warning(
                    "FaceAnimator: scene %s missing portrait (generated_image_path) — skipping",
                    scene.scene_id,
                )
                continue

            if scene.start_time_sec is None or scene.end_time_sec is None:
                logger.warning(
                    "FaceAnimator: scene %s missing timestamps — skipping",
                    scene.scene_id,
                )
                continue

            # Slice the monolithic audio to this scene's window.
            slice_path = output_dir / f"face_{scene.scene_id}.wav"
            self._slice_audio(
                audio_path,
                slice_path,
                start_sec=scene.start_time_sec,
                end_sec=scene.end_time_sec,
            )

            logger.info(
                "FaceAnimator: animating scene %s (%d/%d)",
                scene.scene_id, idx + 1, total,
            )

            result = self.animate(
                portrait_path=Path(scene.generated_image_path),
                audio_path=slice_path,
                output_path=out_path,
                scene_id=scene.scene_id,
            )

            scene.face_animation_path = str(result.video_path)

        return plan

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _probe_audio_duration(audio_path: Path) -> float:
        """Return the duration of a WAV file in seconds, or 0.0 on failure."""
        try:
            import io  # noqa: PLC0415
            import wave  # noqa: PLC0415
            data = Path(audio_path).read_bytes()
            with wave.open(io.BytesIO(data)) as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0

    @staticmethod
    def _slice_audio(
        source: Path,
        destination: Path,
        *,
        start_sec: float,
        end_sec: float,
    ) -> None:
        """Write ``source[start_sec:end_sec]`` to ``destination`` as WAV.

        Uses the stdlib ``wave`` module so no moviepy/ffmpeg dependency
        is pulled in. If the source isn't a readable WAV, the destination
        is left untouched (caller's resume logic will warn on missing file).
        """
        try:
            import wave  # noqa: PLC0415
            with wave.open(str(source), "rb") as wf:
                framerate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                start_frame = max(0, int(start_sec * framerate))
                end_frame = max(start_frame, int(end_sec * framerate))
                wf.setpos(start_frame)
                frames = wf.readframes(end_frame - start_frame)

            destination.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(destination), "wb") as out:
                out.setnchannels(n_channels)
                out.setsampwidth(sampwidth)
                out.setframerate(framerate)
                out.writeframes(frames)
        except Exception as exc:
            logger.warning(
                "FaceAnimator: failed to slice audio %s [%.2f, %.2f]: %s",
                source, start_sec, end_sec, exc,
            )

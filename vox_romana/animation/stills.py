"""StillsExtractor — keyframe stills from video or SD-generated images.

This module bridges the animation pipeline and the document pipeline.
The document pipeline (report generator, show notes, etc.) can use
still images to illustrate articles and summaries. StillsExtractor
provides those images from whichever source is available:

    Priority 1 — Video file present:
        Extract a frame at the midpoint of each scene using
        ``VideoFileClip.get_frame(t)`` from moviepy. This guarantees
        the still matches exactly what appears in the finished video
        (Ken Burns in mid-motion, text overlays composited in).

    Priority 2 — No video, but SD-generated images present:
        Copy the raw SD image for each scene directly to ``output_dir``.
        No video decoding needed. The image won't have text overlays or
        motion effects, but it's the source material the video was built
        from, which is still useful as an illustration.

    Priority 3 — Neither video nor images:
        Scene is skipped. The returned list contains only successfully
        extracted stills.

OUTPUT:

    Files are named ``still_<scene_id>.png`` and saved to ``output_dir``.
    Returns a list of Paths to the written files, in scene order.

LAZY IMPORTS:

    moviepy and PIL are imported lazily (inside methods) so that:
    a) Tests can mock them without the libraries installed.
    b) The whole animation package doesn't break if moviepy is absent
       when only stills (from SD images) are needed.

EXAMPLE:

    from pathlib import Path
    from vox_romana.animation.stills import StillsExtractor

    stills = StillsExtractor.extract(
        plan=plan,
        video_path=Path("out/video.mp4"),   # or None for image-only mode
        output_dir=Path("out/stills"),
    )
    for path in stills:
        print("Still:", path)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .models import AnimationPlan

logger = logging.getLogger(__name__)


class StillsExtractor:
    """Extracts keyframe stills from a rendered video or SD images.

    Stateless — all methods are class methods. Instantiation is optional.
    """

    @classmethod
    def extract(
        cls,
        plan: AnimationPlan,
        video_path: Path | None,
        output_dir: Path,
    ) -> list[Path]:
        """Extract one still per scene and save to output_dir.

        Args:
            plan:
                The AnimationPlan used to produce the video. SceneSpecs
                provide timing (for video extraction) and
                ``generated_image_path`` (for image fallback).

            video_path:
                Path to the rendered video file. If provided and the file
                exists, frames are extracted from the video at scene midpoints.
                Pass ``None`` (or a non-existent path) to skip video extraction
                and fall back to SD images.

            output_dir:
                Directory where still images are written. Created if absent.

        Returns:
            List of Paths to extracted stills, one per scene that had a
            source (video frame or SD image). Empty list if no source was
            available for any scene.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        use_video = video_path is not None and video_path.exists()

        if use_video:
            logger.info(
                "StillsExtractor: extracting %d stills from video %s",
                len(plan.scenes), video_path,
            )
            return cls._extract_from_video(plan, video_path, output_dir)  # type: ignore[arg-type]
        else:
            if video_path is not None:
                logger.debug(
                    "StillsExtractor: video %s not found — falling back to SD images",
                    video_path,
                )
            logger.info(
                "StillsExtractor: copying %d SD images as stills", len(plan.scenes)
            )
            return cls._extract_from_images(plan, output_dir)

    # ── Video extraction ──────────────────────────────────────────────────

    @classmethod
    def _extract_from_video(
        cls,
        plan: AnimationPlan,
        video_path: Path,
        output_dir: Path,
    ) -> list[Path]:
        """Extract frames at scene midpoints from a rendered video.

        Uses moviepy's ``VideoFileClip.get_frame(t)`` which returns a
        NumPy array (H×W×3, uint8). PIL converts this to PNG.

        Falls back to the scene's SD image if:
        - The scene has no timestamps (can't compute midpoint).
        - Frame extraction fails for any reason.
        """
        try:
            import moviepy.editor as mpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "moviepy is required for video frame extraction. "
                "Install with: pip install 'moviepy>=1.0.3,<2.0'"
            ) from exc

        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for saving video frames as stills. "
                "Install with: pip install 'Pillow>=10.4,<11.0'"
            ) from exc

        stills: list[Path] = []
        video_clip = None

        try:
            video_clip = mpy.VideoFileClip(str(video_path))
            video_duration = video_clip.duration

            for scene in plan.scenes:
                out_path = output_dir / f"still_{scene.scene_id}.png"

                # Compute midpoint — fall back to SD image if no timing.
                midpoint = cls._scene_midpoint(scene, video_duration)
                if midpoint is None:
                    logger.debug(
                        "StillsExtractor: scene %s has no timestamps — "
                        "falling back to SD image",
                        scene.scene_id,
                    )
                    fallback = cls._copy_sd_image(scene, out_path)
                    if fallback:
                        stills.append(fallback)
                    continue

                try:
                    frame = video_clip.get_frame(midpoint)  # NumPy H×W×3 uint8
                    img = Image.fromarray(frame)
                    img.save(str(out_path))
                    logger.debug(
                        "StillsExtractor: scene %s → %s (t=%.2fs)",
                        scene.scene_id, out_path.name, midpoint,
                    )
                    stills.append(out_path)
                except Exception as exc:
                    logger.warning(
                        "StillsExtractor: failed to extract frame for scene %s "
                        "(t=%.2fs): %s — falling back to SD image",
                        scene.scene_id, midpoint, exc,
                    )
                    fallback = cls._copy_sd_image(scene, out_path)
                    if fallback:
                        stills.append(fallback)
        finally:
            if video_clip is not None:
                try:
                    video_clip.close()
                except Exception:
                    pass

        logger.info("StillsExtractor: extracted %d stills", len(stills))
        return stills

    # ── Image copy extraction ─────────────────────────────────────────────

    @classmethod
    def _extract_from_images(
        cls,
        plan: AnimationPlan,
        output_dir: Path,
    ) -> list[Path]:
        """Copy each scene's SD-generated image to output_dir as a still.

        Scenes without ``generated_image_path`` or whose file doesn't exist
        are skipped (with a logged warning).
        """
        stills: list[Path] = []

        for scene in plan.scenes:
            out_path = output_dir / f"still_{scene.scene_id}.png"
            result = cls._copy_sd_image(scene, out_path)
            if result:
                stills.append(result)

        logger.info("StillsExtractor: copied %d stills", len(stills))
        return stills

    # ── Shared helpers ────────────────────────────────────────────────────

    @staticmethod
    def _scene_midpoint(scene: object, video_duration: float) -> float | None:
        """Return the scene's midpoint timestamp, or None if not available.

        Clamps to [0, video_duration] so no out-of-bounds frame request
        is made.
        """
        from .models import SceneSpec  # local import to avoid re-export noise  # noqa: PLC0415

        if not isinstance(scene, SceneSpec):
            return None
        if scene.start_time_sec is None or scene.end_time_sec is None:
            return None

        mid = (scene.start_time_sec + scene.end_time_sec) / 2.0
        return max(0.0, min(mid, video_duration - 0.001))

    @staticmethod
    def _copy_sd_image(scene: object, out_path: Path) -> Path | None:
        """Copy the scene's generated SD image to out_path.

        Returns the output Path on success, None on failure.
        """
        from .models import SceneSpec  # local import  # noqa: PLC0415

        if not isinstance(scene, SceneSpec):
            return None
        if not scene.generated_image_path:
            logger.debug(
                "StillsExtractor: scene %s has no generated_image_path — skipping",
                scene.scene_id,
            )
            return None

        src = Path(scene.generated_image_path)
        if not src.exists():
            logger.warning(
                "StillsExtractor: image file %s not found for scene %s — skipping",
                src, scene.scene_id,
            )
            return None

        try:
            shutil.copy2(str(src), str(out_path))
            logger.debug(
                "StillsExtractor: scene %s → %s (from SD image)",
                scene.scene_id, out_path.name,
            )
            return out_path
        except OSError as exc:
            logger.warning(
                "StillsExtractor: failed to copy %s → %s: %s",
                src, out_path, exc,
            )
            return None

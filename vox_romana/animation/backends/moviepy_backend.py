"""MoviePyBackend — functional video compositing backend.

This is the primary (and currently only functional) rendering backend.
Given a fully populated AnimationPlan, it composes a video by:

    1. Loading each scene's generated image (or a black fallback).
    2. Applying the scene's ``animation_hint`` (Ken Burns motion effects).
    3. Rendering ``text_overlay`` and ``speaker_label`` as text clips
       laid over the image.
    4. Concatenating scenes with the specified transitions.
    5. Overlaying the show's audio track (if provided).
    6. Exporting to MP4 or WebM at the plan's resolution and fps.

LAZY LOADING:

    moviepy is imported inside methods (not at module level) so that
    importing this backend in a test environment without moviepy installed
    doesn't cause ImportError. Tests mock moviepy entirely.

KEN BURNS EFFECTS:

    True Ken Burns via moviepy uses the ``resize()`` + ``set_position()``
    approach: the image clip starts at a slightly zoomed-in scale and
    slowly drifts toward 1.0 (zoom-out) or 1.0 → 1.1 (zoom-in). This is
    implemented with ``clip.fl_time()`` (time remapping) plus a resize
    lambda that scales as a function of time.

    For panning, the image is wider than the frame (e.g., 2× width) and
    the position is animated from left to right.

    All effects run entirely within moviepy — no ffmpeg filter graphs or
    subprocess calls.

TEXT OVERLAYS:

    Text is composited using moviepy's ``TextClip``. On macOS, ImageMagick
    must be installed (``brew install imagemagick``) for TextClip to work.
    If ImageMagick is absent, text clips are silently skipped (with a
    logged warning) rather than crashing the render.

FALLBACK IMAGE:

    If a scene's ``generated_image_path`` is missing or the file doesn't
    exist, a solid black ``ColorClip`` is used. The video still renders;
    only the background image is wrong.

TRANSITIONS:

    cut        — instantaneous, no effect
    fade       — fade to black then in from black (0.5s each half)
    crossfade  — direct dissolve to next clip (0.5s)
    wipe       — fade-out + fade-in (same as fade, simpler than true wipe)

    True directional wipes require ffmpeg's xfade filter or manual numpy
    manipulation. For now, wipe is implemented as fade for simplicity.

AUDIO:

    The audio file is attached to the final composite clip. If the audio
    is longer than the assembled video (e.g. because a music bumper adds
    time that scene timestamps don't fully account for), the video is
    extended by freezing the last frame for the overshoot duration — audio
    is never truncated. Differences of less than 50 ms are ignored.

EXAMPLE:

    from pathlib import Path
    from vox_romana.animation.backends.moviepy_backend import MoviePyBackend

    backend = MoviePyBackend()
    output = backend.render(plan, audio_path=Path("show.wav"),
                            output_path=Path("out/video.mp4"))
    print("Video written to:", output)
"""

from __future__ import annotations

import logging
from pathlib import Path

# Pillow ≥ 10.0 removed Image.ANTIALIAS (replaced by Image.LANCZOS).
# moviepy 1.x still references Image.ANTIALIAS in its resize filter.
# Patch it back in so moviepy doesn't crash on modern Pillow.
try:
    from PIL import Image as _PILImage
    if not hasattr(_PILImage, "ANTIALIAS"):
        _PILImage.ANTIALIAS = _PILImage.LANCZOS  # type: ignore[attr-defined]
except ImportError:
    pass

from ..models import AnimationPlan, SceneSpec
from .base import BaseAnimationBackend

logger = logging.getLogger(__name__)

# Transition duration in seconds (for fade / crossfade).
_TRANSITION_DURATION = 0.5

# Ken Burns effect: how much to zoom during the scene (fraction of 1.0).
# e.g., 0.05 → zoom from 1.00 to 1.05 over the scene duration.
_KEN_BURNS_ZOOM_DELTA = 0.05

# Pan distance as a fraction of frame width (for pan_left / pan_right).
_KEN_BURNS_PAN_FRACTION = 0.08

# Fallback solid colour when an image file is absent.
_FALLBACK_COLOR = (10, 10, 20)  # dark navy

# Text overlay position: bottom-left, with small margin.
_OVERLAY_POSITION = ("left", "bottom")
_OVERLAY_MARGIN_PX = 40

# Speaker-label card: white text near bottom-left, larger than overlay.
_LABEL_FONTSIZE = 32
_OVERLAY_FONTSIZE = 28

# ── Font resolution ───────────────────────────────────────────────────────────

def _resolve_font() -> str | None:
    """Return an absolute path to a usable TrueType/TTC font file, or None.

    ImageMagick (Homebrew) does not inherit macOS system fonts, so passing a
    font *name* such as "Helvetica" fails with "unable to read font".
    Passing an absolute path bypasses the font registry and works via
    FreeType directly.  We probe candidates in preference order and cache
    the first hit at module import time.
    """
    import platform as _platform
    from pathlib import Path as _Path

    _CANDIDATES: list[str]
    if _platform.system() == "Darwin":
        _CANDIDATES = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/opt/homebrew/share/fonts/dejavu-fonts-ttf/DejaVuSans.ttf",
            "/usr/local/share/fonts/DejaVuSans.ttf",
        ]
    else:
        _CANDIDATES = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
    for candidate in _CANDIDATES:
        if _Path(candidate).exists():
            return candidate
    return None


# Resolved once at import time; None means use ImageMagick's built-in default.
_RESOLVED_FONT: str | None = _resolve_font()


class MoviePyBackend(BaseAnimationBackend):
    """MoviePy-based video compositor — the functional backend.

    All video compositing is done through moviepy's API. The moviepy
    library is imported lazily so tests can mock it without installing
    the real package.
    """

    @property
    def name(self) -> str:
        return "moviepy"

    # ── Main entry point ──────────────────────────────────────────────────

    def render(
        self,
        plan: AnimationPlan,
        audio_path: Path | None,
        output_path: Path,
    ) -> Path:
        """Render AnimationPlan → video file via moviepy.

        Args:
            plan:
                Fully populated AnimationPlan. SceneSpecs should have
                ``generated_image_path`` and timing set. The backend
                degrades gracefully when these are absent.
            audio_path:
                Path to WAV/MP3 for the audio track. None = silent video.
            output_path:
                Target path for the output video. Extension determines
                codec (`.mp4` → H.264/AAC, `.webm` → VP9/Opus).

        Returns:
            Path to the written video file.

        Raises:
            ImportError: If moviepy is not installed.
        """
        try:
            import moviepy.editor as mpy  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "moviepy is required for video rendering. "
                "Install with: pip install 'moviepy>=1.0.3,<2.0'"
            ) from exc

        if not plan.scenes:
            raise ValueError("AnimationPlan has no scenes — nothing to render.")

        width, height = plan.resolution
        fps = plan.fps

        logger.info(
            "MoviePyBackend: rendering %d scenes at %dx%d @ %dfps → %s",
            len(plan.scenes), width, height, fps, output_path,
        )

        # Build one moviepy clip per scene.
        scene_clips = [
            self._build_scene_clip(scene, width=width, height=height, mpy=mpy)
            for scene in plan.scenes
        ]

        # Concatenate with transitions.
        final_clip = self._concatenate_with_transitions(
            scene_clips, plan.scenes, mpy=mpy,
        )

        # Attach audio if provided.
        if audio_path and audio_path.exists():
            logger.debug("MoviePyBackend: attaching audio from %s", audio_path)
            audio_clip = mpy.AudioFileClip(str(audio_path))
            # If the audio is longer than the video (e.g. because the intro
            # music duration wasn't fully accounted for in scene timing, or
            # mixer pauses add a small tail), extend the video by freezing
            # the last frame rather than truncating the audio.
            overshoot = audio_clip.duration - final_clip.duration
            if overshoot > 0.05:  # ignore sub-50ms rounding differences
                logger.debug(
                    "MoviePyBackend: audio is %.2fs longer than video — "
                    "freezing last frame to fill gap",
                    overshoot,
                )
                freeze_t = max(0.0, final_clip.duration - 0.001)
                freeze = (
                    final_clip.to_ImageClip(t=freeze_t)
                    .set_duration(overshoot)
                )
                final_clip = mpy.concatenate_videoclips([final_clip, freeze])
            final_clip = final_clip.set_audio(audio_clip)

        # Determine codec from extension.
        suffix = output_path.suffix.lower()
        codec, audio_codec = self._choose_codecs(suffix)

        logger.info(
            "MoviePyBackend: writing video (codec=%s, audio=%s) ...",
            codec, audio_codec,
        )

        final_clip.write_videofile(
            str(output_path),
            fps=fps,
            codec=codec,
            audio_codec=audio_codec,
            logger=None,    # Suppress moviepy's own progress bars.
        )

        logger.info("MoviePyBackend: done → %s", output_path)
        return output_path

    # ── Per-scene clip construction ───────────────────────────────────────

    def _build_scene_clip(
        self,
        scene: SceneSpec,
        *,
        width: int,
        height: int,
        mpy: object,
    ) -> object:
        """Build a moviepy clip for one scene.

        Steps:
            1. Load background (image or black fallback).
            2. Set duration from timing (or a default 3 s fallback).
            3. Apply Ken Burns / static motion hint.
            4. Composite text_overlay and speaker_label on top.
        """
        duration = scene.duration_sec or 3.0

        # ── Background ────────────────────────────────────────────────
        bg = self._load_background(scene, width=width, height=height,
                                   duration=duration, mpy=mpy)

        # ── Motion hint ───────────────────────────────────────────────
        bg = self._apply_animation_hint(bg, scene.animation_hint,
                                        width=width, height=height,
                                        duration=duration, mpy=mpy)

        # ── Text layers ───────────────────────────────────────────────
        layers = [bg]

        if scene.speaker_label:
            label_clip = self._make_text_clip(
                scene.speaker_label,
                fontsize=_LABEL_FONTSIZE,
                duration=min(duration, 4.0),  # show label for max 4 s
                width=width, height=height,
                position=(_OVERLAY_MARGIN_PX, height - _OVERLAY_MARGIN_PX - _LABEL_FONTSIZE * 3),
                mpy=mpy,
            )
            if label_clip is not None:
                layers.append(label_clip)

        if scene.text_overlay:
            overlay_clip = self._make_text_clip(
                scene.text_overlay,
                fontsize=_OVERLAY_FONTSIZE,
                duration=duration,
                width=width, height=height,
                position=(_OVERLAY_MARGIN_PX, height - _OVERLAY_MARGIN_PX - _OVERLAY_FONTSIZE),
                mpy=mpy,
            )
            if overlay_clip is not None:
                layers.append(overlay_clip)

        # ── Face-animation overlay (SadTalker mp4 from FaceAnimator) ──
        # When the scene has a talking-head clip we composite it on top
        # of the background as a centred lower-third. The clip is loaded
        # lazily so backends without face animation still work.
        face_path = getattr(scene, "face_animation_path", None)
        if face_path and Path(face_path).exists():
            face_clip = self._load_face_clip(
                face_path,
                duration=duration,
                width=width,
                height=height,
                mpy=mpy,
            )
            if face_clip is not None:
                layers.append(face_clip)

        if len(layers) == 1:
            return layers[0]

        return mpy.CompositeVideoClip(layers, size=(width, height))  # type: ignore[attr-defined]

    # ── Face-animation overlay loading ────────────────────────────────────

    def _load_face_clip(
        self,
        face_path: str,
        *,
        duration: float,
        width: int,
        height: int,
        mpy: object,
    ) -> object | None:
        """Load a SadTalker mp4 as a positioned overlay clip.

        The clip is centred horizontally and anchored to the lower third
        of the frame. Failures to load are logged and return ``None`` so
        compositing falls back to the still background gracefully.
        """
        try:
            face_clip = mpy.VideoFileClip(str(face_path))  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning(
                "MoviePyBackend: failed to load face clip %s: %s",
                face_path, exc,
            )
            return None

        # Crop or extend to scene duration so concatenation timing is preserved.
        if face_clip.duration > duration:
            face_clip = face_clip.subclip(0, duration)

        # Position: centred horizontally, anchored ~10% above the bottom.
        try:
            face_clip = face_clip.set_position(("center", "bottom"))
        except Exception:  # pragma: no cover — defensive
            pass

        return face_clip

    def _load_background(
        self,
        scene: SceneSpec,
        *,
        width: int,
        height: int,
        duration: float,
        mpy: object,
    ) -> object:
        """Load the scene's generated image or a solid-colour fallback."""
        if scene.generated_image_path:
            img_path = Path(scene.generated_image_path)
            if img_path.exists():
                try:
                    clip = mpy.ImageClip(str(img_path)).set_duration(duration)  # type: ignore[attr-defined]
                    # Scale to cover: resize proportionally so both dimensions
                    # are >= target, then center-crop to exact frame size.
                    # This is the CSS background-size:cover equivalent —
                    # no aspect-ratio distortion regardless of source dims.
                    img_w, img_h = clip.size  # type: ignore[attr-defined]
                    scale = max(width / img_w, height / img_h)
                    clip = clip.resize(scale)  # type: ignore[attr-defined]
                    clip = clip.crop(  # type: ignore[attr-defined]
                        x_center=clip.w / 2,  # type: ignore[attr-defined]
                        y_center=clip.h / 2,  # type: ignore[attr-defined]
                        width=width,
                        height=height,
                    )
                    return clip
                except Exception as exc:
                    logger.warning(
                        "MoviePyBackend: could not load image %s (%s) — using fallback",
                        img_path, exc,
                    )

        logger.debug(
            "MoviePyBackend: no image for scene %s — using black fallback",
            scene.scene_id,
        )
        return mpy.ColorClip(  # type: ignore[attr-defined]
            size=(width, height),
            color=_FALLBACK_COLOR,
        ).set_duration(duration)

    def _apply_animation_hint(
        self,
        clip: object,
        hint: str,
        *,
        width: int,
        height: int,
        duration: float,
        mpy: object,
    ) -> object:
        """Apply a Ken Burns or other motion effect to a clip.

        All Ken Burns effects are implemented as resize + crop lambdas so
        they run purely inside moviepy without ffmpeg filter graphs.
        """
        if hint == "static" or duration < 0.1:
            return clip

        # Common helper: zoom in over time.
        if hint == "ken_burns_zoom_in":
            return clip.resize(  # type: ignore[attr-defined]
                lambda t: 1.0 + (_KEN_BURNS_ZOOM_DELTA * t / duration)
            ).crop(x_center=width / 2, y_center=height / 2, width=width, height=height)  # type: ignore[attr-defined]

        if hint == "ken_burns_zoom_out":
            zoom_ratio = 1.0 + _KEN_BURNS_ZOOM_DELTA
            return clip.resize(  # type: ignore[attr-defined]
                lambda t: zoom_ratio - (_KEN_BURNS_ZOOM_DELTA * t / duration)
            ).crop(x_center=width / 2, y_center=height / 2, width=width, height=height)  # type: ignore[attr-defined]

        if hint == "ken_burns_pan_left":
            pan_px = int(width * _KEN_BURNS_PAN_FRACTION)
            # Expand to wider image, pan left → right viewport movement.
            # moviepy 1.x crop() cannot accept a callable for x1 (it
            # evaluates x2 = x1 + width inline, crashing with TypeError).
            # Use fl() to slice the numpy frame directly instead.
            wide = clip.resize(width=width + pan_px)  # type: ignore[attr-defined]
            def _pan_left(get_frame, t):  # noqa: E306
                frame = get_frame(t)
                x1 = int(pan_px * (1 - t / duration))
                return frame[:height, x1:x1 + width]
            return wide.fl(_pan_left)  # type: ignore[attr-defined]

        if hint == "ken_burns_pan_right":
            pan_px = int(width * _KEN_BURNS_PAN_FRACTION)
            wide = clip.resize(width=width + pan_px)  # type: ignore[attr-defined]
            def _pan_right(get_frame, t):  # noqa: E306
                frame = get_frame(t)
                x1 = int(pan_px * t / duration)
                return frame[:height, x1:x1 + width]
            return wide.fl(_pan_right)  # type: ignore[attr-defined]

        if hint == "parallax":
            # Subtle vertical drift — parallax requires multi-layer compositing
            # which we don't have for single images; treat as a gentle zoom-in.
            return clip.resize(  # type: ignore[attr-defined]
                lambda t: 1.0 + (_KEN_BURNS_ZOOM_DELTA * 0.5 * t / duration)
            ).crop(x_center=width / 2, y_center=height / 2, width=width, height=height)  # type: ignore[attr-defined]

        if hint == "text_pulse":
            # Text-pulse is applied to text layers, not the background image.
            # Background stays static.
            return clip

        # Unknown hint — log and return unchanged.
        logger.debug("MoviePyBackend: unknown animation_hint %r — static", hint)
        return clip

    def _make_text_clip(
        self,
        text: str,
        *,
        fontsize: int,
        duration: float,
        width: int,
        height: int,
        position: tuple[int, int],
        mpy: object,
    ) -> object | None:
        """Create a text overlay clip, returning None if ImageMagick is absent."""
        # Use an absolute font path so ImageMagick (Homebrew) can find it via
        # FreeType without relying on its own font-name registry, which does
        # not include macOS system fonts.  If no candidate path was found at
        # import time we omit the font kwarg and let ImageMagick pick its
        # own default rather than crashing.
        base_kwargs: dict = {
            "fontsize": fontsize,
            "color": "white",
            "stroke_color": "black",
            "stroke_width": 1,
            "method": "caption",
            "size": (width - _OVERLAY_MARGIN_PX * 2, None),
        }
        if _RESOLVED_FONT is not None:
            base_kwargs["font"] = _RESOLVED_FONT
        try:
            txt = mpy.TextClip(  # type: ignore[attr-defined]
                text,
                **base_kwargs,
            ).set_duration(duration).set_position(position)
            return txt
        except Exception as exc:
            logger.warning(
                "MoviePyBackend: TextClip failed (ImageMagick installed?) — "
                "skipping overlay %r: %s", text[:40], exc,
            )
            return None

    # ── Transition / concatenation ────────────────────────────────────────

    def _concatenate_with_transitions(
        self,
        clips: list,
        scenes: list[SceneSpec],
        *,
        mpy: object,
    ) -> object:
        """Concatenate clips, applying inter-scene transitions.

        For cut (most common), uses moviepy's ``concatenate_videoclips``
        directly with ``method="compose"`` for clean joins.

        For fade/crossfade/wipe, we apply fade-out/in effects on the
        relevant clips before concatenating.
        """
        if len(clips) == 1:
            return clips[0]

        processed: list = []
        for idx, (clip, scene) in enumerate(zip(clips, scenes)):
            is_first = idx == 0
            is_last = idx == len(clips) - 1

            transition_in = scene.transition_in
            transition_out = scene.transition_out

            # Apply fade-in on first scene if it requests it.
            if is_first and transition_in in ("fade", "crossfade", "wipe"):
                clip = clip.fadein(_TRANSITION_DURATION)  # type: ignore[attr-defined]

            # Apply fade-out on last scene or crossfade-in to next.
            if transition_out == "crossfade" and not is_last:
                clip = clip.crossfadeout(_TRANSITION_DURATION)  # type: ignore[attr-defined]
            elif transition_out in ("fade", "wipe"):
                clip = clip.fadeout(_TRANSITION_DURATION)  # type: ignore[attr-defined]

            processed.append(clip)

        # For crossfade, use padding=-transition_duration so clips overlap.
        has_crossfade = any(
            s.transition_out == "crossfade" for s in scenes[:-1]
        )
        if has_crossfade:
            return mpy.concatenate_videoclips(  # type: ignore[attr-defined]
                processed,
                method="compose",
                padding=-_TRANSITION_DURATION,
            )

        return mpy.concatenate_videoclips(processed, method="compose")  # type: ignore[attr-defined]

    # ── Codec selection ───────────────────────────────────────────────────

    @staticmethod
    def _choose_codecs(suffix: str) -> tuple[str, str]:
        """Return (video_codec, audio_codec) for the given file extension."""
        if suffix == ".webm":
            return "libvpx-vp9", "libopus"
        # Default: H.264 + AAC — broadest compatibility.
        return "libx264", "aac"

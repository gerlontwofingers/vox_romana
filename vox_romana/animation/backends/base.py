"""Abstract base class for animation rendering backends.

All rendering backends implement this interface. The AnimationRenderer
dispatches to the correct backend by name; backends only need to implement
``render()``.

DESIGN NOTES:

    - ``render()`` is the only required method. Backends receive the full
      AnimationPlan (which contains all scene images, timing, transitions,
      style decisions) and the optional audio path.

    - ``audio_path`` is None when rendering a video-only preview (no TTS
      output available yet) or when the caller wants a silent video.

    - Backends must handle the case where ``scene.generated_image_path``
      is None — the ImageGenerator may not have run. A reasonable fallback
      is a solid-color frame.

    - ``output_path`` does not need to exist; backends create it.
      The parent directory is guaranteed to exist by the AnimationRenderer
      before calling ``render()``.

    - The return value is the path where the video was actually written
      (may differ from ``output_path`` if the backend appended an extension).
"""

from __future__ import annotations

import abc
from pathlib import Path

from ..models import AnimationPlan


class BaseAnimationBackend(abc.ABC):
    """Abstract base for all animation rendering backends.

    Subclass and implement :meth:`render`.
    """

    @abc.abstractmethod
    def render(
        self,
        plan: AnimationPlan,
        audio_path: Path | None,
        output_path: Path,
    ) -> Path:
        """Render the AnimationPlan into a video file.

        Args:
            plan:
                The fully populated AnimationPlan. All SceneSpec entries
                should have ``generated_image_path`` set (from
                ImageGenerator) and ``start_time_sec`` / ``end_time_sec``
                set (from TimelineSync). Backends should degrade gracefully
                if these are absent.

            audio_path:
                Path to the show's audio file (WAV or MP3). The backend
                overlays this on the video. None = no audio track.

            output_path:
                Desired output file path. The backend writes here (or
                to a path derived from this, e.g. with an extension).
                The parent directory already exists.

        Returns:
            The actual path of the written video file.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable backend name (defaults to class name)."""
        return type(self).__name__

"""FFmpegBackend — direct ffmpeg rendering stub.

Direct ffmpeg gives maximum control over codec parameters, HDR output,
hardware-accelerated encoding (VideoToolbox on macOS, NVENC on NVIDIA),
and streaming-optimised formats. It requires the ``ffmpeg`` system binary.

When ffmpeg backend support is added, this stub should be replaced with a
real implementation. The approach would typically be:
    1. Write scene images + timestamps to a concat demuxer file.
    2. Build an ffmpeg filter graph (Ken Burns via zoompan filter, overlays
       via drawtext/overlay filters, transitions via xfade).
    3. Shell out via subprocess.run (or ffmpeg-python library).

CONTRIBUTING:

    To implement ffmpeg backend support:
        1. Ensure ffmpeg is installed: brew install ffmpeg
        2. Implement ``render()`` in this class.
        3. Remove the NotImplementedError.
        4. Add tests to tests/test_animation_renderer.py.
"""

from __future__ import annotations

from pathlib import Path

from ..models import AnimationPlan
from .base import BaseAnimationBackend


class FFmpegBackend(BaseAnimationBackend):
    """Direct FFmpeg rendering backend — NOT YET IMPLEMENTED.

    Raises ``NotImplementedError`` immediately. Use ``MoviePyBackend``
    (the default) for all current production work.
    """

    def render(
        self,
        plan: AnimationPlan,
        audio_path: Path | None,
        output_path: Path,
    ) -> Path:
        raise NotImplementedError(
            "The ffmpeg backend is not yet implemented. "
            "Use render_backend='moviepy' for production work. "
            "To contribute an ffmpeg implementation, see "
            "vox_romana/animation/backends/ffmpeg_backend.py."
        )

    @property
    def name(self) -> str:
        return "ffmpeg"

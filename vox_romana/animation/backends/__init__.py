"""Animation rendering backends package.

Each backend implements :class:`BaseAnimationBackend` and handles the
actual video compositing. The :class:`AnimationRenderer` (``renderer.py``)
dispatches to the correct backend based on ``AnimationPlan.render_backend``.

Backends:
    moviepy  — FUNCTIONAL. Ken Burns, text overlays, transitions, audio sync.
    manim    — STUB.  Raises NotImplementedError with helpful message.
    ffmpeg   — STUB.  Raises NotImplementedError with helpful message.
"""

from .base import BaseAnimationBackend
from .ffmpeg_backend import FFmpegBackend
from .manim_backend import ManifoldBackend
from .moviepy_backend import MoviePyBackend

__all__ = [
    "BaseAnimationBackend",
    "FFmpegBackend",
    "ManifoldBackend",
    "MoviePyBackend",
]

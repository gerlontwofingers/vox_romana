"""ManifoldBackend — Manim rendering stub.

Manim is a mathematical animation engine originally written by 3Blue1Brown.
It excels at animated math visualizations, graph theory, and data
storytelling — not the kind of editorial video production this pipeline
currently targets.

When manim support is added, this stub should be replaced with a real
implementation. Until then it raises a clear NotImplementedError so the
failure mode is obvious rather than silent.

CONTRIBUTING:

    To implement manim backend support:
        1. Install manim: pip install manim
        2. Implement ``render()`` in this class.
        3. Remove the NotImplementedError.
        4. Add tests to tests/test_animation_renderer.py.

    Manim scenes are defined as Python classes, not composited from
    images, so the implementation will look quite different from
    MoviePyBackend. The plan's color_palette, font_family, and
    animation_hint fields map well to Manim's style system.
"""

from __future__ import annotations

from pathlib import Path

from ..models import AnimationPlan
from .base import BaseAnimationBackend


class ManifoldBackend(BaseAnimationBackend):
    """Manim animation backend — NOT YET IMPLEMENTED.

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
            "The manim backend is not yet implemented. "
            "Use render_backend='moviepy' for production work. "
            "To contribute a manim implementation, see "
            "vox_romana/animation/backends/manim_backend.py."
        )

    @property
    def name(self) -> str:
        return "manim"

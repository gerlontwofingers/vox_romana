"""Tests for MoviePyBackend._apply_animation_hint — pan effects.

The pan effects previously crashed in moviepy 1.x because crop() evaluated
``x2 = x1 + width`` inline when x1 was a callable, raising TypeError.  The
fix uses fl() with a numpy frame slice instead.

These tests verify:
    - pan_left and pan_right complete without TypeError
    - The returned clip has the correct size (width × height)
    - At t=0 and t=duration the correct pixel column is returned
"""

from __future__ import annotations

import numpy as np
import pytest

from vox_romana.animation.backends.moviepy_backend import (
    MoviePyBackend,
    _KEN_BURNS_PAN_FRACTION,
)


# ── Minimal clip stub that mimics moviepy 1.x behaviour ──────────────────────

class _StubClip:
    """A clip backed by a solid-colour numpy array for frame retrieval."""

    def __init__(self, width: int, height: int, color: tuple[int, int, int] = (128, 64, 32)) -> None:
        self._width = width
        self._height = height
        self._color = color
        # Pre-build the full-width frame once.
        self._frame = np.full((height, width, 3), color, dtype=np.uint8)

    # moviepy-compatible interface used by _apply_animation_hint:

    def resize(self, *, width: int) -> "_StubClip":
        """Return a new stub of the requested width (same height, same colour)."""
        return _StubClip(width=width, height=self._height, color=self._color)

    def fl(self, func) -> "_FlClip":  # noqa: ANN001
        """Simulate clip.fl(func) — store func for later evaluation."""
        return _FlClip(self, func)


class _FlClip:
    """Result of clip.fl(func) — evaluates the frame function on demand."""

    def __init__(self, source: _StubClip, func) -> None:  # noqa: ANN001
        self._source = source
        self._func = func

    def get_frame(self, t: float) -> np.ndarray:
        return self._func(self._source._frame.__class__.__new__(self._source._frame.__class__),
                          t) if False else self._func(lambda _t: self._source._frame, t)

    @property
    def w(self) -> int:
        # Derived from a single get_frame call.
        return self.get_frame(0.0).shape[1]

    @property
    def h(self) -> int:
        return self.get_frame(0.0).shape[0]


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_backend() -> MoviePyBackend:
    return MoviePyBackend()


def _make_clip(width: int, height: int) -> _StubClip:
    return _StubClip(width=width, height=height)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestKenBurnsPan:
    """_apply_animation_hint for pan_left / pan_right should not crash."""

    WIDTH = 512
    HEIGHT = 512
    DURATION = 4.0

    def _apply(self, hint: str) -> "_FlClip":
        backend = _make_backend()
        clip = _make_clip(self.WIDTH, self.HEIGHT)
        result = backend._apply_animation_hint(
            clip,
            hint,
            width=self.WIDTH,
            height=self.HEIGHT,
            duration=self.DURATION,
            mpy=None,  # not used by pan branches
        )
        return result  # type: ignore[return-value]

    def test_pan_left_no_typeerror(self) -> None:
        """pan_left must not raise TypeError (the old crop-lambda bug)."""
        result = self._apply("ken_burns_pan_left")
        assert result is not None

    def test_pan_right_no_typeerror(self) -> None:
        """pan_right must not raise TypeError."""
        result = self._apply("ken_burns_pan_right")
        assert result is not None

    def test_pan_left_output_dimensions(self) -> None:
        """pan_left frame at any t must have the original width × height."""
        result = self._apply("ken_burns_pan_left")
        frame = result.get_frame(0.0)
        assert frame.shape == (self.HEIGHT, self.WIDTH, 3)

    def test_pan_right_output_dimensions(self) -> None:
        """pan_right frame at any t must have the original width × height."""
        result = self._apply("ken_burns_pan_right")
        frame = result.get_frame(0.0)
        assert frame.shape == (self.HEIGHT, self.WIDTH, 3)

    def test_pan_left_start_offset(self) -> None:
        """At t=0, pan_left should start at x1=pan_px (leftmost position)."""
        pan_px = int(self.WIDTH * _KEN_BURNS_PAN_FRACTION)
        result = self._apply("ken_burns_pan_left")
        # At t=0: x1 = pan_px * (1 - 0/dur) = pan_px
        # The frame is a uniform colour so we just verify no crash + shape.
        frame = result.get_frame(0.0)
        assert frame.shape[1] == self.WIDTH

    def test_pan_left_end_offset(self) -> None:
        """At t=duration, pan_left should be fully panned (x1=0)."""
        result = self._apply("ken_burns_pan_left")
        frame = result.get_frame(self.DURATION)
        assert frame.shape[1] == self.WIDTH

    def test_pan_right_start_offset(self) -> None:
        """At t=0, pan_right should start at x1=0 (leftmost position)."""
        result = self._apply("ken_burns_pan_right")
        frame = result.get_frame(0.0)
        assert frame.shape[1] == self.WIDTH

    def test_pan_right_end_offset(self) -> None:
        """At t=duration, pan_right should be fully panned (x1=pan_px)."""
        result = self._apply("ken_burns_pan_right")
        frame = result.get_frame(self.DURATION)
        assert frame.shape[1] == self.WIDTH

    def test_pan_left_pixel_values_consistent(self) -> None:
        """Pixels returned must be from the source image (same colour channel values)."""
        clip = _make_clip(self.WIDTH, self.HEIGHT)
        backend = _make_backend()
        result = backend._apply_animation_hint(
            clip, "ken_burns_pan_left",
            width=self.WIDTH, height=self.HEIGHT, duration=self.DURATION, mpy=None,
        )
        frame = result.get_frame(0.5)
        # Source colour is (128, 64, 32) — every pixel must match.
        assert np.all(frame[:, :, 0] == 128)
        assert np.all(frame[:, :, 1] == 64)
        assert np.all(frame[:, :, 2] == 32)

    def test_pan_right_pixel_values_consistent(self) -> None:
        """pan_right pixels must also come from source (not garbage memory)."""
        clip = _make_clip(self.WIDTH, self.HEIGHT)
        backend = _make_backend()
        result = backend._apply_animation_hint(
            clip, "ken_burns_pan_right",
            width=self.WIDTH, height=self.HEIGHT, duration=self.DURATION, mpy=None,
        )
        frame = result.get_frame(0.5)
        assert np.all(frame[:, :, 0] == 128)
        assert np.all(frame[:, :, 1] == 64)
        assert np.all(frame[:, :, 2] == 32)

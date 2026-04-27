"""Tests for video/animation dependency checks added to preflight.py.

Strategy:
    - check_imagemagick() is tested by mocking shutil.which and subprocess.
    - check_video_python_deps() is tested by mocking sys.modules.
    - _run_all() integration check verifies the video section is present.

All tests run without any real system tools or Python packages installed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from preflight import (
    Result,
    check_imagemagick,
    check_video_python_deps,
    _run_all,
)


# ── check_imagemagick ─────────────────────────────────────────────────────────

class TestCheckImagemagick:
    def test_found_via_magick_binary(self) -> None:
        """When `magick` is on PATH, result is ok with version info."""
        def _which(cmd: str) -> str | None:
            return "/usr/local/bin/magick" if cmd == "magick" else None

        with patch("shutil.which", side_effect=_which):
            with patch("subprocess.check_output", return_value="ImageMagick 7.1.1-15"):
                result = check_imagemagick()

        assert result.ok is True
        assert result.required is False
        assert "magick" in result.note or "ImageMagick" in result.note

    def test_found_via_convert_fallback(self) -> None:
        """Falls back to `convert` when `magick` is absent."""
        def _which(cmd: str) -> str | None:
            return "/usr/bin/convert" if cmd == "convert" else None

        with patch("shutil.which", side_effect=_which):
            with patch("subprocess.check_output", return_value="ImageMagick 6.9.13"):
                result = check_imagemagick()

        assert result.ok is True
        assert result.required is False

    def test_not_found(self) -> None:
        """When neither binary is on PATH, result is not ok and not required."""
        with patch("shutil.which", return_value=None):
            result = check_imagemagick()

        assert result.ok is False
        assert result.required is False
        assert "brew install imagemagick" in result.note

    def test_subprocess_error_still_returns_ok(self) -> None:
        """`--version` failing does not mark the check as failed."""
        def _which(cmd: str) -> str | None:
            return "/usr/local/bin/magick" if cmd == "magick" else None

        with patch("shutil.which", side_effect=_which):
            with patch("subprocess.check_output", side_effect=Exception("oops")):
                result = check_imagemagick()

        assert result.ok is True
        assert result.required is False

    def test_result_is_not_required(self) -> None:
        """imagemagick should never be a hard requirement."""
        with patch("shutil.which", return_value=None):
            result = check_imagemagick()
        assert result.required is False

    def test_result_name(self) -> None:
        with patch("shutil.which", return_value=None):
            result = check_imagemagick()
        assert result.name == "imagemagick"


# ── check_video_python_deps ───────────────────────────────────────────────────

class TestCheckVideoPythonDeps:
    def test_returns_three_results(self) -> None:
        """Must check exactly diffusers, moviepy, and Pillow."""
        results = check_video_python_deps()
        assert len(results) == 3

    def test_all_optional(self) -> None:
        """All video Python deps are optional (not required)."""
        results = check_video_python_deps()
        for r in results:
            assert r.required is False, f"{r.name} should not be required"

    def test_diffusers_present(self) -> None:
        """diffusers importable → ok."""
        mock_diffusers = MagicMock()
        with patch.dict("sys.modules", {"diffusers": mock_diffusers}):
            results = check_video_python_deps()
        diffusers_result = next(r for r in results if r.name == "diffusers")
        assert diffusers_result.ok is True

    def test_diffusers_absent(self) -> None:
        """diffusers not importable → not ok, not required."""
        with patch.dict("sys.modules", {"diffusers": None}):
            results = check_video_python_deps()
        diffusers_result = next(r for r in results if r.name == "diffusers")
        assert diffusers_result.ok is False
        assert diffusers_result.required is False

    def test_moviepy_present(self) -> None:
        mock_mpy = MagicMock()
        with patch.dict("sys.modules", {"moviepy": mock_mpy}):
            results = check_video_python_deps()
        moviepy_result = next(r for r in results if r.name == "moviepy")
        assert moviepy_result.ok is True

    def test_moviepy_absent(self) -> None:
        with patch.dict("sys.modules", {"moviepy": None}):
            results = check_video_python_deps()
        moviepy_result = next(r for r in results if r.name == "moviepy")
        assert moviepy_result.ok is False
        assert moviepy_result.required is False

    def test_pillow_present(self) -> None:
        mock_pil = MagicMock()
        with patch.dict("sys.modules", {"PIL": mock_pil}):
            results = check_video_python_deps()
        pillow_result = next(r for r in results if r.name == "Pillow")
        assert pillow_result.ok is True

    def test_pillow_absent(self) -> None:
        with patch.dict("sys.modules", {"PIL": None}):
            results = check_video_python_deps()
        pillow_result = next(r for r in results if r.name == "Pillow")
        assert pillow_result.ok is False
        assert pillow_result.required is False


# ── _run_all integration ──────────────────────────────────────────────────────

class TestRunAllVideoSection:
    """Smoke-tests that _run_all() includes the video checks."""

    def _names(self) -> list[str]:
        # Patch everything that makes real system calls so this is fast.
        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.check_output", side_effect=Exception),
        ):
            results = _run_all()
        return [r.name for r in results]

    def test_imagemagick_in_run_all(self) -> None:
        assert "imagemagick" in self._names()

    def test_diffusers_in_run_all(self) -> None:
        assert "diffusers" in self._names()

    def test_moviepy_in_run_all(self) -> None:
        assert "moviepy" in self._names()

    def test_pillow_in_run_all(self) -> None:
        assert "Pillow" in self._names()

    def test_video_checks_are_optional(self) -> None:
        """None of the video additions should be hard requirements."""
        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.check_output", side_effect=Exception),
        ):
            results = _run_all()

        video_names = {"imagemagick", "diffusers", "moviepy", "Pillow"}
        for r in results:
            if r.name in video_names:
                assert r.required is False, f"{r.name} should be optional"

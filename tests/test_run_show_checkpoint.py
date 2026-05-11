"""Tests for run_show.py checkpoint helpers (Phase 6) and pre-flight brief validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from vox_romana.cli.runners import _should_use_checkpoint


def _args(checkpoint=None):
    return argparse.Namespace(checkpoint=checkpoint)


class TestShouldUseCheckpoint:
    # ── Manual overrides ────────────────────────────────────────

    def test_explicit_off_always_wins(self):
        assert _should_use_checkpoint(_args(checkpoint=False), "audiobook", 60.0) is False
        assert _should_use_checkpoint(_args(checkpoint=False), "podcast", 30.0) is False

    def test_explicit_on_always_wins(self):
        assert _should_use_checkpoint(_args(checkpoint=True), "debate", 5.0) is True

    # ── Auto-enable rules ───────────────────────────────────────

    def test_audiobook_auto_on_regardless_of_duration(self):
        assert _should_use_checkpoint(_args(), "audiobook", 5.0) is True
        assert _should_use_checkpoint(_args(), "audiobook", None) is True

    def test_long_run_auto_on(self):
        assert _should_use_checkpoint(_args(), "podcast", 30.0) is True
        assert _should_use_checkpoint(_args(), "news", 20.0) is True

    def test_short_run_default_off(self):
        assert _should_use_checkpoint(_args(), "podcast", 10.0) is False
        assert _should_use_checkpoint(_args(), "debate", 5.0) is False

    def test_no_duration_default_off_for_non_audiobook(self):
        assert _should_use_checkpoint(_args(), "podcast", None) is False

    # ── Boundary ────────────────────────────────────────────────

    def test_exactly_twenty_min_enables(self):
        assert _should_use_checkpoint(_args(), "podcast", 20.0) is True

    def test_just_under_twenty_disables(self):
        assert _should_use_checkpoint(_args(), "podcast", 19.99) is False


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Pre-flight brief validation
# ═══════════════════════════════════════════════════════════════════════════

def _make_preflight_args(tmp_path: Path, fact_check: str = "quick") -> argparse.Namespace:
    """Build a minimal args namespace for pre-flight tests."""
    prompt_file = tmp_path / "brief.txt"
    prompt_file.write_text("The Earth is flat and was created in 4004 BC.")
    return argparse.Namespace(
        prompt=prompt_file,
        fact_check=fact_check,
        show_type=None,
        tag=None,
        quiet=True,
        research=None,
        duration=None,
        tolerance=None,
        content_mode=None,
        reference_material=None,
        output=tmp_path / "output",
        checkpoint=None,
        no_log_audio_duration=False,
    )


class TestPreflightBriefValidation:
    """Pre-flight brief validation in main() warns on fabricated claims."""

    def test_preflight_warns_on_fabricated_claim_in_brief(self, tmp_path, capsys):
        """When the fact-checker finds fabrications in the brief, a WARNING is printed."""
        from vox_romana.core.editing.models import FactCheckReport, FactCheckResult

        fabricated = FactCheckResult(
            claim="The Earth is flat",
            verdict="fabricated",
            explanation="The Earth is an oblate spheroid, not flat.",
            source="llm_knowledge",
        )
        report = FactCheckReport(
            content_id="brief",
            mode="quick",
            results=[fabricated],
            summary="One fabricated claim found.",
        )

        mock_checker = MagicMock()
        mock_checker.check_content.return_value = report

        args = _make_preflight_args(tmp_path, fact_check="quick")
        args.output.mkdir(parents=True, exist_ok=True)
        out_file = tmp_path / "show.json"

        with (
            patch("vox_romana.cli.runners.DeepSeekClient"),
            patch("vox_romana.cli.runners.FactChecker") as MockFC,
            patch("vox_romana.cli.runners._RUNNERS", {"news": MagicMock(return_value=out_file)}),
            patch("vox_romana.cli.runners.ShowRouter") as MockRouter,
        ):
            MockFC.build_default.return_value = mock_checker
            router_instance = MockRouter.return_value
            router_instance.classify.return_value = MagicMock(
                show_type="news", confidence=0.9, needs_research=False, reasoning="test"
            )

            from vox_romana.cli.runners import run_generate
            run_generate(args)

        captured = capsys.readouterr()
        assert "PRE-FLIGHT WARNING" in captured.out
        assert "The Earth is flat" in captured.out
        assert "oblate spheroid" in captured.out

    def test_preflight_skipped_when_fact_check_off(self, tmp_path):
        """When --fact-check off, FactChecker.build_default is never called."""
        args = _make_preflight_args(tmp_path, fact_check="off")
        args.output.mkdir(parents=True, exist_ok=True)
        out_file = tmp_path / "show.json"

        mock_fc = MagicMock()

        with (
            patch("vox_romana.cli.runners.DeepSeekClient"),
            patch("vox_romana.cli.runners.FactChecker", mock_fc),
            patch("vox_romana.cli.runners._RUNNERS", {"news": MagicMock(return_value=out_file)}),
            patch("vox_romana.cli.runners.ShowRouter") as MockRouter,
        ):
            router_instance = MockRouter.return_value
            router_instance.classify.return_value = MagicMock(
                show_type="news", confidence=0.9, needs_research=False, reasoning="test"
            )

            from vox_romana.cli.runners import run_generate
            run_generate(args)

        mock_fc.build_default.assert_not_called()

    def test_preflight_no_warning_when_brief_is_accurate(self, tmp_path, capsys):
        """When fact-checker finds no fabrications, no WARNING is printed."""
        from vox_romana.core.editing.models import FactCheckReport, FactCheckResult

        verified = FactCheckResult(
            claim="Water is H2O",
            verdict="verified",
            explanation="Confirmed by llm knowledge.",
            source="llm_knowledge",
        )
        report = FactCheckReport(
            content_id="brief",
            mode="quick",
            results=[verified],
            summary="No fabrications.",
        )

        mock_checker = MagicMock()
        mock_checker.check_content.return_value = report

        args = _make_preflight_args(tmp_path, fact_check="quick")
        args.output.mkdir(parents=True, exist_ok=True)
        out_file = tmp_path / "show.json"

        with (
            patch("vox_romana.cli.runners.DeepSeekClient"),
            patch("vox_romana.cli.runners.FactChecker") as MockFC,
            patch("vox_romana.cli.runners._RUNNERS", {"news": MagicMock(return_value=out_file)}),
            patch("vox_romana.cli.runners.ShowRouter") as MockRouter,
        ):
            MockFC.build_default.return_value = mock_checker
            router_instance = MockRouter.return_value
            router_instance.classify.return_value = MagicMock(
                show_type="news", confidence=0.9, needs_research=False, reasoning="test"
            )

            from vox_romana.cli.runners import run_generate
            run_generate(args)

        captured = capsys.readouterr()
        assert "PRE-FLIGHT WARNING" not in captured.out

    def test_preflight_forwards_perspective_to_fact_checker(self, tmp_path):
        """run_generate forwards args.perspective into brief preflight checking."""
        from vox_romana.core.editing.models import FactCheckReport

        mock_checker = MagicMock()
        mock_checker.check_content.return_value = FactCheckReport(
            content_id="brief",
            mode="quick",
            results=[],
            summary="clean",
        )

        args = _make_preflight_args(tmp_path, fact_check="quick")
        args.perspective = "from Guatemala's geographic perspective"
        args.output.mkdir(parents=True, exist_ok=True)
        out_file = tmp_path / "show.json"

        with (
            patch("vox_romana.cli.runners.DeepSeekClient"),
            patch("vox_romana.cli.runners.FactChecker") as MockFC,
            patch("vox_romana.cli.runners._RUNNERS", {"news": MagicMock(return_value=out_file)}),
            patch("vox_romana.cli.runners.ShowRouter") as MockRouter,
        ):
            MockFC.build_default.return_value = mock_checker
            router_instance = MockRouter.return_value
            router_instance.classify.return_value = MagicMock(
                show_type="news", confidence=0.9, needs_research=False, reasoning="test"
            )

            from vox_romana.cli.runners import run_generate
            run_generate(args)

        assert mock_checker.check_content.call_args.kwargs["perspective"] == (
            "from Guatemala's geographic perspective"
        )

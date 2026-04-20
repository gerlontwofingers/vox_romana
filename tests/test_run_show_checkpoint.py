"""Tests for run_show.py checkpoint helpers (Phase 6)."""

from __future__ import annotations

import argparse

import pytest

from run_show import _should_use_checkpoint


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

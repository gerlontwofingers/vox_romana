"""Tests for vox_romana.core.duration — pure-function helpers and SpeechRateAdvisor."""

from __future__ import annotations

import pytest

from vox_romana.core.duration import (
    WPM_DEFAULTS,
    estimate_total_words,
    get_wpm,
    parse_duration_from_text,
    seconds_to_mmss,
    seconds_to_words,
    tolerance_window,
    words_to_seconds,
    SpeechRateAdvisor,
)


# ── get_wpm ──────────────────────────────────────────────────────────────

class TestGetWpm:
    def test_known_types(self):
        assert get_wpm("debate") == 150
        assert get_wpm("news") == 165
        assert get_wpm("podcast") == 150
        assert get_wpm("audiobook") == 140

    def test_unknown_falls_back(self):
        assert get_wpm("unknown_type") == 150


# ── words_to_seconds / seconds_to_words ──────────────────────────────────

class TestConversions:
    def test_words_to_seconds_default(self):
        # 150 words at 150 wpm = 60 seconds
        assert words_to_seconds(150, content_type="debate") == pytest.approx(60.0)

    def test_words_to_seconds_explicit_wpm(self):
        assert words_to_seconds(300, wpm=100) == pytest.approx(180.0)

    def test_seconds_to_words_roundtrip(self):
        words = seconds_to_words(120.0, content_type="news")
        # 2 minutes at 165 wpm = 330 words
        assert words == 330

    def test_zero_words(self):
        assert words_to_seconds(0) == 0.0


# ── seconds_to_mmss ──────────────────────────────────────────────────────

class TestSecondsToMmss:
    def test_exact_minutes(self):
        assert seconds_to_mmss(1800) == "30:00"

    def test_with_seconds(self):
        assert seconds_to_mmss(95) == "1:35"

    def test_zero(self):
        assert seconds_to_mmss(0) == "0:00"


# ── estimate_total_words ─────────────────────────────────────────────────

class TestEstimateTotalWords:
    def test_30_min_debate(self):
        assert estimate_total_words(30, "debate") == 4500

    def test_custom_wpm(self):
        assert estimate_total_words(10, wpm=200) == 2000


# ── tolerance_window ─────────────────────────────────────────────────────

class TestToleranceWindow:
    def test_default_10_percent(self):
        lo, hi = tolerance_window(1000)
        assert lo == 900
        assert hi == 1100

    def test_custom_percent(self):
        lo, hi = tolerance_window(1000, tolerance_pct=5.0)
        assert lo == 950
        assert hi == 1050

    def test_zero_tolerance(self):
        lo, hi = tolerance_window(500, tolerance_pct=0.0)
        assert lo == 500
        assert hi == 500


# ── parse_duration_from_text ─────────────────────────────────────────────

class TestParseDurationFromText:
    def test_minutes(self):
        assert parse_duration_from_text("a 30 minute debate") == 30.0

    def test_mins_abbreviation(self):
        assert parse_duration_from_text("45 min podcast") == 45.0

    def test_hours(self):
        assert parse_duration_from_text("a 2 hour audiobook") == 120.0

    def test_hours_and_minutes(self):
        assert parse_duration_from_text("1 hour 30 minutes") == 90.0

    def test_fractional_hours(self):
        assert parse_duration_from_text("1.5 hour show") == 90.0

    def test_hyphenated(self):
        assert parse_duration_from_text("a 20-minute episode") == 20.0

    def test_no_duration(self):
        assert parse_duration_from_text("discuss AI ethics") is None


# ── SpeechRateAdvisor ────────────────────────────────────────────────────

class TestSpeechRateAdvisor:
    def test_without_catalog_uses_defaults(self):
        advisor = SpeechRateAdvisor()
        assert advisor.effective_wpm("debate") == 150
        assert advisor.effective_wpm("news") == 165

    def test_estimate_words(self):
        advisor = SpeechRateAdvisor()
        assert advisor.estimate_words(30, "debate") == 4500

    def test_tolerance_window(self):
        advisor = SpeechRateAdvisor()
        lo, hi = advisor.tolerance_window(1000, tolerance_pct=10.0)
        assert lo == 900
        assert hi == 1100


# ── BaseConductor budget helpers ─────────────────────────────────────────

from vox_romana.core.config import ContentConfig
from vox_romana.core.conductor import BaseConductor, ConductorResult


class _StubConfig(ContentConfig):
    """Minimal config for testing BaseConductor helpers."""

    content_type: str = "debate"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _StubConductor(BaseConductor):
    """Concrete subclass that stubs abstract methods."""

    def _build_agents(self) -> None:
        pass

    def _run_turn_loop(self) -> list[dict]:
        return []

    def _package_result(self, raw_events: list[dict]) -> ConductorResult:
        return ConductorResult(config=self.config, events=raw_events)


def _make_conductor(target_words: int | None = None, words_generated: int = 0):
    """Create a stub conductor with pre-set word counters."""
    cfg = _StubConfig(content_type="debate")
    if target_words is not None:
        cfg.target_words = target_words
    cond = _StubConductor(cfg)
    cond._words_generated = words_generated
    return cond


class TestBudgetStatusForPrompt:
    def test_returns_empty_when_no_target(self):
        cond = _make_conductor(target_words=None)
        assert cond._budget_status_for_prompt(5) == ""

    def test_returns_status_when_target_set(self):
        cond = _make_conductor(target_words=5000, words_generated=3000)
        result = cond._budget_status_for_prompt(4)
        assert "BUDGET STATUS:" in result
        assert "3,000" in result
        assert "5,000" in result
        assert "60%" in result
        assert "2,000" in result
        assert "4 step(s)" in result

    def test_behind_pace_warning(self):
        # 10% done (500/5000) but 8 of 10 steps completed → expected 80%
        cond = _make_conductor(target_words=5000, words_generated=500)
        result = cond._budget_status_for_prompt(2, total_steps=10)
        assert "BEHIND PACE" in result

    def test_no_warning_when_on_track(self):
        # 80% done with 1 step remaining out of 5 total — on track
        cond = _make_conductor(target_words=5000, words_generated=4000)
        result = cond._budget_status_for_prompt(1, total_steps=5)
        assert "BEHIND PACE" not in result

    def test_no_warning_without_total_steps(self):
        # Without total_steps, behind-pace detection is skipped
        cond = _make_conductor(target_words=5000, words_generated=500)
        result = cond._budget_status_for_prompt(2)
        assert "BEHIND PACE" not in result

    def test_zero_remaining_steps(self):
        cond = _make_conductor(target_words=5000, words_generated=5000)
        result = cond._budget_status_for_prompt(0)
        assert "BUDGET STATUS:" in result


class TestRebalancedTarget:
    def test_returns_planned_when_no_target(self):
        cond = _make_conductor(target_words=None)
        assert cond._rebalanced_target(400, 5) == 400

    def test_returns_planned_when_on_track(self):
        # 2500/5000 generated, 5 remaining → fair share = 500
        # planned=600 > 500, so planned wins
        cond = _make_conductor(target_words=5000, words_generated=2500)
        assert cond._rebalanced_target(600, 5) == 600

    def test_increases_when_behind(self):
        # 1000/5000 generated, 4 remaining → fair share = 1000
        # planned=400 < 1000, so fair share wins
        cond = _make_conductor(target_words=5000, words_generated=1000)
        assert cond._rebalanced_target(400, 4) == 1000

    def test_zero_remaining_steps(self):
        cond = _make_conductor(target_words=5000, words_generated=3000)
        assert cond._rebalanced_target(400, 0) == 400

    def test_never_shrinks_planned(self):
        # Fully on track: fair share < planned → should return planned
        cond = _make_conductor(target_words=5000, words_generated=4500)
        # remaining=500, 5 steps → fair share=100. planned=400 wins.
        assert cond._rebalanced_target(400, 5) == 400

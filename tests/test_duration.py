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
        assert get_wpm("audiobook") == 155

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

    # ── Word-form number support ──────────────────────────────────────

    def test_word_hours_three(self):
        assert parse_duration_from_text("three hour podcast") == 180.0

    def test_word_hours_two(self):
        assert parse_duration_from_text("a two hour documentary") == 120.0

    def test_word_minutes_twelve(self):
        assert parse_duration_from_text("a twelve minute segment") == 12.0

    def test_word_hours_and_minutes(self):
        assert parse_duration_from_text("two hour thirty minute show") == 150.0

    def test_word_hours_uppercase(self):
        assert parse_duration_from_text("THREE HOUR SHOW") == 180.0

    def test_word_hours_mixed_case(self):
        assert parse_duration_from_text("make me a Three Hour podcast") == 180.0


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


# ── Hard length contract helpers (Phase 1+2) ─────────────────────


class TestLengthContractBlock:
    def test_empty_when_target_zero(self):
        assert BaseConductor._length_contract_block(0) == ""

    def test_empty_when_target_negative(self):
        assert BaseConductor._length_contract_block(-10) == ""

    def test_includes_hard_range(self):
        block = BaseConductor._length_contract_block(100, tolerance_pct=10.0)
        assert "between 90 and 110 words" in block
        assert "target: 100" in block
        assert "HARD" in block.upper()

    def test_default_includes_word_count_clause(self):
        block = BaseConductor._length_contract_block(100)
        assert "word_count" in block

    def test_word_count_clause_optional(self):
        block = BaseConductor._length_contract_block(
            100, require_word_count=False,
        )
        assert "word_count" not in block

    def test_tolerance_pct_widens_range(self):
        block = BaseConductor._length_contract_block(100, tolerance_pct=20.0)
        assert "between 80 and 120 words" in block

    def test_tiny_target_still_yields_at_least_one_margin(self):
        block = BaseConductor._length_contract_block(5, tolerance_pct=10.0)
        # 5 * 0.1 = 0.5 → margin floored to 1 → range [4, 6]
        assert "between 4 and 6 words" in block


class TestValidateWordCount:
    def test_returns_none_when_in_range(self):
        result = BaseConductor._validate_word_count(
            declared=100, actual=100, target_words=100, tolerance_pct=10.0,
        )
        assert result is None

    def test_returns_none_when_at_lower_bound(self):
        result = BaseConductor._validate_word_count(
            declared=90, actual=90, target_words=100, tolerance_pct=10.0,
        )
        assert result is None

    def test_returns_none_when_at_upper_bound(self):
        result = BaseConductor._validate_word_count(
            declared=110, actual=110, target_words=100, tolerance_pct=10.0,
        )
        assert result is None

    def test_returns_warning_when_below(self):
        result = BaseConductor._validate_word_count(
            declared=80, actual=80, target_words=100, tolerance_pct=10.0,
        )
        assert result is not None
        assert "actual=80" in result
        assert "target=100" in result

    def test_returns_warning_when_above(self):
        result = BaseConductor._validate_word_count(
            declared=120, actual=120, target_words=100, tolerance_pct=10.0,
        )
        assert result is not None
        assert "120" in result

    def test_flags_declared_vs_actual_mismatch(self):
        # Model lied about its word_count
        result = BaseConductor._validate_word_count(
            declared=100, actual=60, target_words=100, tolerance_pct=10.0,
        )
        assert result is not None
        assert "declared word_count=100" in result
        assert "actual=60" in result

    def test_returns_none_when_target_zero(self):
        result = BaseConductor._validate_word_count(
            declared=100, actual=100, target_words=0,
        )
        assert result is None

    def test_handles_none_declared(self):
        result = BaseConductor._validate_word_count(
            declared=None, actual=80, target_words=100, tolerance_pct=10.0,
        )
        assert result is not None
        assert "declared" not in result  # No declared/actual annotation


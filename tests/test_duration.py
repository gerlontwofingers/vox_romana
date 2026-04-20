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

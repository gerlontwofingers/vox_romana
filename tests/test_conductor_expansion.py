"""Tests for BaseConductor's expansion-pass plumbing (Phase 7)."""

from __future__ import annotations

import pytest

from vox_romana.core.budget_tracker import BudgetTracker
from vox_romana.core.conductor import BaseConductor, ConductorResult
from vox_romana.core.config import ContentConfig


class _StubConfig(ContentConfig):
    content_type: str = "podcast"


class _StubConductor(BaseConductor):
    """Captures expansion calls; returns user-defined extra events."""

    def __init__(self, *args, expansion_events=None, **kwargs):
        self.expansion_events = expansion_events or []
        self.expansion_calls = []
        super().__init__(*args, **kwargs)

    def _build_agents(self):
        pass

    def _run_turn_loop(self):
        return [{"text": "primary"}]

    def _package_result(self, raw_events):
        return ConductorResult(config=self.config, events=raw_events)

    def _expansion_pass(self, *, deficit_words, deficit_seconds):
        self.expansion_calls.append((deficit_words, deficit_seconds))
        return list(self.expansion_events)


def _make(*, target_words=None, words_generated=0,
          tracker=None, expansion_events=None):
    cfg = _StubConfig(content_type="podcast")
    if target_words is not None:
        cfg.target_words = target_words
    cond = _StubConductor(
        cfg,
        budget_tracker=tracker,
        expansion_events=expansion_events,
    )
    cond._words_generated = words_generated
    return cond


# ── No expansion path ────────────────────────────────────────────


class TestNoExpansion:
    def test_no_target_no_tracker_skips(self):
        cond = _make()
        result = cond.run()
        assert cond.expansion_calls == []
        # Only the primary event survives.
        assert len(result.events) == 1

    def test_on_target_skips(self):
        # 1000 words generated, 1000 target → no shortfall.
        cond = _make(target_words=1000, words_generated=1000,
                     expansion_events=[{"text": "extra"}])
        cond.run()
        assert cond.expansion_calls == []

    def test_within_tolerance_skips(self):
        # 950 / 1000 (5% short) → still inside 10% default tolerance.
        cond = _make(target_words=1000, words_generated=950,
                     expansion_events=[{"text": "extra"}])
        cond.run()
        assert cond.expansion_calls == []


# ── Expansion driven by word counter ─────────────────────────────


class TestWordDrivenExpansion:
    def test_short_below_word_lo_triggers(self):
        # 800 / 1000 → below 90% lower bound (default 10%)
        cond = _make(target_words=1000, words_generated=800,
                     expansion_events=[{"text": "extra"}])
        result = cond.run()
        assert len(cond.expansion_calls) == 1
        deficit_words, _ = cond.expansion_calls[0]
        # 1000 * (1 - 0.10) = 900 → deficit at least 100
        assert deficit_words >= 100
        assert result.events[-1]["text"] == "extra"


# ── Expansion driven by budget tracker ───────────────────────────


class TestTrackerDrivenExpansion:
    def test_tracker_short_triggers(self):
        # Total 600s / 10 segments. Three segments came in at 30s each
        # vs 60s targets → 90s short. Tolerance is 5% of 600 = 30s.
        tracker = BudgetTracker(600.0, 10)
        for i in range(3):
            tracker.record_segment(i, actual_seconds=30.0)
        cond = _make(target_words=1500, words_generated=750,
                     tracker=tracker,
                     expansion_events=[{"text": "extra"}])
        cond.run()
        assert len(cond.expansion_calls) == 1
        _, deficit_seconds = cond.expansion_calls[0]
        assert deficit_seconds == pytest.approx(90.0, abs=1.0)

    def test_tracker_on_pace_skips(self):
        tracker = BudgetTracker(600.0, 10)
        for i in range(3):
            tracker.record_segment(i, actual_seconds=60.0)
        cond = _make(tracker=tracker,
                     expansion_events=[{"text": "extra"}])
        cond.run()
        assert cond.expansion_calls == []


# ── Failure handling ─────────────────────────────────────────────


class TestExpansionFailures:
    def test_expansion_exception_recorded_and_run_continues(self):
        cond = _make(target_words=1000, words_generated=500)

        def boom(*, deficit_words, deficit_seconds):
            raise RuntimeError("nope")

        cond._expansion_pass = boom
        result = cond.run()
        # Run completed and no extra events appended.
        assert len(result.events) == 1
        assert any("expansion" in w for w in cond._length_warnings)

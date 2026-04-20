"""Tests for vox_romana.core.budget_tracker — pure-function rebalancer."""

from __future__ import annotations

import pytest

from vox_romana.core.budget_tracker import BudgetTracker, SegmentRecord


# ── Construction ──────────────────────────────────────────────────


class TestConstruction:
    def test_basic_construction(self):
        t = BudgetTracker(total_target_seconds=600.0, num_segments=10)
        assert t.total_target_seconds == 600.0
        assert t.num_segments == 10
        assert t.segments_done == 0
        assert t.records == []

    def test_zero_segments_rejected(self):
        with pytest.raises(ValueError):
            BudgetTracker(total_target_seconds=600.0, num_segments=0)

    def test_negative_segments_rejected(self):
        with pytest.raises(ValueError):
            BudgetTracker(total_target_seconds=600.0, num_segments=-1)

    def test_zero_total_rejected(self):
        with pytest.raises(ValueError):
            BudgetTracker(total_target_seconds=0, num_segments=10)


# ── Even split with no records ─────────────────────────────────────


class TestInitialTargets:
    def test_first_segment_gets_fair_share(self):
        t = BudgetTracker(1800.0, 10)
        assert t.segment_target_seconds(0) == pytest.approx(180.0)

    def test_all_initial_targets_equal_when_recorded_at_pace(self):
        # If every segment hits its rebalanced target exactly, all targets
        # equal the original fair share.
        t = BudgetTracker(1800.0, 10)
        observed = []
        for i in range(10):
            tgt = t.segment_target_seconds(i)
            observed.append(tgt)
            t.record_segment(i, actual_seconds=tgt, target_seconds=tgt)
        for v in observed:
            assert v == pytest.approx(180.0)

    def test_out_of_range_index_raises(self):
        t = BudgetTracker(600.0, 5)
        with pytest.raises(IndexError):
            t.segment_target_seconds(5)
        with pytest.raises(IndexError):
            t.segment_target_seconds(-1)


# ── Rebalancing after recording ────────────────────────────────────


class TestRebalancing:
    def test_short_segment_grows_remaining(self):
        # 10 segments × 60s each = 600s total. First segment runs 30s short.
        t = BudgetTracker(600.0, 10)
        t.record_segment(0, actual_seconds=30.0, target_seconds=60.0)
        # Remaining 9 segments should now share 570s = 63.33s each.
        assert t.segment_target_seconds(1) == pytest.approx(570.0 / 9)

    def test_long_segment_shrinks_remaining(self):
        t = BudgetTracker(600.0, 10)
        t.record_segment(0, actual_seconds=90.0, target_seconds=60.0)
        # Remaining 9 segments share 510s.
        assert t.segment_target_seconds(1) == pytest.approx(510.0 / 9)

    def test_perfect_segment_keeps_target_steady(self):
        t = BudgetTracker(600.0, 10)
        t.record_segment(0, actual_seconds=60.0, target_seconds=60.0)
        assert t.segment_target_seconds(1) == pytest.approx(60.0)

    def test_drift_accumulates(self):
        t = BudgetTracker(600.0, 10)
        # Three back-to-back 30s shorts ⇒ 90s deficit.
        for i in range(3):
            t.record_segment(i, actual_seconds=30.0)
        # Remaining 7 segments share 600 - 90 = 510s.
        expected = 510.0 / 7
        assert t.segment_target_seconds(3) == pytest.approx(expected)

    def test_overrun_clamps_remaining_budget_to_zero(self):
        # If we blow the entire budget on one segment, remaining = 0.
        t = BudgetTracker(600.0, 10)
        t.record_segment(0, actual_seconds=700.0, target_seconds=60.0)
        assert t.remaining_budget_seconds() == 0.0
        # Subsequent targets are 0 (rather than negative).
        assert t.segment_target_seconds(1) == pytest.approx(0.0)


# ── Cumulative delta ───────────────────────────────────────────────


class TestCumulativeDelta:
    def test_delta_zero_when_no_records(self):
        t = BudgetTracker(600.0, 10)
        assert t.cumulative_delta() == pytest.approx(0.0)

    def test_negative_delta_when_short(self):
        t = BudgetTracker(600.0, 10)
        t.record_segment(0, actual_seconds=30.0, target_seconds=60.0)
        # Delta = 30 - 60 = -30
        assert t.cumulative_delta() == pytest.approx(-30.0)

    def test_positive_delta_when_long(self):
        t = BudgetTracker(600.0, 10)
        t.record_segment(0, actual_seconds=90.0, target_seconds=60.0)
        assert t.cumulative_delta() == pytest.approx(30.0)

    def test_delta_accumulates_against_original_plan(self):
        # Delta uses the *original* fair share (60s/segment), not the
        # rebalanced moving target. This way 'delta' reflects total drift.
        t = BudgetTracker(600.0, 10)
        t.record_segment(0, actual_seconds=30.0)
        t.record_segment(1, actual_seconds=30.0)
        # Two segments planned at 60s = 120s, actual = 60s, delta = -60s.
        assert t.cumulative_delta() == pytest.approx(-60.0)


# ── Predicates ─────────────────────────────────────────────────────


class TestPredicates:
    def test_is_short_by_true_when_deficit_exceeds_threshold(self):
        t = BudgetTracker(600.0, 10)
        t.record_segment(0, actual_seconds=10.0, target_seconds=60.0)
        # delta = -50
        assert t.is_short_by(30.0) is True
        assert t.is_short_by(50.0) is True
        assert t.is_short_by(60.0) is False

    def test_is_long_by_true_when_overage_exceeds_threshold(self):
        t = BudgetTracker(600.0, 10)
        t.record_segment(0, actual_seconds=120.0, target_seconds=60.0)
        # delta = +60
        assert t.is_long_by(30.0) is True
        assert t.is_long_by(60.0) is True
        assert t.is_long_by(90.0) is False


# ── Records and segments_remaining ────────────────────────────────


class TestRecords:
    def test_segment_record_delta(self):
        rec = SegmentRecord(index=0, target_seconds=60.0, actual_seconds=45.0)
        assert rec.delta == pytest.approx(-15.0)

    def test_records_returned_as_copy(self):
        t = BudgetTracker(600.0, 10)
        t.record_segment(0, actual_seconds=60.0)
        recs = t.records
        recs.clear()
        # Original list should still have the record.
        assert len(t.records) == 1

    def test_segments_remaining_decreases(self):
        t = BudgetTracker(600.0, 10)
        assert t.segments_remaining() == 10
        t.record_segment(0, actual_seconds=60.0)
        assert t.segments_remaining() == 9

    def test_negative_actual_rejected(self):
        t = BudgetTracker(600.0, 10)
        with pytest.raises(ValueError):
            t.record_segment(0, actual_seconds=-5.0)


# ── Status report formatting ──────────────────────────────────────


class TestStatusReport:
    def test_status_includes_totals_and_delta(self):
        t = BudgetTracker(1800.0, 10)
        t.record_segment(0, actual_seconds=150.0, target_seconds=180.0)
        report = t.status_report()
        assert "2:30" in report  # 150s
        assert "30:00" in report  # 1800s
        assert "9 segment" in report

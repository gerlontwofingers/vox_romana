"""BudgetTracker — dynamic time-budget rebalancing across segments.

Inspired by the broadcast producer's clipboard. A producer back-times a
broadcast: if the show ends at 11:00:00 sharp and segment 3 ran 30 seconds
short, segments 4 and onward each absorb a portion of that 30 seconds.

The tracker doesn't know anything about words, voices, or conductors. It
only knows about *seconds* and *segments remaining*. Word-budget
calculations live in the conductor (which knows the WPM model).

USAGE:

    tracker = BudgetTracker(total_target_seconds=1800, num_segments=10)
    for i in range(10):
        target = tracker.segment_target_seconds(i)
        # ... generate + render the segment ...
        tracker.record_segment(i, actual_seconds=measured_duration)

    print(tracker.status_report())
    # → "Total: 25:30 / 30:00. Delta: -4:30. 2 segments remain."

DESIGN NOTES:
    * `record_segment` accepts the actual seconds *and* (optionally) the
      original target so the tracker can keep history for diagnostics.
    * `segment_target_seconds` always re-derives from current state
      (cumulative_actual + cumulative_remaining_target). It never returns
      stale values — call it just before generating each segment.
    * `cumulative_delta` is `actual_total - target_total_for_segments_done`.
      Negative means we're behind, positive means we're ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .duration import seconds_to_mmss


@dataclass
class SegmentRecord:
    """One row of the tracker's history."""

    index: int
    target_seconds: float
    actual_seconds: float

    @property
    def delta(self) -> float:
        """Actual - target. Negative = short, positive = long."""
        return self.actual_seconds - self.target_seconds


class BudgetTracker:
    """Rebalances per-segment time budgets as actuals come in.

    Constructor:
        total_target_seconds: The hard target for the entire show.
        num_segments:         Total number of segments planned.

    Behavior:
        Initially each segment gets ``total_target_seconds / num_segments``.
        Once a segment is recorded, remaining segments get
        ``(total_target - cumulative_actual) / segments_left``.
    """

    def __init__(self, total_target_seconds: float, num_segments: int) -> None:
        if num_segments <= 0:
            raise ValueError(f"num_segments must be > 0, got {num_segments}")
        if total_target_seconds <= 0:
            raise ValueError(
                f"total_target_seconds must be > 0, got {total_target_seconds}"
            )
        self._total_target: float = float(total_target_seconds)
        self._num_segments: int = int(num_segments)
        self._records: list[SegmentRecord] = []

    # ── Public API ──────────────────────────────────────────────────

    @property
    def total_target_seconds(self) -> float:
        return self._total_target

    @property
    def num_segments(self) -> int:
        return self._num_segments

    @property
    def records(self) -> list[SegmentRecord]:
        return list(self._records)

    @property
    def cumulative_actual_seconds(self) -> float:
        return sum(r.actual_seconds for r in self._records)

    @property
    def segments_done(self) -> int:
        return len(self._records)

    def segments_remaining(self, current_index: int | None = None) -> int:
        """How many segments still to come (inclusive of current)."""
        if current_index is None:
            return max(0, self._num_segments - self.segments_done)
        return max(0, self._num_segments - current_index)

    def cumulative_delta(self) -> float:
        """Actual - planned-target-for-completed-segments.

        Negative means we are short on time so far; positive means long.
        Uses the *original* fair-share target as the baseline, not the
        rebalanced target — that way 'delta' tells you cumulative drift
        from the original plan, not the moving rebalanced one.
        """
        fair_share = self._total_target / self._num_segments
        planned = fair_share * self.segments_done
        return self.cumulative_actual_seconds - planned

    def remaining_budget_seconds(self) -> float:
        """How many seconds are left in the total budget."""
        return max(0.0, self._total_target - self.cumulative_actual_seconds)

    def segment_target_seconds(self, current_index: int) -> float:
        """Rebalanced target for the segment about to be generated.

        Args:
            current_index: 0-based index of the segment we're about to do.

        Returns:
            Target seconds for this single segment, factoring in any drift.
        """
        if current_index < 0 or current_index >= self._num_segments:
            raise IndexError(
                f"current_index {current_index} out of range "
                f"[0, {self._num_segments})"
            )
        remaining = self._num_segments - current_index
        if remaining <= 0:
            return 0.0
        return self.remaining_budget_seconds() / remaining

    def record_segment(
        self,
        index: int,
        actual_seconds: float,
        target_seconds: float | None = None,
    ) -> None:
        """Record the measured duration of a completed segment.

        Args:
            index:          0-based segment index.
            actual_seconds: Measured TTS duration of the segment.
            target_seconds: Original target for this segment (for diagnostics).
                            If omitted, derived from current rebalance state.
        """
        if actual_seconds < 0:
            raise ValueError(
                f"actual_seconds must be >= 0, got {actual_seconds}"
            )
        if target_seconds is None:
            target_seconds = self.segment_target_seconds(index)
        self._records.append(
            SegmentRecord(
                index=index,
                target_seconds=float(target_seconds),
                actual_seconds=float(actual_seconds),
            )
        )

    def status_report(self) -> str:
        """Human-readable status line for logging."""
        total_actual = self.cumulative_actual_seconds
        delta = self.cumulative_delta()
        sign = "+" if delta >= 0 else "-"
        parts = [
            f"Total: {seconds_to_mmss(total_actual)} / "
            f"{seconds_to_mmss(self._total_target)}",
            f"Delta: {sign}{seconds_to_mmss(abs(delta))}",
            f"{self.segments_remaining()} segment(s) remain",
        ]
        return ". ".join(parts) + "."

    # ── Convenience predicates for retry / expansion logic ─────────

    def is_short_by(self, threshold_seconds: float) -> bool:
        """True when we are at least ``threshold_seconds`` behind plan."""
        return self.cumulative_delta() <= -abs(threshold_seconds)

    def is_long_by(self, threshold_seconds: float) -> bool:
        """True when we are at least ``threshold_seconds`` ahead of plan."""
        return self.cumulative_delta() >= abs(threshold_seconds)

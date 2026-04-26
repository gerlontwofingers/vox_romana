"""Tests for TimelineSync — scene timestamp computation from TTS durations.

These tests exercise pure computation with no mocking required.
TimelineSync is stateless and depends only on AnimationPlan / SceneSpec.
"""

from __future__ import annotations

import pytest

from vox_romana.animation.models import AnimationPlan, SceneSpec
from vox_romana.animation.sync import DEFAULT_FALLBACK_DURATION_SEC, TimelineSync


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_plan(*scene_event_groups: list[int]) -> AnimationPlan:
    """Create an AnimationPlan with one SceneSpec per event group.

    Args:
        *scene_event_groups: Each positional arg is a list of global_orders
            for one scene. E.g. _make_plan([0, 1], [2, 3]) → 2 scenes.
    """
    scenes = [
        SceneSpec(
            scene_id=f"scene_{idx:02d}",
            event_global_orders=list(orders),
        )
        for idx, orders in enumerate(scene_event_groups)
    ]
    return AnimationPlan(
        plan_id="test-plan",
        brief="test brief",
        style="cinematic",
        color_palette=[],
        scenes=scenes,
        render_backend="moviepy",
        resolution=(1920, 1080),
        fps=30,
        image_model="stabilityai/stable-diffusion-xl-base-1.0",
    )


# ── Basic sync ─────────────────────────────────────────────────────────────

class TestSyncBasic:
    def test_returns_same_plan_object(self):
        """sync() mutates in place and returns the same plan."""
        plan = _make_plan([0])
        result = TimelineSync.sync(plan, {0: 2.5})
        assert result is plan

    def test_single_scene_single_event(self):
        """One scene, one event: starts at 0, ends at event duration."""
        plan = _make_plan([0])
        TimelineSync.sync(plan, {0: 3.0})

        assert plan.scenes[0].start_time_sec == pytest.approx(0.0)
        assert plan.scenes[0].end_time_sec == pytest.approx(3.0)

    def test_single_scene_multiple_events(self):
        """Scene covers events 0+1+2: end = sum of their durations."""
        plan = _make_plan([0, 1, 2])
        TimelineSync.sync(plan, {0: 2.5, 1: 4.1, 2: 3.8})

        assert plan.scenes[0].start_time_sec == pytest.approx(0.0)
        assert plan.scenes[0].end_time_sec == pytest.approx(2.5 + 4.1 + 3.8)

    def test_two_scenes_no_overlap(self):
        """Two sequential scenes: second starts where first ends."""
        plan = _make_plan([0, 1], [2, 3])
        event_durations = {0: 2.5, 1: 4.1, 2: 3.8, 3: 2.0}
        TimelineSync.sync(plan, event_durations)

        # Scene 0: events 0 + 1 = 2.5 + 4.1 = 6.6
        assert plan.scenes[0].start_time_sec == pytest.approx(0.0)
        assert plan.scenes[0].end_time_sec == pytest.approx(6.6)

        # Scene 1: events 2 + 3 → starts after event 1 ends
        assert plan.scenes[1].start_time_sec == pytest.approx(6.6)
        assert plan.scenes[1].end_time_sec == pytest.approx(6.6 + 3.8 + 2.0)

    def test_three_scenes_sequential(self):
        """Three scenes: cumulative timestamps chain correctly."""
        plan = _make_plan([0], [1], [2])
        event_durations = {0: 1.0, 1: 2.0, 2: 3.0}
        TimelineSync.sync(plan, event_durations)

        assert plan.scenes[0].start_time_sec == pytest.approx(0.0)
        assert plan.scenes[0].end_time_sec == pytest.approx(1.0)

        assert plan.scenes[1].start_time_sec == pytest.approx(1.0)
        assert plan.scenes[1].end_time_sec == pytest.approx(3.0)

        assert plan.scenes[2].start_time_sec == pytest.approx(3.0)
        assert plan.scenes[2].end_time_sec == pytest.approx(6.0)


# ── is_synced property ─────────────────────────────────────────────────────

class TestIsSyncedProperty:
    def test_is_synced_true_after_sync(self):
        """plan.is_synced becomes True after sync with complete durations."""
        plan = _make_plan([0, 1], [2])
        assert not plan.is_synced  # pre-sync: None timestamps
        TimelineSync.sync(plan, {0: 1.0, 1: 2.0, 2: 1.5})
        assert plan.is_synced

    def test_is_synced_false_without_sync(self):
        """plan.is_synced is False when timestamps are still None."""
        plan = _make_plan([0])
        assert not plan.is_synced


# ── Missing / partial event_durations ─────────────────────────────────────

class TestMissingEventDurations:
    def test_missing_events_default_to_zero(self):
        """Events not in event_durations contribute 0.0 duration."""
        plan = _make_plan([0, 1])
        # Event 1 missing from event_durations
        TimelineSync.sync(plan, {0: 3.0})

        # start = 0, end = duration of event 0 (event 1 has 0 duration)
        assert plan.scenes[0].start_time_sec == pytest.approx(0.0)
        assert plan.scenes[0].end_time_sec == pytest.approx(3.0)

    def test_all_events_missing_uses_fallback(self):
        """Scene with no known durations gets DEFAULT_FALLBACK_DURATION_SEC."""
        plan = _make_plan([99, 100])  # events not in durations dict
        TimelineSync.sync(plan, {})  # empty durations

        scene = plan.scenes[0]
        assert scene.start_time_sec == pytest.approx(0.0)
        assert scene.end_time_sec == pytest.approx(
            0.0 + DEFAULT_FALLBACK_DURATION_SEC
        )

    def test_partial_events_partial_duration(self):
        """Only some events have durations; missing ones are 0.0."""
        plan = _make_plan([0, 1, 2], [3])
        # Only 0, 2, 3 provided; 1 defaults to 0.
        TimelineSync.sync(plan, {0: 2.0, 2: 1.5, 3: 4.0})

        # Scene 0 covers events 0, 1, 2 → 2.0 + 0.0 + 1.5 = 3.5
        assert plan.scenes[0].start_time_sec == pytest.approx(0.0)
        assert plan.scenes[0].end_time_sec == pytest.approx(3.5)

        # Scene 1 covers event 3 → starts at 3.5, ends at 7.5
        assert plan.scenes[1].start_time_sec == pytest.approx(3.5)
        assert plan.scenes[1].end_time_sec == pytest.approx(7.5)

    def test_empty_event_durations_dict(self):
        """Empty dict → all events contribute 0, scenes get fallback."""
        plan = _make_plan([0], [1])
        TimelineSync.sync(plan, {})

        for scene in plan.scenes:
            assert scene.duration_sec == pytest.approx(DEFAULT_FALLBACK_DURATION_SEC)


# ── Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_plan_is_noop(self):
        """An AnimationPlan with no scenes returns without error."""
        plan = AnimationPlan(
            plan_id="test-plan",
            brief="test brief",
            style="cinematic",
            color_palette=[],
            scenes=[],
            render_backend="moviepy",
            resolution=(1920, 1080),
            fps=30,
        )
        result = TimelineSync.sync(plan, {0: 1.0})
        assert result is plan
        assert result.scenes == []

    def test_scene_with_no_event_orders_gets_fallback(self):
        """Scene with empty event_global_orders gets fallback duration."""
        plan = _make_plan([])  # no events assigned to scene
        TimelineSync.sync(plan, {0: 5.0})

        scene = plan.scenes[0]
        # Should not crash, and scene gets a non-zero duration
        assert scene.end_time_sec is not None
        assert scene.end_time_sec > (scene.start_time_sec or 0.0)

    def test_extra_events_in_durations_not_in_scenes(self):
        """Extra events in event_durations that aren't in any scene are included
        in the running cursor, shifting later scenes correctly."""
        # Events 0, 1, 2 in durations; only event 2 assigned to a scene.
        # Events 0 and 1 still consume audio time.
        plan = _make_plan([2])
        TimelineSync.sync(plan, {0: 1.0, 1: 2.0, 2: 3.0})

        # Scene should start AFTER events 0 and 1
        assert plan.scenes[0].start_time_sec == pytest.approx(3.0)
        assert plan.scenes[0].end_time_sec == pytest.approx(6.0)

    def test_single_scene_fractional_durations(self):
        """Floating-point durations accumulate without precision loss."""
        plan = _make_plan([0, 1, 2, 3])
        TimelineSync.sync(plan, {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4})

        assert plan.scenes[0].start_time_sec == pytest.approx(0.0)
        assert plan.scenes[0].end_time_sec == pytest.approx(1.0)

    def test_single_event_zero_duration_gets_fallback(self):
        """A scene whose sole event has 0.0 duration gets the fallback."""
        plan = _make_plan([0])
        TimelineSync.sync(plan, {0: 0.0})

        scene = plan.scenes[0]
        assert scene.end_time_sec == pytest.approx(DEFAULT_FALLBACK_DURATION_SEC)

    def test_sync_is_idempotent(self):
        """Calling sync twice overwrites previous timestamps cleanly."""
        plan = _make_plan([0, 1])
        TimelineSync.sync(plan, {0: 1.0, 1: 2.0})
        first_end = plan.scenes[0].end_time_sec

        # Second call with different durations should overwrite
        TimelineSync.sync(plan, {0: 5.0, 1: 5.0})
        assert plan.scenes[0].end_time_sec != pytest.approx(first_end)
        assert plan.scenes[0].end_time_sec == pytest.approx(10.0)


# ── Many events ────────────────────────────────────────────────────────────

class TestManyEvents:
    def test_large_plan_cumulative_sum(self):
        """20 events across 4 scenes: cumulative sum is correct throughout."""
        # Events 0-4 → scene 0, 5-9 → scene 1, 10-14 → scene 2, 15-19 → scene 3
        plan = _make_plan(
            list(range(0, 5)),
            list(range(5, 10)),
            list(range(10, 15)),
            list(range(15, 20)),
        )
        durations = {i: 1.0 for i in range(20)}  # 1s per event
        TimelineSync.sync(plan, durations)

        assert plan.scenes[0].start_time_sec == pytest.approx(0.0)
        assert plan.scenes[0].end_time_sec == pytest.approx(5.0)
        assert plan.scenes[1].start_time_sec == pytest.approx(5.0)
        assert plan.scenes[1].end_time_sec == pytest.approx(10.0)
        assert plan.scenes[2].start_time_sec == pytest.approx(10.0)
        assert plan.scenes[2].end_time_sec == pytest.approx(15.0)
        assert plan.scenes[3].start_time_sec == pytest.approx(15.0)
        assert plan.scenes[3].end_time_sec == pytest.approx(20.0)

    def test_variable_event_durations(self):
        """Events with different durations: each scene absorbs its events."""
        plan = _make_plan([0, 1], [2, 3], [4])
        durations = {0: 1.5, 1: 2.5, 2: 3.0, 3: 0.5, 4: 4.0}
        TimelineSync.sync(plan, durations)

        # Scene 0: 1.5 + 2.5 = 4.0
        assert plan.scenes[0].end_time_sec == pytest.approx(4.0)

        # Scene 1 starts at 4.0: 3.0 + 0.5 = 3.5 → ends at 7.5
        assert plan.scenes[1].start_time_sec == pytest.approx(4.0)
        assert plan.scenes[1].end_time_sec == pytest.approx(7.5)

        # Scene 2 starts at 7.5: 4.0 → ends at 11.5
        assert plan.scenes[2].start_time_sec == pytest.approx(7.5)
        assert plan.scenes[2].end_time_sec == pytest.approx(11.5)


# ── duration_sec property ──────────────────────────────────────────────────

class TestDurationSecProperty:
    def test_duration_sec_is_difference(self):
        """SceneSpec.duration_sec == end_time_sec - start_time_sec."""
        plan = _make_plan([0, 1])
        TimelineSync.sync(plan, {0: 2.0, 1: 3.0})

        scene = plan.scenes[0]
        assert scene.duration_sec == pytest.approx(
            scene.end_time_sec - scene.start_time_sec
        )

    def test_duration_sec_none_before_sync(self):
        """Before sync, duration_sec is None."""
        plan = _make_plan([0])
        assert plan.scenes[0].duration_sec is None

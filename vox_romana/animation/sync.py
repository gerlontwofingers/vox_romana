"""TimelineSync — maps TTS event durations onto AnimationPlan scene timestamps.

This module bridges the audio world (TTS-rendered events, each with a known
duration) and the video world (SceneSpec start/end times needed by the
renderer).

THE PROBLEM:

    When the AnimationDirector creates the AnimationPlan, it doesn't know
    how long each event will be — TTS hasn't run yet. The plan knows WHICH
    events belong to each scene (via ``event_global_orders``), but not WHEN
    each scene starts or ends.

    After TTS runs, we have a duration for every AudioEvent (by
    ``global_order``). TimelineSync takes that mapping and fills in
    ``start_time_sec`` and ``end_time_sec`` on every SceneSpec so the
    renderer can sync images to audio.

THE CALCULATION:

    1. Sort all events by global_order to get a canonical play order.
    2. Walk events left-to-right, accumulating a running time cursor.
    3. For each SceneSpec:
         start_time_sec = cursor when the FIRST event in this scene begins.
         end_time_sec   = cursor after the LAST event in this scene ends.

    Events not assigned to any scene are included in the running cursor
    (they still consume audio time) but don't affect scene boundaries.

    Scenes with no events in ``event_durations`` get a fallback duration
    of ``DEFAULT_FALLBACK_DURATION_SEC`` (so the image still shows for a
    reasonable time even if TTS data is missing).

EVENT DURATIONS FORMAT:

    ``event_durations`` maps ``global_order: int → duration_seconds: float``.
    This matches the format used by the TTS renderer's output tracking.

EXAMPLE:

    from vox_romana.animation.sync import TimelineSync

    # From TTS rendering
    event_durations = {0: 2.5, 1: 4.1, 2: 3.8, 3: 2.0}

    # From AnimationDirector
    plan.scenes[0].event_global_orders = [0, 1]
    plan.scenes[1].event_global_orders = [2, 3]

    TimelineSync.sync(plan, event_durations)

    plan.scenes[0].start_time_sec  # → 0.0
    plan.scenes[0].end_time_sec    # → 6.6   (2.5 + 4.1)
    plan.scenes[1].start_time_sec  # → 6.6
    plan.scenes[1].end_time_sec    # → 12.4  (6.6 + 3.8 + 2.0)
"""

from __future__ import annotations

import logging

from .models import AnimationPlan, SceneSpec

logger = logging.getLogger(__name__)


# Duration assigned to a scene when none of its events have a known TTS duration.
# Prevents zero-length scenes from crashing the renderer.
DEFAULT_FALLBACK_DURATION_SEC: float = 3.0


class TimelineSync:
    """Computes and assigns scene timestamps from TTS event durations.

    Stateless — all methods are class methods. Instantiation is optional.
    """

    @classmethod
    def sync(
        cls,
        plan: AnimationPlan,
        event_durations: dict[int, float],
    ) -> AnimationPlan:
        """Fill in ``start_time_sec`` and ``end_time_sec`` on every SceneSpec.

        Mutates the plan's SceneSpec objects in place. Returns the same
        plan for convenient chaining:
            plan = TimelineSync.sync(plan, event_durations)

        Args:
            plan:
                AnimationPlan with at least one scene. SceneSpec objects
                must have ``event_global_orders`` populated.
            event_durations:
                Mapping of ``AudioEvent.global_order → duration_sec``.
                Missing entries default to 0.0 (event effectively skipped;
                scene may use the fallback duration if ALL its events are
                missing).

        Returns:
            The mutated AnimationPlan.
        """
        if not plan.scenes:
            logger.debug("TimelineSync: no scenes — nothing to sync")
            return plan

        # Build a set of all global_orders that appear in any scene.
        # We need to know which events drive scene boundaries.
        all_scene_orders: set[int] = set()
        for scene in plan.scenes:
            all_scene_orders.update(scene.event_global_orders)

        # Collect ALL global_orders (from both scene events and any
        # "between-scene" events), sorted ascending.
        all_orders = sorted(
            set(event_durations.keys()) | all_scene_orders
        )

        # Build a cumulative time map: global_order → start_time_sec.
        # The start time of event N is the sum of all durations before it.
        cumulative: dict[int, float] = {}
        cursor = 0.0
        for order in all_orders:
            cumulative[order] = cursor
            cursor += event_durations.get(order, 0.0)

        # Total audio duration after the last event.
        total_duration = cursor

        # Assign timestamps to scenes.
        for scene in plan.scenes:
            cls._assign_scene_timestamps(
                scene,
                cumulative=cumulative,
                total_duration=total_duration,
                event_durations=event_durations,
            )

        logger.debug(
            "TimelineSync: synced %d scenes, total audio %.2fs",
            len(plan.scenes), total_duration,
        )
        return plan

    # ── Private helpers ───────────────────────────────────────────────────

    @classmethod
    def _assign_scene_timestamps(
        cls,
        scene: SceneSpec,
        *,
        cumulative: dict[int, float],
        total_duration: float,
        event_durations: dict[int, float],
    ) -> None:
        """Assign start_time_sec and end_time_sec to one scene.

        Uses the first and last event_global_orders in the scene.
        Falls back to DEFAULT_FALLBACK_DURATION_SEC when the scene has
        no events with known durations.
        """
        orders = sorted(scene.event_global_orders)

        if not orders:
            # Scene has no events at all — should not normally happen,
            # but handle it gracefully.
            logger.warning(
                "TimelineSync: scene %r has no event_global_orders — "
                "assigning fallback duration",
                scene.scene_id,
            )
            # Place after the previous scene (or at start if first).
            scene.start_time_sec = scene.start_time_sec or 0.0
            scene.end_time_sec = (scene.start_time_sec or 0.0) + DEFAULT_FALLBACK_DURATION_SEC
            return

        first_order = orders[0]
        last_order = orders[-1]

        start = cumulative.get(first_order)
        if start is None:
            # Event not in cumulative (not in event_durations and not
            # encountered in the sorted all_orders walk). This can happen
            # if an event_global_order references an event that was never
            # rendered by TTS (e.g. a non-utterance break).
            # Place this scene immediately after the last known event.
            start = total_duration
            logger.debug(
                "TimelineSync: scene %r event %d not in cumulative — "
                "placing at end",
                scene.scene_id, first_order,
            )

        # End time = start of last event + its duration.
        last_event_start = cumulative.get(last_order, start)
        last_event_duration = event_durations.get(last_order, 0.0)
        end = last_event_start + last_event_duration

        # If start == end (all events have zero duration or are missing),
        # use the fallback.
        if end <= start:
            end = start + DEFAULT_FALLBACK_DURATION_SEC
            logger.debug(
                "TimelineSync: scene %r has zero computed duration — "
                "using fallback %.1fs",
                scene.scene_id, DEFAULT_FALLBACK_DURATION_SEC,
            )

        scene.start_time_sec = start
        scene.end_time_sec = end

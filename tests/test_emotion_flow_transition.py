"""Tests for vox_romana.emotion_flow.transition — TransitionBuilderAgent."""

from __future__ import annotations

import numpy as np
import pytest

from vox_romana.emotion_flow.transition import TransitionBuilderAgent, _wilson_ci
from vox_romana.emotion_flow.models import (
    EMOTION_LABELS,
    EmotionSequence,
    EmotionTag,
    SpeechSegment,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_segment(idx: int, speech_id: str = "s1") -> SpeechSegment:
    return SpeechSegment(
        text=f"Segment {idx}",
        original_text=f"Segment {idx}",
        source_lang="en",
        translation_confidence=1.0,
        translation_flagged=False,
        segment_index=idx,
        speech_id=speech_id,
    )


def _make_tag(dominant: str) -> EmotionTag:
    scores = {l: 0.0 for l in EMOTION_LABELS}
    scores[dominant] = 1.0
    return EmotionTag(dominant=dominant, dominant_score=1.0, scores=scores)


def _make_sequence(
    dominants: list[str],
    speech_ids: list[str] | None = None,
) -> EmotionSequence:
    """Build an EmotionSequence with given dominant emotion labels per segment."""
    if speech_ids is None:
        speech_ids = ["s1"] * len(dominants)
    segs = [_make_segment(i, speech_id=speech_ids[i]) for i in range(len(dominants))]
    tags = [_make_tag(d) for d in dominants]
    unique_ids = list(dict.fromkeys(speech_ids))
    return EmotionSequence(
        leader_name="TestLeader",
        segments=segs,
        tags=tags,
        speech_ids=unique_ids,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TransitionBuilderAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestTransitionBuilderAgent:
    def test_builds_from_minimal_two_segment_sequence(self):
        seq = _make_sequence(["joy", "fear"])
        agent = TransitionBuilderAgent()
        tm = agent.build(seq)
        assert tm.n_transitions == 1

    def test_matrix_is_row_normalized(self):
        # Use default smoothing so that unvisited rows are non-zero (k=0 would
        # leave rows with no observations as 0/0 = NaN).
        seq = _make_sequence(["joy", "fear", "joy", "anger"])
        agent = TransitionBuilderAgent()  # smoothing_k=0.1 by default
        tm = agent.build(seq)
        row_sums = tm.matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_count_matrix_records_raw_counts(self):
        # joy→fear twice, joy→anger once
        seq = _make_sequence(["joy", "fear", "joy", "fear", "joy", "anger"])
        agent = TransitionBuilderAgent()
        tm = agent.build(seq)
        joy_idx = EMOTION_LABELS.index("joy")
        fear_idx = EMOTION_LABELS.index("fear")
        anger_idx = EMOTION_LABELS.index("anger")
        assert tm.count_matrix[joy_idx, fear_idx] == 2
        assert tm.count_matrix[joy_idx, anger_idx] == 1

    def test_smoothing_lifts_zero_cells(self):
        # With k=0, unobserved cells are 0; with k>0, all cells > 0
        seq = _make_sequence(["joy", "fear"])
        tm_no_smooth = TransitionBuilderAgent(smoothing_k=0.0).build(seq)
        tm_smoothed = TransitionBuilderAgent(smoothing_k=0.5).build(seq)
        # Non-observed cells should be > 0 after smoothing
        joy_idx = EMOTION_LABELS.index("joy")
        anger_idx = EMOTION_LABELS.index("anger")
        assert tm_no_smooth.matrix[joy_idx, anger_idx] == pytest.approx(0.0, abs=1e-6)
        assert tm_smoothed.matrix[joy_idx, anger_idx] > 0.0

    def test_cross_speech_transitions_skipped_by_default(self):
        # joy (speech 1) → fear (speech 2) should be excluded
        seq = _make_sequence(
            ["joy", "fear"],
            speech_ids=["speech_1", "speech_2"],
        )
        agent = TransitionBuilderAgent(skip_cross_speech=True)
        tm = agent.build(seq)
        assert tm.n_transitions == 0

    def test_cross_speech_transitions_included_when_disabled(self):
        seq = _make_sequence(
            ["joy", "fear"],
            speech_ids=["speech_1", "speech_2"],
        )
        agent = TransitionBuilderAgent(skip_cross_speech=False)
        tm = agent.build(seq)
        assert tm.n_transitions == 1

    def test_within_speech_transitions_counted_correctly(self):
        # Two speeches: first has joy→anger, second has fear→neutral
        seq = _make_sequence(
            ["joy", "anger", "fear", "neutral"],
            speech_ids=["s1", "s1", "s2", "s2"],
        )
        agent = TransitionBuilderAgent(smoothing_k=0.0)
        tm = agent.build(seq)
        assert tm.n_transitions == 2
        joy_idx = EMOTION_LABELS.index("joy")
        anger_idx = EMOTION_LABELS.index("anger")
        assert tm.count_matrix[joy_idx, anger_idx] == 1

    def test_wilson_ci_shape(self):
        seq = _make_sequence(["joy", "fear", "joy", "neutral"])
        agent = TransitionBuilderAgent()
        tm = agent.build(seq)
        n = len(EMOTION_LABELS)
        assert tm.ci_low.shape == (n, n)
        assert tm.ci_high.shape == (n, n)

    def test_ci_low_le_ci_high(self):
        seq = _make_sequence(["anger", "joy", "anger", "fear", "neutral"])
        tm = TransitionBuilderAgent().build(seq)
        assert np.all(tm.ci_low <= tm.ci_high)

    def test_n_transitions_matches_count_matrix_sum(self):
        seq = _make_sequence(["joy", "fear", "anger", "neutral", "sadness"])
        tm = TransitionBuilderAgent().build(seq)
        assert tm.n_transitions == int(tm.count_matrix.sum())

    def test_single_segment_returns_near_uniform(self):
        seq = _make_sequence(["joy"])
        agent = TransitionBuilderAgent()
        tm = agent.build(seq)
        n = len(EMOTION_LABELS)
        # With no transitions, smoothing makes all cells equal → uniform matrix
        np.testing.assert_allclose(
            tm.matrix, np.full((n, n), 1.0 / n, dtype=np.float32), atol=1e-5
        )

    def test_leader_name_propagated(self):
        seq = _make_sequence(["joy", "fear"])
        tm = TransitionBuilderAgent().build(seq)
        assert tm.leader_name == "TestLeader"

    def test_emotion_labels_stored(self):
        seq = _make_sequence(["joy", "anger"])
        tm = TransitionBuilderAgent().build(seq)
        assert tm.emotion_labels == list(EMOTION_LABELS)

    def test_smoothing_k_stored(self):
        seq = _make_sequence(["joy", "anger"])
        tm = TransitionBuilderAgent(smoothing_k=0.25).build(seq)
        assert tm.smoothing_k == pytest.approx(0.25)


# ─────────────────────────────────────────────────────────────────────────────
# _wilson_ci unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWilsonCI:
    def test_zero_n_returns_0_1(self):
        lo, hi = _wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_zero_count_low_end(self):
        lo, hi = _wilson_ci(0, 100)
        assert lo == pytest.approx(0.0, abs=0.01)
        assert hi < 0.05  # CI should be very narrow near 0

    def test_full_count_high_end(self):
        lo, hi = _wilson_ci(100, 100)
        assert lo > 0.95  # CI should be very narrow near 1
        assert hi == pytest.approx(1.0, abs=0.01)

    def test_half_count_centered(self):
        lo, hi = _wilson_ci(50, 100)
        # Midpoint should be near 0.5
        assert abs((lo + hi) / 2 - 0.5) < 0.05

    def test_ci_contained_in_0_1(self):
        for k, n in [(0, 10), (5, 10), (10, 10), (1, 100), (99, 100)]:
            lo, hi = _wilson_ci(k, n)
            assert 0.0 <= lo <= hi <= 1.0

    def test_larger_n_gives_narrower_ci(self):
        lo10, hi10 = _wilson_ci(5, 10)
        lo100, hi100 = _wilson_ci(50, 100)
        assert (hi10 - lo10) > (hi100 - lo100)

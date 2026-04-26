"""Tests for vox_romana.emotion_flow.semantic — SemanticFlowAgent."""

from __future__ import annotations

import numpy as np
import pytest

from vox_romana.emotion_flow.semantic import (
    SemanticFlowAgent,
    _cosine_distances,
    _emotion_cosine_distances,
)
from vox_romana.emotion_flow.models import (
    EMOTION_LABELS,
    EmotionSequence,
    EmotionTag,
    SpeechSegment,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_segment(idx: int) -> SpeechSegment:
    return SpeechSegment(
        text=f"Segment text number {idx} for testing purposes.",
        original_text=f"Segment text number {idx} for testing purposes.",
        source_lang="en",
        translation_confidence=1.0,
        translation_flagged=False,
        segment_index=idx,
        speech_id="s1",
    )


def _make_tag(dominant: str) -> EmotionTag:
    scores = {l: 0.0 for l in EMOTION_LABELS}
    scores[dominant] = 1.0
    return EmotionTag(dominant=dominant, dominant_score=1.0, scores=scores)


def _make_sequence(dominants: list[str]) -> EmotionSequence:
    segs = [_make_segment(i) for i in range(len(dominants))]
    tags = [_make_tag(d) for d in dominants]
    return EmotionSequence(
        leader_name="TestLeader",
        segments=segs,
        tags=tags,
        speech_ids=["s1"],
    )


def _identity_embed(n: int, dim: int = 4) -> np.ndarray:
    """Deterministic L2-normalised embeddings (cyclic basis vectors)."""
    basis = np.eye(dim, dtype=np.float32)
    return np.array([basis[i % dim] for i in range(n)])


# ─────────────────────────────────────────────────────────────────────────────
# SemanticFlowAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticFlowAgent:
    def test_n_pairs_is_len_minus_one(self):
        seq = _make_sequence(["joy", "fear", "anger", "neutral", "sadness"])
        agent = SemanticFlowAgent(_embed_fn=lambda texts: _identity_embed(len(texts)))
        result = agent.compute(seq)
        assert result.n_pairs == 4

    def test_topic_shifts_length_matches_n_pairs(self):
        seq = _make_sequence(["joy", "fear", "anger", "sadness"])
        agent = SemanticFlowAgent(_embed_fn=lambda texts: _identity_embed(len(texts)))
        result = agent.compute(seq)
        assert len(result.topic_shifts) == result.n_pairs

    def test_emotion_shifts_length_matches_n_pairs(self):
        seq = _make_sequence(["joy", "fear", "anger", "sadness"])
        agent = SemanticFlowAgent(_embed_fn=lambda texts: _identity_embed(len(texts)))
        result = agent.compute(seq)
        assert len(result.emotion_shifts) == result.n_pairs

    def test_pearson_r_in_range(self):
        seq = _make_sequence(["joy", "anger", "joy", "anger", "joy"])
        agent = SemanticFlowAgent(_embed_fn=lambda texts: _identity_embed(len(texts)))
        result = agent.compute(seq)
        assert -1.0 <= result.pearson_r <= 1.0

    def test_p_value_in_range(self):
        seq = _make_sequence(["joy", "anger", "fear", "neutral", "sadness"])
        agent = SemanticFlowAgent(_embed_fn=lambda texts: _identity_embed(len(texts)))
        result = agent.compute(seq)
        assert 0.0 <= result.p_value <= 1.0

    def test_single_segment_returns_zero_result(self):
        seq = _make_sequence(["joy"])
        agent = SemanticFlowAgent(_embed_fn=lambda texts: _identity_embed(len(texts)))
        result = agent.compute(seq)
        assert result.n_pairs == 0
        assert result.pearson_r == 0.0
        assert result.p_value == 1.0
        assert result.topic_shifts == []
        assert result.emotion_shifts == []

    def test_perfectly_correlated_shifts_give_r_one(self):
        """joy→anger→anger→joy produces identical topic and emotion shifts.

        Tags: joy, anger, anger, joy
        - emotion_shifts: [dist(joy,anger), dist(anger,anger), dist(anger,joy)]
                        = [1.0, 0.0, 1.0]
        Embeddings: [1,0], [0,1], [0,1], [1,0]
        - topic_shifts: [dist([1,0],[0,1]), dist([0,1],[0,1]), dist([0,1],[1,0])]
                      = [1.0, 0.0, 1.0]
        Both sequences [1.0, 0.0, 1.0] → Pearson r = 1.0
        """
        seq = _make_sequence(["joy", "anger", "anger", "joy"])
        fixed_embeddings = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
        )
        agent = SemanticFlowAgent(_embed_fn=lambda texts: fixed_embeddings)
        result = agent.compute(seq)
        assert result.pearson_r == pytest.approx(1.0, abs=1e-6)

    def test_constant_emotion_shifts_returns_zero_r(self):
        """When one shift sequence is constant, Pearson r is undefined → returns 0.0."""
        seq = _make_sequence(["joy", "joy", "joy", "joy"])
        agent = SemanticFlowAgent(_embed_fn=lambda texts: _identity_embed(len(texts)))
        result = agent.compute(seq)
        # emotion_shifts all 0.0 (constant) → pearson_r = 0.0
        assert result.pearson_r == pytest.approx(0.0, abs=1e-6)

    def test_topic_shifts_in_valid_range(self):
        """Cosine distances must be in [0, 2]."""
        seq = _make_sequence(["joy", "fear", "anger", "neutral", "sadness", "disgust"])
        agent = SemanticFlowAgent(_embed_fn=lambda texts: _identity_embed(len(texts)))
        result = agent.compute(seq)
        for d in result.topic_shifts:
            assert 0.0 <= d <= 2.0

    def test_emotion_shifts_in_valid_range(self):
        seq = _make_sequence(["joy", "anger", "fear", "neutral"])
        agent = SemanticFlowAgent(_embed_fn=lambda texts: _identity_embed(len(texts)))
        result = agent.compute(seq)
        for d in result.emotion_shifts:
            assert 0.0 <= d <= 2.0


# ─────────────────────────────────────────────────────────────────────────────
# _cosine_distances unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCosineDistances:
    def test_identical_embeddings_zero_distance(self):
        e = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        dists = _cosine_distances(e)
        assert len(dists) == 1
        assert dists[0] == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_embeddings_distance_one(self):
        e = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        dists = _cosine_distances(e)
        assert dists[0] == pytest.approx(1.0, abs=1e-6)

    def test_opposite_embeddings_distance_two(self):
        e = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        dists = _cosine_distances(e)
        assert dists[0] == pytest.approx(2.0, abs=1e-6)

    def test_zero_vector_returns_one(self):
        e = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        dists = _cosine_distances(e)
        assert dists[0] == pytest.approx(1.0, abs=1e-6)

    def test_length_is_n_minus_one(self):
        e = np.eye(5, dtype=np.float32)
        dists = _cosine_distances(e)
        assert len(dists) == 4


# ─────────────────────────────────────────────────────────────────────────────
# _emotion_cosine_distances unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionCosineDistances:
    def test_same_emotion_zero_distance(self):
        tags = [_make_tag("joy"), _make_tag("joy")]
        dists = _emotion_cosine_distances(tags)
        assert len(dists) == 1
        assert dists[0] == pytest.approx(0.0, abs=1e-6)

    def test_different_emotions_nonzero_distance(self):
        tags = [_make_tag("joy"), _make_tag("anger")]
        dists = _emotion_cosine_distances(tags)
        assert dists[0] > 0.0

    def test_length_is_n_minus_one(self):
        tags = [_make_tag(l) for l in EMOTION_LABELS]
        dists = _emotion_cosine_distances(tags)
        assert len(dists) == len(EMOTION_LABELS) - 1

    def test_distance_in_valid_range(self):
        tags = [_make_tag("joy"), _make_tag("anger"), _make_tag("fear")]
        dists = _emotion_cosine_distances(tags)
        for d in dists:
            assert 0.0 <= d <= 2.0

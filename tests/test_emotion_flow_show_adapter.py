"""Tests for EmotionFlowShowAdapter.

All tests run fully offline — no LLM calls, no I/O (except tmp_path for
file-loading tests).  The adapter is a pure translation layer so every
assertion targets the mapping logic:

    dominant_cycle      → segments
    null_test           → director_brief.show_arc
    semantic_flow       → tone
    transition_matrix   → key_facts
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vox_romana.emotion_flow.models import (
    EMOTION_LABELS,
    EmotionFlowReport,
    NullTestResult,
    SemanticFlowResult,
    TransitionMatrix,
)
from vox_romana.emotion_flow.show_adapter import EmotionFlowShowAdapter
from vox_romana.shows.news.config import NewsConfig

# ── Test fixtures ──────────────────────────────────────────────────────────────


def _make_null_test(is_structured: bool = True) -> NullTestResult:
    return NullTestResult(
        permutation_p=0.01 if is_structured else 0.80,
        chi2_p=0.01 if is_structured else 0.80,
        chi2_statistic=10.0 if is_structured else 0.5,
        g_test_p=0.01 if is_structured else 0.80,
        g_statistic=10.0 if is_structured else 0.5,
        cramers_v=0.45 if is_structured else 0.05,
        entropy_observed=1.5,
        entropy_random_mean=2.1,
        entropy_ci=(1.8, 2.4),
        is_structured=is_structured,
        n_permutations=100,
    )


def _make_semantic_flow(pearson_r: float = 0.6) -> SemanticFlowResult:
    return SemanticFlowResult(
        pearson_r=pearson_r,
        p_value=0.01,
        topic_shifts=[0.1, 0.2, 0.3],
        emotion_shifts=[0.1, 0.2, 0.3],
        n_pairs=3,
    )


def _make_transition_matrix(n: int = 7) -> TransitionMatrix:
    labels = EMOTION_LABELS[:n]
    rng = np.random.default_rng(0)
    raw = rng.random((n, n)).astype(np.float32)
    # Zero the diagonal so off-diagonal transitions are clearly present.
    np.fill_diagonal(raw, 0.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    matrix = raw / np.where(row_sums == 0.0, 1.0, row_sums)
    return TransitionMatrix(
        leader_name="test_leader",
        matrix=matrix,
        count_matrix=np.ones((n, n), dtype=np.int32),
        ci_low=np.zeros((n, n), dtype=np.float32),
        ci_high=np.ones((n, n), dtype=np.float32),
        emotion_labels=labels,
        smoothing_k=0.1,
        n_transitions=n * n,
    )


def _make_report(
    leader_name: str = "Test Leader",
    dominant_cycle: list[str] | None = None,
    is_structured: bool = True,
    pearson_r: float = 0.6,
    with_matrix: bool = True,
    with_null_test: bool = True,
    with_semantic: bool = True,
) -> EmotionFlowReport:
    if dominant_cycle is None:
        dominant_cycle = ["neutral", "anger", "joy"]
    return EmotionFlowReport(
        leader_name=leader_name,
        n_speeches=3,
        n_segments=12,
        dominant_cycle=dominant_cycle,
        null_test=_make_null_test(is_structured) if with_null_test else None,
        semantic_flow=_make_semantic_flow(pearson_r) if with_semantic else None,
        transition_matrix=_make_transition_matrix() if with_matrix else None,
    )


# ── TestEmotionFlowShowAdapterConstruction ─────────────────────────────────────


class TestEmotionFlowShowAdapterConstruction:
    def test_default_construction(self):
        adapter = EmotionFlowShowAdapter()
        assert adapter.reporter_name == "Emotion Flow"
        assert adapter.n_segments == 6
        assert adapter.target_duration_minutes == 10

    def test_custom_reporter_name(self):
        adapter = EmotionFlowShowAdapter(reporter_name="My Show")
        assert adapter.reporter_name == "My Show"

    def test_custom_n_segments(self):
        adapter = EmotionFlowShowAdapter(n_segments=4)
        assert adapter.n_segments == 4

    def test_custom_target_duration(self):
        adapter = EmotionFlowShowAdapter(target_duration_minutes=20)
        assert adapter.target_duration_minutes == 20


# ── TestAdaptBasic ─────────────────────────────────────────────────────────────


class TestAdaptBasic:
    def test_returns_news_config(self):
        adapter = EmotionFlowShowAdapter()
        result = adapter.adapt(_make_report())
        assert isinstance(result, NewsConfig)

    def test_has_anchor(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report())
        assert len(config.anchors) == 1

    def test_title_contains_leader_name(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(leader_name="Tealc"))
        assert "Tealc" in config.title

    def test_brief_contains_leader_name(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(leader_name="Tealc"))
        assert "Tealc" in config.brief

    def test_reporter_name_in_title(self):
        adapter = EmotionFlowShowAdapter(reporter_name="VoxAnalytics")
        config = adapter.adapt(_make_report())
        assert "VoxAnalytics" in config.title

    def test_content_type_is_news(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report())
        assert config.content_type == "news"

    def test_target_segments_matches_actual_segment_count(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report())
        assert config.target_segments == len(config.segments)

    def test_has_show_host(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report())
        assert config.show_host is not None

    def test_has_director_brief(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report())
        assert config.director_brief is not None

    def test_topic_contains_leader_name(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(leader_name="Ronon"))
        assert "Ronon" in (config.topic or "")


# ── TestAdaptDominantCycle ─────────────────────────────────────────────────────


class TestAdaptDominantCycle:
    def test_cycle_emotions_produce_headline_segments(self):
        adapter = EmotionFlowShowAdapter(n_segments=6)
        config = adapter.adapt(_make_report(dominant_cycle=["anger", "joy"]))
        headings = [s.headline.lower() for s in config.segments]
        assert any("anger" in h for h in headings)
        assert any("joy" in h for h in headings)

    def test_empty_cycle_produces_at_least_two_segments(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(dominant_cycle=[]))
        # overview + synthesis
        assert len(config.segments) >= 2

    def test_duplicate_emotions_deduplicated_in_segments(self):
        adapter = EmotionFlowShowAdapter(n_segments=8)
        config = adapter.adapt(_make_report(dominant_cycle=["joy", "anger", "joy", "anger"]))
        headings = [s.headline.lower() for s in config.segments]
        joy_headings = [h for h in headings if "joy" in h]
        assert len(joy_headings) <= 1

    def test_single_emotion_cycle_produces_segments(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(dominant_cycle=["neutral"]))
        assert len(config.segments) >= 1

    def test_n_segments_cap_respected(self):
        adapter = EmotionFlowShowAdapter(n_segments=3)
        config = adapter.adapt(
            _make_report(dominant_cycle=["anger", "joy", "fear", "sadness", "neutral"])
        )
        assert len(config.segments) <= 3

    def test_overview_is_first_segment(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(dominant_cycle=["joy", "fear"]))
        assert "landscape" in config.segments[0].headline.lower() or "overview" in config.segments[0].angle.lower()

    def test_synthesis_is_last_segment(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(dominant_cycle=["joy"]))
        last = config.segments[-1]
        assert "numbers" in last.headline.lower() or "synthesis" in last.headline.lower() or "tell us" in last.headline.lower()


# ── TestAdaptStructuredFlag ────────────────────────────────────────────────────


class TestAdaptStructuredFlag:
    def test_structured_true_reflected_in_show_arc(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(is_structured=True))
        arc = config.director_brief.show_arc.lower()
        assert "structured" in arc

    def test_structured_false_reflected_in_show_arc(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(is_structured=False))
        arc = config.director_brief.show_arc.lower()
        assert "stochastic" in arc or "exploratory" in arc

    def test_no_null_test_still_has_director_brief(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(with_null_test=False))
        assert config.director_brief is not None
        assert config.director_brief.show_arc  # non-empty

    def test_cramers_v_appears_in_arc_when_null_test_present(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(is_structured=True))
        assert "0.45" in config.director_brief.show_arc


# ── TestAdaptSemanticCoherence ─────────────────────────────────────────────────


class TestAdaptSemanticCoherence:
    def test_high_pearson_r_gives_analytical_tone(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(pearson_r=0.8))
        assert config.tone == "analytical"

    def test_mid_pearson_r_gives_investigative_tone(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(pearson_r=0.35))
        assert config.tone == "investigative"

    def test_low_pearson_r_gives_exploratory_tone(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(pearson_r=0.05))
        assert config.tone == "exploratory"

    def test_no_semantic_flow_defaults_to_analytical(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(with_semantic=False))
        assert config.tone == "analytical"

    def test_negative_pearson_r_uses_absolute_value(self):
        adapter = EmotionFlowShowAdapter()
        # |r| = 0.8 → analytical
        config = adapter.adapt(_make_report(pearson_r=-0.8))
        assert config.tone == "analytical"


# ── TestAdaptTransitionFacts ───────────────────────────────────────────────────


class TestAdaptTransitionFacts:
    def test_transition_facts_present_when_matrix_available(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt(_make_report(with_matrix=True))
        all_facts = [f for seg in config.segments for f in seg.key_facts]
        assert len(all_facts) > 0

    def test_top_transitions_empty_when_no_matrix(self):
        result = EmotionFlowShowAdapter._top_transitions(None)
        assert result == []

    def test_top_transitions_skips_self_loops(self):
        # Build a matrix where only the diagonal is non-zero.
        n = 3
        labels = ["anger", "joy", "neutral"]
        matrix = np.eye(n, dtype=np.float32)  # 100% self-loops
        tm = TransitionMatrix(
            leader_name="t",
            matrix=matrix,
            count_matrix=np.zeros((n, n), dtype=np.int32),
            ci_low=np.zeros((n, n), dtype=np.float32),
            ci_high=np.zeros((n, n), dtype=np.float32),
            emotion_labels=labels,
            smoothing_k=0.0,
            n_transitions=3,
        )
        result = EmotionFlowShowAdapter._top_transitions(tm)
        assert result == []  # no cross-emotion transitions


# ── TestAdaptToDict ────────────────────────────────────────────────────────────


class TestAdaptToDict:
    def test_adapt_to_dict_returns_dict(self):
        adapter = EmotionFlowShowAdapter()
        result = adapter.adapt_to_dict(_make_report())
        assert isinstance(result, dict)

    def test_adapt_to_dict_is_json_serializable(self):
        adapter = EmotionFlowShowAdapter()
        result = adapter.adapt_to_dict(_make_report())
        dumped = json.dumps(result)  # must not raise
        assert isinstance(dumped, str)

    def test_adapt_to_dict_has_brief_key(self):
        adapter = EmotionFlowShowAdapter()
        result = adapter.adapt_to_dict(_make_report())
        assert "brief" in result

    def test_adapt_to_dict_has_segments_key_with_entries(self):
        adapter = EmotionFlowShowAdapter()
        result = adapter.adapt_to_dict(_make_report())
        assert "segments" in result
        assert len(result["segments"]) > 0


# ── TestAdaptFromJson ──────────────────────────────────────────────────────────


class TestAdaptFromJson:
    def _minimal_json(
        self,
        leader_name: str = "TestLeader",
        dominant_cycle: list[str] | None = None,
        null_test: dict | None = None,
        semantic_flow: dict | None = None,
        transition_matrix: dict | None = None,
    ) -> str:
        data = {
            "leader_name": leader_name,
            "n_speeches": 2,
            "n_segments": 8,
            "dominant_cycle": dominant_cycle or ["joy", "neutral"],
            "null_test": null_test,
            "semantic_flow": semantic_flow,
            "transition_matrix": transition_matrix,
            "back_translation_agreement": 1.0,
            "n_flagged_segments": 0,
        }
        return json.dumps(data)

    def test_adapt_from_json_returns_news_config(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt_from_json(self._minimal_json())
        assert isinstance(config, NewsConfig)

    def test_adapt_from_json_leader_name_propagated(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt_from_json(self._minimal_json(leader_name="Vala"))
        assert "Vala" in config.title

    def test_adapt_from_json_with_null_test(self):
        adapter = EmotionFlowShowAdapter()
        nt = {
            "permutation_p": 0.01,
            "chi2_p": 0.01,
            "chi2_statistic": 12.0,
            "g_test_p": 0.01,
            "g_statistic": 12.0,
            "cramers_v": 0.50,
            "entropy_observed": 1.5,
            "entropy_random_mean": 2.1,
            "entropy_ci": [1.8, 2.4],
            "is_structured": True,
            "n_permutations": 200,
        }
        config = adapter.adapt_from_json(self._minimal_json(null_test=nt))
        assert "structured" in config.director_brief.show_arc.lower()

    def test_adapt_from_file_loads_json_file(self, tmp_path: Path):
        adapter = EmotionFlowShowAdapter()
        report_data = {
            "leader_name": "FileLeader",
            "n_speeches": 2,
            "n_segments": 8,
            "dominant_cycle": ["fear", "joy"],
            "null_test": None,
            "semantic_flow": None,
            "transition_matrix": None,
            "back_translation_agreement": 0.95,
            "n_flagged_segments": 1,
        }
        p = tmp_path / "report.json"
        p.write_text(json.dumps(report_data), encoding="utf-8")
        config = adapter.adapt_from_file(p)
        assert isinstance(config, NewsConfig)
        assert "FileLeader" in config.title

    def test_adapt_from_dict_minimal_data(self):
        adapter = EmotionFlowShowAdapter()
        config = adapter.adapt_from_dict({"leader_name": "Min", "dominant_cycle": []})
        assert isinstance(config, NewsConfig)

    def test_adapt_from_dict_with_transition_matrix(self):
        adapter = EmotionFlowShowAdapter()
        matrix = [[0.0, 0.6, 0.4], [0.3, 0.0, 0.7], [0.5, 0.5, 0.0]]
        data = {
            "leader_name": "MatrixLeader",
            "dominant_cycle": ["anger"],
            "transition_matrix": {
                "matrix": matrix,
                "emotion_labels": ["anger", "joy", "neutral"],
                "smoothing_k": 0.1,
                "n_transitions": 9,
            },
        }
        config = adapter.adapt_from_dict(data)
        assert isinstance(config, NewsConfig)
        all_facts = [f for seg in config.segments for f in seg.key_facts]
        # Facts should mention cross-emotion transitions.
        assert any(
            word in f
            for f in all_facts
            for word in ("anger", "joy", "neutral")
        )

    def test_adapt_from_dict_with_semantic_flow(self):
        adapter = EmotionFlowShowAdapter()
        data = {
            "leader_name": "SemanticLeader",
            "dominant_cycle": ["surprise"],
            "semantic_flow": {
                "pearson_r": 0.65,
                "p_value": 0.02,
                "topic_shifts": [0.1, 0.2],
                "emotion_shifts": [0.1, 0.2],
                "n_pairs": 2,
            },
        }
        config = adapter.adapt_from_dict(data)
        assert config.tone == "analytical"  # |r|=0.65 > 0.5

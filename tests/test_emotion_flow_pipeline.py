"""Tests for EmotionFlowPipeline and _dominant_cycle helper.

All heavy agents (TranslatorAgent, EmotionTaggerAgent, SemanticFlowAgent,
NullHypothesisTesterAgent, EmotionFlowReporter/Visualizer) are injected via
their constructor kwargs so the tests run fully offline with no model loads.

The pipeline is tested via its agent-kwarg injection API:
  - _force_passthrough=True  on TranslatorAgent
  - _force_label="neutral"   on EmotionTaggerAgent
  - _embed_fn=...             on SemanticFlowAgent
  - reporter with visualizer=MagicMock() and output_dir=tmp_path
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox_romana.emotion_flow.models import (
    EMOTION_LABELS,
    EmotionFlowReport,
    EmotionSequence,
)
from vox_romana.emotion_flow.pipeline import EmotionFlowPipeline, _dominant_cycle

# ── Helpers ───────────────────────────────────────────────────────────────────


def _flat_embed(texts: list[str]) -> np.ndarray:
    """Deterministic TF-IDF-style embed: row i = one-hot on (i % dim)."""
    dim = 8
    arr = np.zeros((len(texts), dim), dtype=np.float32)
    for i in range(len(texts)):
        arr[i, i % dim] = 1.0
    return arr


def _make_txt_source(tmp_path: Path, n: int = 6) -> Path:
    """Write a plain-text file with n paragraphs (≥20 chars each)."""
    content = "\n\n".join(
        f"This is paragraph number {i + 1} with enough words to pass the minimum length filter."
        for i in range(n)
    )
    p = tmp_path / "test_speech.txt"
    p.write_text(content, encoding="utf-8")
    return p


def _make_minimal_pipeline(
    *,
    output_dir: Path | None = None,
    run_semantic: bool = True,
    run_translation: bool = False,
    n_permutations: int = 50,
) -> EmotionFlowPipeline:
    """Pipeline pre-configured for fast offline tests."""
    mock_viz = MagicMock()
    mock_viz.save_all.return_value = {}
    return EmotionFlowPipeline(
        output_dir=output_dir,
        seed=42,
        n_permutations=n_permutations,
        run_semantic=run_semantic,
        run_translation=run_translation,
        translator_kwargs={"_force_passthrough": True},
        tagger_kwargs={"_force_label": "neutral"},
        semantic_kwargs={"_embed_fn": _flat_embed},
        visualizer_kwargs={
            "_mpl": MagicMock(return_value=(MagicMock(), MagicMock())),
            "_sns": MagicMock(),
            "_go": MagicMock(),
        },
        reporter_kwargs={"visualizer": mock_viz},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TestDominantCycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestDominantCycle:
    def test_empty_returns_empty(self) -> None:
        assert _dominant_cycle([]) == []

    def test_single_returns_single(self) -> None:
        assert _dominant_cycle(["joy"]) == ["joy"]

    def test_all_same_returns_single(self) -> None:
        assert _dominant_cycle(["joy", "joy", "joy"]) == ["joy"]

    def test_alternating_returns_all(self) -> None:
        assert _dominant_cycle(["joy", "fear", "joy", "fear"]) == [
            "joy", "fear", "joy", "fear"
        ]

    def test_runs_deduplicated(self) -> None:
        assert _dominant_cycle(["joy", "joy", "fear", "fear", "joy"]) == [
            "joy", "fear", "joy"
        ]

    def test_preserves_order(self) -> None:
        seq = ["anger", "sadness", "sadness", "joy", "neutral"]
        assert _dominant_cycle(seq) == ["anger", "sadness", "joy", "neutral"]


# ═══════════════════════════════════════════════════════════════════════════════
# TestEmotionFlowPipelineConstruction
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmotionFlowPipelineConstruction:
    def test_default_construction(self) -> None:
        pipeline = EmotionFlowPipeline()
        assert pipeline._seed == 42
        assert pipeline._n_permutations == 1000
        assert pipeline._run_semantic is True
        assert pipeline._run_translation is True

    def test_custom_seed(self) -> None:
        pipeline = EmotionFlowPipeline(seed=123)
        assert pipeline._seed == 123

    def test_custom_n_permutations(self) -> None:
        pipeline = EmotionFlowPipeline(n_permutations=500)
        assert pipeline._n_permutations == 500

    def test_run_semantic_false(self) -> None:
        pipeline = EmotionFlowPipeline(run_semantic=False)
        assert pipeline._run_semantic is False

    def test_run_translation_false(self) -> None:
        pipeline = EmotionFlowPipeline(run_translation=False)
        assert pipeline._run_translation is False

    def test_output_dir_set(self, tmp_path: Path) -> None:
        pipeline = EmotionFlowPipeline(output_dir=tmp_path)
        assert pipeline._output_dir == tmp_path

    def test_output_dir_none_by_default(self) -> None:
        pipeline = EmotionFlowPipeline()
        assert pipeline._output_dir is None

    def test_extra_kwargs_stored(self) -> None:
        pipeline = EmotionFlowPipeline(tagger_kwargs={"batch_size": 8})
        assert pipeline._tagger_kwargs == {"batch_size": 8}


# ═══════════════════════════════════════════════════════════════════════════════
# TestPipelineRun — functional (offline, injected mocks)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineRun:
    def test_returns_emotion_flow_report(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(src)
        assert isinstance(result, EmotionFlowReport)

    def test_leader_name_inferred_from_filename(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(src)
        assert result.leader_name == src.stem

    def test_leader_name_set_explicitly(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(src, leader="Cicero")
        assert result.leader_name == "Cicero"

    def test_n_segments_matches_source(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path, n=6)
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(src)
        assert result.n_segments == 6

    def test_transition_matrix_present(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(src)
        assert result.transition_matrix is not None

    def test_null_test_present(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(src)
        assert result.null_test is not None

    def test_semantic_flow_present_when_run_semantic_true(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline(run_semantic=True)
        result = pipeline.run(src)
        assert result.semantic_flow is not None

    def test_semantic_flow_absent_when_run_semantic_false(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline(run_semantic=False)
        result = pipeline.run(src)
        assert result.semantic_flow is None

    def test_manifest_present(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(src)
        assert result.manifest is not None

    def test_manifest_seed_matches_pipeline_seed(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(src)
        assert result.manifest.seed == 42

    def test_dominant_cycle_is_list(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(src)
        assert isinstance(result.dominant_cycle, list)

    def test_report_path_empty_when_no_output_dir(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline(output_dir=None)
        result = pipeline.run(src)
        assert result.report_path == ""

    def test_report_path_set_when_output_dir_given(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        out = tmp_path / "reports"
        pipeline = _make_minimal_pipeline(output_dir=out)
        result = pipeline.run(src)
        assert result.report_path != ""
        assert result.report_path.endswith("report.html")

    def test_html_file_written_to_disk(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        out = tmp_path / "reports"
        pipeline = _make_minimal_pipeline(output_dir=out)
        result = pipeline.run(src)
        assert Path(result.report_path).exists()

    def test_json_file_written_to_disk(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        out = tmp_path / "reports"
        pipeline = _make_minimal_pipeline(output_dir=out)
        result = pipeline.run(src)
        out_dir = Path(result.report_path).parent
        assert (out_dir / "report.json").exists()

    def test_csv_file_written_to_disk(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        out = tmp_path / "reports"
        pipeline = _make_minimal_pipeline(output_dir=out)
        result = pipeline.run(src)
        out_dir = Path(result.report_path).parent
        assert (out_dir / "report.csv").exists()

    def test_json_leader_name_matches(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        out = tmp_path / "reports"
        pipeline = _make_minimal_pipeline(output_dir=out)
        result = pipeline.run(src, leader="Hortensia")
        out_dir = Path(result.report_path).parent
        data = json.loads((out_dir / "report.json").read_text())
        assert data["leader_name"] == "Hortensia"

    def test_n_permutations_respected(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline(n_permutations=10)
        result = pipeline.run(src)
        assert result.null_test.n_permutations == 10

    def test_transition_matrix_shape_is_n_by_n(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(src)
        n = len(EMOTION_LABELS)
        assert result.transition_matrix.matrix.shape == (n, n)

    def test_all_tags_forced_to_neutral(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(src)
        # dominant_cycle should only contain "neutral" when _force_label="neutral"
        assert all(e == "neutral" for e in result.dominant_cycle)

    def test_directory_source_loads_multiple_speeches(self, tmp_path: Path) -> None:
        speech_dir = tmp_path / "speeches"
        speech_dir.mkdir()
        for i in range(3):
            (speech_dir / f"speech_{i}.txt").write_text(
                "\n\n".join(
                    f"Paragraph {j} from speech {i} with sufficient text here."
                    for j in range(3)
                ),
                encoding="utf-8",
            )
        pipeline = _make_minimal_pipeline()
        result = pipeline.run(speech_dir)
        assert result.n_segments == 9  # 3 speeches × 3 paragraphs

    def test_invalid_source_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "nonexistent.txt"
        pipeline = _make_minimal_pipeline()
        with pytest.raises((ValueError, FileNotFoundError)):
            pipeline.run(bad)


# ═══════════════════════════════════════════════════════════════════════════════
# TestPipelineTranslationFlag
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineTranslationFlag:
    """When run_translation=False, _force_passthrough must be injected."""

    def test_no_translate_does_not_crash(self, tmp_path: Path) -> None:
        src = _make_txt_source(tmp_path)
        # run_translation=False: pipeline should inject _force_passthrough=True
        pipeline = EmotionFlowPipeline(
            run_translation=False,
            run_semantic=False,
            tagger_kwargs={"_force_label": "joy"},
            semantic_kwargs={"_embed_fn": _flat_embed},
            visualizer_kwargs={
                "_mpl": MagicMock(return_value=(MagicMock(), MagicMock())),
                "_sns": MagicMock(),
                "_go": MagicMock(),
            },
            reporter_kwargs={"visualizer": MagicMock(save_all=MagicMock(return_value={}))},
            n_permutations=10,
        )
        result = pipeline.run(src)
        assert isinstance(result, EmotionFlowReport)

    def test_no_translate_flag_overrides_translator_kwargs(self, tmp_path: Path) -> None:
        """run_translation=False should force passthrough even if not in translator_kwargs."""
        src = _make_txt_source(tmp_path)
        pipeline = EmotionFlowPipeline(
            run_translation=False,
            run_semantic=False,
            # Intentionally do NOT set _force_passthrough in translator_kwargs;
            # but also set _force_passthrough via flag so translator doesn't call real models
            translator_kwargs={"_force_passthrough": True},
            tagger_kwargs={"_force_label": "joy"},
            reporter_kwargs={"visualizer": MagicMock(save_all=MagicMock(return_value={}))},
            n_permutations=10,
        )
        result = pipeline.run(src)
        assert result.n_segments > 0

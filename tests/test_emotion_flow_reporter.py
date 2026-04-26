"""Tests for EmotionFlowReporter.

render_json / render_csv / render_html are tested with in-memory reports.
build() is tested with a real tmp_path directory; the visualizer is injected
as a MagicMock to avoid matplotlib/plotly during reporter tests.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from vox_romana.emotion_flow.models import (
    EMOTION_LABELS,
    EmotionFlowReport,
    NullTestResult,
    ReproducibilityManifest,
    SemanticFlowResult,
    TransitionMatrix,
)
from vox_romana.emotion_flow.reporter import EmotionFlowReporter, _NumpyEncoder

# ── Fixtures ──────────────────────────────────────────────────────────────────

N = len(EMOTION_LABELS)


def _make_tm(leader: str = "Caesar") -> TransitionMatrix:
    matrix = np.eye(N, dtype=np.float32)
    count_matrix = (np.eye(N) * 5).astype(np.int32)
    ci_low = np.zeros((N, N), dtype=np.float32)
    ci_high = np.ones((N, N), dtype=np.float32)
    return TransitionMatrix(
        leader_name=leader,
        matrix=matrix,
        count_matrix=count_matrix,
        ci_low=ci_low,
        ci_high=ci_high,
        emotion_labels=list(EMOTION_LABELS),
        smoothing_k=0.1,
        n_transitions=int(count_matrix.sum()),
    )


def _make_null_test() -> NullTestResult:
    return NullTestResult(
        permutation_p=0.02,
        chi2_p=0.03,
        chi2_statistic=38.0,
        g_test_p=0.04,
        g_statistic=37.0,
        cramers_v=0.65,
        entropy_observed=1.1,
        entropy_random_mean=2.4,
        entropy_ci=(1.9, 2.9),
        is_structured=True,
        n_permutations=500,
    )


def _make_semantic() -> SemanticFlowResult:
    return SemanticFlowResult(
        pearson_r=0.72,
        p_value=0.001,
        topic_shifts=[0.2, 0.4, 0.6],
        emotion_shifts=[0.3, 0.5, 0.7],
        n_pairs=3,
    )


def _make_manifest() -> ReproducibilityManifest:
    return ReproducibilityManifest(
        seed=42,
        python_version="3.12.0",
        package_versions={"numpy": "1.26", "torch": "2.4"},
        model_versions={"tagger": "j-hartmann/emotion-english-distilroberta-base"},
        hardware="Apple M3",
        timestamp="2026-04-23T12:00:00Z",
        git_commit="abc1234",
        deterministic_mode=True,
    )


def _make_report(
    *,
    with_tm: bool = True,
    with_null: bool = True,
    with_semantic: bool = False,
    with_manifest: bool = False,
    narrative: str = "",
) -> EmotionFlowReport:
    return EmotionFlowReport(
        leader_name="Caesar",
        n_speeches=3,
        n_segments=42,
        dominant_cycle=["joy", "neutral", "sadness"],
        transition_matrix=_make_tm() if with_tm else None,
        null_test=_make_null_test() if with_null else None,
        semantic_flow=_make_semantic() if with_semantic else None,
        manifest=_make_manifest() if with_manifest else None,
        narrative=narrative,
    )


def _make_mock_visualizer() -> MagicMock:
    """Visualizer mock that returns {} from save_all (no file I/O)."""
    mock_viz = MagicMock()
    mock_viz.save_all.return_value = {}
    return mock_viz


# ═══════════════════════════════════════════════════════════════════════════════
# TestNumpyEncoder
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumpyEncoder:
    def test_encodes_ndarray_as_list(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = json.loads(json.dumps(arr, cls=_NumpyEncoder))
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_encodes_numpy_float(self) -> None:
        val = np.float32(3.14)
        result = json.loads(json.dumps(val, cls=_NumpyEncoder))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-3

    def test_encodes_numpy_int(self) -> None:
        val = np.int32(99)
        result = json.loads(json.dumps(val, cls=_NumpyEncoder))
        assert result == 99
        assert isinstance(result, int)

    def test_encodes_numpy_bool(self) -> None:
        val = np.bool_(True)
        result = json.loads(json.dumps(val, cls=_NumpyEncoder))
        assert result is True


# ═══════════════════════════════════════════════════════════════════════════════
# TestRenderJson
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderJson:
    def test_returns_valid_json_string(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_json(_make_report())
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_leader_name_present(self) -> None:
        reporter = EmotionFlowReporter()
        parsed = json.loads(reporter.render_json(_make_report()))
        assert parsed["leader_name"] == "Caesar"

    def test_n_segments_correct(self) -> None:
        reporter = EmotionFlowReporter()
        parsed = json.loads(reporter.render_json(_make_report()))
        assert parsed["n_segments"] == 42

    def test_dominant_cycle_is_list(self) -> None:
        reporter = EmotionFlowReporter()
        parsed = json.loads(reporter.render_json(_make_report()))
        assert isinstance(parsed["dominant_cycle"], list)
        assert parsed["dominant_cycle"] == ["joy", "neutral", "sadness"]

    def test_numpy_matrix_serialized_as_nested_list(self) -> None:
        reporter = EmotionFlowReporter()
        parsed = json.loads(reporter.render_json(_make_report(with_tm=True)))
        matrix = parsed["transition_matrix"]["matrix"]
        assert isinstance(matrix, list)
        assert isinstance(matrix[0], list)

    def test_null_test_fields_present(self) -> None:
        reporter = EmotionFlowReporter()
        parsed = json.loads(reporter.render_json(_make_report(with_null=True)))
        nt = parsed["null_test"]
        assert "chi2_p" in nt
        assert "permutation_p" in nt
        assert "is_structured" in nt

    def test_none_fields_serialized_as_null(self) -> None:
        reporter = EmotionFlowReporter()
        parsed = json.loads(reporter.render_json(_make_report(with_semantic=False)))
        assert parsed["semantic_flow"] is None

    def test_transition_matrix_none_serialized_as_null(self) -> None:
        reporter = EmotionFlowReporter()
        parsed = json.loads(reporter.render_json(_make_report(with_tm=False)))
        assert parsed["transition_matrix"] is None

    def test_manifest_serialized_when_present(self) -> None:
        reporter = EmotionFlowReporter()
        parsed = json.loads(reporter.render_json(_make_report(with_manifest=True)))
        assert parsed["manifest"]["seed"] == 42
        assert parsed["manifest"]["python_version"] == "3.12.0"

    def test_output_is_indented_human_readable(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_json(_make_report())
        # Indented JSON contains newlines
        assert "\n" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TestRenderCsv
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderCsv:
    def test_returns_string(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_csv(_make_report(with_tm=True))
        assert isinstance(result, str)

    def test_empty_string_when_no_tm(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_csv(_make_report(with_tm=False))
        assert result == ""

    def test_header_row_contains_all_emotion_labels(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_csv(_make_report(with_tm=True))
        reader = csv.reader(result.splitlines())
        header = next(reader)
        for label in EMOTION_LABELS:
            assert label in header

    def test_n_data_rows_equals_n_emotions(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_csv(_make_report(with_tm=True))
        rows = list(csv.reader(result.splitlines()))
        # First row is header; rest are data
        assert len(rows) == N + 1

    def test_first_column_is_from_label(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_csv(_make_report(with_tm=True))
        rows = list(csv.reader(result.splitlines()))
        from_labels = [rows[i + 1][0] for i in range(N)]
        assert from_labels == list(EMOTION_LABELS)

    def test_values_are_parseable_as_floats(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_csv(_make_report(with_tm=True))
        rows = list(csv.reader(result.splitlines()))
        for row in rows[1:]:
            for cell in row[1:]:  # skip the "from" label column
                float(cell)  # should not raise

    def test_diagonal_values_close_to_one_for_identity_matrix(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_csv(_make_report(with_tm=True))
        rows = list(csv.reader(result.splitlines()))
        # identity matrix has 1.0 on diagonal; with smoothing they're ~1.0
        for i in range(N):
            val = float(rows[i + 1][i + 1])  # +1 for header offset, +1 for label col
            assert val > 0.5, f"Diagonal value at ({i},{i}) = {val} — expected near 1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# TestRenderHtml
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderHtml:
    def test_returns_string(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report())
        assert isinstance(result, str)

    def test_doctype_present(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report())
        assert result.startswith("<!DOCTYPE html>")

    def test_leader_name_in_html(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report())
        assert "Caesar" in result

    def test_n_segments_in_html(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report())
        assert "42" in result

    def test_n_speeches_in_html(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report())
        assert "3" in result

    def test_dominant_cycle_in_html(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report())
        assert "joy" in result
        assert "sadness" in result

    def test_structured_badge_when_is_structured_true(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report(with_null=True))
        assert "STRUCTURED" in result

    def test_pearson_r_in_html_when_semantic_present(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report(with_semantic=True))
        assert "0.7200" in result  # _fmt_f(0.72) = "0.7200"

    def test_narrative_in_html_when_present(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report(narrative="A detailed narrative."))
        assert "A detailed narrative." in result

    def test_narrative_section_absent_when_empty(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report(narrative=""))
        assert "Narrative Summary" not in result

    def test_manifest_section_present_when_provided(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report(with_manifest=True))
        assert "Reproducibility" in result
        assert "abc1234" in result

    def test_closing_html_tag_present(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report())
        assert result.strip().endswith("</html>")

    def test_inline_css_present(self) -> None:
        reporter = EmotionFlowReporter()
        result = reporter.render_html(_make_report())
        assert "<style>" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TestBuild
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuild:
    def test_no_output_dir_returns_report_unchanged(self) -> None:
        reporter = EmotionFlowReporter()
        report = _make_report()
        result = reporter.build(report)
        assert result is report

    def test_no_output_dir_report_path_not_set(self) -> None:
        reporter = EmotionFlowReporter()
        report = _make_report()
        reporter.build(report)
        assert report.report_path == ""

    def test_html_file_created(self, tmp_path: Path) -> None:
        reporter = EmotionFlowReporter(
            output_dir=tmp_path, visualizer=_make_mock_visualizer()
        )
        result = reporter.build(_make_report())
        assert Path(result.report_path).exists()
        assert Path(result.report_path).suffix == ".html"

    def test_json_file_created(self, tmp_path: Path) -> None:
        reporter = EmotionFlowReporter(
            output_dir=tmp_path, visualizer=_make_mock_visualizer()
        )
        result = reporter.build(_make_report())
        out_dir = Path(result.report_path).parent
        assert (out_dir / "report.json").exists()

    def test_csv_file_created(self, tmp_path: Path) -> None:
        reporter = EmotionFlowReporter(
            output_dir=tmp_path, visualizer=_make_mock_visualizer()
        )
        result = reporter.build(_make_report())
        out_dir = Path(result.report_path).parent
        assert (out_dir / "report.csv").exists()

    def test_report_path_set_on_returned_report(self, tmp_path: Path) -> None:
        reporter = EmotionFlowReporter(
            output_dir=tmp_path, visualizer=_make_mock_visualizer()
        )
        result = reporter.build(_make_report())
        assert result.report_path != ""

    def test_report_path_points_to_html(self, tmp_path: Path) -> None:
        reporter = EmotionFlowReporter(
            output_dir=tmp_path, visualizer=_make_mock_visualizer()
        )
        result = reporter.build(_make_report())
        assert result.report_path.endswith("report.html")

    def test_output_dir_created_if_not_exists(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "deep" / "nested" / "dir"
        reporter = EmotionFlowReporter(
            output_dir=new_dir, visualizer=_make_mock_visualizer()
        )
        reporter.build(_make_report())
        assert new_dir.exists()

    def test_subdirectory_named_with_leader_slug(self, tmp_path: Path) -> None:
        reporter = EmotionFlowReporter(
            output_dir=tmp_path, visualizer=_make_mock_visualizer()
        )
        result = reporter.build(_make_report())
        out_dir = Path(result.report_path).parent
        assert "caesar" in out_dir.name

    def test_visualizer_save_all_called(self, tmp_path: Path) -> None:
        mock_viz = _make_mock_visualizer()
        reporter = EmotionFlowReporter(output_dir=tmp_path, visualizer=mock_viz)
        report = _make_report()
        reporter.build(report)
        mock_viz.save_all.assert_called_once()

    def test_html_file_content_is_valid_html(self, tmp_path: Path) -> None:
        reporter = EmotionFlowReporter(
            output_dir=tmp_path, visualizer=_make_mock_visualizer()
        )
        result = reporter.build(_make_report())
        content = Path(result.report_path).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Caesar" in content

    def test_json_file_contains_valid_json(self, tmp_path: Path) -> None:
        reporter = EmotionFlowReporter(
            output_dir=tmp_path, visualizer=_make_mock_visualizer()
        )
        result = reporter.build(_make_report())
        out_dir = Path(result.report_path).parent
        parsed = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        assert parsed["leader_name"] == "Caesar"

    def test_csv_file_has_correct_row_count(self, tmp_path: Path) -> None:
        reporter = EmotionFlowReporter(
            output_dir=tmp_path, visualizer=_make_mock_visualizer()
        )
        result = reporter.build(_make_report(with_tm=True))
        out_dir = Path(result.report_path).parent
        rows = list(csv.reader((out_dir / "report.csv").read_text().splitlines()))
        assert len(rows) == N + 1  # header + one row per emotion

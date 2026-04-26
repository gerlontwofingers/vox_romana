"""Tests for EmotionFlowVisualizer.

All matplotlib, seaborn, and plotly calls are fully mocked via constructor
injection (_mpl, _sns, _go) — no display or real model is needed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from vox_romana.emotion_flow.models import (
    EMOTION_LABELS,
    EmotionFlowReport,
    NullTestResult,
    TransitionMatrix,
)
from vox_romana.emotion_flow.visualizer import EMOTION_COLOURS, EmotionFlowVisualizer

# ── Fixtures ──────────────────────────────────────────────────────────────────

N = len(EMOTION_LABELS)


def _make_tm(leader: str = "Caesar") -> TransitionMatrix:
    """Minimal TransitionMatrix with a known probability matrix."""
    matrix = np.eye(N, dtype=np.float32)
    count_matrix = (np.eye(N) * 10).astype(np.int32)
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


def _make_null_test(
    structured: bool = True,
) -> NullTestResult:
    return NullTestResult(
        permutation_p=0.01,
        chi2_p=0.01,
        chi2_statistic=42.0,
        g_test_p=0.01,
        g_statistic=43.0,
        cramers_v=0.7,
        entropy_observed=1.2,
        entropy_random_mean=2.5,
        entropy_ci=(2.0, 3.0),
        is_structured=structured,
        n_permutations=1000,
    )


def _make_report(
    *,
    with_tm: bool = True,
    with_null: bool = True,
) -> EmotionFlowReport:
    return EmotionFlowReport(
        leader_name="Caesar",
        n_speeches=2,
        n_segments=20,
        dominant_cycle=["joy", "neutral", "sadness"],
        transition_matrix=_make_tm() if with_tm else None,
        null_test=_make_null_test() if with_null else None,
    )


def _make_mock_mpl() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (mock_plt, mock_fig, mock_ax)."""
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    return mock_plt, mock_fig, mock_ax


# ═══════════════════════════════════════════════════════════════════════════════
# TestPlotTransitionHeatmap
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlotTransitionHeatmap:
    def _viz(self, mock_plt: MagicMock, mock_sns: MagicMock) -> EmotionFlowVisualizer:
        return EmotionFlowVisualizer(_mpl=mock_plt, _sns=mock_sns)

    def test_returns_figure_object(self) -> None:
        mock_plt, mock_fig, _ = _make_mock_mpl()
        mock_sns = MagicMock()
        viz = self._viz(mock_plt, mock_sns)
        result = viz.plot_transition_heatmap(_make_tm())
        assert result is mock_fig

    def test_subplots_called_once(self) -> None:
        mock_plt, _, _ = _make_mock_mpl()
        mock_sns = MagicMock()
        viz = self._viz(mock_plt, mock_sns)
        viz.plot_transition_heatmap(_make_tm())
        mock_plt.subplots.assert_called_once()

    def test_seaborn_heatmap_called_once(self) -> None:
        mock_plt, _, _ = _make_mock_mpl()
        mock_sns = MagicMock()
        viz = self._viz(mock_plt, mock_sns)
        viz.plot_transition_heatmap(_make_tm())
        mock_sns.heatmap.assert_called_once()

    def test_seaborn_heatmap_receives_matrix(self) -> None:
        mock_plt, _, _ = _make_mock_mpl()
        mock_sns = MagicMock()
        tm = _make_tm()
        viz = self._viz(mock_plt, mock_sns)
        viz.plot_transition_heatmap(tm)
        first_positional_arg = mock_sns.heatmap.call_args[0][0]
        assert first_positional_arg is tm.matrix

    def test_seaborn_heatmap_receives_emotion_labels_as_xticklabels(self) -> None:
        mock_plt, _, _ = _make_mock_mpl()
        mock_sns = MagicMock()
        tm = _make_tm()
        viz = self._viz(mock_plt, mock_sns)
        viz.plot_transition_heatmap(tm)
        kwargs = mock_sns.heatmap.call_args[1]
        assert kwargs["xticklabels"] == tm.emotion_labels

    def test_seaborn_heatmap_receives_emotion_labels_as_yticklabels(self) -> None:
        mock_plt, _, _ = _make_mock_mpl()
        mock_sns = MagicMock()
        tm = _make_tm()
        viz = self._viz(mock_plt, mock_sns)
        viz.plot_transition_heatmap(tm)
        kwargs = mock_sns.heatmap.call_args[1]
        assert kwargs["yticklabels"] == tm.emotion_labels

    def test_title_contains_leader_name(self) -> None:
        mock_plt, _, mock_ax = _make_mock_mpl()
        mock_sns = MagicMock()
        tm = _make_tm("Cicero")
        viz = self._viz(mock_plt, mock_sns)
        viz.plot_transition_heatmap(tm)
        title_arg = mock_ax.set_title.call_args[0][0]
        assert "Cicero" in title_arg

    def test_xlabel_is_to(self) -> None:
        mock_plt, _, mock_ax = _make_mock_mpl()
        mock_sns = MagicMock()
        viz = self._viz(mock_plt, mock_sns)
        viz.plot_transition_heatmap(_make_tm())
        xlabel_arg = mock_ax.set_xlabel.call_args[0][0]
        assert xlabel_arg == "To"

    def test_ylabel_is_from(self) -> None:
        mock_plt, _, mock_ax = _make_mock_mpl()
        mock_sns = MagicMock()
        viz = self._viz(mock_plt, mock_sns)
        viz.plot_transition_heatmap(_make_tm())
        ylabel_arg = mock_ax.set_ylabel.call_args[0][0]
        assert ylabel_arg == "From"

    def test_tight_layout_called(self) -> None:
        mock_plt, mock_fig, _ = _make_mock_mpl()
        mock_sns = MagicMock()
        viz = self._viz(mock_plt, mock_sns)
        viz.plot_transition_heatmap(_make_tm())
        mock_fig.tight_layout.assert_called_once()

    def test_cmap_is_ylorrd(self) -> None:
        mock_plt, _, _ = _make_mock_mpl()
        mock_sns = MagicMock()
        viz = self._viz(mock_plt, mock_sns)
        viz.plot_transition_heatmap(_make_tm())
        kwargs = mock_sns.heatmap.call_args[1]
        assert kwargs["cmap"] == "YlOrRd"


# ═══════════════════════════════════════════════════════════════════════════════
# TestPlotSankey
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlotSankey:
    def test_returns_figure_object(self) -> None:
        mock_go = MagicMock()
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig
        viz = EmotionFlowVisualizer(_go=mock_go)
        result = viz.plot_sankey(_make_tm())
        assert result is mock_fig

    def test_go_figure_called_once(self) -> None:
        mock_go = MagicMock()
        viz = EmotionFlowVisualizer(_go=mock_go)
        viz.plot_sankey(_make_tm())
        mock_go.Figure.assert_called_once()

    def test_sankey_called_with_two_times_n_nodes(self) -> None:
        mock_go = MagicMock()
        viz = EmotionFlowVisualizer(_go=mock_go)
        viz.plot_sankey(_make_tm())
        sankey_kwargs = mock_go.Sankey.call_args[1]
        node_labels = sankey_kwargs["node"]["label"]
        assert len(node_labels) == 2 * N

    def test_sankey_source_nodes_in_range_0_to_n_minus_1(self) -> None:
        mock_go = MagicMock()
        viz = EmotionFlowVisualizer(_go=mock_go)
        viz.plot_sankey(_make_tm(), threshold=0.0)
        sankey_kwargs = mock_go.Sankey.call_args[1]
        sources = sankey_kwargs["link"]["source"]
        assert all(0 <= s < N for s in sources)

    def test_sankey_target_nodes_in_range_n_to_2n_minus_1(self) -> None:
        mock_go = MagicMock()
        viz = EmotionFlowVisualizer(_go=mock_go)
        viz.plot_sankey(_make_tm(), threshold=0.0)
        sankey_kwargs = mock_go.Sankey.call_args[1]
        targets = sankey_kwargs["link"]["target"]
        assert all(N <= t < 2 * N for t in targets)

    def test_threshold_filters_low_probability_links(self) -> None:
        mock_go = MagicMock()
        # identity matrix: diagonal = 1.0, off-diagonal = 0.0
        viz = EmotionFlowVisualizer(_go=mock_go)
        viz.plot_sankey(_make_tm(), threshold=0.5)
        sankey_kwargs = mock_go.Sankey.call_args[1]
        # Only self-loops (prob=1.0) should survive
        assert len(sankey_kwargs["link"]["value"]) == N

    def test_all_link_values_above_threshold(self) -> None:
        mock_go = MagicMock()
        viz = EmotionFlowVisualizer(_go=mock_go)
        viz.plot_sankey(_make_tm(), threshold=0.5)
        sankey_kwargs = mock_go.Sankey.call_args[1]
        for v in sankey_kwargs["link"]["value"]:
            assert v >= 0.5

    def test_title_contains_leader_name(self) -> None:
        mock_go = MagicMock()
        mock_fig = MagicMock()
        mock_go.Figure.return_value = mock_fig
        tm = _make_tm("Augustus")
        viz = EmotionFlowVisualizer(_go=mock_go)
        viz.plot_sankey(tm)
        update_kwargs = mock_fig.update_layout.call_args[1]
        assert "Augustus" in update_kwargs["title_text"]

    def test_node_colors_length_is_2n(self) -> None:
        mock_go = MagicMock()
        viz = EmotionFlowVisualizer(_go=mock_go)
        viz.plot_sankey(_make_tm())
        sankey_kwargs = mock_go.Sankey.call_args[1]
        assert len(sankey_kwargs["node"]["color"]) == 2 * N

    def test_link_colors_match_link_count(self) -> None:
        mock_go = MagicMock()
        viz = EmotionFlowVisualizer(_go=mock_go)
        viz.plot_sankey(_make_tm(), threshold=0.0)
        sankey_kwargs = mock_go.Sankey.call_args[1]
        assert len(sankey_kwargs["link"]["color"]) == len(
            sankey_kwargs["link"]["value"]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestPlotEntropyRate
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlotEntropyRate:
    def test_returns_figure_object(self) -> None:
        mock_plt, mock_fig, _ = _make_mock_mpl()
        viz = EmotionFlowVisualizer(_mpl=mock_plt)
        result = viz.plot_entropy_rate(_make_null_test())
        assert result is mock_fig

    def test_subplots_called_once(self) -> None:
        mock_plt, _, _ = _make_mock_mpl()
        viz = EmotionFlowVisualizer(_mpl=mock_plt)
        viz.plot_entropy_rate(_make_null_test())
        mock_plt.subplots.assert_called_once()

    def test_two_axhlines_drawn(self) -> None:
        mock_plt, _, mock_ax = _make_mock_mpl()
        viz = EmotionFlowVisualizer(_mpl=mock_plt)
        viz.plot_entropy_rate(_make_null_test())
        assert mock_ax.axhline.call_count == 2

    def test_one_axhspan_drawn(self) -> None:
        mock_plt, _, mock_ax = _make_mock_mpl()
        viz = EmotionFlowVisualizer(_mpl=mock_plt)
        viz.plot_entropy_rate(_make_null_test())
        mock_ax.axhspan.assert_called_once()

    def test_observed_entropy_in_axhline(self) -> None:
        mock_plt, _, mock_ax = _make_mock_mpl()
        nt = _make_null_test()
        viz = EmotionFlowVisualizer(_mpl=mock_plt)
        viz.plot_entropy_rate(nt)
        drawn_y_values = [c[0][0] for c in mock_ax.axhline.call_args_list]
        assert nt.entropy_observed in drawn_y_values

    def test_null_mean_in_axhline(self) -> None:
        mock_plt, _, mock_ax = _make_mock_mpl()
        nt = _make_null_test()
        viz = EmotionFlowVisualizer(_mpl=mock_plt)
        viz.plot_entropy_rate(nt)
        drawn_y_values = [c[0][0] for c in mock_ax.axhline.call_args_list]
        assert nt.entropy_random_mean in drawn_y_values

    def test_ci_band_uses_correct_range(self) -> None:
        mock_plt, _, mock_ax = _make_mock_mpl()
        nt = _make_null_test()
        viz = EmotionFlowVisualizer(_mpl=mock_plt)
        viz.plot_entropy_rate(nt)
        span_args = mock_ax.axhspan.call_args[0]
        assert span_args[0] == nt.entropy_ci[0]
        assert span_args[1] == nt.entropy_ci[1]

    def test_custom_title_applied(self) -> None:
        mock_plt, _, mock_ax = _make_mock_mpl()
        viz = EmotionFlowVisualizer(_mpl=mock_plt)
        viz.plot_entropy_rate(_make_null_test(), title="My Custom Title")
        set_title_arg = mock_ax.set_title.call_args[0][0]
        assert set_title_arg == "My Custom Title"

    def test_default_title_when_empty(self) -> None:
        mock_plt, _, mock_ax = _make_mock_mpl()
        viz = EmotionFlowVisualizer(_mpl=mock_plt)
        viz.plot_entropy_rate(_make_null_test(), title="")
        set_title_arg = mock_ax.set_title.call_args[0][0]
        assert "Entropy" in set_title_arg

    def test_ylabel_contains_entropy(self) -> None:
        mock_plt, _, mock_ax = _make_mock_mpl()
        viz = EmotionFlowVisualizer(_mpl=mock_plt)
        viz.plot_entropy_rate(_make_null_test())
        ylabel_arg = mock_ax.set_ylabel.call_args[0][0]
        assert "Entropy" in ylabel_arg

    def test_tight_layout_called(self) -> None:
        mock_plt, mock_fig, _ = _make_mock_mpl()
        viz = EmotionFlowVisualizer(_mpl=mock_plt)
        viz.plot_entropy_rate(_make_null_test())
        mock_fig.tight_layout.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# TestSaveAll
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveAll:
    """Test save_all() in isolation by mocking the plot methods on the instance."""

    def _patched_viz(self) -> tuple[EmotionFlowVisualizer, MagicMock, MagicMock, MagicMock]:
        """Return (viz, mock_heatmap_fig, mock_sankey_fig, mock_entropy_fig)."""
        mock_plt = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())

        viz = EmotionFlowVisualizer(_mpl=mock_plt, _go=MagicMock(), _sns=MagicMock())

        heatmap_fig = MagicMock()
        sankey_fig = MagicMock()
        entropy_fig = MagicMock()

        viz.plot_transition_heatmap = MagicMock(return_value=heatmap_fig)
        viz.plot_sankey = MagicMock(return_value=sankey_fig)
        viz.plot_entropy_rate = MagicMock(return_value=entropy_fig)

        return viz, heatmap_fig, sankey_fig, entropy_fig

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested"
        viz, _, _, _ = self._patched_viz()
        viz.save_all(_make_report(), target)
        assert target.exists()

    def test_returns_dict(self, tmp_path: Path) -> None:
        viz, _, _, _ = self._patched_viz()
        result = viz.save_all(_make_report(), tmp_path)
        assert isinstance(result, dict)

    def test_heatmap_key_present_when_tm_available(self, tmp_path: Path) -> None:
        viz, _, _, _ = self._patched_viz()
        result = viz.save_all(_make_report(with_tm=True), tmp_path)
        assert "heatmap" in result

    def test_sankey_key_present_when_tm_available(self, tmp_path: Path) -> None:
        viz, _, _, _ = self._patched_viz()
        result = viz.save_all(_make_report(with_tm=True), tmp_path)
        assert "sankey" in result

    def test_entropy_rate_key_present_when_null_test_available(self, tmp_path: Path) -> None:
        viz, _, _, _ = self._patched_viz()
        result = viz.save_all(_make_report(with_null=True), tmp_path)
        assert "entropy_rate" in result

    def test_no_heatmap_key_when_tm_absent(self, tmp_path: Path) -> None:
        viz, _, _, _ = self._patched_viz()
        result = viz.save_all(_make_report(with_tm=False), tmp_path)
        assert "heatmap" not in result

    def test_no_sankey_key_when_tm_absent(self, tmp_path: Path) -> None:
        viz, _, _, _ = self._patched_viz()
        result = viz.save_all(_make_report(with_tm=False), tmp_path)
        assert "sankey" not in result

    def test_no_entropy_key_when_null_test_absent(self, tmp_path: Path) -> None:
        viz, _, _, _ = self._patched_viz()
        result = viz.save_all(_make_report(with_null=False), tmp_path)
        assert "entropy_rate" not in result

    def test_heatmap_savefig_called(self, tmp_path: Path) -> None:
        viz, heatmap_fig, _, _ = self._patched_viz()
        viz.save_all(_make_report(with_tm=True), tmp_path)
        heatmap_fig.savefig.assert_called_once()

    def test_sankey_write_html_called(self, tmp_path: Path) -> None:
        viz, _, sankey_fig, _ = self._patched_viz()
        viz.save_all(_make_report(with_tm=True), tmp_path)
        sankey_fig.write_html.assert_called_once()

    def test_heatmap_path_ends_with_png(self, tmp_path: Path) -> None:
        viz, _, _, _ = self._patched_viz()
        result = viz.save_all(_make_report(with_tm=True), tmp_path)
        assert result["heatmap"].suffix == ".png"

    def test_sankey_path_ends_with_html(self, tmp_path: Path) -> None:
        viz, _, _, _ = self._patched_viz()
        result = viz.save_all(_make_report(with_tm=True), tmp_path)
        assert result["sankey"].suffix == ".html"

    def test_entropy_rate_path_ends_with_png(self, tmp_path: Path) -> None:
        viz, _, _, _ = self._patched_viz()
        result = viz.save_all(_make_report(with_null=True), tmp_path)
        assert result["entropy_rate"].suffix == ".png"


# ═══════════════════════════════════════════════════════════════════════════════
# TestEmotionColours
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmotionColours:
    def test_all_emotion_labels_have_colour(self) -> None:
        for label in EMOTION_LABELS:
            assert label in EMOTION_COLOURS

    def test_all_colours_are_valid_hex(self) -> None:
        for label, colour in EMOTION_COLOURS.items():
            assert colour.startswith("#"), f"Colour for {label} should start with #"
            assert len(colour) == 7, f"Colour for {label} should be 6-digit hex"

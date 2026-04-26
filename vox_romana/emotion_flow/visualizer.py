"""EmotionFlowVisualizer — publication-ready figures for EmotionFlowReport.

Three output types:
  * Transition heatmap  — seaborn heatmap of row-normalised probabilities
  * Sankey diagram      — plotly Sankey of emotion flow (transitions ≥ threshold)
  * Entropy-rate plot   — observed entropy vs. permutation-null CI (matplotlib)

All rendering dependencies (matplotlib, seaborn, plotly) are imported lazily
so that the module can be imported in environments where those packages are
absent (e.g. minimal CI).

For testing, inject ``_mpl`` (matplotlib.pyplot mock), ``_sns`` (seaborn mock),
and ``_go`` (plotly.graph_objects mock) via the constructor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .models import EMOTION_LABELS, EmotionFlowReport, NullTestResult, TransitionMatrix

logger = logging.getLogger(__name__)

# ── Emotion colour palette ────────────────────────────────────────────────────

EMOTION_COLOURS: dict[str, str] = {
    "anger":    "#e74c3c",
    "disgust":  "#8e44ad",
    "fear":     "#f39c12",
    "joy":      "#27ae60",
    "neutral":  "#95a5a6",
    "sadness":  "#2980b9",
    "surprise": "#f1c40f",
}

# Default figure DPI for saved PNGs
_DEFAULT_DPI: int = 150


class EmotionFlowVisualizer:
    """Produce figures for an :class:`~.models.EmotionFlowReport`.

    Args:
        _mpl: Injected ``matplotlib.pyplot`` module (for testing).
        _sns: Injected ``seaborn`` module (for testing).
        _go:  Injected ``plotly.graph_objects`` module (for testing).

    All three default to ``None``, which triggers lazy real imports.
    """

    def __init__(
        self,
        *,
        _mpl: Any | None = None,
        _sns: Any | None = None,
        _go: Any | None = None,
    ) -> None:
        self._mpl = _mpl
        self._sns = _sns
        self._go = _go

    # ── Dependency accessors ──────────────────────────────────────────────────

    def _get_mpl(self) -> Any:
        if self._mpl is not None:
            return self._mpl
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless; safe to call even after prior use
            import matplotlib.pyplot as plt
            return plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for heatmap/entropy plots. "
                "Install it with:  pip install 'matplotlib>=3.9'"
            ) from exc

    def _get_sns(self) -> Any:
        if self._sns is not None:
            return self._sns
        try:
            import seaborn as sns
            return sns
        except ImportError as exc:
            raise ImportError(
                "seaborn is required for the transition heatmap. "
                "Install it with:  pip install 'seaborn>=0.13'"
            ) from exc

    def _get_go(self) -> Any:
        if self._go is not None:
            return self._go
        try:
            import plotly.graph_objects as go
            return go
        except ImportError as exc:
            raise ImportError(
                "plotly is required for the Sankey diagram. "
                "Install it with:  pip install 'plotly>=5.24'"
            ) from exc

    # ── Figure constructors ───────────────────────────────────────────────────

    def plot_transition_heatmap(self, tm: TransitionMatrix) -> Any:
        """Return a matplotlib Figure: seaborn heatmap of transition probabilities.

        The heatmap shows the row-normalised probability matrix with ``YlOrRd``
        colour scale, annotated cell values, and emotion labels on both axes.
        """
        plt = self._get_mpl()
        sns = self._get_sns()

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            tm.matrix,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=tm.emotion_labels,
            yticklabels=tm.emotion_labels,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="#cccccc",
        )
        ax.set_title(
            f"Emotion Transition Matrix \u2014 {tm.leader_name}",
            fontsize=13,
            pad=14,
        )
        ax.set_xlabel("To", fontsize=11)
        ax.set_ylabel("From", fontsize=11)
        fig.tight_layout()
        return fig

    def plot_sankey(
        self,
        tm: TransitionMatrix,
        *,
        threshold: float = 0.05,
    ) -> Any:
        """Return a plotly Figure: Sankey diagram of emotion transitions.

        Only transitions with probability ≥ ``threshold`` are drawn.
        Source nodes (0…N-1) represent "from" emotion states; target nodes
        (N…2N-1) represent "to" states so that self-loops are visible.

        Args:
            tm:        Transition matrix to visualise.
            threshold: Minimum transition probability to include as a link.
        """
        go = self._get_go()

        labels = tm.emotion_labels
        n = len(labels)

        # Nodes: first N are "from", second N are "to"
        node_labels = [f"{l} (from)" for l in labels] + [f"{l} (to)" for l in labels]
        node_colors = [EMOTION_COLOURS.get(l, "#888888") for l in labels] * 2

        sources: list[int] = []
        targets: list[int] = []
        values: list[float] = []
        link_colors: list[str] = []

        for i, src_label in enumerate(labels):
            for j in range(n):
                prob = float(tm.matrix[i, j])
                if prob >= threshold:
                    sources.append(i)
                    targets.append(n + j)
                    values.append(prob)
                    # Semi-transparent version of the source colour
                    hex_col = EMOTION_COLOURS.get(src_label, "#888888")
                    link_colors.append(hex_col + "99")

        fig = go.Figure(
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        )
        fig.update_layout(
            title_text=f"Emotion Transition Flows \u2014 {tm.leader_name}",
            title_font_size=16,
            font_size=12,
            height=600,
        )
        return fig

    def plot_entropy_rate(
        self,
        null_test: NullTestResult,
        *,
        title: str = "",
    ) -> Any:
        """Return a matplotlib Figure: observed entropy rate vs. null CI.

        A shaded band shows the 95% permutation null confidence interval.
        A dashed line marks the null mean. A solid red line marks the
        observed entropy rate.
        """
        plt = self._get_mpl()

        fig, ax = plt.subplots(figsize=(7, 4))

        ci_lo, ci_hi = null_test.entropy_ci

        # Null distribution band
        ax.axhspan(ci_lo, ci_hi, alpha=0.25, color="#aaaaaa", label="Null 95% CI")
        ax.axhline(
            null_test.entropy_random_mean,
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            label="Null mean",
        )

        # Observed value
        ax.axhline(
            null_test.entropy_observed,
            color="#e74c3c",
            linewidth=2.0,
            label=f"Observed H = {null_test.entropy_observed:.3f}",
        )

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel("Entropy Rate (bits)", fontsize=11)
        ax.set_title(
            title or "Entropy Rate vs. Null Distribution",
            fontsize=12,
            pad=10,
        )
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    # ── Bulk save ─────────────────────────────────────────────────────────────

    def save_all(
        self,
        report: EmotionFlowReport,
        output_dir: Path,
    ) -> dict[str, Path]:
        """Save all available visualizations to ``output_dir``.

        Which figures are saved depends on what data is present in the report:
        - ``heatmap.png``       — requires ``report.transition_matrix``
        - ``sankey.html``       — requires ``report.transition_matrix``
        - ``entropy_rate.png``  — requires ``report.null_test``

        Args:
            report:     Populated :class:`~.models.EmotionFlowReport`.
            output_dir: Directory in which to write the files.

        Returns:
            Mapping of short name → saved :class:`pathlib.Path`.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        if report.transition_matrix is not None:
            # Heatmap (PNG)
            try:
                fig = self.plot_transition_heatmap(report.transition_matrix)
                path = output_dir / "heatmap.png"
                fig.savefig(path, dpi=_DEFAULT_DPI, bbox_inches="tight")
                self._close(fig)
                saved["heatmap"] = path
                logger.info("Saved heatmap → %s", path)
            except Exception:
                logger.exception("Failed to save heatmap")

            # Sankey (HTML)
            try:
                fig = self.plot_sankey(report.transition_matrix)
                path = output_dir / "sankey.html"
                fig.write_html(str(path), include_plotlyjs="cdn")
                saved["sankey"] = path
                logger.info("Saved Sankey → %s", path)
            except Exception:
                logger.exception("Failed to save Sankey diagram")

        if report.null_test is not None:
            # Entropy-rate plot (PNG)
            try:
                fig = self.plot_entropy_rate(
                    report.null_test,
                    title=f"Entropy Rate \u2014 {report.leader_name}",
                )
                path = output_dir / "entropy_rate.png"
                fig.savefig(path, dpi=_DEFAULT_DPI, bbox_inches="tight")
                self._close(fig)
                saved["entropy_rate"] = path
                logger.info("Saved entropy rate → %s", path)
            except Exception:
                logger.exception("Failed to save entropy rate plot")

        return saved

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _close(self, fig: Any) -> None:
        """Close a matplotlib figure to free memory."""
        try:
            plt = self._get_mpl()
            plt.close(fig)
        except Exception:
            pass

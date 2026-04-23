"""EmotionFlowReporter — generate HTML, JSON, and CSV reports.

:class:`EmotionFlowReporter` is the output layer of the emotion_flow pipeline.
Pass a populated :class:`~.models.EmotionFlowReport` and an output directory;
the reporter writes:

  * ``report.html``        — self-contained academic HTML (inline CSS)
  * ``report.json``        — full structured JSON (numpy arrays as lists)
  * ``report.csv``         — transition probability matrix (from × to)
  * visualizations         — delegated to :class:`~.visualizer.EmotionFlowVisualizer`

QUICK START::

    from vox_romana.emotion_flow.reporter import EmotionFlowReporter
    from vox_romana.emotion_flow.models import EmotionFlowReport

    reporter = EmotionFlowReporter(output_dir="output/emotion_flow/")
    report   = reporter.build(populated_report)
    print(report.report_path)   # path to saved HTML file
"""

from __future__ import annotations

import csv
import dataclasses
import html as _html_lib
import io
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .models import EMOTION_LABELS, EmotionFlowReport
from .visualizer import EmotionFlowVisualizer

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _esc(text: str) -> str:
    return _html_lib.escape(str(text), quote=True)


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_")


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "&lt;0.001"
    return f"{p:.3f}"


def _fmt_f(v: float) -> str:
    return f"{v:.4f}"


# ── Custom JSON encoder ───────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy arrays and scalars to Python types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ── Inline CSS ────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    font-size: 14px; line-height: 1.55; color: #222; background: #fff;
}
.container { max-width: 980px; margin: 40px auto; padding: 0 24px; }
header { border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 36px; }
h1 { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
h2 { font-size: 16px; font-weight: 600; color: #333;
     border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-bottom: 14px; }
section { margin-bottom: 40px; }
.meta { font-size: 12px; color: #777; margin-bottom: 6px; }
.badge {
    display: inline-block; padding: 4px 14px; border-radius: 3px;
    font-weight: 700; font-size: 13px; letter-spacing: .5px;
}
.badge-structured     { background: #d4edda; color: #155724; }
.badge-not-structured { background: #fff3cd; color: #856404; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 10px; }
th { background: #f5f5f5; border: 1px solid #ddd; padding: 7px 11px;
     text-align: left; font-weight: 600; white-space: nowrap; }
td { border: 1px solid #ddd; padding: 7px 11px; vertical-align: top; }
tr:nth-child(even) td { background: #fafafa; }
.num { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
       font-size: 12px; }
.pass { color: #155724; font-weight: 600; }
.fail { color: #721c24; }
.narrative { background: #f9f9f9; border-left: 3px solid #ccc;
             padding: 12px 16px; font-size: 13px; line-height: 1.6;
             white-space: pre-wrap; }
.empty { color: #999; font-style: italic; }
footer { border-top: 1px solid #ddd; padding-top: 14px; margin-top: 40px;
         color: #777; font-size: 11px; line-height: 1.6; }
""".strip()


# ── EmotionFlowReporter ───────────────────────────────────────────────────────

class EmotionFlowReporter:
    """Generate HTML, JSON, and CSV reports from an :class:`~.models.EmotionFlowReport`.

    Args:
        output_dir: If provided, :meth:`build` writes files here and sets
                    ``report.report_path``.  If ``None``, :meth:`build`
                    returns the report unchanged.
        visualizer: Optional :class:`~.visualizer.EmotionFlowVisualizer`.
                    If ``None``, a default instance is used.
    """

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        visualizer: EmotionFlowVisualizer | None = None,
    ) -> None:
        self.output_dir: Path | None = Path(output_dir) if output_dir else None
        self._viz = visualizer if visualizer is not None else EmotionFlowVisualizer()

    # ── Pipeline entry point ─────────────────────────────────────────────────

    def build(self, report: EmotionFlowReport) -> EmotionFlowReport:
        """Write all output files and return the report with ``report_path`` set.

        If ``output_dir`` was not configured, returns the report unchanged.

        Files written to ``{output_dir}/{slug}_{timestamp}/``:
          * ``report.html``
          * ``report.json``
          * ``report.csv``
          * visualizations via :class:`~.visualizer.EmotionFlowVisualizer`
        """
        if self.output_dir is None:
            return report

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = _slug(report.leader_name)
        out_dir = self.output_dir / f"{slug}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Visualizations (best-effort; individual failures are logged but do not abort)
        self._viz.save_all(report, out_dir)

        # HTML
        html_path = out_dir / "report.html"
        html_path.write_text(self.render_html(report), encoding="utf-8")

        # JSON
        json_path = out_dir / "report.json"
        json_path.write_text(self.render_json(report), encoding="utf-8")

        # CSV (transition matrix)
        csv_path = out_dir / "report.csv"
        csv_path.write_text(self.render_csv(report), encoding="utf-8")

        report.report_path = str(html_path)
        logger.info("EmotionFlowReport saved → %s", out_dir)
        return report

    # ── Serialisers ──────────────────────────────────────────────────────────

    def render_json(self, report: EmotionFlowReport) -> str:
        """Serialise the report to a JSON string.

        Numpy arrays are converted to nested lists; all other types use
        standard JSON encoding.  The output is human-readable (indent=2).
        """
        data = dataclasses.asdict(report)
        return json.dumps(data, cls=_NumpyEncoder, indent=2, ensure_ascii=False)

    def render_csv(self, report: EmotionFlowReport) -> str:
        """Serialise the transition probability matrix to a CSV string.

        Rows represent the "from" emotion state; columns represent the "to"
        emotion state.  Header row uses the emotion label names.

        Returns an empty string if ``report.transition_matrix`` is ``None``.
        """
        if report.transition_matrix is None:
            return ""

        tm = report.transition_matrix
        labels = tm.emotion_labels
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Header row: blank leading cell + "to" labels
        writer.writerow(["from \\ to"] + list(labels))

        for i, from_label in enumerate(labels):
            row: list[Any] = [from_label]
            row.extend(f"{tm.matrix[i, j]:.6f}" for j in range(len(labels)))
            writer.writerow(row)

        return buf.getvalue()

    def render_html(self, report: EmotionFlowReport) -> str:
        """Render the report as a self-contained HTML document (inline CSS).

        Returns the full HTML string; does not write to disk.
        """
        generated = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
        title = f"Emotion Flow Report \u2014 {_esc(report.leader_name)}"

        sections: list[str] = [
            self._html_header(report, generated),
            self._html_overview(report),
        ]
        if report.transition_matrix is not None:
            sections.append(self._html_transition(report))
        if report.null_test is not None:
            sections.append(self._html_null_test(report))
        if report.semantic_flow is not None:
            sections.append(self._html_semantic(report))
        if report.manifest is not None:
            sections.append(self._html_manifest(report))
        if report.narrative:
            sections.append(self._html_narrative(report))
        sections.append(self._html_footer(generated))

        body = "\n".join(sections)
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '  <meta charset="UTF-8">\n'
            '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"  <title>{title}</title>\n"
            f"  <style>{_CSS}</style>\n"
            "</head>\n"
            "<body>\n"
            '<div class="container">\n'
            f"{body}\n"
            "</div>\n"
            "</body>\n"
            "</html>\n"
        )

    # ── HTML section builders ─────────────────────────────────────────────────

    def _html_header(self, report: EmotionFlowReport, generated: str) -> str:
        return (
            "<header>\n"
            f"  <h1>Emotion Flow Report &mdash; {_esc(report.leader_name)}</h1>\n"
            f'  <p class="meta">Generated {_esc(generated)}</p>\n'
            f'  <p class="meta">'
            f"{report.n_speeches} speech(es) &bull; "
            f"{report.n_segments} segment(s)"
            f"</p>\n"
            "</header>\n"
        )

    def _html_overview(self, report: EmotionFlowReport) -> str:
        cycle_html = (
            " &rarr; ".join(_esc(e) for e in report.dominant_cycle)
            if report.dominant_cycle
            else '<span class="empty">none</span>'
        )
        flagged_class = "fail" if report.n_flagged_segments > 0 else "pass"
        return (
            "<section>\n"
            "  <h2>Overview</h2>\n"
            "  <table>\n"
            "    <tr><th>Metric</th><th>Value</th></tr>\n"
            f"    <tr><td>Dominant emotion cycle</td>"
            f'<td class="num">{cycle_html}</td></tr>\n'
            f"    <tr><td>Back-translation agreement</td>"
            f'<td class="num">{_fmt_f(report.back_translation_agreement)}</td></tr>\n'
            f"    <tr><td>Flagged segments</td>"
            f'<td class="{flagged_class}">{report.n_flagged_segments}</td></tr>\n'
            "  </table>\n"
            "</section>\n"
        )

    def _html_transition(self, report: EmotionFlowReport) -> str:
        tm = report.transition_matrix
        labels = tm.emotion_labels  # type: ignore[union-attr]
        n = len(labels)

        # Build header row
        header_cells = "".join(
            f"<th>{_esc(l)}</th>" for l in ["from \\ to"] + list(labels)
        )

        # Build data rows
        rows_html: list[str] = []
        for i, from_label in enumerate(labels):
            cells = f"<td><strong>{_esc(from_label)}</strong></td>"
            for j in range(n):
                prob = float(tm.matrix[i, j])  # type: ignore[index]
                intensity = "fail" if prob > 0.4 else ("num" if prob > 0.1 else "num")
                cells += f'<td class="{intensity}">{prob:.3f}</td>'
            rows_html.append(f"    <tr>{cells}</tr>")

        return (
            "<section>\n"
            "  <h2>Transition Matrix</h2>\n"
            f'  <p class="meta">n_transitions = {tm.n_transitions}, '  # type: ignore[union-attr]
            f"smoothing k = {tm.smoothing_k:.2f}</p>\n"  # type: ignore[union-attr]
            "  <table>\n"
            f"    <tr>{header_cells}</tr>\n"
            + "\n".join(rows_html)
            + "\n  </table>\n"
            "</section>\n"
        )

    def _html_null_test(self, report: EmotionFlowReport) -> str:
        nt = report.null_test
        badge_cls = "badge-structured" if nt.is_structured else "badge-not-structured"  # type: ignore[union-attr]
        badge_txt = "STRUCTURED" if nt.is_structured else "NOT STRUCTURED"  # type: ignore[union-attr]

        return (
            "<section>\n"
            "  <h2>Statistical Tests</h2>\n"
            f'  <p><span class="badge {badge_cls}">{badge_txt}</span></p>\n'
            "  <table>\n"
            "    <tr><th>Test</th><th>Statistic</th><th>p-value</th></tr>\n"
            f"    <tr><td>Chi-square</td>"
            f'<td class="num">{_fmt_f(nt.chi2_statistic)}</td>'  # type: ignore[union-attr]
            f"<td>{_fmt_p(nt.chi2_p)}</td></tr>\n"  # type: ignore[union-attr]
            f"    <tr><td>G-test (log-likelihood)</td>"
            f'<td class="num">{_fmt_f(nt.g_statistic)}</td>'  # type: ignore[union-attr]
            f"<td>{_fmt_p(nt.g_test_p)}</td></tr>\n"  # type: ignore[union-attr]
            f"    <tr><td>Permutation (n={nt.n_permutations})</td>"  # type: ignore[union-attr]
            f'<td class="num">&mdash;</td>'
            f"<td>{_fmt_p(nt.permutation_p)}</td></tr>\n"  # type: ignore[union-attr]
            f"    <tr><td>Cram&eacute;r's V</td>"
            f'<td class="num">{_fmt_f(nt.cramers_v)}</td><td>&mdash;</td></tr>\n'  # type: ignore[union-attr]
            f"    <tr><td>Entropy rate (observed)</td>"
            f'<td class="num">{_fmt_f(nt.entropy_observed)}</td><td>&mdash;</td></tr>\n'  # type: ignore[union-attr]
            f"    <tr><td>Entropy rate (null mean)</td>"
            f'<td class="num">{_fmt_f(nt.entropy_random_mean)}</td><td>&mdash;</td></tr>\n'  # type: ignore[union-attr]
            "  </table>\n"
            "</section>\n"
        )

    def _html_semantic(self, report: EmotionFlowReport) -> str:
        sf = report.semantic_flow
        r_class = "pass" if abs(sf.pearson_r) >= 0.3 else "num"  # type: ignore[union-attr]
        return (
            "<section>\n"
            "  <h2>Semantic &amp; Emotion Flow Correlation</h2>\n"
            "  <table>\n"
            "    <tr><th>Metric</th><th>Value</th></tr>\n"
            f"    <tr><td>Pearson r (topic &harr; emotion shifts)</td>"
            f'<td class="{r_class}">{_fmt_f(sf.pearson_r)}</td></tr>\n'  # type: ignore[union-attr]
            f"    <tr><td>p-value</td>"
            f"<td>{_fmt_p(sf.p_value)}</td></tr>\n"  # type: ignore[union-attr]
            f"    <tr><td>Segment pairs</td>"
            f'<td class="num">{sf.n_pairs}</td></tr>\n'  # type: ignore[union-attr]
            "  </table>\n"
            "</section>\n"
        )

    def _html_manifest(self, report: EmotionFlowReport) -> str:
        m = report.manifest
        pkg_rows = "".join(
            f"    <tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>\n"
            for k, v in (m.package_versions.items() if m else {})  # type: ignore[union-attr]
        )
        return (
            "<section>\n"
            "  <h2>Reproducibility</h2>\n"
            "  <table>\n"
            "    <tr><th>Field</th><th>Value</th></tr>\n"
            f"    <tr><td>Seed</td><td>{m.seed}</td></tr>\n"  # type: ignore[union-attr]
            f"    <tr><td>Python</td><td>{_esc(m.python_version)}</td></tr>\n"  # type: ignore[union-attr]
            f"    <tr><td>Git commit</td><td>{_esc(m.git_commit)}</td></tr>\n"  # type: ignore[union-attr]
            f"    <tr><td>Timestamp</td><td>{_esc(m.timestamp)}</td></tr>\n"  # type: ignore[union-attr]
            f"    <tr><td>Hardware</td><td>{_esc(m.hardware)}</td></tr>\n"  # type: ignore[union-attr]
            f"    <tr><td>Deterministic mode</td><td>{'Yes' if m.deterministic_mode else 'No'}</td></tr>\n"  # type: ignore[union-attr]
            + pkg_rows
            + "  </table>\n"
            "</section>\n"
        )

    def _html_narrative(self, report: EmotionFlowReport) -> str:
        return (
            "<section>\n"
            "  <h2>Narrative Summary</h2>\n"
            f'  <pre class="narrative">{_esc(report.narrative)}</pre>\n'
            "</section>\n"
        )

    def _html_footer(self, generated: str) -> str:
        return (
            "<footer>\n"
            f"  Generated {_esc(generated)} &bull; vox_romana emotion_flow pipeline\n"
            "</footer>\n"
        )

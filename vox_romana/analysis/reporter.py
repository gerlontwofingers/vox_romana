"""End-to-end orchestrator and HTML/text report renderer.

:class:`AnalysisReporter` is the single entry point for the full
``vox_romana.analysis`` pipeline.  Pass two :class:`~.models.SpeakerCorpus`
objects and get back a fully-populated :class:`~.models.AnalysisReport` with
an optional self-contained HTML file saved to disk.

QUICK START::

    from vox_romana.analysis import CorpusLoader
    from vox_romana.analysis.reporter import AnalysisReporter

    corpora = CorpusLoader.from_vox_romana_json("output/debates/debate.json")
    a, b = corpora["Alice"], corpora["Bob"]

    reporter = AnalysisReporter(output_dir="output/reports/")
    report   = reporter.build(a, b)
    print(report.hypothesis.verdict)  # "REJECTED" or "NOT_REJECTED"
    print(report.report_path)         # path to the saved HTML file

The ``render_html`` and ``render_text`` methods can also be called
independently on any existing :class:`~.models.AnalysisReport`.
"""

from __future__ import annotations

import html as _html_lib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .embedder import CorpusEmbedder
from .hypothesis import HypothesisTester
from .models import (
    AnalysisReport,
    HypothesisResult,
    SilenceReport,
    SimilarityReport,
    SpeakerCorpus,
    StanceRecord,
)
from .silence import SilenceDetector
from .similarity import SimilarityAnalyzer
from .stance import StanceExtractor
from .topics import TopicModeler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Inline CSS — self-contained academic style, no external resources
# ─────────────────────────────────────────────────────────────────────────────

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
.meta  { font-size: 12px; color: #777; margin-bottom: 6px; }
.speakers { font-size: 15px; font-weight: 600; margin-bottom: 10px; }
.badge {
    display: inline-block; padding: 4px 14px; border-radius: 3px;
    font-weight: 700; font-size: 13px; letter-spacing: .5px;
}
.badge-rejected     { background: #d4edda; color: #155724; }
.badge-not-rejected { background: #fff3cd; color: #856404; }
.badge-high   { background: #f8d7da; color: #721c24; }
.badge-mod    { background: #fff3cd; color: #856404; }
.badge-dist   { background: #d4edda; color: #155724; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 10px; }
th { background: #f5f5f5; border: 1px solid #ddd; padding: 7px 11px;
     text-align: left; font-weight: 600; white-space: nowrap; }
td { border: 1px solid #ddd; padding: 7px 11px; vertical-align: top; }
tr:nth-child(even) td { background: #fafafa; }
.pass { color: #155724; font-weight: 600; }
.fail { color: #721c24; }
.conf { color: #777; font-size: 12px; }
.quote { font-style: italic; color: #555; font-size: 12px; }
.avoidance-list { list-style: none; padding: 0; }
.avoidance-list li { padding: 3px 0; border-bottom: 1px solid #f0f0f0; }
.avoidance-list li:last-child { border-bottom: none; }
.empty  { color: #999; font-style: italic; }
footer { border-top: 1px solid #ddd; padding-top: 14px; margin-top: 40px;
         color: #777; font-size: 11px; line-height: 1.6; }
""".strip()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return _html_lib.escape(str(text), quote=True)


def _slug(text: str) -> str:
    """Convert a speaker name to a filesystem-safe slug."""
    return _SLUG_RE.sub("_", text.lower()).strip("_")


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "&lt;0.001"
    return f"{p:.3f}"


def _fmt_f(v: float) -> str:
    return f"{v:.3f}"


def _verdict_class(verdict: str) -> str:
    v = verdict.upper()
    if v == "REJECTED":
        return "badge-rejected"
    if v == "NOT_REJECTED":
        return "badge-not-rejected"
    # similarity verdicts
    if "high" in v.lower():
        return "badge-high"
    if "mod" in v.lower():
        return "badge-mod"
    return "badge-dist"


# ─────────────────────────────────────────────────────────────────────────────
# AnalysisReporter
# ─────────────────────────────────────────────────────────────────────────────


class AnalysisReporter:
    """Orchestrates the analysis pipeline and renders citable HTML/text reports.

    Args:
        output_dir: If provided, :meth:`build` will save the HTML report here
                    and set ``AnalysisReport.report_path``.  If ``None``,
                    :meth:`build` only returns the in-memory object.
    """

    def __init__(self, *, output_dir: str | Path | None = None) -> None:
        self.output_dir: Path | None = Path(output_dir) if output_dir else None

    # ─────────────────────────────────────────────────────────────────────────
    # Full-pipeline entry point
    # ─────────────────────────────────────────────────────────────────────────

    def build(
        self,
        corpus_a: SpeakerCorpus,
        corpus_b: SpeakerCorpus,
        *,
        hypothesis: HypothesisResult | None = None,
        similarity: SimilarityReport | None = None,
        silence: SilenceReport | None = None,
        stances: list[StanceRecord] | None = None,
        n_topics: int = 10,
        random_state: int = 42,
        n_permutations: int = 1_000,
        use_llm: bool = True,
        embedder: CorpusEmbedder | None = None,
    ) -> AnalysisReport:
        """Run the complete analysis pipeline and return a populated report.

        Any pre-computed component (``hypothesis``, ``similarity``,
        ``silence``, ``stances``) is used as-is; all others are computed
        fresh.

        Args:
            corpus_a:       First speaker's corpus.
            corpus_b:       Second speaker's corpus.
            hypothesis:     Pre-built hypothesis result, or ``None`` to compute.
            similarity:     Pre-built similarity report, or ``None`` to compute.
            silence:        Pre-built silence report, or ``None`` to compute.
            stances:        Pre-built stance records, or ``None`` to compute.
            n_topics:       Number of LDA topics.
            random_state:   RNG seed for reproducibility.
            n_permutations: Permutations for the centroid test.
            use_llm:        If ``False``, skip LLM stance extraction entirely.
            embedder:       Optional pre-configured :class:`~.embedder.CorpusEmbedder`.

        Returns:
            :class:`~.models.AnalysisReport` with all components populated.
            If ``output_dir`` was set, ``report.report_path`` points to the
            saved HTML file.
        """
        logger.info(
            "Building analysis report: %s vs. %s",
            corpus_a.name,
            corpus_b.name,
        )

        # 1 — Embeddings
        emb = embedder or CorpusEmbedder()
        emb_a, emb_b = emb.embed_pair(corpus_a, corpus_b)

        # 2 — Shared topic model (used by silence + stance; optionally reused
        #     by hypothesis if not pre-built)
        tm = TopicModeler(n_topics=n_topics, random_state=random_state)
        dist_a, dist_b = tm.fit_transform(corpus_a, corpus_b)

        # 3 — Hypothesis battery
        if hypothesis is None:
            tester = HypothesisTester(
                n_permutations=n_permutations,
                n_topics=n_topics,
                random_state=random_state,
            )
            hypothesis = tester.run(
                corpus_a, corpus_b, emb_a, emb_b, topic_modeler=tm
            )

        # 4 — Similarity / derivativeness
        if similarity is None:
            similarity = SimilarityAnalyzer().analyze(
                corpus_a, corpus_b, emb_a, emb_b
            )

        # 5 — Silence / topic avoidance
        if silence is None:
            silence = SilenceDetector().detect(
                corpus_a, corpus_b, tm, dist_a, dist_b
            )

        # 6 — Stance extraction (LLM; optional)
        if stances is None:
            if use_llm:
                stances = StanceExtractor().extract(
                    corpus_a, corpus_b, tm, dist_a, dist_b
                )
            else:
                stances = []

        # 7 — Assemble report
        report = AnalysisReport(
            speaker_a=corpus_a.name,
            speaker_b=corpus_b.name,
            hypothesis=hypothesis,
            similarity=similarity,
            silence=silence,
            stances=stances,
        )

        # 8 — Render HTML if output directory configured
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fname = self._report_filename(corpus_a.name, corpus_b.name)
            out_path = self.output_dir / fname
            self.render_html(
                report,
                output_path=out_path,
                corpus_a=corpus_a,
                corpus_b=corpus_b,
            )
            report.report_path = str(out_path)

        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Renderers
    # ─────────────────────────────────────────────────────────────────────────

    def render_html(
        self,
        report: AnalysisReport,
        *,
        output_path: str | Path | None = None,
        corpus_a: SpeakerCorpus | None = None,
        corpus_b: SpeakerCorpus | None = None,
    ) -> str:
        """Render the report as a self-contained HTML document.

        Args:
            report:      The :class:`~.models.AnalysisReport` to render.
            output_path: If provided, write the HTML to this path.
            corpus_a:    Optional corpus for the stats section.
            corpus_b:    Optional corpus for the stats section.

        Returns:
            The full HTML string.
        """
        generated = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
        title = f"Rhetorical Analysis: {_esc(report.speaker_a)} vs. {_esc(report.speaker_b)}"

        sections = []
        sections.append(self._html_header(report, generated))
        if corpus_a is not None and corpus_b is not None:
            sections.append(self._html_corpus_stats(corpus_a, corpus_b))
        sections.append(self._html_hypothesis(report.hypothesis))
        sections.append(self._html_similarity(report.similarity))
        sections.append(self._html_stance(report.stances, report.speaker_a, report.speaker_b))
        sections.append(self._html_silence(report.silence, report.speaker_a, report.speaker_b))
        sections.append(self._html_footer())

        body = "\n".join(sections)
        doc = (
            f"<!DOCTYPE html>\n"
            f'<html lang="en">\n'
            f"<head>\n"
            f'  <meta charset="UTF-8">\n'
            f'  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"  <title>{title}</title>\n"
            f"  <style>{_CSS}</style>\n"
            f"</head>\n"
            f"<body>\n"
            f'<div class="container">\n'
            f"{body}\n"
            f"</div>\n"
            f"</body>\n"
            f"</html>\n"
        )

        if output_path is not None:
            Path(output_path).write_text(doc, encoding="utf-8")
            logger.info("HTML report saved to %s", output_path)

        return doc

    def render_text(
        self,
        report: AnalysisReport,
        *,
        corpus_a: SpeakerCorpus | None = None,
        corpus_b: SpeakerCorpus | None = None,
    ) -> str:
        """Render the report as a plain-text summary.

        Args:
            report:  The report to render.
            corpus_a: Optional corpus for corpus stats.
            corpus_b: Optional corpus for corpus stats.

        Returns:
            Multi-line text string (no ANSI codes).
        """
        sep = "=" * 60
        lines: list[str] = []
        lines.append(sep)
        lines.append("RHETORICAL ANALYSIS REPORT")
        lines.append(sep)
        lines.append(
            f"{report.speaker_a} vs. {report.speaker_b}"
        )
        lines.append(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        lines.append("")

        # Corpus stats
        if corpus_a is not None and corpus_b is not None:
            lines.append("CORPUS STATISTICS")
            lines.append(
                f"  {corpus_a.name}: {corpus_a.utterance_count} utterances, "
                f"{corpus_a.word_count:,} words"
            )
            lines.append(
                f"  {corpus_b.name}: {corpus_b.utterance_count} utterances, "
                f"{corpus_b.word_count:,} words"
            )
            lines.append("")

        # Hypothesis
        if report.hypothesis is not None:
            hyp = report.hypothesis
            lines.append(
                f"HYPOTHESIS BATTERY  [{hyp.verdict}]  "
                f"{hyp.tests_passed}/{hyp.total_tests} tests rejected H0"
            )
            for t in hyp.test_results:
                marker = "PASS" if t.passed else "FAIL"
                lines.append(
                    f"  {marker:<4}  {t.name:<30}  "
                    f"p={t.p_value:.3f}  stat={t.statistic:.3f}  "
                    f"CI=[{t.ci_low:.3f}, {t.ci_high:.3f}]"
                )
            lines.append("")

        # Similarity
        if report.similarity is not None:
            sim = report.similarity
            lines.append(f"SIMILARITY  [{sim.verdict.upper()}]")
            lines.append(f"  N-gram overlap:      {sim.ngram_overlap:.3f}")
            lines.append(f"  Semantic similarity: {sim.semantic_similarity:.3f}")
            lines.append(f"  Burrows' Delta:      {sim.burrows_delta:.3f}")
            lines.append("")

        # Stances
        if report.stances:
            lines.append("STANCE ANALYSIS")
            for rec in report.stances:
                conf = f"conf={rec.confidence:.2f}"
                lines.append(
                    f"  {rec.speaker:<20}  {rec.topic_label:<30}  "
                    f"{rec.position:<10}  {conf}"
                )
            lines.append("")

        # Silence
        if report.silence is not None:
            sil = report.silence
            lines.append("TOPIC AVOIDANCE")
            both = ", ".join(sil.avoided_by_both) or "(none)"
            by_a = ", ".join(sil.avoided_by_a) or "(none)"
            by_b = ", ".join(sil.avoided_by_b) or "(none)"
            lines.append(f"  Avoided by both:          {both}")
            lines.append(f"  Avoided by {report.speaker_a:<12}: {by_a}")
            lines.append(f"  Avoided by {report.speaker_b:<12}: {by_b}")
            if sil.reference_topics_missed:
                missed = ", ".join(sil.reference_topics_missed)
                lines.append(f"  Reference topics missed:  {missed}")
            lines.append("")

        lines.append(sep)
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Private HTML-section builders
    # ─────────────────────────────────────────────────────────────────────────

    def _html_header(self, report: AnalysisReport, generated: str) -> str:
        if report.hypothesis is not None:
            verdict = report.hypothesis.verdict
            v_class = _verdict_class(verdict)
            verdict_html = (
                f'<div style="margin-top:10px">'
                f'<span class="badge {v_class}">{_esc(verdict)}</span>'
                f"</div>"
            )
        else:
            verdict_html = ""

        return (
            f"<header>\n"
            f'  <div class="meta">Generated: {_esc(generated)}</div>\n'
            f'  <h1>Rhetorical Analysis Report</h1>\n'
            f'  <div class="speakers">'
            f'{_esc(report.speaker_a)} &nbsp;vs.&nbsp; {_esc(report.speaker_b)}'
            f"  </div>\n"
            f"  {verdict_html}\n"
            f"</header>"
        )

    def _html_corpus_stats(
        self,
        corpus_a: SpeakerCorpus,
        corpus_b: SpeakerCorpus,
    ) -> str:
        rows = ""
        for c in (corpus_a, corpus_b):
            rows += (
                f"<tr><td>{_esc(c.name)}</td>"
                f"<td>{c.utterance_count:,}</td>"
                f"<td>{c.word_count:,}</td></tr>\n"
            )
        return (
            f'<section id="stats">\n'
            f"  <h2>Corpus Statistics</h2>\n"
            f"  <table>\n"
            f"    <thead><tr><th>Speaker</th><th>Utterances</th><th>Words</th></tr></thead>\n"
            f"    <tbody>{rows}</tbody>\n"
            f"  </table>\n"
            f"</section>"
        )

    def _html_hypothesis(self, hyp: HypothesisResult | None) -> str:
        if hyp is None:
            return ""
        v_class = _verdict_class(hyp.verdict)
        rows = ""
        for t in hyp.test_results:
            marker = f'<span class="pass">&#10003; PASS</span>' if t.passed else f'<span class="fail">&#10007; FAIL</span>'
            rows += (
                f"<tr>"
                f"<td>{_esc(t.name)}</td>"
                f"<td>{_fmt_f(t.statistic)}</td>"
                f"<td>{_fmt_p(t.p_value)}</td>"
                f"<td>{_fmt_f(t.effect_size)}</td>"
                f"<td>[{_fmt_f(t.ci_low)}, {_fmt_f(t.ci_high)}]</td>"
                f"<td>{marker}</td>"
                f"<td>{_esc(t.notes)}</td>"
                f"</tr>\n"
            )
        summary = (
            f"{hyp.tests_passed}/{hyp.total_tests} tests rejected H\u2080 "
            f"(FDR-corrected, threshold \u22653/6)"
        )
        return (
            f'<section id="hypothesis">\n'
            f"  <h2>Hypothesis Battery</h2>\n"
            f'  <p style="margin-bottom:12px">'
            f'    Verdict: <span class="badge {v_class}">{_esc(hyp.verdict)}</span>'
            f"    &nbsp; {_esc(summary)}"
            f"  </p>\n"
            f"  <table>\n"
            f"    <thead><tr>"
            f"<th>Test</th><th>Statistic</th><th>p-value</th>"
            f"<th>Effect Size</th><th>95% CI</th><th>Result</th><th>Notes</th>"
            f"</tr></thead>\n"
            f"    <tbody>{rows}</tbody>\n"
            f"  </table>\n"
            f"</section>"
        )

    def _html_similarity(self, sim: SimilarityReport | None) -> str:
        if sim is None:
            return ""
        v_class = _verdict_class(sim.verdict)
        sty_rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{_esc(str(v))}</td></tr>\n"
            for k, v in (sim.stylometric or {}).items()
        )
        sty_section = (
            f"  <h3 style='font-size:14px;margin:14px 0 8px'>Stylometric Features</h3>\n"
            f"  <table><thead><tr><th>Feature</th><th>Value</th></tr></thead>"
            f"<tbody>{sty_rows}</tbody></table>\n"
            if sty_rows
            else ""
        )
        return (
            f'<section id="similarity">\n'
            f"  <h2>Similarity Analysis</h2>\n"
            f'  <p style="margin-bottom:12px">'
            f'    Verdict: <span class="badge {v_class}">{_esc(sim.verdict)}</span>'
            f"  </p>\n"
            f"  <table>\n"
            f"    <thead><tr><th>Measure</th><th>Score</th></tr></thead>\n"
            f"    <tbody>\n"
            f"      <tr><td>N-gram overlap (Jaccard)</td><td>{_fmt_f(sim.ngram_overlap)}</td></tr>\n"
            f"      <tr><td>Semantic similarity (cosine)</td><td>{_fmt_f(sim.semantic_similarity)}</td></tr>\n"
            f"      <tr><td>Burrows&#39; Delta</td><td>{_fmt_f(sim.burrows_delta)}</td></tr>\n"
            f"    </tbody>\n"
            f"  </table>\n"
            f"  {sty_section}"
            f"</section>"
        )

    def _html_stance(
        self,
        stances: list[StanceRecord],
        speaker_a: str,
        speaker_b: str,
    ) -> str:
        if not stances:
            return (
                f'<section id="stance">\n'
                f"  <h2>Stance Analysis</h2>\n"
                f'  <p class="empty">No stance data (LLM extraction disabled or no topics above threshold).</p>\n'
                f"</section>"
            )
        rows = ""
        for rec in stances:
            pos = _esc(rec.position)
            rows += (
                f"<tr>"
                f"<td>{_esc(rec.speaker)}</td>"
                f"<td>{_esc(rec.topic_label)}</td>"
                f'<td class="{_esc(rec.position)}">{pos}</td>'
                f'<td class="conf">{rec.confidence:.2f}</td>'
                f"<td>{_esc(rec.summary)}</td>"
                f'<td class="quote">{_esc(rec.evidence_quote)}</td>'
                f"</tr>\n"
            )
        return (
            f'<section id="stance">\n'
            f"  <h2>Stance Analysis</h2>\n"
            f"  <table>\n"
            f"    <thead><tr>"
            f"<th>Speaker</th><th>Topic</th><th>Position</th>"
            f"<th>Confidence</th><th>Summary</th><th>Evidence Quote</th>"
            f"</tr></thead>\n"
            f"    <tbody>{rows}</tbody>\n"
            f"  </table>\n"
            f"</section>"
        )

    def _html_silence(
        self,
        silence: SilenceReport | None,
        speaker_a: str,
        speaker_b: str,
    ) -> str:
        if silence is None:
            return ""

        def _list_html(items: list[str], empty_msg: str = "(none)") -> str:
            if not items:
                return f'<span class="empty">{empty_msg}</span>'
            items_html = "".join(f"<li>{_esc(item)}</li>\n" for item in items)
            return f'<ul class="avoidance-list">{items_html}</ul>'

        return (
            f'<section id="silence">\n'
            f"  <h2>Topic Avoidance</h2>\n"
            f"  <table>\n"
            f"    <thead><tr><th>Category</th><th>Topics</th></tr></thead>\n"
            f"    <tbody>\n"
            f"      <tr><td>Avoided by both</td><td>"
            f"{_list_html(silence.avoided_by_both)}</td></tr>\n"
            f"      <tr><td>Avoided by {_esc(speaker_a)}</td><td>"
            f"{_list_html(silence.avoided_by_a)}</td></tr>\n"
            f"      <tr><td>Avoided by {_esc(speaker_b)}</td><td>"
            f"{_list_html(silence.avoided_by_b)}</td></tr>\n"
            + (
                f"      <tr><td>Reference topics missed</td><td>"
                f"{_list_html(silence.reference_topics_missed)}</td></tr>\n"
                if silence.reference_topics_missed
                else ""
            )
            + f"    </tbody>\n"
            f"  </table>\n"
            f"</section>"
        )

    @staticmethod
    def _html_footer() -> str:
        return (
            "<footer>\n"
            "  <strong>Methodology</strong><br>\n"
            "  Corpus embeddings: ChromaDB all-MiniLM-L6-v2 (384-dim) or TF-IDF fallback. "
            "  Topic model: LDA (scikit-learn). "
            "  Hypothesis battery: six non-parametric tests (permutation centroid, "
            "  Mann-Whitney U, KS on topics, JSD bootstrap, Fisher exact, Burrows Delta) "
            "  with Benjamini-Hochberg FDR correction. "
            "  H\u2080 rejected when \u22653/6 tests agree at FDR-corrected p&lt;0.05. "
            "  Stance extraction: DeepSeek LLM with graceful fallback to &ldquo;ambiguous&rdquo;. "
            "  Silence detection: LDA topic-weight thresholds. "
            "  Report generated by vox_romana.analysis.\n"
            "</footer>"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _report_filename(speaker_a: str, speaker_b: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        a = _slug(speaker_a)[:20]
        b = _slug(speaker_b)[:20]
        return f"analysis_{a}_vs_{b}_{ts}.html"

"""Analysis mode runner: rhetorical comparison of two speaker corpora.

Extracted from run.py as part of the cli/ refactor so the logic is
importable and unit-testable independently of the master CLI's argparse
layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_analysis(args: argparse.Namespace) -> None:
    """Run analysis mode: rhetorical comparison of two speaker corpora.

    Loads two speaker corpora (from a vox_romana debate JSON via
    ``--script-input``, or from two text files via the positional
    ``prompt`` and ``--analysis-file-b``), runs the full analysis
    pipeline, and writes a self-contained HTML report.

    Args:
        args: Parsed namespace from run.py's parse_args().
    """
    from vox_romana.analysis.corpus import CorpusLoader
    from vox_romana.analysis.reporter import AnalysisReporter
    from vox_romana.core.paths import OUTPUT_DIR as _ANALYSIS_OUTPUT_DIR

    # ── Resolve input files ──────────────────────────────────────────
    json_path: Path | None = None
    text_a: Path | None = None
    text_b: Path | None = None

    if args.script_input is not None and args.script_input.suffix.lower() == ".json":
        json_path = args.script_input
    elif args.analysis_file_b is not None and args.prompt is not None:
        text_a = args.prompt
        text_b = args.analysis_file_b
    elif args.script_input is not None:
        # Single non-JSON file via --script-input is invalid here
        print(
            "Error: --mode analysis requires either a debate JSON "
            "(--script-input PATH.json) or two text files "
            "(positional FILE_A and --analysis-file-b FILE_B).",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(
            "Error: --mode analysis requires either --script-input "
            "pointing at a debate JSON, or a positional FILE_A plus "
            "--analysis-file-b FILE_B.",
            file=sys.stderr,
        )
        sys.exit(1)

    min_words = args.analysis_min_words

    # ── Load corpora ─────────────────────────────────────────────────
    if json_path is not None:
        corpora = CorpusLoader.from_vox_romana_json(json_path, min_words=min_words)
        speakers = list(corpora.keys())
        if len(speakers) < 2:
            print(
                f"Error: found only {len(speakers)} speaker(s) in {json_path}; "
                "need at least 2 to compare.",
                file=sys.stderr,
            )
            sys.exit(1)

        def _resolve(hint: str | None, exclude: str | None) -> str:
            if hint is not None:
                matches = [s for s in speakers if hint.lower() in s.lower()]
                if not matches:
                    print(
                        f"Error: no speaker matching {hint!r} in {speakers}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                return matches[0]
            for s in speakers:
                if s != exclude:
                    return s
            return speakers[0]

        name_a = _resolve(args.speaker_a, None)
        name_b = _resolve(args.speaker_b, name_a)
        if name_a == name_b:
            print(
                f"Error: speaker A and B resolved to the same speaker: {name_a!r}",
                file=sys.stderr,
            )
            sys.exit(1)

        corpus_a = corpora[name_a]
        corpus_b = corpora[name_b]
    else:
        assert text_a is not None and text_b is not None
        name_a = args.speaker_a or text_a.stem
        name_b = args.speaker_b or text_b.stem
        corpus_a = CorpusLoader.from_text_file(
            text_a, speaker_name=name_a, min_words=min_words
        )
        corpus_b = CorpusLoader.from_text_file(
            text_b, speaker_name=name_b, min_words=min_words
        )

    if not args.quiet:
        print(
            f"  Loaded corpora: {corpus_a.name} ({corpus_a.utterance_count} utts) "
            f"vs. {corpus_b.name} ({corpus_b.utterance_count} utts)"
        )

    # ── Build report ─────────────────────────────────────────────────
    if args.analysis_no_html:
        output_dir = None
    else:
        output_dir = args.analysis_output or (
            _ANALYSIS_OUTPUT_DIR / "text" / "analysis"
        )

    reporter = AnalysisReporter(output_dir=output_dir)
    report = reporter.build(
        corpus_a,
        corpus_b,
        n_topics=args.analysis_n_topics,
        n_permutations=args.analysis_permutations,
        use_llm=not args.analysis_no_llm,
    )

    # ── Output ───────────────────────────────────────────────────────
    if args.analysis_no_html:
        summary = reporter.render_text(report, corpus_a=corpus_a, corpus_b=corpus_b)
        print(summary)

    if not args.quiet:
        if report.report_path:
            print(f"  HTML report \u2192 {report.report_path}")
        if report.hypothesis is not None:
            verdict = report.hypothesis.verdict
            tests_passed = report.hypothesis.tests_passed
            total = report.hypothesis.total_tests
            print(
                f"  Hypothesis verdict: {verdict}  "
                f"({tests_passed}/{total} tests)"
            )

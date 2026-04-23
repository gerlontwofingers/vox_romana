#!/usr/bin/env python3
"""vox_romana CLI — rhetorical analysis of two speaker corpora.

Loads two speaker corpora from a vox_romana JSON debate file (or plain text
files), runs the full analysis pipeline, and writes a self-contained HTML
report to disk.

USAGE::

    # Analyze a completed debate JSON
    python run_analysis.py output/text/debates/ubi_debate.json

    # Specify which speakers to compare (by name substring, case-insensitive)
    python run_analysis.py output/text/debates/ubi.json \\
        --speaker-a "Economist A" --speaker-b "Economist B"

    # Print text summary to stdout without saving HTML
    python run_analysis.py output/text/debates/ubi.json --no-html

    # Save to a custom output directory
    python run_analysis.py output/text/debates/ubi.json \\
        --output-dir output/reports/

    # Faster run — fewer permutations, skip LLM stance extraction
    python run_analysis.py output/text/debates/ubi.json \\
        --permutations 500 --no-llm

    # Analyze two separate text files (one speaker per file)
    python run_analysis.py speaker_a.txt speaker_b.txt \\
        --speaker-a "Alice" --speaker-b "Bob"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_analysis")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_analysis",
        description="Rhetorical analysis of two speaker corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help=(
            "One vox_romana debate JSON (two speakers compared automatically) "
            "or two plain-text files (one speaker each)."
        ),
    )
    p.add_argument(
        "--speaker-a",
        metavar="NAME",
        default=None,
        help="Speaker A name or substring (case-insensitive) when loading from JSON.",
    )
    p.add_argument(
        "--speaker-b",
        metavar="NAME",
        default=None,
        help="Speaker B name or substring (case-insensitive) when loading from JSON.",
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        default="output/text/analysis",
        help="Directory for the HTML report (default: output/text/analysis/).",
    )
    p.add_argument(
        "--n-topics",
        type=int,
        default=10,
        metavar="N",
        help="Number of LDA topics (default: 10).",
    )
    p.add_argument(
        "--permutations",
        type=int,
        default=1_000,
        metavar="N",
        help="Permutations for the centroid test (default: 1000).",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip DeepSeek LLM stance extraction.",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Print text summary to stdout only; do not save an HTML report.",
    )
    p.add_argument(
        "--text",
        action="store_true",
        help="Also print the text summary to stdout (always on when --no-html).",
    )
    p.add_argument(
        "--min-words",
        type=int,
        default=3,
        metavar="N",
        help="Minimum words per utterance when loading corpora (default: 3).",
    )
    return p


def _load_corpora(args: argparse.Namespace):
    """Return (corpus_a, corpus_b) from the provided file(s)."""
    from vox_romana.analysis.corpus import CorpusLoader

    files = [Path(f) for f in args.files]

    if len(files) == 2:
        # Two separate text files — one speaker per file
        name_a = args.speaker_a or files[0].stem
        name_b = args.speaker_b or files[1].stem
        corpus_a = CorpusLoader.from_text_file(
            files[0], speaker_name=name_a, min_words=args.min_words
        )
        corpus_b = CorpusLoader.from_text_file(
            files[1], speaker_name=name_b, min_words=args.min_words
        )
        return corpus_a, corpus_b

    if len(files) == 1:
        path = files[0]
        suffix = path.suffix.lower()

        if suffix == ".json":
            corpora = CorpusLoader.from_vox_romana_json(
                path, min_words=args.min_words
            )
        else:
            logger.error(
                "Single-file input must be a .json debate file.  "
                "For text files, supply two: FILE_A FILE_B."
            )
            sys.exit(1)

        speakers = list(corpora.keys())
        if len(speakers) < 2:
            logger.error(
                "Found only %d speaker(s) in the JSON.  "
                "Need at least 2 to compare.",
                len(speakers),
            )
            sys.exit(1)

        # Resolve speaker-a / speaker-b by substring match
        def _resolve(name_hint: str | None, exclude: str | None) -> str:
            if name_hint is not None:
                matches = [
                    s for s in speakers
                    if name_hint.lower() in s.lower()
                ]
                if not matches:
                    logger.error(
                        "No speaker matching %r in %s",
                        name_hint,
                        speakers,
                    )
                    sys.exit(1)
                return matches[0]
            # default: pick the first (or second) not already chosen
            for s in speakers:
                if s != exclude:
                    return s
            return speakers[0]

        name_a = _resolve(args.speaker_a, None)
        name_b = _resolve(args.speaker_b, name_a)
        if name_a == name_b:
            logger.error("Speaker A and Speaker B resolved to the same speaker: %r", name_a)
            sys.exit(1)

        return corpora[name_a], corpora[name_b]

    logger.error("Supply one JSON file or two text files.")
    sys.exit(1)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Load corpora ──────────────────────────────────────────────────────────
    corpus_a, corpus_b = _load_corpora(args)
    logger.info(
        "Loaded corpora: %s (%d utts) vs. %s (%d utts)",
        corpus_a.name, corpus_a.utterance_count,
        corpus_b.name, corpus_b.utterance_count,
    )

    # ── Build report ──────────────────────────────────────────────────────────
    from vox_romana.analysis.reporter import AnalysisReporter

    output_dir = None if args.no_html else args.output_dir

    reporter = AnalysisReporter(output_dir=output_dir)
    report = reporter.build(
        corpus_a,
        corpus_b,
        n_topics=args.n_topics,
        n_permutations=args.permutations,
        use_llm=not args.no_llm,
    )

    # ── Output ────────────────────────────────────────────────────────────────
    if args.no_html or args.text:
        summary = reporter.render_text(report, corpus_a=corpus_a, corpus_b=corpus_b)
        print(summary)

    if report.report_path:
        logger.info("HTML report: %s", report.report_path)

    if report.hypothesis is not None:
        verdict = report.hypothesis.verdict
        tests_passed = report.hypothesis.tests_passed
        total = report.hypothesis.total_tests
        logger.info(
            "Hypothesis verdict: %s  (%d/%d tests)", verdict, tests_passed, total
        )


if __name__ == "__main__":
    main()

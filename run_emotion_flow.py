#!/usr/bin/env python3
"""vox_romana CLI — Cross-Lingual Emotion Flow Analysis.

Analyse emotion transitions in a speaker's corpus and produce an
academic-grade report with publication-ready visualisations.

WHAT IT DOES:

    1. Load speech segments from a text file, a directory of text files,
       or a vox_romana JSON (debate/news run output).
    2. Translate segments to English (Helsinki-NLP opus-mt models).
    3. Tag each segment with 7-class emotion scores
       (j-hartmann/emotion-english-distilroberta-base).
    4. Build a first-order Markov transition matrix with Wilson-score CIs.
    5. Optionally compute Pearson r between topic drift and emotion drift
       (requires sentence-transformers / CorpusEmbedder).
    6. Run chi-square, G-test, and permutation null-hypothesis tests.
    7. Render HTML + JSON + CSV reports and PNG/HTML visualisations.

USAGE:

    # Single text file — infers speaker name from filename:
    python run_emotion_flow.py speeches/caesar.txt

    # Directory of .txt files, explicit leader name:
    python run_emotion_flow.py speeches/ --leader "Caesar"

    # vox_romana debate JSON (multi-speaker — leader arg required):
    python run_emotion_flow.py output/debates/debate.json --leader "Tealc"

    # Skip translation (already-English corpus):
    python run_emotion_flow.py speeches/caesar.txt --no-translate

    # Skip semantic embedding (faster, no sentence-transformers needed):
    python run_emotion_flow.py speeches/caesar.txt --no-semantic

    # Custom output directory + seed:
    python run_emotion_flow.py speeches/caesar.txt \\
        --output output/emotion_flow/ --seed 123

    # More permutations for publication-quality p-values:
    python run_emotion_flow.py speeches/caesar.txt --permutations 10000

    # Quiet mode — only print the report path:
    python run_emotion_flow.py speeches/caesar.txt --quiet

OUTPUT:

    All files are saved under:
        {output-dir}/{speaker_slug}_{timestamp}/

    Files:
        report.html      — self-contained academic HTML report
        report.json      — full structured results (all stats + metadata)
        report.csv       — transition probability matrix (from × to)
        heatmap.png      — seaborn transition heatmap
        sankey.html      — plotly Sankey emotion flow diagram
        entropy_rate.png — observed entropy vs. null CI plot

ENVIRONMENT:

    No API keys required for offline corpora.
    HuggingFace models are downloaded on first run and cached in ~/.cache/
    (override with --cache-dir).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vox_romana.core.paths import OUTPUT_DIR as _OUTPUT_DIR


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_emotion_flow",
        description="Cross-Lingual Emotion Flow Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Positional
    p.add_argument(
        "source",
        type=Path,
        help="Path to a .txt file, directory of .txt files, or vox_romana JSON.",
    )

    # Core options
    p.add_argument(
        "--leader",
        metavar="NAME",
        default=None,
        help=(
            "Speaker/leader name filter.  Required for multi-speaker JSON. "
            "Inferred from filename when omitted."
        ),
    )
    p.add_argument(
        "--output",
        metavar="DIR",
        type=Path,
        default=_OUTPUT_DIR / "emotion_flow",
        help=f"Output directory.  (default: {_OUTPUT_DIR / 'emotion_flow'})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="RNG seed for reproducibility.  (default: 42)",
    )
    p.add_argument(
        "--permutations",
        type=int,
        default=1000,
        metavar="N",
        help="Number of permutation draws for null-hypothesis test.  (default: 1000)",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=20,
        metavar="N",
        help="Minimum segment character count; shorter segments are skipped.  (default: 20)",
    )

    # Feature flags
    p.add_argument(
        "--no-translate",
        action="store_true",
        default=False,
        help="Skip translation step (treat corpus as already English).",
    )
    p.add_argument(
        "--no-semantic",
        action="store_true",
        default=False,
        help="Skip semantic-flow correlation (no sentence-transformers needed).",
    )

    # Model / caching
    p.add_argument(
        "--cache-dir",
        metavar="DIR",
        type=Path,
        default=None,
        help="HuggingFace model cache directory.  (default: system default ~/.cache/)",
    )

    # Output control
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress progress output; only print the final report path.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return p


def _setup_logging(quiet: bool, verbose: bool) -> None:
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.quiet, args.verbose)
    logger = logging.getLogger("run_emotion_flow")

    # Validate source path
    source: Path = args.source
    if not source.exists():
        print(f"ERROR: source path does not exist: {source}", file=sys.stderr)
        return 1

    # Import pipeline here to keep startup fast (heavy deps load lazily)
    from vox_romana.emotion_flow.pipeline import EmotionFlowPipeline

    pipeline = EmotionFlowPipeline(
        output_dir=args.output,
        seed=args.seed,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        n_permutations=args.permutations,
        run_semantic=not args.no_semantic,
        run_translation=not args.no_translate,
    )

    if not args.quiet:
        leader_display = args.leader or source.stem
        print(f"Analysing: {source}  (leader={leader_display!r})")
        print(f"Output directory: {args.output}")
        print(f"Seed: {args.seed}  |  Permutations: {args.permutations}")
        if args.no_translate:
            print("Translation: OFF")
        if args.no_semantic:
            print("Semantic flow: OFF")
        print()

    try:
        report = pipeline.run(
            source,
            leader=args.leader,
            min_chars=args.min_chars,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Pipeline failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print()
        _print_summary(report)
        print()

    if report.report_path:
        print(report.report_path)
    else:
        print("(no output dir — report not saved)")

    return 0


def _print_summary(report: Any) -> None:  # type: ignore[name-defined]
    """Print a one-page text summary of the report."""
    sep = "─" * 60
    print(sep)
    print(f"EMOTION FLOW REPORT — {report.leader_name}")
    print(sep)
    print(f"  Speeches : {report.n_speeches}")
    print(f"  Segments : {report.n_segments}")
    print(f"  Flagged  : {report.n_flagged_segments}")
    print(f"  BT agree : {report.back_translation_agreement:.3f}")

    if report.dominant_cycle:
        cycle_str = " → ".join(report.dominant_cycle[:10])
        if len(report.dominant_cycle) > 10:
            cycle_str += " …"
        print(f"  Cycle    : {cycle_str}")

    if report.null_test is not None:
        nt = report.null_test
        structured = "YES" if nt.is_structured else "NO"
        print(f"\n  Structured transitions : {structured}")
        print(f"  Chi-square p           : {nt.chi2_p:.4f}")
        print(f"  G-test p               : {nt.g_test_p:.4f}")
        print(f"  Permutation p          : {nt.permutation_p:.4f}")
        print(f"  Cramér's V             : {nt.cramers_v:.4f}")
        print(f"  Entropy observed       : {nt.entropy_observed:.4f}")

    if report.semantic_flow is not None:
        sf = report.semantic_flow
        print(f"\n  Semantic-emotion r     : {sf.pearson_r:.4f}")
        print(f"  p-value                : {sf.p_value:.4f}")

    if report.transition_matrix is not None:
        tm = report.transition_matrix
        print(f"\n  Transitions            : {tm.n_transitions}")

    print(sep)


# Support both `python run_emotion_flow.py` and `python -m vox_romana.emotion_flow`
from typing import Any  # noqa: E402 — import below __main__ guard intentional

if __name__ == "__main__":
    sys.exit(main())

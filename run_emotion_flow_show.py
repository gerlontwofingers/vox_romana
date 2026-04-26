#!/usr/bin/env python3
"""run_emotion_flow_show.py — EmotionFlowReport → NewsConfig → show JSON.

Takes the report.json produced by run_emotion_flow.py and converts it into
a pre-filled NewsConfig that can be fed directly into the news show pipeline,
skipping the bootstrap/architect stage entirely.

    report.json → EmotionFlowShowAdapter → NewsConfig → (optionally) show JSON

USAGE:
    # Adapt only — writes show_config.json to the output directory:
    python run_emotion_flow_show.py output/emotion_flow/slug_ts/report.json

    # Adapt + run show pipeline (requires DEEPSEEK_API_KEY):
    python run_emotion_flow_show.py report.json --run

    # Override segment count and output location:
    python run_emotion_flow_show.py report.json --n-segments 5 --output shows/

OUTPUT FILES:
    <output>/show_config.json   — the adapted NewsConfig (always written)
    <output>/show.json          — full show JSON for render_show.py (only with --run)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from vox_romana.core.paths import OUTPUT_DIR as _OUTPUT_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_emotion_flow_show",
        description="Convert an EmotionFlowReport into a NewsConfig (and optionally a show).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_emotion_flow_show.py output/emotion_flow/slug/report.json\n"
            "  python run_emotion_flow_show.py report.json --run --output shows/\n"
            "  python run_emotion_flow_show.py report.json --n-segments 5 --quiet\n"
        ),
    )
    parser.add_argument(
        "report",
        help="Path to report.json produced by run_emotion_flow.py.",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="DIR",
        default=str(_OUTPUT_DIR / "shows" / "emotion_flow"),
        help=f"Output directory for config and show files. (default: {_OUTPUT_DIR / 'shows' / 'emotion_flow'})",
    )
    parser.add_argument(
        "--n-segments",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of show segments. (default: 6)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        metavar="MIN",
        help="Target show duration in minutes. (default: 10)",
    )
    parser.add_argument(
        "--reporter-name",
        default="Emotion Flow",
        metavar="NAME",
        help="Show / reporter name used in the title. (default: 'Emotion Flow')",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Also run the show pipeline to generate show.json (requires DEEPSEEK_API_KEY).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: report file not found: {report_path}", file=sys.stderr)
        return 1

    # Lazy import so the module is importable even without heavy deps installed.
    from vox_romana.emotion_flow.show_adapter import EmotionFlowShowAdapter

    adapter = EmotionFlowShowAdapter(
        reporter_name=args.reporter_name,
        n_segments=args.n_segments,
        target_duration_minutes=args.duration,
    )

    if not args.quiet:
        print(f"Loading report: {report_path}")

    config = adapter.adapt_from_file(report_path)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always write the adapted NewsConfig.
    config_path = output_dir / "show_config.json"
    config_dict = asdict(config)
    config_path.write_text(
        json.dumps(config_dict, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if not args.quiet:
        print(f"Show config written to: {config_path}")
        _print_summary(config, verbose=args.verbose)

    if args.run:
        return _run_show(config, args, output_dir)

    return 0


def _print_summary(config: object, *, verbose: bool = False) -> None:
    from vox_romana.shows.news.config import NewsConfig

    if not isinstance(config, NewsConfig):
        return

    print()
    print(f"  Title:    {config.title}")
    print(f"  Tone:     {config.tone}")
    print(f"  Anchors:  {', '.join(a.name for a in config.anchors)}")
    print(f"  Segments: {len(config.segments)}")
    for i, seg in enumerate(config.segments, 1):
        tone_label = seg.tone or "neutral"
        print(f"    {i}. [{tone_label}] {seg.headline}")
    if config.director_brief and verbose:
        print()
        arc = config.director_brief.show_arc
        print(f"  Arc: {arc[:160]}{'...' if len(arc) > 160 else ''}")
    print()


def _run_show(config: object, args: argparse.Namespace, output_dir: Path) -> int:
    """Run the full show pipeline on the pre-filled NewsConfig."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from vox_romana.core.client import DeepSeekClient
        from vox_romana.shows.news.conductor import NewsConductor
        from vox_romana.shows.news.serializer import NewsSerializer
    except ImportError as exc:
        print(f"Error: cannot import show pipeline: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("Running show pipeline…")

    client = DeepSeekClient()
    conductor = NewsConductor(config, client=client)  # type: ignore[arg-type]
    result = conductor.run()
    merged = result.to_merged_news_show()

    show_path = output_dir / "show.json"
    NewsSerializer().save(merged, show_path)

    if not args.quiet:
        print(f"Show written to: {show_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

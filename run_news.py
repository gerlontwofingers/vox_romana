#!/usr/bin/env python3
"""run_news.py — generate a news show from a prompt file.

Same pattern as run_debate.py: hand it a text file with a brief,
get a JSON file with a complete news broadcast ready for TTS.

    prompt file → bootstrap (config) → conductor (news show) → JSON output

USAGE:
    python run_news.py prompts/tech_news.txt
    python run_news.py prompts/breaking.txt --research
    python run_news.py prompts/weekly_roundup.txt --segments 6 --tone "conversational"

PROMPT FILE FORMAT:
    Plain text. Describe the news show you want in natural language.
    Example:
        Top 5 AI stories this week. Cover the major model releases,
        regulatory developments, and one human interest story.
        Authoritative but accessible tone. Think NPR meets tech podcast.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from vox_romana.core.client import DeepSeekClient
from vox_romana.shows.news.bootstrap import NewsBootstrap
from vox_romana.shows.news.conductor import NewsConductor
from vox_romana.shows.news.serializer import NewsSerializer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a news show from a prompt file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_news.py prompts/tech_news.txt\n"
            "  python run_news.py prompts/breaking.txt --research\n"
            "  python run_news.py prompts/weekly.txt --segments 6\n"
        ),
    )

    parser.add_argument(
        "prompt",
        type=Path,
        help="Path to a plain-text prompt file describing the news show.",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Directory to save the output JSON (default: output/)",
    )

    parser.add_argument(
        "--research",
        action="store_true",
        help="Auto-research topics via Google before generating.",
    )

    parser.add_argument(
        "--no-memory",
        action="store_false",
        dest="memory",
        default=True,
        help="Disable memory-first lookup during --research. By default, the "
        "local knowledge base (Chroma + FTS5) is consulted before any web "
        "search; use this flag to force web-only research.",
    )

    parser.add_argument(
        "--segments",
        type=int,
        default=None,
        help="Override number of segments (default: 5).",
    )

    parser.add_argument(
        "--tone",
        type=str,
        default=None,
        help="Override broadcast tone (e.g., 'urgent', 'conversational').",
    )

    parser.add_argument(
        "--tag", "-t",
        type=str,
        default=None,
        help="Label baked into the output filename.",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    return parser.parse_args()


def log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"  {msg}", flush=True)


def main() -> None:
    args = parse_args()

    if not args.prompt.exists():
        print(f"Error: prompt file not found: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    brief = args.prompt.read_text().strip()
    if not brief:
        print(f"Error: prompt file is empty: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    overrides = {}
    if args.segments is not None:
        overrides["target_segments"] = args.segments
    if args.tone is not None:
        overrides["tone"] = args.tone

    # ── Phase 1: Bootstrap ───────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  VOX ROMANA — News Show Generator")
    print(f"{'═' * 60}")
    log(f"Prompt: {args.prompt}", quiet=args.quiet)
    if args.tag:
        log(f"Tag:    {args.tag}", quiet=args.quiet)
    log(f"Brief:  {brief[:80]}{'...' if len(brief) > 80 else ''}", quiet=args.quiet)
    print()

    client = DeepSeekClient()

    log("Phase 1: Bootstrap (rundown planning, anchor profiling)...", quiet=args.quiet)
    t0 = time.time()

    bootstrap_kwargs = dict(
        reference_material=None,
        fact_check="off",
        overrides=overrides,
        auto_research=False,
    )

    if args.research:
        bootstrap_kwargs["auto_research"] = True
        if args.memory:
            try:
                from vox_romana.core.catalog import Catalog
                from vox_romana.memory.recall import MemoryRecall
                catalog = Catalog()
                memory = MemoryRecall(catalog=catalog)
                bootstrap_kwargs["memory"] = memory
                log("  Memory-first search: ON", quiet=args.quiet)
            except Exception as e:
                log(f"  Memory-first search: OFF (init failed: {e})", quiet=args.quiet)
        try:
            from vox_romana.core.search import GoogleSearchProvider
            search_provider = GoogleSearchProvider()
            bootstrap_kwargs["search_provider"] = search_provider
            log("  Research mode ON — will search Google first", quiet=args.quiet)
        except ValueError as e:
            log(f"  Warning: --research requested but Google keys missing: {e}", quiet=args.quiet)
            log(f"  Continuing with memory + LLM only...", quiet=args.quiet)

    bootstrap = NewsBootstrap(client=client)
    config = bootstrap.from_brief(brief, **bootstrap_kwargs)

    t1 = time.time()
    log(f"  Done in {t1 - t0:.1f}s", quiet=args.quiet)
    log(f"  Title: {config.title}", quiet=args.quiet)
    anchor_names = ", ".join(a.name for a in config.anchors)
    log(f"  Anchors: {anchor_names}", quiet=args.quiet)
    log(f"  Segments: {len(config.segments)} | Tone: {config.tone}", quiet=args.quiet)
    print()

    # ── Phase 2: Conductor ───────────────────────────────────────────
    log("Phase 2: Conductor (generating the broadcast)...", quiet=args.quiet)
    t2 = time.time()

    conductor = NewsConductor(config, client=client)
    result = conductor.run()

    t3 = time.time()
    log(f"  Done in {t3 - t2:.1f}s ({result.total_llm_calls} LLM calls)", quiet=args.quiet)
    log(f"  Generated {len(result.news_events)} events across {len(result.news_segments)} segments", quiet=args.quiet)
    print()

    # ── Phase 3: Serialize ───────────────────────────────────────────
    log("Phase 3: Saving output...", quiet=args.quiet)

    merged = result.to_merged_news_show()
    serializer = NewsSerializer()

    stem = args.prompt.stem
    tag_part = f"_{args.tag}" if args.tag else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{stem}{tag_part}_{timestamp}.json"
    output_path = args.output / filename

    serializer.save(merged, output_path)

    print()
    print(f"{'═' * 60}")
    log(f"Output saved to: {output_path}", quiet=args.quiet)
    log(f"Total time: {t3 - t0:.1f}s", quiet=args.quiet)
    print(f"{'═' * 60}")
    print()

    if args.quiet:
        print(output_path)


if __name__ == "__main__":
    main()

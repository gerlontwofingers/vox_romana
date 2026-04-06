#!/usr/bin/env python3
"""vox_romana CLI — generate any show type from a prompt file.

Unified entry point. Hand it a brief and it figures out the rest:

    prompt file → router (classify) → bootstrap → conductor → JSON output

The router reads your brief and decides whether it's a debate, a news
show, or (eventually) something else. Then it dispatches to the right
pipeline. You never need to choose between run_debate.py and run_news.py.

USAGE:
    # Just point at a prompt file — the router figures out the rest:
    python run_show.py prompts/ubi_debate.txt
    python run_show.py prompts/tech_news.txt

    # Force a show type (skip the router):
    python run_show.py prompts/ambiguous.txt --type debate
    python run_show.py prompts/ambiguous.txt --type news

    # With research, tags, overrides:
    python run_show.py prompts/ubi_debate.txt --research --tag batch_01
    python run_show.py prompts/tech_news.txt --segments 6 --tone urgent

    # Quiet mode:
    python run_show.py prompts/ubi_debate.txt --quiet

PROMPT FILE FORMAT:
    Plain text. No special syntax. Describe what you want in natural
    language. The router + architect will handle everything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from vox_romana.core.client import DeepSeekClient
from vox_romana.core.router import ShowRouter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_show",
        description="Generate a show (debate, news, etc.) from a prompt file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_show.py prompts/ubi_debate.txt\n"
            "  python run_show.py prompts/tech_news.txt --research\n"
            "  python run_show.py prompts/ambiguous.txt --type debate\n"
            "  python run_show.py prompts/topic.txt --tag v2 --quiet\n"
        ),
    )

    parser.add_argument(
        "prompt",
        type=Path,
        help="Path to a plain-text prompt file.",
    )

    parser.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["debate", "news"],
        dest="show_type",
        help="Force a show type (skip the router).",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Directory to save the output JSON (default: output/).",
    )

    parser.add_argument(
        "--research",
        action="store_true",
        default=None,
        dest="research",
        help="Force research ON. By default the router auto-detects from "
        "the brief whether research is needed.",
    )

    parser.add_argument(
        "--no-research",
        action="store_false",
        dest="research",
        help="Force research OFF (skip even if the router recommends it).",
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

    # ── Debate-specific overrides ────────────────────────────────────
    debate_group = parser.add_argument_group("debate overrides")
    debate_group.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Override target debate turns.",
    )
    debate_group.add_argument(
        "--tone",
        type=str,
        default=None,
        help="Override tone (works for both debates and news).",
    )
    debate_group.add_argument(
        "--format",
        type=str,
        default=None,
        dest="debate_format",
        help="Override debate format (e.g. 'oxford', 'crossfire').",
    )

    # ── News-specific overrides ──────────────────────────────────────
    news_group = parser.add_argument_group("news overrides")
    news_group.add_argument(
        "--segments",
        type=int,
        default=None,
        help="Override number of news segments.",
    )

    return parser.parse_args(argv)


def log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"  {msg}", flush=True)


def _run_debate(brief: str, client: DeepSeekClient, args: argparse.Namespace) -> Path:
    """Run the debate pipeline, return the output path."""
    from vox_romana.shows.debates.bootstrap import DebateBootstrap
    from vox_romana.shows.debates.conductor import DebateConductor
    from vox_romana.shows.debates.serializer import DebateSerializer

    overrides = {}
    if args.turns is not None:
        overrides["target_turns"] = args.turns
    if args.tone is not None:
        overrides["tone"] = args.tone
    if args.debate_format is not None:
        overrides["debate_format"] = args.debate_format

    # ── Bootstrap ────────────────────────────────────────────────────
    log("Phase 2: Bootstrap (casting, profiling, directing)...", quiet=args.quiet)
    t0 = time.time()

    bootstrap_kwargs = dict(
        reference_material=None,
        fact_check="off",
        overrides=overrides,
        auto_research=False,
    )

    if args.research:
        bootstrap_kwargs["auto_research"] = True
        try:
            from vox_romana.core.search import GoogleSearchProvider
            search_provider = GoogleSearchProvider()
            bootstrap_kwargs["search_provider"] = search_provider
            log("  Research mode ON (Google)", quiet=args.quiet)
        except (ValueError, ImportError):
            log("  Research mode ON (LLM knowledge — no Google keys)", quiet=args.quiet)

    bootstrap = DebateBootstrap(client=client)
    config = bootstrap.from_brief(brief, **bootstrap_kwargs)

    t1 = time.time()
    log(f"  Done in {t1 - t0:.1f}s", quiet=args.quiet)
    log(f"  Title: {config.title}", quiet=args.quiet)
    log(f"  Participants: {', '.join(p.name for p in config.participants)}", quiet=args.quiet)
    log(f"  Format: {config.debate_format} | Tone: {config.tone} | Turns: {config.target_turns}", quiet=args.quiet)
    print()

    # ── Conductor ────────────────────────────────────────────────────
    log("Phase 3: Conductor (running the debate)...", quiet=args.quiet)
    t2 = time.time()

    conductor = DebateConductor(config, client=client)
    result = conductor.run()

    t3 = time.time()
    log(f"  Done in {t3 - t2:.1f}s ({result.total_llm_calls} LLM calls)", quiet=args.quiet)
    log(f"  Generated {len(result.debate_events)} events", quiet=args.quiet)
    print()

    # ── Serialize ────────────────────────────────────────────────────
    log("Phase 4: Saving output...", quiet=args.quiet)

    merged = result.to_merged_debate()
    serializer = DebateSerializer()

    stem = args.prompt.stem
    tag_part = f"_{args.tag}" if args.tag else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{stem}{tag_part}_{timestamp}.json"
    output_path = args.output / filename

    serializer.save_merged(merged, output_path)
    return output_path


def _run_news(brief: str, client: DeepSeekClient, args: argparse.Namespace) -> Path:
    """Run the news pipeline, return the output path."""
    from vox_romana.shows.news.bootstrap import NewsBootstrap
    from vox_romana.shows.news.conductor import NewsConductor
    from vox_romana.shows.news.serializer import NewsSerializer

    overrides = {}
    if args.segments is not None:
        overrides["target_segments"] = args.segments
    if args.tone is not None:
        overrides["tone"] = args.tone

    # ── Bootstrap ────────────────────────────────────────────────────
    log("Phase 2: Bootstrap (rundown planning, anchor profiling)...", quiet=args.quiet)
    t0 = time.time()

    bootstrap_kwargs = dict(
        reference_material=None,
        fact_check="off",
        overrides=overrides,
        auto_research=False,
    )

    if args.research:
        bootstrap_kwargs["auto_research"] = True
        try:
            from vox_romana.core.search import GoogleSearchProvider
            search_provider = GoogleSearchProvider()
            bootstrap_kwargs["search_provider"] = search_provider
            log("  Research mode ON (Google)", quiet=args.quiet)
        except (ValueError, ImportError):
            log("  Research mode ON (LLM knowledge — no Google keys)", quiet=args.quiet)

    bootstrap = NewsBootstrap(client=client)
    config = bootstrap.from_brief(brief, **bootstrap_kwargs)

    t1 = time.time()
    log(f"  Done in {t1 - t0:.1f}s", quiet=args.quiet)
    log(f"  Title: {config.title}", quiet=args.quiet)
    anchor_names = ", ".join(a.name for a in config.anchors)
    log(f"  Anchors: {anchor_names}", quiet=args.quiet)
    log(f"  Segments: {len(config.segments)} | Tone: {config.tone}", quiet=args.quiet)
    print()

    # ── Conductor ────────────────────────────────────────────────────
    log("Phase 3: Conductor (generating the broadcast)...", quiet=args.quiet)
    t2 = time.time()

    conductor = NewsConductor(config, client=client)
    result = conductor.run()

    t3 = time.time()
    log(f"  Done in {t3 - t2:.1f}s ({result.total_llm_calls} LLM calls)", quiet=args.quiet)
    log(f"  Generated {len(result.news_events)} events across {len(result.news_segments)} segments", quiet=args.quiet)
    print()

    # ── Serialize ────────────────────────────────────────────────────
    log("Phase 4: Saving output...", quiet=args.quiet)

    merged = result.to_merged_news_show()
    serializer = NewsSerializer()

    stem = args.prompt.stem
    tag_part = f"_{args.tag}" if args.tag else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{stem}{tag_part}_{timestamp}.json"
    output_path = args.output / filename

    serializer.save(merged, output_path)
    return output_path


# ── Dispatch table ───────────────────────────────────────────────────────
_RUNNERS = {
    "debate": _run_debate,
    "news": _run_news,
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.prompt.exists():
        print(f"Error: prompt file not found: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    brief = args.prompt.read_text().strip()
    if not brief:
        print(f"Error: prompt file is empty: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    client = DeepSeekClient()

    # ── Phase 1: Route ───────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  VOX ROMANA — Show Generator")
    print(f"{'═' * 60}")
    log(f"Prompt: {args.prompt}", quiet=args.quiet)
    if args.tag:
        log(f"Tag:    {args.tag}", quiet=args.quiet)
    log(f"Brief:  {brief[:80]}{'...' if len(brief) > 80 else ''}", quiet=args.quiet)
    print()

    if args.show_type:
        show_type = args.show_type
        log(f"Phase 1: Show type forced → {show_type}", quiet=args.quiet)
    else:
        log("Phase 1: Classifying brief...", quiet=args.quiet)
        t_route = time.time()

        router = ShowRouter(client)
        result = router.classify(brief)

        log(
            f"  Classified as: {result.show_type} "
            f"(confidence: {result.confidence:.0%})",
            quiet=args.quiet,
        )
        if result.needs_research:
            log("  Research recommended: yes", quiet=args.quiet)
        log(f"  Reasoning: {result.reasoning}", quiet=args.quiet)
        log(f"  Done in {time.time() - t_route:.1f}s", quiet=args.quiet)
        show_type = result.show_type

        # If user didn't explicitly pass --research or --no-research,
        # let the router decide.
        if args.research is None:
            args.research = result.needs_research

    print()

    # ── Phase 2–4: Run the pipeline ──────────────────────────────────
    runner = _RUNNERS[show_type]
    t_start = time.time()
    output_path = runner(brief, client, args)
    t_end = time.time()

    print()
    print(f"{'═' * 60}")
    log(f"Show type: {show_type}", quiet=args.quiet)
    log(f"Output:    {output_path}", quiet=args.quiet)
    log(f"Total:     {t_end - t_start:.1f}s", quiet=args.quiet)
    print(f"{'═' * 60}")
    print()

    if args.quiet:
        print(output_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""vox_romana CLI — generate any show type from a prompt file.

Unified entry point that replaces run_debate.py and run_news.py with a
single command. Hand it a brief and the system figures out the rest:

    prompt file → router → bootstrap → conductor → editor → JSON output

THE BROADCAST PRODUCTION ANALOGY:

    Think of vox_romana as a television network's production pipeline:

    1. YOUR PROMPT FILE is the show pitch. You walk into the network
       office, drop a one-pager on the desk, and say "make this."
       Natural language. No special format. Just describe the show.

    2. THE ROUTER is the programming scheduler. It reads your pitch and
       decides which studio floor should produce it — Studio A (debates)
       or Studio B (news). If the pitch is ambiguous, you can force the
       assignment with --type. The scheduler also flags whether the show
       needs a research team (fact-checking, web lookups).

    3. THE BOOTSTRAP is pre-production. A team of AI agents reads your
       pitch and does all the planning you didn't explicitly specify:
       casting (who speaks), profiling (their personalities), directing
       (the dramatic arc), and runsheet planning (segment order, beats).
       You control as much or as little as you want — leave things out
       and the team fills in smart defaults.

    4. THE CONDUCTOR is the live recording. Each agent takes their turn
       at the mic, the moderator/anchor keeps things moving, and the
       director checks in periodically for pacing adjustments.

    5. THE EDITOR is the quality control booth. After each chunk of
       content is generated, the editor reviews it for speaker clarity,
       repetition, and narrative quality. If a chunk fails review, the
       conductor regenerates it with detailed feedback — up to
       --max-retries times. This is the new editing pipeline from the
       design spec (see agent_instructions.txt sections E-F).

    6. THE JSON OUTPUT is the final cut — ready for the sound booth
       (render_show.py) to produce audio.

USAGE:

    # The simplest case — the router auto-detects show type:
    python run_show.py prompts/ubi_debate.txt
    python run_show.py prompts/tech_news.txt

    # Force a show type (bypass the router):
    python run_show.py prompts/ambiguous_topic.txt --type debate
    python run_show.py prompts/ambiguous_topic.txt --type news
    python run_show.py prompts/podcast/soccer.txt --type podcast

    # Enable web research (Google if keys are set, LLM fallback otherwise):
    python run_show.py prompts/ubi_debate.txt --research
    python run_show.py prompts/tech_news.txt --research

    # Explicitly disable research even if the router recommends it:
    python run_show.py prompts/ubi_debate.txt --no-research

    # Override parameters that the Architect would normally decide:
    python run_show.py prompts/ubi_debate.txt --turns 50 --tone "cable news"
    python run_show.py prompts/tech_news.txt --segments 6 --tone "urgent"

    # Control the editing pipeline:
    python run_show.py prompts/ubi_debate.txt --max-retries 5
    python run_show.py prompts/quick_test.txt --max-retries 0

    # Tag outputs for batch management:
    python run_show.py prompts/ubi_debate.txt --tag batch_01
    python run_show.py prompts/ubi_debate.txt --tag experiment_high_temp

    # Quiet mode (just the output path — ideal for scripting):
    python run_show.py prompts/ubi_debate.txt --quiet

    # Combine freely:
    python run_show.py prompts/ubi_debate.txt \
        --research --turns 50 --tone "unhinged" --tag v3 --max-retries 4

    # Pipe into the renderer for a complete text-to-speech pipeline:
    python run_show.py prompts/ubi_debate.txt --quiet | xargs python render_show.py

PROMPT FILE FORMAT:

    Plain text. No special syntax. No YAML, no JSON, no tags.
    Just describe the show you want, like you're pitching it to a
    producer. The router and architect will handle everything.

    Example (debate):
        Four economists debate UBI in an Oxford-style format.
        Two pro, two con. Aggressive moderator who interrupts.
        50 turns. Cable news energy but with real data.

    Example (news):
        Top 5 AI stories this week. Cover the major model releases,
        regulatory developments, and one human interest story.
        Authoritative but accessible. Think NPR meets tech podcast.

    The more detail you put in, the more control you have. The less
    detail, the more creative freedom the AI gets. Both are valid.

ENVIRONMENT:
    Requires DEEPSEEK_API_KEY in .env (or exported in shell).
    Optional: GOOGLE_API_KEY + GOOGLE_CSE_ID for --research mode.

OUTPUT:
    Saves a JSON file to the output directory. The filename encodes
    the prompt stem, optional tag, and a timestamp:
        {prompt_stem}_{tag}_{YYYYMMDD_HHMMSS}.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from vox_romana.core.client import DeepSeekClient
from vox_romana.core.duration import estimate_total_words, parse_duration_from_text
from vox_romana.core.router import ShowRouter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    The only REQUIRED argument is the path to a prompt file. Everything
    else is optional — the system infers what it needs from the brief.

    Override flags (--turns, --tone, --segments, etc.) are applied AFTER
    the Architect reads your prompt. Think of them as last-minute notes
    slipped to the production team: "Actually, make it 50 turns."

    Args:
        argv: Arguments to parse. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed namespace with all options.
    """
    parser = argparse.ArgumentParser(
        prog="run_show",
        description="Generate a show (debate, news, etc.) from a prompt file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Auto-detect show type from the brief:\n"
            "  python run_show.py prompts/ubi_debate.txt\n"
            "  python run_show.py prompts/tech_news.txt\n"
            "\n"
            "  # Force show type:\n"
            "  python run_show.py prompts/ambiguous.txt --type debate\n"
            "  python run_show.py prompts/podcast/soccer.txt --type podcast\n"
            "\n"
            "  # With research and overrides:\n"
            "  python run_show.py prompts/ubi_debate.txt --research --turns 50\n"
            "  python run_show.py prompts/tech_news.txt --segments 6 --tone urgent\n"
            "\n"
            "  # Control editor retry behavior:\n"
            "  python run_show.py prompts/ubi_debate.txt --max-retries 5\n"
            "\n"
            "  # Tag for batch management:\n"
            "  python run_show.py prompts/ubi_debate.txt --tag batch_01 --quiet\n"
            "\n"
            "  # Pipe directly into the renderer:\n"
            "  python run_show.py prompts/ubi_debate.txt --quiet | xargs python render_show.py\n"
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
        choices=["debate", "news", "podcast"],
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
    # These only matter when the show type is "debate". Ignored for news.
    debate_group = parser.add_argument_group("debate overrides")
    debate_group.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Override target debate turns (default: ~35).",
    )
    debate_group.add_argument(
        "--tone",
        type=str,
        default=None,
        help=(
            "Override tone (works for both debates and news). "
            "Examples: 'academic', 'cable news', 'conversational', 'urgent'."
        ),
    )
    debate_group.add_argument(
        "--format",
        type=str,
        default=None,
        dest="debate_format",
        help="Override debate format (e.g. 'oxford', 'crossfire', 'public_forum').",
    )

    # ── News-specific overrides ──────────────────────────────────────
    # These only matter when the show type is "news". Ignored for debates.
    news_group = parser.add_argument_group("news overrides")
    news_group.add_argument(
        "--segments",
        type=int,
        default=None,
        help="Override number of news segments (default: ~5).",
    )

    # ── Duration control ─────────────────────────────────────────────
    duration_group = parser.add_argument_group("duration control")
    duration_group.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Target duration in minutes.  Overrides any duration mentioned "
            "in the prompt file.  The conductor will aim within ±tolerance%%."
        ),
    )
    duration_group.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=(
            "Acceptable deviation from target duration, in %% (default: 10). "
            "E.g. --tolerance 5 means ±5%%."
        ),
    )
    # ── Checkpoint rendering (Phase 6) ────────────────────────────
    # When enabled, the conductor renders each segment's text through
    # TTS as it's generated, measures the actual audio duration, and
    # rebalances remaining-segment word targets in real time.
    duration_group.add_argument(
        "--checkpoint",
        action="store_true",
        default=None,
        dest="checkpoint",
        help=(
            "Force-enable checkpoint rendering (real-time TTS measurement "
            "for each segment). Auto-enabled for runs >= 20 min or "
            "audiobook content. Requires the configured TTS engine."
        ),
    )
    duration_group.add_argument(
        "--no-checkpoint",
        action="store_false",
        dest="checkpoint",
        help="Disable checkpoint rendering even when it would auto-enable.",
    )

    # ── Editing pipeline ─────────────────────────────────────────────
    # The editor reviews each generated chunk and can request retakes.
    # max_retries controls how persistent the editor is. Setting it to 0
    # effectively skips editor review (useful for quick drafts).
    editing_group = parser.add_argument_group("editing pipeline")
    editing_group.add_argument(
        "--max-retries",
        type=int,
        default=None,
        dest="max_retries",
        help=(
            "Max editor retry attempts per chunk (default: 3). "
            "The editor reviews each chunk for quality (speaker clarity, "
            "repetition, narrative coherence). Failed chunks are regenerated "
            "with detailed feedback. Set to 0 for no editor review."
        ),
    )

    return parser.parse_args(argv)


def log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"  {msg}", flush=True)


# ── Checkpoint plumbing (Phase 6) ─────────────────────────────────


def _should_use_checkpoint(args: argparse.Namespace, content_type: str,
                           target_minutes: float | None) -> bool:
    """Decide whether to enable checkpoint rendering for this run.

    Rules (in order):
      * --no-checkpoint always wins (user override OFF).
      * --checkpoint always wins (user override ON).
      * Audiobooks: ON by default.
      * target_duration_minutes >= 20: ON by default.
      * Otherwise: OFF.
    """
    if args.checkpoint is False:
        return False
    if args.checkpoint is True:
        return True
    if content_type == "audiobook":
        return True
    if target_minutes is not None and target_minutes >= 20.0:
        return True
    return False


def _build_checkpoint_kwargs(
    args: argparse.Namespace,
    content_type: str,
    *,
    target_minutes: float | None,
    num_segments: int,
    quiet: bool = False,
) -> dict:
    """Build kwargs to pass into a Conductor for checkpoint + tracker.

    Returns an empty dict when checkpointing is disabled. The conductor
    accepts ``checkpoint_renderer=`` and ``budget_tracker=`` keyword args.

    Imports the TTS engine lazily so users without TTS installed aren't
    forced to depend on it for non-checkpoint runs.
    """
    if not _should_use_checkpoint(args, content_type, target_minutes):
        return {}

    if num_segments <= 0 or target_minutes is None:
        log(
            "  Checkpoint requested but no target duration / segments — skipping.",
            quiet=quiet,
        )
        return {}

    try:
        from vox_romana.core.budget_tracker import BudgetTracker
        from vox_romana.tts.checkpoint import CheckpointRenderer
        from vox_romana.tts.engines.piper_engine import PiperEngine
    except ImportError as exc:
        log(f"  Checkpoint disabled (TTS unavailable: {exc!r})", quiet=quiet)
        return {}

    try:
        engine = PiperEngine()
    except Exception as exc:  # noqa: BLE001
        log(f"  Checkpoint disabled (engine init failed: {exc!r})", quiet=quiet)
        return {}

    renderer = CheckpointRenderer(engine)
    tracker = BudgetTracker(
        total_target_seconds=target_minutes * 60.0,
        num_segments=num_segments,
    )
    log(
        f"  Checkpoint ON — measuring {num_segments} segments against "
        f"{target_minutes:.1f} min budget.",
        quiet=quiet,
    )
    return {
        "checkpoint_renderer": renderer,
        "budget_tracker": tracker,
    }


def _resolve_duration(args: argparse.Namespace, brief: str) -> dict:
    """Return duration-related overrides.  Priority: CLI > brief > nothing."""
    overrides: dict = {}
    duration = args.duration
    if duration is None:
        duration = parse_duration_from_text(brief)
    if duration is not None:
        overrides["target_duration_minutes"] = duration
    if args.tolerance is not None:
        overrides["tolerance_percent"] = args.tolerance
    return overrides


def _run_debate(brief: str, client: DeepSeekClient, args: argparse.Namespace) -> Path:
    """Run the debate pipeline, return the output path.

    This is the debate studio floor. The brief has already been classified
    by the router — this function handles everything from pre-production
    (bootstrap) through recording (conductor) to final cut (serializer).

    Args:
        brief:  The full text of the user's prompt file.
        client: Shared DeepSeek API client.
        args:   Parsed CLI arguments (overrides, research, tag, etc.).

    Returns:
        Path to the saved JSON output file.
    """
    from vox_romana.shows.debates.bootstrap import DebateBootstrap
    from vox_romana.shows.debates.conductor import DebateConductor
    from vox_romana.shows.debates.serializer import DebateSerializer

    # ── Build overrides from CLI flags ───────────────────────────────
    # These stomp whatever the Architect decided. If the user said
    # --turns 50, that's what they get, regardless of what the AI
    # thought was appropriate.
    overrides = _resolve_duration(args, brief)
    if "target_duration_minutes" in overrides:
        overrides["target_words"] = estimate_total_words(
            overrides["target_duration_minutes"], "debate",
        )
    if args.turns is not None:
        overrides["target_turns"] = args.turns
    if args.tone is not None:
        overrides["tone"] = args.tone
    if args.debate_format is not None:
        overrides["debate_format"] = args.debate_format
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries

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

    checkpoint_kwargs = _build_checkpoint_kwargs(
        args, "debate",
        target_minutes=overrides.get("target_duration_minutes"),
        num_segments=config.target_turns or 0,
        quiet=args.quiet,
    )
    conductor = DebateConductor(config, client=client, **checkpoint_kwargs)
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
    """Run the news pipeline, return the output path.

    Same structure as _run_debate but dispatches to the news studio:
    bootstrap → conductor → serializer, with news-specific config.

    Args:
        brief:  The full text of the user's prompt file.
        client: Shared DeepSeek API client.
        args:   Parsed CLI arguments (overrides, research, tag, etc.).

    Returns:
        Path to the saved JSON output file.
    """
    from vox_romana.shows.news.bootstrap import NewsBootstrap
    from vox_romana.shows.news.conductor import NewsConductor
    from vox_romana.shows.news.serializer import NewsSerializer

    # ── Build overrides from CLI flags ───────────────────────────────
    overrides = _resolve_duration(args, brief)
    if "target_duration_minutes" in overrides:
        overrides["target_words"] = estimate_total_words(
            overrides["target_duration_minutes"], "news",
        )
    if args.segments is not None:
        overrides["target_segments"] = args.segments
    if args.tone is not None:
        overrides["tone"] = args.tone
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries

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

    checkpoint_kwargs = _build_checkpoint_kwargs(
        args, "news",
        target_minutes=overrides.get("target_duration_minutes"),
        num_segments=len(config.segments),
        quiet=args.quiet,
    )
    conductor = NewsConductor(config, client=client, **checkpoint_kwargs)
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


def _run_podcast(brief: str, client: DeepSeekClient, args: argparse.Namespace) -> Path:
    """Run the podcast pipeline, return the output path.

    Podcast studio floor. The brief has been classified as a podcast by
    the router — this function handles bootstrap → conductor → serializer.

    Podcast generation uses segment-level generation: one LLM call
    produces a full multi-speaker segment (3-15 utterances), unlike
    debates which generate one turn at a time.

    Args:
        brief:  The full text of the user's prompt file.
        client: Shared DeepSeek API client.
        args:   Parsed CLI arguments (overrides, research, tag, etc.).

    Returns:
        Path to the saved JSON output file.
    """
    from vox_romana.shows.podcast.bootstrap import PodcastBootstrap
    from vox_romana.shows.podcast.conductor import PodcastConductor
    from vox_romana.shows.podcast.serializer import PodcastSerializer

    # ── Build overrides from CLI flags ───────────────────────────────
    overrides = _resolve_duration(args, brief)
    target_words = None
    if "target_duration_minutes" in overrides:
        target_words = estimate_total_words(
            overrides["target_duration_minutes"], "podcast",
        )
        overrides["target_words"] = target_words
    if args.tone is not None:
        overrides["tone"] = args.tone
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries

    # ── Bootstrap ────────────────────────────────────────────────────
    log("Phase 2: Bootstrap (casting hosts, profiling, directing)...", quiet=args.quiet)
    t0 = time.time()

    bootstrap_kwargs = dict(
        overrides=overrides,
        auto_research=False,
    )
    if target_words is not None:
        bootstrap_kwargs["target_words"] = target_words

    if args.research:
        bootstrap_kwargs["auto_research"] = True
        try:
            from vox_romana.core.search import GoogleSearchProvider
            search_provider = GoogleSearchProvider()
            bootstrap_kwargs["search_provider"] = search_provider
            log("  Research mode ON (Google)", quiet=args.quiet)
        except (ValueError, ImportError):
            log("  Research mode ON (LLM knowledge — no Google keys)", quiet=args.quiet)

    bootstrap = PodcastBootstrap(client=client)
    config = bootstrap.from_brief(brief, **bootstrap_kwargs)

    t1 = time.time()
    log(f"  Done in {t1 - t0:.1f}s", quiet=args.quiet)
    log(f"  Title: {config.title}", quiet=args.quiet)
    log(f"  Show:  {config.show_name}", quiet=args.quiet)
    host_names = ", ".join(h.name for h in config.hosts)
    log(f"  Hosts: {host_names}", quiet=args.quiet)
    log(f"  Segments: {len(config.segments)} | Tone: {config.tone}", quiet=args.quiet)
    print()

    # ── Conductor ────────────────────────────────────────────────────
    log("Phase 3: Conductor (generating the episode)...", quiet=args.quiet)
    t2 = time.time()

    checkpoint_kwargs = _build_checkpoint_kwargs(
        args, "podcast",
        target_minutes=overrides.get("target_duration_minutes"),
        num_segments=len(config.segments),
        quiet=args.quiet,
    )
    conductor = PodcastConductor(config, client=client, **checkpoint_kwargs)
    result = conductor.run()

    t3 = time.time()
    log(f"  Done in {t3 - t2:.1f}s ({result.total_llm_calls} LLM calls)", quiet=args.quiet)
    log(f"  Generated {len(result.podcast_events)} events across {len(result.podcast_segments)} segments", quiet=args.quiet)
    print()

    # ── Serialize ────────────────────────────────────────────────────
    log("Phase 4: Saving output...", quiet=args.quiet)

    merged = result.to_merged_podcast()
    serializer = PodcastSerializer()

    stem = args.prompt.stem
    tag_part = f"_{args.tag}" if args.tag else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{stem}{tag_part}_{timestamp}.json"
    # Podcasts go to output/text/podcasts/ alongside debates/ and news/.
    podcast_dir = args.output / "text" / "podcasts"
    podcast_dir.mkdir(parents=True, exist_ok=True)
    output_path = podcast_dir / filename

    serializer.save(merged, output_path)
    return output_path


# ── Dispatch table ───────────────────────────────────────────────────────
_RUNNERS = {
    "debate": _run_debate,
    "news": _run_news,
    "podcast": _run_podcast,
}


def main(argv: list[str] | None = None) -> None:
    """Run the full show generation pipeline.

    This is the top-level entry point. It validates input, routes the
    brief to the right studio, and orchestrates the full pipeline.

    Args:
        argv: CLI arguments. Defaults to sys.argv[1:] if None.
              Exposed as a parameter so tests can call main() directly.
    """
    args = parse_args(argv)

    if not args.prompt.exists():
        print(f"Error: prompt file not found: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    brief = args.prompt.read_text().strip()
    if not brief:
        print(f"Error: prompt file is empty: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Create a shared client ─────────────────────────────────────
    # One client for the whole run so bootstrap + conductor + editor
    # all share the same connection pool.
    client = DeepSeekClient()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: ROUTE — classify the brief → pick the right studio
    # ══════════════════════════════════════════════════════════════════
    # The router makes a single LLM call to decide: debate or news?
    # If you passed --type, we skip this entirely.
    print(f"\n{'═' * 60}")
    print(f"  VOX ROMANA — Show Generator")
    print(f"{'═' * 60}")
    log(f"Prompt: {args.prompt}", quiet=args.quiet)
    if args.tag:
        log(f"Tag:    {args.tag}", quiet=args.quiet)
    log(f"Brief:  {brief[:80]}{'...' if len(brief) > 80 else ''}", quiet=args.quiet)

    # ── Resolve duration ──────────────────────────────────────────────
    duration_overrides = _resolve_duration(args, brief)
    if "target_duration_minutes" in duration_overrides:
        dur = duration_overrides["target_duration_minutes"]
        src = "CLI" if args.duration is not None else "brief"
        log(f"Duration: {dur:.1f} min (source: {src})", quiet=args.quiet)
    if "tolerance_percent" in duration_overrides:
        log(f"Tolerance: ±{duration_overrides['tolerance_percent']:.0f}%", quiet=args.quiet)

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

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2–4: RUN THE PIPELINE — bootstrap → conductor → serialize
    # ══════════════════════════════════════════════════════════════════
    # The dispatch table routes to the right studio floor based on the
    # classified show type. Each runner handles its own phases internally.
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

    return output_path


if __name__ == "__main__":
    main()

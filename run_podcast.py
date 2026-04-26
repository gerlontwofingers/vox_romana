#!/usr/bin/env python3
"""vox_romana CLI — generate a podcast episode from a prompt file.

This is the front door for podcast generation. You hand it a text file
containing a podcast brief, and it runs the full pipeline:

    prompt file → bootstrap (config) → conductor (podcast) → JSON output

THE RESTAURANT ANALOGY (podcast edition):

    1. YOUR PROMPT FILE is the customer's order. "Three women discuss
       soccer rivalries with chaotic energy." That's it. Vibes. Intent.

    2. THE BOOTSTRAP is the head chef. It reads your brief and spins up
       three specialist agents:

       - ARCHITECT:  Decides the show name, host cast, personality
                     traits, segment structure, tone. Extracts everything
                     you described — or invents smart defaults for what
                     you left out.

       - HOST PROFILER: Takes each host stub and builds a full
                        personality: voice fingerprint, behavioral
                        triggers, emotional arc, tangent topics, TTS
                        voice hints.

       - DIRECTOR:  Plans the episode arc: mood progression per segment,
                    pacing notes, closing strategy.

    3. THE CONDUCTOR is the recording studio in action. It generates
       each segment by asking the LLM to write dialogue for the full
       host cast at once (segment-level generation). After every few
       segments, it compresses the context and optionally checks in
       with the Director for pacing adjustments.

    4. THE JSON OUTPUT is the finished episode. Ready to feed to TTS
       for audio rendering.

ONE ARG TO RULE THEM ALL:

    The prompt file IS the entire specification. Write everything you
    want in plain English.

    Example prompt file:

        Three women discuss the greatest rivalries in women's soccer.
        The show is called "Trash Pandas." Tone: playful chaos with
        moments of genuine insight. One host is a stats nerd, one is
        a casual fan with pop culture tangents, and one is a former
        player who gets heated about certain teams. 8 segments
        including an intro and outro.

    From that paragraph, the system will:
    - Name the show "Trash Pandas"
    - Cast 3 women with distinct personalities
    - Plan 8 segments (intro, banter, deep dives, drama, outro)
    - Set the tone to "playful chaos with insight"
    - Build voice profiles for each host

    Shorter is fine too:

        Two guys argue about the best horror movies.

    The Architect will fill in everything else with smart defaults.

OVERRIDE FLAGS (safety valves, not required):

    --tone, --show-name let you override the Architect's decisions AFTER
    it reads your prompt. They're applied last and always win.

USAGE:
    # The common case — just point at a prompt file and go:
    python run_podcast.py prompts/podcast/soccer_rivalries.txt

    # Custom output directory:
    python run_podcast.py prompts/podcast/soccer_rivalries.txt --output results/

    # With auto-research (grounds the podcast in real web data):
    python run_podcast.py prompts/podcast/soccer_rivalries.txt --research

    # Override something the Architect chose:
    python run_podcast.py prompts/podcast/soccer_rivalries.txt --tone "unhinged"

    # Tag the output for easy filtering later:
    python run_podcast.py prompts/podcast/soccer_rivalries.txt --tag "batch_01"

    # Quiet mode — just the output path, for scripting:
    python run_podcast.py prompts/podcast/soccer_rivalries.txt -q

    # Full combo — research, override tone, tag, custom output:
    python run_podcast.py prompts/podcast/soccer_rivalries.txt \\
        --research --tone "chaotic energy" --tag "v2" --output results/

PROMPT FILE FORMAT:
    Plain text. No special syntax. Just describe the podcast you want,
    like you're pitching it to a friend. Put the file anywhere — the
    path is the only required CLI arg.

ENVIRONMENT:
    Requires DEEPSEEK_API_KEY in .env (or exported in shell).
    Optional: GOOGLE_API_KEY + GOOGLE_CSE_ID for --research mode.

OUTPUT:
    Saves a JSON file to output/text/podcasts/ with the full episode:
    events, segment structure, host metadata, voice casting info.
    Filename: {prompt_stem}_{tag}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Load .env BEFORE any vox_romana imports ──
from dotenv import load_dotenv
load_dotenv()

from vox_romana import (
    PodcastBootstrap,
    PodcastConductor,
    DeepSeekClient,
)
from vox_romana.core.paths import OUTPUT_DIR as _OUTPUT_DIR
from vox_romana.shows.podcast.serializer import PodcastSerializer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    The only REQUIRED argument is the path to a prompt file. That file
    contains everything — cast, show name, tone, segment count — in
    plain English.

    Optional flags override the Architect's decisions AFTER it reads
    the prompt. Think of them as last-minute post-it notes.
    """
    parser = argparse.ArgumentParser(
        description="Generate a podcast episode from a prompt file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_podcast.py prompts/podcast/soccer.txt\n"
            "  python run_podcast.py prompts/podcast/soccer.txt --research\n"
            "  python run_podcast.py prompts/podcast/soccer.txt --tone 'unhinged'\n"
            "  python run_podcast.py prompts/podcast/soccer.txt --tag 'v2' -q\n"
        ),
    )

    # ── Required ─────────────────────────────────────────────────────
    parser.add_argument(
        "prompt",
        type=Path,
        help="Path to a plain-text prompt file describing the podcast episode.",
    )

    # ── Output ───────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=_OUTPUT_DIR,
        help=f"Base output directory (default: {_OUTPUT_DIR}). Files go to text/podcasts/ under this.",
    )

    # ── Research ─────────────────────────────────────────────────────
    parser.add_argument(
        "--research",
        action="store_true",
        help="Auto-research the topic before generating. Uses Google if API "
        "keys are set (GOOGLE_API_KEY + GOOGLE_CSE_ID), falls back to LLM "
        "knowledge otherwise.",
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
        "--university-session",
        type=str,
        default=None,
        dest="university_session",
        help=(
            "Session ID of a completed University deep-research run. "
            "When set, Scholar notes are searched first (highest-priority tier). "
            "E.g. --university-session a1b2c3d4e5f6"
        ),
    )
    # ── Overrides ────────────────────────────────────────────────────
    # Post-it notes on the Architect's plan. Applied last, always win.
    parser.add_argument(
        "--tone",
        type=str,
        default=None,
        help="Override the podcast tone (e.g. 'playful', 'chaotic energy', 'serious')",
    )
    parser.add_argument(
        "--show-name",
        type=str,
        default=None,
        help="Override the show name (e.g. 'Trash Pandas')",
    )

    # ── Fact-checking ─────────────────────────────────────────────────
    parser.add_argument(
        "--fact-check",
        choices=["off", "quick", "thorough"],
        default="off",
        dest="fact_check",
        help=(
            "Fact-check mode for generated content (default: off). "
            "'quick': verify claims against --reference-material only. "
            "'thorough': also web-searches unverifiable claims. "
            "Segments with fabricated facts are regenerated up to --max-retries times."
        ),
    )
    parser.add_argument(
        "--reference-material",
        type=Path,
        default=None,
        dest="reference_material",
        help="Path to a plain-text file with factual reference material for fact-checking.",
    )

    # ── Tagging ──────────────────────────────────────────────────────
    parser.add_argument(
        "--tag", "-t",
        type=str,
        default=None,
        help="A label baked into the output filename for easy filtering (e.g. --tag 'batch_01')",
    )

    # ── Verbosity ────────────────────────────────────────────────────
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output — just print the final output path",
    )

    return parser.parse_args()


def log(msg: str, *, quiet: bool = False) -> None:
    """Print a progress message (unless in quiet mode)."""
    if not quiet:
        print(f"  {msg}", flush=True)


def main() -> None:
    """Run the full podcast generation pipeline.

    Three phases, same as the debate pipeline:
      1. Bootstrap: brief → PodcastConfig (Architect + Host Profiler + Director)
      2. Conductor: config → PodcastConductorResult (segment-level generation)
      3. Serialize: result → JSON file

    .. deprecated::
        Use ``run_show.py`` instead. ``run_show.py`` auto-routes to the
        podcast pipeline when the brief is podcast-shaped. ``run_podcast.py``
        will be removed in a future release.
    """
    import warnings
    warnings.warn(
        "run_podcast.py is deprecated; use 'python run_show.py <prompt>' "
        "which auto-routes to the podcast pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    args = parse_args()

    # ── Validate the prompt file ─────────────────────────────────────
    if not args.prompt.exists():
        print(f"Error: prompt file not found: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    brief = args.prompt.read_text().strip()
    if not brief:
        print(f"Error: prompt file is empty: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    # ── Create output directory ──────────────────────────────────────
    # Podcasts go under output/text/podcasts/ to stay organized alongside
    # debates/ and news/ in the existing output structure.
    output_dir = args.output / "text" / "podcasts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build overrides dict from CLI flags ──────────────────────────
    overrides = {}
    if args.tone is not None:
        overrides["tone"] = args.tone
    if args.show_name is not None:
        overrides["show_name"] = args.show_name

    # ══════════════════════════════════════════════════════════════════
    # BOOTSTRAP — brief → fully populated PodcastConfig
    # ══════════════════════════════════════════════════════════════════
    # Pre-production. Three AI agents read your brief:
    #   ARCHITECT:      Decides show name, host cast, segment structure.
    #   HOST PROFILER:  Builds full personality for each host (voice,
    #                   triggers, arc, tangent topics).
    #   DIRECTOR:       Plans the episode arc and pacing notes.
    print(f"\n{'═' * 60}")
    print(f"  VOX ROMANA — Podcast Generator")
    print(f"{'═' * 60}")
    log(f"Prompt: {args.prompt}", quiet=args.quiet)
    if args.tag:
        log(f"Tag:    {args.tag}", quiet=args.quiet)
    log(f"Brief:  {brief[:80]}{'...' if len(brief) > 80 else ''}", quiet=args.quiet)
    print()

    # Shared client for bootstrap + conductor.
    client = DeepSeekClient()

    log("Bootstrap (casting hosts, profiling, directing)...", quiet=args.quiet)
    t0 = time.time()

    bootstrap_kwargs = dict(
        overrides=overrides,
        auto_research=False,
    )

    # ── Fact-check configuration ─────────────────────────────────────
    if args.fact_check != "off":
        reference_text: str | None = None
        if args.reference_material is not None:
            if not args.reference_material.exists():
                print(f"Error: reference material file not found: {args.reference_material}", file=sys.stderr)
                sys.exit(1)
            reference_text = args.reference_material.read_text().strip()
        bootstrap_kwargs["fact_check"] = args.fact_check
        bootstrap_kwargs["reference_material"] = reference_text
        log(f"  Fact-check: {args.fact_check}", quiet=args.quiet)
    else:
        bootstrap_kwargs["fact_check"] = "off"

    # ── Optional: auto-research ──────────────────────────────────────
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
            log("  Research mode ON (Google)", quiet=args.quiet)
        except (ValueError, ImportError):
            log("  Research mode ON (LLM knowledge — no Google keys)", quiet=args.quiet)

        # ── Tier 0: Prepend University session (highest priority) ──
        university_session_id = getattr(args, "university_session", None)
        if university_session_id:
            try:
                from vox_romana.core.search import (
                    ChainSearchProvider,
                    build_research_provider,
                )
                from vox_romana.university.search_provider import UniversitySearchProvider
                univ = UniversitySearchProvider(university_session_id)
                base = build_research_provider(
                    bootstrap_kwargs.pop("search_provider", None),
                    bootstrap_kwargs.pop("memory", None),
                )
                providers = [univ] + ([base] if base is not None else [])
                bootstrap_kwargs["search_provider"] = (
                    ChainSearchProvider(providers) if len(providers) > 1 else providers[0]
                )
                log(f"  University session: {university_session_id}", quiet=args.quiet)
            except Exception as e:
                log(f"  University session: OFF (load failed: {e})", quiet=args.quiet)

    bootstrap = PodcastBootstrap(client=client)
    config = bootstrap.from_brief(brief, **bootstrap_kwargs)

    t1 = time.time()
    log(f"  Done in {t1 - t0:.1f}s", quiet=args.quiet)
    log(f"  Title: {config.title}", quiet=args.quiet)
    log(f"  Show:  {config.show_name}", quiet=args.quiet)
    log(f"  Hosts: {', '.join(h.name for h in config.hosts)}", quiet=args.quiet)
    log(f"  Segments: {len(config.segments)} | Tone: {config.tone}", quiet=args.quiet)
    print()

    # ══════════════════════════════════════════════════════════════════
    # CONDUCTOR — config → live podcast episode
    # ══════════════════════════════════════════════════════════════════
    # Recording studio is live. The Conductor generates each segment
    # by asking the LLM to write dialogue for the full host cast at
    # once (segment-level generation, not turn-by-turn like debates).
    #
    # After every few segments, it compresses the context window and
    # optionally checks in with the Director for pacing adjustments.
    log("Conductor (generating the episode)...", quiet=args.quiet)
    t2 = time.time()

    conductor = PodcastConductor(config, client=client)
    result = conductor.run()

    t3 = time.time()
    log(f"  Done in {t3 - t2:.1f}s ({result.total_llm_calls} LLM calls)", quiet=args.quiet)
    log(f"  Generated {len(result.podcast_events)} events across {len(result.podcast_segments)} segments", quiet=args.quiet)
    print()

    # ══════════════════════════════════════════════════════════════════
    # SERIALIZE — save to JSON
    # ══════════════════════════════════════════════════════════════════
    # Convert the in-memory podcast to a JSON file. This file contains
    # every line of dialogue, segment structure, host metadata, voice
    # casting info — everything TTS needs to render audio.
    log("Saving output...", quiet=args.quiet)

    merged = result.to_merged_podcast()
    serializer = PodcastSerializer()

    # Build filename: {classification}_{prompt_stem}_{tag}_{timestamp}.json
    stem = args.prompt.stem
    tag_part = f"_{args.tag}" if args.tag else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"podcast_{stem}{tag_part}_{timestamp}.json"
    output_path = output_dir / filename

    serializer.save(merged, output_path)

    print()
    print(f"{'═' * 60}")
    log(f"Output saved to: {output_path}", quiet=args.quiet)
    log(f"Total time: {t3 - t0:.1f}s", quiet=args.quiet)
    print(f"{'═' * 60}")
    print()

    # In quiet mode, just print the path (useful for scripting).
    if args.quiet:
        print(output_path)


if __name__ == "__main__":
    main()

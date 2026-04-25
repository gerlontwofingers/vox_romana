#!/usr/bin/env python3
"""vox_romana CLI — generate a debate from a prompt file.

This is the front door. You hand it a text file containing a debate brief,
and it runs the full pipeline:

    prompt file → bootstrap (config) → conductor (debate) → JSON output

THE RESTAURANT ANALOGY:

    Think of vox_romana like a high-end restaurant:

    1. YOUR PROMPT FILE is the customer's order. You don't say "I want
       200g of protein with 45ml of sauce." You say "I want the steak,
       medium-rare, with the peppercorn sauce, and make it a big portion."
       Natural language. Vibes. Intent.

    2. THE BOOTSTRAP is the head chef reading your order. The chef decides:
       how many cooks to assign (participants), which station runs lead
       (moderator), the plating style (tone/format), and how many courses
       (turns). You didn't specify any of that — the chef inferred it from
       your order. That's the Architect, Profile Builder, and Director
       doing their thing.

    3. THE CONDUCTOR is the kitchen in action. Each cook works their
       station, the expeditor (moderator) keeps things moving, and the
       head chef (director) checks in periodically to make sure the
       meal is on track.

    4. THE JSON OUTPUT is the plated dish. Ready to serve (to TTS).

ONE ARG TO RULE THEM ALL:

    The prompt file IS the entire specification. Write everything you want
    in plain English — the number of participants, the moderator style,
    the tone, the length, the format — and the Architect (an LLM agent)
    will extract it all and build the config.

    Example prompt file:

        Four economists debate UBI in an Oxford-style format.
        Two pro, two con. Aggressive moderator who interrupts.
        50 turns. Cable news energy but with real data.

    From that single paragraph, the system will:
    - Cast 4 economists with distinct personalities and positions
    - Assign an aggressive moderator character
    - Set the format to Oxford-style
    - Target 50 turns
    - Set the tone to "cable news with substance"

    You don't NEED to specify all of that. A brief like "Two people argue
    about pineapple on pizza" works fine — the Architect will fill in
    sensible defaults for everything you left out (2 participants, a
    neutral moderator, ~35 turns, balanced tone).

    The more detail you put in the prompt, the more control you have.
    The less detail, the more creative freedom the AI gets. Both are valid.

OVERRIDE FLAGS (safety valves, not required):

    The --turns, --tone, and --format flags exist as escape hatches.
    They override whatever the Architect decided, AFTER it reads the prompt.
    Think of them as telling the waiter "actually, make that well-done"
    after the chef already started cooking. They're applied last and
    they always win.

    These are for when the Architect misinterprets your prompt or when
    you want to experiment without editing the prompt file. 99% of the
    time, the prompt alone is enough.

USAGE:
    # The common case — just point at a prompt file and go:
    python run_debate.py prompts/ubi_debate.txt

    # Custom output directory:
    python run_debate.py prompts/ubi_debate.txt --output results/

    # With auto-research (grounds the debate in real web data):
    python run_debate.py prompts/ubi_debate.txt --research

    # Override something the Architect chose (rare):
    python run_debate.py prompts/ubi_debate.txt --turns 50 --tone "unhinged"

    # Quiet mode — just the output path, for scripting:
    python run_debate.py prompts/ubi_debate.txt --quiet

PROMPT FILE FORMAT:
    Plain text. No special syntax. No YAML, no JSON, no tags.
    Just describe the debate you want, like you're telling a friend.
    Put the file anywhere under prompts/ — the path is the only CLI arg.

    See prompts/ubi_debate.txt for a working example.

ENVIRONMENT:
    Requires DEEPSEEK_API_KEY in .env (or exported in shell).
    Optional: GOOGLE_API_KEY + GOOGLE_CSE_ID for --research mode.

OUTPUT:
    Saves a JSON file to the output directory with the full debate:
    events, metadata, participant profiles, and dramatic arc.
    Filename: {prompt_stem}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Load .env BEFORE any vox_romana imports that might need the key ──
from dotenv import load_dotenv
load_dotenv()

from vox_romana import (
    DebateBootstrap,
    DebateConductor,
    DeepSeekClient,
)
from vox_romana.core.paths import OUTPUT_DIR as _OUTPUT_DIR
from vox_romana.shows.debates.serializer import DebateSerializer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    The only REQUIRED argument is the path to a prompt file. That file
    contains everything — cast, format, tone, turn count — in plain English.

    The optional flags (--turns, --tone, --format) are safety valves.
    They override the Architect's decisions AFTER it reads your prompt.
    Think of them as post-it notes stuck on the chef's finished plan:
    "Actually, make it 50 turns" or "Change the tone to unhinged."

    Most of the time, you won't need them. The prompt is king.
    """
    parser = argparse.ArgumentParser(
        description="Generate a debate from a prompt file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_debate.py prompts/ubi_debate.txt\n"
            "  python run_debate.py prompts/ubi_debate.txt --research\n"
            "  python run_debate.py prompts/ubi_debate.txt --turns 50 --tone 'cable news'\n"
        ),
    )

    # ── Required ─────────────────────────────────────────────────────
    # This is it. The one ring. The prompt file path.
    # Everything about the debate lives in this file as plain English.
    parser.add_argument(
        "prompt",
        type=Path,
        help="Path to a plain-text prompt file (e.g. prompts/ubi_debate.txt). "
             "This is the ONLY required arg — the prompt describes everything.",
    )

    # ── Output ───────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=_OUTPUT_DIR,
        help=f"Directory to save the output JSON (default: {_OUTPUT_DIR})",
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
    # ── Overrides ────────────────────────────────────────────────────
    # These are the "actually, change this one thing" flags.
    # They're applied AFTER the Architect reads your prompt and makes
    # its decisions. The override always wins over the Architect.
    #
    # You almost never need these. If your prompt says "50 turns, Oxford
    # format, cable news energy," the Architect will honor that. These
    # flags are for when the Architect misreads your intent, or when
    # you want to quickly experiment without editing the prompt file.
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Override the target number of debate turns (default: ~35)",
    )
    parser.add_argument(
        "--tone",
        type=str,
        default=None,
        help="Override the debate tone (e.g. 'academic', 'cable news', 'unhinged')",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        dest="debate_format",
        help="Override the debate format (e.g. 'oxford', 'public_forum', 'crossfire')",
    )

    # ── Tagging ───────────────────────────────────────────────────────
    # A free-text label that gets baked into the output filename and
    # stored in the JSON metadata. Use it however you want:
    #   - batch IDs:   --tag "batch_042"
    #   - experiments: --tag "high_temp_test"
    #   - topics:      --tag "ubi_v2"
    #   - anything:    --tag "for_sarah"
    #
    # Later, when you have a payload/ directory full of JSON files,
    # this tag makes grep/sort/filter trivial.
    # ── Fact-checking ─────────────────────────────────────────────────
    parser.add_argument(
        "--fact-check",
        choices=["off", "quick", "thorough"],
        default="off",
        dest="fact_check",
        help=(
            "Fact-check mode (default: off). "
            "'quick': verify against --reference-material. "
            "'thorough': also web-searches unverifiable claims. "
            "Fabricated facts trigger regeneration up to max_retries."
        ),
    )
    parser.add_argument(
        "--reference-material",
        type=Path,
        default=None,
        dest="reference_material",
        help="Path to a plain-text file with factual reference material for fact-checking.",
    )

    # ── Tagging ─────────────────────────────────────────────────
    parser.add_argument(
        "--tag", "-t",
        type=str,
        default=None,
        help="A label baked into the output filename for easy filtering later (e.g. --tag 'batch_01')",
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
    """Run the full debate generation pipeline."""
    args = parse_args()

    # ── Validate the prompt file exists ──────────────────────────────
    if not args.prompt.exists():
        print(f"Error: prompt file not found: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    brief = args.prompt.read_text().strip()
    if not brief:
        print(f"Error: prompt file is empty: {args.prompt}", file=sys.stderr)
        sys.exit(1)

    # ── Create output directory ──────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)

    # ── Build overrides dict from CLI flags ──────────────────────────
    # These are the post-it notes from the CLI flags. If the user passed
    # --turns 50, that goes into overrides and will stomp whatever the
    # Architect chose. If no flags were passed, this dict stays empty
    # and the Architect's decisions stand unchallenged.
    overrides = {}
    if args.turns is not None:
        overrides["target_turns"] = args.turns
    if args.tone is not None:
        overrides["tone"] = args.tone
    if args.debate_format is not None:
        overrides["debate_format"] = args.debate_format

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: BOOTSTRAP — brief → fully populated config
    # ══════════════════════════════════════════════════════════════════
    # This is pre-production. The brief (your entire prompt file) gets
    # read by a chain of three AI agents:
    #
    #   ARCHITECT:       Reads the brief, decides cast size, format, tone,
    #                    turn count, moderator style. Extracts everything
    #                    you described — or fills in smart defaults for
    #                    anything you left out.
    #
    #   PROFILE BUILDER: Takes each participant stub from the Architect
    #                    and builds a full personality: voice, triggers,
    #                    emotional arc, key positions.
    #
    #   DIRECTOR:        Reads the full cast and plans the dramatic arc:
    #                    when tensions should rise, when to pivot, how
    #                    to close.
    #
    # The result is a DebateConfig — a complete production plan that the
    # Conductor can execute without any more human input.
    print(f"\n{'═' * 60}")
    print(f"  VOX ROMANA — Debate Generator")
    print(f"{'═' * 60}")
    log(f"Prompt: {args.prompt}", quiet=args.quiet)
    if args.tag:
        log(f"Tag:    {args.tag}", quiet=args.quiet)
    log(f"Brief:  {brief[:80]}{'...' if len(brief) > 80 else ''}", quiet=args.quiet)
    print()

    # Create a shared client so bootstrap + conductor reuse the same connection.
    client = DeepSeekClient()

    log("Phase 1: Bootstrap (casting, profiling, directing)...", quiet=args.quiet)
    t0 = time.time()

    bootstrap_kwargs = dict(
        reference_material=None,
        fact_check="off",
        overrides=overrides,
        auto_research=False,
    )

    # ── Fact-check configuration ───────────────────────────────────────
    if args.fact_check != "off":
        reference_text: str | None = None
        if args.reference_material is not None:
            if not args.reference_material.exists():
                print(f"Error: reference material not found: {args.reference_material}", file=sys.stderr)
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
        # Memory-first: consult the local knowledge base before any paid
        # search. Free, sub-millisecond, and domain-tuned. The bootstrap
        # automatically wraps memory + web search into a chain.
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

    bootstrap = DebateBootstrap(client=client)
    config = bootstrap.from_brief(brief, **bootstrap_kwargs)

    t1 = time.time()
    log(f"  Done in {t1 - t0:.1f}s", quiet=args.quiet)
    log(f"  Title: {config.title}", quiet=args.quiet)
    log(f"  Participants: {', '.join(p.name for p in config.participants)}", quiet=args.quiet)
    log(f"  Format: {config.debate_format} | Tone: {config.tone} | Turns: {config.target_turns}", quiet=args.quiet)
    print()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: CONDUCTOR — config → live debate
    # ══════════════════════════════════════════════════════════════════
    # Now we're rolling cameras. The Conductor takes the config from
    # Phase 1 and runs the actual debate: picking who speaks next,
    # generating their dialogue, having the moderator interject,
    # checking in with the Director for pacing adjustments.
    #
    # Each turn is one LLM call. A 35-turn debate = ~40 LLM calls
    # (turns + moderator interjections + director check-ins).
    # Takes roughly 30-90 seconds depending on the model.
    log("Phase 2: Conductor (running the debate)...", quiet=args.quiet)
    t2 = time.time()

    conductor = DebateConductor(config, client=client)
    result = conductor.run()

    t3 = time.time()
    log(f"  Done in {t3 - t2:.1f}s ({result.total_llm_calls} LLM calls)", quiet=args.quiet)
    log(f"  Generated {len(result.debate_events)} events", quiet=args.quiet)
    print()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: SERIALIZE — save to JSON
    # ══════════════════════════════════════════════════════════════════
    # The debate is done. Now we plate the dish — convert the in-memory
    # debate events into a JSON file. This file contains everything:
    # every line of dialogue, speaker metadata, stage directions,
    # emotional cues, and the full config that produced it.
    #
    # This JSON is the input for the next stage (TTS), or you can just
    # read it to see what the AI came up with.
    log("Phase 3: Saving output...", quiet=args.quiet)

    merged = result.to_merged_debate()
    serializer = DebateSerializer()

    # Build a filename from the prompt file name + tag + timestamp.
    # Examples:
    #   debate_ubi_debate_20260405_143022.json           (no tag)
    #   debate_ubi_debate_batch_01_20260405_143022.json  (with --tag batch_01)
    stem = args.prompt.stem
    tag_part = f"_{args.tag}" if args.tag else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"debate_{stem}{tag_part}_{timestamp}.json"
    output_path = args.output / filename

    serializer.save_merged(merged, output_path)

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

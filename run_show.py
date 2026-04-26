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
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from vox_romana.core.paths import OUTPUT_DIR as _OUTPUT_DIR


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
        choices=["debate", "news", "podcast", "audiobook"],
        dest="show_type",
        help="Force a show type (skip the router).",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=_OUTPUT_DIR,
        help=f"Directory to save the output JSON (default: {_OUTPUT_DIR}).",
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
        "--no-memory",
        action="store_false",
        dest="memory",
        default=True,
        help="Disable memory-first lookup during research. By default, the "
        "local knowledge base (Chroma + FTS5) is consulted before any web "
        "search; use this flag to force web-only research.",
    )

    parser.add_argument(
        "--university-session",
        type=str,
        default=None,
        dest="university_session",
        help=(
            "Session ID of a completed University deep-research run. When set, "
            "the Scholar's notes are searched first (before memory or Google), "
            "grounding the show in your prior deep-research. "
            "E.g. --university-session a1b2c3d4e5f6"
        ),
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
    # ── Checkpoint rendering ────────────────────────────
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
    # Optional engine hint for word-budget calibration. When the run.py
    # full pipeline forwards this, ``estimate_total_words`` uses a per-
    # engine WPM (Piper runs faster than Kokoro) so target_words better
    # matches the actual delivery speed.
    duration_group.add_argument(
        "--engine",
        type=str,
        default=None,
        choices=["kokoro", "piper", "both"],
        help=(
            "TTS engine hint for duration math (forwarded by run.py). "
            "Adjusts word-budget targets to the engine's typical pace."
        ),
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
    editing_group.add_argument(
        "--fact-check",
        choices=["off", "quick", "thorough"],
        default="thorough",
        dest="fact_check",
        help=(
            "Fact-check mode (default: thorough). "
            "'off': no checking. "
            "'quick': verify against --reference-material. "
            "'thorough': also web-searches unverifiable claims. "
            "Fabricated facts and cross-segment contradictions trigger "
            "regeneration up to max_retries."
        ),
    )
    editing_group.add_argument(
        "--reference-material",
        type=Path,
        default=None,
        dest="reference_material",
        help="Path to a plain-text file with factual reference material for fact-checking.",
    )
    editing_group.add_argument(
        "--content-mode",
        choices=["auto", "factual", "fictional-universe", "pure-fiction"],
        default="auto",
        dest="content_mode",
        help=(
            "Content truth-mode (default: auto — detected from the brief). "
            "'factual': real-world claims, fabrications block publication. "
            "'fictional-universe': established canon (Star Wars, Tolkien, …), "
            "violations are WARNINGS not blockers. "
            "'pure-fiction': original fiction, fact-checking is skipped entirely."
        ),
    )

    editing_group.add_argument(
        "--allow-bootstrap-fabrications",
        action="store_true",
        default=False,
        dest="allow_bootstrap_fabrications",
        help=(
            "Skip the bootstrap preflight abort when fabricated facts are detected "
            "in the generated outline/rundown. Prints a warning and continues. "
            "Not recommended — use only when you have verified the content manually."
        ),
    )

    # ── Series grouping ──────────────────────────────────────────────
    parser.add_argument(
        "--series-id",
        type=str,
        default=None,
        dest="series_id",
        help=(
            "Series identifier for multi-episode shows. When set, voice "
            "assignments for recurring characters are locked in the registry "
            "so the same character always gets the same voice across episodes. "
            "E.g. --series-id my-science-podcast"
        ),
    )

    return parser.parse_args(argv)


def log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"  {msg}", flush=True)


def main(argv: list[str] | None = None) -> Path:
    """Generate a show from a prompt file.

    Thin CLI shim — parses args and delegates to
    ``vox_romana.cli.runners.run_generate``.  All runner
    implementations live in that module.

    Args:
        argv: CLI arguments. Defaults to sys.argv[1:] if None.
              Exposed as a parameter so tests can call main() directly.

    Returns:
        Path to the generated JSON file.
    """
    args = parse_args(argv)
    from vox_romana.cli.runners import run_generate
    return run_generate(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""run.py — master CLI for the vox_romana production pipeline.

One command to generate text, render audio, or do both.  This replaces
the old two-step workflow of ``run_show.py`` → ``render_show.py`` with
a single entry point that handles the full pipeline end-to-end.

    prompt file → classify → bootstrap → conductor → editor → JSON → TTS → audio

THE PRODUCTION STUDIO ANALOGY:

    Think of run.py as the executive producer. You hand them a brief and
    say "make me a show." They walk it through every department:

    1. THE WRITERS' ROOM (run_show.py) generates the script from your
       prompt — casting, profiling, directing, conducting, editing.

    2. THE SOUND BOOTH (render_show.py) records the script into audio
       using the TTS engine of your choice (Kokoro or Piper).

    3. THE DATA SCIENTIST (WPM logging) records how long the audio
       actually came out, so future duration estimates get better.

    run.py coordinates all three.  You can also ask it to do just one
    step (--mode generate or --mode render) if you want manual control.

MODES:

    full (default):
        Generate the show script AND render it to audio in one shot.
        This is the batteries-included experience. After rendering,
        actual audio duration is logged to the catalog's wpm_stats
        table for the SpeechRateAdvisor feedback loop.

        python run.py prompts/ubi_debate.txt
        python run.py prompts/tech_news.txt --engine kokoro --format mp3

    generate:
        Text generation only. Produces the show JSON but no audio.
        Equivalent to calling run_show.py directly.

        python run.py prompts/ubi_debate.txt --mode generate
        python run.py prompts/trash_pandas_01.txt --mode generate --tag v2

    render:
        Audio rendering only from an existing show JSON. Requires
        --script-input pointing to a previously generated JSON file.
        Equivalent to calling render_show.py directly.

        python run.py --mode render --script-input output/text/debates/ubi.json
        python run.py --mode render --script-input output/text/news/tech.json \\
            --engine piper --audio-output my_show.mp3

USAGE:

    # === FULL MODE (generate + render) ===============================

    # Simplest: auto-detect show type, default engine (kokoro), mp3:
    python run.py prompts/ubi_debate.txt

    # Force show type, custom engine, wav format:
    python run.py prompts/ubi_debate.txt --type debate --engine piper -f wav

    # With research and duration target:
    python run.py prompts/tech_news.txt --research --duration 15 --tolerance 5

    # Override turns, tag the run, custom audio output:
    python run.py prompts/ubi_debate.txt --turns 50 --tag v3 \\
        --audio-output output/audio/debates/ubi_v3.mp3

    # Quiet mode (just prints the audio path at the end):
    python run.py prompts/ubi_debate.txt --quiet

    # Disable WPM logging (benchmarks, experiments):
    python run.py prompts/ubi_debate.txt --no-log-audio-duration

    # Manual voice assignment for the render step:
    python run.py prompts/ubi_debate.txt \\
        --voice-map '{"Dr. Chen": "af_heart", "Dr. Vance": "am_adam"}'

    # === GENERATE MODE (text only) ===================================

    # Just produce the JSON script:
    python run.py prompts/ubi_debate.txt --mode generate

    # With overrides:
    python run.py prompts/tech_news.txt --mode generate \\
        --segments 8 --tone urgent --tag breaking

    # === RENDER MODE (audio only) ====================================

    # Render an existing JSON to audio:
    python run.py --mode render --script-input output/text/debates/ubi.json

    # Custom engine and output:
    python run.py --mode render --script-input output/text/news/tech.json \\
        --engine piper --audio-output my_news.wav --format wav

    # Disable WPM logging for this render:
    python run.py --mode render --script-input output/show.json \\
        --no-log-audio-duration

    # === PODCAST EXAMPLES ============================================

    python run.py prompts/podcast/soccer.txt --type podcast --duration 30
    python run.py prompts/podcast/soccer.txt --mode generate --type podcast
    python run.py --mode render --script-input output/text/podcasts/soccer.json

ENVIRONMENT:
    DEEPSEEK_API_KEY in .env (required for generate/full modes).
    No API keys needed for render mode — all TTS is local.
    Optional: GOOGLE_API_KEY + GOOGLE_CSE_ID for --research mode.

OUTPUT:
    generate mode → JSON file in output/text/{type}/
    render mode   → Audio file next to the JSON (or --audio-output)
    full mode     → Both JSON and audio files
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the master pipeline.

    Merges flags from both run_show.py (generation) and render_show.py
    (rendering) into a single namespace, plus adds master-level flags
    like --mode, --script-input, --script-output, --audio-output, and
    --no-log-audio-duration.

    Args:
        argv: Arguments to parse. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed namespace with all options.
    """
    parser = argparse.ArgumentParser(
        prog="run",
        description=(
            "Master CLI for vox_romana. Generate a show, render audio, "
            "or do both in one shot."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full pipeline (generate + render):\n"
            "  python run.py prompts/ubi_debate.txt\n"
            "  python run.py prompts/tech_news.txt --engine kokoro\n"
            "\n"
            "  # Generate only (JSON, no audio):\n"
            "  python run.py prompts/ubi_debate.txt --mode generate\n"
            "\n"
            "  # Render only (audio from existing JSON):\n"
            "  python run.py --mode render --script-input output/text/debates/ubi.json\n"
            "\n"
            "  # Full with overrides:\n"
            "  python run.py prompts/ubi_debate.txt --turns 50 --duration 20 \\\n"
            "      --tag v3 --engine kokoro --format wav\n"
            "\n"
            "  # Quiet mode:\n"
            "  python run.py prompts/ubi_debate.txt --quiet\n"
            "\n"
            "  # Disable WPM logging:\n"
            "  python run.py prompts/ubi_debate.txt --no-log-audio-duration\n"
        ),
    )

    # ── Master-level flags ───────────────────────────────────────────
    parser.add_argument(
        "prompt",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Path to a plain-text prompt file. Required for generate "
            "and full modes. Not needed for render mode."
        ),
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="full",
        choices=["generate", "render", "full"],
        help=(
            "Pipeline mode. "
            "'generate' = text only (JSON). "
            "'render' = audio only (from existing JSON). "
            "'full' = generate + render (default)."
        ),
    )

    parser.add_argument(
        "--script-input",
        type=Path,
        default=None,
        help=(
            "Path to an existing show JSON file. Required for render "
            "mode. Ignored in generate and full modes."
        ),
    )

    parser.add_argument(
        "--script-output",
        type=Path,
        default=None,
        help=(
            "Custom output path for the generated JSON. If not set, "
            "run_show.py picks the default (output/text/{type}/)."
        ),
    )

    parser.add_argument(
        "--audio-output",
        type=Path,
        default=None,
        help=(
            "Custom output path for the rendered audio file. If not set, "
            "the audio goes next to the JSON with the right extension."
        ),
    )

    parser.add_argument(
        "--no-log-audio-duration",
        action="store_true",
        default=False,
        help=(
            "Disable logging actual audio duration to the catalog. "
            "By default, after rendering, the actual WPM is recorded "
            "so future duration estimates improve over time. Use this "
            "for benchmarks or dry runs."
        ),
    )

    # ── Generation flags (forwarded to run_show.py) ──────────────────
    gen_group = parser.add_argument_group("generation (forwarded to run_show.py)")

    gen_group.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["debate", "news", "podcast"],
        dest="show_type",
        help="Force a show type (skip the router).",
    )

    gen_group.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Base output directory for generated JSON (default: output/).",
    )

    gen_group.add_argument(
        "--research",
        action="store_true",
        default=None,
        dest="research",
        help="Force research ON.",
    )

    gen_group.add_argument(
        "--no-research",
        action="store_false",
        dest="research",
        help="Force research OFF.",
    )

    gen_group.add_argument(
        "--tag", "-t",
        type=str,
        default=None,
        help="Label baked into the output filename.",
    )

    gen_group.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Override target debate turns.",
    )

    gen_group.add_argument(
        "--tone",
        type=str,
        default=None,
        help="Override tone (works for all show types).",
    )

    gen_group.add_argument(
        "--debate-format",
        type=str,
        default=None,
        dest="debate_format",
        help="Override debate format (e.g. 'oxford', 'crossfire').",
    )

    gen_group.add_argument(
        "--segments",
        type=int,
        default=None,
        help="Override number of news segments.",
    )

    gen_group.add_argument(
        "--max-retries",
        type=int,
        default=None,
        dest="max_retries",
        help="Max editor retry attempts per chunk (default: 3).",
    )

    # ── Duration control (forwarded to run_show.py) ──────────────────
    dur_group = parser.add_argument_group("duration control")

    dur_group.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Target duration in minutes.",
    )

    dur_group.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Acceptable deviation from target, in %% (default: 10).",
    )

    # ── Render flags (forwarded to render_show.py) ───────────────────
    render_group = parser.add_argument_group("rendering (forwarded to render_show.py)")

    render_group.add_argument(
        "--engine", "-e",
        type=str,
        default="kokoro",
        choices=["kokoro", "piper"],
        help="TTS engine (default: kokoro).",
    )

    render_group.add_argument(
        "--format", "-f",
        type=str,
        default="mp3",
        choices=["mp3", "wav", "flac", "ogg"],
        help="Audio format (default: mp3).",
    )

    render_group.add_argument(
        "--voice-map",
        type=str,
        default=None,
        help='JSON mapping of speaker names to voice IDs.',
    )

    render_group.add_argument(
        "--voice-policy",
        type=str,
        default="auto",
        choices=["auto", "matched"],
        help="Voice assignment strategy (default: auto).",
    )

    render_group.add_argument(
        "--model-dir",
        type=str,
        default="models/piper",
        help="Piper model directory (default: models/piper).",
    )

    render_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID for Kokoro.",
    )

    render_group.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip volume normalization.",
    )

    render_group.add_argument(
        "--no-strip-directions",
        action="store_true",
        default=False,
        dest="no_strip_directions",
        help=(
            "Disable stripping of parenthetical stage directions (e.g. "
            "'(laughs)', '(dramatic pause)') before synthesis. Stripping "
            "is ON by default because no supported TTS engine can act on "
            "these cues."
        ),
    )

    # ── Shared flags ─────────────────────────────────────────────────
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help=(
            "Suppress progress output. In full mode, only prints the "
            "final audio path."
        ),
    )

    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Argv builders for sub-scripts
# ─────────────────────────────────────────────────────────────────────────────
# run_show.py and render_show.py both accept argv: list[str] as their
# main() parameter. These helpers build the right argv from our parsed
# args namespace, forwarding only the flags each sub-script knows about.

def _build_generate_argv(args: argparse.Namespace) -> list[str]:
    """Build an argv list for run_show.main().

    Always includes --quiet so we can reliably parse the output path
    from stdout.  run.py provides its own user-facing progress
    messages ("Step 1/2", "Step 2/2").

    Args:
        args: Parsed namespace from our parse_args().

    Returns:
        List of string arguments suitable for run_show.main(argv).
    """
    argv = [str(args.prompt), "--quiet"]

    if args.show_type:
        argv += ["--type", args.show_type]
    if args.output != Path("output"):
        argv += ["--output", str(args.output)]
    if args.research is True:
        argv.append("--research")
    elif args.research is False:
        argv.append("--no-research")
    if args.tag:
        argv += ["--tag", args.tag]
    if args.turns is not None:
        argv += ["--turns", str(args.turns)]
    if args.tone is not None:
        argv += ["--tone", args.tone]
    if args.debate_format is not None:
        argv += ["--format", args.debate_format]
    if args.segments is not None:
        argv += ["--segments", str(args.segments)]
    if args.max_retries is not None:
        argv += ["--max-retries", str(args.max_retries)]
    if args.duration is not None:
        argv += ["--duration", str(args.duration)]
    if args.tolerance is not None:
        argv += ["--tolerance", str(args.tolerance)]

    return argv


def _build_render_argv(
    json_path: str | Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build an argv list for render_show.main().

    Args:
        json_path: Path to the show JSON to render.
        args:      Parsed namespace from our parse_args().

    Returns:
        List of string arguments suitable for render_show.main(argv).
    """
    argv = [str(json_path)]

    argv += ["--engine", args.engine]
    argv += ["--format", args.format]

    if args.audio_output:
        argv += ["--output", str(args.audio_output)]
    if args.voice_map:
        argv += ["--voice-map", args.voice_map]
    if args.voice_policy != "auto":
        argv += ["--voice-policy", args.voice_policy]
    if args.model_dir != "models/piper":
        argv += ["--model-dir", args.model_dir]
    if args.model:
        argv += ["--model", args.model]
    if args.no_normalize:
        argv.append("--no-normalize")
    if args.no_log_audio_duration:
        argv.append("--no-log-audio-duration")
    if args.no_strip_directions:
        argv.append("--no-strip-directions")
    if args.quiet:
        argv.append("--quiet")

    return argv


# ─────────────────────────────────────────────────────────────────────────────
# Mode runners
# ─────────────────────────────────────────────────────────────────────────────

def _mode_generate(args: argparse.Namespace) -> Path:
    """Run generate mode: prompt → JSON.

    Calls run_show.main() with --quiet to capture the output path.
    The JSON path is printed to stdout by run_show.py when in quiet mode.

    Args:
        args: Parsed namespace from parse_args().

    Returns:
        Path to the generated JSON file.
    """
    import run_show

    argv = _build_generate_argv(args)

    # run_show.py in --quiet mode prints just the output path to stdout.
    # Capture it so we can return the path for the full-mode pipeline.
    captured = io.StringIO()
    with redirect_stdout(captured):
        run_show.main(argv)

    json_path = captured.getvalue().strip().splitlines()[-1]
    return Path(json_path)


def _mode_render(json_path: str | Path, args: argparse.Namespace) -> None:
    """Run render mode: JSON → audio.

    Calls render_show.main() with the appropriate flags.

    Args:
        json_path: Path to the show JSON to render.
        args:      Parsed namespace from parse_args().
    """
    import render_show

    argv = _build_render_argv(json_path, args)
    render_show.main(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Run the vox_romana master pipeline.

    Dispatches to generate, render, or full mode based on --mode.
    In full mode, generates the script first, then renders it to audio.

    Args:
        argv: CLI arguments. Defaults to sys.argv[1:] if None.
              Exposed as a parameter so tests can call main() directly.
    """
    args = parse_args(argv)

    # ── Validate mode-specific requirements ──────────────────────────
    if args.mode in ("generate", "full") and args.prompt is None:
        print(
            "Error: prompt file is required for "
            f"'{args.mode}' mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "render" and args.script_input is None:
        print(
            "Error: --script-input is required for 'render' mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "render" and args.script_input is not None:
        if not args.script_input.exists():
            print(
                f"Error: script file not found: {args.script_input}",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── Banner ───────────────────────────────────────────────────────
    if not args.quiet:
        print(f"\n{'═' * 60}")
        print(f"  VOX ROMANA — Master Pipeline")
        print(f"{'═' * 60}")
        print(f"  Mode: {args.mode}")
        if args.prompt:
            print(f"  Prompt: {args.prompt}")
        if args.script_input:
            print(f"  Script input: {args.script_input}")
        print()

    # ── Dispatch ─────────────────────────────────────────────────────

    if args.mode == "generate":
        # Generate only — produce JSON, no audio.
        json_path = _mode_generate(args)
        if not args.quiet:
            print(f"\n{'═' * 60}")
            print(f"  Generated: {json_path}")
            print(f"{'═' * 60}\n")
        else:
            print(json_path)

    elif args.mode == "render":
        # Render only — audio from existing JSON.
        _mode_render(args.script_input, args)

    elif args.mode == "full":
        # Full pipeline — generate then render.
        if not args.quiet:
            print("  Step 1/2: Generating show script...\n")

        json_path = _mode_generate(args)

        if not args.quiet:
            print(f"  Script: {json_path}")
            print(f"\n  Step 2/2: Rendering audio...\n")

        _mode_render(json_path, args)

        if not args.quiet:
            print(f"\n{'═' * 60}")
            print(f"  Pipeline complete!")
            print(f"  Script: {json_path}")
            audio_suffix = f".{args.format}"
            audio_path = args.audio_output or json_path.with_suffix(audio_suffix)
            print(f"  Audio:  {audio_path}")
            print(f"{'═' * 60}\n")
        else:
            # In quiet mode, print the audio path for scripting.
            audio_suffix = f".{args.format}"
            audio_path = args.audio_output or json_path.with_suffix(audio_suffix)
            print(audio_path)


if __name__ == "__main__":
    main()

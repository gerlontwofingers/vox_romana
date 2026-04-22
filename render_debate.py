#!/usr/bin/env python3
"""render_debate.py — turn a debate JSON into a spoken audio file.

If run_debate.py is the kitchen that cooks the meal, this script is the
waiter who delivers it to the table — with garnish, of course.

    debate JSON → TTS engine → mixer → audio file (mp3/wav/flac)

THE DRIVE-THROUGH ANALOGY:

    You pull up to the speaker with your order ticket (the JSON file).
    The person at the window asks two questions:

        1. "Which speaker system?" → --engine (piper or kokoro)
        2. "Would you like fries?" → --format, --output, --voice-map

    Then the kitchen (TTS engine) cooks your order, the mixer bags it up,
    and you drive off with your steaming hot audio file.

ENGINES:

    piper   — Local, free, CPU-only. Fast and reliable but flat delivery.
              Requires: pip install piper-tts + a downloaded .onnx model.
              Good for: testing, quick drafts, non-Apple machines.

    kokoro  — Local, free, GPU-accelerated on Apple Silicon via MLX.
              Requires: pip install mlx mlx-audio (already installed).
              Model: auto-downloads ~2GB on first use, cached after that.
              Good for: production audio, emotional delivery, final renders.
              Features: emotion control via `instruct` parameter.

USAGE:

    # Simplest: render with Kokoro (default engine).
    python render_debate.py output/ubi_debate_test_01_20260405_100526.json

    # Pick your engine.
    python render_debate.py output/debate.json --engine piper
    python render_debate.py output/debate.json --engine kokoro

    # Control the output.
    python render_debate.py output/debate.json --output output/debate.wav --format wav

    # Quiet mode (no progress output — just the audio).
    python render_debate.py output/debate.json --quiet

VOICE ASSIGNMENT:

    By default, voices are auto-assigned round-robin from the engine's
    available voices. For manual control, pass a voice map as JSON:

        python render_debate.py output/debate.json \\
            --voice-map '{"Dr. Maya Chen": "af_heart", "Dr. Robert Vance": "am_adam"}'

    Kokoro voices: af_heart, am_adam, bf_emma, bm_daniel, ... (see kokoro_engine.py)
    Piper voices: depends on which .onnx models you've downloaded.

FULL OPTIONS:

    python render_debate.py --help
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Load environment variables from .env ──────────────────────────────
# (Not strictly needed for TTS, but keeps the pattern consistent with
# run_debate.py in case engines ever need API keys.)
from dotenv import load_dotenv
load_dotenv()

from vox_romana.tts.engine import VoiceConfig
from vox_romana.tts.renderer import TTSRenderer


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def make_engine(engine_name: str, **kwargs):
    """Build a TTS engine by name.

    This is the "valet" — you say which car you want and it goes to the
    garage to fetch it. You don't need to know where the keys are.

    Args:
        engine_name: "kokoro" or "piper".
        **kwargs:    Engine-specific arguments (e.g. model_dir for Piper).

    Returns:
        A BaseTTSEngine instance, ready to synthesize.

    Raises:
        ValueError: If the engine name is unknown.
    """
    engine_name = engine_name.lower().strip()

    if engine_name == "kokoro":
        from vox_romana.tts.engines.kokoro_engine import KokoroEngine
        model = kwargs.get("model", None)
        if model:
            return KokoroEngine(model=model)
        return KokoroEngine()

    elif engine_name == "piper":
        from vox_romana.tts.engines.piper_engine import PiperEngine
        model_dir = kwargs.get("model_dir", "piper-voices")
        return PiperEngine(model_dir=model_dir)

    elif engine_name == "both":
        from vox_romana.tts.engines.multi_engine import MultiEngine
        from vox_romana.tts.engines.kokoro_engine import KokoroEngine
        from vox_romana.tts.engines.piper_engine import PiperEngine
        model = kwargs.get("model", None)
        model_dir = kwargs.get("model_dir", "piper-voices")
        kokoro = KokoroEngine(model=model) if model else KokoroEngine()
        piper = PiperEngine(model_dir=model_dir)
        return MultiEngine([kokoro, piper])

    else:
        raise ValueError(
            f"Unknown engine: {engine_name!r}. "
            f"Available engines: kokoro, piper, both"
        )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments to parse (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with all options.
    """
    parser = argparse.ArgumentParser(
        prog="render_debate",
        description=(
            "Turn a debate JSON file into a spoken audio file. "
            "Uses local TTS engines (Kokoro for Apple Silicon GPU, "
            "Piper for CPU). No cloud APIs needed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python render_debate.py output/debate.json\n"
            "  python render_debate.py output/debate.json --engine piper\n"
            "  python render_debate.py output/debate.json --output my_debate.mp3\n"
            "  python render_debate.py output/debate.json --format wav --quiet\n"
        ),
    )

    # ── Required ─────────────────────────────────────────────────────
    parser.add_argument(
        "json_file",
        type=str,
        help=(
            "Path to the debate JSON file (the output of run_debate.py). "
            "This is your order ticket."
        ),
    )

    # ── Engine ───────────────────────────────────────────────────────
    parser.add_argument(
        "--engine", "-e",
        type=str,
        default="kokoro",
        choices=["kokoro", "piper", "both"],
        help=(
            "Which TTS engine to use. "
            "kokoro = Apple Silicon GPU (best quality, default). "
            "piper = CPU-only (faster setup, lower quality). "
            "(default: kokoro)"
        ),
    )

    # ── Output ───────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=(
            "Output audio file path. If not specified, auto-generated "
            "from the input filename (e.g. debate.json → debate.mp3). "
        ),
    )

    parser.add_argument(
        "--format", "-f",
        type=str,
        default="mp3",
        choices=["mp3", "wav", "flac", "ogg"],
        help="Audio format (default: mp3). Anything ffmpeg supports.",
    )

    # ── Voice control ────────────────────────────────────────────────
    parser.add_argument(
        "--voice-map",
        type=str,
        default=None,
        help=(
            'JSON string mapping speaker names to voice IDs. '
            'Example: \'{"Dr. Chen": "af_heart", "Dr. Vance": "am_adam"}\' '
            'If not provided, voices are auto-assigned round-robin.'
        ),
    )

    # ── Engine-specific ──────────────────────────────────────────────
    parser.add_argument(
        "--model-dir",
        type=str,
        default="piper-voices",
        help="Piper model directory (default: piper-voices). Ignored for Kokoro.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "HuggingFace model ID for Kokoro. "
            "Default: mlx-community/Kokoro-82M-bf16. "
            "Ignored for Piper."
        ),
    )

    # ── Flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip volume normalization (not recommended).",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output. Just produce the file.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the render pipeline.

    Args:
        argv: Arguments (defaults to sys.argv[1:]).
    """
    args = parse_args(argv)

    # ── Validate input ───────────────────────────────────────────────
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    if not json_path.suffix == ".json":
        print(f"Warning: File doesn't end in .json: {json_path}", file=sys.stderr)

    # ── Determine output path ────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate: same name, new extension.
        output_path = json_path.with_suffix(f".{args.format}")

    # ── Parse voice map ──────────────────────────────────────────────
    voice_map = None
    if args.voice_map:
        try:
            raw_map = json.loads(args.voice_map)
            voice_map = {
                name: VoiceConfig(voice_id=vid)
                for name, vid in raw_map.items()
            }
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in --voice-map: {e}", file=sys.stderr)
            sys.exit(1)

    # ── Build the engine ─────────────────────────────────────────────
    engine_kwargs = {}
    if args.engine == "piper":
        engine_kwargs["model_dir"] = args.model_dir
    elif args.engine == "kokoro" and args.model:
        engine_kwargs["model"] = args.model

    if not args.quiet:
        print(f"\n  Engine: {args.engine}")

    try:
        engine = make_engine(args.engine, **engine_kwargs)
    except Exception as e:
        print(f"Error creating {args.engine} engine: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Build the renderer ───────────────────────────────────────────
    renderer = TTSRenderer(engine, voice_map=voice_map)

    # ── Render! ──────────────────────────────────────────────────────
    if not args.quiet:
        print(f"  Input:  {json_path}")
        print(f"  Output: {output_path}")
        print()

    start_time = time.time()

    result = renderer.render_file(
        json_path,
        output_path,
        format=args.format,
        normalize=not args.no_normalize,
        quiet=args.quiet,
    )

    elapsed = time.time() - start_time

    # ── Report ───────────────────────────────────────────────────────
    if not args.quiet:
        duration_sec = result.duration_ms / 1000
        print(f"\n  Done!")
        print(f"  Audio:    {result.output_path}")
        print(f"  Duration: {duration_sec:.1f}s")
        print(f"  Events:   {result.rendered} rendered, {result.skipped} skipped")
        print(f"  Time:     {elapsed:.1f}s")
        print()


if __name__ == "__main__":
    main()

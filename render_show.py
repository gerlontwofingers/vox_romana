#!/usr/bin/env python3
"""render_show.py — turn any show JSON into a spoken audio file.

Unified renderer that replaces render_debate.py and render_news.py with a
single command. Works for debates, news shows, or any future show type
that produces an "events" key in its JSON output.

    show JSON → TTS engine → mixer → audio file (mp3/wav/flac)

THE SOUND BOOTH ANALOGY:

    This script is the recording studio at the end of the production line.
    The show has already been written and edited (by run_show.py). Now
    it needs to be recorded, mixed, and mastered into an audio file.

    1. YOUR JSON FILE is the script. Every line of dialogue, every speaker
       tag, every emotional cue — it's all written down. The script arrived
       from the writers' room (run_show.py) fully finalized.

    2. THE ENGINE is the voice talent booth. You pick which booth to use:
       kokoro (Apple Silicon GPU — the premium studio with warm delivery)
       or piper (CPU-only — the quick-and-dirty demo booth). Both booths
       produce audio from a script; they just differ in quality and speed.

    3. THE VOICE MAP is the casting sheet. By default, the renderer
       auto-assigns voices round-robin from the engine's roster, using
       gender hints from the show JSON if available. For manual control,
       pass --voice-map with a JSON mapping of speaker names to voice IDs.

    4. THE OUTPUT is the mastered recording — a single audio file in your
       choice of format (mp3, wav, flac, ogg). Volume-normalized by
       default so nothing clips or whispers.

ENGINES:

    kokoro (default):
        Local, free, GPU-accelerated on Apple Silicon via MLX.
        Requires: pip install mlx mlx-audio (already in requirements.txt).
        Model: auto-downloads ~2GB on first use, cached after that.
        Best for: production audio, emotional delivery, final renders.
        Features: emotion-aware synthesis via the instruct parameter.

    piper:
        Local, free, CPU-only. Fast and lightweight but flatter delivery.
        Requires: pip install piper-tts + a downloaded .onnx model.
        Best for: testing, quick drafts, non-Apple machines, CI pipelines.

USAGE:

    # Simplest: render the output of run_show.py with the default engine.
    python render_show.py output/text/debates/ubi_debate_20260405_100526.json
    python render_show.py output/text/news/tech_news_20260406_003942.json

    # Pick your engine:
    python render_show.py output/show.json --engine kokoro
    python render_show.py output/show.json --engine piper

    # Custom output path and format:
    python render_show.py output/show.json --output my_show.wav --format wav
    python render_show.py output/show.json --output final.flac --format flac

    # Manual voice casting:
    python render_show.py output/debate.json \\
        --voice-map '{"Dr. Maya Chen": "af_heart", "Dr. Vance": "am_adam"}'

    # Use a specific Kokoro model from HuggingFace:
    python render_show.py output/show.json --model mlx-community/Kokoro-82M-bf16

    # Use a specific Piper model directory:
    python render_show.py output/show.json --engine piper --model-dir models/piper

    # Skip volume normalization (not usually recommended):
    python render_show.py output/show.json --no-normalize

    # Quiet mode (no progress output — good for scripting):
    python render_show.py output/show.json --quiet

    # Full pipeline: generate a show then immediately render it.
    python run_show.py prompts/ubi_debate.txt --quiet | xargs python render_show.py

VOICE IDS:

    Kokoro voices follow a naming convention:
        af_*  = American female (af_heart, af_bella, af_sarah, ...)
        am_*  = American male   (am_adam, am_michael, ...)
        bf_*  = British female  (bf_emma, bf_isabella, ...)
        bm_*  = British male    (bm_daniel, bm_george, ...)

    Piper voices depend on which .onnx models you've downloaded.
    See https://github.com/rhasspy/piper for the full catalog.

ENVIRONMENT:
    No API keys required. All TTS engines run locally.
    DEEPSEEK_API_KEY is loaded from .env for consistency but not used here.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from vox_romana.tts.engine import VoiceConfig
from vox_romana.tts.renderer import TTSRenderer


def make_engine(engine_name: str, **kwargs):
    """Build a TTS engine by name.

    This is the booth selector — you say which recording booth you want
    and it sets everything up. You don't need to know about ONNX runtimes
    or MLX model loading.

    Args:
        engine_name: "kokoro" or "piper" (case-insensitive).
        **kwargs:    Engine-specific arguments forwarded to the constructor.
                     kokoro accepts: model (HuggingFace model ID).
                     piper accepts: model_dir (path to .onnx models).

    Returns:
        A BaseTTSEngine instance, ready to synthesize audio.

    Raises:
        ValueError: If the engine name isn't recognized.
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
        model_dir = kwargs.get("model_dir", "models/piper")
        return PiperEngine(model_dir=model_dir)

    else:
        raise ValueError(
            f"Unknown engine: {engine_name!r}. "
            f"Available engines: kokoro, piper"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the render pipeline.

    The only REQUIRED argument is the path to a show JSON file (the
    output of run_show.py, run_debate.py, or run_news.py).

    Args:
        argv: Arguments to parse. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed namespace with all options.
    """
    parser = argparse.ArgumentParser(
        prog="render_show",
        description=(
            "Turn a show JSON file into a spoken audio file. "
            "Works for any show type (debate, news, etc.) — the "
            "renderer reads the 'events' key and doesn't care "
            "about the show type."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Render with the default Kokoro engine:\n"
            "  python render_show.py output/text/debates/ubi_debate_20260405.json\n"
            "  python render_show.py output/text/news/tech_news_20260406.json\n"
            "\n"
            "  # Pick your engine:\n"
            "  python render_show.py output/show.json --engine piper\n"
            "  python render_show.py output/show.json --engine kokoro\n"
            "\n"
            "  # Custom output path and format:\n"
            "  python render_show.py output/show.json -o final.wav -f wav\n"
            "  python render_show.py output/show.json -o podcast.flac -f flac\n"
            "\n"
            "  # Manual voice assignment:\n"
            "  python render_show.py output/debate.json \\\n"
            '      --voice-map \'{"Host": "af_heart", "Guest": "am_adam"}\'\n'
            "\n"
            "  # Quiet mode for scripting:\n"
            "  python render_show.py output/show.json --quiet\n"
            "\n"
            "  # Full pipeline (generate + render in one shot):\n"
            "  python run_show.py prompts/ubi_debate.txt --quiet | \\\n"
            "      xargs python render_show.py --quiet\n"
        ),
    )

    # ── Required ─────────────────────────────────────────────────────
    parser.add_argument(
        "json_file",
        type=str,
        help=(
            "Path to the show JSON file (the output of run_show.py). "
            "This is the script that the sound booth will record."
        ),
    )

    # ── Engine selection ─────────────────────────────────────────────
    parser.add_argument(
        "--engine", "-e",
        type=str,
        default="kokoro",
        choices=["kokoro", "piper"],
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
            "from the input filename (e.g. show.json → show.mp3)."
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

    parser.add_argument(
        "--voice-policy",
        type=str,
        default="auto",
        choices=["auto", "matched"],
        dest="voice_policy",
        help=(
            "How to assign TTS voices. "
            "'auto' = gender-based round-robin (legacy default). "
            "'matched' = catalog-based matching using voice_hints from the "
            "show JSON (picks voices that fit each character's casting notes). "
            "(default: auto)"
        ),
    )

    # ── Engine-specific options ──────────────────────────────────────
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/piper",
        help="Piper model directory (default: models/piper). Ignored for Kokoro.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "HuggingFace model ID for Kokoro "
            "(default: mlx-community/Kokoro-82M-bf16). "
            "Ignored for Piper."
        ),
    )

    # ── Flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip volume normalization (not recommended for final output).",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output. Just produce the audio file.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the render pipeline: JSON → TTS → audio file.

    Args:
        argv: CLI arguments. Defaults to sys.argv[1:] if None.
              Exposed as a parameter so tests can call main() directly.
    """
    args = parse_args(argv)

    # ── Validate input ───────────────────────────────────────────────
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    # ── Determine output path ────────────────────────────────────────
    # If the user didn't specify --output, generate one by swapping the
    # extension: show.json → show.mp3 (or .wav, .flac, .ogg).
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = json_path.with_suffix(f".{args.format}")

    # ── Parse voice map ──────────────────────────────────────────────
    # The voice map is optional JSON that pins specific speakers to
    # specific voice IDs. Without it, voices are auto-assigned.
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
    # Forward engine-specific kwargs (model_dir for Piper, model for Kokoro).
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

    # ── Build the renderer and render ────────────────────────────────
    renderer = TTSRenderer(engine, voice_map=voice_map)

    if not args.quiet:
        print(f"  Input:  {json_path}")
        print(f"  Output: {output_path}")
        print(f"  Voice policy: {args.voice_policy}")
        print()

    start_time = time.time()

    # Load the show JSON so we can apply the voice policy before rendering.
    with open(json_path) as f:
        show_data = json.load(f)

    # Under 'auto' policy, strip voice_hints so the renderer falls back
    # to the simpler gender-based round-robin assignment. Under 'matched',
    # voice_hints flow through to the catalog-matching algorithm.
    if args.voice_policy == "auto":
        show_data.pop("voice_hints", None)

    if not args.quiet:
        title = show_data.get("title", json_path.stem)
        print(f"\n  Rendering: {title}")
        print(f"  Source: {json_path}")
        print()

    result = renderer.render(
        show_data,
        output_path,
        format=args.format,
        normalize=not args.no_normalize,
        quiet=args.quiet,
    )

    elapsed = time.time() - start_time

    # ── Report results ───────────────────────────────────────────────
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

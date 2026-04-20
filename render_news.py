#!/usr/bin/env python3
"""render_news.py — turn a news show JSON into a spoken audio file.

Same pattern as render_debate.py. Takes the JSON output of run_news.py
and feeds it through the TTS engine to produce an audio file.

    news JSON → TTS engine → mixer → audio file (mp3/wav/flac)

The TTSRenderer is content-agnostic — it reads the "events" key from the
JSON and assigns voices to speakers. Works identically for debates and
news shows.

USAGE:
    python render_news.py output/tech_news_20260405_100526.json
    python render_news.py output/news.json --engine piper
    python render_news.py output/news.json --output output/news.wav --format wav
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
    """Build a TTS engine by name."""
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
    parser = argparse.ArgumentParser(
        prog="render_news",
        description=(
            "Turn a news show JSON file into a spoken audio file. "
            "Uses local TTS engines (Kokoro for Apple Silicon GPU, "
            "Piper for CPU)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python render_news.py output/news.json\n"
            "  python render_news.py output/news.json --engine piper\n"
            "  python render_news.py output/news.json --output my_news.mp3\n"
        ),
    )

    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the news show JSON file (output of run_news.py).",
    )

    parser.add_argument(
        "--engine", "-e",
        type=str,
        default="kokoro",
        choices=["kokoro", "piper"],
        help="TTS engine (default: kokoro).",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output audio file path.",
    )

    parser.add_argument(
        "--format", "-f",
        type=str,
        default="mp3",
        choices=["mp3", "wav", "flac", "ogg"],
        help="Audio format (default: mp3).",
    )

    parser.add_argument(
        "--voice-map",
        type=str,
        default=None,
        help='JSON mapping speaker names to voice IDs.',
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/piper",
        help="Piper model directory.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID for Kokoro.",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip volume normalization.",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = json_path.with_suffix(f".{args.format}")

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

    renderer = TTSRenderer(engine, voice_map=voice_map)

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

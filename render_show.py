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
    python render_show.py output/show.json --engine piper --model-dir piper-voices

    # Skip volume normalization (not usually recommended):
    python render_show.py output/show.json --no-normalize

    # Quiet mode (no progress output — good for scripting):
    python render_show.py output/show.json --quiet

    # Disable WPM logging (for benchmarks or dry runs):
    python render_show.py output/show.json --no-log-audio-duration

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
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from vox_romana.tts.engine import VoiceConfig
from vox_romana.tts.renderer import TTSRenderer, strip_stage_directions

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Content-type detection from show JSON keys
# ─────────────────────────────────────────────────────────────────────────────
# Each show type stamps a unique ID field into its serialized JSON.
# We inspect those fields to figure out what kind of show we rendered,
# so the WPM observation lands in the right catalog bucket.

_CONTENT_TYPE_KEYS: dict[str, str] = {
    "debate_id": "debate",
    "show_id": "news",
    "podcast_id": "podcast",
    "book_id": "audiobook",
}


def _detect_content_type(show_data: dict) -> str:
    """Infer the content type from the show JSON's top-level keys.

    Checks for known ID fields stamped by each serializer:
        debate_id  → "debate"
        show_id    → "news"
        podcast_id → "podcast"
        book_id    → "audiobook"

    Returns:
        A content type string, or "unknown" if no key matched.
    """
    for key, content_type in _CONTENT_TYPE_KEYS.items():
        if key in show_data:
            return content_type
    return "unknown"


def _count_event_words(show_data: dict) -> int:
    """Count total spoken words across all events in the show JSON.

    Events live under the "events" key. Each event has a "text" field
    containing the spoken dialogue. Stage directions are stripped before
    counting so the WPM estimate reflects only the words actually spoken.

    Returns:
        Total word count across all events (0 if no events found).
    """
    return sum(
        len(strip_stage_directions(event.get("text", "")).split())
        for event in show_data.get("events", [])
    )


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


def _build_music_engine(args: argparse.Namespace):
    """Build a MusicGenEngine if --music was requested, else return None.

    The engine is constructed only when ``args.music`` is True. On any
    import or construction error (e.g. transformers not installed), the
    function prints a warning and returns None so the rest of the render
    continues without music.

    Args:
        args: Parsed namespace from parse_args().

    Returns:
        A MusicGenEngine instance, or None if music was not requested
        or the engine could not be constructed.
    """
    if not args.music:
        return None

    try:
        from vox_romana.tts.music_engine import MusicGenEngine, MODEL_ALIASES
        from vox_romana.core.paths import HF_CACHE_DIR
        model_name = MODEL_ALIASES.get(args.music_model, f"facebook/musicgen-{args.music_model}")
        return MusicGenEngine(model_name=model_name, cache_dir=HF_CACHE_DIR)
    except Exception as exc:
        import sys
        print(
            f"Warning: could not initialise MusicGenEngine ({exc}). "
            "Continuing without music generation.",
            file=sys.stderr,
        )
        return None


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
            "  # Render with the default engine (both — all Kokoro + Piper voices):\n"
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
        default="both",
        choices=["kokoro", "piper", "both"],
        help=(
            "Which TTS engine to use. "
            "both = all Kokoro + Piper voices (default). "
            "kokoro = Apple Silicon GPU only. "
            "piper = CPU-only. "
            "(default: both)"
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
        default="piper-voices",
        help="Piper model directory (default: piper-voices). Ignored for Kokoro.",
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

    # ── Audio duration feedback ──────────────────────────────────────
    # By default, after rendering we log the actual audio duration and
    # word count to the catalog's wpm_stats table. This feeds the
    # SpeechRateAdvisor's learning loop — future duration estimates
    # improve as we accumulate real observations.
    #
    # Disable with --no-log-audio-duration for benchmarks, dry runs,
    # or when you don't want to pollute the catalog.
    #
    # Examples:
    #   # Normal render — WPM logged automatically:
    #   python render_show.py output/text/debates/ubi.json
    #
    #   # Disable logging (dry run / benchmark):
    #   python render_show.py output/show.json --no-log-audio-duration
    parser.add_argument(
        "--no-log-audio-duration",
        action="store_true",
        default=False,
        help=(
            "Disable logging actual audio duration to the catalog. "
            "By default, the renderer records WPM observations so "
            "future duration estimates improve over time."
        ),
    )

    # ── Parallel TTS rendering ──────────────────────────────────────────────
    # When > 1, spoken events are submitted to a ThreadPoolExecutor and
    # mixed in script order once their futures resolve. Music cues stay
    # sequential. Default is 1 (sequential, zero behaviour change).
    #
    # Engine notes:
    #   - Piper:  safe at any worker count (subprocess-per-call)
    #   - Kokoro: NOT validated (MLX/MPS may serialize internally)
    #   - MusicGen: stays sequential regardless
    parser.add_argument(
        "--tts-workers",
        type=int,
        default=1,
        help=(
            "Number of TTS worker threads (default: 1, sequential). "
            "Values > 1 parallelize spoken-event synthesis. Recommended "
            "for Piper only — Kokoro (MLX/MPS) parallel safety is not "
            "yet validated."
        ),
    )

    # ── Stage direction stripping ───────────────────────────────────────────────
    # LLMs often write stage directions inline in utterance text,
    # e.g. "(laughs) Well, that's funny." or "(dramatic pause) Gold."
    # Standard TTS engines can't act on these cues, so they're spoken
    # literally, which sounds broken.
    #
    # Stripping is ON by default because no supported TTS engine
    # (Piper, Kokoro) can perform the gestures/expressions. Pass
    # --no-strip-directions to disable if you need the raw text.
    parser.add_argument(
        "--no-strip-directions",
        action="store_false",
        default=True,
        dest="strip_directions",
        help=(
            "Disable stripping of parenthetical stage directions from "
            "utterance text before synthesis. By default, blocks like "
            "'(laughs)' and '(dramatic pause)' are removed because no "
            "supported TTS engine can act on them (they would be spoken "
            "literally otherwise). The original JSON is never modified."
        ),
    )

    parser.add_argument(
        "--no-whisper-voices",
        action="store_true",
        default=False,
        dest="no_whisper_voices",
        help=(
            "Exclude airy/whispery voices (e.g. af_kore, af_sky) from "
            "voice assignment. Useful when the listener prefers voices "
            "with more presence. Applies to both catalog-matched and "
            "gender-only fallback assignment."
        ),
    )

    # ── Music generation ────────────────────────────────────────────────
    parser.add_argument(
        "--music",
        action="store_true",
        default=False,
        dest="music",
        help=(
            "Enable generative music for bumper events and [music: …] cues "
            "in the show script. Uses MusicGen (facebook/musicgen-small by "
            "default) via HuggingFace Transformers. The model auto-downloads "
            "on first use (~300 MB for small). Requires torch and scipy "
            "(already in requirements.txt)."
        ),
    )
    parser.add_argument(
        "--music-style",
        type=str,
        default="",
        dest="music_style",
        metavar="DESCRIPTION",
        help=(
            "Default music style/genre used when a bumper event carries no "
            "explicit [music: …] cue. Examples: \"lo-fi jazz podcast bed\", "
            "\"orchestral cinematic transition\". Ignored when --music is not set."
        ),
    )
    parser.add_argument(
        "--music-model",
        type=str,
        default="small",
        dest="music_model",
        choices=["small", "medium", "large"],
        metavar="{small,medium,large}",
        help=(
            "MusicGen model size. \"small\" (default, ~300 MB) is fast on MPS "
            "and good enough for bumpers. \"medium\" (~1.5 GB) and \"large\" "
            "(~3.3 GB) produce higher quality at the cost of speed. "
            "Maps to facebook/musicgen-{size}. Ignored when --music is not set."
        ),
    )
    parser.add_argument(
        "--music-duration",
        type=float,
        default=8.0,
        dest="music_duration",
        metavar="SECONDS",
        help=(
            "Duration in seconds of each generated music clip (default: 8.0). "
            "Applies to all music cues unless the cue specifies its own length. "
            "MusicGen supports up to ~30 seconds per generation call. "
            "Ignored when --music is not set."
        ),
    )
    parser.add_argument(
        "--series-id",
        type=str,
        default=None,
        dest="series_id",
        help=(
            "Series identifier for multi-episode shows. When set, the "
            "voice registry locks each character to the same voice "
            "across all episodes that share this series ID. Falls back "
            "to the show JSON's series_id / show_name when omitted."
        ),
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
    elif args.engine == "both":
        if args.model_dir:
            engine_kwargs["model_dir"] = args.model_dir
        if args.model:
            engine_kwargs["model"] = args.model
    if not args.quiet:
        print(f"\n  Engine: {args.engine}")

    try:
        engine = make_engine(args.engine, **engine_kwargs)
    except Exception as e:
        print(f"Error creating {args.engine} engine: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Build the renderer and render ────────────────────────────────
    registry = None
    series_key = getattr(args, "series_id", None)
    try:
        from vox_romana.tts.voice_registry import VoiceRegistry
        registry = VoiceRegistry()
    except Exception as exc:
        logger.warning("VoiceRegistry unavailable, voice locking disabled: %s", exc)

    renderer = TTSRenderer(
        engine,
        voice_map=voice_map,
        strip_directions=args.strip_directions,
        registry=registry,
        series_key=series_key,
        exclude_whispery=args.no_whisper_voices,
        music_engine=_build_music_engine(args),
        music_style=args.music_style,
        music_duration=args.music_duration,
        parallel_workers=args.tts_workers,
    )

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
    duration_sec = result.duration_ms / 1000

    if not args.quiet:
        print(f"\n  Done!")
        print(f"  Audio:    {result.output_path}")
        print(f"  Duration: {duration_sec:.1f}s")
        print(f"  Events:   {result.rendered} rendered, {result.skipped} skipped")
        print(f"  Time:     {elapsed:.1f}s")

    # ── Log actual audio duration to catalog ─────────────────────────
    # This is the feedback loop that makes duration estimates smarter.
    # After rendering, we know the REAL audio length (from the TTS
    # engine) and the word count (from the script). Recording this
    # observation lets the SpeechRateAdvisor improve its WPM estimates
    # for future runs.
    #
    # Why this matters:
    #   - Static WPM defaults (150 wpm for debate, 165 for news) are
    #     just starting points.
    #   - Real WPM varies by speaker pacing, engine, and content style.
    #   - After a few logged renders, the advisor switches from static
    #     defaults to running averages from the catalog.
    #   - If the optional WPMModel (sklearn) is installed, it trains an
    #     even more accurate predictor from the accumulated data.
    if not args.no_log_audio_duration:
        content_type = _detect_content_type(show_data)
        word_count = _count_event_words(show_data)

        if content_type != "unknown" and word_count > 0 and duration_sec > 0:
            try:
                from vox_romana.core.catalog import Catalog
                from vox_romana.core.duration import SpeechRateAdvisor

                catalog = Catalog()
                advisor = SpeechRateAdvisor(catalog)
                advisor.log_observation(content_type, word_count, duration_sec)

                actual_wpm = (word_count / duration_sec) * 60.0
                if not args.quiet:
                    print(f"  WPM logged: {actual_wpm:.0f} "
                          f"({content_type}, {word_count} words, "
                          f"{duration_sec:.1f}s)")
            except Exception as e:
                # Logging is best-effort — never crash the render pipeline.
                if not args.quiet:
                    print(f"  WPM logging skipped: {e}")
        elif not args.quiet:
            if content_type == "unknown":
                print("  WPM logging skipped: unknown content type")
            elif word_count == 0:
                print("  WPM logging skipped: no words in events")
    elif not args.quiet:
        print("  WPM logging: disabled (--no-log-audio-duration)")

    if not args.quiet:
        print()


if __name__ == "__main__":
    main()

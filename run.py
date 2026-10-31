#!/usr/bin/env python3
"""run.py — master CLI for the vox_romana production pipeline.

One command to generate text, render audio/video, or do both.  This replaces
the old two-step workflow of ``run_show.py`` → ``render_show.py`` with
a single entry point that handles the full pipeline end-to-end.

    prompt file → classify → bootstrap → conductor → editor → JSON → TTS → audio → optional video

THE PRODUCTION STUDIO ANALOGY:

    Think of run.py as the executive producer. You hand them a brief and
    say "make me a show." They walk it through every department:

    1. THE WRITERS' ROOM (run_show.py) generates the script from your
       prompt — casting, profiling, directing, conducting, editing.

    2. THE SOUND BOOTH (render_show.py) records the script into audio
       using the TTS engine of your choice (Kokoro or Piper).

    3. THE ANIMATION STUDIO (--mode video) turns the rendered audio into
       an MP4 using the animation planner, image generation, timeline sync,
       and MoviePy compositing stack.

    4. THE DATA SCIENTIST (WPM logging) records how long the audio
       actually came out, so future duration estimates get better.

    run.py coordinates all four.  You can also ask it to do just one
    step (--mode generate or --mode render) if you want manual control.

MODES:

    publish:
        Render an existing show JSON into a formatted human-readable
        document (TXT and/or PDF).  No LLM required; the output is
        fully deterministic.  Supported layouts:

            debate   → stage-play script (ALL CAPS speakers, dialogue
                        indented, parenthetical stage directions).
            news     → broadcast rundown (SLUG headers, role cues).
            podcast  → Rev/Otter transcript (Speaker: + paragraphs).
            audiobook→ book chapters with chapter headings.

        python run.py --mode publish \\
            --script-input output/text/debates/ubi.json
        python run.py --mode publish \\
            --script-input output/text/debates/ubi.json \\
            --publish-format both \\
            --publish-output ~/Desktop

        Add --publish to any other mode to auto-publish after generation:

        python run.py prompts/ubi_debate.txt --mode generate --publish
        python run.py prompts/ubi_debate.txt --publish --publish-format pdf

    full (default):
        Generate the show script AND render it to audio in one shot.
        This is the batteries-included experience. After rendering,
        actual audio duration is logged to the catalog's wpm_stats
        table for the SpeechRateAdvisor feedback loop.

        python run.py prompts/ubi_debate.txt
        python run.py prompts/tech_news.txt --engine kokoro --format mp3
        python run.py prompts/ai_ethics_lecture.txt --type lecture --engine both

        # --name gives the output files a fixed stem (no timestamp):
        python run.py prompts/ubi_debate.txt --name ubi_v4
        # → output/text/debates/ubi_v4.json  +  output/text/debates/ubi_v4.mp3

    generate:
        Text generation only. Produces the show JSON but no audio.
        Equivalent to calling run_show.py directly.

        python run.py prompts/ubi_debate.txt --mode generate
        python run.py prompts/trash_pandas_01.txt --mode generate --tag v2

        # Fixed output stem:
        python run.py prompts/ubi_debate.txt --mode generate --name ubi_v4
        # → output/text/debates/ubi_v4.json

    render:
        Audio rendering only from an existing show JSON. Requires
        --script-input pointing to a previously generated JSON file.
        Equivalent to calling render_show.py directly.

        python run.py --mode render --script-input output/text/debates/ubi.json
        python run.py --mode render --script-input output/text/news/tech.json \\
            --engine piper --audio-output my_show.mp3

    video:
        Full pipeline (generate + render) plus an animation pass that
        produces an MP4 using Stable Diffusion + Ken Burns compositing.

        Platform choices (--video-platform, default: youtube):
            youtube  → 1920×1080 landscape MP4
            tiktok   → 1080×1920 portrait MP4
            both     → landscape + portrait in a single SDXL pass
            fast     → 512×512, SD 1.5, 4 steps — dev / diagnostic run

        python run.py prompts/ubi_debate.txt --mode video
        # → video_youtube.mp4  (default platform)

        python run.py prompts/ubi_debate.txt --mode video \\
            --video-platform tiktok
        # → video_tiktok.mp4

        python run.py prompts/ubi_debate.txt --mode video \\
            --video-platform both
        # → video_youtube.mp4  +  video_tiktok.mp4

        # Add generated bumper music when the script includes music cues:
        python run.py prompts/period_drama.txt --mode video \
            --music \
            --music-style "subtle harpsichord music" \
            --music-duration 12 \
            --music-model large

        # --name controls the video stem too:
        python run.py prompts/ubi_debate.txt --mode video \\
            --name ubi_v4 --video-platform both
        # → ubi_v4_youtube.mp4  +  ubi_v4_tiktok.mp4

        # Quick diagnostic render (SD 1.5, 512×512, 4 steps, 15 fps):
        python run.py prompts/ubi_debate.txt --mode video \\
            --video-platform fast
        # → video_fast.mp4

        # Custom visual style and output directory:
        python run.py prompts/tech_news.txt --mode video \\
            --visual-brief "clean studio, cool blue tones" \\
            --video-output ~/Desktop/news_video/ \\
            --video-platform youtube

USAGE:

    # === FULL MODE (generate + render) ===============================

    # Simplest: auto-detect show type, default engine (both = all voices), mp3:
    python run.py prompts/ubi_debate.txt

    # Force show type, custom engine, wav format:
    python run.py prompts/ubi_debate.txt --type debate --engine piper -f wav

    # With research and duration target:
    python run.py prompts/tech_news.txt --research --duration 15 --tolerance 5

    # Override turns, tag the run, custom audio output:
    python run.py prompts/ubi_debate.txt --turns 50 --tag v3 \\
        --audio-output output/audio/debates/ubi_v3.mp3

    # Fixed output stem — same stem used for .json, .mp3, .mp4:
    python run.py prompts/ubi_debate.txt --name ubi_v4
    # → output/text/debates/ubi_v4.json  +  ubi_v4.mp3

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

    # Fixed output stem:
    python run.py prompts/ubi_debate.txt --mode generate --name ubi_v4

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

    # === VIDEO MODE (generate + render + animate) ====================

    # Default platform (youtube = 1920×1080 landscape):
    python run.py prompts/ubi_debate.txt --mode video

    # TikTok portrait (1080×1920):
    python run.py prompts/ubi_debate.txt --mode video --video-platform tiktok

    # Both landscape + portrait in one SDXL pass:
    python run.py prompts/ubi_debate.txt --mode video --video-platform both

    # Fast diagnostic run (SD 1.5, 512×512, 4 steps):
    python run.py prompts/ubi_debate.txt --mode video --video-platform fast

    # Fixed output stem applied to all video files:
    python run.py prompts/ubi_debate.txt --mode video \\
        --name ubi_v4 --video-platform both
    # → output/.../video/ubi_v4_youtube.mp4  +  ubi_v4_tiktok.mp4

    # Custom visual brief and explicit output directory:
    python run.py prompts/tech_news.txt --mode video \\
        --visual-brief "clean studio, cool blue tones" \\
        --video-output ~/Desktop/news_video/ \\
        --video-platform youtube

    # Write video directly to a named file:
    python run.py prompts/ubi_debate.txt --mode video \\
        --video-output ~/Desktop/ubi_final.mp4

    # === PUBLISH MODE (human-readable document from JSON) ===========

    # Render an existing JSON to a formatted TXT document:
    python run.py --mode publish --script-input output/text/debates/ubi.json

    # Both TXT and PDF:
    python run.py --mode publish \\
        --script-input output/text/debates/ubi.json \\
        --publish-format both

    # Custom output directory:
    python run.py --mode publish \\
        --script-input output/text/news/tech.json \\
        --publish-format txt --publish-output ~/Desktop

    # Auto-publish after generation (adds the document step to generate/full):
    python run.py prompts/ubi_debate.txt --mode generate --publish
    python run.py prompts/ubi_debate.txt --publish --publish-format both

    # === INGEST MODE (build the knowledge base) ======================
    #
    # Feed documents into the local memory store so future --research
    # runs consult your material first (free, sub-ms) before falling
    # back to Google.  The pipeline:
    #
    #   1. Text is extracted from each file (PDF, DOCX, HTML, CSV, …)
    #   2. Extracted text is split into overlapping chunks (~800 chars)
    #   3. Chunks are written to data/memory/documents/ as JSONL
    #   4. ChromaVectorIndex embeds the chunks (all-MiniLM-L6-v2)
    #   5. Any future  run.py … --research  queries memory FIRST
    #
    # Optional packages for extra formats (plain text / JSON / XML /
    # CSV work without any extras):
    #   pip install beautifulsoup4 lxml python-docx openpyxl python-pptx
    #   pip install ebooklib odfpy striprtf  # epub, odt, rtf

    # Ingest a whole directory of papers:
    python run.py --mode ingest --ingest-path papers/

    # Ingest a single PDF with a tag for later filtering:
    python run.py --mode ingest --ingest-path report.pdf --ingest-tag research-2026

    # Preview what would be ingested — no files written:
    python run.py --mode ingest --ingest-path docs/ --dry-run

    # Force re-ingest files already processed (e.g. after an update):
    python run.py --mode ingest --ingest-path docs/ --force-reingest

    # Multiple source paths in one run:
    python run.py --mode ingest --ingest-path papers/ --ingest-path extra.pdf

    # Custom chunk size (larger = fewer but richer chunks):
    python run.py --mode ingest --ingest-path docs/ --chunk-size 1200

    # === MUSIC MODE (generate a standalone audio clip) =============
    #
    # Give it a natural-language description of the music you want
    # (inline text or a prompt file) and get back an audio file.
    # No show pipeline, no JSON — just MusicGen → audio.
    #
    # Defaults are set for quality when running on good hardware:
    #   model  = large  (~3.3 GB, MPS-accelerated on Apple Silicon)
    #   duration = 30 s  (practical per-call max for MusicGen)
    #   format = mp3  (override with --format wav for lossless)
    #   output = output/music/<slug>_<timestamp>.<format>
    #
    # Inline prompt:
    python run.py --mode music \
        --music-prompt "heavy metal with harpsichord, dark Ozzy Osbourne style"
    #
    # Prompt from file:
    python run.py prompts/my_music_brief.txt --mode music
    #
    # Custom length, lossless output:
    python run.py --mode music \
        --music-prompt "lo-fi jazz rainy day" \
        --music-duration 20 --format wav --audio-output ~/Desktop/lofi.wav
    #
    # Fixed output stem (→ output/music/theme.mp3):
    python run.py --mode music \
        --music-prompt "dramatic orchestral sting" --name theme
    #
    # Lighter model for quick tests:
    python run.py --mode music \
        --music-prompt "energetic synthwave" \
        --music-model small --music-duration 8

ENVIRONMENT:
    DEEPSEEK_API_KEY in .env (required for generate/full modes).
    No API keys needed for render mode — all TTS is local.
    Optional: GOOGLE_API_KEY + GOOGLE_CSE_ID for --research mode.

OUTPUT:
    generate mode → JSON file in output/text/{type}/
    render mode   → Audio file next to the JSON (or --audio-output)
    full mode     → Both JSON and audio files
    video mode    → MP4 next to the audio file in a video/ subdirectory
                    (or --video-output).  'both' platform produces two files.
    music mode    → Audio file in output/music/ (or --audio-output)

NAMING (--name / -n STEM):
    Pass --name STEM to fix the output filename across all modes.  The
    stem is used without an extension and without a path separator.

        python run.py prompts/ubi_debate.txt --name ubi_v4
        # → ubi_v4.json  ubi_v4.mp3

        python run.py prompts/ubi_debate.txt --mode video --name ubi_v4
        # → ubi_v4.json  ubi_v4.mp3  ubi_v4_youtube.mp4  (or _tiktok / _fast)

        python run.py --mode music --music-prompt "lo-fi jazz" --name lofi
        # → output/music/lofi.mp3

    Without --name, outputs use the auto-generated pattern:
        {type}_{prompt_stem}[_{tag}]_{timestamp}.{ext}
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
load_dotenv()
from vox_romana.core.paths import OUTPUT_DIR as _OUTPUT_DIR, PIPER_VOICES_DIR as _PIPER_VOICES_DIR
from vox_romana.core.languages import (
    LANGUAGE_CODES,
    normalize_lang_arg,
    language_display,
    supported_languages,
)

logger = logging.getLogger(__name__)


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
            "  # Publish (human-readable document from existing JSON):\n"
            "  python run.py --mode publish --script-input output/text/debates/ubi.json\n"
            "  python run.py --mode publish --script-input output/text/debates/ubi.json --publish-format both\n"
            "  python run.py prompts/ubi_debate.txt --mode generate --publish\n"
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
        choices=["generate", "render", "full", "ingest", "dedupe", "publish", "music", "analysis", "emotion-flow", "university", "video", "series", "audition", "package", "voice-prep", "voice-assign"],
        help=(
            "Pipeline mode. "
            "'generate' = text only (JSON). "
            "'render' = audio only (from existing JSON). "
            "'full' = generate + render (default). "
            "'video' = generate + render + animate (requires animation deps). "
            "'ingest' = feed documents into the knowledge store. "
            "'dedupe' = scan corpus for near-duplicate documents and report clusters. "
            "'publish' = format an existing show JSON as a human-readable "
            "document (TXT/PDF). "
            "'music' = generate a standalone audio clip from a text prompt "
            "(inline via --music-prompt or from a prompt file). "
            "'series' = manage series records (create/list/inspect). "
            "'package' = generate SRT captions + YouTube show notes from "
            "an existing show JSON (--script-input). "
            "See the USAGE block above for full examples."
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

    parser.add_argument(
        "--skip-resource-monitor",
        action="store_true",
        default=False,
        dest="skip_resource_monitor",
        help=(
            "Disable the background disk/RAM monitor. "
            "Use in CI or tests where resources are intentionally constrained."
        ),
    )

    parser.add_argument(
        "--tts-workers",
        type=int,
        default=1,
        help=(
            "Number of TTS worker threads (default: 1, sequential). "
            "Values > 1 parallelize spoken-event synthesis in the render "
            "step. Recommended for Piper only — Kokoro (MLX/MPS) parallel "
            "safety is not yet validated."
        ),
    )

    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        dest="name",
        metavar="STEM",
        help=(
            "Output filename stem for the primary artifact. No extension, no "
            "path separator. '--name ubi_v4' → ubi_v4.json, ubi_v4.mp3, "
            "ubi_v4.mp4. Combines with --output (directory). Overrides the "
            "auto-generated {type}_{prompt}_{tag}_{timestamp} stem. When both "
            "--name and --tag are given, --name wins and no timestamp is "
            "appended."
        ),
    )

    parser.add_argument(
        "--production-id",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Production identifier for longitudinal tracking. Stored in "
            "run metadata and used by the single-run ingest path."
        ),
    )

    # ── Generation flags (forwarded to run_show.py) ──────────────────
    gen_group = parser.add_argument_group("generation (forwarded to run_show.py)")

    gen_group.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["debate", "news", "podcast", "audiobook", "lecture", "radio_drama"],
        dest="show_type",
        help="Force a show type (skip the router).",
    )

    gen_group.add_argument(
        "--era",
        type=str,
        default=None,
        choices=["1940s", "1970s", "present", "future"],
        dest="era",
        help=(
            "Radio drama only: era dial for announcer register, music style, "
            "and vocabulary (default: 1940s)."
        ),
    )

    gen_group.add_argument(
        "--genre",
        type=str,
        default=None,
        dest="genre",
        help=(
            "Radio drama only: genre knob (e.g. 'noir mystery', 'sci-fi', "
            "'horror')."
        ),
    )

    gen_group.add_argument(
        "--output", "-o",
        type=Path,
        default=_OUTPUT_DIR,
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
        "--university-session",
        type=str,
        default=None,
        dest="university_session",
        help=(
            "Session ID of a completed University deep-research run. "
            "Grounds the show in Scholar notes (highest-priority search tier). "
            "E.g. --university-session a1b2c3d4e5f6"
        ),
    )

    gen_group.add_argument(
        "--series",
        type=str,
        default=None,
        dest="series",
        metavar="SERIES_ID",
        help=(
            "Series ID. Generation prepends the series writers'-bible and "
            "recent-episode summaries to the brief, AND locks character "
            "voices across episodes. Create one with: "
            "`python run.py --mode series create \"My Show\"`."
        ),
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
        help=(
            "Max fact-check attempts per chunk (default: 10). If it still "
            "fails, the chunk is rewritten to drop the unverifiable claims."
        ),
    )

    gen_group.add_argument(
        "--fact-check",
        choices=["off", "quick", "thorough"],
        default="thorough",
        dest="fact_check",
        help=(
            "Fact-check mode (default: thorough). "
            "'off': no checking. "
            "'quick': LLM-only verification. "
            "'thorough': LLM + web search for flagged claims."
        ),
    )

    gen_group.add_argument(
        "--reference-material",
        type=Path,
        default=None,
        dest="reference_material",
        help="Path to a plain-text file with factual reference material.",
    )

    gen_group.add_argument(
        "--content-mode",
        choices=["auto", "factual", "fictional-universe", "pure-fiction"],
        default="auto",
        dest="content_mode",
        help=(
            "Content truth-mode (default: auto — detected from the brief). "
            "'factual': fabrications block publication. "
            "'fictional-universe': canon violations are warnings. "
            "'pure-fiction': fact-checking is skipped entirely."
        ),
    )

    gen_group.add_argument(
        "--perspective",
        type=str,
        default=None,
        dest="perspective",
        help=(
            "Optional framing string for observer-relative language, for example "
            "\"from Guatemala's geographic perspective\". Passed through to "
            "the config and fact-checker as-is."
        ),
    )

    gen_group.add_argument(
        "--allow-bootstrap-fabrications",
        action="store_true",
        default=False,
        dest="allow_bootstrap_fabrications",
        help=(
            "Skip the bootstrap preflight abort when fabricated facts are detected "
            "in the generated outline/rundown. Prints a warning and continues. "
            "Not recommended."
        ),
    )

    gen_group.add_argument(
        "--crew-director",
        type=str,
        default=None,
        dest="crew_director",
        metavar="CREW_ID",
        help=(
            "Assign a crew director to this run. CREW_ID is a directory name "
            "under the crew root (--crew-dir). The director's profile.yaml is "
            "injected into every director check-in prompt, and an episodic "
            "memory entry is written to the director's short_term.jsonl at run end."
        ),
    )

    gen_group.add_argument(
        "--crew-factchecker",
        type=str,
        default=None,
        dest="crew_factchecker",
        metavar="CREW_ID",
        help=(
            "Assign a crew fact-checker to this run. CREW_ID is a directory name "
            "under the crew root. The fact-checker's profile is available for "
            "injection and an episodic memory entry is written at run end."
        ),
    )

    gen_group.add_argument(
        "--crew-dir",
        type=str,
        default=None,
        dest="crew_dir",
        metavar="PATH",
        help=(
            "Base directory for crew data (profile.yaml, short_term.jsonl, "
            "long_term.jsonl). Defaults to ~/.vox_romana/crew/."
        ),
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

    dur_group.add_argument(
        "--checkpoint",
        action="store_true",
        default=None,
        dest="checkpoint",
        help=(
            "Force-enable checkpoint rendering (real-time TTS measurement "
            "per segment for accurate duration tracking). Auto-enabled for "
            "runs >= 20 min. Forwarded to run_show.py."
        ),
    )
    dur_group.add_argument(
        "--no-checkpoint",
        action="store_false",
        dest="checkpoint",
        help="Disable checkpoint rendering even when it would auto-enable.",
    )

    # ── Render flags (forwarded to render_show.py) ───────────────────
    render_group = parser.add_argument_group("rendering (forwarded to render_show.py)")

    render_group.add_argument(
        "--engine", "-e",
        type=str,
        default="both",
        choices=["kokoro", "piper", "both"],
        help="TTS engine (default: both — uses all Kokoro + Piper voices). Use 'kokoro' or 'piper' for a single engine.",
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
        default=str(_PIPER_VOICES_DIR),
        help="Piper model directory (default: piper-voices).",
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

    render_group.add_argument(
        "--no-whisper-voices",
        action="store_true",
        default=False,
        dest="no_whisper_voices",
        help=(
            "Exclude airy/whispery voices (e.g. af_kore, af_sky) from "
            "voice assignment. Useful when the listener prefers voices "
            "with more presence."
        ),
    )
    render_group.add_argument(
        "--hqvoices",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="hqvoices",
        help=(
            "Only auto-assign high-quality voices (default: on). Kokoro "
            "voices graded C or better and Piper medium/high voices; "
            "synthetic-sounding low-tier voices are skipped. Missing "
            "high-quality Piper voices are downloaded when the show needs "
            "them. Use --no-hqvoices to draw from every installed voice. "
            "Explicit --voice-map entries are never changed."
        ),
    )
    render_group.add_argument(
        "--music",
        action="store_true",
        default=True,
        dest="music",
        help=(
            "Enable generative music for bumper events and [music: …] cues. "
            "Uses MusicGen via HuggingFace Transformers. Model auto-downloads "
            "on first use (~300 MB for small). Enabled by default."
        ),
    )
    render_group.add_argument(
        "--no-music",
        action="store_false",
        dest="music",
        help="Disable generative music intro/outro (enabled by default).",
    )
    render_group.add_argument(
        "--music-style",
        type=str,
        default="",
        dest="music_style",
        metavar="DESCRIPTION",
        help=(
            "Default music style used when a bumper cue has no explicit "
            "[music: …] description. E.g. 'lo-fi jazz podcast bed'. "
            "Ignored when --music is not set."
        ),
    )
    render_group.add_argument(
        "--music-model",
        type=str,
        default="small",
        dest="music_model",
        choices=["small", "medium", "large"],
        metavar="{small,medium,large}",
        help=(
            "MusicGen model size (small/medium/large). "
            "Defaults to 'small' (~300 MB). Ignored when --music is not set."
        ),
    )
    render_group.add_argument(
        "--music-duration",
        type=float,
        default=8.0,
        dest="music_duration",
        metavar="SECONDS",
        help=(
            "Duration in seconds of each generated music clip (default: 8.0 "
            "for show bumpers; 30.0 for --mode music). "
            "Ignored when --music is not set outside of music mode."
        ),
    )
    render_group.add_argument(
        "--music-volume",
        type=float,
        default=0.0,
        dest="music_volume",
        metavar="DB",
        help=(
            "Adjust music level relative to speech in dB. Negative = quieter "
            "(e.g. --music-volume -6). Default 0. Ignored when --music is not set."
        ),
    )
    render_group.add_argument(
        "--music-prompt",
        type=str,
        default=None,
        dest="music_prompt",
        metavar="TEXT",
        help=(
            "Inline text prompt for --mode music. "
            "E.g. 'heavy metal with harpsichord, dark Ozzy Osbourne style'. "
            "Mutually exclusive with a positional prompt file. "
            "Ignored in all other modes."
        ),
    )
    # ── Knowledge ingestion flags ────────────────────────────────
    ingest_group = parser.add_argument_group(
        "knowledge ingestion (--mode ingest)"
    )
    ingest_group.add_argument(
        "--ingest-path",
        dest="ingest_path",
        metavar="PATH",
        action="append",
        default=None,
        help=(
            "File or directory to ingest. May be repeated: "
            "--ingest-path docs/ --ingest-path extra.pdf"
        ),
    )
    ingest_group.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Recurse into sub-directories (default: on).",
    )
    ingest_group.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Do not recurse into sub-directories.",
    )
    ingest_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Report what would be ingested without writing any files.",
    )
    ingest_group.add_argument(
        "--force-reingest",
        dest="force_reingest",
        action="store_true",
        default=False,
        help="Re-ingest files that have already been processed.",
    )
    ingest_group.add_argument(
        "--chunk-size",
        dest="ingest_chunk_size",
        type=int,
        default=None,
        metavar="N",
        help="Target characters per chunk (default: 800).",
    )
    ingest_group.add_argument(
        "--chunk-overlap",
        dest="ingest_chunk_overlap",
        type=int,
        default=None,
        metavar="N",
        help="Overlap characters between consecutive chunks (default: 100).",
    )
    ingest_group.add_argument(
        "--ingest-tag",
        dest="ingest_tag",
        type=str,
        default="",
        metavar="TAG",
        help="Tag string embedded in every chunk (e.g. 'batch-2026-04').",
    )
    ingest_group.add_argument(
        "--no-embed",
        dest="no_embed",
        action="store_true",
        default=False,
        help=(
            "Skip Phase 2 (ChromaDB vector embedding) after ingest. "
            "Useful for large batches where you want to embed separately."
        ),
    )
    # ── Deduplication flags (--mode dedupe) ───────────────────────────
    dedupe_group = parser.add_argument_group(
        "deduplication (--mode dedupe)"
    )
    dedupe_group.add_argument(
        "--dedupe-path",
        dest="dedupe_path",
        metavar="DIR",
        default=None,
        help=(
            "Directory of JSONL corpus files to scan for near-duplicates. "
            "Defaults to the configured documents directory."
        ),
    )
    dedupe_group.add_argument(
        "--dedupe-threshold",
        dest="dedupe_threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Cosine similarity threshold (default: dedup.similarity_threshold "
            "from vox_romana.yaml, or 0.97)."
        ),
    )
    dedupe_group.add_argument(
        "--apply",
        dest="dedupe_apply",
        action="store_true",
        default=False,
        help=(
            "Remove the lower-chunk-count JSONL in each near-duplicate pair. "
            "Prompts for confirmation. Without --apply the scan is read-only."
        ),
    )
    # ── Animation flags (--mode video) ──────────────────────────────
    anim_group = parser.add_argument_group(
        "animation (--mode video)"
    )
    anim_group.add_argument(
        "--visual-brief",
        dest="visual_brief",
        type=str,
        default=None,
        metavar="TEXT",
        help=(
            "Visual style hint passed to the AnimationDirector LLM. "
            "Describes the desired look and feel of the video "
            "(e.g. 'academic documentary, warm lighting'). "
            "Only used with --mode video."
        ),
    )
    anim_group.add_argument(
        "--video-output",
        dest="video_output",
        type=Path,
        default=None,
        metavar="DIR_OR_FILE",
        help=(
            "Output directory for rendered video files, OR a full file path "
            "(e.g. ~/Desktop/ubi.mp4). When a full file path is given the "
            "filename is used directly. Defaults to a 'video/' subdirectory "
            "next to the audio file. Only used with --mode video."
        ),
    )
    anim_group.add_argument(
        "--video-platform",
        dest="video_platform",
        type=str,
        default="youtube",
        choices=["youtube", "tiktok", "both", "fast"],
        metavar="{youtube,tiktok,both,fast}",
        help=(
            "Target platform for video output (default: youtube). "
            "'youtube' = 1920×1080 landscape. "
            "'tiktok' = 1080×1920 portrait. "
            "'both' = landscape + portrait in one SDXL pass. "
            "'fast' = 512×512, SD 1.5, 4 inference steps — "
            "dev/diagnostic speed run. Only used with --mode video."
        ),
    )
    anim_group.add_argument(
        "--face-animation",
        dest="face_animation",
        action="store_true",
        default=None,
        help=(
            "Force-enable talking-head face animation (SadTalker). "
            "Generated portraits are lip-synced to per-scene audio and "
            "composited as a lower-third overlay. Requires the SadTalker "
            "package to be installed. When neither --face-animation nor "
            "--no-face-animation is given, the runner auto-detects from "
            "the visual brief (keywords: 'anchor', 'news', 'talking head', "
            "'podcast') and the AnimationDirector's style choice "
            "('talking_heads'). Only used with --mode video."
        ),
    )
    anim_group.add_argument(
        "--no-face-animation",
        dest="face_animation",
        action="store_false",
        help=(
            "Force-disable talking-head face animation even when the "
            "visual brief or director would have triggered it. Useful "
            "for fast preview renders. Only used with --mode video."
        ),
    )
    anim_group.add_argument(
        "--face-animation-model",
        dest="face_animation_model",
        type=str,
        default="sadtalker",
        choices=["sadtalker", "live_portrait"],
        metavar="{sadtalker,live_portrait}",
        help=(
            "Backend used to render talking-head clips (default: "
            "sadtalker). 'live_portrait' is a stub for a future "
            "LivePortrait integration. Only used with --mode video."
        ),
    )
    # ── Publish flags (--mode publish or --publish add-on) ───────────
    pub_group = parser.add_argument_group(
        "publishing (--mode publish  /  --publish add-on)"
    )
    pub_group.add_argument(
        "--publish",
        dest="publish",
        action="store_true",
        default=False,
        help=(
            "After generating (or rendering) a show, also produce a "
            "formatted human-readable document.  Equivalent to running "
            "--mode publish on the output JSON.  Works with --mode generate "
            "and --mode full."
        ),
    )
    pub_group.add_argument(
        "--publish-format",
        dest="publish_format",
        type=str,
        default="txt",
        choices=["txt", "pdf", "html", "both"],
        metavar="{txt,pdf,html,both}",
        help=(
            "Output format(s) for publish mode.  "
            "'txt' (default) — plain UTF-8 text, always works.  "
            "'html' — self-contained HTML document (requires jinja2).  "
            "'pdf' — requires weasyprint + jinja2 (pip install weasyprint jinja2).  "
            "'both' — TXT and PDF."
        ),
    )
    pub_group.add_argument(
        "--publish-output",
        dest="publish_output",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory where published documents are written.  "
            "Defaults to the same directory as the show JSON."
        ),
    )
    pub_group.add_argument(
        "--polish",
        dest="polish",
        action="store_true",
        default=False,
        help=(
            "Enable the optional LLM polish pass when publishing.  "
            "The polish agent rewrites spoken-style phrasing into written "
            "prose.  Off by default (not yet implemented)."
        ),
    )

    # ── Audition flags (--mode audition) ─────────────────────────────
    # Audition mode generates candidate assets (voices / music / visuals)
    # and optionally approves them into a series asset pool so future
    # episodes of that series reuse them automatically.
    audition_group = parser.add_argument_group(
        "audition (--mode audition)"
    )
    audition_group.add_argument(
        "--audition-type",
        dest="audition_type",
        type=str,
        default=None,
        choices=["voice", "music", "visual"],
        help=(
            "Kind of asset to audition.  Required for --mode audition.  "
            "'voice' tries N TTS voices; 'music' generates N MusicGen "
            "clips from --music-brief; 'visual' is reserved (stub)."
        ),
    )
    audition_group.add_argument(
        "--speaker",
        dest="speaker",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Character name for voice auditions.  Used as a label so "
            "you can tell which clips belong to which character."
        ),
    )
    audition_group.add_argument(
        "--music-brief",
        dest="music_brief",
        type=str,
        default=None,
        metavar="TEXT",
        help=(
            "Prompt for music auditions, e.g. \"upbeat jazz podcast "
            "intro, piano and bass\".  Required for --audition-type music."
        ),
    )
    audition_group.add_argument(
        "--audition-count",
        dest="audition_count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "How many candidates to generate.  Defaults: 4 voices, "
            "3 music takes."
        ),
    )
    audition_group.add_argument(
        "--approve-all",
        dest="approve_all",
        action="store_true",
        default=False,
        help=(
            "After generating the slate, approve every candidate into "
            "the series pool with no prompts.  Requires --series."
        ),
    )
    audition_group.add_argument(
        "--no-approve",
        dest="no_approve",
        action="store_true",
        default=False,
        help=(
            "Log the audition session but skip approval entirely.  "
            "Use this to review clips before approving via "
            "'--mode series approve --audition-id …'."
        ),
    )

    # ── YouTube packaging flags (--mode package / --package add-on) ──────
    pkg_group = parser.add_argument_group(
        "YouTube packaging (--mode package  /  --package add-on)"
    )
    pkg_group.add_argument(
        "--package",
        dest="package",
        action="store_true",
        default=False,
        help=(
            "After rendering, also generate SRT captions and YouTube show "
            "notes.  Works as an add-on with --mode full and --mode video.  "
            "Equivalent to running --mode package on the output JSON.  "
            "Show notes require DEEPSEEK_API_KEY; captions are always free."
        ),
    )
    pkg_group.add_argument(
        "--package-output",
        dest="package_output",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory for packaged output files (.srt, _show_notes.txt).  "
            "Defaults to the same directory as the show JSON."
        ),
    )
    pkg_group.add_argument(
        "--no-package-notes",
        dest="no_package_notes",
        action="store_true",
        default=False,
        help=(
            "Skip the LLM show-notes generation step; produce SRT captions "
            "only.  Useful when DEEPSEEK_API_KEY is unavailable or you just "
            "need the subtitle file."
        ),
    )

    # ── Analysis flags (--mode analysis) ─────────────────────────────
    analysis_group = parser.add_argument_group(
        "rhetorical analysis (--mode analysis)"
    )
    analysis_group.add_argument(
        "--analysis-file-b",
        dest="analysis_file_b",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Second speaker text file.  When supplied, the positional "
            "prompt arg is treated as speaker A's text file (text-pair "
            "mode).  Ignored when --script-input points to a debate JSON."
        ),
    )
    analysis_group.add_argument(
        "--speaker-a",
        dest="speaker_a",
        type=str,
        default=None,
        metavar="NAME",
        help="Speaker A name or substring (case-insensitive).",
    )
    analysis_group.add_argument(
        "--speaker-b",
        dest="speaker_b",
        type=str,
        default=None,
        metavar="NAME",
        help="Speaker B name or substring (case-insensitive).",
    )
    analysis_group.add_argument(
        "--analysis-output",
        dest="analysis_output",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory for the HTML analysis report "
            "(default: output/text/analysis/)."
        ),
    )
    analysis_group.add_argument(
        "--n-topics",
        dest="analysis_n_topics",
        type=int,
        default=10,
        metavar="N",
        help="Number of LDA topics (default: 10).",
    )
    analysis_group.add_argument(
        "--permutations",
        dest="analysis_permutations",
        type=int,
        default=1000,
        metavar="N",
        help="Permutation test count (default: 1000).",
    )
    analysis_group.add_argument(
        "--no-llm",
        dest="analysis_no_llm",
        action="store_true",
        default=False,
        help="Skip DeepSeek LLM stance extraction.",
    )
    analysis_group.add_argument(
        "--no-html",
        dest="analysis_no_html",
        action="store_true",
        default=False,
        help="Print text summary to stdout only; do not save HTML.",
    )
    analysis_group.add_argument(
        "--min-words",
        dest="analysis_min_words",
        type=int,
        default=3,
        metavar="N",
        help="Minimum words per utterance (default: 3).",
    )

    # ── Emotion-flow flags (--mode emotion-flow) ──────────────────────
    ef_group = parser.add_argument_group(
        "emotion flow analysis (--mode emotion-flow)"
    )
    ef_group.add_argument(
        "--leader",
        dest="ef_leader",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Speaker/leader name filter for emotion-flow analysis.  "
            "Required for multi-speaker JSON.  Inferred from filename "
            "when omitted."
        ),
    )
    ef_group.add_argument(
        "--emotion-output",
        dest="emotion_output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory for emotion-flow reports (default: output/emotion_flow/).",
    )
    ef_group.add_argument(
        "--no-translate",
        dest="ef_no_translate",
        action="store_true",
        default=False,
        help="Skip translation step (treat corpus as already English).",
    )
    ef_group.add_argument(
        "--no-semantic",
        dest="ef_no_semantic",
        action="store_true",
        default=False,
        help="Skip semantic-flow correlation (no sentence-transformers needed).",
    )
    ef_group.add_argument(
        "--ef-permutations",
        dest="ef_permutations",
        type=int,
        default=1000,
        metavar="N",
        help="Permutation draws for emotion-flow null-hypothesis test (default: 1000).",
    )
    ef_group.add_argument(
        "--ef-seed",
        dest="ef_seed",
        type=int,
        default=42,
        metavar="N",
        help="RNG seed for emotion-flow analysis (default: 42).",
    )
    ef_group.add_argument(
        "--ef-cache-dir",
        dest="ef_cache_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="HuggingFace model cache directory override for emotion-flow.",
    )
    ef_group.add_argument(
        "--to-show",
        dest="ef_to_show",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a report.json produced by emotion-flow analysis.  "
            "Switches into show-adapter mode: converts the report to a "
            "NewsConfig (and optionally runs the show pipeline)."
        ),
    )
    ef_group.add_argument(
        "--run-show",
        dest="ef_run_show",
        action="store_true",
        default=False,
        help=(
            "When --to-show is set, also run the show pipeline to generate "
            "show.json (requires DEEPSEEK_API_KEY)."
        ),
    )
    ef_group.add_argument(
        "--n-segments",
        dest="ef_n_segments",
        type=int,
        default=None,
        metavar="N",
        help="Segment count for the adapted emotion-flow show (default: 6).",
    )

    # ── Language / localisation flags ────────────────────────────────
    lang_group = parser.add_argument_group(
        "language / localisation"
    )
    lang_group.add_argument(
        "--root-lang",
        dest="root_lang",
        type=str,
        default="en",
        metavar="LANG",
        help=(
            "Language the show is generated in.  Defaults to 'en'.  "
            "The LLM produces all dialogue in this language.  "
            "Use a BCP-47 code (en, es, fr, de, pt, …)."
        ),
    )
    lang_group.add_argument(
        "--out-lang",
        dest="out_lang",
        type=str,
        default=None,
        metavar="LANG[,LANG…]|all",
        help=(
            "Comma-separated list of output languages, or 'all' for every "
            "language that has TTS voice coverage installed.  "
            "Example: --out-lang en,es   or   --out-lang all.  "
            "When omitted, only the root language is produced."
        ),
    )
    lang_group.add_argument(
        "--keep-translated",
        dest="keep_translated",
        action="store_true",
        default=False,
        help=(
            "Keep intermediate translated JSON files on disk after "
            "rendering.  By default these are deleted once their audio "
            "and published outputs have been written."
        ),
    )
    lang_group.add_argument(
        "--translation-model",
        dest="translation_model",
        type=str,
        default=None,
        metavar="MODEL",
        help=(
            "Override the LLM model used for translation passes.  "
            "Defaults to the same model the generator used."
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
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help=(
            "Show diagnostic detail: per-beat length warnings, "
            "fact-check retry messages, and budget status. Off by default."
        ),
    )

    # ── LLM lens rack ────────────────────────────────────────────────
    llm_group = parser.add_argument_group("LLM lens rack")
    llm_group.add_argument(
        "--llm-backend",
        dest="llm_backend",
        type=str,
        default="auto",
        choices=["cloud", "local", "auto"],
        help=(
            "Default LLM backend.  'cloud' forces cloud-deepseek; "
            "'local' forces local-r1-distill (requires Ollama); "
            "'auto' (default) honours the llm: block in vox_romana.yaml."
        ),
    )
    llm_group.add_argument(
        "--llm-lens",
        dest="llm_lens",
        action="append",
        default=None,
        metavar="ROLE=LENS",
        help=(
            "Per-role lens override, repeatable.  E.g. "
            "--llm-lens translator=local-qwen-polyglot "
            "--llm-lens 'character:Salvatore=local-llama-dim'.  "
            "Roles include: director, conductor, editor, fact_checker, "
            "translator, character:<Name>, default."
        ),
    )
    llm_group.add_argument(
        "--character-lang",
        dest="character_lang",
        action="append",
        default=None,
        metavar="NAME=CODE",
        help=(
            "Per-character language override, repeatable.  E.g. "
            "--character-lang 'Marcus=zh' --character-lang 'Giulia=it'. "
            "NAME is the speaker display name; CODE is a BCP-47 code (en, zh, es, fr, …)."
        ),
    )

    voice_prep_group = parser.add_argument_group(
        "voice-prep (--mode voice-prep)",
        "Clean a reference audio file and register it as a named voice-clone profile.",
    )
    voice_prep_group.add_argument(
        "--vp-source",
        dest="vp_source",
        type=Path,
        default=None,
        metavar="AUDIO_FILE",
        help=(
            "Path to the reference audio file (MP3, WAV, FLAC, …). "
            "Required for voice-prep mode."
        ),
    )
    voice_prep_group.add_argument(
        "--vp-name",
        dest="vp_name",
        type=str,
        default=None,
        metavar="VOICE_NAME",
        help=(
            "Short alphanumeric name for the voice profile (e.g. marcus_deep). "
            "Stored as vc_<name>. Required for voice-prep mode."
        ),
    )
    voice_prep_group.add_argument(
        "--vp-gender",
        dest="vp_gender",
        type=str,
        default="neutral",
        metavar="GENDER",
        help="Gender hint for the voice (male / female / neutral). Default: neutral.",
    )
    voice_prep_group.add_argument(
        "--vp-tone",
        dest="vp_tone",
        type=str,
        default="",
        metavar="DESCRIPTION",
        help='Free-text tone description (e.g. "deep, measured, resonant").',
    )
    voice_prep_group.add_argument(
        "--vp-best-for",
        dest="vp_best_for",
        type=str,
        default="",
        metavar="USE_CASE",
        help='Best-use hint (e.g. "narration", "dialogue", "lecture").',
    )
    voice_prep_group.add_argument(
        "--vp-tags",
        dest="vp_tags",
        nargs="+",
        default=None,
        metavar="TAG",
        help="Space-separated tags (e.g. --vp-tags deep male measured).",
    )
    voice_prep_group.add_argument(
        "--vp-force",
        dest="vp_force",
        action="store_true",
        default=False,
        help="Overwrite an existing profile with the same name.",
    )
    voice_prep_group.add_argument(
        "--skip-demucs",
        dest="skip_demucs",
        action="store_true",
        default=False,
        help="Skip the demucs vocal-separation stage (faster, but no stem isolation).",
    )

    voice_assign_group = parser.add_argument_group(
        "voice-assign (--mode voice-assign)",
        "Lock a cloned voice to a series character and record it in the catalog.",
    )
    voice_assign_group.add_argument(
        "--voice-id",
        dest="voice_id",
        type=str,
        default=None,
        metavar="VC_ID",
        help='Cloned voice ID to assign (must start with "vc_").',
    )
    voice_assign_group.add_argument(
        "--va-character",
        dest="va_character",
        type=str,
        default=None,
        metavar="CHARACTER_NAME",
        help="Character display name to lock the voice to.",
    )
    voice_assign_group.add_argument(
        "--series-id",
        dest="series_id",
        type=str,
        default=None,
        metavar="SERIES_ID",
        help="Series ID (from --mode series create or catalog list).",
    )

    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Argv builder for render sub-script
# ─────────────────────────────────────────────────────────────────────────────
# render_show.py accepts argv: list[str] as its main() parameter.  This
# helper builds the right argv from our parsed args namespace, forwarding
# only the flags render_show knows about.  The generate path no longer uses
# argv serialization — it calls vox_romana.cli.runners.run_generate directly.
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
    elif getattr(args, "name", None):
        # --name sets the audio stem when no explicit --audio-output was given.
        # We derive the path from the JSON's parent directory.
        argv += ["--output", str(Path(json_path).parent / f"{args.name}.{args.format}")]
    if args.voice_map:
        argv += ["--voice-map", args.voice_map]
    if args.voice_policy != "auto":
        argv += ["--voice-policy", args.voice_policy]
    argv += ["--model-dir", args.model_dir]
    if args.model:
        argv += ["--model", args.model]
    if args.no_normalize:
        argv.append("--no-normalize")
    if args.no_log_audio_duration:
        argv.append("--no-log-audio-duration")
    if args.tts_workers != 1:
        argv += ["--tts-workers", str(args.tts_workers)]
    if args.no_strip_directions:
        argv.append("--no-strip-directions")
    if args.no_whisper_voices:
        argv.append("--no-whisper-voices")
    if not args.hqvoices:
        argv.append("--no-hqvoices")
    if args.music:
        argv.append("--music")
    if args.music_style:
        argv += ["--music-style", args.music_style]
    if args.music_model != "small":
        argv += ["--music-model", args.music_model]
    if args.music_duration != 8.0:
        argv += ["--music-duration", str(args.music_duration)]
    if args.music_volume != 0.0:
        argv += ["--music-volume", str(args.music_volume)]
    if args.quiet:
        argv.append("--quiet")
    if getattr(args, "verbose", False):
        argv.append("--verbose")

    return argv


# ─────────────────────────────────────────────────────────────────────────────
# Mode runners
# ─────────────────────────────────────────────────────────────────────────────

def _mode_generate(args: argparse.Namespace) -> Path:
    """Run generate mode: prompt → JSON.

    Delegates to ``vox_romana.cli.runners.run_generate`` directly,
    passing the already-parsed args namespace.  No argv serialization.

    Args:
        args: Parsed namespace from parse_args().

    Returns:
        Path to the generated JSON file.
    """
    from vox_romana.cli.runners import run_generate
    return run_generate(args)


def _mode_render(json_path: str | Path, args: argparse.Namespace):
    """Run render mode: JSON → audio.

    Delegates to ``vox_romana.cli.runners.run_render`` directly,
    passing the argv list built by ``_build_render_argv``.

    Args:
        json_path: Path to the show JSON to render.
        args:      Parsed namespace from parse_args().

    Returns:
        RenderResult with stats including per-event audio durations.
    if args.hqvoices:
        # Fetch missing high-quality voices first so the preflight below
        # only warns about voices that are still missing afterwards.
        from vox_romana.tts.hq_voices import ensure_hq_voices
        ensure_hq_voices(
            json_path,
            engine=args.engine,
            model_dir=args.model_dir,
            quiet=getattr(args, "quiet", False),
        )
    """
    from vox_romana.tts.voice_preflight import check_piper_voices
    check_piper_voices(
        json_path,
        model_dir=args.model_dir,
        engine=args.engine,
        quiet=getattr(args, "quiet", False),
    )

    from vox_romana.cli.runners import run_render

    argv = _build_render_argv(json_path, args)
    return run_render(argv)


def _mode_render_isolated(json_path: str | Path, args: argparse.Namespace):
    """Run render mode in a child process and return lightweight metadata.

    This is used by ``--mode video`` so the TTS / MusicGen process can exit
    before animation begins. Some ML runtimes (MLX, torch, ONNX Runtime)
    keep large caches resident even after Python references are dropped; a
    process boundary is the only reliable way to return that memory to the OS.

    Returns:
        A namespace exposing ``event_durations`` for animation timing.
    """
    argv = _build_render_argv(json_path, args)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as meta_file:
        meta_path = Path(meta_file.name)

    script = (
        "from pathlib import Path; "
        "import json, sys; "
        "from vox_romana.cli.runners import run_render; "
        "result = run_render(sys.argv[1:-1]); "
        "payload = {'event_durations': result.event_durations if result is not None else None}; "
        "Path(sys.argv[-1]).write_text(json.dumps(payload), encoding='utf-8')"
    )

    audio_suffix = f".{args.format}"
    audio_path = Path(args.audio_output) if args.audio_output else Path(json_path).with_suffix(audio_suffix)

    try:
        completed = subprocess.run([sys.executable, "-c", script, *argv, str(meta_path)])

        payload = None
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))

        late_render_crash = (
            completed.returncode < 0
            and payload is not None
            and audio_path.exists()
        )
        if completed.returncode != 0 and not late_render_crash:
            sys.exit(completed.returncode)
        if late_render_crash:
            print(
                f"Warning: render subprocess exited with signal {-completed.returncode} "
                "after writing audio; continuing into animation.",
                file=sys.stderr,
            )

        if payload is None:
            raise RuntimeError("render subprocess produced no timing metadata")

        event_durations = payload.get("event_durations")
        if isinstance(event_durations, dict):
            event_durations = {
                int(event_index): duration
                for event_index, duration in event_durations.items()
            }
        return SimpleNamespace(event_durations=event_durations)
    finally:
        meta_path.unlink(missing_ok=True)


def _mode_video(args: argparse.Namespace) -> None:
    """Run video mode: generate → render → animate.

    Three-phase pipeline:
      1. Generate the show JSON (same as --mode generate).
      2. Render the audio (same as --mode render).
      3. Run the animation pipeline: hardware scan → bootstrap →
         director (1 LLM call) → video render.

    The animation phase is driven by ``vox_romana.cli.animation_runner``
    so the heavy animation imports (torch, diffusers, moviepy) are never
    loaded for non-video modes.

    Platform dispatch is controlled by ``args.video_platform``:
      - ``youtube`` (default) — 1920×1080 landscape MP4.
      - ``tiktok``            — 1080×1920 portrait MP4.
      - ``both``              — landscape + portrait in one SDXL pass;
                                returns the YouTube path as the primary result.
      - ``fast``              — 512×512, SD 1.5, 4 inference steps, 15 fps;
                                intended for quick dev/diagnostic runs.

    Output naming:
      - ``args.name`` sets a fixed filename stem (no timestamp).
        With ``both``, filenames become ``{name}_youtube.mp4`` and
        ``{name}_tiktok.mp4``.  Without ``--name`` the defaults are
        ``video_{platform}.mp4`` (or ``video_fast.mp4``).
      - ``args.video_output`` can be a directory (default:
        ``{audio_parent}/video/``) or a full file path for single-platform
        runs.

    Args:
        args: Parsed namespace from parse_args().  Key attributes read:
              ``args.video_platform``, ``args.video_output``, ``args.name``,
              ``args.visual_brief``, ``args.quiet``.
    """
    from vox_romana.cli.animation_runner import run_animation

    pipeline_start = time.time()

    if not args.quiet:
        print(f"\n{'─' * 60}")
        print("  Step 1/3: Generating script…")
        print(f"{'─' * 60}\n")

    t0 = time.time()
    json_path = _mode_generate(args)
    if not args.quiet:
        print(f"\n  Done in {time.time() - t0:.0f}s")

    if not args.quiet:
        print(f"\n{'─' * 60}")
        print("  Step 2/3: Rendering audio…")
        print(f"{'─' * 60}\n")

    t0 = time.time()
    render_result = _mode_render_isolated(json_path, args)
    if not args.quiet:
        print(f"\n  Done in {time.time() - t0:.0f}s")

    # Belt-and-suspenders GC after render: the renderer already called
    # teardown() on all TTS engines, but an explicit collect() here ensures
    # any remaining Python-managed model references are freed before the
    # resource monitor's pre-animate RAM check.
    import gc as _gc
    _gc.collect()

    audio_suffix = f".{args.format}"
    audio_path = Path(args.audio_output) if args.audio_output else json_path.with_suffix(audio_suffix)

    # Mid-pipeline resource checkpoint: audio render consumed disk/RAM;
    # re-check before the much heavier animation phase starts.
    monitor = getattr(args, "_resource_monitor", None)
    if monitor is not None:
        from vox_romana.core.resource_monitor import ResourceError
        try:
            monitor.checkpoint("pre-animate")
        except ResourceError as _re:
            print(f"\nError: {_re}", file=sys.stderr)
            monitor.stop()
            sys.exit(1)

    if not args.quiet:
        platform = getattr(args, "video_platform", "youtube")
        print(f"\n{'─' * 60}")
        print(f"  Step 3/3: Animating [{platform}]…")
        print(f"{'─' * 60}\n")

    t0 = time.time()
    video_path = run_animation(
        args,
        json_path,
        audio_path,
        event_durations=render_result.event_durations if render_result is not None else None,
    )
    if not args.quiet:
        print(f"\n  Done in {time.time() - t0:.0f}s")

    # ── Optional: YouTube packaging (--package add-on) ──────────────
    if getattr(args, "package", False):
        if not args.quiet:
            print(f"\n{'─' * 60}")
            print("  Packaging for YouTube…")
            print(f"{'─' * 60}")
        _mode_package(
            json_path,
            args,
            event_durations=render_result.event_durations if render_result is not None else None,
        )

    if not args.quiet:
        total_elapsed = time.time() - pipeline_start
        print(f"\n{'═' * 60}")
        print(f"  Pipeline complete!  (total: {total_elapsed:.0f}s)")
        print(f"  Script: {json_path}")
        print(f"  Audio:  {audio_path}")
        if video_path:
            print(f"  Video:  {video_path}")
        print(f"{'═' * 60}\n")
    elif video_path:
        print(video_path)


def _mode_dedupe(args: argparse.Namespace) -> None:
    """Scan corpus for near-duplicate documents and report (or remove) them.

    Builds a FingerprintStore from all JSONL files in the corpus directory,
    then calls ``nearest_pairs()`` to find documents with cosine similarity
    above the threshold.  By default the scan is read-only.  Pass ``--apply``
    to remove the lower-chunk-count file in each pair (with confirmation).
    """
    import sys
    from vox_romana.memory.fingerprint_store import FingerprintStore
    from vox_romana.memory.ingest import _dedup_config
    import vox_romana.memory.ingest as _ingest_mod

    # Resolve corpus directory.
    corpus_dir = Path(args.dedupe_path) if args.dedupe_path else _ingest_mod.DEFAULT_DOCUMENT_DIR
    if not corpus_dir.exists():
        print(f"Error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    cfg_threshold, _ = _dedup_config()
    threshold = args.dedupe_threshold if args.dedupe_threshold is not None else cfg_threshold

    fp_path = corpus_dir / "fingerprints.npz"
    store = FingerprintStore(store_path=fp_path)

    if not args.quiet:
        print(f"\n{'─' * 70}")
        print(f"  Deduplication scan")
        print(f"  Corpus:    {corpus_dir}")
        print(f"  Threshold: {threshold}")
        print(f"{'─' * 70}")
        print("  Building fingerprints (this may take a moment)…", flush=True)

    n = store.build_parallel(corpus_dir, verbose=not args.quiet)
    pairs = store.nearest_pairs(threshold=threshold)

    if not args.quiet:
        print(f"  Fingerprinted {n} documents.")
        if not pairs:
            print("  No near-duplicate pairs found.")
            print(f"{'─' * 70}\n")
            return
        print(f"\n  Found {len(pairs)} near-duplicate pair(s):\n")
        col = 38
        print(f"  {'Source A':<{col}}  {'Source B':<{col}}  {'Sim':>6}")
        print(f"  {'-' * col}  {'-' * col}  {'-' * 6}")
        for src_a, src_b, sim in pairs:
            a = Path(src_a).name
            b = Path(src_b).name
            if len(a) > col:
                a = "…" + a[-(col - 1):]
            if len(b) > col:
                b = "…" + b[-(col - 1):]
            print(f"  {a:<{col}}  {b:<{col}}  {sim:.4f}")
        print()

    if not pairs:
        return

    if not args.dedupe_apply:
        if not args.quiet:
            print("  (Read-only scan — pass --apply to remove duplicates.)")
            print(f"{'─' * 70}\n")
        return

    # ── --apply: remove lower-chunk-count file per pair ──────────────
    # Count chunks per JSONL.
    def _count_chunks(jsonl_path: Path) -> int:
        try:
            return sum(1 for _ in jsonl_path.open())
        except Exception:
            return 0

    # Build a map: source string → Path of the JSONL that contains it.
    # The FingerprintStore sources are file paths (absolute or relative).
    to_remove: set[Path] = set()
    for src_a, src_b, _sim in pairs:
        p_a = Path(src_a)
        p_b = Path(src_b)
        if not p_a.exists() or not p_b.exists():
            continue
        keep = p_a if _count_chunks(p_a) >= _count_chunks(p_b) else p_b
        victim = p_b if keep == p_a else p_a
        to_remove.add(victim)

    if not to_remove:
        print("  Nothing to remove (files not found on disk).")
        print(f"{'─' * 70}\n")
        return

    if not args.quiet:
        print(f"  Will remove {len(to_remove)} file(s):")
        for p in sorted(to_remove):
            print(f"    {p.name}")
        print()

    confirm = input("  Confirm removal? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        print(f"{'─' * 70}\n")
        return

    removed = 0
    for p in sorted(to_remove):
        try:
            p.unlink()
            store.remove(str(p))
            removed += 1
            if not args.quiet:
                print(f"  Removed: {p.name}")
        except Exception as exc:
            print(f"  ERROR removing {p.name}: {exc}", file=sys.stderr)

    if not args.quiet:
        print(f"\n  Done: {removed} file(s) removed.")
        print(f"{'─' * 70}\n")


def _mode_ingest(args: argparse.Namespace) -> None:
    """Run ingest mode: feed documents into the knowledge store.

    Walks each --ingest-path, calls DocumentIngester, then prints a
    per-file summary table and a grand total.

    Args:
        args: Parsed namespace from parse_args().
    """
    from vox_romana.memory.ingest import (
        DocumentIngester,
        IngestStats,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    chunk_size = args.ingest_chunk_size or CHUNK_SIZE
    chunk_overlap = args.ingest_chunk_overlap or CHUNK_OVERLAP

    ingester = DocumentIngester(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_stats = []
    for raw_path in args.ingest_path:
        p = Path(raw_path)
        if p.is_dir():
            stats_list = ingester.ingest_directory(
                p,
                recursive=args.recursive,
                dry_run=args.dry_run,
                force_reingest=args.force_reingest,
                tag=args.ingest_tag,
            )
            all_stats.extend(stats_list)
        elif p.is_file():
            try:
                stats = ingester.ingest(
                    p,
                    dry_run=args.dry_run,
                    force_reingest=args.force_reingest,
                    tag=args.ingest_tag,
                )
                all_stats.append(stats)
            except Exception as exc:
                import sys
                print(f"  ERROR: {p}: {exc}", file=sys.stderr)
                all_stats.append(
                    IngestStats(
                        source=str(p),
                        format=p.suffix.lower(),
                        skipped=True,
                        skip_reason=f"error: {exc}",
                        failed=True,
                    )
                )
        else:
            import sys
            print(f"  WARNING: path not found: {raw_path}", file=sys.stderr)

    if not args.quiet:
        # Print per-file summary table.
        label = "[DRY RUN] " if args.dry_run else ""
        print(f"\n{'─' * 70}")
        print(f"  {label}Ingest summary")
        print(f"{'─' * 70}")
        col_w = 50
        print(f"  {'Source':<{col_w}}  {'Fmt':<7}  {'Chunks':>6}  {'Status'}")
        print(f"  {'-' * col_w}  {'-' * 7}  {'-' * 6}  {'-' * 10}")
        for s in all_stats:
            src = Path(s.source).name
            if len(src) > col_w:
                src = "..." + src[-(col_w - 3):]
            if s.skipped:
                status = f"skipped ({s.skip_reason})"
            else:
                status = "ok"
            print(
                f"  {src:<{col_w}}  {s.format or '?':<7}  {s.chunks:>6}  {status}"
            )
        ingested = [s for s in all_stats if not s.skipped]
        skipped = [s for s in all_stats if s.skipped]
        failed = [s for s in all_stats if s.failed]
        total_chunks = sum(s.chunks for s in ingested)
        total_chars = sum(s.chars for s in ingested)
        print(f"{'─' * 70}")
        print(
            f"  Total: {len(ingested)} ingested, {len(skipped)} skipped "
            f"({len(failed)} failed) — {total_chunks} chunks, {total_chars:,} chars"
        )
        if failed:
            print(f"  {len(failed)} file(s) failed to ingest:")
            for s in failed:
                print(f"    - {Path(s.source).name}: {s.skip_reason}")
        print(f"{'─' * 70}\n")

    # ── Phase 2: Update vector index after ingest ─────────────────────
    if not args.dry_run and not args.no_embed:
        try:
            from vox_romana.memory.vector_index import ChromaVectorIndex
            import vox_romana.memory.ingest as _ingest_mod
            if ChromaVectorIndex.is_available():
                index = ChromaVectorIndex()
                migration_stats = index.migrate_documents(
                    _ingest_mod.DEFAULT_DOCUMENT_DIR,
                    verbose=not args.quiet,
                )
                if not args.quiet and migration_stats["files"] > 0:
                    print(
                        f"  Vector index: +{migration_stats['loaded']} chunk(s) "
                        f"embedded from {migration_stats['files']} new file(s)\n"
                    )
        except Exception as exc:
            logger.warning("Failed to update vector index: %s", exc)


def _mode_music(args: argparse.Namespace) -> Path:
    """Run music mode: text prompt → standalone audio file.

    Accepts a prompt from ``args.music_prompt`` (inline text) or
    ``args.prompt`` (plain-text file).  Generates audio via
    :class:`MusicGenEngine`, converts to the requested format, and
    writes the result to the first matching path below:

      1. ``args.audio_output``  — explicit ``--audio-output`` path.
      2. ``output/music/{args.name}.{format}``  — when ``--name`` is set.
      3. ``output/music/{slug}_{timestamp}.{format}``  — auto-generated.

    Args:
        args: Parsed namespace from parse_args().  Reads
              ``args.music_prompt``, ``args.prompt``, ``args.music_model``,
              ``args.music_duration``, ``args.format``, ``args.audio_output``,
              ``args.name``, ``args.quiet``.

    Returns:
        Path to the generated audio file.
    """
    import re
    import datetime
    from io import BytesIO

    from vox_romana.tts.music_engine import MusicGenEngine, MODEL_ALIASES
    from vox_romana.core.paths import HF_CACHE_DIR

    # ── Resolve prompt text ───────────────────────────────────────────
    if args.music_prompt:
        prompt_text = args.music_prompt.strip()
    else:
        prompt_text = args.prompt.read_text(encoding="utf-8").strip()

    if not prompt_text:
        print("Error: music prompt is empty.", file=sys.stderr)
        sys.exit(1)

    # ── Resolve model name ────────────────────────────────────────────
    model_name = MODEL_ALIASES.get(args.music_model, args.music_model)

    # ── Generate ──────────────────────────────────────────────────────
    if not args.quiet:
        print(f"  Model:    {model_name}")
        print(f"  Duration: {args.music_duration}s")
        print(f"  Prompt:   {prompt_text[:80]}{'…' if len(prompt_text) > 80 else ''}")
        print()

    engine = MusicGenEngine(model_name=model_name, cache_dir=HF_CACHE_DIR)
    result = engine.generate(prompt_text, duration=args.music_duration)

    # ── Determine output path ─────────────────────────────────────────
    if args.audio_output:
        out_path = Path(args.audio_output)
    elif getattr(args, "name", None):
        # --name overrides the auto-generated slug+timestamp stem.
        out_dir = _OUTPUT_DIR / "music"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.name}.{args.format}"
    else:
        # Auto-name: output/music/<slug>_<timestamp>.<format>
        slug = re.sub(r"[^a-z0-9]+", "_", prompt_text[:40].lower()).strip("_")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _OUTPUT_DIR / "music"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug}_{ts}.{args.format}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Write audio (convert format if needed) ────────────────────────
    if args.format == "wav":
        out_path.write_bytes(result.audio)
    else:
        from pydub import AudioSegment
        segment = AudioSegment.from_wav(BytesIO(result.audio))
        segment.export(str(out_path), format=args.format)

    if not args.quiet:
        print(f"  Audio → {out_path}")
    else:
        print(out_path)

    return out_path


def _mode_analysis(args: argparse.Namespace) -> None:
    """Delegate to vox_romana.cli.analysis.run_analysis()."""
    from vox_romana.cli.analysis import run_analysis
    run_analysis(args)


def _mode_emotion_flow(args: argparse.Namespace) -> None:
    """Delegate to vox_romana.cli.emotion_flow.run_emotion_flow()."""
    from vox_romana.cli.emotion_flow import run_emotion_flow
    run_emotion_flow(args)


def _mode_audition(args: argparse.Namespace) -> Path | None:
    """Delegate to vox_romana.cli.audition_runner.run_audition().

    Audition mode generates candidate assets (voices / music / visuals)
    and optionally approves them into a series asset pool.  Lives in a
    separate module so heavy TTS/MusicGen imports stay deferred.
    """
    from vox_romana.cli.audition_runner import run_audition
    return run_audition(args)


def _mode_package(
    json_path: str | Path,
    args: argparse.Namespace,
    event_durations: dict | None = None,
) -> None:
    """Generate SRT captions and YouTube show notes for a rendered episode.

    Reads the show JSON at ``json_path``, calls ``package_episode()`` from
    ``vox_romana.package``, and prints the paths of files written.

    Args:
        json_path:       Path to the show JSON.
        args:            Parsed namespace.  Reads ``args.package_output``,
                         ``args.no_package_notes``, ``args.quiet``.
        event_durations: Per-event audio durations from a ``RenderResult``.
                         Pass ``None`` to use fallback timing (~3 s/event).
    """
    from vox_romana.package import package_episode

    generate_notes = not getattr(args, "no_package_notes", False)
    client = None
    if generate_notes:
        try:
            from vox_romana.core.client import DeepSeekClient
            client = DeepSeekClient()
        except ValueError:
            print(
                "  Warning: DEEPSEEK_API_KEY not set — show notes skipped. "
                "Set the key or pass --no-package-notes to suppress this warning.",
                file=sys.stderr,
            )
            generate_notes = False

    dest_dir = getattr(args, "package_output", None)

    result = package_episode(
        json_path=Path(json_path),
        event_durations=event_durations,
        dest_dir=dest_dir,
        generate_notes=generate_notes,
        client=client,
    )

    if not args.quiet:
        print()
        if result.srt_path:
            print(f"  Captions  → {result.srt_path}")
        if result.notes_path:
            print(f"  Notes     → {result.notes_path}")


def _mode_publish(
    json_path: str | Path,
    args: argparse.Namespace,
) -> None:
    """Run publish mode: show JSON → formatted TXT and/or PDF.

    Uses the ``vox_romana.publish`` package.  PDF output is silently
    skipped (with a warning) when WeasyPrint is not installed.

    Args:
        json_path: Path to the show JSON to publish.
        args:      Parsed namespace from parse_args().
    """
    from vox_romana.publish import publish_document, PDF_AVAILABLE, JINJA_AVAILABLE

    fmt = args.publish_format  # "txt", "pdf", "html", or "both"
    if fmt == "both":
        formats: tuple[str, ...] = ("txt", "pdf")
    else:
        formats = (fmt,)

    dest_dir = args.publish_output  # Path | None

    result = publish_document(
        src=json_path,
        formats=formats,
        dest_dir=dest_dir,
        root_lang=getattr(args, "root_lang", "en"),
    )

    for warning in result.warnings:
        print(f"  ⚠  {warning}")

    if not args.quiet:
        print(f"\n  Published [{result.show_type}] — {result.sections} section(s)")
        if result.txt_path:
            print(f"  TXT → {result.txt_path}")
        if result.pdf_path:
            print(f"  PDF → {result.pdf_path}")
        if result.html_path:
            print(f"  HTML → {result.html_path}")
        if "pdf" in formats and not PDF_AVAILABLE:
            print(
                "  (PDF requires: pip install weasyprint jinja2 "
                "+ brew install pango)"
            )
        if "html" in formats and not JINJA_AVAILABLE:
            print("  (HTML requires: pip install jinja2)")


# ─────────────────────────────────────────────────────────────────────────────
# Translation + multi-language fan-out
# ─────────────────────────────────────────────────────────────────────────────

def _mode_voice_prep(args: argparse.Namespace) -> None:
    """Run voice-prep mode: clean reference audio + register as voice profile.

    Delegates to :func:`vox_romana.cli.voice_prep_cli.run_voice_prep`.

    Args:
        args: Parsed namespace from parse_args().
    """
    from vox_romana.cli.voice_prep_cli import run_voice_prep
    run_voice_prep(args)


def _mode_voice_assign(args: argparse.Namespace) -> None:
    """Run voice-assign mode: lock cloned voice to a series character.

    Delegates to :func:`vox_romana.cli.voice_assign_cli.run_voice_assign`.

    Args:
        args: Parsed namespace from parse_args().
    """
    from vox_romana.cli.voice_assign_cli import run_voice_assign
    run_voice_assign(args)


def _mode_translate(
    src_json: Path,
    target_lang: str,
    args: argparse.Namespace,
) -> Path:
    """Translate an existing show JSON into ``target_lang``.

    Reads ``src_json``, runs every spoken event + structural label through
    :class:`LLMTranslator` via :func:`translate_show_data`, then writes
    the translated dict to ``<src>_<target_lang>.json`` next to the
    source file.  The new JSON's ``language`` field is set to
    ``target_lang`` so downstream render and publish steps know which
    voices and labels to use.

    Args:
        src_json:    Path to the root-language JSON.
        target_lang: BCP-47 language code (e.g. ``"es"``).
        args:        Parsed namespace from :func:`parse_args`.  Reads
                     ``args.root_lang``, ``args.translation_model``,
                     ``args.quiet``.

    Returns:
        Path to the translated JSON file.
    """
    import json

    from vox_romana.core.client import DeepSeekClient
    from vox_romana.translate import LLMTranslator, translate_show_data

    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = DeepSeekClient()
    translator = LLMTranslator(
        client=client,
        model=args.translation_model,
    )

    if not args.quiet:
        print(
            f"  Translating {src_json.name}: "
            f"{args.root_lang} → {target_lang}…"
        )

    translated = translate_show_data(
        data,
        target_lang=target_lang,
        translator=translator,
        source_lang=args.root_lang,
    )

    out_path = src_json.with_name(f"{src_json.stem}_{target_lang}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(f"  Translated → {out_path}")

    return out_path


def _suffix_audio_output(
    audio_output: Path | None,
    language: str,
) -> Path | None:
    """Return ``audio_output`` with ``_{language}`` inserted before the suffix.

    No-op when ``audio_output`` is ``None``.  Used so that an explicit
    ``--audio-output debate.mp3`` becomes ``debate_es.mp3`` for the
    Spanish pass of a multi-language run, instead of being overwritten.
    """
    if audio_output is None:
        return None
    return audio_output.with_name(f"{audio_output.stem}_{language}{audio_output.suffix}")


def _run_language_matrix(
    root_json: Path,
    args: argparse.Namespace,
    *,
    do_render: bool,
    do_publish: bool,
) -> dict[str, Path]:
    """Render and/or publish ``root_json`` once per language.

    For the root language the original JSON is used directly.  For each
    non-root language we first call :func:`_mode_translate` to produce a
    suffixed JSON, then run render and/or publish against that file.
    Translated JSON files are deleted afterwards unless
    ``args.keep_translated`` is set.

    Args:
        root_json:  The freshly generated root-language JSON path.
        args:       Parsed namespace.  Reads ``args.out_lang_list``,
                    ``args.root_lang``, ``args.audio_output``,
                    ``args.keep_translated``, ``args.quiet``.
        do_render:  When True, run :func:`_mode_render` for each language.
        do_publish: When True, run :func:`_mode_publish` for each language.

    Returns:
        Mapping of language code → JSON path used for that language's
        outputs.  The root language entry is ``root_json`` itself; other
        entries are the translated files (which may have been deleted by
        the time the caller inspects them).
    """
    paths: dict[str, Path] = {}
    original_audio_output = args.audio_output

    for lang in args.out_lang_list:
        is_root = (lang == args.root_lang)

        if is_root:
            target_json = root_json
        else:
            if not args.quiet:
                print(f"\n{'─' * 60}")
                print(f"  Translating to {lang}…")
                print(f"{'─' * 60}\n")
            target_json = _mode_translate(root_json, lang, args)

        paths[lang] = target_json

        # ── Render ───────────────────────────────────────────────────
        if do_render:
            if not args.quiet:
                print(f"\n{'─' * 60}")
                print(f"  Rendering audio [{lang}]…")
                print(f"{'─' * 60}\n")
            # Ensure explicit --audio-output doesn't overwrite across langs.
            if not is_root:
                args.audio_output = _suffix_audio_output(
                    original_audio_output, lang
                )
            else:
                args.audio_output = original_audio_output
            try:
                _mode_render(target_json, args)
            finally:
                args.audio_output = original_audio_output

        # ── Publish ──────────────────────────────────────────────────
        if do_publish:
            if not args.quiet:
                print(f"\n{'─' * 60}")
                print(f"  Publishing script [{lang}]…")
                print(f"{'─' * 60}\n")
            # The translated JSON's stem already carries the _{lang}
            # suffix; tell publish_document so it doesn't double-suffix.
            saved_root_lang = args.root_lang
            try:
                if not is_root:
                    args.root_lang = lang
                _mode_publish(target_json, args)
            finally:
                args.root_lang = saved_root_lang

        # ── Cleanup translated JSON ──────────────────────────────────
        if not is_root and not args.keep_translated:
            try:
                target_json.unlink()
                if not args.quiet:
                    print(f"  Removed temporary {target_json.name}")
            except OSError as exc:
                logger.warning(
                    "Could not delete translated JSON %s: %s",
                    target_json, exc,
                )

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _extract_university_argv(argv: list[str]) -> list[str] | None:
    """If ``--mode university`` (or ``-m university``) is in ``argv``, return
    everything after ``university`` as the argv to forward to the University
    CLI; otherwise return None.

    University mode uses a subcommand grammar (study/list/inspect/ask/chat/
    extend/export) that does not fit run.py's flat namespace, so we intercept
    early and hand the remainder off to ``vox_romana.university.cli.main``.
    """
    for i, tok in enumerate(argv):
        if tok in ("--mode", "-m") and i + 1 < len(argv) and argv[i + 1] == "university":
            return list(argv[i + 2:])
        if tok in ("--mode=university", "-m=university"):
            return list(argv[i + 1:])
    return None


def _extract_series_argv(argv: list[str]) -> list[str] | None:
    """If ``--mode series`` is in ``argv``, return the remainder for the
    Series CLI; otherwise return None.

    Series mode uses a subcommand grammar (create/list/inspect) that does
    not fit run.py's flat namespace.
    """
    for i, tok in enumerate(argv):
        if tok in ("--mode", "-m") and i + 1 < len(argv) and argv[i + 1] == "series":
            return list(argv[i + 2:])
        if tok in ("--mode=series", "-m=series"):
            return list(argv[i + 1:])
    return None


def main(argv: list[str] | None = None) -> None:
    """Run the vox_romana master pipeline.

    Dispatches to generate, render, or full mode based on --mode.
    In full mode, generates the script first, then renders it to audio.

    Args:
        argv: CLI arguments. Defaults to sys.argv[1:] if None.
              Exposed as a parameter so tests can call main() directly.
    """
    # University mode has its own subcommand grammar and is dispatched
    # before the master parser runs.
    raw_argv = list(sys.argv[1:]) if argv is None else list(argv)
    uni_argv = _extract_university_argv(raw_argv)
    if uni_argv is not None:
        from vox_romana.university.cli import main as university_main
        university_main(uni_argv)
        return

    # Series mode also has its own subcommand grammar (create/list/inspect).
    series_argv = _extract_series_argv(raw_argv)
    if series_argv is not None:
        from vox_romana.cli.series import main as series_main
        series_main(series_argv)
        return

    args = parse_args(argv)

    # ── Music mode: apply quality defaults before validation ──────────
    # The render-group flags use low defaults suited for show bumpers.
    # In music mode we upgrade them to the best quality that fits the
    # hardware (M1 Max 32 GB: large model ~3.3 GB, runs on MPS).
    if args.mode == "music":
        if args.music_model == "small":   # sentinel default → upgrade
            args.music_model = "large"
        if args.music_duration == 8.0:    # sentinel default → upgrade
            args.music_duration = 30.0

    # ── Validate mode-specific requirements ──────────────────────────
    if args.mode == "ingest" and not args.ingest_path:
        print(
            "Error: --ingest-path is required for 'ingest' mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "dedupe" and args.dedupe_threshold is not None:
        if not (0.0 < args.dedupe_threshold <= 1.0):
            print(
                "Error: --dedupe-threshold must be in (0.0, 1.0].",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.mode in ("generate", "full", "video") and args.prompt is None:
        print(
            "Error: prompt file is required for "
            f"'{args.mode}' mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "music":
        has_file_prompt = args.prompt is not None
        has_inline_prompt = bool(args.music_prompt)
        if not has_file_prompt and not has_inline_prompt:
            print(
                "Error: --mode music requires either a prompt file "
                "(positional argument) or --music-prompt TEXT.",
                file=sys.stderr,
            )
            sys.exit(1)
        if has_file_prompt and has_inline_prompt:
            print(
                "Error: provide a prompt file OR --music-prompt, not both.",
                file=sys.stderr,
            )
            sys.exit(1)
        if has_file_prompt and not args.prompt.exists():
            print(
                f"Error: prompt file not found: {args.prompt}",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.mode == "render" and args.script_input is None:
        print(
            "Error: --script-input is required for 'render' mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.publish and args.mode not in ("generate", "full"):
        print(
            "Error: --publish can only be used with --mode generate or --mode full "
            "(use --mode publish for standalone publishing).",
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

    if args.mode == "publish" and args.script_input is None:
        print(
            "Error: --script-input is required for 'publish' mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "publish" and args.script_input is not None:
        if not args.script_input.exists():
            print(
                f"Error: script file not found: {args.script_input}",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.mode == "analysis":
        # analysis takes either --script-input PATH.json or
        # positional FILE_A + --analysis-file-b FILE_B.
        if args.script_input is None and args.analysis_file_b is None:
            print(
                "Error: --mode analysis requires --script-input PATH.json "
                "or a positional FILE_A plus --analysis-file-b FILE_B.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.script_input is not None and not args.script_input.exists():
            print(
                f"Error: script file not found: {args.script_input}",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.analysis_file_b is not None:
            if args.prompt is None:
                print(
                    "Error: --analysis-file-b requires a positional FILE_A "
                    "as the first argument.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not args.prompt.exists():
                print(
                    f"Error: speaker A file not found: {args.prompt}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not args.analysis_file_b.exists():
                print(
                    f"Error: speaker B file not found: {args.analysis_file_b}",
                    file=sys.stderr,
                )
                sys.exit(1)

    if args.mode == "emotion-flow":
        # Requires either --script-input / positional prompt (analysis),
        # or --to-show PATH (show-adapter).  The two are mutually exclusive.
        if args.ef_to_show is not None:
            if not args.ef_to_show.exists():
                print(
                    f"Error: emotion-flow report not found: {args.ef_to_show}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if args.ef_run_show and args.script_input is not None:
                # --script-input is irrelevant in to-show mode; warn rather
                # than hard-error to stay friendly.
                pass
        else:
            source = args.script_input or args.prompt
            if source is None:
                print(
                    "Error: --mode emotion-flow requires --script-input PATH "
                    "(or a positional prompt file), or --to-show report.json.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not source.exists():
                print(
                    f"Error: source path not found: {source}",
                    file=sys.stderr,
                )
                sys.exit(1)

    # ── Validate and normalise language args ─────────────────────────
    # Inject voice_profiles_dir onto args for voice-prep / voice-assign modes.
    if args.mode in ("voice-prep", "voice-assign"):
        from vox_romana.core.paths import VOICE_PROFILES_DIR
        args.voice_profiles_dir = VOICE_PROFILES_DIR

    if args.mode == "voice-prep":
        if args.vp_source is None:
            print("Error: --vp-source is required for 'voice-prep' mode.", file=sys.stderr)
            sys.exit(1)
        if args.vp_name is None:
            print("Error: --vp-name is required for 'voice-prep' mode.", file=sys.stderr)
            sys.exit(1)

    if args.mode == "voice-assign":
        if args.voice_id is None:
            print("Error: --voice-id is required for 'voice-assign' mode.", file=sys.stderr)
            sys.exit(1)
        if args.va_character is None:
            print("Error: --va-character is required for 'voice-assign' mode.", file=sys.stderr)
            sys.exit(1)
        if args.series_id is None:
            print("Error: --series-id is required for 'voice-assign' mode.", file=sys.stderr)
            sys.exit(1)

    # Parse --character-lang NAME=CODE overrides into a dict
    from vox_romana.core.llm_config import parse_character_langs
    try:
        args.character_langs = parse_character_langs(getattr(args, "character_lang", None))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Normalise root_lang
    args.root_lang = args.root_lang.strip().lower()
    if args.root_lang not in LANGUAGE_CODES:
        valid_sample = ", ".join(sorted(LANGUAGE_CODES)[:8])
        print(
            f"Error: unknown language code '{args.root_lang}'.  "
            f"Valid codes include: {valid_sample}, …",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Resource monitor ─────────────────────────────────────────────
    # Start the background disk+RAM sampler before any model loading.
    # Modes that don't write large outputs (ingest, dedupe, analysis, …)
    # still benefit from at least knowing disk state at launch.
    if not getattr(args, "skip_resource_monitor", False):
        from vox_romana.core.resource_monitor import ResourceMonitor, ResourceError
        _resource_monitor = ResourceMonitor.for_mode(args.mode, args)
        _resource_monitor.start()
        # First checkpoint: verify there is enough headroom right now.
        # This catches the case where the disk is already nearly full before
        # we even start generating.  Later checkpoints fire mid-pipeline.
        try:
            snapshot = _resource_monitor.checkpoint("pre-flight")
            if not args.quiet:
                ram_str = (
                    f" | RAM avail: {snapshot.available_ram_gb:.1f} GB"
                    if snapshot.available_ram_gb is not None
                    else ""
                )
                print(
                    f"  Resources: {snapshot.free_disk_gb:.1f} GB disk free"
                    f" ({snapshot.disk_path}){ram_str}"
                )
        except ResourceError as _re:
            print(f"\nError: {_re}", file=sys.stderr)
            _resource_monitor.stop()
            sys.exit(1)
        # Store on args so _mode_video can call checkpoint() again
        # between pipeline steps.
        args._resource_monitor = _resource_monitor
    else:
        args._resource_monitor = None

    # Normalise out_lang → resolved list (includes root_lang if not already)
    if args.out_lang is None:
        # No --out-lang supplied → produce only the root language.
        args.out_lang_list: list[str] = [args.root_lang]
    else:
        try:
            parsed_langs = normalize_lang_arg(args.out_lang)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        # Always include root_lang so users don't accidentally lose the
        # primary output by specifying only non-root codes.
        if args.root_lang not in parsed_langs:
            parsed_langs = sorted({args.root_lang} | set(parsed_langs))
        args.out_lang_list = parsed_langs

    # ── Banner ───────────────────────────────────────────────────────
    if not args.quiet:
        print(f"\n{'═' * 60}")
        print(f"  VOX ROMANA — Master Pipeline")
        print(f"{'═' * 60}")
        print(f"  Mode: {args.mode}")
        if args.mode == "ingest" and args.ingest_path:
            for p in args.ingest_path:
                print(f"  Ingest path: {p}")
            if args.dry_run:
                print("  (dry run — no files will be written)")
        if args.mode == "music":
            if args.music_prompt:
                print(f"  Prompt:   {args.music_prompt[:80]}{'…' if len(args.music_prompt) > 80 else ''}")
            else:
                print(f"  Prompt file: {args.prompt}")
            print(f"  Model:    {args.music_model}")
            print(f"  Duration: {args.music_duration}s")
            print(f"  Format:   {args.format}")
        if args.prompt and args.mode != "music":
            print(f"  Prompt: {args.prompt}")
        if args.script_input:
            print(f"  Script input: {args.script_input}")
        if args.mode == "publish" or args.publish:
            fmt_label = {
                "txt": "TXT",
                "pdf": "PDF",
                "html": "HTML",
                "both": "TXT + PDF",
            }[args.publish_format]
            print(f"  Publish format: {fmt_label}")
            if args.publish_output:
                print(f"  Publish output: {args.publish_output}")
        if args.mode == "analysis":
            if args.script_input:
                print(f"  Source: {args.script_input}")
            elif args.prompt and args.analysis_file_b:
                print(f"  Speaker A: {args.prompt}")
                print(f"  Speaker B: {args.analysis_file_b}")
            print(f"  Topics: {args.analysis_n_topics}  Permutations: {args.analysis_permutations}")
            if args.analysis_no_llm:
                print("  LLM stance extraction: off")
            if args.analysis_no_html:
                print("  HTML output: off (text summary only)")
        if args.mode == "emotion-flow":
            if args.ef_to_show is not None:
                print(f"  Mode:         show-adapter")
                print(f"  Report:       {args.ef_to_show}")
                if args.ef_n_segments:
                    print(f"  Segments:     {args.ef_n_segments}")
                if args.ef_run_show:
                    print("  Run show:     yes")
            else:
                source = args.script_input or args.prompt
                print(f"  Source:       {source}")
                print(f"  Permutations: {args.ef_permutations}  Seed: {args.ef_seed}")
                if args.ef_no_translate:
                    print("  Translation:  OFF")
                if args.ef_no_semantic:
                    print("  Semantic:     OFF")
        # Language matrix (not shown for music/analysis/emotion-flow — language-agnostic)
        if args.mode not in ("music", "analysis", "emotion-flow"):
            if len(args.out_lang_list) == 1 and args.out_lang_list[0] == args.root_lang:
                print(f"  Language: {language_display([args.root_lang])}")
            else:
                print(f"  Root language: {language_display([args.root_lang])}")
                print(f"  Output languages: {language_display(args.out_lang_list)}")
        print()

    # ── Dispatch ─────────────────────────────────────────────────────

    if args.mode == "music":
        # Standalone music generation — prompt → audio file.
        out_path = _mode_music(args)
        if not args.quiet:
            print(f"\n{'═' * 60}")
            print(f"  Music generation complete!")
            print(f"  Audio → {out_path}")
            print(f"{'═' * 60}\n")

    elif args.mode == "analysis":
        # Standalone rhetorical analysis — two corpora → HTML report.
        _mode_analysis(args)
        if not args.quiet:
            print(f"\n{'═' * 60}")
            print(f"  Analysis complete!")
            print(f"{'═' * 60}\n")

    elif args.mode == "emotion-flow":
        # Cross-lingual emotion flow analysis or show adaptation.
        _mode_emotion_flow(args)
        if not args.quiet:
            print(f"\n{'═' * 60}")
            print(f"  Emotion-flow complete!")
            print(f"{'═' * 60}\n")

    elif args.mode == "publish":
        # Standalone publish — read existing JSON → TXT/PDF.
        # Fan out across every language in --out-lang.
        _run_language_matrix(
            args.script_input,
            args,
            do_render=False,
            do_publish=True,
        )
        if not args.quiet:
            print(f"\n{'═' * 60}")
            print(f"  Publish complete!")
            print(f"{'═' * 60}\n")

    elif args.mode == "generate":
        # Generate only — produce JSON, no audio.
        json_path = _mode_generate(args)
        # Translate (and optionally publish) into each non-root language.
        non_root_requested = any(
            lang != args.root_lang for lang in args.out_lang_list
        )
        if non_root_requested or args.publish:
            # In generate mode the translated JSONs ARE the deliverable
            # for non-root languages, so force-keep them unless we're
            # also publishing (then user's --keep-translated wins).
            saved_keep = args.keep_translated
            if not args.publish:
                args.keep_translated = True
            try:
                _run_language_matrix(
                    json_path,
                    args,
                    do_render=False,
                    do_publish=args.publish,
                )
            finally:
                args.keep_translated = saved_keep
        if not args.quiet:
            print(f"\n{'═' * 60}")
            print(f"  Generated: {json_path}")
            print(f"{'═' * 60}\n")
        else:
            print(json_path)

    elif args.mode == "render":
        # Render only — audio from existing JSON, fanned across langs.
        _run_language_matrix(
            args.script_input,
            args,
            do_render=True,
            do_publish=False,
        )

    elif args.mode == "ingest":
        _mode_ingest(args)

    elif args.mode == "dedupe":
        _mode_dedupe(args)

    elif args.mode == "package":
        # Standalone packaging — read existing JSON → SRT + show notes.
        if args.script_input is None:
            print(
                "Error: --script-input is required for 'package' mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.script_input.exists():
            print(
                f"Error: File not found: {args.script_input}",
                file=sys.stderr,
            )
            sys.exit(1)
        _mode_package(args.script_input, args, event_durations=None)
        if not args.quiet:
            print(f"\n{'\u2550' * 60}")
            print("  Packaging complete!")
            print(f"{'\u2550' * 60}\n")

    elif args.mode == "voice-prep":
        _mode_voice_prep(args)

    elif args.mode == "voice-assign":
        _mode_voice_assign(args)

    elif args.mode == "video":
        _mode_video(args)
    elif args.mode == "audition":
        _mode_audition(args)
    elif args.mode == "full":
        # Full pipeline — generate then render (+ optional publish),
        # fanning across every requested output language.
        json_path = _mode_generate(args)

        step_n = 3 if args.publish else 2
        if not args.quiet:
            print(f"\n{'─' * 60}")
            print(f"  Step 2/{step_n}: Rendering audio…")
            print(f"{'─' * 60}\n")

        _run_language_matrix(
            json_path,
            args,
            do_render=True,
            do_publish=args.publish,
        )

        # ── Optional: YouTube packaging (--package add-on) ───────────
        if getattr(args, "package", False):
            if not args.quiet:
                print(f"\n{'─' * 60}")
                print("  Packaging for YouTube…")
                print(f"{'─' * 60}")
            # event_durations not available here (absorbed by _run_language_matrix);
            # captions will use fallback timing (~3 s per event).
            _mode_package(json_path, args, event_durations=None)

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

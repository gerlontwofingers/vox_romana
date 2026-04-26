"""Show runners for vox_romana — studio floor dispatch functions.

Contains all per-show-type pipeline functions (_run_debate, _run_news,
_run_podcast, _run_audiobook), the dispatch table (_RUNNERS), and
run_generate() — the core generate logic called from both run_show.py
(standalone CLI shim) and run.py (master pipeline) without serializing
args back into an argv list.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from vox_romana.core.client import DeepSeekClient
from vox_romana.core.duration import estimate_total_words, parse_duration_from_text
from vox_romana.core.fact_checker import FactChecker
from vox_romana.core.paths import OUTPUT_DIR as _OUTPUT_DIR
from vox_romana.core.router import ShowRouter


def log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"  {msg}", flush=True)


def _enable_research(bootstrap_kwargs: dict, args) -> None:
    """Wire University + Memory + Google into ``bootstrap_kwargs``.

    Centralized here so all show types (debate / news / podcast / audiobook)
    use the exact same tiered fallback ladder:

        University (Scholar session notes, if --university-session given)
            ↓  (if not enough material)
        Memory (Chroma + FTS5, free, local)
            ↓  (if still not enough)
        Google (paid, web)
            ↓  (last resort)
        LLM-only (no search at all)

    Any tier failure is non-fatal — we log and continue down the ladder.
    """
    bootstrap_kwargs["auto_research"] = True

    providers: list = []

    # ── Tier 0: University session (pinned deep-research notes) ──────
    university_session_id = getattr(args, "university_session", None)
    if university_session_id:
        try:
            from vox_romana.university.search_provider import UniversitySearchProvider
            providers.append(UniversitySearchProvider(university_session_id))
            log(f"  University session: {university_session_id}", quiet=args.quiet)
        except Exception as e:
            log(
                f"  University session: OFF (load failed: {e})",
                quiet=args.quiet,
            )

    # ── Tier 1: Local memory (Chroma + FTS5) ─────────────────────────
    if getattr(args, "memory", True):
        try:
            from vox_romana.core.catalog import Catalog
            from vox_romana.core.search import MemorySearchProvider
            from vox_romana.memory.recall import MemoryRecall
            catalog = Catalog()
            memory = MemoryRecall(catalog=catalog)
            providers.append(MemorySearchProvider(memory))
            log("  Memory-first search: ON", quiet=args.quiet)
        except Exception as e:
            log(f"  Memory-first search: OFF (init failed: {e})", quiet=args.quiet)

    # ── Tier 2: Google Custom Search ──────────────────────────────────
    try:
        from vox_romana.core.search import GoogleSearchProvider
        providers.append(GoogleSearchProvider())
        log("  Research mode ON (Google)", quiet=args.quiet)
    except (ValueError, ImportError):
        log("  Research mode ON (LLM knowledge — no Google keys)", quiet=args.quiet)

    # ── Wire up the provider chain ────────────────────────────────────
    if len(providers) == 1:
        bootstrap_kwargs["search_provider"] = providers[0]
    elif len(providers) > 1:
        from vox_romana.core.search import ChainSearchProvider
        bootstrap_kwargs["search_provider"] = ChainSearchProvider(providers)
    # If 0 providers were loaded successfully, leave search_provider unset
    # so the bootstrap falls back to LLM-only research.


# ── Word/duration accounting helpers ───────────────────────────


def _count_words_in_events(events) -> int:
    """Sum the words across a list of event dicts or typed event objects."""
    total = 0
    for evt in events:
        text = getattr(evt, "text", None)
        if text is None and hasattr(evt, "get"):
            text = evt.get("text", "")
        if text:
            total += len(text.split())
    return total


def _log_length_summary(
    *,
    actual_words: int,
    target_words: int | None,
    target_minutes: float | None,
    content_type: str,
    length_warnings: list[str] | None = None,
    quiet: bool = False,
) -> None:
    """Emit a one-line summary of generated word count vs target.

    Always shows actual words + estimated duration. When a target is
    set, also shows the target and the percentage achieved. Length
    warnings (from the conductor) are surfaced afterwards.
    """
    from vox_romana.core.duration import get_wpm, words_to_seconds

    wpm = get_wpm(content_type)
    est_seconds = words_to_seconds(actual_words, wpm=wpm)
    est_minutes = est_seconds / 60.0

    if target_words is not None and target_words > 0:
        pct = (actual_words / target_words) * 100
        target_min_str = (
            f" / {target_minutes:.0f} min" if target_minutes else ""
        )
        log(
            f"  {actual_words:,} words \u2248 {est_minutes:.0f} min "
            f"(target {target_words:,} words{target_min_str}, {pct:.0f}%)",
            quiet=quiet,
        )
    else:
        log(
            f"  {actual_words:,} words \u2248 {est_minutes:.0f} min (no target)",
            quiet=quiet,
        )

    if length_warnings:
        for w in length_warnings[:5]:  # cap noise
            log(f"  \u26a0 {w}", quiet=quiet)
        if len(length_warnings) > 5:
            log(
                f"  \u26a0 \u2026 and {len(length_warnings) - 5} more length warning(s)",
                quiet=quiet,
            )


# ── Checkpoint plumbing ─────────────────────────────────


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


# ── Show runners ──────────────────────────────────────────────────────────


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
    overrides = _resolve_duration(args, brief)
    if "target_duration_minutes" in overrides:
        overrides["target_words"] = estimate_total_words(
            overrides["target_duration_minutes"], "debate",
            engine=args.engine,
        )
    if args.turns is not None:
        overrides["target_turns"] = args.turns
    if args.tone is not None:
        overrides["tone"] = args.tone
    if args.debate_format is not None:
        overrides["debate_format"] = args.debate_format
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries
    if args.allow_bootstrap_fabrications:
        overrides["allow_bootstrap_fabrications"] = True

    # ── Bootstrap ────────────────────────────────────────────────────
    log("Bootstrap (casting, profiling, directing)...", quiet=args.quiet)
    t0 = time.time()

    reference_text: str | None = None
    if args.reference_material is not None:
        if not args.reference_material.exists():
            print(f"Error: reference material not found: {args.reference_material}", file=sys.stderr)
            sys.exit(1)
        reference_text = args.reference_material.read_text().strip()

    if args.content_mode != "auto":
        overrides["content_mode"] = args.content_mode

    bootstrap_kwargs = dict(
        reference_material=reference_text,
        fact_check=args.fact_check,
        overrides=overrides,
        auto_research=False,
    )
    if args.fact_check != "off":
        log(f"  Fact-check: {args.fact_check}", quiet=args.quiet)
    if args.content_mode != "auto":
        log(f"  Content mode: {args.content_mode}", quiet=args.quiet)

    if args.research:
        _enable_research(bootstrap_kwargs, args)

    bootstrap = DebateBootstrap(client=client)
    config = bootstrap.from_brief(brief, **bootstrap_kwargs)

    t1 = time.time()
    log(f"  Done in {t1 - t0:.1f}s", quiet=args.quiet)
    log(f"  Title: {config.title}", quiet=args.quiet)
    log(f"  Participants: {', '.join(p.name for p in config.participants)}", quiet=args.quiet)
    log(f"  Format: {config.debate_format} | Tone: {config.tone} | Turns: {config.target_turns}", quiet=args.quiet)
    print()

    # ── Conductor ────────────────────────────────────────────────────
    log("Conductor (running the debate)...", quiet=args.quiet)
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
    _log_length_summary(
        actual_words=_count_words_in_events(result.debate_events),
        target_words=overrides.get("target_words"),
        target_minutes=overrides.get("target_duration_minutes"),
        content_type="debate",
        length_warnings=getattr(conductor, "_length_warnings", None),
        quiet=args.quiet,
    )
    print()

    # ── Serialize ────────────────────────────────────────────────────
    log("Saving output...", quiet=args.quiet)

    merged = result.to_merged_debate(series_id=getattr(args, "series_id", None))
    serializer = DebateSerializer()

    stem = args.prompt.stem
    tag_part = f"_{args.tag}" if args.tag else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"debate_{stem}{tag_part}_{timestamp}.json"
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
            engine=args.engine,
        )
    if args.segments is not None:
        overrides["target_segments"] = args.segments
    if args.tone is not None:
        overrides["tone"] = args.tone
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries
    if args.allow_bootstrap_fabrications:
        overrides["allow_bootstrap_fabrications"] = True

    # ── Bootstrap ────────────────────────────────────────────────────
    log("Bootstrap (rundown planning, anchor profiling)...", quiet=args.quiet)
    t0 = time.time()

    reference_text_news: str | None = None
    if args.reference_material is not None:
        if not args.reference_material.exists():
            print(f"Error: reference material not found: {args.reference_material}", file=sys.stderr)
            sys.exit(1)
        reference_text_news = args.reference_material.read_text().strip()

    if args.content_mode != "auto":
        overrides["content_mode"] = args.content_mode

    bootstrap_kwargs = dict(
        reference_material=reference_text_news,
        fact_check=args.fact_check,
        overrides=overrides,
        auto_research=False,
    )
    if args.fact_check != "off":
        log(f"  Fact-check: {args.fact_check}", quiet=args.quiet)
    if args.content_mode != "auto":
        log(f"  Content mode: {args.content_mode}", quiet=args.quiet)

    if args.research:
        _enable_research(bootstrap_kwargs, args)

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
    log("Conductor (generating the broadcast)...", quiet=args.quiet)
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
    _log_length_summary(
        actual_words=_count_words_in_events(result.news_events),
        target_words=overrides.get("target_words"),
        target_minutes=overrides.get("target_duration_minutes"),
        content_type="news",
        length_warnings=getattr(conductor, "_length_warnings", None),
        quiet=args.quiet,
    )
    print()

    # ── Serialize ────────────────────────────────────────────────────
    log("Saving output...", quiet=args.quiet)

    merged = result.to_merged_news_show(series_id=getattr(args, "series_id", None))
    serializer = NewsSerializer()

    stem = args.prompt.stem
    tag_part = f"_{args.tag}" if args.tag else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"news_{stem}{tag_part}_{timestamp}.json"
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
            engine=args.engine,
        )
        overrides["target_words"] = target_words
    if args.tone is not None:
        overrides["tone"] = args.tone
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries
    if args.allow_bootstrap_fabrications:
        overrides["allow_bootstrap_fabrications"] = True

    # ── Bootstrap ────────────────────────────────────────────────────
    log("Bootstrap (casting hosts, profiling, directing)...", quiet=args.quiet)
    t0 = time.time()

    reference_text_podcast: str | None = None
    if args.reference_material is not None:
        if not args.reference_material.exists():
            print(f"Error: reference material not found: {args.reference_material}", file=sys.stderr)
            sys.exit(1)
        reference_text_podcast = args.reference_material.read_text().strip()

    if args.content_mode != "auto":
        overrides["content_mode"] = args.content_mode

    bootstrap_kwargs = dict(
        overrides=overrides,
        auto_research=False,
        fact_check=args.fact_check,
        reference_material=reference_text_podcast,
    )
    if target_words is not None:
        bootstrap_kwargs["target_words"] = target_words
    if args.fact_check != "off":
        log(f"  Fact-check: {args.fact_check}", quiet=args.quiet)
    if args.content_mode != "auto":
        log(f"  Content mode: {args.content_mode}", quiet=args.quiet)

    if args.research:
        _enable_research(bootstrap_kwargs, args)

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
    log("Conductor (generating the episode)...", quiet=args.quiet)
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
    _log_length_summary(
        actual_words=_count_words_in_events(result.podcast_events),
        target_words=overrides.get("target_words") or target_words,
        target_minutes=overrides.get("target_duration_minutes"),
        content_type="podcast",
        length_warnings=getattr(conductor, "_length_warnings", None),
        quiet=args.quiet,
    )
    print()

    # ── Serialize ────────────────────────────────────────────────────
    log("Saving output...", quiet=args.quiet)

    merged = result.to_merged_podcast(series_id=getattr(args, "series_id", None))
    serializer = PodcastSerializer()

    stem = args.prompt.stem
    tag_part = f"_{args.tag}" if args.tag else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"podcast_{stem}{tag_part}_{timestamp}.json"
    output_path = args.output / filename

    serializer.save(merged, output_path)
    return output_path


def _run_audiobook(brief: str, client: DeepSeekClient, args: argparse.Namespace) -> Path:
    """Run the audiobook pipeline, return the output path.

    Audiobook studio floor. The brief has been classified as an audiobook by
    the router — this function handles bootstrap → conductor → serializer.

    Audiobook generation is chapter-by-chapter: the narrator agent delivers
    each chapter in sequence, maintaining continuity via sliding-window
    session compression.

    Args:
        brief:  The full text of the user's prompt file.
        client: Shared DeepSeek API client.
        args:   Parsed CLI arguments (overrides, research, tag, etc.).

    Returns:
        Path to the saved JSON output file.
    """
    from vox_romana.shows.audiobooks.bootstrap import AudiobookBootstrap
    from vox_romana.shows.audiobooks.conductor import AudiobookConductor
    from vox_romana.shows.audiobooks.serializer import AudiobookSerializer

    # ── Build overrides from CLI flags ───────────────────────────────
    overrides = _resolve_duration(args, brief)
    target_words = None
    if "target_duration_minutes" in overrides:
        target_words = estimate_total_words(
            overrides["target_duration_minutes"], "audiobook",
            engine=args.engine,
        )
        overrides["target_words"] = target_words
    if args.tone is not None:
        overrides["tone"] = args.tone
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries
    if args.allow_bootstrap_fabrications:
        overrides["allow_bootstrap_fabrications"] = True

    # ── Bootstrap ────────────────────────────────────────────────────
    log("Bootstrap (planning chapters, profiling narrator, directing)...", quiet=args.quiet)
    t0 = time.time()

    reference_text_audiobook: str | None = None
    if args.reference_material is not None:
        if not args.reference_material.exists():
            print(f"Error: reference material not found: {args.reference_material}", file=sys.stderr)
            sys.exit(1)
        reference_text_audiobook = args.reference_material.read_text().strip()

    bootstrap_kwargs = dict(
        overrides=overrides,
        auto_research=False,
        fact_check=args.fact_check,
        reference_material=reference_text_audiobook,
    )
    if target_words is not None:
        bootstrap_kwargs["target_words"] = target_words
    if args.fact_check != "off":
        log(f"  Fact-check: {args.fact_check}", quiet=args.quiet)

    if args.research:
        _enable_research(bootstrap_kwargs, args)

    bootstrap = AudiobookBootstrap(client=client)
    config = bootstrap.from_brief(brief, **bootstrap_kwargs)

    t1 = time.time()
    log(f"  Done in {t1 - t0:.1f}s", quiet=args.quiet)
    log(f"  Title:    {config.title}", quiet=args.quiet)
    log(f"  Genre:    {config.genre}", quiet=args.quiet)
    narrator_name = config.narrator.name if config.narrator else "(no narrator)"
    log(f"  Narrator: {narrator_name}", quiet=args.quiet)
    log(f"  Chapters: {len(config.chapters)}", quiet=args.quiet)
    print()

    # ── Conductor ────────────────────────────────────────────────────
    log("Conductor (generating the audiobook)...", quiet=args.quiet)
    t2 = time.time()

    checkpoint_kwargs = _build_checkpoint_kwargs(
        args, "audiobook",
        target_minutes=overrides.get("target_duration_minutes"),
        num_segments=len(config.chapters),
        quiet=args.quiet,
    )
    conductor = AudiobookConductor(config, client=client, **checkpoint_kwargs)
    result = conductor.run()

    t3 = time.time()
    log(f"  Done in {t3 - t2:.1f}s ({result.total_llm_calls} LLM calls)", quiet=args.quiet)
    log(f"  Generated {len(result.audiobook_events)} events across {len(result.audiobook_chapters)} chapters", quiet=args.quiet)
    _log_length_summary(
        actual_words=_count_words_in_events(result.audiobook_events),
        target_words=overrides.get("target_words") or target_words,
        target_minutes=overrides.get("target_duration_minutes"),
        content_type="audiobook",
        length_warnings=getattr(conductor, "_length_warnings", None),
        quiet=args.quiet,
    )
    print()

    # ── Serialize ────────────────────────────────────────────────────
    log("Saving output...", quiet=args.quiet)

    merged = result.to_merged_audiobook()
    serializer = AudiobookSerializer()

    stem = args.prompt.stem
    tag_part = f"_{args.tag}" if args.tag else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"audiobook_{stem}{tag_part}_{timestamp}.json"
    output_path = args.output / filename

    serializer.save(merged, output_path)
    return output_path


# ── Dispatch table ───────────────────────────────────────────────────────
_RUNNERS = {
    "debate": _run_debate,
    "news": _run_news,
    "podcast": _run_podcast,
    "audiobook": _run_audiobook,
}


def run_generate(args: argparse.Namespace) -> Path:
    """Core generate pipeline — routing, bootstrap, conduct, serialize.

    Extracted from run_show.main() so it can be called from both
    run_show.py (standalone CLI shim) and run.py (master pipeline)
    without serializing args back into an argv list.

    Args:
        args: Parsed CLI namespace. Must have at minimum: prompt, output,
              show_type, research, quiet, tag, fact_check, content_mode,
              duration, tolerance, checkpoint, engine, turns, tone,
              debate_format, segments, max_retries, reference_material,
              allow_bootstrap_fabrications.

    Returns:
        Path to the generated JSON file.
    """
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

    # ── Pre-flight: validate brief for factual accuracy ──────────────
    if args.fact_check != "off":
        _preflight_checker = FactChecker.build_default(client)
        _preflight_report = _preflight_checker.check_content(
            brief, chunk_id="brief", mode="quick", content_mode="factual"
        )
        if _preflight_report.has_fabrications:
            print("\n\u26a0  PRE-FLIGHT WARNING: Your brief may contain incorrect claims.")
            print("   The fact-checker flagged the following based on LLM knowledge:")
            for _r in _preflight_report.fabricated_claims:
                print(f"   \u2022 {_r.claim}")
                print(f"     \u2192 {_r.explanation}")
            print("   The show will still be generated. Verify these facts before publishing.\n")

    # ══════════════════════════════════════════════════════════════════
    # ROUTE — classify the brief → pick the right studio
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
        log(f"Tolerance: \u00b1{duration_overrides['tolerance_percent']:.0f}%", quiet=args.quiet)

    print()

    if args.show_type:
        show_type = args.show_type
        log(f"Show type forced \u2192 {show_type}", quiet=args.quiet)
    else:
        log("Classifying brief...", quiet=args.quiet)
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
    # RUN THE PIPELINE — bootstrap → conductor → serialize
    # ══════════════════════════════════════════════════════════════════
    # The dispatch table routes to the right studio floor based on the
    # classified show type. Each runner handles its own steps internally.
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

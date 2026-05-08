"""Show runners for vox_romana — studio floor dispatch functions.

Contains all per-show-type pipeline functions (_run_debate, _run_news,
_run_podcast, _run_audiobook), the dispatch table (_RUNNERS), and
run_generate() — the core generate logic called from both run_show.py
(standalone CLI shim) and run.py (master pipeline) without serializing
args back into an argv list.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

from vox_romana.core.client import DeepSeekClient
from vox_romana.core.duration import estimate_total_words, parse_duration_from_text
from vox_romana.core.fact_checker import FactChecker
from vox_romana.core.router import ShowRouter


def log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"  {msg}", flush=True)


_PRONUNCIATION_BLOCK_RE = re.compile(
    r"(?:^|\n)Pronunciations:\s*\n((?:[ \t]*[^\n]+\n?)*)",
    re.IGNORECASE,
)


def _parse_pronunciations(brief: str) -> tuple[str, dict[str, Any]]:
    """Strip the optional ``Pronunciations:`` block from *brief*.

    Syntax (one entry per line, inside the block)::

        Pronunciations:
        Okonkwo = oh-KOHN-kwoh | /oʊˈkɒŋkwoʊ/
        Nguyen  = nwin

    Returns a ``(cleaned_brief, overrides)`` tuple where *overrides* maps
    each source phrase to ``{"spoken": str, "ipa": str | None}``.
    Lines that don't contain ``=`` are silently skipped.
    """
    m = _PRONUNCIATION_BLOCK_RE.search(brief)
    if not m:
        return brief, {}

    block_text = m.group(1)
    overrides: dict[str, Any] = {}
    for line in block_text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        phrase, _, rest = line.partition("=")
        phrase = phrase.strip()
        if not phrase:
            continue
        parts = [p.strip() for p in rest.split("|", 1)]
        spoken = parts[0] if parts else ""
        ipa = parts[1] if len(parts) > 1 else None
        if spoken:
            overrides[phrase] = {"spoken": spoken, "ipa": ipa}

    cleaned = _PRONUNCIATION_BLOCK_RE.sub("", brief).strip()
    return cleaned, overrides


def _build_series_context(catalog, series_id: str) -> str:
    """Format a series record + recent episodes into a SERIES CONTEXT block.

    The returned string is prepended to the brief inside each bootstrap's
    ``from_brief()`` so every Architect / Profile Builder / Director sees
    the recurring writers'-bible before reading the user's specific prompt.

    Returns an empty string if the series has no bible AND no prior
    episodes (the caller can still proceed — context just adds nothing).
    """
    series = catalog.get_series(series_id)
    if series is None:
        raise ValueError(f"Unknown series: {series_id}")

    recent = catalog.get_recent_episode_summaries(series_id, n=5)

    parts: list[str] = []
    parts.append("SERIES CONTEXT")
    parts.append("==============")
    parts.append(f"Series: {series.title}")
    if series.show_type:
        parts.append(f"Show type: {series.show_type}")
    if series.tone:
        parts.append(f"Tone: {series.tone}")
    if series.bible:
        parts.append("")
        parts.append("Writers' bible:")
        parts.append(series.bible)
    if recent:
        parts.append("")
        parts.append("Recent episodes (most recent first):")
        for ep in recent:
            ep_no = ep["episode_number"]
            label = f"  • Episode {ep_no}: " if ep_no else "  • "
            parts.append(f"{label}{ep['title']}")
    if series.audience_profile is not None:
        parts.append("")
        parts.append("AUDIENCE")
        parts.append("========")
        parts.append(series.audience_profile.to_prompt_text())
    parts.append("──────────────")
    return "\n".join(parts)


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
    verbose: bool = False,
) -> None:
    """Emit a one-line summary of generated word count vs target.

    Always shows actual words + estimated duration. When a target is
    set, also shows the target and the percentage achieved. Length
    warnings (from the conductor) are surfaced afterwards — only when
    ``verbose`` is True (they are kept silent by default to keep clean
    runs uncluttered; the conductor still records them).
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

    if verbose and length_warnings:
        for w in length_warnings[:5]:  # cap noise
            log(f"  \u26a0 {w}", quiet=quiet)
        if len(length_warnings) > 5:
            log(
                f"  \u26a0 \u2026 and {len(length_warnings) - 5} more length warning(s)",
                quiet=quiet,
            )
    elif length_warnings and not quiet:
        # One-line nudge so the user knows there were warnings without
        # spilling them. Re-run with --verbose to see them.
        log(
            f"  ({len(length_warnings)} length warning(s) suppressed; "
            f"re-run with --verbose to see them)",
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


# ── Shared runner helpers ────────────────────────────────────────────────


def _read_reference_material(args: argparse.Namespace) -> str | None:
    """Read reference material text, or return None if not provided."""
    if args.reference_material is None:
        return None
    if not args.reference_material.exists():
        print(
            f"Error: reference material not found: {args.reference_material}",
            file=sys.stderr,
        )
        sys.exit(1)
    return args.reference_material.read_text().strip()


def _build_bootstrap_kwargs(
    args: argparse.Namespace,
    overrides: dict,
    *,
    reference_text: str | None,
    target_words: int | None = None,
) -> dict:
    """Assemble ``bootstrap_kwargs`` common to all show runners.

    Builds the base dict, appends optional series context and target_words,
    logs fact-check and content-mode settings, and wires up research
    providers if ``args.research`` is set.
    """
    bootstrap_kwargs: dict = dict(
        reference_material=reference_text,
        fact_check=args.fact_check,
        overrides=overrides,
        auto_research=False,
    )
    if target_words is not None:
        bootstrap_kwargs["target_words"] = target_words
    if getattr(args, "series_context", None):
        bootstrap_kwargs["series_context"] = args.series_context
    if args.fact_check != "off":
        log(f"  Fact-check: {args.fact_check}", quiet=args.quiet)
    if args.content_mode != "auto":
        log(f"  Content mode: {args.content_mode}", quiet=args.quiet)
    if args.research:
        _enable_research(bootstrap_kwargs, args)
    return bootstrap_kwargs


def _build_output_path(args: argparse.Namespace, content_type: str) -> Path:
    """Return the output JSON ``Path`` for a generated show.

    Uses ``args.name`` as the filename stem when provided; otherwise
    auto-generates ``{content_type}_{prompt_stem}[_{tag}]_{timestamp}.json``.
    """
    if getattr(args, "name", None):
        filename = f"{args.name}.json"
    else:
        stem = args.prompt.stem
        tag_part = f"_{args.tag}" if args.tag else ""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{content_type}_{stem}{tag_part}_{timestamp}.json"
    return args.output / filename


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

    reference_text = _read_reference_material(args)

    if args.content_mode != "auto":
        overrides["content_mode"] = args.content_mode

    bootstrap_kwargs = _build_bootstrap_kwargs(args, overrides, reference_text=reference_text)

    bootstrap = DebateBootstrap(client=client)
    config = bootstrap.from_brief(brief, **bootstrap_kwargs)

    # ── Clamp target_turns so it fits the word budget ────────────────
    # The Architect doesn't know the target duration when it picks
    # target_turns, so for short shows it routinely overshoots.
    # Reserve some words for the host intro+outro narration, then
    # divide the remainder by an average per-turn word count.
    if config.target_words and config.target_turns:
        AVG_WORDS_PER_TURN = 40
        HOST_OVERHEAD_WORDS = 280  # intro + outro + closing turns spillover
        budget_for_turns = max(80, config.target_words - HOST_OVERHEAD_WORDS)
        max_turns = max(4, budget_for_turns // AVG_WORDS_PER_TURN)
        if config.target_turns > max_turns:
            log(
                f"  Clamping target_turns: {config.target_turns} -> {max_turns} "
                f"(fits {config.target_words}-word budget)",
                quiet=args.quiet,
            )
            config.target_turns = max_turns

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
    conductor = DebateConductor(
        config,
        client=client,
        lens_rack=getattr(args, "lens_rack", None),
        **checkpoint_kwargs,
    )
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
        verbose=getattr(args, "verbose", False),
    )
    print()

    # ── Serialize ────────────────────────────────────────────────────
    log("Saving output...", quiet=args.quiet)

    merged = result.to_merged_debate(series_id=getattr(args, "series_id", None))
    pron = getattr(args, "pronunciation_overrides", {})
    if pron:
        merged = dataclasses.replace(merged, pronunciation_overrides=pron)
    serializer = DebateSerializer()

    output_path = _build_output_path(args, "debate")

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

    reference_text_news = _read_reference_material(args)

    if args.content_mode != "auto":
        overrides["content_mode"] = args.content_mode

    bootstrap_kwargs = _build_bootstrap_kwargs(args, overrides, reference_text=reference_text_news)

    bootstrap = NewsBootstrap(client=client)
    config = bootstrap.from_brief(brief, **bootstrap_kwargs)

    # ── Trim segment list to fit the word budget ─────────────────────
    # The Architect doesn't know the target duration when it picks
    # target_segments, so for short broadcasts it routinely overshoots.
    # Each segment costs ~150 anchor words + ~30 transition words.
    if config.target_words and config.segments:
        AVG_WORDS_PER_SEGMENT = 180
        HOST_OVERHEAD_WORDS = 120  # intro + outro
        budget_for_segments = max(120, config.target_words - HOST_OVERHEAD_WORDS)
        max_segments = max(1, budget_for_segments // AVG_WORDS_PER_SEGMENT)
        if len(config.segments) > max_segments:
            log(
                f"  Trimming segments: {len(config.segments)} -> {max_segments} "
                f"(fits {config.target_words}-word budget)",
                quiet=args.quiet,
            )
            config.segments = config.segments[:max_segments]

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
        verbose=getattr(args, "verbose", False),
    )
    print()

    # ── Serialize ────────────────────────────────────────────────────
    log("Saving output...", quiet=args.quiet)

    merged = result.to_merged_news_show(series_id=getattr(args, "series_id", None))
    pron = getattr(args, "pronunciation_overrides", {})
    if pron:
        merged = dataclasses.replace(merged, pronunciation_overrides=pron)
    serializer = NewsSerializer()

    output_path = _build_output_path(args, "news")

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

    reference_text_podcast = _read_reference_material(args)

    if args.content_mode != "auto":
        overrides["content_mode"] = args.content_mode

    bootstrap_kwargs = _build_bootstrap_kwargs(
        args, overrides, reference_text=reference_text_podcast, target_words=target_words
    )

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
        verbose=getattr(args, "verbose", False),
    )
    print()

    # ── Serialize ────────────────────────────────────────────────────
    log("Saving output...", quiet=args.quiet)

    merged = result.to_merged_podcast(series_id=getattr(args, "series_id", None))
    pron = getattr(args, "pronunciation_overrides", {})
    if pron:
        merged = dataclasses.replace(merged, pronunciation_overrides=pron)
    serializer = PodcastSerializer()

    output_path = _build_output_path(args, "podcast")

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

    reference_text_audiobook = _read_reference_material(args)

    bootstrap_kwargs = _build_bootstrap_kwargs(
        args, overrides, reference_text=reference_text_audiobook, target_words=target_words
    )

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
        verbose=getattr(args, "verbose", False),
    )
    print()

    # ── Serialize ────────────────────────────────────────────────────
    log("Saving output...", quiet=args.quiet)

    merged = result.to_merged_audiobook()
    pron = getattr(args, "pronunciation_overrides", {})
    if pron:
        merged = dataclasses.replace(merged, pronunciation_overrides=pron)
    serializer = AudiobookSerializer()

    output_path = _build_output_path(args, "audiobook")

    serializer.save(merged, output_path)
    return output_path


def _run_lecture(brief: str, client: DeepSeekClient, args: argparse.Namespace) -> Path:
    """Run the lecture pipeline, return the output path.

    Lecture studio floor. The brief has been classified as a lecture by the
    router — this function handles bootstrap → conductor → serializer.

    Lecture generation is section-by-section: a single professor agent
    delivers each section in sequence (with type-specific overlay prompts:
    intro / concept / case_study / exercise / recap / outro).
    """
    from vox_romana.shows.lecture.bootstrap import LectureBootstrap
    from vox_romana.shows.lecture.conductor import LectureConductor
    from vox_romana.shows.lecture.serializer import LectureSerializer

    # ── Build overrides from CLI flags ─────────────────────────────
    overrides = _resolve_duration(args, brief)
    target_words = None
    if "target_duration_minutes" in overrides:
        target_words = estimate_total_words(
            overrides["target_duration_minutes"], "lecture",
            engine=args.engine,
        )
        overrides["target_words"] = target_words
    if args.tone is not None:
        overrides["tone"] = args.tone
    if args.max_retries is not None:
        overrides["max_retries"] = args.max_retries
    if args.allow_bootstrap_fabrications:
        overrides["allow_bootstrap_fabrications"] = True

    # ── Bootstrap ──────────────────────────────────────────────────
    log("Bootstrap (planning sections, profiling professor, directing)...", quiet=args.quiet)
    t0 = time.time()

    reference_text_lecture = _read_reference_material(args)

    bootstrap_kwargs = _build_bootstrap_kwargs(
        args, overrides, reference_text=reference_text_lecture, target_words=target_words
    )

    bootstrap = LectureBootstrap(client=client)
    config = bootstrap.from_brief(brief, **bootstrap_kwargs)

    t1 = time.time()
    log(f"  Done in {t1 - t0:.1f}s", quiet=args.quiet)
    log(f"  Title:        {config.title}", quiet=args.quiet)
    log(f"  Subject area: {config.subject_area}", quiet=args.quiet)
    professor_name = config.professor.name if config.professor else "(no professor)"
    log(f"  Professor:    {professor_name}", quiet=args.quiet)
    log(f"  Sections:     {len(config.sections)}", quiet=args.quiet)
    print()

    # ── Conductor ──────────────────────────────────────────────────
    log("Conductor (generating the lecture)...", quiet=args.quiet)
    t2 = time.time()

    checkpoint_kwargs = _build_checkpoint_kwargs(
        args, "lecture",
        target_minutes=overrides.get("target_duration_minutes"),
        num_segments=len(config.sections),
        quiet=args.quiet,
    )
    conductor = LectureConductor(config, client=client, **checkpoint_kwargs)
    result = conductor.run()

    t3 = time.time()
    log(f"  Done in {t3 - t2:.1f}s ({result.total_llm_calls} LLM calls)", quiet=args.quiet)
    log(f"  Generated {len(result.lecture_events)} events across {len(result.lecture_sections)} sections", quiet=args.quiet)
    _log_length_summary(
        actual_words=_count_words_in_events(result.lecture_events),
        target_words=overrides.get("target_words") or target_words,
        target_minutes=overrides.get("target_duration_minutes"),
        content_type="lecture",
        length_warnings=getattr(conductor, "_length_warnings", None),
        quiet=args.quiet,
        verbose=getattr(args, "verbose", False),
    )
    print()

    # ── Serialize ──────────────────────────────────────────────────
    log("Saving output...", quiet=args.quiet)

    merged = result.to_merged_lecture()
    pron = getattr(args, "pronunciation_overrides", {})
    if pron:
        merged = dataclasses.replace(merged, pronunciation_overrides=pron)
    serializer = LectureSerializer()

    output_path = _build_output_path(args, "lecture")

    serializer.save(merged, output_path)
    return output_path


# ── Dispatch table ───────────────────────────────────────────────────────
_RUNNERS = {
    "debate": _run_debate,
    "news": _run_news,
    "podcast": _run_podcast,
    "audiobook": _run_audiobook,
    "lecture": _run_lecture,
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

    # ── Strip Pronunciations: block before the LLM ever sees the brief ──
    brief, _pronunciation_overrides = _parse_pronunciations(brief)
    args.pronunciation_overrides = _pronunciation_overrides

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Resolve --series → SERIES CONTEXT block + voice-locking key ──
    # When --series is given, we (1) load the series record from the
    # catalog and format a SERIES CONTEXT block that bootstraps prepend
    # to the brief, and (2) stamp the same id onto args.series_id so the
    # downstream voice registry locks character voices across episodes.
    args.series_context = None
    series_arg = getattr(args, "series", None)
    if series_arg:
        from vox_romana.core.catalog import Catalog
        with Catalog() as _cat:
            try:
                args.series_context = _build_series_context(_cat, series_arg)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        # Reuse the same id for voice locking. Existing code already reads
        # args.series_id via getattr() in each runner.
        args.series_id = series_arg

    # ── Create a shared client ─────────────────────────────────────
    # One client for the whole run so bootstrap + conductor + editor
    # all share the same connection pool.
    client = DeepSeekClient()

    # ── Build the LLM lens rack ────────────────────────────────────
    # Chunk 1 of the burn-down plan.  The rack lets users pick a
    # different LLM per role / per character.  We construct it once
    # here and stash it on args so downstream code (Chunk 5) can call
    # ``args.lens_rack.select(role=..., character=...)`` without
    # plumbing a new parameter through every function.  The default
    # rack picks cloud-deepseek for everything, so existing behaviour
    # is unchanged when no --llm-* flags are passed.
    from vox_romana.core.lens import LLMLensRack
    from vox_romana.core.llm_config import load_llm_config, parse_lens_overrides

    try:
        lens_overrides = parse_lens_overrides(getattr(args, "llm_lens", None))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    args.lens_rack = LLMLensRack.from_config(
        load_llm_config(),
        backend_override=getattr(args, "llm_backend", None),
        lens_overrides=lens_overrides,
    )

    # ── Pre-flight: validate brief for factual accuracy ──────────────
    if args.fact_check != "off":
        _preflight_checker = FactChecker.build_default(client)
        _preflight_report = _preflight_checker.check_content(
            brief, content_id="brief", mode="quick", content_mode="factual"
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


# ─────────────────────────────────────────────────────────────────────────────
# Render pipeline — extracted from render_show.py
# ─────────────────────────────────────────────────────────────────────────────

_render_logger = logging.getLogger(__name__)

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
    from vox_romana.tts.renderer import strip_stage_directions
    return sum(
        len(strip_stage_directions(event.get("text", "")).split())
        for event in show_data.get("events", [])
    )


def make_engine(engine_name: str, **kwargs):
    """Build a TTS engine by name.

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

    Args:
        args: Parsed namespace from _render_parse_args().

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
        print(
            f"Warning: could not initialise MusicGenEngine ({exc}). "
            "Continuing without music generation.",
            file=sys.stderr,
        )
        return None


def _render_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the render pipeline.

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
    )

    # ── Required ─────────────────────────────────────────────────────
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the show JSON file.",
    )

    # ── Engine selection ─────────────────────────────────────────────
    parser.add_argument(
        "--engine", "-e",
        type=str,
        default="both",
        choices=["kokoro", "piper", "both"],
        help="Which TTS engine to use (default: both).",
    )

    # ── Output ───────────────────────────────────────────────────────
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

    # ── Voice control ────────────────────────────────────────────────
    parser.add_argument(
        "--voice-map",
        type=str,
        default=None,
        help='JSON string mapping speaker names to voice IDs.',
    )

    parser.add_argument(
        "--voice-policy",
        type=str,
        default="auto",
        choices=["auto", "matched"],
        dest="voice_policy",
        help="How to assign TTS voices (default: auto).",
    )

    # ── Engine-specific options ──────────────────────────────────────
    parser.add_argument(
        "--model-dir",
        type=str,
        default="piper-voices",
        help="Piper model directory (default: piper-voices).",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID for Kokoro.",
    )

    # ── Flags ────────────────────────────────────────────────────────
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

    parser.add_argument(
        "--no-log-audio-duration",
        action="store_true",
        default=False,
        help="Disable logging actual audio duration to the catalog.",
    )

    parser.add_argument(
        "--tts-workers",
        type=int,
        default=1,
        help="Number of TTS worker threads (default: 1, sequential).",
    )

    parser.add_argument(
        "--no-strip-directions",
        action="store_false",
        default=True,
        dest="strip_directions",
        help="Disable stripping of parenthetical stage directions.",
    )

    parser.add_argument(
        "--no-whisper-voices",
        action="store_true",
        default=False,
        dest="no_whisper_voices",
        help="Exclude airy/whispery voices from voice assignment.",
    )

    # ── Music generation ─────────────────────────────────────────────
    parser.add_argument(
        "--music",
        action="store_true",
        default=False,
        dest="music",
        help="Enable generative music for bumper events.",
    )
    parser.add_argument(
        "--music-style",
        type=str,
        default="",
        dest="music_style",
        metavar="DESCRIPTION",
        help="Default music style/genre.",
    )
    parser.add_argument(
        "--music-model",
        type=str,
        default="small",
        dest="music_model",
        choices=["small", "medium", "large"],
        metavar="{small,medium,large}",
        help="MusicGen model size (default: small).",
    )
    parser.add_argument(
        "--music-duration",
        type=float,
        default=8.0,
        dest="music_duration",
        metavar="SECONDS",
        help="Duration in seconds of each generated music clip (default: 8.0).",
    )
    parser.add_argument(
        "--music-volume",
        type=float,
        default=0.0,
        dest="music_volume",
        metavar="DB",
        help=(
            "Adjust music level relative to speech in dB. Negative = quieter "
            "(e.g. -6). Default 0 — music is already normalised to "
            "-28 dBFS, 8 dB below speech."
        ),
    )
    parser.add_argument(
        "--series-id",
        type=str,
        default=None,
        dest="series_id",
        help="Series identifier for multi-episode voice locking.",
    )

    return parser.parse_args(argv)


def run_render(argv: list[str] | None = None) -> "RenderResult":
    """Run the render pipeline: JSON → TTS → audio file.

    Extracted from render_show.main() so it can be called from both
    render_show.py (standalone CLI shim) and run.py (master pipeline)
    without round-tripping through a subprocess.

    Args:
        argv: CLI arguments. Defaults to sys.argv[1:] if None.
    """
    from vox_romana.tts.engine import VoiceConfig
    from vox_romana.tts.renderer import RenderResult, TTSRenderer

    args = _render_parse_args(argv)

    # ── Validate input ───────────────────────────────────────────────
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    # ── Determine output path ────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
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
        _render_logger.warning("VoiceRegistry unavailable, voice locking disabled: %s", exc)

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
        music_volume_db=args.music_volume,
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
    # to the simpler gender-based round-robin assignment.
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

    return result

"""Audition mode runner: pre-production casting for series production.

The audition stage sits BEFORE generation in a series workflow.  It
generates candidate assets (voices, music beds, visual styles), lets
the user pick which ones to greenlight, and writes the chosen set into
the series asset pool.  Once approved, those assets are reused
automatically by every subsequent episode of that series.

Audition is OPTIONAL.  A series can be produced without ever running an
audition — the pool can also be filled manually via
``python run.py --mode series approve`` with a pre-existing asset path.
The audition runner exists for the common case where you want to
explore options before committing.

Subtypes (selected via ``--audition-type``):

    voice    Generate short TTS clips for N candidate voices.  When
             ``--speaker NAME`` is given, the clips are labelled with
             that character so you can tell at a glance which voice
             plays who.

    music    Generate N short MusicGen clips from ``--music-brief
             TEXT``.  Useful for finding the right intro/outro bed for
             a podcast or news show.

    visual   (Stub for v1)  Will generate N sample stills via the
             diffusion image generator.  Currently logs a warning and
             writes an empty session.

After generating the candidates the runner records the session in the
``auditions`` table (see :class:`Catalog`) and offers three approval
modes:

    --approve-all   Approve every candidate into the series pool with
                    no prompts.  Requires ``--series SERIES_ID``.
    --no-approve    Log the session but skip approval entirely so the
                    user can review the clips and approve later via
                    ``python run.py --mode series approve``.
    (default)       Print the candidate table and exit.  The user
                    approves selected candidates manually with
                    ``--mode series approve --audition-id …``.

Why a separate module rather than inline in run.py?  Same reason as
``cli/animation_runner.py`` and ``cli/analysis.py``: keeps the dispatch
file thin, makes the runner unit-testable without driving argparse,
and defers heavy imports (TTS engines, diffusers) until they're needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any


# Default short scripts used to drive each audition type.  Picked to
# exercise vowels + sibilants + a brief pause without taking long to
# render — five seconds per voice is plenty to judge timbre.
_DEFAULT_VOICE_SCRIPT = (
    "This is an audition take. The quick brown fox jumps over the lazy "
    "dog while the rain in Spain stays mainly on the plain."
)

# A small but representative slate of Kokoro voice IDs covering both
# American + British accents and both genders.  Used when the caller
# doesn't supply --audition-count or supplies a count larger than the
# slate length (we just take the first N).
_DEFAULT_VOICE_SLATE: tuple[str, ...] = (
    "af_heart",    # American female, warm
    "am_adam",     # American male, neutral
    "bf_emma",     # British female, crisp
    "bm_george",   # British male, gravelly
    "af_sky",      # American female, airy
    "am_michael",  # American male, deep
)


def _audition_dir(session_id: str) -> Path:
    """Return the on-disk directory for a given audition session."""
    from vox_romana.core.paths import OUTPUT_DIR
    out = OUTPUT_DIR / "auditions" / session_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_wav(path: Path, audio_bytes: bytes) -> None:
    """Write WAV bytes to disk, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio_bytes)


def _audition_voices(
    args: argparse.Namespace,
    *,
    session_id: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Generate voice candidate clips and return their metadata.

    For each candidate voice ID, synthesizes a short script via
    :class:`KokoroEngine` and writes a WAV next to the session
    directory.  Returns one dict per clip ready to be stored as JSON in
    the ``auditions.results_json`` column.

    Args:
        args:        Parsed CLI namespace.  Reads ``args.speaker``
                     (optional character name), ``args.audition_count``
                     (number of voices to try), ``args.quiet``.
        session_id:  Stable session identifier (used for filenames).
        output_dir:  Directory to write WAV clips into.

    Returns:
        List of result dicts:
        ``{"label", "asset_id", "character_name", "path"}``.
    """
    from vox_romana.tts.engines.kokoro_engine import KokoroEngine

    speaker = getattr(args, "speaker", None) or "Audition"
    count = max(1, int(getattr(args, "audition_count", None) or 4))
    slate = list(_DEFAULT_VOICE_SLATE[:count])

    quiet = bool(getattr(args, "quiet", False))
    engine = KokoroEngine()

    script = _DEFAULT_VOICE_SCRIPT
    if speaker and speaker != "Audition":
        script = f"Hello, my name is {speaker}. {script}"

    results: list[dict[str, Any]] = []
    for voice_id in slate:
        if not quiet:
            print(f"  Synthesizing {voice_id} for {speaker}...", flush=True)
        try:
            audio = engine.synthesize(script, voice=voice_id)
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                print(f"    skipped ({exc!r})", file=sys.stderr)
            continue
        clip_path = output_dir / "voice" / f"{speaker}_{voice_id}.wav"
        _write_wav(clip_path, audio.audio)
        results.append({
            "label": voice_id,
            "asset_id": voice_id,
            "character_name": speaker if speaker != "Audition" else None,
            "path": str(clip_path),
        })
    return results


def _audition_music(
    args: argparse.Namespace,
    *,
    session_id: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Generate music candidate clips and return their metadata.

    Calls :class:`MusicGenEngine` ``audition_count`` times with the
    user-supplied brief.  Each call produces an independent clip so the
    user can pick the best take.

    Args:
        args:        Parsed CLI namespace.  Reads ``args.music_brief``,
                     ``args.audition_count``, ``args.music_model``,
                     ``args.music_duration``, ``args.quiet``.
        session_id:  Stable session identifier (used for filenames).
        output_dir:  Directory to write WAV clips into.

    Returns:
        List of result dicts:
        ``{"label", "asset_id", "path"}``.
    """
    from vox_romana.core.paths import HF_CACHE_DIR
    from vox_romana.tts.music_engine import MODEL_ALIASES, MusicGenEngine

    brief = getattr(args, "music_brief", None)
    if not brief:
        print(
            "Error: --music-brief TEXT is required for music auditions.",
            file=sys.stderr,
        )
        sys.exit(1)

    count = max(1, int(getattr(args, "audition_count", None) or 3))
    duration = float(getattr(args, "music_duration", None) or 8.0)
    model_alias = getattr(args, "music_model", None) or "small"
    model_name = MODEL_ALIASES.get(model_alias, model_alias)
    quiet = bool(getattr(args, "quiet", False))

    engine = MusicGenEngine(model_name=model_name, cache_dir=str(HF_CACHE_DIR))

    results: list[dict[str, Any]] = []
    for i in range(count):
        label = f"take_{i + 1}"
        if not quiet:
            print(f"  Generating music take {i + 1}/{count}...", flush=True)
        try:
            audio = engine.generate(brief, duration=duration)
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                print(f"    skipped ({exc!r})", file=sys.stderr)
            continue
        clip_path = output_dir / "music" / f"{label}.wav"
        _write_wav(clip_path, audio.audio)
        results.append({
            "label": label,
            "asset_id": label,
            "path": str(clip_path),
        })
    return results


def _audition_visual(
    args: argparse.Namespace,
    *,
    session_id: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Stub for visual auditions — not yet wired to the diffuser.

    The image generator currently runs through the animation pipeline's
    ``generate_for_plan`` entry point, which expects a full
    :class:`AnimationPlan`.  Wiring a standalone N-stills entry will
    come in a later shot; for now we log a warning and return an empty
    result list so the rest of the audition flow stays consistent.
    """
    quiet = bool(getattr(args, "quiet", False))
    if not quiet:
        print(
            "  Visual auditions are not yet implemented.  "
            "Use --mode video for visual style exploration.",
            file=sys.stderr,
        )
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Approval logic
# ─────────────────────────────────────────────────────────────────────────────


def _print_results_table(
    results: list[dict[str, Any]],
    *,
    audition_type: str,
) -> None:
    """Render the candidate slate as a small text table on stdout."""
    if not results:
        print("  (no candidates generated)")
        return
    print(f"\n  Candidates ({audition_type}, {len(results)}):")
    print(f"  {'#':>3}  {'Label':<24}  Path")
    print(f"  {'-'*3}  {'-'*24}  {'-'*40}")
    for i, r in enumerate(results, start=1):
        label = r.get("label") or r.get("asset_id") or "?"
        path = r.get("path") or ""
        if len(path) > 60:
            path = "…" + path[-57:]
        print(f"  {i:>3}  {label:<24}  {path}")


def _maybe_approve(
    args: argparse.Namespace,
    *,
    session_id: str,
    results: list[dict[str, Any]],
    catalog,
) -> list[int]:
    """Apply the approval policy chosen by the user.

    Returns the list of approved asset PKs (empty when nothing was
    approved or no series was given).  Will exit on policy violation
    (e.g. ``--approve-all`` without ``--series``).
    """
    if not results:
        return []

    approve_all = bool(getattr(args, "approve_all", False))
    no_approve = bool(getattr(args, "no_approve", False))
    series_id = getattr(args, "series", None)

    if approve_all and no_approve:
        print(
            "Error: --approve-all and --no-approve are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(1)

    if approve_all:
        if not series_id:
            print(
                "Error: --approve-all requires --series SERIES_ID.",
                file=sys.stderr,
            )
            sys.exit(1)
        pks = catalog.approve_all_from_audition(session_id, series_id=series_id)
        if not getattr(args, "quiet", False):
            print(f"  Approved {len(pks)} asset(s) into series {series_id}.")
        return pks

    if no_approve:
        if not getattr(args, "quiet", False):
            print(
                "  Logged audition session — no assets approved.\n"
                f"  Review later with: python run.py --mode series approve "
                f"<SERIES_ID> --audition-id {session_id}"
            )
        return []

    # Default: print instructions for manual approval, no prompting.
    if not getattr(args, "quiet", False):
        ref = f"--series {series_id} " if series_id else "<SERIES_ID> "
        print(
            "\n  To approve assets into a series pool, run:\n"
            f"    python run.py --mode series approve {ref}"
            f"--audition-id {session_id} --approve-all\n"
            "  Or approve individually with --asset-id ASSET_ID."
        )
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry
# ─────────────────────────────────────────────────────────────────────────────


def run_audition(args: argparse.Namespace) -> Path | None:
    """Entry point for ``--mode audition``.

    Generates a slate of candidate assets, writes the session to the
    ``auditions`` catalog table, and applies the user's approval policy.

    Args:
        args: Parsed namespace.  Required: ``args.audition_type`` (one
              of ``voice``, ``music``, ``visual``).  Optional: ``args.series``,
              ``args.speaker``, ``args.music_brief``, ``args.audition_count``,
              ``args.approve_all``, ``args.no_approve``, ``args.quiet``.

    Returns:
        Path to the audition session directory, or ``None`` when nothing
        was generated.
    """
    from vox_romana.core.catalog import Catalog

    audition_type = getattr(args, "audition_type", None)
    if audition_type not in ("voice", "music", "visual"):
        print(
            "Error: --audition-type must be one of: voice, music, visual.",
            file=sys.stderr,
        )
        sys.exit(1)

    series_id = getattr(args, "series", None)
    quiet = bool(getattr(args, "quiet", False))

    session_id = uuid.uuid4().hex[:12]
    output_dir = _audition_dir(session_id)

    if not quiet:
        print(f"\n{'═' * 60}")
        print(f"  AUDITION — {audition_type}")
        print(f"{'═' * 60}")
        print(f"  Session ID: {session_id}")
        if series_id:
            print(f"  Series:     {series_id}")
        print(f"  Output:     {output_dir}")
        print()

    t0 = time.time()
    if audition_type == "voice":
        brief = (
            getattr(args, "speaker", None)
            or "Voice audition (default slate)"
        )
        results = _audition_voices(
            args, session_id=session_id, output_dir=output_dir,
        )
    elif audition_type == "music":
        brief = getattr(args, "music_brief", None) or ""
        results = _audition_music(
            args, session_id=session_id, output_dir=output_dir,
        )
    else:  # visual
        brief = getattr(args, "visual_brief", None) or ""
        results = _audition_visual(
            args, session_id=session_id, output_dir=output_dir,
        )
    elapsed = time.time() - t0

    # Persist the session before approval so it survives even if the
    # approval step fails or the user aborts.
    with Catalog() as catalog:
        catalog.log_audition(
            session_id=session_id,
            series_id=series_id,
            audition_type=audition_type,
            brief=brief,
            results=results,
        )

        if not quiet:
            print(f"\n  Generated {len(results)} candidate(s) in {elapsed:.1f}s")
            _print_results_table(results, audition_type=audition_type)

        # Write a manifest for human inspection.
        manifest = {
            "session_id": session_id,
            "audition_type": audition_type,
            "series_id": series_id,
            "brief": brief,
            "results": results,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )

        _maybe_approve(
            args,
            session_id=session_id,
            results=results,
            catalog=catalog,
        )

    if quiet:
        # In quiet mode, surface only the session_id so scripts can
        # capture it.
        print(session_id)

    return output_dir

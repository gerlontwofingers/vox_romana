"""High-level orchestrator for YouTube episode packaging.

Combines SRT caption generation (``captions.py``) and LLM show-notes
generation (``notes_agent.py``) into a single ``package_episode()`` call.

Example::

    from vox_romana.package import package_episode

    result = package_episode(
        json_path=Path("output/text/debates/ubi.json"),
        event_durations=render_result.event_durations,
        generate_notes=True,
        client=DeepSeekClient(),
    )
    print(result.srt_path)    # output/text/debates/ubi.srt
    print(result.notes_path)  # output/text/debates/ubi_show_notes.txt
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .captions import make_srt
from .notes_agent import extract_chapters, generate_show_notes

logger = logging.getLogger(__name__)


@dataclass
class PackageResult:
    """Result of packaging a rendered episode for YouTube.

    Attributes:
        srt_path:   Path to the ``.srt`` caption file, or ``None`` if
                    SRT generation failed.
        notes_path: Path to the ``_show_notes.txt`` file, or ``None``
                    if notes generation was skipped or failed.
    """

    srt_path: Path | None = None
    notes_path: Path | None = None


def package_episode(
    json_path: Path,
    event_durations: dict[int, float] | None,
    dest_dir: Path | None = None,
    *,
    generate_notes: bool = True,
    client=None,
) -> PackageResult:
    """Generate SRT captions and YouTube show notes for a rendered episode.

    This is the primary entry point for the packaging pipeline.  It reads
    the show JSON, calls :func:`make_srt` for captions and (optionally)
    :func:`generate_show_notes` via the LLM for the description.

    Args:
        json_path: Path to the show JSON file.
        event_durations: Per-event audio durations in seconds, keyed by
            event index.  Pass ``None`` or ``{}`` to use fallback timing
            (3 s per spoken event) — captions will be approximately timed.
        dest_dir: Directory for output files.  Defaults to the same
            directory as ``json_path``.
        generate_notes: When ``True`` (default), call the LLM to generate
            the YouTube description.  Silently skipped when ``client``
            is ``None``.
        client: LLM client with ``.chat(messages, **kwargs) -> str``.
            Required for notes generation; ignored when
            ``generate_notes=False``.

    Returns:
        :class:`PackageResult` with paths to the written files.

    Raises:
        FileNotFoundError: If ``json_path`` does not exist.
        json.JSONDecodeError: If the JSON is malformed.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Show JSON not found: {json_path}")

    durations: dict[int, float] = event_durations or {}
    out_dir = Path(dest_dir) if dest_dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = json_path.stem

    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    events = data.get("events", [])
    result = PackageResult()

    # ── SRT captions ─────────────────────────────────────────────────────
    srt_path = out_dir / f"{stem}.srt"
    try:
        srt_content = make_srt(events, durations)
        srt_path.write_text(srt_content, encoding="utf-8")
        result.srt_path = srt_path
        logger.info("SRT captions written → %s", srt_path)
    except Exception as exc:
        logger.error("SRT generation failed: %s", exc)

    # ── Show notes ────────────────────────────────────────────────────────
    if generate_notes:
        if client is None:
            logger.warning(
                "generate_notes=True but no LLM client supplied — "
                "show notes skipped.  Pass client= or set DEEPSEEK_API_KEY."
            )
        else:
            try:
                chapters = extract_chapters(data, durations)
                notes = generate_show_notes(data, chapters, client=client)
                if notes and notes.strip():
                    notes_path = out_dir / f"{stem}_show_notes.txt"
                    notes_path.write_text(notes.strip(), encoding="utf-8")
                    result.notes_path = notes_path
                    logger.info("Show notes written → %s", notes_path)
            except Exception as exc:
                logger.error("Show notes generation failed: %s", exc)

    return result

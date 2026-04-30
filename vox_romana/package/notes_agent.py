"""LLM agent: generate YouTube show notes with chapter timestamps.

Produces a ~200-word YouTube video description that includes:
- A 1-2 sentence hook
- 3-5 key takeaway bullets
- Chapter list with timestamps in ``0:00`` format
- A call to action

Chapter timestamps are derived by accumulating per-event audio durations
(from ``RenderResult.event_durations``) and noting when the show transitions
between segments, chunks, or chapters.

Typical usage::

    from vox_romana.package.notes_agent import extract_chapters, generate_show_notes

    chapters = extract_chapters(data, render_result.event_durations)
    notes = generate_show_notes(data, chapters, client=DeepSeekClient())
    Path("my_show_notes.txt").write_text(notes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# Event kinds that carry spoken dialogue and advance the audio clock.
_SPOKEN_KINDS: frozenset[str] = frozenset({"utterance", "narration", "interrupt"})

# Timing fallbacks (same as captions.py, replicated to avoid circular import).
_FALLBACK_SPOKEN_SECS: float = 3.0
_FALLBACK_NONSPEECH_SECS: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# ChapterMark
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChapterMark:
    """One YouTube chapter marker.

    Attributes:
        timestamp: YouTube chapter format (``"0:00"`` or ``"1:23:45"``).
        title:     Human-readable section title.
    """

    timestamp: str
    title: str


# ─────────────────────────────────────────────────────────────────────────────
# Chapter extraction
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_youtube_ts(total_secs: float) -> str:
    """Convert seconds to YouTube chapter timestamp format.

    Examples:
        0 s   → ``"0:00"``
        90 s  → ``"1:30"``
        3723 s → ``"1:02:03"``
    """
    total = max(0, int(total_secs))
    secs = total % 60
    mins = (total // 60) % 60
    hours = total // 3600
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def _build_segment_title_map(data: dict) -> dict[str, str]:
    """Build a mapping of segment/chunk id → human-readable title.

    Handles all three show layouts:
    - ``data["segments"]`` — news / podcast (keyed by ``segment_id``)
    - ``data["chunks"]``   — debates (keyed by ``chunk_id`` or ``chunk_order``)
    - no structure         — returns empty dict (caller handles fallback)
    """
    titles: dict[str, str] = {}

    for seg in data.get("segments", []):
        sid = str(seg.get("segment_id") or "")
        if sid:
            label = (
                seg.get("headline")
                or seg.get("title")
                or sid
            )
            titles[sid] = str(label).strip()

    for chunk in data.get("chunks", []):
        cid = str(chunk.get("chunk_id") or chunk.get("chunk_order") or "")
        if cid:
            label = chunk.get("title") or f"Part {cid}"
            titles[cid] = str(label).strip()

    return titles


def extract_chapters(
    data: dict,
    event_durations: dict[int, float],
    *,
    fallback_secs: float = _FALLBACK_SPOKEN_SECS,
) -> list[ChapterMark]:
    """Extract YouTube chapter marks from show JSON + timing data.

    Walks the flat ``data["events"]`` list, accumulating time from
    ``event_durations``.  A new chapter mark is emitted the first time
    an event belonging to a new ``segment_id`` (or ``chunk_id``) is
    encountered.

    YouTube requires the first chapter to start at ``0:00``.  This
    function guarantees that by inserting an ``"Introduction"`` chapter
    at the beginning when needed.

    Args:
        data: Parsed show JSON dict.
        event_durations: Per-event audio durations in seconds (index-keyed).
        fallback_secs: Duration used for events absent from
            ``event_durations`` (default 3.0 s for spoken events).

    Returns:
        List of :class:`ChapterMark` in presentation order.  Always
        starts at ``"0:00"``.  Returns a single ``["0:00 – <title>"]``
        entry for shows with no segment / chunk structure.
    """
    events = data.get("events", [])
    seg_titles = _build_segment_title_map(data)

    # No segment structure → single chapter at 0:00.
    if not seg_titles:
        title = (data.get("title") or "Introduction").strip()
        return [ChapterMark(timestamp="0:00", title=title)]

    chapters: list[ChapterMark] = []
    seen_ids: set[str] = set()
    current_time = 0.0

    for i, event in enumerate(events):
        kind = (event.get("event_kind") or "utterance").lower()
        duration = event_durations.get(
            i,
            fallback_secs if kind in _SPOKEN_KINDS else _FALLBACK_NONSPEECH_SECS,
        )
        duration = max(duration, 0.0)

        seg_id = str(event.get("segment_id") or "")
        if seg_id and seg_id not in seen_ids:
            seen_ids.add(seg_id)
            label = seg_titles.get(seg_id, seg_id)
            chapters.append(
                ChapterMark(timestamp=_fmt_youtube_ts(current_time), title=label)
            )

        current_time += duration

    # YouTube requires first chapter at exactly "0:00".
    if chapters and chapters[0].timestamp != "0:00":
        intro = (data.get("title") or "Introduction").strip()
        chapters.insert(0, ChapterMark(timestamp="0:00", title=intro))

    # Deduplicate consecutive timestamps (can happen when opening events
    # are non-spoken and time hasn't advanced yet).
    deduped: list[ChapterMark] = []
    seen_ts: set[str] = set()
    for ch in chapters:
        if ch.timestamp not in seen_ts:
            seen_ts.add(ch.timestamp)
            deduped.append(ch)

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# LLM notes generator
# ─────────────────────────────────────────────────────────────────────────────


class _ChatClient(Protocol):
    """Structural interface for an LLM client."""

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        ...


_SYSTEM_PROMPT = """\
You are a YouTube content writer specializing in audio/video show descriptions.

Given a show summary and chapter list, write a YouTube video description that:
1. Opens with a 1-2 sentence hook (the most compelling reason to watch).
2. Lists 3-5 key takeaways as bullet points, each under 15 words.
3. Includes the full chapter list VERBATIM — copy the timestamps and titles
   exactly as given. Do NOT add extra commentary between chapters.
4. Ends with a 1-sentence call to action (like/subscribe/comment).

Target length: ~200 words total.
Return ONLY the description text — no markdown wrappers, no title, no preamble.
The chapter list must appear after the bullet points on its own line.
"""


def generate_show_notes(
    data: dict,
    chapters: list[ChapterMark],
    *,
    client: _ChatClient,
    temperature: float = 0.7,
) -> str:
    """Call an LLM to generate a YouTube show description with chapters.

    Args:
        data: Parsed show JSON dict (for title, topic, show_type, tone).
        chapters: Pre-computed chapter marks (from :func:`extract_chapters`).
        client: LLM client with a ``.chat()`` method.
        temperature: Sampling temperature (default 0.7).

    Returns:
        YouTube description string, approximately 200 words, ready to
        paste into the YouTube Studio description field.
    """
    title = (data.get("title") or "Untitled").strip()
    show_type = (data.get("show_type") or _detect_show_type(data)).strip()
    topic = (data.get("topic") or data.get("theme") or "").strip()
    tone = (data.get("tone") or "").strip()

    chapter_text = "\n".join(
        f"{ch.timestamp} {ch.title}" for ch in chapters
    )

    user_msg = (
        f"Show title: {title}\n"
        f"Show type: {show_type}\n"
        + (f"Topic: {topic}\n" if topic else "")
        + (f"Tone: {tone}\n" if tone else "")
        + f"\nChapters:\n{chapter_text}\n\n"
        "Write the YouTube video description now."
    )

    return client.chat(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
    )


def _detect_show_type(data: dict) -> str:
    """Infer show type from JSON structure when show_type field is absent."""
    if data.get("segments"):
        return "news" if data.get("anchors") else "podcast"
    if data.get("chapters"):
        return "audiobook"
    return "debate"

"""Generate SRT captions from a show JSON event list + per-event timing data.

SRT (SubRip Text) is the most universally accepted caption format for
YouTube Studio.  Each spoken event becomes one subtitle entry.  Non-spoken
events (break, bumper, transition) advance the running clock but do not
produce caption entries.

Typical usage::

    events = data["events"]           # from the show JSON
    durations = render_result.event_durations   # from TTSRenderer.render()
    srt = make_srt(events, durations)
    Path("my_show.srt").write_text(srt, encoding="utf-8")
"""

from __future__ import annotations

# Event kinds that carry spoken dialogue we want to caption.
_SPOKEN_KINDS: frozenset[str] = frozenset({"utterance", "narration", "interrupt"})

# Default duration in seconds when event_durations has no entry for an index.
# Used for spoken events that were skipped or had synthesis errors.
_FALLBACK_SPOKEN_SECS: float = 3.0

# Non-spoken events (breaks, bumpers) still advance the clock.
# Use a conservative gap rather than 0 so timestamps don't bunch up.
_FALLBACK_NONSPEECH_SECS: float = 0.5


def _srt_ts(total_secs: float) -> str:
    """Convert seconds to SRT timestamp format ``HH:MM:SS,mmm``.

    Args:
        total_secs: Time in seconds (may be fractional).

    Returns:
        String like ``"00:01:23,456"``.
    """
    ms = round(total_secs * 1000)
    ms = max(ms, 0)
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def make_srt(
    events: list[dict],
    event_durations: dict[int, float],
    *,
    fallback_secs: float = _FALLBACK_SPOKEN_SECS,
    max_caption_chars: int = 80,
) -> str:
    """Build a complete SRT caption string for a rendered show.

    Args:
        events: Flat list of event dicts from ``data["events"]``.
        event_durations: Per-event audio durations in seconds, keyed by
            event index (matching the order of ``events``).  This is
            ``RenderResult.event_durations`` from a completed render.
        fallback_secs: Duration used when ``event_durations`` has no entry
            for a spoken event (default 3.0 s).
        max_caption_chars: Wrap captions longer than this many characters
            at a word boundary into two lines (default 80).

    Returns:
        Complete SRT string ready to write to ``{stem}.srt``.  Empty
        string if there are no captionable events.
    """
    blocks: list[str] = []
    seq = 0
    current_time = 0.0

    for i, event in enumerate(events):
        kind = (event.get("event_kind") or "utterance").lower()
        text = (event.get("text") or "").strip()

        duration = event_durations.get(
            i,
            fallback_secs if kind in _SPOKEN_KINDS else _FALLBACK_NONSPEECH_SECS,
        )
        duration = max(duration, 0.1)  # SRT requires end > start

        if kind in _SPOKEN_KINDS and text:
            seq += 1
            start = _srt_ts(current_time)
            end = _srt_ts(current_time + duration)

            speaker = (event.get("speaker_name") or "").strip()
            line = f"{speaker}: {text}" if speaker else text

            # Wrap long lines at a word boundary.
            if len(line) > max_caption_chars:
                words = line.split()
                mid = max(1, len(words) // 2)
                line = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

            blocks.append(f"{seq}\n{start} --> {end}\n{line}")

        current_time += duration

    return "\n\n".join(blocks) + ("\n" if blocks else "")

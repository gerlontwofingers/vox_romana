from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentResponse:
    prompt: str
    response: str
    tag: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# AudioEvent — the universal event schema for all content types
# ─────────────────────────────────────────────────────────────────────────────
# Every show type (debates, news, podcasts, audiobooks) produces a sequence
# of these. The TTS renderer and audio mixer consume them generically.
#
# DebateEvent was the original name — AudioEvent is the same dataclass,
# promoted to core so all show types share one schema.

@dataclass(slots=True)
class AudioEvent:
    """A single spoken (or non-spoken) event in any content type.

    This is the universal unit of content that flows from conductors
    through parsers into the TTS renderer. Every show type produces
    AudioEvents; the renderer doesn't need to know what kind of show
    generated them.

    Attributes:
        global_order:          Position in the full event sequence.
        local_order:           Position within a segment/chunk.
        speaker_role:          Function: "participant", "moderator", "anchor",
                               "reporter", "narrator", "host", etc.
        speaker_name:          Display name of the speaker.
        target:                Who they're addressing (if specific).
        text:                  The spoken content.
        tone:                  Emotional tones (e.g., ("aggressive", "rising")).
        volume:                "loud", "normal", "soft", etc.
        pace:                  "fast", "measured", "slow", etc.
        pause_before:          Pause hint before this event.
        pause_after:           Pause hint after this event.
        interruption:          Does this event interrupt the previous speaker?
        nonverbal:             Stage directions (("sighs",), ("applause",)).
        event_kind:            "utterance", "narration", "break", "bumper",
                               "transition", "interrupt".
        overlap_with_previous: Should this overlap with the prior event?
        completion_status:     LLM generation status (for debugging).
        segment_id:            Which segment this belongs to (None for
                               non-segmented content like simple debates).
        raw_fields:            Any extra fields from the source data.
    """

    global_order: int
    local_order: int
    speaker_role: str
    speaker_name: str
    target: str | None
    text: str | None
    tone: tuple[str, ...] = ()
    volume: str | None = None
    pace: str | None = None
    pause_before: str | None = None
    pause_after: str | None = None
    interruption: bool = False
    nonverbal: tuple[str, ...] = ()
    event_kind: str = "utterance"
    overlap_with_previous: bool = False
    completion_status: str | None = None
    segment_id: str | None = None
    raw_fields: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tone"] = list(self.tone)
        data["nonverbal"] = list(self.nonverbal)
        return data


# ─────────────────────────────────────────────────────────────────────────────
# FactCard — a verified, citable data point for grounding LLM content
# ─────────────────────────────────────────────────────────────────────────────
# Produced by the DataResearcher agent (or manually). Conductors and
# bootstrap agents receive FactCards as grounding material — they can
# cite them but not invent numbers.  Stored in the catalog via log_study().

@dataclass(frozen=True)
class FactCard:
    """A single verified data point that agents can cite.

    Attributes:
        claim:      Plain English statement (e.g. "Employment rose 4.8%").
        value:      The numeric or categorical value (e.g. 4.8, "increased").
        unit:       Unit of measurement (e.g. "percent", "USD", "count").
        source:     Where this fact comes from (dataset name, URL, org).
        methodology: Brief note on how the stat was derived.
        confidence:  Qualitative confidence level.
    """

    claim: str
    value: str | int | float | None = None
    unit: str = ""
    source: str = ""
    methodology: str = ""
    confidence: str = "verified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

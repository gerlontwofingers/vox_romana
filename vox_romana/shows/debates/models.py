"""Data models for the debate pipeline.

Three model tiers:

1. :class:`DebateSegment` — a parse artifact covering one contiguous segment
   of a raw LLM output (one ``template_id`` + ordered ``events``).
2. :class:`ParsedDebateDocument` — a full LLM output split into its segments,
   as produced by a single parser pass.
3. :class:`MergedDebate` — the canonical, publication-ready view produced by
   :class:`~.merge.DebateDocumentMerger` after reconciling multiple parser
   outputs into a single coherent transcript.

:data:`DebateEvent` is a re-exported alias for
:class:`~vox_romana.core.models.AudioEvent` — it exists for readability
within the debates package; the canonical type lives in ``core``.

.. note::
   ``DebateChunk`` is preserved as a deprecated alias for :class:`DebateSegment`
   to maintain backwards compatibility with older external callers. New code
   should use ``DebateSegment``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ...core.models import AudioEvent

# Intentional re-export: type alias for readability within the debates package.
# AudioEvent is the canonical unified type in vox_romana.core.models.
DebateEvent = AudioEvent


@dataclass(slots=True)
class DebateSegment:
    """A single parsed segment of a debate, produced by one parser.

    A segment corresponds to one contiguous block of LLM-generated output that
    a parser has successfully sliced into structured :class:`DebateEvent` objects.
    Multiple segments produced from the same raw output are ordered by
    ``chunk_order`` and merged by :class:`~.merge.DebateDocumentMerger` into a
    :class:`MergedDebate`.

    Attributes:
        template_id:    Which parser template produced this segment.
        debate_id:      Unique ID for the containing debate run.
        chunk_order:    1-based position of this segment within the full document.
        is_final_chunk: True if no further segments follow in this document.
        participants:   Names of debating participants present in this segment.
        events:         Ordered sequence of :class:`DebateEvent` objects.
    """

    template_id: str
    debate_id: str
    title: str
    chunk_order: int
    is_final_chunk: bool
    participants: tuple[str, ...]
    moderators: tuple[str, ...] = ()
    hosts: tuple[str, ...] = ()
    audience_present: bool = False
    audience_participation: bool | None = None
    topic: str | None = None
    debate_format: str | None = None
    setting: str | None = None
    moderator_mode: str | None = None
    continuity_notes: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    events: tuple[DebateEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["participants"] = list(self.participants)
        data["moderators"] = list(self.moderators)
        data["hosts"] = list(self.hosts)
        data["events"] = [event.to_dict() for event in self.events]
        return data


@dataclass(slots=True)
class ParsedDebateDocument:
    """The full output of one parser run over a raw debate text.

    Produced by :class:`~.pipeline.DebateParserRegistry` when a parser
    successfully processes the raw LLM output. Multiple
    ``ParsedDebateDocument`` instances (one per parser) are fed to
    :class:`~.merge.DebateDocumentMerger` to produce a single
    :class:`MergedDebate`.

    Attributes:
        template_id: Which parser template produced this document.
        parser_name: Human-readable parser identifier.
        raw_text:    The full, unmodified LLM text this was parsed from.
        chunks:      Ordered tuple of :class:`DebateSegment` instances.
    """

    template_id: str
    parser_name: str
    raw_text: str
    chunks: tuple[DebateSegment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "parser_name": self.parser_name,
            "raw_text": self.raw_text,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


@dataclass(slots=True)
class MergedDebate:
    """The canonical, publication-ready debate produced after merge.

    :class:`~.merge.DebateDocumentMerger` reconciles one or more
    :class:`ParsedDebateDocument` instances into a single ``MergedDebate``.
    This is the final structured output handed to
    :class:`~.serializer.DebateSerializer` for saving to disk.

    Attributes:
        template_id:   Parser template that produced the primary source.
        debate_id:     Unique ID for this debate run.
        title:         Human-readable debate title.
        chunks:        All chunks that contributed to this debate.
        events:        Flattened, ordered tuple of all :class:`DebateEvent`
                       objects — the complete spoken transcript.
        parser_names:  Names of all parsers that contributed chunks.
        participants:  Names of all debating participants.
        voice_casting: ``{name: voice_gender}`` for TTS renderer.
        voice_hints:   ``{name: casting_note}`` for the voice catalog matcher.
        tts_styles:    ``{name: tts_style}`` persistent TTS style per speaker.
    """

    template_id: str
    debate_id: str
    title: str
    chunks: tuple[DebateSegment, ...]
    events: tuple[DebateEvent, ...]
    parser_names: tuple[str, ...] = ()
    participants: tuple[str, ...] = ()
    moderators: tuple[str, ...] = ()
    hosts: tuple[str, ...] = ()
    audience_present: bool = False
    audience_participation: bool | None = None
    topic: str | None = None
    debate_format: str | None = None
    setting: str | None = None
    moderator_mode: str | None = None
    continuity_notes: tuple[str, ...] = ()
    source_count: int = 0
    language: str = "en"
    voice_casting: dict[str, str] = field(default_factory=dict)
    voice_hints: dict[str, str] = field(default_factory=dict)
    tts_styles: dict[str, str] = field(default_factory=dict)
    series_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "debate_id": self.debate_id,
            "title": self.title,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "events": [event.to_dict() for event in self.events],
            "parser_names": list(self.parser_names),
            "participants": list(self.participants),
            "moderators": list(self.moderators),
            "hosts": list(self.hosts),
            "audience_present": self.audience_present,
            "audience_participation": self.audience_participation,
            "topic": self.topic,
            "debate_format": self.debate_format,
            "setting": self.setting,
            "moderator_mode": self.moderator_mode,
            "continuity_notes": list(self.continuity_notes),
            "source_count": self.source_count,
            "language": self.language,
            "voice_casting": self.voice_casting,
            "voice_hints": self.voice_hints,
            "tts_styles": self.tts_styles,
            "series_id": self.series_id,
        }


# Backwards-compatibility alias. The class was renamed DebateChunk → DebateSegment
# because "segment" matches the rest of the codebase (NewsSegment, PodcastSegment)
# and the parse-artifact role is one debate "segment", not a generic chunk.
# External callers and older imports continue to work via this alias; new code
# should import DebateSegment.
DebateChunk = DebateSegment

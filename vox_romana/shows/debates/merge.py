from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import DebateChunk, DebateEvent, MergedDebate, ParsedDebateDocument


class DebateMergeError(ValueError):
    """Raised when parsed debate chunks cannot be merged safely."""


class DebateDocumentMerger:
    """Merges parsed debate chunks into durable, ordered debate objects."""

    def merge_document(self, document: ParsedDebateDocument) -> MergedDebate:
        return self.merge_documents((document,))[0]

    def merge_documents(
        self,
        documents: Iterable[ParsedDebateDocument],
    ) -> tuple[MergedDebate, ...]:
        document_list = tuple(documents)
        if not document_list:
            return ()

        grouped_chunks: dict[tuple[str, str], list[DebateChunk]] = defaultdict(list)
        parser_names: dict[tuple[str, str], set[str]] = defaultdict(set)
        source_counts: dict[tuple[str, str], int] = defaultdict(int)

        for document in document_list:
            source_counts[(document.template_id, "__document__")] += 1
            for chunk in document.chunks:
                key = (chunk.template_id, chunk.debate_id)
                grouped_chunks[key].append(chunk)
                parser_names[key].add(document.parser_name)
                source_counts[key] += 1

        merged: list[MergedDebate] = []
        for key in sorted(grouped_chunks, key=lambda item: (item[1], item[0])):
            template_id, debate_id = key
            chunks = tuple(sorted(grouped_chunks[key], key=_chunk_sort_key))
            merged.append(
                self._merge_group(
                    template_id=template_id,
                    debate_id=debate_id,
                    chunks=chunks,
                    parser_names=tuple(sorted(parser_names[key])),
                    source_count=source_counts[key],
                )
            )
        return tuple(merged)

    def _merge_group(
        self,
        *,
        template_id: str,
        debate_id: str,
        chunks: tuple[DebateChunk, ...],
        parser_names: tuple[str, ...],
        source_count: int,
    ) -> MergedDebate:
        if not chunks:
            raise DebateMergeError("Cannot merge an empty debate chunk group.")

        title = _coalesce_scalar(chunks, "title")
        topic = _coalesce_scalar(chunks, "topic")
        debate_format = _coalesce_scalar(chunks, "debate_format")
        setting = _coalesce_scalar(chunks, "setting")
        moderator_mode = _coalesce_scalar(chunks, "moderator_mode")
        audience_present = _coalesce_bool(chunks, "audience_present")
        audience_participation = _coalesce_optional_bool(chunks, "audience_participation")

        continuity_notes = tuple(
            chunk.continuity_notes
            for chunk in chunks
            if chunk.continuity_notes is not None
        )
        participants = _ordered_unique_name_union(chunk.participants for chunk in chunks)
        moderators = _ordered_unique_name_union(chunk.moderators for chunk in chunks)
        events = _merge_events(chunks)

        return MergedDebate(
            template_id=template_id,
            debate_id=debate_id,
            title=title,
            chunks=chunks,
            events=events,
            parser_names=parser_names,
            participants=participants,
            moderators=moderators,
            audience_present=audience_present,
            audience_participation=audience_participation,
            topic=topic,
            debate_format=debate_format,
            setting=setting,
            moderator_mode=moderator_mode,
            continuity_notes=continuity_notes,
            source_count=source_count,
        )


def _merge_events(chunks: tuple[DebateChunk, ...]) -> tuple[DebateEvent, ...]:
    events_by_global_order: dict[int, DebateEvent] = {}
    for chunk in chunks:
        for event in sorted(chunk.events, key=_event_sort_key):
            existing = events_by_global_order.get(event.global_order)
            if existing is None:
                events_by_global_order[event.global_order] = event
                continue
            if existing.to_dict() != event.to_dict():
                raise DebateMergeError(
                    "Conflicting events share GLOBAL order "
                    f"{event.global_order} in debate '{chunk.debate_id}'."
                )
    return tuple(sorted(events_by_global_order.values(), key=_event_sort_key))


def _coalesce_scalar(chunks: tuple[DebateChunk, ...], field_name: str) -> str | None:
    values = [getattr(chunk, field_name) for chunk in chunks if getattr(chunk, field_name) is not None]
    if not values:
        return None
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise DebateMergeError(
                f"Conflicting values for '{field_name}' in debate '{chunks[0].debate_id}'."
            )
    return first


def _coalesce_bool(chunks: tuple[DebateChunk, ...], field_name: str) -> bool:
    values = {getattr(chunk, field_name) for chunk in chunks}
    if len(values) != 1:
        raise DebateMergeError(
            f"Conflicting values for '{field_name}' in debate '{chunks[0].debate_id}'."
        )
    return values.pop()


def _coalesce_optional_bool(chunks: tuple[DebateChunk, ...], field_name: str) -> bool | None:
    values = [getattr(chunk, field_name) for chunk in chunks if getattr(chunk, field_name) is not None]
    if not values:
        return None
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise DebateMergeError(
                f"Conflicting values for '{field_name}' in debate '{chunks[0].debate_id}'."
            )
    return first


def _ordered_unique_name_union(groups: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for names in groups:
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def _chunk_sort_key(chunk: DebateChunk) -> tuple[int, int, int]:
    first_event_order = min((event.global_order for event in chunk.events), default=-1)
    return (chunk.chunk_order, first_event_order, len(chunk.events))


def _event_sort_key(event: DebateEvent) -> tuple[int, int, str, str]:
    return (event.global_order, event.local_order, event.speaker_role, event.speaker_name)

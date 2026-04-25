from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable

from .merge import DebateDocumentMerger
from .models import DebateEvent, DebateSegment, ParsedDebateDocument


class DebateParserError(ValueError):
    """Raised when a debate document cannot be parsed."""


class DebateTemplateParser(ABC):
    template_id: str

    @abstractmethod
    def can_parse(self, text: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse(self, text: str) -> ParsedDebateDocument:
        raise NotImplementedError


class DebateTemplate01Parser(DebateTemplateParser):
    template_id = "template_01"
    _chunk_pattern = re.compile(
        r"=== DEBATE_CHUNK_START ===\n(.*?)\n=== DEBATE_CHUNK_END ===",
        re.DOTALL,
    )
    _utterance_pattern = re.compile(
        r"--- UTTERANCE START ---\n(.*?)\n--- UTTERANCE END ---",
        re.DOTALL,
    )

    def can_parse(self, text: str) -> bool:
        return (
            "TEMPLATE_ID: template_01" in text
            or "DEBATE TEMPLATE 01" in text
            or (
                "=== DEBATE_CHUNK_START ===" in text
                and "--- UTTERANCE START ---" in text
            )
        )

    def parse(self, text: str) -> ParsedDebateDocument:
        chunks = [self._parse_chunk(match.group(1)) for match in self._chunk_pattern.finditer(text)]
        if not chunks:
            raise DebateParserError("No template_01 debate chunks were found.")
        return ParsedDebateDocument(
            template_id=self.template_id,
            parser_name=self.__class__.__name__,
            raw_text=text,
            chunks=tuple(chunks),
        )

    def _parse_chunk(self, chunk_text: str) -> DebateSegment:
        if "--- UTTERANCE START ---" not in chunk_text:
            raise DebateParserError("template_01 chunk is missing utterance blocks.")

        metadata_text = chunk_text.split("--- UTTERANCE START ---", 1)[0]
        metadata = _parse_key_value_block(metadata_text)
        utterances = [
            self._parse_utterance(match.group(1))
            for match in self._utterance_pattern.finditer(chunk_text)
        ]
        if not utterances:
            raise DebateParserError("template_01 chunk must contain at least one utterance.")

        return DebateSegment(
            template_id=metadata.get("TEMPLATE_ID", self.template_id),
            debate_id=_require(metadata, "DEBATE_ID"),
            title=_require(metadata, "DEBATE_TITLE"),
            chunk_order=_parse_int(_require(metadata, "CHUNK_ORDER"), "CHUNK_ORDER"),
            is_final_chunk=_parse_yes_no(_require(metadata, "IS_FINAL_CHUNK"), "IS_FINAL_CHUNK"),
            participants=_split_list(_require(metadata, "PARTICIPANT_NAMES")),
            moderators=_split_list(metadata.get("MODERATOR_NAMES", "NONE")),
            audience_present=_parse_yes_no(_require(metadata, "AUDIENCE_PRESENT"), "AUDIENCE_PRESENT"),
            audience_participation=_parse_optional_yes_no(
                metadata.get("AUDIENCE_PARTICIPATION", "NONE"),
                "AUDIENCE_PARTICIPATION",
            ),
            topic=_normalize_value(metadata.get("DEBATE_TOPIC")),
            debate_format=_normalize_value(metadata.get("DEBATE_FORMAT")),
            setting=_normalize_value(metadata.get("SETTING")),
            moderator_mode=_normalize_value(metadata.get("MODERATOR_MODE")),
            continuity_notes=_normalize_value(metadata.get("NOTES_FOR_CONTINUITY")),
            metadata=metadata,
            events=tuple(utterances),
        )

    def _parse_utterance(self, utterance_text: str) -> DebateEvent:
        fields = _parse_key_value_block(utterance_text)
        return DebateEvent(
            global_order=_parse_int(_require(fields, "GLOBAL_UTTERANCE_ORDER"), "GLOBAL_UTTERANCE_ORDER"),
            local_order=_parse_int(_require(fields, "CHUNK_UTTERANCE_ORDER"), "CHUNK_UTTERANCE_ORDER"),
            speaker_role=_require(fields, "SPEAKER_ROLE"),
            speaker_name=_require(fields, "SPEAKER_NAME"),
            target=_normalize_value(fields.get("TARGET")),
            text=_normalize_value(fields.get("SPOKEN_TEXT")),
            tone=_split_list(fields.get("TONE", "NONE")),
            volume=_normalize_value(fields.get("VOLUME")),
            pace=_normalize_value(fields.get("PACE")),
            pause_before=_normalize_value(fields.get("PAUSE_BEFORE")),
            pause_after=_normalize_value(fields.get("PAUSE_AFTER")),
            interruption=_parse_yes_no(fields.get("INTERRUPTION", "NO"), "INTERRUPTION"),
            nonverbal=_split_list(fields.get("NONVERBAL", "NONE")),
            event_kind="utterance",
            raw_fields=fields,
        )


class DebateTemplate02Parser(DebateTemplateParser):
    template_id = "template_02"
    _segment_pattern = re.compile(
        r"=== DEBATE_SEQUENCE_START ===\n(.*?)\n=== DEBATE_SEQUENCE_END ===",
        re.DOTALL,
    )
    _event_pattern = re.compile(
        r"\+\+\+ EVENT START \+\+\+\n(.*?)\n\+\+\+ EVENT END \+\+\+",
        re.DOTALL,
    )

    def can_parse(self, text: str) -> bool:
        return (
            "TEMPLATE_ID: template_02" in text
            or "DEBATE TEMPLATE 02" in text
            or (
                "=== DEBATE_SEQUENCE_START ===" in text
                and "+++ EVENT START +++" in text
            )
        )

    def parse(self, text: str) -> ParsedDebateDocument:
        chunks = [
            self._parse_segment(match.group(1))
            for match in self._segment_pattern.finditer(text)
        ]
        if not chunks:
            raise DebateParserError("No template_02 debate segments were found.")
        return ParsedDebateDocument(
            template_id=self.template_id,
            parser_name=self.__class__.__name__,
            raw_text=text,
            chunks=tuple(chunks),
        )

    def _parse_segment(self, segment_text: str) -> DebateSegment:
        if "+++ EVENT START +++" not in segment_text:
            raise DebateParserError("template_02 segment is missing event blocks.")

        metadata_text = segment_text.split("+++ EVENT START +++", 1)[0]
        metadata = _parse_key_value_block(metadata_text)
        events = [
            self._parse_event(match.group(1))
            for match in self._event_pattern.finditer(segment_text)
        ]
        if not events:
            raise DebateParserError("template_02 segment must contain at least one event.")

        return DebateSegment(
            template_id=metadata.get("TEMPLATE_ID", self.template_id),
            debate_id=_require(metadata, "DEBATE_ID"),
            title=_require(metadata, "DEBATE_TITLE"),
            chunk_order=_parse_int(_require(metadata, "SEGMENT_ORDER"), "SEGMENT_ORDER"),
            is_final_chunk=_parse_yes_no(_require(metadata, "IS_FINAL_SEGMENT"), "IS_FINAL_SEGMENT"),
            participants=_split_list(_require(metadata, "PARTICIPANT_NAMES")),
            moderators=_split_list(metadata.get("MODERATOR_NAMES", "NONE")),
            audience_present=_parse_yes_no(_require(metadata, "AUDIENCE_PRESENT"), "AUDIENCE_PRESENT"),
            audience_participation=_parse_optional_yes_no(
                metadata.get("AUDIENCE_PARTICIPATION", "NONE"),
                "AUDIENCE_PARTICIPATION",
            ),
            topic=_normalize_value(metadata.get("DEBATE_TOPIC")),
            debate_format=_normalize_value(metadata.get("DEBATE_STYLE")),
            setting=_normalize_value(metadata.get("SETTING")),
            moderator_mode=_normalize_value(metadata.get("MODERATOR_MODE")),
            continuity_notes=_normalize_value(metadata.get("CARRY_FORWARD")),
            metadata=metadata,
            events=tuple(events),
        )

    def _parse_event(self, event_text: str) -> DebateEvent:
        fields = _parse_key_value_block(event_text)
        event_kind = _require(fields, "EVENT_KIND")
        interrupts_previous = _parse_yes_no(
            fields.get("INTERRUPTS_PREVIOUS", "NO"),
            "INTERRUPTS_PREVIOUS",
        )
        return DebateEvent(
            global_order=_parse_int(_require(fields, "GLOBAL_EVENT_ORDER"), "GLOBAL_EVENT_ORDER"),
            local_order=_parse_int(_require(fields, "SEGMENT_EVENT_ORDER"), "SEGMENT_EVENT_ORDER"),
            speaker_role=_require(fields, "SPEAKER_ROLE"),
            speaker_name=_require(fields, "SPEAKER_NAME"),
            target=_normalize_value(fields.get("ADDRESSEE")),
            text=_normalize_value(fields.get("TEXT")),
            tone=_split_list(fields.get("TONE_SHIFT", "NONE")),
            volume=_normalize_value(fields.get("VOLUME")),
            pace=_normalize_value(fields.get("PACE")),
            pause_before=_normalize_value(fields.get("PAUSE_BEFORE")),
            pause_after=_normalize_value(fields.get("PAUSE_AFTER")),
            interruption=interrupts_previous or event_kind == "interruption",
            nonverbal=_split_list(fields.get("NONVERBAL", "NONE")),
            event_kind=event_kind,
            overlap_with_previous=_parse_yes_no(
                fields.get("OVERLAP_WITH_PREVIOUS", "NO"),
                "OVERLAP_WITH_PREVIOUS",
            ),
            completion_status=_normalize_value(fields.get("COMPLETION_STATUS")),
            raw_fields=fields,
        )


class DebateParserRegistry:
    """Registry-backed parser selector for debate template documents."""

    def __init__(self, parsers: Iterable[DebateTemplateParser] | None = None):
        self._parsers: dict[str, DebateTemplateParser] = {}
        for parser in parsers or self.default_parsers():
            self.register(parser)

    @staticmethod
    def default_parsers() -> tuple[DebateTemplateParser, ...]:
        return (DebateTemplate01Parser(), DebateTemplate02Parser())

    def register(self, parser: DebateTemplateParser) -> None:
        self._parsers[parser.template_id] = parser

    def available_template_ids(self) -> tuple[str, ...]:
        return tuple(self._parsers)

    def detect_parser(self, text: str) -> DebateTemplateParser:
        explicit_template_id = _extract_template_id(text)
        if explicit_template_id:
            parser = self._parsers.get(explicit_template_id)
            if parser is None:
                raise DebateParserError(
                    f"No parser registered for template ID '{explicit_template_id}'."
                )
            return parser

        matches = [parser for parser in self._parsers.values() if parser.can_parse(text)]
        if not matches:
            raise DebateParserError("Unable to detect a registered debate template.")
        if len(matches) > 1:
            names = ", ".join(parser.template_id for parser in matches)
            raise DebateParserError(
                f"Debate template detection is ambiguous. Matching parsers: {names}."
            )
        return matches[0]

    def parse(self, text: str) -> ParsedDebateDocument:
        parser = self.detect_parser(text)
        return parser.parse(text)

    def parse_many(self, texts: Iterable[str]) -> tuple[ParsedDebateDocument, ...]:
        return tuple(self.parse(text) for text in texts)

    def parse_and_merge(self, texts: Iterable[str]):
        documents = self.parse_many(texts)
        return DebateDocumentMerger().merge_documents(documents)


def _extract_template_id(text: str) -> str | None:
    match = re.search(r"^TEMPLATE_ID:\s*([^\n]+)$", text, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def _parse_key_value_block(block_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in block_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def _require(fields: dict[str, str], key: str) -> str:
    value = fields.get(key)
    if value is None or not value.strip():
        raise DebateParserError(f"Required field '{key}' is missing.")
    return value


def _parse_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise DebateParserError(f"Field '{field_name}' must be an integer.") from exc


def _parse_yes_no(value: str, field_name: str) -> bool:
    normalized = value.strip().upper()
    if normalized == "YES":
        return True
    if normalized == "NO":
        return False
    raise DebateParserError(f"Field '{field_name}' must be YES or NO.")


def _parse_optional_yes_no(value: str, field_name: str) -> bool | None:
    normalized = value.strip().upper()
    if normalized == "NONE":
        return None
    return _parse_yes_no(value, field_name)


def _normalize_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.upper() == "NONE":
        return None
    return stripped


def _split_list(value: str) -> tuple[str, ...]:
    normalized = _normalize_value(value)
    if normalized is None:
        return ()
    return tuple(item.strip() for item in normalized.split(";") if item.strip())

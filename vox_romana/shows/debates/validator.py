"""Validates parsed debate chunks and collects problems without hard-crashing.

The idea: LLMs mess up. A lot. Instead of blowing up the whole pipeline when
DeepSeek drops a field or invents a new tone value, we collect every problem
into a ValidationReport so you can decide what to do about it.

Each issue is either a WARNING (suspicious but survivable) or an ERROR
(structurally broken — will probably break downstream).

Generic primitives (``Severity``, ``ValidationIssue``, ``ValidationReport``)
live in :mod:`vox_romana.core.validation` and are re-exported here for
backward compatibility — older code can still
``from vox_romana.shows.debates.validator import ValidationReport``.
"""

from __future__ import annotations

from ...core.validation import Severity, ValidationIssue, ValidationReport
from .models import DebateEvent, DebateSegment, ParsedDebateDocument

# ── Allowed values ─────────────────────────────────────────────────────────
# These mirror the template definitions. If you add a new template with new
# allowed values, extend these sets — the validator checks against them.

ALLOWED_TONES = frozenset({
    "neutral", "calm", "firm", "passionate", "skeptical", "reflective",
    "urgent", "confrontational", "respectful", "analytical", "playful",
    "solemn", "escalating", "deflective",
})

ALLOWED_VOLUMES = frozenset({
    "soft", "moderate", "loud", "rising", "falling",
})

ALLOWED_PACES = frozenset({
    "slow", "medium", "fast", "accelerating", "decelerating",
})

ALLOWED_PAUSES = frozenset({
    "none", "short", "medium", "long",
})

ALLOWED_ROLES = frozenset({
    "participant", "moderator", "audience",
})

ALLOWED_EVENT_KINDS = frozenset({
    "utterance", "speech", "interruption", "reaction",
    "moderation", "stage_direction",
})

ALLOWED_COMPLETION_STATUSES = frozenset({
    "complete", "cut_off", "fragment", "trailing", "handoff",
})


__all__ = [
    "ALLOWED_COMPLETION_STATUSES",
    "ALLOWED_EVENT_KINDS",
    "ALLOWED_PACES",
    "ALLOWED_PAUSES",
    "ALLOWED_ROLES",
    "ALLOWED_TONES",
    "ALLOWED_VOLUMES",
    "DebateValidator",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
]


# ── Validator ──────────────────────────────────────────────────────────────

class DebateValidator:
    """Checks a parsed document for structural problems and suspicious values.

    Usage:
        report = DebateValidator().validate(parsed_document)
        if not report.is_valid:
            print(report.summary())
    """

    def validate(self, document: ParsedDebateDocument) -> ValidationReport:
        report = ValidationReport()

        if not document.chunks:
            report.issues.append(ValidationIssue(
                Severity.ERROR, None, None, "chunks", "Document has no chunks.",
            ))
            return report

        # All chunks should agree on debate_id
        debate_ids = {chunk.debate_id for chunk in document.chunks}
        if len(debate_ids) > 1:
            report.issues.append(ValidationIssue(
                Severity.ERROR, None, None, "debate_id",
                f"Multiple debate IDs found: {sorted(debate_ids)}",
            ))
        report.debate_id = next(iter(debate_ids))

        # Check chunk ordering
        chunk_orders = [chunk.chunk_order for chunk in document.chunks]
        if sorted(chunk_orders) != chunk_orders:
            report.issues.append(ValidationIssue(
                Severity.WARNING, None, None, "chunk_order",
                f"Chunks are not in ascending order: {chunk_orders}",
            ))
        if len(set(chunk_orders)) != len(chunk_orders):
            report.issues.append(ValidationIssue(
                Severity.ERROR, None, None, "chunk_order",
                f"Duplicate chunk orders: {chunk_orders}",
            ))

        # Check each chunk individually
        all_global_orders: list[int] = []
        all_speaker_names: set[str] = set()

        for chunk in document.chunks:
            self._validate_chunk(chunk, report)
            all_global_orders.extend(e.global_order for e in chunk.events)
            all_speaker_names.update(e.speaker_name for e in chunk.events)

        # Global event order should be unique across all chunks
        if len(set(all_global_orders)) != len(all_global_orders):
            seen: set[int] = set()
            dupes: set[int] = set()
            for order in all_global_orders:
                if order in seen:
                    dupes.add(order)
                seen.add(order)
            report.issues.append(ValidationIssue(
                Severity.ERROR, None, None, "global_order",
                f"Duplicate global event orders across chunks: {sorted(dupes)}",
            ))

        return report

    def _validate_chunk(self, chunk: DebateSegment, report: ValidationReport) -> None:
        co = chunk.chunk_order

        # Must have at least 2 participants
        if len(chunk.participants) < 2:
            report.issues.append(ValidationIssue(
                Severity.ERROR, co, None, "participants",
                f"Need at least 2 participants, got {len(chunk.participants)}.",
            ))

        # Must have at least 1 event
        if not chunk.events:
            report.issues.append(ValidationIssue(
                Severity.ERROR, co, None, "events",
                "Chunk has no events.",
            ))
            return

        # Local event order should start at 1 and be contiguous
        local_orders = [e.local_order for e in chunk.events]
        expected = list(range(1, len(local_orders) + 1))
        if local_orders != expected:
            report.issues.append(ValidationIssue(
                Severity.WARNING, co, None, "local_order",
                f"Expected local orders {expected}, got {local_orders}.",
            ))

        # Collect known names for speaker checks
        known_names: set[str] = set(chunk.participants)
        known_names.update(chunk.moderators)
        # Audience members are dynamic, so we accept "Audience Member *"
        # patterns without requiring them in the participant list.

        for event in chunk.events:
            self._validate_event(event, chunk, known_names, report)

    def _validate_event(
        self,
        event: DebateEvent,
        chunk: DebateSegment,
        known_names: set[str],
        report: ValidationReport,
    ) -> None:
        co = chunk.chunk_order
        eo = event.global_order

        # Speaker name should match a known participant, moderator, or audience pattern
        if event.speaker_name not in known_names:
            is_audience_member = (
                event.speaker_role == "audience"
                or event.speaker_name.lower().startswith("audience member")
            )
            if not is_audience_member:
                report.issues.append(ValidationIssue(
                    Severity.WARNING, co, eo, "speaker_name",
                    f"'{event.speaker_name}' not in declared participants or moderators.",
                ))

        # Speaker role
        if event.speaker_role not in ALLOWED_ROLES:
            report.issues.append(ValidationIssue(
                Severity.WARNING, co, eo, "speaker_role",
                f"Unexpected role '{event.speaker_role}'.",
            ))

        # Tone values
        for tone in event.tone:
            if tone.lower() not in ALLOWED_TONES:
                report.issues.append(ValidationIssue(
                    Severity.WARNING, co, eo, "tone",
                    f"Unexpected tone '{tone}'.",
                ))

        # Volume
        if event.volume and event.volume.lower() not in ALLOWED_VOLUMES:
            report.issues.append(ValidationIssue(
                Severity.WARNING, co, eo, "volume",
                f"Unexpected volume '{event.volume}'.",
            ))

        # Pace
        if event.pace and event.pace.lower() not in ALLOWED_PACES:
            report.issues.append(ValidationIssue(
                Severity.WARNING, co, eo, "pace",
                f"Unexpected pace '{event.pace}'.",
            ))

        # Pauses
        for pause_field in ("pause_before", "pause_after"):
            val = getattr(event, pause_field)
            if val and val.lower() not in ALLOWED_PAUSES:
                report.issues.append(ValidationIssue(
                    Severity.WARNING, co, eo, pause_field,
                    f"Unexpected pause value '{val}'.",
                ))

        # Event kind
        if event.event_kind not in ALLOWED_EVENT_KINDS:
            report.issues.append(ValidationIssue(
                Severity.WARNING, co, eo, "event_kind",
                f"Unexpected event kind '{event.event_kind}'.",
            ))

        # Completion status
        if event.completion_status and event.completion_status not in ALLOWED_COMPLETION_STATUSES:
            report.issues.append(ValidationIssue(
                Severity.WARNING, co, eo, "completion_status",
                f"Unexpected completion status '{event.completion_status}'.",
            ))

        # A speech/utterance event with no text is suspicious
        if event.event_kind in ("utterance", "speech", "moderation"):
            if not event.text:
                report.issues.append(ValidationIssue(
                    Severity.WARNING, co, eo, "text",
                    f"Speech-type event ('{event.event_kind}') has no spoken text.",
                ))

"""Shared validation primitives for show-output documents.

Every show type needs a way to inspect a parsed/merged document and
collect structural problems without crashing the pipeline. The base
pieces are identical:

  * ``Severity`` — enum: WARNING vs ERROR.
  * ``ValidationIssue`` — one problem with location + message.
  * ``ValidationReport`` — collection with ``warnings`` / ``errors`` /
    ``is_valid`` accessors and a human-readable ``summary()``.

Show-specific validators (``DebateValidator``, etc.) live in their own
packages and emit issues into a :class:`ValidationReport`. Show-specific
issue location fields (``chunk_order``, ``event_order``) are kept on
:class:`ValidationIssue` because they're generic enough that other show
types can reuse them — chunks/events are a common pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """How seriously to take a validation issue."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class ValidationIssue:
    """One problem found during validation."""

    severity: Severity
    chunk_order: int | None  # None for document-level issues
    event_order: int | None  # None for chunk-level issues
    field_name: str
    message: str

    def __str__(self) -> str:
        location = ""
        if self.chunk_order is not None:
            location += f"chunk {self.chunk_order}"
        if self.event_order is not None:
            location += f" event {self.event_order}"
        location = location.strip() or "document"
        return f"[{self.severity.value.upper()}] {location} / {self.field_name}: {self.message}"


@dataclass(slots=True)
class ValidationReport:
    """A collection of validation issues for a single document.

    The optional ``debate_id`` field is named historically — it carries
    any document identifier (debate id, news show id, audiobook id, …).
    """

    debate_id: str | None = None
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def is_valid(self) -> bool:
        """True if there are zero errors. Warnings alone don't fail validation."""
        return not self.errors

    def summary(self) -> str:
        lines = [f"Validation for '{self.debate_id or 'unknown'}': "
                 f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)

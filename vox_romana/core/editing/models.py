"""Data models for the editing pipeline.

These are the building blocks for the Director outline, Editor verdicts,
chunk results, and fact-check reports. All are plain dataclasses — no
behavior, just data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass
class OutlineSection:
    """A single section in the Director's show outline."""

    id: str
    title: str
    expected_duration: str = ""
    key_beats: list[str] = field(default_factory=list)
    speaker_assignments: list[str] = field(default_factory=list)
    acceptance_criteria: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutlineSection:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class OutlineSnapshot:
    """A versioned snapshot of the Director's outline.

    Every time the outline changes, a new snapshot is created. This is
    what gets broadcast to the cast and persisted for audit.
    """

    outline_id: str
    version: int
    timestamp: str
    summary: str
    sections: list[OutlineSection]
    changed_sections: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sections"] = [s.to_dict() for s in self.sections]
        return data

    def to_broadcast_text(self) -> str:
        """Format for injection into agent context windows."""
        lines = [
            f"OUTLINE UPDATE (v{self.version}): {self.summary}",
        ]
        if self.changed_sections:
            lines.append(
                f"Changed sections: {', '.join(self.changed_sections)}"
            )
        lines.append("Current sections:")
        for s in self.sections:
            beats = ", ".join(s.key_beats) if s.key_beats else "no specific beats"
            speakers = (
                ", ".join(s.speaker_assignments)
                if s.speaker_assignments
                else "TBD"
            )
            lines.append(
                f"  [{s.id}] {s.title} ({s.expected_duration}) — "
                f"beats: {beats} | speakers: {speakers}"
            )
        return "\n".join(lines)


@dataclass
class EditorVerdict:
    """The Editor's judgment on a single chunk."""

    status: str  # "pass" or "fail"
    chunk_id: str
    outline_version: int
    summary: str
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditorVerdict:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class ChunkResult:
    """The result of generating (and optionally retrying) a single chunk."""

    chunk_id: str
    content: list[dict[str, Any]]
    outline_version: int
    attempt_number: int
    editor_verdict: EditorVerdict | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.editor_verdict:
            data["editor_verdict"] = self.editor_verdict.to_dict()
        return data


# ─────────────────────────────────────────────────────────────────────────────
# Fact-check models
# ─────────────────────────────────────────────────────────────────────────────
# These three dataclasses represent the output of the FactChecker agent.
# They flow from the FactChecker → ChunkEditor → conductor retry loop.
#
# FlaggedClaim   — a specific assertion extracted from generated text that
#                  the FactChecker needs to verify (e.g. a date, a statistic,
#                  a person's name or cause of death).
#
# FactCheckResult — the verdict on one claim after checking it against the
#                   reference_material (quick mode) and/or a web search
#                   (thorough mode).
#
# ChunkFactCheckReport — the full report for one generated segment/chunk.
#                        has_fabrications drives the blocking retry logic.


@dataclass
class FlaggedClaim:
    """A concrete factual assertion extracted from generated text.

    Attributes:
        text:         The verbatim claim as it appears in the output
                      (e.g. "Dosio died on August 12, 2022").
        reason:       Why this claim was flagged for checking
                      (e.g. "specific date about a named real person").
        search_query: An optional web-search query to verify this claim
                      in thorough mode (e.g. "Ernie Dosio death date Gabon").
    """

    text: str
    reason: str
    search_query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlaggedClaim:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class FactCheckResult:
    """The verdict on a single flagged claim.

    Attributes:
        claim:       The original claim text (mirrors FlaggedClaim.text).
        verdict:     One of:
                       "verified"          — confirmed by reference_material or search.
                       "unverifiable"      — no reference material or search result
                                            addressed this claim; cannot confirm or deny.
                       "fabricated"        — directly contradicted by reference_material,
                                            search results, or established canon
                                            (for content_mode="fictional-universe").
                       "contradicts_prior" — contradicts a fact previously asserted in
                                            an EARLIER approved segment/turn/chapter of
                                            the SAME show. Drives the intra-show
                                            consistency gate.
        explanation: Human-readable explanation of the verdict. Flows into the
                     regeneration rejection note so the LLM knows exactly what
                     to correct.
        source:      Where the verification (or contradiction) came from —
                     e.g. "reference_material", "google: <url>",
                     "prior_context", "none".
    """

    claim: str
    verdict: Literal["verified", "unverifiable", "fabricated", "contradicts_prior"]
    explanation: str
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactCheckResult:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class ChunkFactCheckReport:
    """Full fact-check report for one generated segment or chunk.

    Attributes:
        chunk_id:  Identifier matching the chunk being checked.
        mode:      The fact-check mode used ("quick" or "thorough").
        results:   One FactCheckResult per extracted claim.
        summary:   A one-paragraph prose summary of findings, suitable
                   for injection into a rejection note.

    Properties:
        has_fabrications: True if any result has verdict == "fabricated".
                          This is the boolean that drives the blocking retry
                          logic in the conductor.
    """

    chunk_id: str
    mode: str
    results: list[FactCheckResult] = field(default_factory=list)
    summary: str = ""

    @property
    def has_fabrications(self) -> bool:
        """Return True if any claim was found to be fabricated."""
        return any(r.verdict == "fabricated" for r in self.results)

    @property
    def fabricated_claims(self) -> list[FactCheckResult]:
        """Return only the results with verdict == 'fabricated'."""
        return [r for r in self.results if r.verdict == "fabricated"]

    @property
    def has_contradictions(self) -> bool:
        """Return True if any claim contradicts prior approved content.

        Drives the intra-show consistency gate in the conductor retry loop —
        treated equivalently to ``has_fabrications`` for blocking purposes
        when ``content_mode == "factual"``.
        """
        return any(r.verdict == "contradicts_prior" for r in self.results)

    @property
    def contradicted_claims(self) -> list[FactCheckResult]:
        """Return only the results with verdict == 'contradicts_prior'."""
        return [r for r in self.results if r.verdict == "contradicts_prior"]

    def to_dict(self) -> dict[str, Any]:
        data = {
            "chunk_id": self.chunk_id,
            "mode": self.mode,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "has_fabrications": self.has_fabrications,
            "has_contradictions": self.has_contradictions,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkFactCheckReport:
        results = [
            FactCheckResult.from_dict(r) for r in data.get("results", [])
        ]
        return cls(
            chunk_id=data.get("chunk_id", ""),
            mode=data.get("mode", "quick"),
            results=results,
            summary=data.get("summary", ""),
        )

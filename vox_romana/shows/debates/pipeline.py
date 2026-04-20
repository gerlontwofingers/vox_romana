"""End-to-end pipeline: raw LLM text in, validated merged JSON out.

This is the main entry point you'll use most of the time. It wires together
the parser registry, validator, merger, and serializer so you don't have to
remember the order or pass objects between them manually.

    from vox_romana.shows.debates import DebatePipeline

    pipeline = DebatePipeline()

    # Feed it raw text chunks from DeepSeek (list of strings)
    result = pipeline.run(chunks)

    # Check if the LLM behaved
    for report in result.reports:
        if not report.is_valid:
            print(report.summary())

    # Save the good ones
    for debate in result.merged:
        pipeline.save(debate, f"output/{debate.debate_id}.json")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable

from .merge import DebateDocumentMerger, DebateMergeError
from .models import MergedDebate, ParsedDebateDocument
from .parsers import DebateParserError, DebateParserRegistry
from .serializer import DebateSerializer
from .validator import DebateValidator, ValidationReport


@dataclass(slots=True)
class PipelineResult:
    """Everything that came out of a pipeline run, good and bad."""

    # Successfully parsed documents (before merge)
    parsed: tuple[ParsedDebateDocument, ...] = ()

    # Validation reports — one per parsed document
    reports: tuple[ValidationReport, ...] = ()

    # Merged debates (only from documents that passed validation)
    merged: tuple[MergedDebate, ...] = ()

    # Raw texts that failed to parse at all, with the error message
    parse_failures: tuple[tuple[str, str], ...] = ()

    # Merge-level errors, if any
    merge_failures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True if nothing failed and every report is valid."""
        return (
            not self.parse_failures
            and not self.merge_failures
            and all(r.is_valid for r in self.reports)
        )

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(f"Parsed: {len(self.parsed)}  |  "
                      f"Merged: {len(self.merged)}  |  "
                      f"Parse failures: {len(self.parse_failures)}  |  "
                      f"Merge failures: {len(self.merge_failures)}")

        for report in self.reports:
            if report.issues:
                lines.append(report.summary())

        for raw_text, error_msg in self.parse_failures:
            preview = raw_text[:80].replace("\n", " ")
            lines.append(f"[PARSE FAIL] \"{preview}...\" → {error_msg}")

        for msg in self.merge_failures:
            lines.append(f"[MERGE FAIL] {msg}")

        return "\n".join(lines)


class DebatePipeline:
    """Parse → Validate → Merge → Serialize, all in one place.

    You can use individual pieces if you want, or just call run() and save().
    """

    def __init__(
        self,
        registry: DebateParserRegistry | None = None,
        validator: DebateValidator | None = None,
        merger: DebateDocumentMerger | None = None,
        serializer: DebateSerializer | None = None,
    ):
        self.registry = registry or DebateParserRegistry()
        self.validator = validator or DebateValidator()
        self.merger = merger or DebateDocumentMerger()
        self.serializer = serializer or DebateSerializer()

    def run(
        self,
        raw_texts: Iterable[str],
        *,
        strict: bool = False,
    ) -> PipelineResult:
        """Run the full pipeline on a batch of raw LLM output strings.

        Args:
            raw_texts: Each string is one chunk of raw text from the LLM.
            strict: If True, documents with any validation warnings are also
                    excluded from merging. Default only excludes errors.

        Returns:
            A PipelineResult with everything that happened.
        """
        raw_list = list(raw_texts)

        # ── Step 1: Parse ──────────────────────────────────────────────
        parsed: list[ParsedDebateDocument] = []
        parse_failures: list[tuple[str, str]] = []

        for text in raw_list:
            try:
                parsed.append(self.registry.parse(text))
            except DebateParserError as exc:
                parse_failures.append((text, str(exc)))

        # ── Step 2: Validate ───────────────────────────────────────────
        reports: list[ValidationReport] = []
        valid_docs: list[ParsedDebateDocument] = []

        for doc in parsed:
            report = self.validator.validate(doc)
            reports.append(report)

            # Decide if this doc is clean enough to merge
            passes = report.is_valid if not strict else (not report.issues)
            if passes:
                valid_docs.append(doc)

        # ── Step 3: Merge ──────────────────────────────────────────────
        merged: tuple[MergedDebate, ...] = ()
        merge_failures: list[str] = []

        if valid_docs:
            try:
                merged = self.merger.merge_documents(valid_docs)
            except DebateMergeError as exc:
                merge_failures.append(str(exc))

        # ── Done ───────────────────────────────────────────────────────
        return PipelineResult(
            parsed=tuple(parsed),
            reports=tuple(reports),
            merged=merged,
            parse_failures=tuple(parse_failures),
            merge_failures=tuple(merge_failures),
        )

    # ── Saving shortcuts ───────────────────────────────────────────────

    def save(self, debate: MergedDebate, path: str | Path) -> Path:
        """Save a single merged debate to JSON."""
        return self.serializer.save_merged(debate, path)

    def save_all(self, result: PipelineResult, directory: str | Path) -> list[Path]:
        """Save every merged debate in a result to separate JSON files."""
        out_dir = Path(directory)
        paths: list[Path] = []
        for debate in result.merged:
            filename = f"{debate.debate_id}.json"
            paths.append(self.serializer.save_merged(debate, out_dir / filename))
        return paths

    def to_json(self, debate: MergedDebate) -> str:
        """Return a merged debate as a pretty-printed JSON string."""
        return self.serializer.to_json_string(debate)

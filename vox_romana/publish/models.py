"""Data models for formatted documents and publish results.

A ``FormattedDocument`` is the show-type-agnostic intermediate
representation between the raw show JSON and the final TXT/PDF output.
It is structured (sections + lines) rather than free text so that:

  - Plain-text rendering is trivial (concatenate with delimiters).
  - HTML/PDF rendering can apply per-line CSS classes (e.g. centre stage
    directions, italicise non-verbal cues, all-caps speaker labels).
  - The optional LLM polish pass can rewrite individual lines while
    preserving structural metadata.

Lines carry a ``kind`` tag so renderers know what they are looking at.
The set of kinds is deliberately small and conventional:

    title         — document or section title
    heading       — secondary heading (act, scene, segment slug)
    speaker       — speaker label ("RONON DEX:", "ANCHOR ON CAM:")
    dialogue      — spoken text
    narration     — narrator/host prose, also audiobook body text
    direction     — stage direction or non-verbal cue (parenthetical)
    metadata      — front-matter line (cast list, date, topic)
    blank         — explicit blank line
    rule          — horizontal rule / scene break (* * *)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


LineKind = Literal[
    "title",
    "heading",
    "speaker",
    "dialogue",
    "narration",
    "direction",
    "metadata",
    "blank",
    "rule",
]


@dataclass(frozen=True)
class Line:
    """One typed line of formatted output.

    Attributes:
        kind: What this line is (drives rendering).
        text: The line's text. May be empty for ``blank`` and ``rule``.
    """

    kind: LineKind
    text: str = ""


@dataclass
class Section:
    """A logical chunk of the document (act, segment, chapter, etc.).

    Attributes:
        title: Section heading text.
        lines: Ordered lines making up the section's body.
        kind:  Free-form label describing the section type, e.g.
               ``"chunk"``, ``"segment"``, ``"chapter"``, ``"front_matter"``.
    """

    title: str
    lines: list[Line] = field(default_factory=list)
    kind: str = "section"


@dataclass
class FormattedDocument:
    """The structured intermediate form of a publishable show.

    Attributes:
        show_type:    One of ``"debate"``, ``"news"``, ``"podcast"``,
                      ``"audiobook"``.
        title:        The show's title.
        front_matter: Lines shown before the first section (cast, date,
                      topic, format notes).
        sections:     Ordered body sections.
        meta:         Free-form metadata for renderers (e.g. participants
                      list, segment counts).  Not displayed verbatim.
    """

    show_type: str
    title: str
    front_matter: list[Line] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class PublishResult:
    """What ``publish_document`` returns.

    Attributes:
        txt_path: Path to the generated TXT file (None if not requested).
        pdf_path: Path to the generated PDF file (None if not requested
                  or weasyprint unavailable).
        show_type: The detected show type.
        sections: How many body sections the document had.
        warnings: Non-fatal messages (e.g. "weasyprint not installed,
                  PDF skipped").
    """

    txt_path: Path | None = None
    pdf_path: Path | None = None
    show_type: str = ""
    sections: int = 0
    warnings: list[str] = field(default_factory=list)

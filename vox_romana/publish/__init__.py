"""Publish package — turn show JSON into formatted human-readable documents.

Currently supports plain text (always) and PDF (when ``weasyprint`` is
installed).  An optional LLM polish pass (``vox_romana.publish.polisher``)
can clean spoken-style phrasing into written prose; that lives in a separate
module so the deterministic path here has zero LLM dependencies.

Public API:

    >>> from vox_romana.publish import publish_document
    >>> publish_document("output/text/debates/foo.json", formats=("txt", "pdf"))

Or use the lower-level building blocks:

    >>> from vox_romana.publish import format_document, render_txt, render_pdf
    >>> doc = format_document(json_dict)         # auto-detects show type
    >>> render_txt(doc, "foo.txt")
    >>> render_pdf(doc, "foo.pdf")               # raises if weasyprint missing
"""

from __future__ import annotations

from .models import FormattedDocument, PublishResult, Section
from .formatter import (
    DocumentFormatter,
    DebateFormatter,
    NewsFormatter,
    PodcastFormatter,
    AudiobookFormatter,
    detect_show_type,
    format_document,
)
from .outputter import render_txt, render_pdf, publish_document, PDF_AVAILABLE

__all__ = [
    "FormattedDocument",
    "PublishResult",
    "Section",
    "DocumentFormatter",
    "DebateFormatter",
    "NewsFormatter",
    "PodcastFormatter",
    "AudiobookFormatter",
    "detect_show_type",
    "format_document",
    "render_txt",
    "render_pdf",
    "publish_document",
    "PDF_AVAILABLE",
]

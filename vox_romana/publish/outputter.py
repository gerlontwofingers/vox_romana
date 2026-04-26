"""Renderers — turn a ``FormattedDocument`` into TXT or PDF on disk.

.. note::
   This module (``vox_romana.publish.outputter``) writes formatted documents
   to the filesystem as human-readable TXT or print-ready PDF files.
   It is **completely separate** from ``vox_romana.tts.renderer``, which
   synthesises speech audio from :class:`~vox_romana.core.models.AudioEvent`
   sequences.  They share the name "renderer" but operate in different
   domains and have no shared code.

Plain-text rendering is dependency-free and always available.  PDF
rendering uses Jinja2 + WeasyPrint; both are imported lazily so that
``vox_romana.publish`` itself stays import-cheap and so a missing
WeasyPrint install merely disables PDF instead of crashing import.

There is also a high-level :func:`publish_document` convenience that
loads a JSON file, formats it, and writes both TXT and PDF (as
requested) into a destination directory.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable

from .formatter import detect_show_type, format_document
from .models import FormattedDocument, Line, PublishResult, Section


# ─────────────────────────────────────────────────────────────────────────────
# Optional PDF stack — imported lazily.
# ─────────────────────────────────────────────────────────────────────────────

# Detect availability without importing the heavy modules at import time.
def _check_pdf_available() -> bool:
    try:
        import importlib.util
        for mod in ("weasyprint", "jinja2"):
            if importlib.util.find_spec(mod) is None:
                return False
        return True
    except Exception:  # pragma: no cover — defensive
        return False


PDF_AVAILABLE: bool = _check_pdf_available()
"""True iff both ``weasyprint`` and ``jinja2`` import successfully."""


# ─────────────────────────────────────────────────────────────────────────────
# Plain text rendering
# ─────────────────────────────────────────────────────────────────────────────

# Wrap width for prose-ish lines; chosen to match a comfortable
# 80-column terminal while still looking right when printed.
_WRAP = 78
_DIALOGUE_INDENT = "    "
_DIRECTION_INDENT = "        "
_NARRATION_INDENT = "  "

_RULE = "* * *"


def _wrap(text: str, indent: str = "", width: int = _WRAP) -> list[str]:
    """Word-wrap text with a hanging indent. Empty input → empty list."""
    text = text.strip()
    if not text:
        return []
    return textwrap.wrap(
        text,
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [indent + text]


def _render_line_to_txt(line: Line) -> list[str]:
    """Render one ``Line`` as zero or more text rows."""
    if line.kind == "blank":
        return [""]
    if line.kind == "rule":
        # Centre-ish; the rule is short and looks fine left-padded.
        return ["", " " * ((_WRAP - len(_RULE)) // 2) + _RULE, ""]
    if line.kind == "title":
        return [line.text.upper(), "=" * min(len(line.text), _WRAP)]
    if line.kind == "heading":
        return ["", line.text, "-" * min(len(line.text), _WRAP)]
    if line.kind == "speaker":
        # Speaker labels sit at column 0 with a blank line above (added
        # by the caller's section flow).
        return [line.text]
    if line.kind == "dialogue":
        return _wrap(line.text, _DIALOGUE_INDENT)
    if line.kind == "narration":
        return _wrap(line.text, _NARRATION_INDENT)
    if line.kind == "direction":
        return _wrap(line.text, _DIRECTION_INDENT)
    if line.kind == "metadata":
        return _wrap(line.text, "")
    # Unknown kind → render as-is.
    return [line.text]


def _render_section_to_txt(section: Section) -> list[str]:
    """Render a section heading + body lines to text rows."""
    rows: list[str] = []
    if section.title:
        rows.append("")
        rows.append(section.title)
        rows.append("─" * min(len(section.title), _WRAP))
        rows.append("")
    for line in section.lines:
        rows.extend(_render_line_to_txt(line))
    return rows


def render_txt(doc: FormattedDocument, path: str | Path) -> Path:
    """Write ``doc`` as plain UTF-8 text to ``path``.

    Args:
        doc:  The formatted document to render.
        path: Output file path (parent directories are created).

    Returns:
        The resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    title = doc.title.strip() or "Untitled"
    rows.append(title.upper())
    rows.append("=" * min(len(title), _WRAP))
    rows.append("")

    for line in doc.front_matter:
        rows.extend(_render_line_to_txt(line))
    if doc.front_matter:
        rows.append("")

    for sec in doc.sections:
        rows.extend(_render_section_to_txt(sec))

    # Collapse runs of >2 blank lines.
    cleaned: list[str] = []
    blank_run = 0
    for row in rows:
        if row == "":
            blank_run += 1
            if blank_run > 2:
                continue
        else:
            blank_run = 0
        cleaned.append(row)

    out.write_text("\n".join(cleaned).rstrip() + "\n", encoding="utf-8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PDF rendering
# ─────────────────────────────────────────────────────────────────────────────

# Templates and CSS live in ``<repo>/templates/publish/``.
_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates" / "publish"
_CSS_FILE = _TEMPLATE_DIR / "publish.css"
_TEMPLATE_BY_TYPE = {
    "debate":    "debate.html.jinja2",
    "news":      "news.html.jinja2",
    "podcast":   "podcast.html.jinja2",
    "audiobook": "audiobook.html.jinja2",
}


def _render_html(doc: FormattedDocument) -> str:
    """Render ``doc`` to a complete HTML document via Jinja2."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "jinja2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template_name = _TEMPLATE_BY_TYPE.get(
        doc.show_type, "debate.html.jinja2",
    )
    template = env.get_template(template_name)
    return template.render(doc=doc)


def render_pdf(doc: FormattedDocument, path: str | Path) -> Path:
    """Render ``doc`` to PDF at ``path`` using WeasyPrint.

    Args:
        doc:  Formatted document.
        path: Output PDF path.

    Returns:
        The resolved output path.

    Raises:
        RuntimeError: If WeasyPrint or Jinja2 isn't installed.
    """
    if not PDF_AVAILABLE:
        raise RuntimeError(
            "PDF rendering requires 'weasyprint' and 'jinja2'. "
            "Install with: pip install weasyprint jinja2"
        )
    from weasyprint import CSS, HTML  # type: ignore

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    html_str = _render_html(doc)
    stylesheets = []
    if _CSS_FILE.exists():
        stylesheets.append(CSS(filename=str(_CSS_FILE)))

    HTML(string=html_str, base_url=str(_TEMPLATE_DIR)).write_pdf(
        target=str(out),
        stylesheets=stylesheets or None,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# High-level convenience
# ─────────────────────────────────────────────────────────────────────────────

def _default_outputs(
    src: Path,
    dest_dir: Path | None,
    formats: Iterable[str],
    *,
    language: str = "en",
    root_lang: str = "en",
) -> dict[str, Path]:
    """Compute default output paths for each requested format.

    When ``language`` differs from ``root_lang``, a ``_{language}`` suffix
    is inserted before the file extension so multilingual output files
    coexist without overwriting each other.

    Example (root_lang="en", language="es")::

        debate.txt  →  debate_es.txt
        debate.pdf  →  debate_es.pdf
    """
    out_dir = dest_dir or src.parent
    stem = src.stem
    if language != root_lang:
        stem = f"{stem}_{language}"
    return {fmt: out_dir / f"{stem}.{fmt}" for fmt in formats}


def publish_document(
    src: str | Path,
    formats: Iterable[str] = ("txt",),
    dest_dir: str | Path | None = None,
    *,
    root_lang: str = "en",
) -> PublishResult:
    """Load show JSON, format it, and write the requested output formats.

    Args:
        src:       Path to a show JSON file.
        formats:   Iterable of formats to produce.  Currently supported:
                   ``"txt"`` (always) and ``"pdf"`` (requires weasyprint).
        dest_dir:  Where to write outputs.  Defaults to ``src``'s parent.
        root_lang: The project's root language (default ``"en"``).
                   When the document's language differs from this, a
                   ``_{lang}`` suffix is inserted in the output filenames
                   (e.g. ``debate_es.txt``).

    Returns:
        A ``PublishResult`` describing what was written.
    """
    src_path = Path(src)
    data = json.loads(src_path.read_text(encoding="utf-8"))
    doc = format_document(data)
    language = doc.language  # already normalised by the formatter

    fmts = {f.lower() for f in formats}
    unknown = fmts - {"txt", "pdf"}
    if unknown:
        raise ValueError(
            f"Unknown publish formats: {sorted(unknown)} "
            "(supported: 'txt', 'pdf')"
        )

    out_paths = _default_outputs(
        src_path,
        Path(dest_dir) if dest_dir else None,
        fmts,
        language=language,
        root_lang=root_lang,
    )

    result = PublishResult(
        show_type=detect_show_type(data),
        language=language,
        sections=len(doc.sections),
    )

    if "txt" in fmts:
        result.txt_path = render_txt(doc, out_paths["txt"])

    if "pdf" in fmts:
        if PDF_AVAILABLE:
            result.pdf_path = render_pdf(doc, out_paths["pdf"])
        else:
            result.warnings.append(
                "PDF skipped: weasyprint and/or jinja2 not installed."
            )

    return result

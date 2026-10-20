"""Document ingestion — extracts text from user-provided files.

Supported formats (Tier 1 — full extraction):
    .txt, .md, .markdown — Plain text, auto-encoding detection.
    .rst, .log           — Plain text.
    .pdf                 — Requires PyPDF2 (optional).
    .json                — "text" field or concatenated string values.
    .jsonl               — Line-by-line JSON records.
    .html, .htm          — Requires beautifulsoup4 + lxml (optional).
    .csv, .tsv           — stdlib csv module.
    .docx                — Requires python-docx (optional).
    .epub                — Requires ebooklib + beautifulsoup4 (optional).
    .xlsx, .xls          — Requires openpyxl (optional).
    .pptx                — Requires python-pptx (optional).
    .odt                 — Requires odfpy (optional).
    .rtf                 — Requires striprtf (optional).
    .xml                 — stdlib ElementTree.
    .eml                 — stdlib email module.
    .mbox                — stdlib mailbox module.
    .tex                 — LaTeX stripped via regex.
    .fb2                 — XML ebook, stdlib ElementTree.
    .py, .js, .ts, ...   — Source code and config files read as plain text.

Tier 2 — archive recursion:
    .zip                 — stdlib zipfile, recurse into contained files.
    .tar, .gz, .bz2, .tgz — stdlib tarfile, recurse into contained files.

Tier 3 — known-but-unsupported (raises ValueError with install hint):
    .doc, .ppt           — Legacy Office formats (convert via LibreOffice).
    .mobi, .azw, .azw3   — Kindle (no free pure-Python library).
    .djvu                — Requires djvulibre system package.
    .msg                 — Outlook messages (requires extract-msg).

Each document is chunked into overlapping paragraphs with metadata,
then written as JSONL to ``data/memory/documents/``.  The corpus builder
reads these JSONL files as training material.

USAGE:
    from vox_romana.memory.ingest import DocumentIngester

    ingester = DocumentIngester()
    stats = ingester.ingest("papers/ubi_pilot_2024.pdf")
    all_stats = ingester.ingest_directory("papers/")

    # Dry-run (report without writing):
    stats = ingester.ingest("file.pdf", dry_run=True)

    # Force re-ingest even if already processed:
    stats = ingester.ingest("file.pdf", force_reingest=True)
"""

from __future__ import annotations

import csv
import email
import hashlib
import io
import json
import logging
import mailbox
import re
import tarfile
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

from vox_romana.core.paths import DOCUMENT_DIR as DEFAULT_DOCUMENT_DIR

# Chunking defaults.
_CONTENT_HASH_READ_BYTES = 65_536  # 64 KB sample for content-based dedup hash
CHUNK_SIZE = 800       # target chars per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks


@dataclass
class IngestStats:
    """Summary of a single ingest operation."""
    source: str
    chunks: int = 0
    chars: int = 0
    format: str = ""
    skipped: bool = False
    skip_reason: str = ""
    deduped_chunks: int = 0
    # True only for unexpected extractor errors (e.g. corrupt PDF/EPUB),
    # as opposed to intentional skips (dry run, dedup, Tier 3, etc.).
    failed: bool = False


@dataclass
class DocumentChunk:
    """A single chunk from an ingested document."""
    text: str
    title: str = ""
    source: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    format: str = ""
    ingested_at: str = ""
    tag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Encoding detection
# ─────────────────────────────────────────────────────────────────────────────

def _read_text_auto(path: Path) -> str:
    """Read a text file with automatic encoding detection.

    Uses charset-normalizer if available, then falls back to a manual
    chain of common encodings.
    """
    raw = path.read_bytes()
    # Try charset-normalizer (optional dep — already in requirements.txt).
    try:
        from charset_normalizer import from_bytes  # type: ignore[import]
        result = from_bytes(raw).best()
        if result is not None:
            return str(result)
    except ImportError:
        pass
    # Manual fallback chain — ordered by frequency.
    _ENCODING_CHAIN = [
        "utf-8", "utf-16", "utf-16-le", "utf-16-be",
        "iso-8859-1", "windows-1252", "windows-1250",
        "gb18030", "gbk", "shift_jis", "euc-jp",
        "euc-kr", "cp949", "big5", "macroman",
    ]
    for enc in _ENCODING_CHAIN:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Text extractors (per format)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(path: Path) -> str:
    """Read plain text, markdown, or log files with encoding detection."""
    return _read_text_auto(path)


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF using PyPDF2 (optional dep)."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError(
            "PyPDF2 required for PDF ingestion. "
            "Install with: pip install PyPDF2"
        )
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_json(path: Path) -> str:
    """Extract text from a JSON file.

    Priority order for dict objects: ``"text"`` key → concatenation of all
    top-level string values.  For lists, each string element or
    ``{"text": …}`` dict entry is included.  Falls back to empty string if
    no textual content is found.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if "text" in data:
            return str(data["text"])
        # Concatenate all string values.
        parts = [str(v) for v in data.values() if isinstance(v, str)]
        return "\n\n".join(parts)
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n\n".join(parts)
    return ""


def _extract_jsonl(path: Path) -> str:
    """Extract text from a JSONL file (one JSON object per line).

    Applies the same ``"text"`` key priority as ``_extract_json`` to each
    line independently.  Malformed lines are silently skipped.
    """
    parts: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, str):
                parts.append(obj)
            elif isinstance(obj, dict):
                if "text" in obj:
                    parts.append(str(obj["text"]))
                else:
                    parts.extend(str(v) for v in obj.values() if isinstance(v, str))
    return "\n\n".join(parts)


def _extract_html(path: Path) -> str:
    """Extract readable text from an HTML/HTM file.

    Requires ``beautifulsoup4`` + ``lxml`` (``pip install beautifulsoup4 lxml``).
    ``<script>``, ``<style>``, and ``<head>`` tags are removed before text
    extraction so JavaScript, CSS, and metadata don't pollute the output.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "beautifulsoup4 required for HTML ingestion. "
            "Install with: pip install beautifulsoup4 lxml"
        )
    raw = _read_text_auto(path)
    soup = BeautifulSoup(raw, features="lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _extract_csv(path: Path) -> str:
    """Extract text from CSV or TSV files using stdlib csv.

    The dialect (delimiter, quoting style) is auto-detected via
    ``csv.Sniffer`` on the first 4 KB.  For ``.tsv`` files the tab
    delimiter is forced regardless.  Each row is emitted as a
    tab-separated line; empty cells are omitted.
    """
    raw = _read_text_auto(path)
    suffix = path.suffix.lower()
    try:
        dialect = csv.Sniffer().sniff(raw[:4096]) if raw.strip() else None
    except csv.Error:
        dialect = None
    reader = csv.reader(
        io.StringIO(raw),
        delimiter="\t" if suffix == ".tsv" else (
            dialect.delimiter if dialect else ","
        ),
    )
    lines: list[str] = []
    for row in reader:
        cells = [cell.strip() for cell in row if cell.strip()]
        if cells:
            lines.append("  ".join(cells))
    return "\n".join(lines)


def _extract_docx(path: Path) -> str:
    """Extract text from DOCX using python-docx (optional dep)."""
    try:
        import docx as _docx  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "python-docx required for DOCX ingestion. "
            "Install with: pip install python-docx"
        )
    doc = _docx.Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_epub(path: Path) -> str:
    """Extract text from EPUB using ebooklib (optional dep)."""
    try:
        import ebooklib  # type: ignore[import]
        from ebooklib import epub  # type: ignore[import]
        from bs4 import BeautifulSoup  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "ebooklib and beautifulsoup4 required for EPUB ingestion. "
            "Install with: pip install ebooklib beautifulsoup4 lxml"
        )
    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    parts: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), features="lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _extract_xlsx(path: Path) -> str:
    """Extract text from XLSX/XLS using openpyxl (optional dep)."""
    try:
        import openpyxl  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "openpyxl required for XLSX ingestion. "
            "Install with: pip install openpyxl"
        )
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    parts: list[str] = []
    for sheet in wb.worksheets:
        rows: list[str] = []
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                rows.append("  ".join(cells))
        if rows:
            parts.append(f"[Sheet: {sheet.title}]\n" + "\n".join(rows))
    return "\n\n".join(parts)


def _extract_pptx(path: Path) -> str:
    """Extract text from PPTX using python-pptx (optional dep)."""
    try:
        from pptx import Presentation  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "python-pptx required for PPTX ingestion. "
            "Install with: pip install python-pptx"
        )
    prs = Presentation(str(path))
    slides: list[str] = []
    for i, slide in enumerate(prs.slides, 1):
        texts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        texts.append(t)
        if texts:
            slides.append(f"[Slide {i}]\n" + "\n".join(texts))
    return "\n\n".join(slides)


def _extract_odt(path: Path) -> str:
    """Extract text from ODT using odfpy (optional dep)."""
    try:
        from odf import teletype  # type: ignore[import]
        from odf.opendocument import load as odf_load  # type: ignore[import]
        from odf.text import P  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "odfpy required for ODT ingestion. "
            "Install with: pip install odfpy"
        )
    doc = odf_load(str(path))
    paragraphs = doc.getElementsByType(P)
    return "\n\n".join(
        teletype.extractText(p) for p in paragraphs
        if teletype.extractText(p).strip()
    )


def _extract_rtf(path: Path) -> str:
    """Extract text from RTF using striprtf (optional dep)."""
    try:
        from striprtf.striprtf import rtf_to_text  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "striprtf required for RTF ingestion. "
            "Install with: pip install striprtf"
        )
    raw = _read_text_auto(path)
    return rtf_to_text(raw)


def _extract_xml(path: Path) -> str:
    """Extract text from XML using stdlib ElementTree."""
    raw = _read_text_auto(path)
    tree = ElementTree.fromstring(raw.encode("utf-8", errors="replace"))  # noqa: S314
    texts = [t.strip() for t in tree.itertext() if t.strip()]
    return "\n".join(texts)


def _extract_eml(path: Path) -> str:
    """Extract text from email (.eml) files using stdlib email module."""
    raw = path.read_bytes()
    msg = email.message_from_bytes(raw)
    subject = msg.get("Subject", "")
    sender = msg.get("From", "")
    parts: list[str] = []
    if subject:
        parts.append(f"Subject: {subject}")
    if sender:
        parts.append(f"From: {sender}")
    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                try:
                    parts.append(payload.decode(charset, errors="replace"))
                except LookupError:
                    parts.append(payload.decode("utf-8", errors="replace"))
    return "\n\n".join(parts)


def _extract_mbox(path: Path) -> str:
    """Extract text from mbox files using stdlib mailbox module."""
    mbox = mailbox.mbox(str(path), create=False)
    parts: list[str] = []
    try:
        for i, msg in enumerate(mbox):
            subject = msg.get("Subject", f"message_{i}")
            sender = msg.get("From", "")
            body_parts: list[str] = []
            if hasattr(msg, "walk"):
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or "utf-8"
                            try:
                                body_parts.append(
                                    payload.decode(charset, errors="replace")
                                )
                            except LookupError:
                                body_parts.append(
                                    payload.decode("utf-8", errors="replace")
                                )
                body = "\n".join(body_parts).strip()
            if body:
                parts.append(f"[{sender}: {subject}]\n{body}")
    finally:
        mbox.close()
    return "\n\n".join(parts)


def _extract_tex(path: Path) -> str:
    """Extract text from LaTeX by stripping commands."""
    raw = _read_text_auto(path)
    # Remove comments.
    text = re.sub(r"%.*$", "", raw, flags=re.MULTILINE)
    # Expand \command{arg} → arg, \command[opt]{arg} → arg.
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^}]*)\}", r"\2", text)
    # Remove standalone commands and bare braces.
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    text = re.sub(r"[{}]", "", text)
    # Collapse excessive whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_fb2(path: Path) -> str:
    """Extract text from FictionBook 2 (.fb2) XML ebook."""
    raw = _read_text_auto(path)
    try:
        tree = ElementTree.fromstring(raw.encode("utf-8", errors="replace"))  # noqa: S314
    except ElementTree.ParseError as exc:
        raise ValueError(f"Invalid FB2 file {path}: {exc}") from exc
    texts = [t.strip() for t in tree.itertext() if t.strip()]
    return "\n".join(texts)


def _extract_source(path: Path) -> str:
    """Read source code and config files as plain text."""
    return _read_text_auto(path)


# ── Archive helpers ────────────────────────────────────────────────────────

# Archive suffixes that must not be recursed into (to prevent infinite loops).
_ARCHIVE_EXTS: frozenset[str] = frozenset({".zip", ".tar", ".gz", ".bz2", ".tgz"})


def _extract_zip(path: Path) -> str:
    """Recursively extract text from all supported files inside a ZIP.

    Each member is written to a temporary file so format-specific
    extractors (which require a real ``Path``) can operate on it.  The
    temporary file is always deleted in a ``finally`` block.  Nested
    archives (zip-in-zip) are skipped via ``_ARCHIVE_EXTS`` to prevent
    infinite recursion.  Members without a matching extractor are silently
    ignored.
    """
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a valid ZIP file: {path}")
    parts: list[str] = []
    with zipfile.ZipFile(path, "r") as zf:
        for name in sorted(zf.namelist()):
            info = zf.getinfo(name)
            if info.is_dir():
                continue
            suffix = Path(name).suffix.lower()
            if suffix in _ARCHIVE_EXTS:
                continue
            extractor = _EXTRACTORS.get(suffix)
            if extractor is None:
                continue
            with zf.open(name) as member:
                raw_bytes = member.read()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = Path(tmp.name)
            try:
                text = extractor(tmp_path)
                if text.strip():
                    parts.append(f"[{name}]\n{text}")
            except Exception as exc:
                logger.warning("Failed to extract %s from %s: %s", name, path, exc)
            finally:
                tmp_path.unlink(missing_ok=True)
    return "\n\n".join(parts)


def _extract_tar(path: Path) -> str:
    """Recursively extract text from all supported files inside a TAR archive.

    Handles all compression variants supported by ``tarfile.open`` in
    ``r:*`` mode (uncompressed, gzip, bzip2, xz).  Same temp-file and
    skip-nested-archive strategy as ``_extract_zip``.
    """
    parts: list[str] = []
    with tarfile.open(path, "r:*") as tf:
        for member in sorted(tf.getmembers(), key=lambda m: m.name):
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix in _ARCHIVE_EXTS:
                continue
            extractor = _EXTRACTORS.get(suffix)
            if extractor is None:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            raw_bytes = f.read()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = Path(tmp.name)
            try:
                text = extractor(tmp_path)
                if text.strip():
                    parts.append(f"[{member.name}]\n{text}")
            except Exception as exc:
                logger.warning(
                    "Failed to extract %s from %s: %s", member.name, path, exc
                )
            finally:
                tmp_path.unlink(missing_ok=True)
    return "\n\n".join(parts)


# Dispatch table: file extension (lowercase, with leading dot) → extractor
# function.  Each extractor takes a ``Path`` and returns a plain-text string.
#
# To add a new format:
#   1. Write a  def _extract_<fmt>(path: Path) -> str  function above.
#   2. Add one or more extension → function entries in this dict.
#   3. If the format requires an optional dependency, raise ``ImportError``
#      with a ``pip install …`` hint inside the function.
#
# Extensions NOT listed here (but known) live in ``_TIER3_MESSAGES`` with
# human-readable installation guidance.
_EXTRACTORS: dict[str, Any] = {
    # Plain text
    ".txt": _extract_text,
    ".md": _extract_text,
    ".markdown": _extract_text,
    ".rst": _extract_text,
    ".log": _extract_text,
    # Documents
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".odt": _extract_odt,
    ".rtf": _extract_rtf,
    ".epub": _extract_epub,
    ".fb2": _extract_fb2,
    # Office data
    ".xlsx": _extract_xlsx,
    ".xls": _extract_xlsx,
    ".pptx": _extract_pptx,
    # Data / markup
    ".json": _extract_json,
    ".jsonl": _extract_jsonl,
    ".csv": _extract_csv,
    ".tsv": _extract_csv,
    ".xml": _extract_xml,
    ".html": _extract_html,
    ".htm": _extract_html,
    # Email
    ".eml": _extract_eml,
    ".mbox": _extract_mbox,
    # Academic / writing
    ".tex": _extract_tex,
    # Source code & config
    ".py": _extract_source,
    ".js": _extract_source,
    ".ts": _extract_source,
    ".go": _extract_source,
    ".rs": _extract_source,
    ".java": _extract_source,
    ".c": _extract_source,
    ".cpp": _extract_source,
    ".h": _extract_source,
    ".hpp": _extract_source,
    ".sh": _extract_source,
    ".rb": _extract_source,
    ".swift": _extract_source,
    ".kt": _extract_source,
    ".scala": _extract_source,
    ".toml": _extract_source,
    ".yaml": _extract_source,
    ".yml": _extract_source,
    ".ini": _extract_source,
    ".cfg": _extract_source,
    ".conf": _extract_source,
    # Archives
    ".zip": _extract_zip,
    ".tar": _extract_tar,
    ".gz": _extract_tar,
    ".bz2": _extract_tar,
    ".tgz": _extract_tar,
}

# Known-but-unsupported formats — produce a helpful hint instead of a
# generic "unsupported format" error.
_TIER3_MESSAGES: dict[str, str] = {
    ".doc": (
        "Legacy Word (.doc) requires LibreOffice conversion. "
        "Convert first: soffice --convert-to docx file.doc"
    ),
    ".ppt": (
        "Legacy PowerPoint (.ppt) requires LibreOffice conversion. "
        "Convert first: soffice --convert-to pptx file.ppt"
    ),
    ".mobi": (
        ".mobi files have no free pure-Python support. "
        "Convert to .epub via Calibre: ebook-convert file.mobi file.epub"
    ),
    ".azw": (
        ".azw Kindle files are typically DRM-encumbered. "
        "Dedrm and convert to .epub first using Calibre."
    ),
    ".azw3": (
        ".azw3 Kindle files are typically DRM-encumbered. "
        "Dedrm and convert to .epub first using Calibre."
    ),
    ".djvu": (
        ".djvu requires the djvulibre system package (no pure-Python support). "
        "Convert: djvutxt file.djvu > file.txt"
    ),
    ".msg": (
        ".msg Outlook messages require the extract-msg package. "
        "Install with: pip install extract-msg"
    ),
}

# All extensions the ingester is aware of (Tier 1 + Tier 2 + Tier 3).
ALL_KNOWN_EXTENSIONS: frozenset[str] = frozenset(
    set(_EXTRACTORS.keys()) | set(_TIER3_MESSAGES.keys())
)



# ─────────────────────────────────────────────────────────────────────────────
# Content-based dedup hash
# ─────────────────────────────────────────────────────────────────────────────

def _content_hash(path: Path) -> str:
    """Stable 12-hex-char hash based on file content, not path.

    Uses the first ``_CONTENT_HASH_READ_BYTES`` (64 KB) of the file plus
    the total file size.  This is fast even for large files and catches
    same-file-different-path duplicates that a path-based hash would miss.
    """
    h = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as fh:
        h.update(fh.read(_CONTENT_HASH_READ_BYTES))
    h.update(str(size).encode())
    return h.hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping chunks, respecting paragraph boundaries.

    Strategy
    --------
    1. The text is first split on double newlines (``\\n\\n``) to obtain
       natural paragraph units.  Paragraphs are never broken mid-way.
    2. Paragraphs are accumulated into a *current* buffer.  When adding
       the next paragraph would exceed *chunk_size* characters, the buffer
       is flushed as a chunk and the **last paragraph** of the flushed
       chunk is retained as the start of the next buffer (overlap).
    3. If the retained paragraph is longer than *overlap* characters it is
       discarded rather than kept (avoids a single giant paragraph
       dominating every chunk boundary).
    4. **Character-level fallback** — if the text has no paragraph breaks
       at all (e.g. a single long sentence), the function falls back to a
       sliding-window character-level split with a step of
       ``chunk_size - overlap``.

    Args:
        text:       Raw extracted text to split.
        chunk_size: Target maximum characters per chunk (default
                    ``CHUNK_SIZE`` = 800).  Soft limit — a single
                    paragraph longer than this will not be split.
        overlap:    Characters of overlap between consecutive chunks
                    (default ``CHUNK_OVERLAP`` = 100).

    Returns:
        List of non-empty chunk strings.  Empty input → ``[]``.
    """
    # First split on double newlines (paragraphs).
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Keep last paragraph for overlap.
            if overlap > 0 and current:
                last = current[-1]
                current = [last] if len(last) <= overlap else []
                current_len = len(last) if current else 0
            else:
                current = []
                current_len = 0

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    # If no paragraph breaks, fall back to character-level chunking.
    if not chunks and text.strip():
        text = text.strip()
        pos = 0
        while pos < len(text):
            end = min(pos + chunk_size, len(text))
            chunks.append(text[pos:end])
            pos = end - overlap if overlap > 0 else end

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Ingester
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Dedup config helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEDUP_THRESHOLD_DEFAULT = 0.97
_DEDUP_MIN_CHARS_DEFAULT = 500


def _dedup_config() -> tuple[float, int]:
    """Return (similarity_threshold, min_chars) from vox_romana.yaml, or defaults."""
    try:
        from vox_romana.core.paths import _load_config  # type: ignore[attr-defined]
        cfg = _load_config().get("dedup") or {}
        threshold = float(cfg.get("similarity_threshold", _DEDUP_THRESHOLD_DEFAULT))
        min_chars = int(cfg.get("min_chars", _DEDUP_MIN_CHARS_DEFAULT))
        return threshold, min_chars
    except Exception:
        return _DEDUP_THRESHOLD_DEFAULT, _DEDUP_MIN_CHARS_DEFAULT


class DocumentIngester:
    """Ingests user documents into chunked JSONL for corpus building.

    Args:
        output_dir: Directory to write chunked JSONL files.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        near_dedup_threshold: Cosine similarity above which a document is
            considered a near-duplicate of an already-ingested document and
            skipped.  Set to ``>1.0`` to disable.  Defaults to the
            ``dedup.similarity_threshold`` value in *vox_romana.yaml*.
    """

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        near_dedup_threshold: float | None = None,
        fingerprint_store_path: str | Path | None = None,
    ) -> None:
        self._output_dir = Path(output_dir or DEFAULT_DOCUMENT_DIR)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        cfg_threshold, self._near_dedup_min_chars = _dedup_config()
        self._near_dedup_threshold = (
            near_dedup_threshold if near_dedup_threshold is not None else cfg_threshold
        )
        self._fp_store_path = Path(fingerprint_store_path) if fingerprint_store_path else None
        self._fp_store: Any = None  # lazy-loaded FingerprintStore

    def ingest(
        self,
        path: str | Path,
        *,
        title: str = "",
        dry_run: bool = False,
        force_reingest: bool = False,
        tag: str = "",
    ) -> IngestStats:
        """Ingest a single document.

        Args:
            path: Path to the document.
            title: Optional title override (defaults to filename stem).
            dry_run: If True, extract and chunk but do not write output.
            force_reingest: If True, overwrite an existing JSONL for this path.
            tag: Optional tag string embedded in every chunk.

        Returns:
            IngestStats summarising the ingestion.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()

        # Tier 3 — known-but-unsupported: warn and skip gracefully.
        tier3_msg = _TIER3_MESSAGES.get(suffix)
        if tier3_msg:
            logger.warning("Skipping %s: %s", path, tier3_msg)
            return IngestStats(
                source=str(path),
                format=suffix,
                skipped=True,
                skip_reason=tier3_msg,
            )

        extractor = _EXTRACTORS.get(suffix)
        if extractor is None:
            raise ValueError(
                f"Unsupported format: {suffix!r}. "
                f"Supported: {', '.join(sorted(_EXTRACTORS))}"
            )

        # Idempotency — skip if already ingested (unless force_reingest).
        # Hash is based on file *content* (first 64 KB + file size) rather than
        # path so that the same document ingested from a different location is
        # correctly identified as a duplicate.
        doc_hash = _content_hash(path)
        out_path = self._output_dir / f"{doc_hash}.jsonl"
        if out_path.exists() and not force_reingest and not dry_run:
            logger.debug("Already ingested (skipping): %s", path)
            return IngestStats(
                source=str(path),
                format=suffix,
                skipped=True,
                skip_reason="already ingested",
            )

        text = extractor(path)
        if not text.strip():
            logger.warning("Empty document: %s", path)
            return IngestStats(source=str(path), format=suffix)

        # Near-duplicate guard (document-level, using Markov fingerprint).
        # Skipped when: threshold > 1.0, dry_run, force_reingest, doc is too short, or store unavailable.
        if (
            not dry_run
            and not force_reingest
            and self._near_dedup_threshold <= 1.0
            and len(text) >= self._near_dedup_min_chars
        ):
            match = self._check_near_dup(text)
            if match is not None:
                near_source, sim = match
                logger.info(
                    "Near-duplicate of %s (sim=%.4f ≥ %.4f) — skipping: %s",
                    near_source,
                    sim,
                    self._near_dedup_threshold,
                    path,
                )
                return IngestStats(
                    source=str(path),
                    format=suffix,
                    skipped=True,
                    skip_reason=f"near-duplicate of {near_source} (sim={sim:.4f})",
                )

        title = title or path.stem
        chunks = chunk_text(
            text,
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )
        now = datetime.now(timezone.utc).isoformat()

        doc_chunks = [
            DocumentChunk(
                text=chunk,
                title=title,
                source=str(path),
                chunk_index=i,
                total_chunks=len(chunks),
                format=suffix,
                ingested_at=now,
                tag=tag,
            )
            for i, chunk in enumerate(chunks)
        ]

        stats = IngestStats(
            source=str(path),
            chunks=len(doc_chunks),
            chars=sum(len(c.text) for c in doc_chunks),
            format=suffix,
        )

        if dry_run:
            logger.info("Dry-run %s: would write %d chunks", path, stats.chunks)
            stats.skipped = True
            stats.skip_reason = "dry run"
            return stats

        deduped = self._write_chunks(doc_hash, doc_chunks)
        stats.deduped_chunks = deduped
        stats.chunks -= deduped
        # Register fingerprint so future near-duplicate docs are caught.
        self._register_fingerprint(str(path), text)
        logger.info(
            "Ingested %s: %d chunks, %d chars (%d chunk-level dupes skipped)",
            path,
            stats.chunks,
            stats.chars,
            deduped,
        )
        return stats

    def ingest_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
        dry_run: bool = False,
        force_reingest: bool = False,
        tag: str = "",
    ) -> list[IngestStats]:
        """Ingest all supported documents in a directory.

        Args:
            directory: Path to scan.
            recursive: If True, scan subdirectories too.
            dry_run: If True, report without writing.
            force_reingest: If True, re-ingest already-processed files.
            tag: Optional tag embedded in every chunk.

        Returns:
            List of IngestStats, one per file attempted.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        results: list[IngestStats] = []
        pattern = "**/*" if recursive else "*"

        for path in sorted(directory.glob(pattern)):
            if path.is_file() and path.suffix.lower() in ALL_KNOWN_EXTENSIONS:
                try:
                    stats = self.ingest(
                        path,
                        dry_run=dry_run,
                        force_reingest=force_reingest,
                        tag=tag,
                    )
                    results.append(stats)
                except Exception as e:
                    logger.warning("Failed to ingest %s: %s", path, e)
                    results.append(
                        IngestStats(
                            source=str(path),
                            format=path.suffix.lower(),
                            skipped=True,
                            skip_reason=f"error: {e}",
                            failed=True,
                        )
                    )

        return results

    def _get_fp_store(self):
        """Lazy-load FingerprintStore, return None if unavailable."""
        if self._fp_store is None:
            try:
                from vox_romana.memory.fingerprint_store import FingerprintStore
                # Default: store lives alongside the JSONL corpus files so that
                # each output directory maintains its own dedup index.
                path = self._fp_store_path or (self._output_dir / "fingerprints.npz")
                self._fp_store = FingerprintStore(store_path=path)
            except Exception as exc:
                logger.debug("FingerprintStore unavailable: %s", exc)
                self._fp_store = False  # sentinel: don't retry
        return self._fp_store if self._fp_store is not False else None

    def _check_near_dup(self, text: str) -> tuple[str, float] | None:
        """Return (source, similarity) if *text* is a near-duplicate, else None."""
        store = self._get_fp_store()
        if store is None:
            return None
        try:
            return store.find_nearest(text, threshold=self._near_dedup_threshold)
        except Exception as exc:
            logger.debug("FingerprintStore.find_nearest failed: %s", exc)
            return None

    def _register_fingerprint(self, source: str, text: str) -> None:
        """Add document fingerprint to store (best-effort, never raises)."""
        store = self._get_fp_store()
        if store is None:
            return
        try:
            store.add(source, text)
        except Exception as exc:
            logger.debug("FingerprintStore.add failed: %s", exc)

    def _chunk_hashes_path(self) -> Path:
        return self._output_dir / ".chunk_hashes.json"

    def _load_chunk_hashes(self) -> set[str]:
        p = self._chunk_hashes_path()
        if p.exists():
            try:
                return set(json.loads(p.read_text()))
            except Exception:
                return set()
        return set()

    def _save_chunk_hashes(self, hashes: set[str]) -> None:
        self._chunk_hashes_path().write_text(json.dumps(sorted(hashes)))

    def _write_chunks(self, doc_hash: str, chunks: list[DocumentChunk]) -> int:
        """Write chunks to JSONL, skipping exact duplicates already seen.

        Returns the number of chunks that were deduplicated (skipped).
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        known = self._load_chunk_hashes()
        deduped = 0
        new_hashes: set[str] = set()
        path = self._output_dir / f"{doc_hash}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                # Normalise before hashing: collapse whitespace, strip.
                normalised = " ".join(chunk.text.split())
                ch = hashlib.sha256(normalised.encode()).hexdigest()[:16]
                if ch in known:
                    deduped += 1
                    continue
                new_hashes.add(ch)
                f.write(json.dumps(chunk.to_dict(), default=str) + "\n")
        if new_hashes:
            self._save_chunk_hashes(known | new_hashes)
        return deduped

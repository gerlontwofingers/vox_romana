"""Document ingestion — extracts text from user-provided files.

Supported formats:
    .txt, .md   — Plain text (read as-is).
    .pdf        — Requires PyPDF2 (optional, in requirements-data.txt).
    .json       — Reads a "text" field or concatenates string values.

Each document is chunked into overlapping paragraphs with metadata,
then written as JSONL to ``data/memory/documents/``.  The corpus builder
reads these JSONL files as training material.

USAGE:
    from vox_romana.memory.ingest import DocumentIngester

    ingester = DocumentIngester()
    stats = ingester.ingest("papers/ubi_pilot_2024.pdf")
    stats = ingester.ingest_directory("papers/")
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DOCUMENT_DIR = Path("data/memory/documents")

# Chunking defaults.
CHUNK_SIZE = 800       # target chars per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks


@dataclass
class IngestStats:
    """Summary of a single ingest operation."""
    source: str
    chunks: int = 0
    chars: int = 0
    format: str = ""


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Text extractors (per format)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(path: Path) -> str:
    """Read plain text or markdown."""
    return path.read_text(encoding="utf-8", errors="replace")


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
    """Extract text from a JSON file (looks for 'text' field)."""
    with open(path) as f:
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


_EXTRACTORS: dict[str, Any] = {
    ".txt": _extract_text,
    ".md": _extract_text,
    ".markdown": _extract_text,
    ".pdf": _extract_pdf,
    ".json": _extract_json,
}


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries."""
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

class DocumentIngester:
    """Ingests user documents into chunked JSONL for corpus building.

    Args:
        output_dir: Directory to write chunked JSONL files.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
    """

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self._output_dir = Path(output_dir or DEFAULT_DOCUMENT_DIR)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, path: str | Path, *, title: str = "") -> IngestStats:
        """Ingest a single document.

        Args:
            path: Path to the document.
            title: Optional title override (defaults to filename).

        Returns:
            IngestStats summarizing the ingestion.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()
        extractor = _EXTRACTORS.get(suffix)
        if extractor is None:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported: {', '.join(_EXTRACTORS)}"
            )

        text = extractor(path)
        if not text.strip():
            logger.warning("Empty document: %s", path)
            return IngestStats(source=str(path), format=suffix)

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
            )
            for i, chunk in enumerate(chunks)
        ]

        # Write to JSONL (one file per source document, keyed by content hash).
        doc_hash = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        self._write_chunks(doc_hash, doc_chunks)

        stats = IngestStats(
            source=str(path),
            chunks=len(doc_chunks),
            chars=sum(len(c.text) for c in doc_chunks),
            format=suffix,
        )
        logger.info(
            "Ingested %s: %d chunks, %d chars",
            path, stats.chunks, stats.chars,
        )
        return stats

    def ingest_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
    ) -> list[IngestStats]:
        """Ingest all supported documents in a directory.

        Args:
            directory: Path to scan.
            recursive: If True, scan subdirectories too.

        Returns:
            List of IngestStats, one per file processed.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        results: list[IngestStats] = []
        pattern = "**/*" if recursive else "*"

        for path in sorted(directory.glob(pattern)):
            if path.is_file() and path.suffix.lower() in _EXTRACTORS:
                try:
                    stats = self.ingest(path)
                    results.append(stats)
                except Exception as e:
                    logger.warning("Failed to ingest %s: %s", path, e)

        return results

    def _write_chunks(self, doc_hash: str, chunks: list[DocumentChunk]) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"{doc_hash}.jsonl"
        with open(path, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), default=str) + "\n")

"""Vector index backends for vox_romana memory recall.

Pipeline position
-----------------
This module sits in **Phase 2** of the ingest pipeline::

    Phase 1 — ingest.py                Phase 2 — vector_index.py
    ──────────────────────             ──────────────────────────
    DocumentIngester                   ChromaVectorIndex
    └─→ JSONL chunks written           └─→ migrate_documents()
        to data/memory/                    reads JSONL, embeds chunks,
        documents/                         persists to Chroma on disk
                                                   │
                                                   ▼
                                       Phase 3 — recall.py
                                       MemoryRecall.query()
                                       └─→ MemorySearchProvider
                                           (used by all bootstraps
                                            when --research is active)

Populated by ``DocumentIngester`` via ``python run.py --mode ingest``.

Two backends sharing the same ``search()`` interface:

    ChromaVectorIndex — persistent Chroma-backed cosine similarity index.
        Survives process restarts.  Auto-migrates ingested JSONL chunks from
        ``data/memory/documents/`` on first use (via a marker file so chunks
        are only embedded once).
        Requires: chromadb  (pip install chromadb)
        Embedding: Chroma's built-in DefaultEmbeddingFunction
                   (all-MiniLM-L6-v2 via onnxruntime — no sentence-transformers).

    _CosineFallbackIndex — pure-Python/numpy in-memory cosine index.
        Rebuilt from scratch on each process start.
        Used when chromadb is not installed or when an explicit encoder is
        preferred over Chroma.

Both expose:
    add(text, source_id, source_type)       add a single entry
    search(query, *, top_k) → list          cosine similarity search
    size / count                            number of entries

ChromaVectorIndex additionally provides:
    upsert_document_chunk(chunk)            embed a DocumentChunk
    add_catalog_entries(entries)            batch-add catalog rows
    migrate_documents(documents_dir)        load ingested JSONL into Chroma
    delete_by_source(source)               remove chunks by source path
    is_available()                          classmethod — chromadb importable?
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from vox_romana.core.paths import CHROMA_DIR as DEFAULT_CHROMA_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Fallback in-memory cosine index (no chromadb required)
# ─────────────────────────────────────────────────────────────────────────────

class _CosineFallbackIndex:
    """In-memory cosine similarity index over encoder embeddings.

    Rebuilt on every process start — no persistence.
    Used when chromadb is not installed or when a custom encoder is provided.
    """

    def __init__(self, encoder: Any) -> None:
        self._encoder = encoder
        self._texts: list[str] = []
        self._source_ids: list[str] = []
        self._embeddings: Any = None  # numpy (n, dim) or None

    @property
    def size(self) -> int:
        return len(self._texts)

    def add(self, text: str, source_id: str, source_type: str = "") -> None:
        self._texts.append(text)
        self._source_ids.append(source_id)

    def build(self) -> None:
        """Encode all added texts and normalise for cosine similarity."""
        if not self._texts:
            return
        import numpy as np
        self._embeddings = self._encoder.encode(self._texts)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._embeddings = self._embeddings / norms

    def search(
        self, query: str, *, top_k: int = 10
    ) -> list[tuple[str, float, str]]:
        """Return top-k (text, cosine_similarity, source_id) tuples."""
        if self._embeddings is None or not self._texts:
            return []
        import numpy as np
        q_emb = self._encoder.encode(query)
        q_emb = q_emb / (np.linalg.norm(q_emb) or 1.0)
        scores = self._embeddings @ q_emb
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self._texts[idx], float(scores[idx]), self._source_ids[idx])
            for idx in top_indices
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Persistent Chroma-backed index
# ─────────────────────────────────────────────────────────────────────────────

class ChromaVectorIndex:
    """Persistent cosine-similarity index backed by Chroma.

    Uses a single Chroma collection.  All upserts are idempotent by stable
    doc_id, so running ingest or catalog indexing multiple times is safe.

    Embedding uses Chroma's built-in ``DefaultEmbeddingFunction``
    (all-MiniLM-L6-v2 via onnxruntime) — no sentence-transformers needed.

    JSONL document migration is tracked via ``.migrated_files.json`` in the
    persist directory, so chunks are only embedded on first encounter.

    Args:
        persist_dir:      Directory for Chroma SQLite + HNSW data.
        collection_name:  Chroma collection name (default: ``"memory"``).
        _client:          Injected Chroma client — use ``EphemeralClient()``
                          in tests to avoid disk writes.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        *,
        collection_name: str = "memory",
        _client: Any | None = None,
    ) -> None:
        self._persist_dir = Path(persist_dir or DEFAULT_CHROMA_DIR)
        self._collection_name = collection_name
        self._injected_client = _client
        self._client: Any = None
        self._collection: Any = None

    # ── Availability ────────────────────────────────────────────────────────

    @staticmethod
    def is_available() -> bool:
        """True if ``chromadb`` can be imported."""
        try:
            import chromadb  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Lazy init ───────────────────────────────────────────────────────────

    def _get_collection(self) -> Any:
        """Return (creating if needed) the Chroma collection."""
        if self._collection is not None:
            return self._collection
        import chromadb
        if self._injected_client is not None:
            self._client = self._injected_client
        else:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    # ── Size helpers ────────────────────────────────────────────────────────

    def count(self) -> int:
        """Number of entries currently in the collection."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    @property
    def size(self) -> int:
        return self.count()

    # ── Write API ───────────────────────────────────────────────────────────

    def upsert(
        self,
        text: str,
        doc_id: str,
        *,
        source_type: str = "",
        source: str = "",
        fmt: str = "",
        tag: str = "",
        title: str = "",
    ) -> None:
        """Upsert a single text entry.

        Args:
            text:        Text to embed and store.
            doc_id:      Stable unique Chroma ID for this entry.
            source_type: ``"catalog_run"``, ``"catalog_study"``, or
                         ``"document"``.
            source:      Original catalog ID or file path.  Returned by
                         ``search()`` as the third tuple element so that
                         ``MemoryRecall`` can look up catalog records.
            fmt:         File format (e.g. ``".pdf"``).
            tag:         Ingest run tag.
            title:       Document title.
        """
        col = self._get_collection()
        col.upsert(
            documents=[text],
            ids=[doc_id],
            metadatas=[{
                "source_type": source_type,
                "source": source,
                "format": fmt,
                "tag": tag,
                "title": title,
            }],
        )

    def add_catalog_entries(
        self,
        entries: list[tuple[str, str, str]],
    ) -> int:
        """Add catalog entries not yet present in the index.

        Uses a single batch ``col.get()`` to check existing IDs before
        adding, so only newly-seen entries are embedded.

        Args:
            entries: List of ``(text, catalog_id, source_type)`` tuples.
                     ``catalog_id`` is a ``run_id`` or ``study_id`` from the
                     catalog.  It is stored in ``metadata["source"]`` and
                     returned by ``search()``.

        Returns:
            Number of new entries actually added.
        """
        if not entries:
            return 0
        col = self._get_collection()
        # Stable Chroma IDs include a type prefix to avoid collisions.
        doc_ids = [f"cat:{e[2]}:{e[1]}" for e in entries]
        try:
            existing = set(col.get(ids=doc_ids, include=[])["ids"])
        except Exception:
            existing = set()
        new = [(e, did) for e, did in zip(entries, doc_ids) if did not in existing]
        if not new:
            return 0
        col.add(
            documents=[item[0][0] for item in new],
            ids=[item[1] for item in new],
            metadatas=[
                {
                    "source_type": item[0][2],
                    "source": item[0][1],   # catalog_id → used in recall lookup
                    "format": "",
                    "tag": "",
                    "title": "",
                }
                for item in new
            ],
        )
        return len(new)

    def upsert_document_chunk(self, chunk: Any) -> None:
        """Embed and upsert a DocumentChunk (or _ChunkProxy).

        The Chroma doc_id is a SHA256 hash of ``source::chunk_index``,
        so the same chunk ingested twice results in one entry.
        """
        import hashlib
        id_str = f"{chunk.source}::chunk::{chunk.chunk_index}"
        doc_id = "doc:" + hashlib.sha256(id_str.encode()).hexdigest()[:20]
        self.upsert(
            chunk.text,
            doc_id,
            source_type="document",
            source=chunk.source,
            fmt=getattr(chunk, "format", ""),
            tag=getattr(chunk, "tag", ""),
            title=getattr(chunk, "title", ""),
        )

    # ── Document migration ───────────────────────────────────────────────────

    @property
    def _marker_path(self) -> Path:
        """Path to the JSON file tracking which JSONL files were migrated."""
        return self._persist_dir / ".migrated_files.json"

    def _load_marker(self) -> dict[str, int]:
        if self._injected_client is not None:
            return {}  # ephemeral client — no marker needed
        if self._marker_path.exists():
            try:
                return json.loads(self._marker_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_marker(self, marker: dict[str, int]) -> None:
        if self._injected_client is not None:
            return  # ephemeral client — skip marker to avoid disk side-effects
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._marker_path.write_text(
            json.dumps(marker, indent=2), encoding="utf-8"
        )

    def migrate_documents(
        self,
        documents_dir: str | Path,
        *,
        force: bool = False,
    ) -> dict[str, int]:
        """Load ingested JSONL chunks into Chroma.

        Only processes files not yet recorded in ``.migrated_files.json``.
        Safe to call on every startup — already-migrated files are skipped
        in O(1) via the marker file.

        Args:
            documents_dir: Directory containing JSONL from ``DocumentIngester``.
            force:         If True, re-migrate all files (ignores marker).

        Returns:
            ``{"loaded": N, "files": M, "skipped_files": K}``
        """
        documents_dir = Path(documents_dir)
        if not documents_dir.is_dir():
            return {"loaded": 0, "files": 0, "skipped_files": 0}

        marker = {} if force else self._load_marker()
        loaded = 0
        files_new = 0
        files_skipped = 0

        for jf in sorted(documents_dir.glob("*.jsonl")):
            if jf.name in marker and not force:
                files_skipped += 1
                continue
            files_new += 1
            file_count = 0
            with open(jf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        chunk = _ChunkProxy(data)
                        if chunk.text.strip():
                            self.upsert_document_chunk(chunk)
                            file_count += 1
                            loaded += 1
                    except Exception as exc:
                        logger.warning(
                            "Skipping malformed chunk in %s: %s", jf, exc
                        )
            marker[jf.name] = file_count

        if files_new > 0:
            self._save_marker(marker)
            logger.info(
                "Migrated %d chunks from %d JSONL file(s) into Chroma",
                loaded,
                files_new,
            )
        return {"loaded": loaded, "files": files_new, "skipped_files": files_skipped}

    # ── Search API ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_type_filter: str | None = None,
    ) -> list[tuple[str, float, str]]:
        """Find top-k most similar chunks.

        Args:
            query:               Text query.
            top_k:               Maximum results.
            source_type_filter:  Optional ``source_type`` equality filter.

        Returns:
            List of ``(text, cosine_similarity, source)`` sorted by
            similarity descending.

            - ``cosine_similarity = 1 − chroma_cosine_distance ∈ [−1, 1]``
            - ``source`` = ``metadata["source"]`` — the catalog run/study ID
              or file path (NOT the Chroma internal doc_id).
        """
        col = self._get_collection()
        total = col.count()
        if total == 0:
            return []
        n = min(top_k, total)
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": n,
            "include": ["documents", "distances", "metadatas"],
        }
        if source_type_filter:
            kwargs["where"] = {"source_type": source_type_filter}
        try:
            results = col.query(**kwargs)
        except Exception as exc:
            logger.warning("Chroma query failed: %s", exc)
            return []

        output: list[tuple[str, float, str]] = []
        for text, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            # Chroma cosine distance ∈ [0, 2] → similarity ∈ [−1, 1].
            similarity = 1.0 - dist
            source = meta.get("source", "")
            output.append((text, similarity, source))
        return output

    # ── Deletion ─────────────────────────────────────────────────────────────

    def delete_by_source(self, source: str) -> None:
        """Remove all entries whose ``metadata["source"] == source``."""
        col = self._get_collection()
        col.delete(where={"source": source})


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

class _ChunkProxy:
    """Minimal duck-type for a DocumentChunk built from a raw dict."""

    __slots__ = ("text", "source", "chunk_index", "format", "tag", "title")

    def __init__(self, data: dict[str, Any]) -> None:
        self.text = data.get("text", "")
        self.source = data.get("source", "")
        self.chunk_index = data.get("chunk_index", 0)
        self.format = data.get("format", "")
        self.tag = data.get("tag", "")
        self.title = data.get("title", "")

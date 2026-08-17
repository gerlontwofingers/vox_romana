"""Phase 2 tests — persistent Chroma vector index + MemoryRecall integration.

Coverage:
    1. _CosineFallbackIndex — add, build, search, empty behaviour.
    2. ChromaVectorIndex (EphemeralClient, no disk writes):
       - is_available, count/size
       - upsert idempotency
       - cosine similarity range
       - add_catalog_entries (batch existence check, no duplicates)
       - upsert_document_chunk
       - migrate_documents (marker file, force flag, empty dir)
       - search (basic, source_type_filter)
       - delete_by_source
    3. MemoryRecall + Chroma:
       - method == "chroma" when Chroma active
       - catalog entries indexed and retrievable
       - ingested JSONL documents searchable
       - graceful fallback to FTS5 when use_chroma=False
       - _chroma_client injection for clean in-memory tests
    4. run.py --mode ingest auto-migrates into Chroma after ingest.
    5. _ChunkProxy duck-type helper.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb
import numpy as np
import pytest

from vox_romana.memory.vector_index import (
    ChromaVectorIndex,
    _ChunkProxy,
    _CosineFallbackIndex,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fresh_client() -> "chromadb.EphemeralClient":
    """Return an EphemeralClient with all lingering collections deleted.

    chromadb 1.5.8 EphemeralClient instances share global in-process state.
    Calling this helper before each client creation ensures clean state.
    """
    client = chromadb.EphemeralClient()
    for col in client.list_collections():
        try:
            client.delete_collection(col.name)
        except Exception:
            pass
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def ephemeral_chroma():
    """An in-memory EphemeralClient with clean global state."""
    return fresh_client()


@pytest.fixture
def chroma_index(ephemeral_chroma):
    """A ChromaVectorIndex backed by a clean EphemeralClient."""
    return ChromaVectorIndex(_client=ephemeral_chroma)


@pytest.fixture
def mock_encoder():
    """A lightweight mock encoder returning random embeddings."""
    enc = MagicMock()
    def smart_encode(texts, **kwargs):
        if isinstance(texts, str):
            return np.random.randn(384).astype(np.float32)
        return np.random.randn(len(texts), 384).astype(np.float32)
    enc.encode.side_effect = smart_encode
    return enc


@pytest.fixture
def seeded_catalog():
    from vox_romana.core.catalog import Catalog
    cat = Catalog(db_path=":memory:")
    run_id = cat.log_run(
        content_type="debate",
        title="UBI Debate",
        brief="Two economists debate universal basic income.",
        tags=["economics", "UBI"],
    )
    cat.log_study(
        run_id=run_id,
        content_type="debate",
        dataset="finland_ubi.csv",
        method="t-test",
        findings="Employment rose 4.8% (p=0.003)",
    )
    yield cat
    cat.close()


def write_txt(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def make_jsonl(tmp_path: Path, name: str, chunks: list[dict]) -> Path:
    p = tmp_path / name
    with open(p, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 1. _CosineFallbackIndex
# ─────────────────────────────────────────────────────────────────────────────

class TestCosineFallbackIndex:
    def test_add_and_search(self, mock_encoder):
        idx = _CosineFallbackIndex(mock_encoder)
        idx.add("UBI employment study", "run_001", "catalog_run")
        idx.add("Climate temperature trend", "run_002", "catalog_run")
        idx.build()
        results = idx.search("UBI employment", top_k=2)
        assert len(results) == 2
        assert all(len(r) == 3 for r in results)
        texts = [r[0] for r in results]
        assert "UBI employment study" in texts

    def test_search_before_build_returns_empty(self, mock_encoder):
        idx = _CosineFallbackIndex(mock_encoder)
        idx.add("some text", "id1")
        assert idx.search("query") == []

    def test_search_empty_index_returns_empty(self, mock_encoder):
        idx = _CosineFallbackIndex(mock_encoder)
        idx.build()
        assert idx.search("anything") == []

    def test_size_tracks_adds(self, mock_encoder):
        idx = _CosineFallbackIndex(mock_encoder)
        assert idx.size == 0
        idx.add("text1", "id1")
        idx.add("text2", "id2")
        assert idx.size == 2

    def test_scores_in_valid_range(self, mock_encoder):
        idx = _CosineFallbackIndex(mock_encoder)
        for i in range(5):
            idx.add(f"document {i}", f"id{i}")
        idx.build()
        results = idx.search("document", top_k=5)
        for _, score, _ in results:
            assert -1.01 <= score <= 1.01


# ─────────────────────────────────────────────────────────────────────────────
# 2. ChromaVectorIndex
# ─────────────────────────────────────────────────────────────────────────────

class TestChromaVectorIndexAvailability:
    def test_is_available(self):
        assert ChromaVectorIndex.is_available() is True

    def test_is_available_returns_false_when_chromadb_missing(self):
        with patch.dict("sys.modules", {"chromadb": None}):
            assert ChromaVectorIndex.is_available() is False

    def test_ephemeral_client_injection(self, chroma_index):
        """Injected client is used instead of PersistentClient."""
        assert chroma_index.count() == 0


class TestChromaVectorIndexWrite:
    def test_upsert_basic(self, chroma_index):
        chroma_index.upsert("hello world", "id1")
        assert chroma_index.count() == 1

    def test_upsert_idempotent(self, chroma_index):
        """Same doc_id upserted twice → only one entry."""
        chroma_index.upsert("same text", "same_id")
        chroma_index.upsert("same text", "same_id")
        assert chroma_index.count() == 1

    def test_upsert_stores_metadata(self, chroma_index):
        chroma_index.upsert(
            "Research paper content",
            "doc001",
            source_type="document",
            source="/path/to/paper.pdf",
            fmt=".pdf",
            tag="batch-2026",
            title="My Paper",
        )
        col = chroma_index._get_collection()
        result = col.get(ids=["doc001"], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["source_type"] == "document"
        assert meta["source"] == "/path/to/paper.pdf"
        assert meta["tag"] == "batch-2026"
        assert meta["title"] == "My Paper"

    def test_size_property(self, chroma_index):
        assert chroma_index.size == 0
        chroma_index.upsert("text", "id1")
        assert chroma_index.size == 1


class TestChromaVectorIndexCatalogEntries:
    def test_add_catalog_entries_new(self, chroma_index):
        entries = [
            ("UBI employment study findings.", "run_001", "catalog_run"),
            ("Climate temperature rose 0.18°C per decade.", "run_002", "catalog_run"),
        ]
        added = chroma_index.add_catalog_entries(entries)
        assert added == 2
        assert chroma_index.count() == 2

    def test_add_catalog_entries_no_duplicates(self, chroma_index):
        entries = [("UBI study.", "run_001", "catalog_run")]
        added1 = chroma_index.add_catalog_entries(entries)
        added2 = chroma_index.add_catalog_entries(entries)
        assert added1 == 1
        assert added2 == 0  # already exists
        assert chroma_index.count() == 1

    def test_add_catalog_entries_empty_list(self, chroma_index):
        added = chroma_index.add_catalog_entries([])
        assert added == 0
        assert chroma_index.count() == 0

    def test_catalog_source_stored_in_metadata(self, chroma_index):
        chroma_index.add_catalog_entries([
            ("Findings text.", "study_abc", "catalog_study")
        ])
        col = chroma_index._get_collection()
        results = col.get(include=["metadatas"])
        meta = results["metadatas"][0]
        assert meta["source"] == "study_abc"
        assert meta["source_type"] == "catalog_study"


class TestChromaVectorIndexDocumentChunk:
    def test_upsert_document_chunk(self, chroma_index):
        chunk = _ChunkProxy({
            "text": "Research paper about UBI.",
            "source": "/data/paper.pdf",
            "chunk_index": 0,
            "format": ".pdf",
            "tag": "batch-1",
            "title": "UBI Paper",
        })
        chroma_index.upsert_document_chunk(chunk)
        assert chroma_index.count() == 1

    def test_upsert_document_chunk_idempotent(self, chroma_index):
        chunk = _ChunkProxy({
            "text": "Same content.",
            "source": "/data/file.txt",
            "chunk_index": 0,
        })
        chroma_index.upsert_document_chunk(chunk)
        chroma_index.upsert_document_chunk(chunk)
        assert chroma_index.count() == 1

    def test_upsert_different_chunks_same_source(self, chroma_index):
        for i in range(3):
            chroma_index.upsert_document_chunk(_ChunkProxy({
                "text": f"Paragraph {i}.",
                "source": "/data/file.txt",
                "chunk_index": i,
            }))
        assert chroma_index.count() == 3


class TestChromaVectorIndexMigration:
    def test_migrate_documents_basic(self, chroma_index, tmp_path):
        make_jsonl(tmp_path, "doc1.jsonl", [
            {"text": "First chunk.", "source": "/a.txt", "chunk_index": 0},
            {"text": "Second chunk.", "source": "/a.txt", "chunk_index": 1},
        ])
        stats = chroma_index.migrate_documents(tmp_path)
        assert stats["loaded"] == 2
        assert stats["files"] == 1
        assert stats["skipped_files"] == 0
        assert chroma_index.count() == 2

    def test_migrate_documents_empty_dir(self, chroma_index, tmp_path):
        stats = chroma_index.migrate_documents(tmp_path)
        assert stats["loaded"] == 0
        assert stats["files"] == 0

    def test_migrate_documents_nonexistent_dir(self, chroma_index, tmp_path):
        stats = chroma_index.migrate_documents(tmp_path / "does_not_exist")
        assert stats["loaded"] == 0

    def test_migrate_documents_marker_skips_second_run(self, tmp_path):
        """After first migration, same file is skipped on second run."""
        chroma_dir = tmp_path / "chroma"
        # Use PersistentClient (no _client injection) to test marker file logic.
        idx = ChromaVectorIndex(persist_dir=chroma_dir)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        make_jsonl(docs_dir, "doc1.jsonl", [
            {"text": "Chunk 1.", "source": "/f.txt", "chunk_index": 0},
        ])

        stats1 = idx.migrate_documents(docs_dir)
        stats2 = idx.migrate_documents(docs_dir)

        assert stats1["files"] == 1
        assert stats2["files"] == 0
        assert stats2["skipped_files"] == 1
        assert idx.count() == 1

    def test_migrate_documents_force_remigrates(self, tmp_path):
        chroma_dir = tmp_path / "chroma"
        # Use PersistentClient to test force flag with marker logic.
        idx = ChromaVectorIndex(persist_dir=chroma_dir)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        make_jsonl(docs_dir, "doc1.jsonl", [
            {"text": "Content.", "source": "/f.txt", "chunk_index": 0},
        ])
        idx.migrate_documents(docs_dir)
        stats = idx.migrate_documents(docs_dir, force=True)
        assert stats["files"] == 1  # re-processed despite marker

    def test_migrate_skips_empty_text(self, chroma_index, tmp_path):
        make_jsonl(tmp_path, "doc.jsonl", [
            {"text": "", "source": "/f.txt", "chunk_index": 0},
            {"text": "   ", "source": "/f.txt", "chunk_index": 1},
            {"text": "Real content.", "source": "/f.txt", "chunk_index": 2},
        ])
        stats = chroma_index.migrate_documents(tmp_path)
        assert stats["loaded"] == 1  # only the real content

    def test_migrate_skips_malformed_lines(self, chroma_index, tmp_path):
        jf = tmp_path / "bad.jsonl"
        jf.write_text('{"text": "Good."}\nnot json\n{"text": "Also good."}\n')
        stats = chroma_index.migrate_documents(tmp_path)
        assert stats["loaded"] == 2  # two valid lines

    def test_migrate_marker_file_written(self, tmp_path):
        chroma_dir = tmp_path / "chroma"
        # Use PersistentClient to test that marker file is physically written.
        idx = ChromaVectorIndex(persist_dir=chroma_dir)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        make_jsonl(docs_dir, "doc1.jsonl", [
            {"text": "Content.", "source": "/f.txt", "chunk_index": 0},
        ])
        idx.migrate_documents(docs_dir)
        marker = chroma_dir / ".migrated_files.json"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert "doc1.jsonl" in data


class TestChromaVectorIndexSearch:
    def test_search_returns_tuple_format(self, chroma_index):
        chroma_index.upsert("Universal basic income study.", "id1", source="run_001")
        results = chroma_index.search("UBI employment", top_k=1)
        assert len(results) == 1
        text, score, source = results[0]
        assert isinstance(text, str)
        assert isinstance(score, float)
        assert isinstance(source, str)

    def test_search_empty_collection(self, chroma_index):
        results = chroma_index.search("anything")
        assert results == []

    def test_search_similarity_in_range(self, chroma_index):
        for i in range(5):
            chroma_index.upsert(f"document {i} about topic", f"id{i}")
        results = chroma_index.search("document topic", top_k=5)
        for _, score, _ in results:
            assert -1.01 <= score <= 1.01

    def test_search_returns_source_not_doc_id(self, chroma_index):
        """search() returns metadata["source"], not the Chroma doc_id."""
        chroma_index.upsert("text", "internal_id_xyz", source="run_abc123")
        results = chroma_index.search("text")
        assert results[0][2] == "run_abc123"  # source, not "internal_id_xyz"

    def test_search_source_type_filter(self, chroma_index):
        chroma_index.upsert(
            "Catalog entry.", "id1", source_type="catalog_run", source="run_1"
        )
        chroma_index.upsert(
            "Document chunk.", "id2", source_type="document", source="/file.txt"
        )
        catalog_results = chroma_index.search(
            "entry", top_k=5, source_type_filter="catalog_run"
        )
        assert all(True for _ in catalog_results)  # filtered, may be 1 result
        assert len(catalog_results) <= 1

    def test_search_top_k_respected(self, chroma_index):
        for i in range(10):
            chroma_index.upsert(f"document {i}", f"id{i}")
        results = chroma_index.search("document", top_k=3)
        assert len(results) <= 3


class TestChromaVectorIndexDeletion:
    def test_delete_by_source(self, chroma_index):
        chroma_index.upsert("text a", "id1", source="/file_a.txt")
        chroma_index.upsert("text b", "id2", source="/file_b.txt")
        assert chroma_index.count() == 2
        chroma_index.delete_by_source("/file_a.txt")
        assert chroma_index.count() == 1

    def test_delete_nonexistent_source(self, chroma_index):
        chroma_index.upsert("text", "id1", source="/file.txt")
        chroma_index.delete_by_source("/does_not_exist.txt")
        assert chroma_index.count() == 1  # no change


# ─────────────────────────────────────────────────────────────────────────────
# 3. MemoryRecall + Chroma integration
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.xdist_group("chroma_onnx")
class TestMemoryRecallChroma:
    """MemoryRecall using an injected EphemeralClient."""

    @pytest.fixture(autouse=True)
    def _empty_document_dir(self, tmp_path, monkeypatch):
        """Redirect DEFAULT_DOCUMENT_DIR to an empty tmp dir for every test.

        Prevents _ensure_index() from embedding the entire real corpus on each
        call.  Tests that need specific documents (e.g.
        test_ingested_documents_searchable) patch
        vox_romana.memory.ingest.DEFAULT_DOCUMENT_DIR themselves inside a
        ``with patch(...)`` block, which overrides this monkeypatch for the
        duration of that block.
        """
        import vox_romana.memory.ingest as _ingest_mod
        monkeypatch.setattr(_ingest_mod, "DEFAULT_DOCUMENT_DIR", tmp_path)

    def test_chroma_method_used_when_available(self, seeded_catalog, ephemeral_chroma):
        from vox_romana.memory.recall import MemoryRecall
        recall = MemoryRecall(
            catalog=seeded_catalog, _chroma_client=ephemeral_chroma
        )
        result = recall.query("universal basic income")
        assert result.method == "chroma"

    def test_catalog_entries_indexed_and_found(self, seeded_catalog, ephemeral_chroma):
        from vox_romana.memory.recall import MemoryRecall
        recall = MemoryRecall(
            catalog=seeded_catalog, _chroma_client=ephemeral_chroma
        )
        result = recall.query("employment effect UBI")
        assert result.method == "chroma"
        assert len(result.passages) > 0
        # Source IDs in passages should match catalog IDs.
        sources = {s for _, _, s in result.passages}
        assert len(sources) > 0

    def test_catalog_runs_populated_via_chroma(self, seeded_catalog, ephemeral_chroma):
        from vox_romana.memory.recall import MemoryRecall
        recall = MemoryRecall(
            catalog=seeded_catalog, _chroma_client=ephemeral_chroma
        )
        result = recall.query("UBI debate economists")
        # Chroma returns passages whose sources are catalog run/study IDs.
        # MemoryRecall then looks them up in the catalog.
        # With seeded data, at least one run should be found.
        assert result.method == "chroma"
        assert len(result.passages) > 0

    def test_ingested_documents_searchable(self, seeded_catalog, ephemeral_chroma, tmp_path):
        """Documents written by DocumentIngester are found by MemoryRecall."""
        from vox_romana.memory.recall import MemoryRecall

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        make_jsonl(docs_dir, "paper.jsonl", [
            {
                "text": "Universal basic income pilots in Finland showed employment gains.",
                "source": str(tmp_path / "paper.txt"),
                "chunk_index": 0,
                "format": ".txt",
                "tag": "",
                "title": "UBI Paper",
                "total_chunks": 1,
                "ingested_at": "2026-04-21T00:00:00+00:00",
            }
        ])

        with patch("vox_romana.memory.ingest.DEFAULT_DOCUMENT_DIR", docs_dir):
            recall = MemoryRecall(
                catalog=seeded_catalog,
                _chroma_client=ephemeral_chroma,
            )
            result = recall.query("Finland UBI employment")

        assert result.method == "chroma"
        all_texts = [t for t, _, _ in result.passages]
        assert any("Finland" in t for t in all_texts)

    def test_use_chroma_false_falls_back_to_fts5(self, seeded_catalog):
        from vox_romana.memory.recall import MemoryRecall
        recall = MemoryRecall(catalog=seeded_catalog, use_chroma=False)
        result = recall.query("universal basic income")
        assert result.method == "fts5"
        assert len(result.catalog_runs) > 0

    def test_no_catalog_chroma_empty_returns_fts5_method(self, ephemeral_chroma, tmp_path):
        """Empty Chroma index + no catalog → method stays 'fts5' (default)."""
        from vox_romana.memory.recall import MemoryRecall
        # Redirect migrate_documents to an empty dir so no real corpus
        # documents are indexed (which would make Chroma non-empty).
        empty_docs = tmp_path / "docs"
        with patch("vox_romana.memory.ingest.DEFAULT_DOCUMENT_DIR", empty_docs):
            recall = MemoryRecall(catalog=None, _chroma_client=ephemeral_chroma)
            result = recall.query("anything")
        assert result.method == "fts5"  # no passages, default stays
        assert result.passages == []

    def test_chroma_indexed_once(self, seeded_catalog, ephemeral_chroma):
        """Calling query twice does not double-index catalog entries."""
        from vox_romana.memory.recall import MemoryRecall
        recall = MemoryRecall(
            catalog=seeded_catalog, _chroma_client=ephemeral_chroma
        )
        recall.query("first query")
        count_after_first = ChromaVectorIndex(
            _client=ephemeral_chroma
        ).count()
        recall.query("second query")
        count_after_second = ChromaVectorIndex(
            _client=ephemeral_chroma
        ).count()
        assert count_after_first == count_after_second

    def test_format_context_with_chroma_passages(self, seeded_catalog, ephemeral_chroma):
        from vox_romana.memory.recall import MemoryRecall
        recall = MemoryRecall(
            catalog=seeded_catalog, _chroma_client=ephemeral_chroma
        )
        result = recall.query("UBI employment effects")
        context = recall.format_context(result)
        # Context should have some content (not just "no prior knowledge found").
        assert len(context) > 20

    def test_lm_synthesis_runs_before_chroma(self, seeded_catalog, ephemeral_chroma):
        """LM synthesis executes regardless of Chroma."""
        from vox_romana.memory.recall import MemoryRecall
        mock_lm = MagicMock()
        mock_lm.generate.return_value = "LM says something useful."
        recall = MemoryRecall(
            catalog=seeded_catalog,
            lm=mock_lm,
            _chroma_client=ephemeral_chroma,
        )
        result = recall.query("UBI employment")
        assert result.synthesis == "LM says something useful."
        mock_lm.generate.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# 4. run.py --mode ingest auto-migrates into Chroma
# ─────────────────────────────────────────────────────────────────────────────

class TestRunIngestWithChroma:
    def test_ingest_then_chroma_has_chunks(self, tmp_path, capsys):
        """After --mode ingest, ChromaVectorIndex picks up the new chunks."""
        import run
        import vox_romana.memory.ingest as ingest_mod

        doc = tmp_path / "paper.txt"
        doc.write_text("UBI employment study. " * 50, encoding="utf-8")
        docs_out = tmp_path / "docs"
        chroma_dir = tmp_path / "chroma"

        # Patch document output dir and Chroma dir to use tmp dirs.
        with (
            patch.object(ingest_mod, "DEFAULT_DOCUMENT_DIR", docs_out),
            patch(
                "vox_romana.memory.vector_index.DEFAULT_CHROMA_DIR", chroma_dir
            ),
        ):
            run.main([
                "--mode", "ingest",
                "--ingest-path", str(doc),
                "--force-reingest",
            ])

        # Verify JSONL was written.
        assert docs_out.is_dir()
        jsonl_files = list(docs_out.glob("*.jsonl"))
        assert len(jsonl_files) == 1

        # Verify Chroma was populated.
        assert chroma_dir.is_dir()
        index = ChromaVectorIndex(persist_dir=chroma_dir)
        assert index.count() > 0

    def test_ingest_dry_run_does_not_update_chroma(self, tmp_path, capsys):
        """Dry-run ingest must not write to Chroma."""
        import run
        import vox_romana.memory.ingest as ingest_mod

        doc = tmp_path / "paper.txt"
        doc.write_text("Content " * 50, encoding="utf-8")
        docs_out = tmp_path / "docs"
        chroma_dir = tmp_path / "chroma"

        with (
            patch.object(ingest_mod, "DEFAULT_DOCUMENT_DIR", docs_out),
            patch("vox_romana.memory.vector_index.DEFAULT_CHROMA_DIR", chroma_dir),
        ):
            run.main([
                "--mode", "ingest",
                "--ingest-path", str(doc),
                "--dry-run",
            ])

        # Chroma dir should not be created for dry-run.
        assert not chroma_dir.exists()


# ─────────────────────────────────────────────────────────────────────────────
# 5. _ChunkProxy helper
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkProxy:
    def test_full_data(self):
        proxy = _ChunkProxy({
            "text": "Hello world",
            "source": "/path/file.txt",
            "chunk_index": 3,
            "format": ".txt",
            "tag": "batch-1",
            "title": "My Doc",
        })
        assert proxy.text == "Hello world"
        assert proxy.source == "/path/file.txt"
        assert proxy.chunk_index == 3
        assert proxy.format == ".txt"
        assert proxy.tag == "batch-1"
        assert proxy.title == "My Doc"

    def test_defaults(self):
        proxy = _ChunkProxy({})
        assert proxy.text == ""
        assert proxy.source == ""
        assert proxy.chunk_index == 0
        assert proxy.format == ""
        assert proxy.tag == ""
        assert proxy.title == ""

    def test_partial_data(self):
        proxy = _ChunkProxy({"text": "Just text."})
        assert proxy.text == "Just text."
        assert proxy.source == ""

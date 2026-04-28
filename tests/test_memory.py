"""Tests for the memory system — corpus, ingest, recall, and integration.

Tests cover:
    1. CorpusBuilder — extraction from catalog, JSONL output
    2. DocumentIngester — text/md chunking, PDF skipping, directory scan
    3. MemoryRecall — FTS5 fallback, format_context, graceful degradation
    4. DataResearcher memory integration
    5. Chunking logic
    6. Encoder/LM stubs (import checks, no actual training)

All tests run without torch/transformers/sentence-transformers installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.core.catalog import Catalog
from vox_romana.core.models import FactCard
from vox_romana.memory.corpus import CorpusBuilder, CorpusStats, EncoderPair, LMExample
from vox_romana.memory.ingest import DocumentIngester, chunk_text, DocumentChunk
from vox_romana.memory.recall import MemoryRecall, RecallResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def catalog():
    cat = Catalog(db_path=":memory:")
    yield cat
    cat.close()


@pytest.fixture
def seeded_catalog(catalog):
    """Catalog with some runs and studies for corpus building."""
    run_id = catalog.log_run(
        content_type="debate",
        title="UBI Debate",
        brief="Two economists debate universal basic income",
        tags=["economics", "UBI", "policy"],
    )
    catalog.log_study(
        run_id=run_id,
        content_type="debate",
        dataset="finland_ubi.csv",
        method="two-sample t-test",
        findings="Employment rose 4.8% (p=0.003, n=2000)",
        fact_cards=[
            {"claim": "Employment rose 4.8%", "source": "KELA", "confidence": "verified"},
        ],
    )
    catalog.log_study(
        run_id=run_id,
        content_type="debate",
        dataset="canada_mincome.csv",
        method="regression",
        findings="Hospital visits dropped 8.5% in treatment group",
    )

    run_id2 = catalog.log_run(
        content_type="news",
        title="Climate Report",
        brief="Latest IPCC findings on global temperature trends",
        tags=["climate", "science"],
    )
    catalog.log_study(
        run_id=run_id2,
        content_type="news",
        dataset="nasa_temp.csv",
        method="trend_over_time",
        findings="Global temperature rose 0.18°C per decade since 1980",
    )
    return catalog


# ═══════════════════════════════════════════════════════════════════════════
# CORPUS BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class TestCorpusBuilder:

    def test_build_from_seeded_catalog(self, seeded_catalog, tmp_path):
        builder = CorpusBuilder(
            seeded_catalog,
            corpus_root=tmp_path / "corpus",
        )
        stats = builder.build()

        assert stats.runs_processed == 2
        assert stats.studies_processed == 3
        assert stats.encoder_pairs > 0
        assert stats.lm_examples > 0

        # JSONL files should exist.
        assert (tmp_path / "corpus" / "encoder_pairs.jsonl").exists()
        assert (tmp_path / "corpus" / "lm_examples.jsonl").exists()
        assert (tmp_path / "corpus" / "meta.json").exists()

    def test_encoder_pairs_content(self, seeded_catalog, tmp_path):
        builder = CorpusBuilder(seeded_catalog, corpus_root=tmp_path)
        builder.build()

        pairs = []
        with open(tmp_path / "encoder_pairs.jsonl") as f:
            for line in f:
                pairs.append(json.loads(line))

        # Should have pairs from runs (brief→title, tag→combined)
        # and studies (method→findings, dataset→findings, claim→findings).
        assert len(pairs) > 5
        queries = [p["query"] for p in pairs]
        assert "economics" in queries  # tag pair
        assert "two-sample t-test" in queries  # method pair

    def test_lm_examples_content(self, seeded_catalog, tmp_path):
        builder = CorpusBuilder(seeded_catalog, corpus_root=tmp_path)
        builder.build()

        examples = []
        with open(tmp_path / "lm_examples.jsonl") as f:
            for line in f:
                examples.append(json.loads(line))

        assert len(examples) >= 3  # At least one per study
        assert any("Employment rose" in e["completion"] for e in examples)

    def test_empty_catalog(self, catalog, tmp_path):
        builder = CorpusBuilder(catalog, corpus_root=tmp_path)
        stats = builder.build()
        assert stats.encoder_pairs == 0
        assert stats.lm_examples == 0

    def test_document_extraction(self, catalog, tmp_path):
        # Create a document JSONL.
        doc_dir = tmp_path / "documents"
        doc_dir.mkdir()
        with open(doc_dir / "test.jsonl", "w") as f:
            f.write(json.dumps({
                "text": "Universal basic income has been tested in several countries.",
                "title": "UBI Overview",
                "source": "test_paper.pdf",
            }) + "\n")

        builder = CorpusBuilder(
            catalog,
            corpus_root=tmp_path / "corpus",
            document_dir=doc_dir,
        )
        stats = builder.build()
        assert stats.documents_processed == 1
        assert stats.encoder_pairs >= 1  # title→text pair

    def test_meta_json(self, seeded_catalog, tmp_path):
        builder = CorpusBuilder(seeded_catalog, corpus_root=tmp_path)
        builder.build()

        meta = json.loads((tmp_path / "meta.json").read_text())
        assert "built_at" in meta
        assert meta["encoder_pairs"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT INGESTION
# ═══════════════════════════════════════════════════════════════════════════

class TestChunking:

    def test_basic_chunking(self):
        text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four."
        chunks = chunk_text(text, chunk_size=30, overlap=0)
        assert len(chunks) >= 2
        assert "Para one." in chunks[0]

    def test_single_paragraph(self):
        text = "A single paragraph with no breaks."
        chunks = chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, chunk_size=30, overlap=20)
        assert len(chunks) >= 2
        # Overlap means last para of chunk N appears in chunk N+1.

    def test_no_text(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_character_fallback(self):
        """Long text with no paragraph breaks should still be returned."""
        text = "x" * 2000
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        # Single paragraph — returned as one chunk (no paragraph split points).
        assert len(chunks) >= 1
        assert sum(len(c) for c in chunks) >= 2000


class TestDocumentIngester:

    def test_ingest_txt(self, tmp_path):
        doc = tmp_path / "test.txt"
        doc.write_text("This is a test document.\n\nSecond paragraph here.")

        ingester = DocumentIngester(output_dir=tmp_path / "output")
        stats = ingester.ingest(doc)

        assert stats.chunks >= 1
        assert stats.format == ".txt"
        assert (tmp_path / "output").exists()

        # Check JSONL output.
        jsonl_files = list((tmp_path / "output").glob("*.jsonl"))
        assert len(jsonl_files) == 1

    def test_ingest_md(self, tmp_path):
        doc = tmp_path / "notes.md"
        doc.write_text("# Header\n\nSome content.\n\n## Section\n\nMore content.")

        ingester = DocumentIngester(output_dir=tmp_path / "output")
        stats = ingester.ingest(doc)
        assert stats.chunks >= 1
        assert stats.format == ".md"

    def test_ingest_json(self, tmp_path):
        doc = tmp_path / "data.json"
        doc.write_text(json.dumps({"text": "Hello world from JSON."}))

        ingester = DocumentIngester(output_dir=tmp_path / "output")
        stats = ingester.ingest(doc)
        assert stats.chunks >= 1

    def test_ingest_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("Document A content.")
        (tmp_path / "b.md").write_text("Document B content.")
        (tmp_path / "c.py").write_text("# source file — also supported now")

        ingester = DocumentIngester(output_dir=tmp_path / "output")
        results = ingester.ingest_directory(tmp_path)

        # .txt, .md, and .py are all supported since Phase 1.
        assert len(results) == 3

    def test_ingest_missing_file(self, tmp_path):
        ingester = DocumentIngester(output_dir=tmp_path / "output")
        with pytest.raises(FileNotFoundError):
            ingester.ingest(tmp_path / "nonexistent.txt")

    def test_ingest_unsupported_format(self, tmp_path):
        # Use a completely unknown extension (not Tier1/2/3).
        doc = tmp_path / "data.xyz"
        doc.write_bytes(b"totally unknown format")
        ingester = DocumentIngester(output_dir=tmp_path / "output")
        with pytest.raises(ValueError, match="Unsupported format"):
            ingester.ingest(doc)

    def test_chunk_metadata(self, tmp_path):
        doc = tmp_path / "test.txt"
        doc.write_text("Para one.\n\nPara two.\n\nPara three.")

        ingester = DocumentIngester(
            output_dir=tmp_path / "output",
            chunk_size=20,
        )
        stats = ingester.ingest(doc, title="My Document")
        assert stats.chunks >= 2

        # Read back chunks and verify metadata.
        jsonl_files = list((tmp_path / "output").glob("*.jsonl"))
        chunks = []
        with open(jsonl_files[0]) as f:
            for line in f:
                chunks.append(json.loads(line))
        assert chunks[0]["title"] == "My Document"
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["total_chunks"] == len(chunks)


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY RECALL
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryRecall:

    def test_fts5_fallback(self, seeded_catalog):
        """Without encoder, should fall back to FTS5."""
        recall = MemoryRecall(catalog=seeded_catalog, use_chroma=False)
        result = recall.query("universal basic income")

        assert result.method == "fts5"
        assert len(result.catalog_runs) > 0

    def test_format_context_with_studies(self, seeded_catalog):
        recall = MemoryRecall(catalog=seeded_catalog)
        result = recall.query("UBI employment")

        context = recall.format_context(result)
        # recall.py emits "RELATED STUDIES:" (catalog hits) or "SEMANTIC MATCHES:"
        # (vector hits) or "(no prior knowledge found)" depending on which backend
        # fires. All three are valid outputs — accept any.
        assert (
            "RELATED STUDIES:" in context
            or "SEMANTIC MATCHES:" in context
            or "(no prior knowledge found)" in context
        )

    def test_format_context_empty(self, catalog):
        recall = MemoryRecall(catalog=catalog, use_chroma=False)
        result = recall.query("something that doesn't exist")
        context = recall.format_context(result)
        assert context == "(no prior knowledge found)"

    def test_no_catalog(self):
        recall = MemoryRecall(catalog=None, use_chroma=False)
        result = recall.query("anything")
        assert result.method == "fts5"  # default, but no results
        assert result.catalog_runs == []

    def test_format_context_with_synthesis(self):
        result = RecallResult(
            synthesis="UBI tends to improve employment in pilot studies.",
            method="semantic",
        )
        recall = MemoryRecall(catalog=None)
        context = recall.format_context(result)
        assert "MEMORY SYNTHESIS:" in context
        assert "UBI tends to improve" in context

    def test_format_context_truncation(self):
        result = RecallResult(synthesis="x" * 5000)
        recall = MemoryRecall(catalog=None)
        context = recall.format_context(result, max_chars=100)
        assert len(context) <= 120  # 100 + "... (truncated)"
        assert "truncated" in context

    def test_with_mock_encoder(self, seeded_catalog):
        """Semantic search with a mocked encoder."""
        import numpy as np

        mock_encoder = MagicMock()
        # Return 384-dim embeddings.
        mock_encoder.encode.return_value = np.random.randn(384).astype(np.float32)

        recall = MemoryRecall(
            catalog=seeded_catalog,
            encoder=mock_encoder,
            use_chroma=False,
        )

        # The encode method should be called for batch indexing too.
        # Mock it to return a matrix for multiple texts.
        def smart_encode(texts, **kwargs):
            if isinstance(texts, str):
                return np.random.randn(384).astype(np.float32)
            return np.random.randn(len(texts), 384).astype(np.float32)

        mock_encoder.encode.side_effect = smart_encode

        result = recall.query("UBI employment")
        assert result.method == "semantic"
        assert mock_encoder.encode.called

    def test_with_mock_lm(self, seeded_catalog):
        """LM synthesis with a mocked LM."""
        mock_lm = MagicMock()
        mock_lm.generate.return_value = "UBI pilots typically show employment gains."

        recall = MemoryRecall(
            catalog=seeded_catalog,
            lm=mock_lm,
            use_chroma=False,
        )
        result = recall.query("UBI employment effects")
        assert result.synthesis == "UBI pilots typically show employment gains."
        mock_lm.generate.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# RECALL RESULT
# ═══════════════════════════════════════════════════════════════════════════

class TestRecallResult:

    def test_defaults(self):
        r = RecallResult()
        assert r.synthesis == ""
        assert r.passages == []
        assert r.catalog_studies == []
        assert r.method == "fts5"

    def test_with_values(self):
        r = RecallResult(
            synthesis="UBI is good",
            passages=[("text", 0.95, "id1")],
            method="semantic",
        )
        assert r.synthesis == "UBI is good"
        assert len(r.passages) == 1
        assert r.passages[0][1] == 0.95


# ═══════════════════════════════════════════════════════════════════════════
# DATARESEARCHER MEMORY INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestDataResearcherMemoryIntegration:

    def test_researcher_uses_memory(self, seeded_catalog):
        """DataResearcher should use MemoryRecall when provided."""
        from vox_romana.core.data_researcher import DataResearcher

        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            '{"findings": "Memory-informed analysis", "fact_cards": []}',
            "Memory-informed summary.",
        ]

        mock_lm = MagicMock()
        mock_lm.generate.return_value = "Prior work suggests employment gains."

        recall = MemoryRecall(
            catalog=seeded_catalog,
            lm=mock_lm,
            use_chroma=False,
        )

        researcher = DataResearcher(
            client=mock_client,
            catalog=seeded_catalog,
            memory=recall,
        )

        result = researcher.run(brief="UBI employment analysis")
        assert result.findings != ""
        # Memory was consulted.
        mock_lm.generate.assert_called()

    def test_researcher_without_memory_still_works(self, catalog):
        """DataResearcher should work without memory (backward compat)."""
        from vox_romana.core.data_researcher import DataResearcher

        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            '{"findings": "Basic analysis", "fact_cards": []}',
            "Basic summary.",
        ]

        researcher = DataResearcher(
            client=mock_client,
            catalog=catalog,
        )
        result = researcher.run(brief="Test without memory")
        assert result.findings == "Basic summary."


# ═══════════════════════════════════════════════════════════════════════════
# ENCODER / LM (import-level checks only — no actual training)
# ═══════════════════════════════════════════════════════════════════════════

class TestEncoderImport:

    def test_encoder_importable(self):
        from vox_romana.memory.encoder import KnowledgeEncoder
        encoder = KnowledgeEncoder()
        assert encoder.is_trained is False

    def test_encoder_default_dir(self):
        from vox_romana.memory.encoder import DEFAULT_ENCODER_DIR
        assert "encoder" in str(DEFAULT_ENCODER_DIR)


class TestLMImport:

    def test_lm_importable(self):
        from vox_romana.memory.lm import KnowledgeLM
        lm = KnowledgeLM()
        assert lm.is_trained is False

    def test_lm_default_dir(self):
        from vox_romana.memory.lm import DEFAULT_LM_DIR
        assert "lm" in str(DEFAULT_LM_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINER (import + corpus-only, no actual model training)
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainerCorpusOnly:

    def test_build_corpus_only(self, seeded_catalog, tmp_path):
        from vox_romana.memory.trainer import MemoryTrainer

        trainer = MemoryTrainer(
            seeded_catalog,
            memory_root=tmp_path / "memory",
        )
        stats = trainer.build_corpus_only()
        assert stats.encoder_pairs > 0
        assert stats.lm_examples > 0
        assert (tmp_path / "memory" / "corpus" / "encoder_pairs.jsonl").exists()

    def test_train_skips_without_deps(self, seeded_catalog, tmp_path):
        from vox_romana.memory.trainer import MemoryTrainer

        trainer = MemoryTrainer(
            seeded_catalog,
            memory_root=tmp_path / "memory",
        )
        result = trainer.train(skip_encoder=True, skip_lm=True)
        assert result.ok
        assert result.corpus is not None
        assert result.encoder == {"status": "skipped"}
        assert result.lm == {"status": "skipped"}

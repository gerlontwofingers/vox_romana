"""Tests for vox_romana.analysis.embedder — CorpusEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox_romana.analysis.models import SpeakerCorpus


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCES = [
    "The economy requires substantial structural reform to address inequality",
    "Climate change demands immediate and comprehensive policy intervention",
    "Education funding should be increased across all public schools nationwide",
    "Healthcare access is a fundamental right for all citizens everywhere",
    "Immigration policy must balance security concerns with humanitarian values",
    "Tax reform should prioritize working families over large corporate interests",
    "Infrastructure investment creates good jobs and improves quality of life",
    "Criminal justice reform requires addressing systemic bias throughout courts",
]


def _make_corpus(name: str, n: int = 5, seed: int = 0) -> SpeakerCorpus:
    utterances = [_SENTENCES[(seed + i) % len(_SENTENCES)] for i in range(n)]
    return SpeakerCorpus(name=name, utterances=utterances)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF backend (no chromadb required)
# ─────────────────────────────────────────────────────────────────────────────

class TestTFIDF:
    @pytest.fixture
    def embedder(self, tmp_path):
        from vox_romana.analysis.embedder import CorpusEmbedder
        return CorpusEmbedder(cache_dir=tmp_path, _force_backend="tfidf")

    def test_embed_returns_correct_shape(self, embedder):
        corpus = _make_corpus("Alice", n=5)
        emb = embedder.embed(corpus)
        assert emb.ndim == 2
        assert emb.shape[0] == 5
        assert emb.shape[1] > 0

    def test_embed_returns_normalized(self, embedder):
        corpus = _make_corpus("Alice", n=5)
        emb = embedder.embed(corpus)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, np.ones(5), atol=1e-5)

    def test_embed_pair_shared_space(self, embedder):
        a = _make_corpus("Alice", n=4, seed=0)
        b = _make_corpus("Bob", n=6, seed=2)
        emb_a, emb_b = embedder.embed_pair(a, b)
        assert emb_a.shape[1] == emb_b.shape[1]
        assert emb_a.shape[0] == 4
        assert emb_b.shape[0] == 6

    def test_embed_pair_normalized(self, embedder):
        a = _make_corpus("Alice", n=3, seed=0)
        b = _make_corpus("Bob", n=3, seed=1)
        emb_a, emb_b = embedder.embed_pair(a, b)
        np.testing.assert_allclose(np.linalg.norm(emb_a, axis=1), np.ones(3), atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(emb_b, axis=1), np.ones(3), atol=1e-5)

    def test_embed_pair_identical_corpora_high_similarity(self, embedder):
        utterances = ["The policy requires comprehensive reform and long-term investment"] * 5
        a = SpeakerCorpus(name="A", utterances=utterances)
        b = SpeakerCorpus(name="B", utterances=utterances)
        emb_a, emb_b = embedder.embed_pair(a, b)
        centroid_a = emb_a.mean(axis=0)
        centroid_b = emb_b.mean(axis=0)
        norm_a = centroid_a / (np.linalg.norm(centroid_a) or 1.0)
        norm_b = centroid_b / (np.linalg.norm(centroid_b) or 1.0)
        sim = float(norm_a @ norm_b)
        assert sim > 0.99

    def test_embed_texts_returns_correct_shape(self, embedder):
        texts = ["First text here today", "Second text here today", "Third text here today"]
        emb = embedder.embed_texts(texts)
        assert emb.shape[0] == 3
        assert emb.ndim == 2

    def test_embed_texts_normalized(self, embedder):
        texts = ["Alpha beta gamma delta epsilon", "Zeta eta theta iota kappa lambda"]
        emb = embedder.embed_texts(texts)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)

    def test_cache_hit_skips_recompute(self, embedder):
        corpus = _make_corpus("Alice", n=3)
        emb1 = embedder.embed(corpus)
        # Second call should load from disk — np.save must not be called again
        with patch("numpy.save") as mock_save:
            emb2 = embedder.embed(corpus)
            mock_save.assert_not_called()
        np.testing.assert_array_equal(emb1, emb2)

    def test_backend_property(self, embedder):
        assert embedder.backend == "tfidf"

    def test_single_utterance_corpus(self, embedder):
        corpus = SpeakerCorpus(name="Solo", utterances=["Just one utterance here today"])
        emb = embedder.embed(corpus)
        assert emb.shape[0] == 1
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), np.ones(1), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Chroma backend (mocked — no chromadb install needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestChromaMocked:
    @pytest.fixture
    def chroma_embedder(self, tmp_path):
        from vox_romana.analysis.embedder import CorpusEmbedder
        embedder = CorpusEmbedder(cache_dir=tmp_path, _force_backend="chroma")
        mock_fn = MagicMock()
        mock_fn.side_effect = lambda texts: [
            [float(i % 384) / 384.0 for i in range(384)] for _ in texts
        ]
        embedder._chroma_fn = mock_fn
        return embedder

    def test_backend_is_chroma(self, chroma_embedder):
        assert chroma_embedder.backend == "chroma"

    def test_embedding_dim_is_384(self, chroma_embedder):
        assert chroma_embedder.embedding_dim == 384

    def test_embed_shape_is_correct(self, chroma_embedder):
        corpus = _make_corpus("Alice", n=3)
        emb = chroma_embedder.embed(corpus)
        assert emb.shape == (3, 384)

    def test_embed_calls_chroma_fn(self, chroma_embedder):
        corpus = _make_corpus("Alice", n=3)
        chroma_embedder.embed(corpus)
        chroma_embedder._chroma_fn.assert_called()

    def test_embed_pair_uses_embed_twice(self, chroma_embedder):
        a = _make_corpus("Alice", n=2)
        b = _make_corpus("Bob", n=3)
        emb_a, emb_b = chroma_embedder.embed_pair(a, b)
        assert emb_a.shape[0] == 2
        assert emb_b.shape[0] == 3
        assert emb_a.shape[1] == emb_b.shape[1] == 384

    def test_embed_result_is_normalized(self, chroma_embedder):
        corpus = _make_corpus("Alice", n=4)
        emb = chroma_embedder.embed(corpus)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases / general
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneral:
    def test_empty_texts_raises(self, tmp_path):
        from vox_romana.analysis.embedder import CorpusEmbedder
        embedder = CorpusEmbedder(cache_dir=tmp_path, _force_backend="tfidf")
        with pytest.raises(ValueError, match="empty text list"):
            embedder.embed_texts([])

    def test_cache_dir_created_on_demand(self, tmp_path):
        from vox_romana.analysis.embedder import CorpusEmbedder
        cache_dir = tmp_path / "deep" / "nested" / "cache"
        embedder = CorpusEmbedder(cache_dir=cache_dir, _force_backend="tfidf")
        corpus = _make_corpus("Alice", n=2)
        embedder.embed(corpus)
        assert cache_dir.exists()

    def test_float32_dtype(self, tmp_path):
        from vox_romana.analysis.embedder import CorpusEmbedder
        embedder = CorpusEmbedder(cache_dir=tmp_path, _force_backend="tfidf")
        corpus = _make_corpus("Alice", n=3)
        emb = embedder.embed(corpus)
        assert emb.dtype == np.float32

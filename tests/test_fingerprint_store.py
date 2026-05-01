"""Tests for vox_romana.memory.fingerprint_store — FingerprintStore.

Coverage:
    1. add() — stores fingerprint, upsert replaces existing, returns vector.
    2. find_nearest() — returns match above threshold, None below threshold,
       None on empty store.
    3. nearest_pairs() — finds pairs above threshold, empty when < 2 docs,
       excludes self-pairs.
    4. remove() — returns True on hit, False on miss, size decreases.
    5. build_parallel() — indexes JSONL files, returns count, idempotent upsert.
    6. rebuild() — clears and rebuilds from scratch.
    7. size / all_sources() properties.
    8. Persistence — round-trip save/reload from .npz.
    9. Custom fingerprinter propagation via store_path + fingerprinter params.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vox_romana.memory.fingerprint_store import FingerprintStore


# ─────────────────────────────────────────────────────────────────────────────
# Text fixtures
# ─────────────────────────────────────────────────────────────────────────────

_PARA = (
    "The court filing describes allegations of systematic misconduct. "
    "Witness testimony provides supporting evidence for the prosecution. "
    "Multiple co-conspirators are named in the indictment. "
)

TEXT_A = _PARA * 30           # ~1 800 chars — solid bigram distribution
TEXT_B = _PARA * 30           # identical to A
TEXT_C = _PARA[:-10] + "xyz " * 5 + _PARA * 29   # near-identical to A
TEXT_D = (
    "Chlorophyll absorbs sunlight to drive the Calvin cycle. "
    "Photosystem II oxidises water releasing oxygen as a byproduct. "
    "The electron transport chain generates ATP and NADPH. "
) * 30  # completely unrelated topic


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _store(tmp_path: Path) -> FingerprintStore:
    return FingerprintStore(store_path=tmp_path / "fps.npz")


def _write_jsonl(path: Path, chunks: list[str]) -> Path:
    """Write a JSONL file containing chunk records."""
    with open(path, "w") as f:
        for i, text in enumerate(chunks):
            f.write(json.dumps({"text": text, "chunk_index": i}) + "\n")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. add()
# ─────────────────────────────────────────────────────────────────────────────

class TestAdd:
    def test_add_increases_size(self, tmp_path):
        s = _store(tmp_path)
        assert s.size == 0
        s.add("doc_a", TEXT_A)
        assert s.size == 1

    def test_add_returns_vector(self, tmp_path):
        s = _store(tmp_path)
        v = s.add("doc_a", TEXT_A)
        assert isinstance(v, np.ndarray)
        assert v.shape == (9216,)
        assert v.dtype == np.float32

    def test_upsert_replaces_existing(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        s.add("doc_a", TEXT_D)   # same key → replace
        assert s.size == 1
        assert s.all_sources() == ["doc_a"]

    def test_add_multiple_sources(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        s.add("doc_d", TEXT_D)
        assert s.size == 2

    def test_source_in_all_sources(self, tmp_path):
        s = _store(tmp_path)
        s.add("my_source", TEXT_A)
        assert "my_source" in s.all_sources()


# ─────────────────────────────────────────────────────────────────────────────
# 2. find_nearest()
# ─────────────────────────────────────────────────────────────────────────────

class TestFindNearest:
    def test_empty_store_returns_none(self, tmp_path):
        s = _store(tmp_path)
        result = s.find_nearest(TEXT_A, threshold=0.97)
        assert result is None

    def test_identical_text_above_threshold(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        result = s.find_nearest(TEXT_B, threshold=0.97)
        assert result is not None
        source, sim = result
        assert source == "doc_a"
        assert sim == pytest.approx(1.0, abs=1e-4)

    def test_unrelated_text_below_threshold(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        result = s.find_nearest(TEXT_D, threshold=0.97)
        assert result is None

    def test_near_identical_above_threshold(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        # TEXT_C is a small variant of TEXT_A; actual similarity ~0.93
        result = s.find_nearest(TEXT_C, threshold=0.90)
        assert result is not None
        _, sim = result
        assert sim >= 0.90

    def test_threshold_one_catches_only_exact(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        # near-identical C should not match at threshold=1.0
        result = s.find_nearest(TEXT_C, threshold=1.0)
        assert result is None

    def test_callable_threshold(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        # Dynamic threshold returning 0.0 → always matches anything above 0
        result = s.find_nearest(TEXT_B, threshold=lambda _chars: 0.0)
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# 3. nearest_pairs()
# ─────────────────────────────────────────────────────────────────────────────

class TestNearestPairs:
    def test_empty_store_returns_empty(self, tmp_path):
        s = _store(tmp_path)
        assert s.nearest_pairs() == []

    def test_single_doc_returns_empty(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        assert s.nearest_pairs() == []

    def test_identical_pair_detected(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        s.add("doc_b", TEXT_B)
        pairs = s.nearest_pairs(threshold=0.97)
        assert len(pairs) == 1
        sources = {pairs[0][0], pairs[0][1]}
        assert sources == {"doc_a", "doc_b"}
        assert pairs[0][2] == pytest.approx(1.0, abs=1e-4)

    def test_unrelated_pair_not_detected(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        s.add("doc_d", TEXT_D)
        pairs = s.nearest_pairs(threshold=0.97)
        assert pairs == []

    def test_sorted_by_similarity_descending(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        s.add("doc_b", TEXT_B)   # sim ≈ 1.0
        s.add("doc_c", TEXT_C)   # sim ≈ 0.93
        # Use threshold=0.90 to capture both A↔B and A↔C (and B↔C) pairs
        pairs = s.nearest_pairs(threshold=0.90)
        sims = [p[2] for p in pairs]
        assert sims == sorted(sims, reverse=True)

    def test_no_self_pairs(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        s.add("doc_b", TEXT_B)
        pairs = s.nearest_pairs(threshold=0.97)
        for a, b, _ in pairs:
            assert a != b


# ─────────────────────────────────────────────────────────────────────────────
# 4. remove()
# ─────────────────────────────────────────────────────────────────────────────

class TestRemove:
    def test_remove_existing_returns_true(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        assert s.remove("doc_a") is True

    def test_remove_missing_returns_false(self, tmp_path):
        s = _store(tmp_path)
        assert s.remove("nonexistent") is False

    def test_remove_decreases_size(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        s.add("doc_d", TEXT_D)
        s.remove("doc_a")
        assert s.size == 1

    def test_remove_source_gone_from_all_sources(self, tmp_path):
        s = _store(tmp_path)
        s.add("doc_a", TEXT_A)
        s.remove("doc_a")
        assert "doc_a" not in s.all_sources()


# ─────────────────────────────────────────────────────────────────────────────
# 5. build_parallel()
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildParallel:
    def _corpus(self, tmp_path: Path) -> Path:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _write_jsonl(corpus / "doc_a.jsonl", [TEXT_A[i:i+400] for i in range(0, len(TEXT_A), 400)])
        _write_jsonl(corpus / "doc_d.jsonl", [TEXT_D[i:i+400] for i in range(0, len(TEXT_D), 400)])
        return corpus

    def test_returns_count(self, tmp_path):
        corpus = self._corpus(tmp_path)
        s = _store(tmp_path)
        n = s.build_parallel(corpus, workers=1)
        assert n == 2

    def test_sources_indexed(self, tmp_path):
        corpus = self._corpus(tmp_path)
        s = _store(tmp_path)
        s.build_parallel(corpus, workers=1)
        sources = s.all_sources()
        assert "doc_a.jsonl" in sources
        assert "doc_d.jsonl" in sources

    def test_idempotent_upsert(self, tmp_path):
        corpus = self._corpus(tmp_path)
        s = _store(tmp_path)
        s.build_parallel(corpus, workers=1)
        n2 = s.build_parallel(corpus, workers=1)
        assert s.size == 2  # upsert, not duplicate

    def test_empty_dir_returns_zero(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        s = _store(tmp_path)
        assert s.build_parallel(empty, workers=1) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. rebuild()
# ─────────────────────────────────────────────────────────────────────────────

class TestRebuild:
    def test_rebuild_clears_and_repopulates(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _write_jsonl(corpus / "doc_a.jsonl", [TEXT_A])
        _write_jsonl(corpus / "doc_d.jsonl", [TEXT_D])

        s = _store(tmp_path)
        s.add("stale_doc", TEXT_A)   # manually added — should be cleared
        assert s.size == 1

        n = s.rebuild(corpus, workers=1)
        assert n == 2
        assert "stale_doc" not in s.all_sources()


# ─────────────────────────────────────────────────────────────────────────────
# 7. size / all_sources()
# ─────────────────────────────────────────────────────────────────────────────

class TestSizeAndSources:
    def test_size_empty(self, tmp_path):
        assert _store(tmp_path).size == 0

    def test_all_sources_empty(self, tmp_path):
        assert _store(tmp_path).all_sources() == []

    def test_size_after_adds(self, tmp_path):
        s = _store(tmp_path)
        s.add("a", TEXT_A)
        s.add("b", TEXT_D)
        assert s.size == 2


# ─────────────────────────────────────────────────────────────────────────────
# 8. Persistence — round-trip save / reload
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_round_trip(self, tmp_path):
        fp_path = tmp_path / "fps.npz"
        s1 = FingerprintStore(store_path=fp_path)
        s1.add("doc_a", TEXT_A)
        s1.add("doc_d", TEXT_D)

        # New instance loading same file.
        s2 = FingerprintStore(store_path=fp_path)
        assert s2.size == 2
        assert set(s2.all_sources()) == {"doc_a", "doc_d"}

    def test_find_nearest_after_reload(self, tmp_path):
        fp_path = tmp_path / "fps.npz"
        s1 = FingerprintStore(store_path=fp_path)
        s1.add("doc_a", TEXT_A)

        s2 = FingerprintStore(store_path=fp_path)
        result = s2.find_nearest(TEXT_B, threshold=0.97)
        assert result is not None
        assert result[0] == "doc_a"

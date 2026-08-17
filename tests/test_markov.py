"""Tests for vox_romana.core.markov — CharBigramFingerprinter + module seam.

Coverage:
    1. CharBigramFingerprinter.build() — shape, dtype, identical=1.0, empty=zeros,
       single-char=zeros, near-identical high similarity, unrelated low similarity.
    2. CharBigramFingerprinter.pairwise_cosine() — shape, diagonal=1.0,
       symmetry, empty input, 1-D input raises.
    3. Module seam — get_fingerprinter returns default, set_fingerprinter propagates
       to get_fingerprinter(), reset_fingerprinter restores default,
       set_fingerprinter rejects non-Protocol objects.
    4. Protocol conformance — CharBigramFingerprinter satisfies MarkovFingerprinter.
"""

from __future__ import annotations

import numpy as np
import pytest

from vox_romana.core.markov import (
    CharBigramFingerprinter,
    MarkovFingerprinter,
    get_fingerprinter,
    reset_fingerprinter,
    set_fingerprinter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

LONG_TEXT_A = (
    "The court documents describe a network of financiers and politicians. "
    "Witness testimony details systematic abuse over many years at multiple "
    "locations. The indictment lists numerous co-conspirators. "
) * 40  # ~2 800 chars

LONG_TEXT_B = LONG_TEXT_A  # identical

LONG_TEXT_C = LONG_TEXT_A[:-5] + "abcde"  # one-word tweak — should still be very close

LONG_TEXT_D = (
    "Quantum chromodynamics describes the strong interaction between quarks "
    "mediated by gluons. Asymptotic freedom means coupling weakens at short "
    "distances. The running coupling constant changes with energy scale. "
) * 40  # completely unrelated topic


@pytest.fixture(autouse=True)
def restore_fingerprinter():
    """Always restore the default fingerprinter after each test."""
    yield
    reset_fingerprinter()


# ─────────────────────────────────────────────────────────────────────────────
# 1. CharBigramFingerprinter.build()
# ─────────────────────────────────────────────────────────────────────────────


class TestCharBigramFingerprinterBuild:
    fp = CharBigramFingerprinter()

    def test_output_shape(self):
        v = self.fp.build(LONG_TEXT_A)
        assert v.shape == (9216,)

    def test_output_dtype(self):
        v = self.fp.build(LONG_TEXT_A)
        assert v.dtype == np.float32

    def test_empty_string_returns_zeros(self):
        v = self.fp.build("")
        assert np.all(v == 0)
        assert v.shape == (9216,)

    def test_single_char_returns_zeros(self):
        # Only one ASCII char — no bigrams possible.
        v = self.fp.build("x")
        assert np.all(v == 0)

    def test_identical_texts_cosine_one(self):
        va = self.fp.build(LONG_TEXT_A)
        vb = self.fp.build(LONG_TEXT_B)
        sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_near_identical_high_similarity(self):
        va = self.fp.build(LONG_TEXT_A)
        vc = self.fp.build(LONG_TEXT_C)
        sim = float(np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-9))
        assert sim >= 0.97, f"Expected >= 0.97, got {sim:.4f}"

    def test_unrelated_text_low_similarity(self):
        va = self.fp.build(LONG_TEXT_A)
        vd = self.fp.build(LONG_TEXT_D)
        sim = float(np.dot(va, vd) / (np.linalg.norm(va) * np.linalg.norm(vd) + 1e-9))
        assert sim < 0.97, f"Expected < 0.97, got {sim:.4f}"

    def test_non_ascii_input_handled(self):
        # Non-ASCII bytes should be ignored without raising.
        text = "Caf\u00e9 and na\u00efve r\u00e9sum\u00e9s"
        v = self.fp.build(text)
        assert v.shape == (9216,)
        assert v.dtype == np.float32

    def test_numeric_input_cast_to_str(self):
        v = self.fp.build(12345)
        assert v.shape == (9216,)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CharBigramFingerprinter.pairwise_cosine()
# ─────────────────────────────────────────────────────────────────────────────


class TestCharBigramFingerprinterPairwiseCosine:
    fp = CharBigramFingerprinter()

    def _stack(self, *texts):
        vecs = [self.fp.build(t) for t in texts]
        return np.stack(vecs)

    def test_output_shape(self):
        fps = self._stack(LONG_TEXT_A, LONG_TEXT_D)
        sim = self.fp.pairwise_cosine(fps)
        assert sim.shape == (2, 2)

    def test_diagonal_is_one(self):
        fps = self._stack(LONG_TEXT_A, LONG_TEXT_D)
        sim = self.fp.pairwise_cosine(fps)
        assert sim[0, 0] == pytest.approx(1.0, abs=1e-4)
        assert sim[1, 1] == pytest.approx(1.0, abs=1e-4)

    def test_symmetry(self):
        fps = self._stack(LONG_TEXT_A, LONG_TEXT_D)
        sim = self.fp.pairwise_cosine(fps)
        assert sim[0, 1] == pytest.approx(sim[1, 0], abs=1e-5)

    def test_empty_input_returns_empty(self):
        fps = np.zeros((0, 9216), dtype=np.float32)
        sim = self.fp.pairwise_cosine(fps)
        assert sim.shape == (0, 0)

    def test_single_row(self):
        fps = self._stack(LONG_TEXT_A)
        sim = self.fp.pairwise_cosine(fps)
        assert sim.shape == (1, 1)
        assert sim[0, 0] == pytest.approx(1.0, abs=1e-4)

    def test_1d_input_raises(self):
        fps = self.fp.build(LONG_TEXT_A)  # 1-D
        with pytest.raises(ValueError, match="2-D"):
            self.fp.pairwise_cosine(fps)

    def test_identical_pair_off_diagonal_one(self):
        fps = self._stack(LONG_TEXT_A, LONG_TEXT_B)
        sim = self.fp.pairwise_cosine(fps)
        assert sim[0, 1] == pytest.approx(1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Module seam
# ─────────────────────────────────────────────────────────────────────────────


class TestModuleSeam:
    def test_get_fingerprinter_returns_default(self):
        fp = get_fingerprinter()
        assert isinstance(fp, CharBigramFingerprinter)

    def test_set_fingerprinter_propagates(self):
        class _Stub:
            vector_dim = 4

            def build(self, sequence):
                return np.zeros(4, dtype=np.float32)

            def pairwise_cosine(self, fps):
                return np.eye(fps.shape[0], dtype=np.float32)

        stub = _Stub()
        set_fingerprinter(stub)
        assert get_fingerprinter() is stub

    def test_reset_fingerprinter_restores_default(self):
        class _Stub:
            vector_dim = 4

            def build(self, sequence):
                return np.zeros(4, dtype=np.float32)

            def pairwise_cosine(self, fps):
                return np.eye(fps.shape[0], dtype=np.float32)

        set_fingerprinter(_Stub())
        reset_fingerprinter()
        fp = get_fingerprinter()
        assert isinstance(fp, CharBigramFingerprinter)

    def test_set_fingerprinter_rejects_invalid(self):
        with pytest.raises(TypeError, match="MarkovFingerprinter Protocol"):
            set_fingerprinter("not a fingerprinter")  # type: ignore[arg-type]

    def test_set_fingerprinter_rejects_missing_method(self):
        class _Incomplete:
            vector_dim = 4

            def build(self, sequence):
                return np.zeros(4, dtype=np.float32)
            # missing pairwise_cosine

        with pytest.raises(TypeError, match="MarkovFingerprinter Protocol"):
            set_fingerprinter(_Incomplete())  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Protocol conformance
# ─────────────────────────────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_char_bigram_satisfies_protocol(self):
        fp = CharBigramFingerprinter()
        assert isinstance(fp, MarkovFingerprinter)

    def test_vector_dim_attribute(self):
        fp = CharBigramFingerprinter()
        assert fp.vector_dim == 9216

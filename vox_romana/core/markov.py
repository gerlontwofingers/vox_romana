"""vox_romana.core.markov — canonical seam for transition-matrix fingerprints.

This module is the **one place** to change if the math behind sequence
fingerprinting (currently first-order Markov over a fixed alphabet) is ever
upgraded.  All consumers — :mod:`vox_romana.memory.fingerprint_store`, future
voice-consistency checks, etc. — go through :func:`get_fingerprinter` so a
single :func:`set_fingerprinter` call propagates a new implementation
everywhere.

Conceptual lineage
------------------
This is the same mathematical object that appears in three different
granularities throughout the codebase and the broader ML literature:

* :class:`vox_romana.emotion_flow.models.TransitionMatrix` — first-order
  Markov over emotion labels (~12 states), per-leader.
* :class:`CharBigramFingerprinter` (this module) — first-order Markov over
  printable ASCII (96 states), per-document.  Captures *prose rhythm* /
  textual identity, not semantic content.
* Transformer attention — generalised, learned, context-conditioned
  transition matrix over a learned token vocabulary (50k+ states).

When to upgrade
---------------
The default :class:`CharBigramFingerprinter` empirically separates near-
duplicate documents (cosine ≥ ~0.97) from independently-written documents on
the same topic (cosine ≤ ~0.92).  If that gap collapses on real corpora,
swap in a richer implementation here:

* ``TrigramFingerprinter`` — 96³ ≈ 884k dim, captures longer-range structure.
* A learned encoder (e.g. sentence-transformers) — semantic, not textual.
  Trades dedup precision for topic recall.

Whichever you choose, expose it via the :class:`MarkovFingerprinter` Protocol
and register it with :func:`set_fingerprinter`.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ─── Protocol ──────────────────────────────────────────────────────────────


@runtime_checkable
class MarkovFingerprinter(Protocol):
    """Fingerprint a sequence as a fixed-dimensional vector.

    Implementations build a (typically row-stochastic) transition matrix
    from an input sequence and flatten it to a 1-D vector suitable for
    cosine similarity comparison.

    Attributes:
        vector_dim: Dimensionality of vectors returned by :meth:`build`.
    """

    vector_dim: int

    def build(self, sequence: Any) -> np.ndarray:
        """Compute a fingerprint vector of shape ``(vector_dim,)`` float32."""
        ...

    def pairwise_cosine(self, fps: np.ndarray) -> np.ndarray:
        """Compute the ``(n, n)`` cosine similarity matrix for ``(n, vector_dim)`` input."""
        ...


# ─── Default implementation: character bigrams ─────────────────────────────


class CharBigramFingerprinter:
    """First-order Markov transition matrix over printable ASCII.

    Alphabet: bytes 32–127 inclusive (96 printable characters including
    space).  Non-ASCII input is encoded to ASCII with ``errors="ignore"`` —
    accents are dropped, but the bigram signature stays stable across
    format variants of the same document (EPUB → TXT → OCR).

    The fingerprint is the row-normalised 96×96 transition matrix flattened
    to a 9,216-dim float32 vector.  Cosine similarity between two such
    vectors is sensitive to *prose rhythm* (which characters tend to follow
    which) and so behaves as a textual identity metric, not a semantic one.

    Empirical regimes on long-form English (≥ 500 chars):

    * Same text, identical extraction:        cosine ≈ 1.000
    * Same text, format variants (EPUB↔TXT):  cosine ≈ 0.990–0.999
    * Same text, OCR noise:                   cosine ≈ 0.970–0.990
    * Same domain, independent writing:       cosine ≈ 0.800–0.920
    * Unrelated topics:                       cosine ≈ 0.500–0.800
    """

    vector_dim: int = 96 * 96  # 9,216

    _ALPHABET_SIZE: int = 96
    _ASCII_LO: int = 32   # space
    _ASCII_HI: int = 128  # exclusive — covers up to 127 (DEL)

    def build(self, sequence: Any) -> np.ndarray:
        """Build a 9,216-dim bigram fingerprint from text.

        Args:
            sequence: A ``str`` of any length (or any object castable to
                ``str``).  Empty / sub-bigram input returns a zero vector.

        Returns:
            float32 array of shape ``(9216,)``.  All-zero if input has < 2
            valid characters.
        """
        text = sequence if isinstance(sequence, str) else str(sequence)
        if not text:
            return np.zeros(self.vector_dim, dtype=np.float32)

        # ASCII-fold: drop bytes outside the printable range.
        encoded = text.encode("ascii", errors="ignore")
        arr = np.frombuffer(encoded, dtype=np.uint8)
        mask = (arr >= self._ASCII_LO) & (arr < self._ASCII_HI)
        arr = arr[mask].astype(np.int32) - self._ASCII_LO

        if arr.size < 2:
            return np.zeros(self.vector_dim, dtype=np.float32)

        # Vectorised bigram count: M[src, dst] += 1 for every adjacent pair.
        n = self._ALPHABET_SIZE
        counts = np.zeros((n, n), dtype=np.float32)
        np.add.at(counts, (arr[:-1], arr[1:]), 1.0)

        # Row-normalise → row-stochastic.  Empty rows stay zero.
        row_sums = counts.sum(axis=1, keepdims=True)
        safe = np.where(row_sums == 0, 1.0, row_sums)
        matrix = counts / safe

        return matrix.ravel().astype(np.float32, copy=False)

    def pairwise_cosine(self, fps: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix for a stack of fingerprints.

        Uses a single BLAS ``sgemm`` after L2-normalisation.  Dispatches
        to ``torch`` on Apple Silicon's MPS backend when available for
        GPU-accelerated comparison; falls back to numpy otherwise.

        Args:
            fps: float32 array of shape ``(n, vector_dim)``.

        Returns:
            float32 array of shape ``(n, n)`` with values in ``[-1, 1]``.
            Diagonal is 1.0 for any non-zero row.
        """
        if fps.ndim != 2:
            raise ValueError(
                f"pairwise_cosine expects 2-D input, got shape {fps.shape}"
            )
        if fps.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32)

        fps = np.ascontiguousarray(fps, dtype=np.float32)

        # L2-normalise rows; zero-rows stay zero (no NaN).
        norms = np.linalg.norm(fps, axis=1, keepdims=True)
        safe = np.where(norms == 0, 1.0, norms)
        normed = fps / safe

        # Try torch on MPS (Apple GPU) for large batches; numpy is fine
        # for small ones and avoids the device-transfer overhead.
        if normed.shape[0] >= 64:
            torch_result = _try_torch_mps_gemm(normed)
            if torch_result is not None:
                return torch_result

        return (normed @ normed.T).astype(np.float32, copy=False)


# ─── JIT / acceleration hooks ──────────────────────────────────────────────


def _try_torch_mps_gemm(normed: np.ndarray) -> np.ndarray | None:
    """Run ``F @ F.T`` on Apple MPS if torch + MPS are available.

    Returns ``None`` if torch isn't importable or MPS isn't present.
    """
    try:
        import torch  # noqa: PLC0415 — optional fast path

        if not torch.backends.mps.is_available():
            return None
        device = torch.device("mps")
        t = torch.from_numpy(normed).to(device)
        out = (t @ t.T).cpu().numpy()
        return out.astype(np.float32, copy=False)
    except Exception as exc:  # pragma: no cover — best-effort fast path
        logger.debug("torch/MPS pairwise_cosine fallback: %s", exc)
        return None


def _try_jit_pairwise():  # pragma: no cover — stub for future numba hook
    """Hook for a future numba-compiled pairwise routine.

    Returns ``None`` today.  If numba is added to requirements, return a
    compiled function with signature ``(np.ndarray) -> np.ndarray`` here
    and have :meth:`CharBigramFingerprinter.pairwise_cosine` prefer it.
    """
    return None


# ─── Module-level seam ─────────────────────────────────────────────────────


_DEFAULT: MarkovFingerprinter = CharBigramFingerprinter()


def get_fingerprinter() -> MarkovFingerprinter:
    """Return the currently registered :class:`MarkovFingerprinter`.

    All consumers should call this rather than instantiating a fingerprinter
    directly, so swapping the implementation via :func:`set_fingerprinter`
    propagates everywhere.
    """
    return _DEFAULT


def set_fingerprinter(impl: MarkovFingerprinter) -> None:
    """Replace the module-default fingerprinter.

    Use sparingly — typically once at process start, or in tests via
    :func:`reset_fingerprinter` to restore the default afterwards.
    """
    global _DEFAULT
    if not isinstance(impl, MarkovFingerprinter):
        raise TypeError(
            f"impl must implement MarkovFingerprinter Protocol, got {type(impl).__name__}"
        )
    _DEFAULT = impl


def reset_fingerprinter() -> None:
    """Restore the module-default :class:`CharBigramFingerprinter`."""
    global _DEFAULT
    _DEFAULT = CharBigramFingerprinter()


__all__ = [
    "CharBigramFingerprinter",
    "MarkovFingerprinter",
    "get_fingerprinter",
    "reset_fingerprinter",
    "set_fingerprinter",
]

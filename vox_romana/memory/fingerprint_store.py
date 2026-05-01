"""vox_romana.memory.fingerprint_store — persistent document fingerprint store.

Stores one bigram-transition-matrix fingerprint per ingested document in a
single flat ``.npz`` file for fast pairwise comparison.  All math is
delegated to :func:`vox_romana.core.markov.get_fingerprinter` — swapping
the fingerprint algorithm via :func:`~vox_romana.core.markov.set_fingerprinter`
automatically affects all :meth:`FingerprintStore` operations.

Layout on disk::

    {MEMORY_ROOT}/fingerprints.npz
        fps      — float32 (n_docs, vector_dim)
        sources  — object  (n_docs,)   str paths / IDs parallel to fps rows

Concurrency note
----------------
The store is **not** thread-safe for concurrent writes.  Reads are safe
(the file is only opened, never mutated, during :meth:`find_nearest` and
:meth:`all_sources`).  For parallel builds use :meth:`build_parallel` which
collects results in-process before doing a single final write.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

from vox_romana.core.markov import MarkovFingerprinter, get_fingerprinter
from vox_romana.core.paths import FINGERPRINT_STORE_PATH

logger = logging.getLogger(__name__)

_DEFAULT_STORE_PATH = FINGERPRINT_STORE_PATH


# ─── Parallel worker (module-level so it's picklable) ──────────────────────


def _fingerprint_jsonl_slice(
    jsonl_paths: list[str],
) -> dict[str, np.ndarray]:
    """Worker: fingerprint all documents in a list of JSONL files.

    Reads each JSONL, concatenates all chunk texts per file, builds one
    fingerprint per file.  Returns ``{jsonl_filename: fingerprint_vector}``.

    This function is module-level (not a method) so ``ProcessPoolExecutor``
    can pickle it on all platforms.
    """
    import json as _json

    from vox_romana.core.markov import get_fingerprinter as _get_fp

    fp = _get_fp()
    results: dict[str, np.ndarray] = {}
    for jpath in jsonl_paths:
        p = Path(jpath)
        texts: list[str] = []
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                        text = data.get("text", "")
                        if text:
                            texts.append(text)
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("fingerprint worker: skipping %s: %s", p, exc)
            continue
        if texts:
            combined = " ".join(texts)
            results[p.name] = fp.build(combined)
    return results


# ─── FingerprintStore ──────────────────────────────────────────────────────


class FingerprintStore:
    """Persistent store of document fingerprints backed by a single ``.npz``.

    Args:
        store_path:    Path to the ``.npz`` file.  Created on first write.
        fingerprinter: :class:`~vox_romana.core.markov.MarkovFingerprinter`
                       to use.  Defaults to the module-level default returned
                       by :func:`~vox_romana.core.markov.get_fingerprinter`.
    """

    def __init__(
        self,
        store_path: Path | str | None = None,
        fingerprinter: MarkovFingerprinter | None = None,
    ) -> None:
        self._path = Path(store_path) if store_path else _DEFAULT_STORE_PATH
        self._fp: MarkovFingerprinter = fingerprinter or get_fingerprinter()

        # Lazily loaded on first access.
        self._fps: np.ndarray | None = None       # (n, vector_dim) float32
        self._sources: np.ndarray | None = None   # (n,) object (str)

    # ── Internal load / save ───────────────────────────────────────────────

    def _load(self) -> None:
        """Load fingerprints from disk into memory (idempotent)."""
        if self._fps is not None:
            return
        if self._path.exists():
            data = np.load(self._path, allow_pickle=True)
            self._fps = data["fps"].astype(np.float32)
            self._sources = data["sources"].astype(object)
        else:
            self._fps = np.zeros((0, self._fp.vector_dim), dtype=np.float32)
            self._sources = np.empty(0, dtype=object)

    def _save(self) -> None:
        """Persist current in-memory state to the ``.npz`` file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._path, fps=self._fps, sources=self._sources)

    def _invalidate(self) -> None:
        """Force reload on next access (used after external modifications)."""
        self._fps = None
        self._sources = None

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of fingerprints currently in the store."""
        self._load()
        return int(self._fps.shape[0])  # type: ignore[union-attr]

    def all_sources(self) -> list[str]:
        """Return all source identifiers in the store."""
        self._load()
        return list(self._sources)  # type: ignore[arg-type]

    def add(self, source: str, text: str) -> np.ndarray:
        """Fingerprint ``text`` and append it to the store.

        If ``source`` is already present the existing entry is **replaced**
        (upsert semantics) to keep the store consistent after re-ingest.

        Args:
            source: Unique identifier — typically the ingest file path or
                    a content-hash-derived ID.
            text:   Full document text to fingerprint.

        Returns:
            The computed fingerprint vector of shape ``(vector_dim,)``.
        """
        self._load()
        fp_vec = self._fp.build(text)

        assert self._fps is not None
        assert self._sources is not None

        # Upsert: remove existing entry for this source if present.
        existing = np.where(self._sources == source)[0]
        if existing.size:
            keep = np.ones(self._fps.shape[0], dtype=bool)
            keep[existing] = False
            self._fps = self._fps[keep]
            self._sources = self._sources[keep]

        self._fps = np.vstack([self._fps, fp_vec[np.newaxis, :]])
        self._sources = np.append(self._sources, source)
        self._save()
        return fp_vec

    def find_nearest(
        self,
        text: str,
        threshold: float | Callable[[int], float] = 0.97,
    ) -> tuple[str, float] | None:
        """Find the nearest stored document to ``text``.

        Args:
            text:      Document text to query.
            threshold: Minimum cosine similarity to report a match.  Can be
                       a float or a callable ``(doc_chars: int) -> float``
                       for dynamic thresholds (see future work in repo memory
                       ``vox_romana_dynamic_dedup_threshold.md``).

        Returns:
            ``(source, similarity)`` of the closest match above threshold,
            or ``None`` if no match exceeds the threshold.
        """
        self._load()

        assert self._fps is not None
        assert self._sources is not None

        if self._fps.shape[0] == 0:
            return None

        fp_vec = self._fp.build(text)

        # Resolve threshold.
        if callable(threshold):
            t = threshold(len(text))
        else:
            t = float(threshold)

        # Stack query with corpus for a single pairwise_cosine call.
        stacked = np.vstack([fp_vec[np.newaxis, :], self._fps])
        sims = self._fp.pairwise_cosine(stacked)

        # sims[0, 1:] are similarities of query vs each stored doc.
        query_sims = sims[0, 1:]
        best_idx = int(np.argmax(query_sims))
        best_sim = float(query_sims[best_idx])

        if best_sim >= t:
            return (str(self._sources[best_idx]), best_sim)
        return None

    def remove(self, source: str) -> bool:
        """Remove the fingerprint for ``source``.

        Returns:
            ``True`` if the source was found and removed, ``False`` otherwise.
        """
        self._load()
        assert self._fps is not None
        assert self._sources is not None

        idx = np.where(self._sources == source)[0]
        if not idx.size:
            return False
        keep = np.ones(self._fps.shape[0], dtype=bool)
        keep[idx] = False
        self._fps = self._fps[keep]
        self._sources = self._sources[keep]
        self._save()
        return True

    def build_parallel(
        self,
        jsonl_dir: Path | str,
        *,
        workers: int = 4,
        min_chars: int = 500,
        verbose: bool = False,
    ) -> int:
        """Rebuild fingerprints from all JSONL files in ``jsonl_dir``.

        Existing entries whose source name matches a JSONL filename are
        replaced (upsert semantics).  Uses :class:`ProcessPoolExecutor`
        for parallel fingerprinting — embarrassingly parallel across files.

        Args:
            jsonl_dir:  Directory containing ``*.jsonl`` chunk files.
            workers:    Number of worker processes.
            min_chars:  Minimum combined text length to fingerprint a file.
                        Files below this threshold are skipped (too short
                        for a stable bigram distribution).
            verbose:    Print progress to stdout.

        Returns:
            Number of fingerprints added or updated.
        """
        jsonl_dir = Path(jsonl_dir)
        all_files = sorted(jsonl_dir.glob("*.jsonl"))
        if not all_files:
            return 0

        # Partition files across workers.
        paths_str = [str(p) for p in all_files]
        chunk_size = max(1, len(paths_str) // workers)
        slices = [
            paths_str[i : i + chunk_size]
            for i in range(0, len(paths_str), chunk_size)
        ]

        if verbose:
            print(f"  Fingerprinting {len(all_files)} JSONL file(s) across {len(slices)} worker(s)...")

        merged: dict[str, np.ndarray] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_fingerprint_jsonl_slice, s): s for s in slices}
            done = 0
            for fut in as_completed(futs):
                result = fut.result()
                merged.update(result)
                done += 1
                if verbose:
                    print(f"\r  Fingerprinted {done}/{len(slices)} slice(s)...", end="", flush=True)

        if verbose:
            print()  # newline after \r progress

        # Upsert all results into the store.
        self._load()
        added = 0
        for filename, fp_vec in merged.items():
            source = filename  # JSONL filename as source key
            existing = np.where(self._sources == source)[0]
            if existing.size:
                keep = np.ones(self._fps.shape[0], dtype=bool)
                keep[existing] = False
                self._fps = self._fps[keep]
                self._sources = self._sources[keep]
            self._fps = np.vstack([self._fps, fp_vec[np.newaxis, :]])
            self._sources = np.append(self._sources, source)
            added += 1

        if added:
            self._save()

        if verbose:
            print(f"  Fingerprint store: {added} file(s) indexed, {self.size} total.")

        return added

    def rebuild(self, jsonl_dir: Path | str, *, workers: int = 4, verbose: bool = False) -> int:
        """Clear the store and rebuild from scratch from ``jsonl_dir``.

        Use after ``--dedupe --apply`` has removed JSONL files to keep the
        store consistent.

        Returns:
            Number of fingerprints written.
        """
        self._fps = np.zeros((0, self._fp.vector_dim), dtype=np.float32)
        self._sources = np.empty(0, dtype=object)
        return self.build_parallel(jsonl_dir, workers=workers, verbose=verbose)

    def nearest_pairs(
        self,
        threshold: float = 0.97,
    ) -> list[tuple[str, str, float]]:
        """Find all pairs of stored documents above ``threshold``.

        Runs a single :meth:`~vox_romana.core.markov.MarkovFingerprinter.pairwise_cosine`
        call over the full corpus.  Returns unique unordered pairs.

        Returns:
            List of ``(source_a, source_b, similarity)`` sorted by similarity
            descending.
        """
        self._load()
        assert self._fps is not None
        assert self._sources is not None

        n = self._fps.shape[0]
        if n < 2:
            return []

        sims = self._fp.pairwise_cosine(self._fps)

        pairs: list[tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sims[i, j])
                if sim >= threshold:
                    pairs.append((str(self._sources[i]), str(self._sources[j]), sim))

        pairs.sort(key=lambda t: t[2], reverse=True)
        return pairs

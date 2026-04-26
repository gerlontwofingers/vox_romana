"""Derivativeness and stylometric similarity analysis between two corpora.

Produces a :class:`~vox_romana.analysis.models.SimilarityReport` measuring
three orthogonal axes:

1. **N-gram overlap** — weighted character n-gram Jaccard similarity (n=2..4)
   over the full-text of each corpus.  Pure lexical proximity.

2. **Semantic similarity** — mean cross-corpus pairwise cosine similarity
   computed from L2-normalised embeddings.  Captures paraphrase and shared
   framing even when word choice differs.

3. **Burrows' Delta** — mean absolute Z-score difference on the top-100 most
   frequent words.  The classical stylometric distance; 0.0 = identical style.

Alongside these, a ``stylometric`` dict is populated with raw features:
type-token ratio (TTR), mean utterance length, vocabulary size, and lexical
density (content-word fraction).

VERDICT THRESHOLDS::

    ngram_overlap > 0.70  OR  semantic_similarity > 0.85
        → "highly similar"   (likely derivative / plagiarism risk)
    ngram_overlap > 0.40  OR  semantic_similarity > 0.65
        → "moderately similar"
    otherwise
        → "distinct"

USAGE::

    from vox_romana.analysis.similarity import SimilarityAnalyzer
    from vox_romana.analysis.embedder import CorpusEmbedder

    embedder = CorpusEmbedder()
    emb_a, emb_b = embedder.embed_pair(corpus_a, corpus_b)

    report = SimilarityAnalyzer().analyze(corpus_a, corpus_b, emb_a, emb_b)
    print(report.verdict)      # "distinct", "moderately similar", "highly similar"
    print(report.ngram_overlap)
    print(report.semantic_similarity)
    print(report.burrows_delta)
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import numpy as np

from .models import SimilarityReport, SpeakerCorpus

logger = logging.getLogger(__name__)

_TOKENIZE = re.compile(r"\b[a-z]+\b")

# Approximate list of common English function words for lexical density
_FUNCTION_WORDS = frozenset(
    "a an the and or but if in on at to of for with by from as is was are were "
    "be been being have has had do does did will would could should may might "
    "shall must can this that these those it its i me my we our you your he she "
    "him her they them their not no nor so yet both either neither each every "
    "any all some few more most other such only than then there here when where "
    "who which what how just also very too about after before during between "
    "through under over above below up down out off since until while".split()
)


class SimilarityAnalyzer:
    """Computes derivativeness and stylometric similarity between two corpora.

    The analyzer is stateless — create one instance and call :meth:`analyze`
    as many times as needed.

    Args:
        highly_similar_ngram:    N-gram overlap threshold for "highly similar".
        highly_similar_semantic: Semantic similarity threshold for "highly similar".
        moderately_similar_ngram:    Threshold for "moderately similar".
        moderately_similar_semantic: Threshold for "moderately similar".
        ngram_max_n:             Maximum n-gram size for overlap calculation.
    """

    def __init__(
        self,
        *,
        highly_similar_ngram: float = 0.70,
        highly_similar_semantic: float = 0.85,
        moderately_similar_ngram: float = 0.40,
        moderately_similar_semantic: float = 0.65,
        ngram_max_n: int = 4,
    ) -> None:
        self.highly_similar_ngram = highly_similar_ngram
        self.highly_similar_semantic = highly_similar_semantic
        self.moderately_similar_ngram = moderately_similar_ngram
        self.moderately_similar_semantic = moderately_similar_semantic
        self.ngram_max_n = ngram_max_n

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        corpus_a: SpeakerCorpus,
        corpus_b: SpeakerCorpus,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
    ) -> SimilarityReport:
        """Compute all similarity metrics and return a :class:`SimilarityReport`.

        Args:
            corpus_a, corpus_b: Speaker corpora to compare.
            emb_a, emb_b:       L2-normalised embeddings from
                                :class:`~vox_romana.analysis.embedder.CorpusEmbedder`.

        Returns:
            :class:`~vox_romana.analysis.models.SimilarityReport` with all
            fields populated and a human-readable verdict.
        """
        ng = self.ngram_overlap(corpus_a, corpus_b)
        sem = self.semantic_similarity(emb_a, emb_b)
        delta = self.burrows_delta(corpus_a, corpus_b)
        stylo = self.stylometric_features(corpus_a, corpus_b)
        verdict = self._verdict(ng, sem)

        logger.debug(
            "SimilarityAnalyzer: %s vs %s → ngram=%.3f sem=%.3f delta=%.3f verdict=%s",
            corpus_a.name,
            corpus_b.name,
            ng,
            sem,
            delta,
            verdict,
        )

        return SimilarityReport(
            ngram_overlap=ng,
            semantic_similarity=sem,
            burrows_delta=delta,
            stylometric=stylo,
            verdict=verdict,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Individual metrics
    # ─────────────────────────────────────────────────────────────────────────

    def ngram_overlap(
        self, corpus_a: SpeakerCorpus, corpus_b: SpeakerCorpus
    ) -> float:
        """Weighted character n-gram Jaccard overlap (n=2..max_n).

        Each n-gram size is weighted equally.  Returns a value in [0, 1]
        where 1.0 means the two corpora share all character n-grams.

        Uses the full-text of each corpus (all utterances joined).
        """
        text_a = corpus_a.full_text.lower()
        text_b = corpus_b.full_text.lower()
        scores: list[float] = []

        for n in range(2, self.ngram_max_n + 1):
            set_a = self._char_ngrams(text_a, n)
            set_b = self._char_ngrams(text_b, n)
            union = set_a | set_b
            if not union:
                scores.append(0.0)
                continue
            intersection = set_a & set_b
            scores.append(len(intersection) / len(union))

        return float(np.mean(scores)) if scores else 0.0

    def semantic_similarity(
        self, emb_a: np.ndarray, emb_b: np.ndarray
    ) -> float:
        """Mean cross-corpus pairwise cosine similarity.

        Since embeddings are L2-normalised, cosine similarity = dot product.
        Capped at 200 utterances per corpus to keep compute tractable.

        Returns a value in [−1, 1] (practically [0, 1] for semantic embeddings).
        """
        MAX = 200
        a = emb_a[:MAX]
        b = emb_b[:MAX]
        cross = a @ b.T          # (N_a, N_b)
        return float(cross.mean())

    def burrows_delta(
        self, corpus_a: SpeakerCorpus, corpus_b: SpeakerCorpus
    ) -> float:
        """Burrows' Delta stylometric distance (0.0 = identical style).

        Uses top-100 most frequent words from the combined corpus.
        Computes per-utterance relative frequencies, Z-score-normalises on
        the combined statistics, then returns the mean absolute Z-score
        difference between the two speakers' mean vectors.
        """
        all_utts = corpus_a.utterances + corpus_b.utterances
        na = len(corpus_a.utterances)
        n_total = len(all_utts)

        all_tokens = _TOKENIZE.findall(" ".join(all_utts).lower())
        n_words = min(100, max(1, len(set(all_tokens))))
        vocab = [w for w, _ in Counter(all_tokens).most_common(n_words)]
        word_idx = {w: i for i, w in enumerate(vocab)}

        seg_freqs = np.zeros((n_total, n_words), dtype=np.float32)
        for s, utt in enumerate(all_utts):
            toks = _TOKENIZE.findall(utt.lower())
            total = max(len(toks), 1)
            for tok in toks:
                if tok in word_idx:
                    seg_freqs[s, word_idx[tok]] += 1.0 / total

        mean_f = seg_freqs.mean(axis=0)
        std_f = seg_freqs.std(axis=0) + 1e-10

        mean_a = seg_freqs[:na].mean(axis=0)
        mean_b = seg_freqs[na:].mean(axis=0)
        z_a = (mean_a - mean_f) / std_f
        z_b = (mean_b - mean_f) / std_f
        return float(np.abs(z_a - z_b).mean())

    def stylometric_features(
        self, corpus_a: SpeakerCorpus, corpus_b: SpeakerCorpus
    ) -> dict[str, object]:
        """Compute raw stylometric features for both corpora.

        Returns a dict with keys:
            ttr_a, ttr_b           — type-token ratio (vocabulary richness)
            vocab_a, vocab_b       — unique word count
            mean_len_a, mean_len_b — mean utterance length in words
            lexical_density_a/b    — fraction of content words
        """
        def _features(corpus: SpeakerCorpus, suffix: str) -> dict[str, object]:
            tokens = _TOKENIZE.findall(corpus.full_text.lower())
            total = max(len(tokens), 1)
            unique = len(set(tokens))
            ttr = unique / total
            mean_len = float(np.mean([len(u.split()) for u in corpus.utterances]))
            content_words = sum(1 for t in tokens if t not in _FUNCTION_WORDS)
            lex_density = content_words / total
            return {
                f"ttr_{suffix}": round(ttr, 4),
                f"vocab_{suffix}": unique,
                f"mean_len_{suffix}": round(mean_len, 2),
                f"lexical_density_{suffix}": round(lex_density, 4),
            }

        out: dict[str, object] = {}
        out.update(_features(corpus_a, "a"))
        out.update(_features(corpus_b, "b"))
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _char_ngrams(text: str, n: int) -> set[str]:
        """Return the set of character n-grams from a string."""
        if len(text) < n:
            return set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def _verdict(self, ngram: float, semantic: float) -> str:
        if ngram > self.highly_similar_ngram or semantic > self.highly_similar_semantic:
            return "highly similar"
        if ngram > self.moderately_similar_ngram or semantic > self.moderately_similar_semantic:
            return "moderately similar"
        return "distinct"

"""Data models for the vox_romana analysis pipeline.

All dataclasses used across the full analysis system are defined here so
downstream modules import from one place.  Shot 1 uses SpeakerCorpus;
later shots use the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# SpeakerCorpus — the fundamental unit of analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SpeakerCorpus:
    """A collection of utterances attributed to a single speaker.

    Attributes:
        name:       Display name of the speaker.
        utterances: List of individual utterances (sentences, lines, or chunks).
        metadata:   Optional dict of provenance information (source file, etc.).
    """

    name: str
    utterances: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.utterances:
            raise ValueError(
                f"SpeakerCorpus '{self.name}' must have at least one utterance."
            )

    @property
    def full_text(self) -> str:
        """All utterances joined into a single string."""
        return " ".join(self.utterances)

    @property
    def word_count(self) -> int:
        """Total word count across all utterances."""
        return sum(len(u.split()) for u in self.utterances)

    @property
    def utterance_count(self) -> int:
        """Number of utterances."""
        return len(self.utterances)


# ─────────────────────────────────────────────────────────────────────────────
# TestResult — result of a single statistical hypothesis test
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TestResult:
    """Result of a single hypothesis test.

    Attributes:
        name:        Human-readable test name.
        statistic:   Observed test statistic.
        p_value:     FDR-corrected p-value (Benjamini-Hochberg).
        effect_size: Effect size (rank-biserial r, eta², or equivalent).
        ci_low:      95% bootstrap CI lower bound for the test statistic.
        ci_high:     95% bootstrap CI upper bound for the test statistic.
        passed:      True = H0 rejected = speakers ARE statistically distinct
                     on this test at FDR-corrected p < 0.05.
        notes:       Optional human-readable note.
    """

    name: str
    statistic: float
    p_value: float
    effect_size: float
    ci_low: float
    ci_high: float
    passed: bool
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# StanceRecord — inferred speaker position on a topic
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class StanceRecord:
    """A speaker's inferred position on a single topic.

    Produced by StanceExtractor (Shot 3) via LLM calls over topic labels.

    Attributes:
        speaker:        Speaker name.
        topic_id:       LDA topic index.
        topic_label:    Human-readable label from TopicModeler.top_terms().
        position:       Brief stance: "pro", "anti", "neutral", "ambiguous".
        summary:        1–2 sentence summary of the position.
        confidence:     Extractor confidence, 0.0–1.0.
        evidence_quote: Representative quote from the corpus (optional).
    """

    speaker: str
    topic_id: int
    topic_label: str
    position: str
    summary: str
    confidence: float
    evidence_quote: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# HypothesisResult — combined output of the full test battery
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class HypothesisResult:
    """Combined output of the full six-test hypothesis battery.

    Verdict logic: H0 rejected ("REJECTED") only if >= 3 of 6 tests agree
    at FDR-corrected p < 0.05.

    Attributes:
        speaker_a:     Name of the first speaker.
        speaker_b:     Name of the second speaker.
        test_results:  List of TestResult objects (one per test).
        verdict:       "REJECTED" or "NOT_REJECTED".
        fdr_corrected: Whether Benjamini-Hochberg FDR correction was applied.
        notes:         Free-text summary or caveats.
    """

    speaker_a: str
    speaker_b: str
    test_results: list[TestResult]
    verdict: str
    fdr_corrected: bool = True
    notes: str = ""

    @property
    def tests_passed(self) -> int:
        """Number of tests that rejected H0."""
        return sum(1 for t in self.test_results if t.passed)

    @property
    def total_tests(self) -> int:
        """Total number of tests run."""
        return len(self.test_results)

    @property
    def rejection_rate(self) -> float:
        """Fraction of tests that rejected H0 (0.0–1.0)."""
        if not self.test_results:
            return 0.0
        return self.tests_passed / self.total_tests


# ─────────────────────────────────────────────────────────────────────────────
# SilenceReport — topics systematically avoided by one or both speakers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SilenceReport:
    """Topics avoided by one or both speakers.

    Attributes:
        avoided_by_both:         Topic labels that neither speaker addresses.
        avoided_by_a:            Topics present in B's discourse but absent
                                 from A's.
        avoided_by_b:            Topics present in A's discourse but absent
                                 from B's.
        reference_topics_missed: Topics from an external reference corpus
                                 that neither speaker addresses.
    """

    avoided_by_both: list[str]
    avoided_by_a: list[str]
    avoided_by_b: list[str]
    reference_topics_missed: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# SimilarityReport — derivativeness / plagiarism analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SimilarityReport:
    """Derivativeness and stylometric similarity between two corpora.

    Attributes:
        ngram_overlap:       Weighted n-gram overlap score, 0.0–1.0.
        semantic_similarity: Mean pairwise cosine similarity, 0.0–1.0.
        burrows_delta:       Burrows' Delta stylometric distance (0.0 = identical).
        stylometric:         Dict of raw stylometric features (TTR, MTLD, etc.).
        verdict:             "highly similar", "moderately similar", or "distinct".
    """

    ngram_overlap: float
    semantic_similarity: float
    burrows_delta: float
    stylometric: dict[str, Any] = field(default_factory=dict)
    verdict: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# AnalysisReport — top-level result combining all components
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class AnalysisReport:
    """Top-level result combining all analysis components.

    Attributes:
        speaker_a:        Name of the first speaker / corpus.
        speaker_b:        Name of the second speaker / corpus.
        hypothesis:       Full hypothesis test battery result (or None).
        similarity:       Similarity / derivativeness analysis (or None).
        silence:          Silence / avoidance analysis (or None).
        stances:          List of per-speaker stance records (may be empty).
        report_path:      Path to the rendered HTML report (or "").
        catalog_study_id: Catalog study ID if result was logged (or "").
    """

    speaker_a: str
    speaker_b: str
    hypothesis: HypothesisResult | None = None
    similarity: SimilarityReport | None = None
    silence: SilenceReport | None = None
    stances: list[StanceRecord] = field(default_factory=list)
    report_path: str = ""
    catalog_study_id: str = ""

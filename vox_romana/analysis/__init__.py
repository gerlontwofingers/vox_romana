"""vox_romana.analysis — Rhetorical hypothesis testing and corpus analysis.

Public API::

    from vox_romana.analysis import CorpusLoader, CorpusEmbedder
    from vox_romana.analysis import (
        SpeakerCorpus, TestResult, HypothesisResult,
        StanceRecord, SilenceReport, SimilarityReport, AnalysisReport,
    )

Workflow::

    # 1. Load corpora
    corpora = CorpusLoader.from_vox_romana_json("output/text/debates/debate.json")
    corpus_a = corpora["Candidate A"]
    corpus_b = corpora["Candidate B"]

    # 2. Embed
    embedder = CorpusEmbedder()
    emb_a, emb_b = embedder.embed_pair(corpus_a, corpus_b)

    # 3. Model topics and run hypothesis tests (Shot 2)
    from vox_romana.analysis import TopicModeler, HypothesisTester
    result = HypothesisTester().run(corpus_a, corpus_b, emb_a, emb_b)
    print(result.verdict)   # "REJECTED" or "NOT_REJECTED"

    # 4. Similarity, stance, silence (Shot 3)
    from vox_romana.analysis import SimilarityAnalyzer, StanceExtractor, SilenceDetector
    tm = TopicModeler(n_topics=8)
    dist_a, dist_b = tm.fit_transform(corpus_a, corpus_b)
    sim = SimilarityAnalyzer().analyze(corpus_a, corpus_b, emb_a, emb_b)
    stances = StanceExtractor().extract(corpus_a, corpus_b, tm, dist_a, dist_b)
    silence = SilenceDetector().detect(corpus_a, corpus_b, tm, dist_a, dist_b)
"""

from __future__ import annotations

from .corpus import CorpusLoader
from .embedder import CorpusEmbedder
from .hypothesis import HypothesisTester
from .models import (
    AnalysisReport,
    HypothesisResult,
    SilenceReport,
    SimilarityReport,
    SpeakerCorpus,
    StanceRecord,
    TestResult,
)
from .reporter import AnalysisReporter
from .silence import SilenceDetector
from .similarity import SimilarityAnalyzer
from .stance import StanceExtractor
from .topics import TopicModeler

__all__ = [
    "CorpusLoader",
    "CorpusEmbedder",
    "TopicModeler",
    "HypothesisTester",
    "SimilarityAnalyzer",
    "StanceExtractor",
    "SilenceDetector",
    "AnalysisReporter",
    "SpeakerCorpus",
    "TestResult",
    "HypothesisResult",
    "StanceRecord",
    "SilenceReport",
    "SimilarityReport",
    "AnalysisReport",
]

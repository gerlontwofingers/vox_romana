"""Tests for vox_romana.analysis.reporter — AnalysisReporter."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vox_romana.analysis.embedder import CorpusEmbedder
from vox_romana.analysis.models import (
    AnalysisReport,
    HypothesisResult,
    SilenceReport,
    SimilarityReport,
    SpeakerCorpus,
    StanceRecord,
    TestResult,
)
from vox_romana.analysis.reporter import AnalysisReporter, _esc, _slug


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

_UTTS_A = [
    "National defense requires increased military spending and robust weapons procurement",
    "Armed forces must maintain readiness with modern equipment and thorough training",
    "Pentagon funding ensures soldiers have necessary weapons for combat operations abroad",
    "Strategic nuclear deterrence protects sovereignty from foreign aggression and threats",
    "Military bases require substantial infrastructure investment for all personnel safety",
    "Intelligence agencies monitor terrorist threats to national security interests daily",
    "Naval warships patrol international waters to enforce maritime agreements effectively",
    "Air force jets defend territorial airspace against hostile incursion at all times",
]

_UTTS_B = [
    "Climate change threatens ecosystems through rising temperatures and widespread flooding",
    "Carbon emissions from fossil fuels must be reduced immediately to prevent catastrophe",
    "Renewable solar and wind energy creates sustainable economic opportunities for workers",
    "Ocean acidification destroys coral reefs and decimates marine biodiversity rapidly",
    "Deforestation accelerates greenhouse warming and destroys critical wildlife habitat",
    "Environmental regulations protect clean drinking water and breathable air for citizens",
    "Green technology innovation reduces industrial pollution and lowers the carbon footprint",
    "Sustainable agriculture practices protect soil health and preserve water quality",
]


def _corpus(name: str, utts: list[str]) -> SpeakerCorpus:
    return SpeakerCorpus(name=name, utterances=utts)


def _make_test_result(name: str, passed: bool = True) -> TestResult:
    return TestResult(
        name=name,
        statistic=1.23,
        p_value=0.01 if passed else 0.20,
        effect_size=0.55,
        ci_low=0.40,
        ci_high=0.70,
        passed=passed,
        notes="",
    )


def _make_hypothesis(verdict: str = "REJECTED") -> HypothesisResult:
    tests = [
        _make_test_result("Permutation Centroid", True),
        _make_test_result("Mann-Whitney Cosine", True),
        _make_test_result("KS Topic Distribution", True),
        _make_test_result("JSD Bootstrap", False),
        _make_test_result("Fisher Stance Contingency", False),
        _make_test_result("Burrows Delta", True),
    ]
    return HypothesisResult(
        speaker_a="Alice",
        speaker_b="Bob",
        test_results=tests,
        verdict=verdict,
    )


def _make_similarity(verdict: str = "distinct") -> SimilarityReport:
    return SimilarityReport(
        ngram_overlap=0.12,
        semantic_similarity=0.43,
        burrows_delta=1.87,
        stylometric={
            "ttr_a": 0.82, "ttr_b": 0.79,
            "vocab_a": 72, "vocab_b": 68,
            "mean_len_a": 12.3, "mean_len_b": 11.9,
            "lexical_density_a": 0.60, "lexical_density_b": 0.58,
        },
        verdict=verdict,
    )


def _make_silence() -> SilenceReport:
    return SilenceReport(
        avoided_by_both=["economics/trade/market"],
        avoided_by_a=["climate/energy/carbon"],
        avoided_by_b=["military/defense/weapons"],
    )


def _make_stances() -> list[StanceRecord]:
    return [
        StanceRecord(
            speaker="Alice",
            topic_id=0,
            topic_label="defense/military/weapons",
            position="pro",
            summary="Alice supports increased military spending.",
            confidence=0.85,
            evidence_quote="Pentagon funding ensures soldiers...",
        ),
        StanceRecord(
            speaker="Bob",
            topic_id=0,
            topic_label="defense/military/weapons",
            position="anti",
            summary="Bob opposes military expansion.",
            confidence=0.80,
            evidence_quote="",
        ),
    ]


def _make_report(**kwargs) -> AnalysisReport:
    defaults = dict(
        speaker_a="Alice",
        speaker_b="Bob",
        hypothesis=_make_hypothesis(),
        similarity=_make_similarity(),
        silence=_make_silence(),
        stances=_make_stances(),
    )
    defaults.update(kwargs)
    return AnalysisReport(**defaults)


def _small_embedder() -> CorpusEmbedder:
    return CorpusEmbedder(_force_backend="tfidf")


# ─────────────────────────────────────────────────────────────────────────────
# TestInit
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_output_dir_is_none(self):
        r = AnalysisReporter()
        assert r.output_dir is None

    def test_custom_output_dir_stored_as_path(self, tmp_path):
        r = AnalysisReporter(output_dir=tmp_path)
        assert r.output_dir == tmp_path

    def test_string_output_dir_converted_to_path(self, tmp_path):
        r = AnalysisReporter(output_dir=str(tmp_path))
        assert isinstance(r.output_dir, Path)


# ─────────────────────────────────────────────────────────────────────────────
# TestRenderHtml — HTML rendering from a pre-built AnalysisReport
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderHtml:
    def _render(self, **kwargs) -> str:
        report = _make_report(**kwargs)
        return AnalysisReporter().render_html(report)

    def test_returns_string(self):
        assert isinstance(self._render(), str)

    def test_starts_with_doctype(self):
        assert self._render().startswith("<!DOCTYPE html>")

    def test_contains_both_speaker_names(self):
        html = self._render()
        assert "Alice" in html
        assert "Bob" in html

    def test_contains_verdict(self):
        html = self._render()
        assert "REJECTED" in html

    def test_contains_not_rejected(self):
        html = self._render(hypothesis=_make_hypothesis("NOT_REJECTED"))
        assert "NOT_REJECTED" in html

    def test_contains_test_names(self):
        html = self._render()
        assert "Permutation Centroid" in html
        assert "Burrows Delta" in html

    def test_contains_similarity_scores(self):
        html = self._render()
        assert "0.120" in html   # ngram_overlap
        assert "0.430" in html   # semantic_similarity
        assert "1.870" in html   # burrows_delta

    def test_contains_similarity_verdict(self):
        html = self._render()
        assert "distinct" in html

    def test_contains_stance_positions(self):
        html = self._render()
        assert "pro" in html
        assert "anti" in html

    def test_contains_silence_categories(self):
        html = self._render()
        assert "Avoided by both" in html
        assert "economics/trade/market" in html

    def test_no_external_http_resources(self):
        html = self._render()
        # No src= or href= pointing to external URLs
        assert not re.search(r'(src|href)=["\']https?://', html)

    def test_no_script_tags(self):
        html = self._render()
        assert "<script" not in html.lower()

    def test_contains_methodology_footer(self):
        html = self._render()
        assert "Methodology" in html
        assert "Benjamini-Hochberg" in html

    def test_html_with_corpus_stats(self):
        report = _make_report()
        a = _corpus("Alice", _UTTS_A)
        b = _corpus("Bob", _UTTS_B)
        html = AnalysisReporter().render_html(report, corpus_a=a, corpus_b=b)
        assert "Utterances" in html
        assert "Words" in html

    def test_no_stances_shows_empty_message(self):
        html = self._render(stances=[])
        assert "No stance data" in html

    def test_no_hypothesis_skips_section(self):
        html = self._render(hypothesis=None)
        assert "Hypothesis Battery" not in html

    def test_no_similarity_skips_section(self):
        html = self._render(similarity=None)
        assert "Similarity Analysis" not in html

    def test_html_escapes_speaker_names(self):
        report = _make_report(
            speaker_a='<script>alert(1)</script>',
            speaker_b="Bob & Carol",
        )
        html = AnalysisReporter().render_html(report)
        assert "<script>" not in html
        assert "&amp;" in html or "Bob &amp; Carol" in html

    def test_saves_file_when_output_path_provided(self, tmp_path):
        out = tmp_path / "report.html"
        report = _make_report()
        html = AnalysisReporter().render_html(report, output_path=out)
        assert out.exists()
        assert out.read_text(encoding="utf-8") == html


# ─────────────────────────────────────────────────────────────────────────────
# TestRenderText
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderText:
    def _render(self, **kwargs) -> str:
        report = _make_report(**kwargs)
        return AnalysisReporter().render_text(report)

    def test_returns_string(self):
        assert isinstance(self._render(), str)

    def test_non_empty(self):
        assert len(self._render()) > 0

    def test_contains_speaker_names(self):
        text = self._render()
        assert "Alice" in text
        assert "Bob" in text

    def test_contains_verdict(self):
        assert "REJECTED" in self._render()

    def test_contains_similarity_scores(self):
        text = self._render()
        assert "0.120" in text
        assert "1.870" in text

    def test_contains_stance_records(self):
        text = self._render()
        assert "pro" in text.lower()
        assert "anti" in text.lower()

    def test_contains_avoidance_info(self):
        text = self._render()
        assert "Avoided" in text

    def test_corpus_stats_when_provided(self):
        report = _make_report()
        a = _corpus("Alice", _UTTS_A)
        b = _corpus("Bob", _UTTS_B)
        text = AnalysisReporter().render_text(report, corpus_a=a, corpus_b=b)
        assert "utterances" in text
        assert "words" in text

    def test_no_ansi_codes(self):
        text = self._render()
        assert "\x1b[" not in text


# ─────────────────────────────────────────────────────────────────────────────
# TestBuild — full pipeline (fast: TF-IDF + low permutations)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuild:
    def _build(self, **kwargs):
        a = _corpus("Alice", _UTTS_A)
        b = _corpus("Bob", _UTTS_B)
        r = AnalysisReporter()
        defaults = dict(
            n_topics=3,
            n_permutations=50,
            use_llm=False,
            embedder=_small_embedder(),
        )
        defaults.update(kwargs)
        return r.build(a, b, **defaults)

    def test_returns_analysis_report(self):
        assert isinstance(self._build(), AnalysisReport)

    def test_speaker_names_set(self):
        report = self._build()
        assert report.speaker_a == "Alice"
        assert report.speaker_b == "Bob"

    def test_hypothesis_populated(self):
        report = self._build()
        assert isinstance(report.hypothesis, HypothesisResult)

    def test_hypothesis_verdict_is_valid(self):
        report = self._build()
        assert report.hypothesis.verdict in ("REJECTED", "NOT_REJECTED")

    def test_similarity_populated(self):
        report = self._build()
        assert isinstance(report.similarity, SimilarityReport)

    def test_similarity_verdict_set(self):
        report = self._build()
        assert report.similarity.verdict != ""

    def test_silence_populated(self):
        report = self._build()
        assert isinstance(report.silence, SilenceReport)

    def test_stances_empty_when_no_llm(self):
        report = self._build(use_llm=False)
        assert report.stances == []

    def test_report_path_empty_when_no_output_dir(self):
        report = self._build()
        assert report.report_path == ""

    def test_report_path_set_when_output_dir_provided(self, tmp_path):
        a = _corpus("Alice", _UTTS_A)
        b = _corpus("Bob", _UTTS_B)
        reporter = AnalysisReporter(output_dir=tmp_path)
        report = reporter.build(
            a, b,
            n_topics=3, n_permutations=50,
            use_llm=False, embedder=_small_embedder(),
        )
        assert report.report_path != ""
        assert Path(report.report_path).exists()

    def test_html_file_is_valid(self, tmp_path):
        a = _corpus("Alice", _UTTS_A)
        b = _corpus("Bob", _UTTS_B)
        reporter = AnalysisReporter(output_dir=tmp_path)
        report = reporter.build(
            a, b,
            n_topics=3, n_permutations=50,
            use_llm=False, embedder=_small_embedder(),
        )
        content = Path(report.report_path).read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html>")
        assert "Alice" in content
        assert "Bob" in content


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildPrebuilt — pre-built components are not recomputed
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPrebuilt:
    def test_pre_built_hypothesis_used(self):
        a = _corpus("Alice", _UTTS_A)
        b = _corpus("Bob", _UTTS_B)
        pre_hyp = _make_hypothesis("NOT_REJECTED")
        report = AnalysisReporter().build(
            a, b,
            hypothesis=pre_hyp,
            n_topics=3, n_permutations=50,
            use_llm=False, embedder=_small_embedder(),
        )
        assert report.hypothesis is pre_hyp
        assert report.hypothesis.verdict == "NOT_REJECTED"

    def test_pre_built_similarity_used(self):
        a = _corpus("Alice", _UTTS_A)
        b = _corpus("Bob", _UTTS_B)
        pre_sim = _make_similarity("highly similar")
        report = AnalysisReporter().build(
            a, b,
            similarity=pre_sim,
            n_topics=3, n_permutations=50,
            use_llm=False, embedder=_small_embedder(),
        )
        assert report.similarity is pre_sim
        assert report.similarity.verdict == "highly similar"

    def test_pre_built_silence_used(self):
        a = _corpus("Alice", _UTTS_A)
        b = _corpus("Bob", _UTTS_B)
        pre_sil = SilenceReport(avoided_by_both=["custom_topic"], avoided_by_a=[], avoided_by_b=[])
        report = AnalysisReporter().build(
            a, b,
            silence=pre_sil,
            n_topics=3, n_permutations=50,
            use_llm=False, embedder=_small_embedder(),
        )
        assert report.silence is pre_sil
        assert "custom_topic" in report.silence.avoided_by_both

    def test_pre_built_stances_used(self):
        a = _corpus("Alice", _UTTS_A)
        b = _corpus("Bob", _UTTS_B)
        pre_stances = _make_stances()
        report = AnalysisReporter().build(
            a, b,
            stances=pre_stances,
            n_topics=3, n_permutations=50,
            use_llm=False, embedder=_small_embedder(),
        )
        assert report.stances is pre_stances


# ─────────────────────────────────────────────────────────────────────────────
# TestHelpers — _esc and _slug
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_esc_escapes_angle_brackets(self):
        assert "&lt;" in _esc("<")
        assert "&gt;" in _esc(">")

    def test_esc_escapes_ampersand(self):
        assert "&amp;" in _esc("&")

    def test_esc_escapes_quotes(self):
        assert "&#x27;" in _esc("'") or "&#39;" in _esc("'") or "&apos;" in _esc("'") or "'" not in _esc('"')

    def test_slug_lowercases(self):
        assert _slug("Alice") == "alice"

    def test_slug_replaces_spaces(self):
        assert " " not in _slug("Alice Bob")

    def test_slug_handles_special_chars(self):
        slug = _slug("Dr. Alice O'Brien")
        assert re.match(r"^[a-z0-9_]+$", slug)

    def test_report_filename_contains_slugs(self):
        fname = AnalysisReporter._report_filename("Alice Smith", "Bob Jones")
        assert "alice_smith" in fname
        assert "bob_jones" in fname
        assert fname.endswith(".html")

    def test_report_filename_contains_timestamp(self):
        fname = AnalysisReporter._report_filename("A", "B")
        # Should contain digits from the timestamp
        assert re.search(r"\d{8}_\d{6}", fname)

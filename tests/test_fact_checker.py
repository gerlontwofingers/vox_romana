"""Tests for FactChecker — factual accuracy gating for generated content.

This file tests every layer of the FactChecker pipeline in isolation using
canned data and mock LLM clients. No real API calls are made.

TEST PLAN:

  TestParseClaims         — parse_claims() with canned JSON (happy paths + edge cases)
  TestParseResults        — parse_results() verdict normalisation and fallbacks
  TestExtractClaims       — extract_claims() LLM call + parse pipeline (mocked)
  TestVerifyClaims        — verify_claims() quick mode: ref material vs claims
  TestThoroughWithSearch  — _upgrade_with_search() searches unverifiable claims
  TestThoroughFallback    — thorough mode without search_provider falls back to quick
  TestCheckContent        — check_content() end-to-end integration (all modes)
  TestFactCheckReport— has_fabrications, fabricated_claims, to_dict/from_dict
  TestContentEditorBlocking — ContentEditor injects FactChecker result into EditorVerdict
  TestLLMFailureSafety    — all parse errors and LLM exceptions → conservative fallback
  [v2] TestContentModeCheck    — pure-fiction skip, fictional-universe canon prompt
  [v2] TestPriorContextConsistency — _check_consistency, _merge_consistency_results
  [v2] TestTrimPriorContext    — _trim_prior_context truncation helper
  [v2] TestBuildDefault        — build_default() classmethod wiring
  [v2] TestContradictionReport — has_contradictions, contradicted_claims on report

Run with:
    cd /Users/bagheera/repos/vox_romana
    source .venv/bin/activate
    python -m pytest tests/test_fact_checker.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from vox_romana.core.editing.editor import EDITOR_SYSTEM_PROMPT, ContentEditor
from vox_romana.core.editing.models import (
    FactCheckReport,
    EditorVerdict,
    FlaggedClaim,
    FactCheckResult,
    OutlineSection,
    OutlineSnapshot,
)
from vox_romana.core.fact_checker import FactChecker, _strip_fences, _trim_prior_context
from vox_romana.core.search import BaseSearchProvider, SearchResult


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _make_client(*responses: str) -> Mock:
    """Build a mock DeepSeekClient whose .chat() returns responses in order."""
    client = Mock()
    client.chat = Mock(side_effect=list(responses))
    return client


def _make_snapshot(version: int = 1) -> OutlineSnapshot:
    return OutlineSnapshot(
        outline_id="test",
        version=version,
        timestamp="2026-04-24T00:00:00Z",
        summary="Test outline",
        sections=[
            OutlineSection(
                id="intro",
                title="Section 1",
                expected_duration="5 min",
                key_beats=["topic A"],
                speaker_assignments=["Host"],
            )
        ],
    )


def _make_editor_session(verdict_json: str) -> Mock:
    """Return a mock AgentSession that responds with the given verdict JSON."""
    session = Mock()
    session.send = Mock(return_value=verdict_json)
    return session


class _MockSearchProvider(BaseSearchProvider):
    """A search provider that returns canned results."""

    def __init__(self, results: list[SearchResult] | None = None):
        self._results = results or [
            SearchResult(
                title="Dosio obituary",
                url="https://example.com/dosio",
                snippet="Ernie Dosio died on April 24, 2026 in a car accident in Melbourne.",
            ),
        ]

    def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        return self._results[:num_results]


class _FailingSearchProvider(BaseSearchProvider):
    """A search provider that always raises."""

    def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        raise ConnectionError("Search is down!")


# ═══════════════════════════════════════════════════════════════════════════
# TEST: _strip_fences helper
# ═══════════════════════════════════════════════════════════════════════════

class TestStripFences:
    """Unit-test the markdown-fence-stripping helper."""

    def test_plain_json_unchanged(self):
        assert _strip_fences('["a", "b"]') == '["a", "b"]'

    def test_json_fenced(self):
        raw = '```json\n["a"]\n```'
        assert _strip_fences(raw) == '["a"]'

    def test_bare_fenced(self):
        raw = '```\n["a"]\n```'
        assert _strip_fences(raw) == '["a"]'

    def test_strips_surrounding_whitespace(self):
        assert _strip_fences('  {"x": 1}  ') == '{"x": 1}'


# ═══════════════════════════════════════════════════════════════════════════
# TEST: parse_claims
# ═══════════════════════════════════════════════════════════════════════════

class TestParseClaims:
    """parse_claims() with canned JSON — no LLM involved."""

    @pytest.fixture
    def checker(self):
        return FactChecker(client=Mock())

    def test_valid_claims(self, checker):
        raw = json.dumps([
            {"claim": "Dosio died in 2022", "reason": "death date", "search_query": "Dosio death"},
            {"claim": "Pacific AgriLands had 15k acres", "reason": "land size", "search_query": "Pacific AgriLands size"},
        ])
        claims = checker.parse_claims(raw)
        assert len(claims) == 2
        assert claims[0].text == "Dosio died in 2022"
        assert claims[0].reason == "death date"
        assert claims[0].search_query == "Dosio death"
        assert claims[1].text == "Pacific AgriLands had 15k acres"

    def test_markdown_fenced_json(self, checker):
        raw = '```json\n[{"claim": "X died in 2020", "reason": "r", "search_query": "q"}]\n```'
        claims = checker.parse_claims(raw)
        assert len(claims) == 1
        assert claims[0].text == "X died in 2020"

    def test_empty_array(self, checker):
        assert checker.parse_claims("[]") == []

    def test_bad_json_returns_empty(self, checker):
        """Unparseable JSON → conservative empty list, no exception."""
        assert checker.parse_claims("not json at all") == []

    def test_items_missing_claim_key_skipped(self, checker):
        raw = json.dumps([
            {"claim": "Real claim", "reason": "r", "search_query": None},
            {"reason": "no claim field here"},
        ])
        claims = checker.parse_claims(raw)
        assert len(claims) == 1
        assert claims[0].text == "Real claim"

    def test_null_search_query_becomes_none(self, checker):
        raw = json.dumps([{"claim": "c", "reason": "r", "search_query": None}])
        claims = checker.parse_claims(raw)
        assert claims[0].search_query is None

    def test_non_list_returns_empty(self, checker):
        """LLM returned an object instead of array → conservative empty."""
        assert checker.parse_claims('{"claim": "oops"}') == []


# ═══════════════════════════════════════════════════════════════════════════
# TEST: parse_results
# ═══════════════════════════════════════════════════════════════════════════

class TestParseResults:
    """parse_results() verdict normalisation and fallbacks."""

    @pytest.fixture
    def checker(self):
        return FactChecker(client=Mock())

    def test_valid_results(self, checker):
        raw = json.dumps([
            {"claim": "X died in 2022", "verdict": "fabricated", "explanation": "Died in 2026", "source": "ref"},
            {"claim": "Y founded in 2003", "verdict": "verified", "explanation": "Confirmed", "source": "ref"},
            {"claim": "Z had 10k users", "verdict": "unverifiable", "explanation": "Not in ref", "source": "none"},
        ])
        results = checker.parse_results(raw)
        assert len(results) == 3
        assert results[0].verdict == "fabricated"
        assert results[1].verdict == "verified"
        assert results[2].verdict == "unverifiable"

    def test_unknown_verdict_normalised_to_unverifiable(self, checker):
        raw = json.dumps([
            {"claim": "c", "verdict": "nonsense_value", "explanation": "e", "source": "s"},
        ])
        results = checker.parse_results(raw)
        assert results[0].verdict == "unverifiable"

    def test_bad_json_with_fallback_claims(self, checker):
        fallback = [FlaggedClaim(text="fallback claim", reason="r")]
        results = checker.parse_results("not json", fallback_claims=fallback)
        assert len(results) == 1
        assert results[0].claim == "fallback claim"
        assert results[0].verdict == "unverifiable"

    def test_bad_json_without_fallback_returns_empty(self, checker):
        results = checker.parse_results("not json")
        assert results == []

    def test_markdown_fenced_json(self, checker):
        raw = '```json\n[{"claim": "c", "verdict": "verified", "explanation": "e", "source": "s"}]\n```'
        results = checker.parse_results(raw)
        assert len(results) == 1
        assert results[0].verdict == "verified"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: extract_claims (LLM call)
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractClaims:
    """extract_claims() calls the LLM and parses the response."""

    def test_extracts_claims_from_llm_response(self):
        llm_response = json.dumps([
            {"claim": "Dosio died in 2022", "reason": "death date", "search_query": "Dosio death"},
        ])
        checker = FactChecker(client=_make_client(llm_response))
        claims = checker.extract_claims("Ernie Dosio died in 2022 in a car crash.")
        assert len(claims) == 1
        assert claims[0].text == "Dosio died in 2022"

    def test_empty_text_calls_llm_returns_list(self):
        """Empty text: LLM returns [] — we get an empty list back."""
        checker = FactChecker(client=_make_client("[]"))
        claims = checker.extract_claims("   ")
        # Empty chunk is handled at check_chunk level, but extract_claims
        # still calls LLM on non-empty whitespace... wait, whitespace is
        # stripped in check_chunk, not here. Either way [] is valid.
        assert isinstance(claims, list)

    def test_llm_exception_returns_empty(self):
        """LLM raises → conservative empty list, no exception."""
        client = Mock()
        client.chat = Mock(side_effect=RuntimeError("LLM down"))
        checker = FactChecker(client=client)
        claims = checker.extract_claims("Some text with claims.")
        assert claims == []


# ═══════════════════════════════════════════════════════════════════════════
# TEST: verify_claims (quick mode)
# ═══════════════════════════════════════════════════════════════════════════

class TestVerifyClaims:
    """verify_claims() calls the LLM verifier and parses results."""

    def test_verify_fabricated_claim(self):
        """Claim contradicted by reference → fabricated."""
        claims = [FlaggedClaim(text="Dosio died in 2022", reason="death year")]
        response = json.dumps([
            {"claim": "Dosio died in 2022", "verdict": "fabricated",
             "explanation": "Reference says he died in 2026", "source": "reference_material"},
        ])
        checker = FactChecker(client=_make_client(response))
        results = checker.verify_claims(claims, reference_material="Dosio died April 24, 2026.")
        assert len(results) == 1
        assert results[0].verdict == "fabricated"
        assert "2026" in results[0].explanation

    def test_verify_with_no_reference_material(self):
        """No reference → all claims come back unverifiable (not fabricated)."""
        claims = [FlaggedClaim(text="Some claim", reason="r")]
        response = json.dumps([
            {"claim": "Some claim", "verdict": "unverifiable",
             "explanation": "No reference to check against", "source": "none"},
        ])
        checker = FactChecker(client=_make_client(response))
        results = checker.verify_claims(claims, reference_material=None)
        assert results[0].verdict == "unverifiable"

    def test_empty_claims_returns_empty(self):
        checker = FactChecker(client=Mock())
        assert checker.verify_claims([]) == []

    def test_llm_exception_returns_unverifiable_fallbacks(self):
        claims = [FlaggedClaim(text="A claim", reason="r")]
        client = Mock()
        client.chat = Mock(side_effect=RuntimeError("LLM down"))
        checker = FactChecker(client=client)
        results = checker.verify_claims(claims, reference_material="some ref")
        assert len(results) == 1
        assert results[0].verdict == "unverifiable"
        assert results[0].claim == "A claim"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: thorough mode with search
# ═══════════════════════════════════════════════════════════════════════════

class TestThoroughWithSearch:
    """_upgrade_with_search() searches unverifiable claims and upgrades verdicts."""

    def test_upgrades_unverifiable_to_fabricated(self):
        """Unverifiable claim → search finds contradiction → fabricated."""
        flagged = [FlaggedClaim(text="Dosio died in 2022", reason="r", search_query="Dosio death")]
        quick_results = [
            FactCheckResult(claim="Dosio died in 2022", verdict="unverifiable",
                            explanation="Not in reference", source="none"),
        ]
        # LLM call for search verifier returns "fabricated"
        search_verifier_response = json.dumps([
            {"claim": "Dosio died in 2022", "verdict": "fabricated",
             "explanation": "Wikipedia says April 24, 2026",
             "source": "web search: Dosio obituary"},
        ])
        checker = FactChecker(
            client=_make_client(search_verifier_response),
            search_provider=_MockSearchProvider(),
        )
        merged = checker._upgrade_with_search(quick_results, flagged)
        assert merged[0].verdict == "fabricated"

    def test_leaves_verified_untouched(self):
        """Verified results from quick mode should not be re-checked."""
        flagged = [FlaggedClaim(text="Founded 2003", reason="r", search_query="q")]
        quick_results = [
            FactCheckResult(claim="Founded 2003", verdict="verified",
                            explanation="Confirmed in ref", source="reference_material"),
        ]
        checker = FactChecker(client=Mock(), search_provider=_MockSearchProvider())
        merged = checker._upgrade_with_search(quick_results, flagged)
        # Verified result not touched; no LLM call needed.
        assert merged[0].verdict == "verified"
        checker._client.chat.assert_not_called()

    def test_search_failure_leaves_unverifiable(self):
        """Search raises → leave result as unverifiable, no crash."""
        flagged = [FlaggedClaim(text="Claim X", reason="r", search_query="q")]
        quick_results = [
            FactCheckResult(claim="Claim X", verdict="unverifiable",
                            explanation="Not in ref", source="none"),
        ]
        checker = FactChecker(
            client=Mock(),
            search_provider=_FailingSearchProvider(),
        )
        merged = checker._upgrade_with_search(quick_results, flagged)
        assert merged[0].verdict == "unverifiable"

    def test_no_search_query_leaves_unverifiable(self):
        """Claim without a search_query: don't attempt a search."""
        flagged = [FlaggedClaim(text="Claim X", reason="r", search_query=None)]
        quick_results = [
            FactCheckResult(claim="Claim X", verdict="unverifiable",
                            explanation="Not in ref", source="none"),
        ]
        checker = FactChecker(client=Mock(), search_provider=_MockSearchProvider())
        merged = checker._upgrade_with_search(quick_results, flagged)
        # No search query → no call to search provider
        assert merged[0].verdict == "unverifiable"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: thorough mode fallback (no search provider)
# ═══════════════════════════════════════════════════════════════════════════

class TestThoroughFallback:
    """mode='thorough' with search_provider=None falls back to quick."""

    def test_warns_and_continues(self, caplog):
        """Logs a warning and proceeds with quick mode (no crash)."""
        # Extractor returns one claim; verifier returns unverifiable.
        extractor_resp = json.dumps([
            {"claim": "Some claim", "reason": "r", "search_query": "q"},
        ])
        verifier_resp = json.dumps([
            {"claim": "Some claim", "verdict": "unverifiable",
             "explanation": "Not found", "source": "none"},
        ])
        checker = FactChecker(
            client=_make_client(extractor_resp, verifier_resp),
            search_provider=None,
        )
        import logging
        with caplog.at_level(logging.WARNING, logger="vox_romana.core.fact_checker"):
            report = checker.check_content(
                "Some claim text",
                content_id="seg_001",
                reference_material="Some ref",
                mode="thorough",
            )
        assert "no search_provider" in caplog.text.lower() or "falling back" in caplog.text.lower()
        assert isinstance(report, FactCheckReport)
        assert not report.has_fabrications

    def test_quick_result_returned(self):
        """Result is the quick-mode result (unverifiable, not fabricated)."""
        extractor_resp = json.dumps([
            {"claim": "Claim A", "reason": "r", "search_query": "q"},
        ])
        verifier_resp = json.dumps([
            {"claim": "Claim A", "verdict": "unverifiable",
             "explanation": "Not found", "source": "none"},
        ])
        checker = FactChecker(
            client=_make_client(extractor_resp, verifier_resp),
            search_provider=None,
        )
        report = checker.check_content(
            "Claim A is notable",
            content_id="seg_002",
            reference_material=None,
            mode="thorough",
        )
        assert len(report.results) == 1
        assert report.results[0].verdict == "unverifiable"
        assert not report.has_fabrications


# ═══════════════════════════════════════════════════════════════════════════
# TEST: check_content (full integration)
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckChunk:
    """check_content() end-to-end with mocked LLM calls."""

    def test_empty_chunk_returns_empty_report(self):
        checker = FactChecker(client=Mock())
        report = checker.check_content("   ", content_id="seg_000")
        assert not report.has_fabrications
        assert report.results == []
        assert "empty" in report.summary.lower()

    def test_no_claims_extracted_returns_clean_report(self):
        checker = FactChecker(client=_make_client("[]"))
        report = checker.check_content("Just opinions and vibes.", content_id="seg_001")
        assert not report.has_fabrications
        assert "no checkable" in report.summary.lower()

    def test_fabricated_claim_detected(self):
        extractor_resp = json.dumps([
            {"claim": "Dosio died in 2022", "reason": "r", "search_query": "q"},
        ])
        verifier_resp = json.dumps([
            {"claim": "Dosio died in 2022", "verdict": "fabricated",
             "explanation": "Reference says 2026", "source": "reference_material"},
        ])
        checker = FactChecker(client=_make_client(extractor_resp, verifier_resp))
        report = checker.check_content(
            "Ernie Dosio died in 2022.",
            content_id="seg_002",
            reference_material="Dosio died April 24, 2026.",
            mode="quick",
        )
        assert report.has_fabrications
        assert len(report.fabricated_claims) == 1
        assert "Dosio died in 2022" in report.fabricated_claims[0].claim
        assert "FABRICATED" in report.summary

    def test_verified_claim_passes(self):
        extractor_resp = json.dumps([
            {"claim": "Founded in 2003", "reason": "r", "search_query": None},
        ])
        verifier_resp = json.dumps([
            {"claim": "Founded in 2003", "verdict": "verified",
             "explanation": "Confirmed in ref", "source": "reference_material"},
        ])
        checker = FactChecker(client=_make_client(extractor_resp, verifier_resp))
        report = checker.check_content(
            "The company was founded in 2003.",
            content_id="seg_003",
            reference_material="Founded 2003.",
            mode="quick",
        )
        assert not report.has_fabrications
        assert len(report.results) == 1
        assert report.results[0].verdict == "verified"

    def test_content_id_in_report(self):
        checker = FactChecker(client=_make_client("[]"))
        report = checker.check_content("some text", content_id="unique_seg_id")
        assert report.content_id == "unique_seg_id"

    def test_mode_stored_in_report(self):
        checker = FactChecker(client=_make_client("[]"))
        report = checker.check_content("text", mode="thorough")
        assert report.mode == "thorough"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: FactCheckReport
# ═══════════════════════════════════════════════════════════════════════════

class TestFactCheckReport:
    """has_fabrications, fabricated_claims, to_dict/from_dict."""

    def test_has_fabrications_true(self):
        results = [
            FactCheckResult(claim="c1", verdict="fabricated", explanation="e", source="s"),
            FactCheckResult(claim="c2", verdict="verified", explanation="e", source="s"),
        ]
        report = FactCheckReport(content_id="x", mode="quick", results=results)
        assert report.has_fabrications is True

    def test_has_fabrications_false(self):
        results = [
            FactCheckResult(claim="c1", verdict="verified", explanation="e", source="s"),
            FactCheckResult(claim="c2", verdict="unverifiable", explanation="e", source="s"),
        ]
        report = FactCheckReport(content_id="x", mode="quick", results=results)
        assert report.has_fabrications is False

    def test_fabricated_claims_filters_correctly(self):
        results = [
            FactCheckResult(claim="fab", verdict="fabricated", explanation="e", source="s"),
            FactCheckResult(claim="ok", verdict="verified", explanation="e", source="s"),
        ]
        report = FactCheckReport(content_id="x", mode="quick", results=results)
        fab = report.fabricated_claims
        assert len(fab) == 1
        assert fab[0].claim == "fab"

    def test_to_dict_and_from_dict_roundtrip(self):
        results = [
            FactCheckResult(claim="c", verdict="fabricated", explanation="e", source="s"),
        ]
        report = FactCheckReport(
            content_id="seg_001", mode="quick", results=results, summary="Test summary"
        )
        d = report.to_dict()
        assert d["content_id"] == "seg_001"
        assert d["mode"] == "quick"
        assert d["has_fabrications"] is True

        r2 = FactCheckReport.from_dict(d)
        assert r2.content_id == "seg_001"
        assert r2.has_fabrications is True
        assert r2.results[0].claim == "c"

    def test_empty_results_no_fabrications(self):
        report = FactCheckReport(content_id="x", mode="quick", results=[])
        assert report.has_fabrications is False
        assert report.fabricated_claims == []


# ═══════════════════════════════════════════════════════════════════════════
# TEST: ContentEditor blocking hook
# ═══════════════════════════════════════════════════════════════════════════

class TestContentEditorBlocking:
    """ContentEditor injects FactChecker result into EditorVerdict."""

    def _make_editor(self, fact_checker=None, reference_material=None):
        """Build a ContentEditor with a mock session that always passes."""
        pass_verdict = json.dumps({
            "status": "pass",
            "summary": "Looks good.",
            "issues": [],
            "recommendations": [],
        })
        session = _make_editor_session(pass_verdict)
        return ContentEditor(
            session,
            fact_checker=fact_checker,
            reference_material=reference_material,
        )

    def test_no_fact_checker_passes_normally(self):
        editor = self._make_editor(fact_checker=None)
        snapshot = _make_snapshot()
        verdict = editor.review("Some clean text", snapshot, "seg_001")
        assert verdict.status == "pass"
        assert verdict.issues == []

    def test_fabricated_claim_flips_verdict_to_fail(self):
        """Editor passes LLM review but FactChecker finds fabrication → fail."""
        extractor_resp = json.dumps([
            {"claim": "Dosio died in 2022", "reason": "death year", "search_query": "q"},
        ])
        verifier_resp = json.dumps([
            {"claim": "Dosio died in 2022", "verdict": "fabricated",
             "explanation": "Reference says 2026", "source": "reference_material"},
        ])
        fact_checker = FactChecker(client=_make_client(extractor_resp, verifier_resp))
        editor = self._make_editor(
            fact_checker=fact_checker,
            reference_material="Ernie Dosio died April 24, 2026.",
        )
        snapshot = _make_snapshot()
        verdict = editor.review("Dosio died in 2022.", snapshot, "seg_001")

        assert verdict.status == "fail"
        assert any("FABRICATED" in issue for issue in verdict.issues)
        assert any("Dosio died in 2022" in issue for issue in verdict.issues)

    def test_fabricated_claim_adds_recommendation(self):
        """Recommendations include the correction instruction."""
        extractor_resp = json.dumps([
            {"claim": "Revenue was $304k", "reason": "financial figure", "search_query": "q"},
        ])
        verifier_resp = json.dumps([
            {"claim": "Revenue was $304k", "verdict": "fabricated",
             "explanation": "No revenue figure in reference material", "source": "none"},
        ])
        fact_checker = FactChecker(client=_make_client(extractor_resp, verifier_resp))
        editor = self._make_editor(fact_checker=fact_checker, reference_material="Some ref.")
        verdict = editor.review("Revenue was $304k.", _make_snapshot(), "seg_001")
        assert any("Revenue was $304k" in rec for rec in verdict.recommendations)

    def test_verified_claim_passes(self):
        """FactChecker verifies a claim → editor stays pass."""
        extractor_resp = json.dumps([
            {"claim": "Founded 2003", "reason": "founding year", "search_query": None},
        ])
        verifier_resp = json.dumps([
            {"claim": "Founded 2003", "verdict": "verified",
             "explanation": "Confirmed in reference", "source": "reference_material"},
        ])
        fact_checker = FactChecker(client=_make_client(extractor_resp, verifier_resp))
        editor = self._make_editor(fact_checker=fact_checker, reference_material="Founded 2003.")
        verdict = editor.review("Founded 2003.", _make_snapshot(), "seg_001")
        assert verdict.status == "pass"

    def test_fact_checker_exception_does_not_block(self):
        """FactChecker raises unexpectedly → editor still returns a verdict."""
        bad_checker = Mock(spec=FactChecker)
        bad_checker.check_content = Mock(side_effect=RuntimeError("Unexpected!"))
        editor = self._make_editor(fact_checker=bad_checker)
        verdict = editor.review("Some text.", _make_snapshot(), "seg_001")
        # Should not propagate exception — returns the LLM verdict.
        assert verdict.status in ("pass", "fail")


# ═══════════════════════════════════════════════════════════════════════════
# TEST: LLM failure safety
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMFailureSafety:
    """All parse errors and LLM exceptions → conservative fallbacks."""

    def test_extractor_bad_json_returns_no_fabrications(self):
        """Extractor returns garbage JSON → empty claims → no fabrications."""
        checker = FactChecker(client=_make_client("this is not json"))
        report = checker.check_content(
            "Some text with possible claims.",
            content_id="seg_001",
            reference_material="Some ref",
            mode="quick",
        )
        # Empty claims → report with no fabrications (not a block).
        assert not report.has_fabrications

    def test_verifier_bad_json_returns_unverifiable_not_fabricated(self):
        """Verifier returns garbage → fallback is unverifiable, never fabricated."""
        extractor_resp = json.dumps([
            {"claim": "Some claim", "reason": "r", "search_query": None},
        ])
        checker = FactChecker(client=_make_client(extractor_resp, "GARBAGE"))
        report = checker.check_content(
            "Some claim is real.",
            content_id="seg_001",
            reference_material="ref text",
            mode="quick",
        )
        assert not report.has_fabrications
        if report.results:
            assert all(r.verdict == "unverifiable" for r in report.results)

    def test_complete_llm_failure_returns_empty_report(self):
        """Both LLM calls fail → empty report, no crash, no fabrications."""
        client = Mock()
        client.chat = Mock(side_effect=RuntimeError("Both calls dead"))
        checker = FactChecker(client=client)
        report = checker.check_content(
            "Claim about something real.",
            content_id="seg_001",
            reference_material="ref",
            mode="quick",
        )
        assert isinstance(report, FactCheckReport)
        assert not report.has_fabrications


# ═══════════════════════════════════════════════════════════════════════════
# [v2] TEST: content_mode in check_content
# ═══════════════════════════════════════════════════════════════════════════

class TestContentModeCheck:
    """check_content() behaves correctly for each content_mode."""

    def test_pure_fiction_skips_all_llm_calls(self):
        """content_mode='pure-fiction' returns immediately with zero LLM calls."""
        client = Mock()
        checker = FactChecker(client=client)
        report = checker.check_content(
            "A dragon flew over the Crystal Mountains.",
            content_id="seg_001",
            mode="quick",
            content_mode="pure-fiction",
        )
        client.chat.assert_not_called()
        assert not report.has_fabrications
        assert "pure-fiction" in report.summary

    def test_pure_fiction_empty_results(self):
        """pure-fiction report has empty results list."""
        checker = FactChecker(client=Mock())
        report = checker.check_content(
            "Zara the elf cast a spell.",
            content_id="seg_x",
            content_mode="pure-fiction",
        )
        assert report.results == []
        assert not report.has_fabrications

    def test_fictional_universe_uses_canon_verifier_prompt(self):
        """content_mode='fictional-universe' sends CANON_VERIFIER_PROMPT to LLM."""
        from vox_romana.core.fact_checker import CANON_VERIFIER_PROMPT, CLAIM_VERIFIER_PROMPT

        extractor_resp = json.dumps([
            {"claim": "Luke is from Kansas", "reason": "birthplace", "search_query": None},
        ])
        verifier_resp = json.dumps([
            {"claim": "Luke is from Kansas", "verdict": "fabricated",
             "explanation": "Luke is from Tatooine per Star Wars canon", "source": "reference_material"},
        ])
        client = _make_client(extractor_resp, verifier_resp)
        checker = FactChecker(client=client)
        report = checker.check_content(
            "Luke Skywalker grew up on a farm in Kansas.",
            content_id="seg_sw",
            reference_material="Luke Skywalker is from Tatooine.",
            mode="quick",
            content_mode="fictional-universe",
        )
        # The second call (verifier) must use CANON_VERIFIER_PROMPT, not CLAIM_VERIFIER_PROMPT.
        verifier_call = client.chat.call_args_list[1]
        messages = verifier_call[1]["messages"] if "messages" in verifier_call[1] else verifier_call[0][0]
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert "canon" in system_content.lower()
        assert "Tatooine" in system_content or "fictional" in system_content.lower()

    def test_fictional_universe_fabrication_detected(self):
        """Canon violation flagged as fabricated in fictional-universe mode."""
        extractor_resp = json.dumps([
            {"claim": "Luke is from Kansas", "reason": "birthplace", "search_query": None},
        ])
        verifier_resp = json.dumps([
            {"claim": "Luke is from Kansas", "verdict": "fabricated",
             "explanation": "Canon says Tatooine", "source": "reference_material"},
        ])
        checker = FactChecker(client=_make_client(extractor_resp, verifier_resp))
        report = checker.check_content(
            "Luke grew up in Kansas.",
            content_id="seg_sw",
            reference_material="Luke is from Tatooine.",
            content_mode="fictional-universe",
        )
        assert report.has_fabrications
        assert "Luke is from Kansas" in report.fabricated_claims[0].claim

    def test_factual_mode_uses_claim_verifier_prompt(self):
        """content_mode='factual' (default) uses CLAIM_VERIFIER_PROMPT."""
        from vox_romana.core.fact_checker import CLAIM_VERIFIER_PROMPT

        extractor_resp = json.dumps([
            {"claim": "Dosio died in 2022", "reason": "death year", "search_query": None},
        ])
        verifier_resp = json.dumps([
            {"claim": "Dosio died in 2022", "verdict": "fabricated",
             "explanation": "Ref says 2026", "source": "reference_material"},
        ])
        client = _make_client(extractor_resp, verifier_resp)
        checker = FactChecker(client=client)
        checker.check_content(
            "Dosio died in 2022.",
            content_id="seg_001",
            reference_material="Dosio died 2026.",
            content_mode="factual",
        )
        verifier_call = client.chat.call_args_list[1]
        messages = verifier_call[1]["messages"] if "messages" in verifier_call[1] else verifier_call[0][0]
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        # CLAIM_VERIFIER_PROMPT is about real-world verification, not canon
        assert "canon" not in system_content.lower()


# ═══════════════════════════════════════════════════════════════════════════
# [v2] TEST: prior_context consistency pass
# ═══════════════════════════════════════════════════════════════════════════

class TestPriorContextConsistency:
    """_check_consistency and _merge_consistency_results."""

    def test_consistency_pass_called_when_prior_context_provided(self):
        """When prior_context is non-empty, a consistency LLM call is made."""
        extractor_resp = json.dumps([
            {"claim": "The trial began in 2017", "reason": "timeline", "search_query": None},
        ])
        verifier_resp = json.dumps([
            {"claim": "The trial began in 2017", "verdict": "unverifiable",
             "explanation": "Not in ref", "source": "none"},
        ])
        consistency_resp = json.dumps([
            {"claim": "The trial began in 2017", "verdict": "contradicts_prior",
             "explanation": "Earlier segment said trial started in 2019", "source": "prior_context"},
        ])
        client = _make_client(extractor_resp, verifier_resp, consistency_resp)
        checker = FactChecker(client=client)
        report = checker.check_content(
            "The trial began in 2017.",
            content_id="seg_002",
            mode="quick",
            prior_context="In segment 1, we reported the trial started in 2019.",
        )
        assert client.chat.call_count == 3  # extractor + verifier + consistency
        assert report.has_contradictions
        assert len(report.contradicted_claims) == 1

    def test_no_prior_context_skips_consistency_pass(self):
        """When prior_context is None, only 2 LLM calls are made (extract + verify)."""
        extractor_resp = json.dumps([
            {"claim": "Revenue was $1M", "reason": "financial", "search_query": None},
        ])
        verifier_resp = json.dumps([
            {"claim": "Revenue was $1M", "verdict": "unverifiable",
             "explanation": "Not in ref", "source": "none"},
        ])
        client = _make_client(extractor_resp, verifier_resp)
        checker = FactChecker(client=client)
        report = checker.check_content(
            "Revenue was $1M.",
            content_id="seg_001",
            mode="quick",
            prior_context=None,
        )
        assert client.chat.call_count == 2
        assert not report.has_contradictions

    def test_empty_prior_context_skips_consistency_pass(self):
        """Whitespace-only prior_context is treated as absent."""
        extractor_resp = json.dumps([
            {"claim": "Dosio died in 2026", "reason": "death", "search_query": None},
        ])
        verifier_resp = json.dumps([
            {"claim": "Dosio died in 2026", "verdict": "verified",
             "explanation": "Matches ref", "source": "reference_material"},
        ])
        client = _make_client(extractor_resp, verifier_resp)
        checker = FactChecker(client=client)
        checker.check_content(
            "Dosio died in 2026.",
            content_id="seg_001",
            prior_context="   ",
        )
        assert client.chat.call_count == 2

    def test_merge_does_not_overwrite_fabricated_with_contradicts_prior(self):
        """A 'fabricated' verdict from the primary pass is NOT downgraded."""
        primary = [
            FactCheckResult(claim="X was 32", verdict="fabricated",
                            explanation="Ref says X was 40", source="reference_material"),
        ]
        consistency = [
            FactCheckResult(claim="X was 32", verdict="contradicts_prior",
                            explanation="Earlier said X was 40", source="prior_context"),
        ]
        merged = FactChecker._merge_consistency_results(primary, consistency)
        assert merged[0].verdict == "fabricated"  # not downgraded

    def test_merge_replaces_unverifiable_with_contradicts_prior(self):
        """'unverifiable' primary entry IS replaced by contradicts_prior."""
        primary = [
            FactCheckResult(claim="Trial started in 2017", verdict="unverifiable",
                            explanation="Not in ref", source="none"),
        ]
        consistency = [
            FactCheckResult(claim="Trial started in 2017", verdict="contradicts_prior",
                            explanation="Earlier said 2019", source="prior_context"),
        ]
        merged = FactChecker._merge_consistency_results(primary, consistency)
        assert merged[0].verdict == "contradicts_prior"

    def test_merge_consistency_verified_dropped(self):
        """'verified' verdicts from the consistency pass are dropped."""
        primary = [
            FactCheckResult(claim="X died in 2026", verdict="verified",
                            explanation="In ref", source="reference_material"),
        ]
        consistency = [
            FactCheckResult(claim="X died in 2026", verdict="verified",
                            explanation="Consistent with prior", source="prior_context"),
        ]
        merged = FactChecker._merge_consistency_results(primary, consistency)
        assert len(merged) == 1
        assert merged[0].verdict == "verified"

    def test_consistency_llm_failure_does_not_block(self):
        """If the consistency LLM call fails, the primary results still pass."""
        extractor_resp = json.dumps([
            {"claim": "X was 32", "reason": "age", "search_query": None},
        ])
        verifier_resp = json.dumps([
            {"claim": "X was 32", "verdict": "unverifiable",
             "explanation": "Not in ref", "source": "none"},
        ])
        client = Mock()
        # extractor → ok, verifier → ok, consistency → raises
        client.chat = Mock(side_effect=[
            extractor_resp,
            verifier_resp,
            RuntimeError("LLM down"),
        ])
        checker = FactChecker(client=client)
        report = checker.check_content(
            "X was 32.",
            content_id="seg_001",
            prior_context="Earlier we said X was 40.",
        )
        # Consistency failure → no contradictions, no crash.
        assert not report.has_contradictions
        assert not report.has_fabrications


# ═══════════════════════════════════════════════════════════════════════════
# [v2] TEST: _trim_prior_context
# ═══════════════════════════════════════════════════════════════════════════

class TestTrimPriorContext:
    """_trim_prior_context() keeps most-recent content and adds a marker."""

    def test_short_text_returned_unchanged(self):
        text = "Hello world."
        assert _trim_prior_context(text, max_chars=100) == text

    def test_exact_length_not_trimmed(self):
        text = "a" * 100
        assert _trim_prior_context(text, max_chars=100) == text

    def test_long_text_trimmed_to_max_chars(self):
        text = "a" * 5000
        result = _trim_prior_context(text, max_chars=4000)
        assert len(result) <= 4000

    def test_trim_keeps_most_recent_content(self):
        """After trimming, the TAIL (most recent) of the text is preserved."""
        # Use the default 4000-char limit. The text is 10000 A's + a unique suffix.
        # _trim_prior_context keeps the TAIL, so the suffix must survive.
        text = "A" * 10000 + "TAIL_SUFFIX"
        result = _trim_prior_context(text)
        assert "TAIL_SUFFIX" in result
        assert len(result) <= 4000

    def test_trim_adds_marker(self):
        """Trimmed result is prefixed with the '…earlier content trimmed…' marker."""
        text = "x" * 5000
        result = _trim_prior_context(text, max_chars=4000)
        assert result.startswith("[")
        assert "trimmed" in result.lower()

    def test_default_max_is_4000(self):
        """Default max_chars is 4000."""
        text = "a" * 5000
        result = _trim_prior_context(text)
        assert len(result) <= 4000


# ═══════════════════════════════════════════════════════════════════════════
# [v2] TEST: build_default classmethod
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildDefault:
    """FactChecker.build_default() wires up build_fact_check_provider()."""

    def test_returns_fact_checker_instance(self):
        client = Mock()
        with patch("vox_romana.core.fact_checker.build_fact_check_provider", return_value=None):
            checker = FactChecker.build_default(client)
        assert isinstance(checker, FactChecker)
        assert checker._client is client

    def test_uses_google_when_available(self):
        """build_default uses the provider returned by build_fact_check_provider."""
        client = Mock()
        mock_provider = Mock()
        with patch("vox_romana.core.fact_checker.build_fact_check_provider", return_value=mock_provider):
            checker = FactChecker.build_default(client)
        assert checker._search is mock_provider

    def test_falls_back_to_none_when_no_credentials(self):
        """When build_fact_check_provider returns None, search is None (quick-only)."""
        client = Mock()
        with patch("vox_romana.core.fact_checker.build_fact_check_provider", return_value=None):
            checker = FactChecker.build_default(client)
        assert checker._search is None


# ═══════════════════════════════════════════════════════════════════════════
# [v2] TEST: has_contradictions + contradicted_claims on FactCheckReport
# ═══════════════════════════════════════════════════════════════════════════

class TestContradictionReport:
    """FactCheckReport.has_contradictions and .contradicted_claims."""

    def test_has_contradictions_false_when_no_contradicts_prior(self):
        results = [
            FactCheckResult(claim="c1", verdict="verified", explanation="e", source="s"),
            FactCheckResult(claim="c2", verdict="unverifiable", explanation="e", source="s"),
            FactCheckResult(claim="c3", verdict="fabricated", explanation="e", source="s"),
        ]
        report = FactCheckReport(content_id="x", mode="quick", results=results)
        assert report.has_contradictions is False

    def test_has_contradictions_true_when_contradicts_prior_present(self):
        results = [
            FactCheckResult(claim="c1", verdict="contradicts_prior",
                            explanation="earlier said otherwise", source="prior_context"),
        ]
        report = FactCheckReport(content_id="x", mode="quick", results=results)
        assert report.has_contradictions is True

    def test_contradicted_claims_filters_correctly(self):
        results = [
            FactCheckResult(claim="ok", verdict="verified", explanation="e", source="s"),
            FactCheckResult(claim="bad", verdict="contradicts_prior",
                            explanation="prior conflict", source="prior_context"),
        ]
        report = FactCheckReport(content_id="x", mode="quick", results=results)
        contradicted = report.contradicted_claims
        assert len(contradicted) == 1
        assert contradicted[0].claim == "bad"

    def test_to_dict_includes_has_contradictions(self):
        results = [
            FactCheckResult(claim="bad", verdict="contradicts_prior",
                            explanation="conflict", source="prior_context"),
        ]
        report = FactCheckReport(content_id="x", mode="quick", results=results)
        d = report.to_dict()
        assert "has_contradictions" in d
        assert d["has_contradictions"] is True

    def test_fabricated_and_contradicted_can_coexist(self):
        results = [
            FactCheckResult(claim="fab", verdict="fabricated", explanation="e", source="ref"),
            FactCheckResult(claim="contra", verdict="contradicts_prior",
                            explanation="conflict", source="prior_context"),
        ]
        report = FactCheckReport(content_id="x", mode="quick", results=results)
        assert report.has_fabrications is True
        assert report.has_contradictions is True
        assert len(report.fabricated_claims) == 1
        assert len(report.contradicted_claims) == 1

    def test_parse_results_accepts_contradicts_prior_verdict(self):
        """parse_results normalises 'contradicts_prior' as a valid verdict."""
        raw = json.dumps([
            {"claim": "trial 2017", "verdict": "contradicts_prior",
             "explanation": "earlier said 2019", "source": "prior_context"},
        ])
        checker = FactChecker(client=Mock())
        results = checker.parse_results(raw)
        assert len(results) == 1
        assert results[0].verdict == "contradicts_prior"


# ═══════════════════════════════════════════════════════════════════════════
# [v2] TEST: LLM knowledge fallback (no reference_material + factual mode)
# ═══════════════════════════════════════════════════════════════════════════

class TestKnowledgeVerifier:
    """When reference_material is absent and content_mode='factual', verify_claims()
    uses KNOWLEDGE_VERIFIER_PROMPT so the LLM applies its training knowledge."""

    def test_no_ref_material_uses_knowledge_prompt(self):
        """System message must be KNOWLEDGE_VERIFIER_PROMPT when no ref supplied."""
        from vox_romana.core.fact_checker import KNOWLEDGE_VERIFIER_PROMPT
        claims = [FlaggedClaim(text="Dosio died on April 17 2026", reason="death date")]
        response = json.dumps([
            {"claim": "Dosio died on April 17 2026", "verdict": "verified",
             "explanation": "Consistent with training knowledge", "source": "llm_knowledge"},
        ])
        client = _make_client(response)
        checker = FactChecker(client=client)
        checker.verify_claims(claims, reference_material=None, content_mode="factual")
        call_args = client.chat.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert system_content == KNOWLEDGE_VERIFIER_PROMPT

    def test_empty_string_ref_also_uses_knowledge_prompt(self):
        """Whitespace-only reference_material also triggers the knowledge path."""
        from vox_romana.core.fact_checker import KNOWLEDGE_VERIFIER_PROMPT
        claims = [FlaggedClaim(text="Some claim", reason="r")]
        response = json.dumps([
            {"claim": "Some claim", "verdict": "unverifiable",
             "explanation": "Uncertain", "source": "llm_knowledge"},
        ])
        client = _make_client(response)
        checker = FactChecker(client=client)
        checker.verify_claims(claims, reference_material="   ", content_mode="factual")
        call_args = client.chat.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert system_content == KNOWLEDGE_VERIFIER_PROMPT

    def test_knowledge_verifier_can_return_fabricated(self):
        """LLM knowledge path can flag a claim as fabricated → has_fabrications True."""
        extractor_resp = json.dumps([
            {"claim": "Dosio died in 2022", "reason": "death year", "search_query": "Dosio death"},
        ])
        verifier_resp = json.dumps([
            {"claim": "Dosio died in 2022", "verdict": "fabricated",
             "explanation": "Training data shows he died in April 2026", "source": "llm_knowledge"},
        ])
        checker = FactChecker(client=_make_client(extractor_resp, verifier_resp))
        report = checker.check_content(
            "Ernie Dosio passed away in 2022.",
            content_id="seg_001",
            reference_material=None,
            mode="quick",
        )
        assert report.has_fabrications
        assert report.fabricated_claims[0].source == "llm_knowledge"

    def test_knowledge_verifier_can_return_verified(self):
        """LLM correctly verifies a known fact → no fabrication."""
        extractor_resp = json.dumps([
            {"claim": "Pacific AgriLands is in Galt CA", "reason": "location", "search_query": None},
        ])
        verifier_resp = json.dumps([
            {"claim": "Pacific AgriLands is in Galt CA", "verdict": "verified",
             "explanation": "Consistent with business records", "source": "llm_knowledge"},
        ])
        checker = FactChecker(client=_make_client(extractor_resp, verifier_resp))
        report = checker.check_content(
            "Pacific AgriLands is in Galt, CA.",
            content_id="seg_001",
            reference_material=None,
        )
        assert not report.has_fabrications
        assert report.results[0].verdict == "verified"

    def test_knowledge_verifier_conservative_unverifiable(self):
        """When the LLM is uncertain, it returns unverifiable → no block."""
        extractor_resp = json.dumps([
            {"claim": "Revenue was $304,954", "reason": "revenue figure", "search_query": None},
        ])
        verifier_resp = json.dumps([
            {"claim": "Revenue was $304,954", "verdict": "unverifiable",
             "explanation": "Cannot verify private company revenue from training data",
             "source": "llm_knowledge"},
        ])
        checker = FactChecker(client=_make_client(extractor_resp, verifier_resp))
        report = checker.check_content(
            "Revenue was $304,954.",
            content_id="seg_001",
            reference_material=None,
        )
        assert not report.has_fabrications
        assert report.results[0].verdict == "unverifiable"

    def test_fictional_universe_no_ref_skips_knowledge_path(self):
        """fictional-universe with no ref uses CANON_VERIFIER_PROMPT, not KNOWLEDGE."""
        from vox_romana.core.fact_checker import CANON_VERIFIER_PROMPT, KNOWLEDGE_VERIFIER_PROMPT
        claims = [FlaggedClaim(text="Luke is from Kansas", reason="birthplace")]
        response = json.dumps([
            {"claim": "Luke is from Kansas", "verdict": "unverifiable",
             "explanation": "No reference provided", "source": "none"},
        ])
        client = _make_client(response)
        checker = FactChecker(client=client)
        checker.verify_claims(claims, reference_material=None, content_mode="fictional-universe")
        call_args = client.chat.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert system_content == CANON_VERIFIER_PROMPT
        assert system_content != KNOWLEDGE_VERIFIER_PROMPT

    def test_knowledge_verifier_exception_returns_unverifiable(self):
        """LLM raises on the knowledge verifier call → conservative unverifiable, no crash."""
        claims = [FlaggedClaim(text="Some claim", reason="r")]
        client = Mock()
        client.chat = Mock(side_effect=RuntimeError("LLM down"))
        checker = FactChecker(client=client)
        results = checker.verify_claims(claims, reference_material=None, content_mode="factual")
        assert len(results) == 1
        assert results[0].verdict == "unverifiable"
        assert results[0].claim == "Some claim"

    def test_reference_material_provided_does_not_use_knowledge_prompt(self):
        """When reference_material IS provided, CLAIM_VERIFIER_PROMPT is used (not knowledge)."""
        from vox_romana.core.fact_checker import CLAIM_VERIFIER_PROMPT, KNOWLEDGE_VERIFIER_PROMPT
        claims = [FlaggedClaim(text="Dosio died April 17 2026", reason="death date")]
        response = json.dumps([
            {"claim": "Dosio died April 17 2026", "verdict": "verified",
             "explanation": "Confirmed in ref", "source": "reference_material"},
        ])
        client = _make_client(response)
        checker = FactChecker(client=client)
        checker.verify_claims(
            claims,
            reference_material="Ernie Dosio died April 17 2026 in Gabon.",
            content_mode="factual",
        )
        call_args = client.chat.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert system_content == CLAIM_VERIFIER_PROMPT
        assert system_content != KNOWLEDGE_VERIFIER_PROMPT

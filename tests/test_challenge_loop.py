"""Tests for the fact-check challenge loop (Chunk 5 / Phase 3).

Covers:
  - FactChecker.adjudicate_justification — LLM-driven re-evaluation of a
    "fabricated" verdict given a justification from the generating agent.
  - BaseConductor._challenge_fabrications — orchestrates the round-trip
    between fact-checker and a registered generator session.

Run with:
    cd /Users/bagheera/repos/vox_romana
    python -m pytest tests/test_challenge_loop.py -q
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from vox_romana.core.conductor import BaseConductor, ConductorResult
from vox_romana.core.config import ContentConfig
from vox_romana.core.editing.models import FactCheckReport, FactCheckResult
from vox_romana.core.fact_checker import FactChecker


class _StubConductor(BaseConductor):
    def _build_agents(self) -> None:  # pragma: no cover - not used
        pass

    def _run_turn_loop(self) -> list[dict]:  # pragma: no cover
        return []

    def _package_result(self, raw_events):  # pragma: no cover
        return ConductorResult(config=self.config, events=raw_events)


@pytest.fixture
def conductor():
    return _StubConductor(ContentConfig(content_type="news"), client=Mock())


# ─────────────────────────────────────────────────────────────────────────────
# FactChecker.adjudicate_justification
# ─────────────────────────────────────────────────────────────────────────────


class TestAdjudicateJustification:
    def _checker(self, llm_response: str) -> FactChecker:
        client = Mock()
        client.chat.return_value = llm_response
        return FactChecker(client=client)

    def test_downgrades_to_verified_when_adjudicator_accepts(self):
        checker = self._checker(json.dumps({
            "verdict": "verified",
            "explanation": "Perspective alignment confirmed.",
        }))
        original = FactCheckResult(
            claim="the northern border",
            verdict="fabricated",
            explanation="Guatemala has no northern border.",
            source="reference_material",
        )

        new = checker.adjudicate_justification(
            original,
            "From Guatemala, Mexico is to the north — that is the northern border.",
            perspective="from Guatemala",
        )

        assert new.verdict == "verified"
        assert new.source == "challenge_loop"
        assert "Perspective" in new.explanation

    def test_downgrades_to_unverifiable(self):
        checker = self._checker(json.dumps({
            "verdict": "unverifiable",
            "explanation": "Justification raises doubt but no source.",
        }))
        original = FactCheckResult(
            claim="X happened in 1923",
            verdict="fabricated",
            explanation="Reference says 1932.",
        )

        new = checker.adjudicate_justification(original, "I think it was 1923.")

        assert new.verdict == "unverifiable"
        assert new.source == "challenge_loop"

    def test_keeps_fabricated_when_adjudicator_rejects(self):
        checker = self._checker(json.dumps({
            "verdict": "fabricated",
            "explanation": "Justification was vague.",
        }))
        original = FactCheckResult(
            claim="X", verdict="fabricated", explanation="wrong",
        )

        new = checker.adjudicate_justification(original, "creative licence")

        assert new.verdict == "fabricated"
        assert new.source == "challenge_loop"

    def test_non_fabricated_results_pass_through_unchanged(self):
        checker = self._checker("{}")
        client = checker._client
        original = FactCheckResult(
            claim="X", verdict="verified", explanation="ok", source="reference_material",
        )

        new = checker.adjudicate_justification(original, "anything")

        assert new is original
        client.chat.assert_not_called()

    def test_empty_justification_skips_llm_call(self):
        checker = self._checker("{}")
        client = checker._client
        original = FactCheckResult(
            claim="X", verdict="fabricated", explanation="wrong",
        )

        new = checker.adjudicate_justification(original, "   ")

        assert new is original
        client.chat.assert_not_called()

    def test_unparseable_llm_response_keeps_original_verdict(self):
        checker = self._checker("not json at all")
        original = FactCheckResult(
            claim="X", verdict="fabricated", explanation="wrong",
        )

        new = checker.adjudicate_justification(original, "justification")

        assert new is original

    def test_invalid_verdict_keeps_original(self):
        checker = self._checker(json.dumps({"verdict": "bogus"}))
        original = FactCheckResult(
            claim="X", verdict="fabricated", explanation="wrong",
        )

        new = checker.adjudicate_justification(original, "justification")

        assert new is original

    def test_perspective_is_injected_into_user_message(self):
        checker = self._checker(json.dumps({"verdict": "verified", "explanation": "ok"}))
        original = FactCheckResult(
            claim="our northern border", verdict="fabricated", explanation="wrong",
        )

        checker.adjudicate_justification(
            original, "Mexico is north of Guatemala.", perspective="from Guatemala",
        )

        user_msg = checker._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "from Guatemala" in user_msg
        assert "PERSPECTIVE" in user_msg


# ─────────────────────────────────────────────────────────────────────────────
# BaseConductor._challenge_fabrications
# ─────────────────────────────────────────────────────────────────────────────


class TestChallengeFabrications:
    def _make_report(self, *results: FactCheckResult) -> FactCheckReport:
        return FactCheckReport(
            content_id="seg1",
            mode="quick",
            results=list(results),
            summary="",
        )

    def _fabricated(self, claim="X") -> FactCheckResult:
        return FactCheckResult(
            claim=claim, verdict="fabricated",
            explanation="bad", source="reference_material",
        )

    def test_no_fabrications_returns_report_unchanged(self, conductor):
        report = self._make_report(FactCheckResult(
            claim="A", verdict="verified", explanation="", source="ref",
        ))

        out = conductor._challenge_fabrications(
            report, generator_agent="anchor", fact_checker=Mock(),
        )

        assert out is report
        assert out.results[0].verdict == "verified"

    def test_unregistered_generator_is_skipped(self, conductor):
        report = self._make_report(self._fabricated())
        checker = Mock()

        out = conductor._challenge_fabrications(
            report, generator_agent="ghost", fact_checker=checker,
        )

        assert out.results[0].verdict == "fabricated"
        checker.adjudicate_justification.assert_not_called()

    def test_no_fact_checker_is_skipped(self, conductor):
        report = self._make_report(self._fabricated())
        # Register a session so registry check passes.
        sess = Mock()
        conductor._register_session("anchor", sess)

        out = conductor._challenge_fabrications(
            report, generator_agent="anchor", fact_checker=None,
        )

        assert out.results[0].verdict == "fabricated"
        sess.send.assert_not_called()

    def test_challenge_round_trip_downgrades_verdict(self, conductor):
        sess = Mock()
        sess.send.return_value = "Mexico is to the north of Guatemala."
        conductor._register_session("anchor", sess)

        checker = Mock()
        checker.adjudicate_justification.return_value = FactCheckResult(
            claim="X", verdict="verified",
            explanation="confirmed", source="challenge_loop",
        )

        report = self._make_report(self._fabricated())
        out = conductor._challenge_fabrications(
            report, generator_agent="anchor", fact_checker=checker,
            perspective="from Guatemala",
        )

        sess.send.assert_called_once()
        sent = sess.send.call_args.args[0]
        assert "fact_checker" in sent
        assert "from Guatemala" in sent
        checker.adjudicate_justification.assert_called_once()
        adj_kwargs = checker.adjudicate_justification.call_args.kwargs
        assert adj_kwargs["perspective"] == "from Guatemala"
        assert out.results[0].verdict == "verified"
        assert out.results[0].source == "challenge_loop"

    def test_only_fabricated_claims_are_challenged(self, conductor):
        sess = Mock()
        sess.send.return_value = "justification"
        conductor._register_session("anchor", sess)

        checker = Mock()
        checker.adjudicate_justification.return_value = FactCheckResult(
            claim="bad", verdict="verified", explanation="ok",
            source="challenge_loop",
        )

        verified = FactCheckResult(claim="A", verdict="verified", explanation="")
        fabricated = self._fabricated(claim="bad")
        unverifiable = FactCheckResult(claim="C", verdict="unverifiable", explanation="")

        report = self._make_report(verified, fabricated, unverifiable)
        out = conductor._challenge_fabrications(
            report, generator_agent="anchor", fact_checker=checker,
        )

        assert sess.send.call_count == 1
        assert out.results[0].verdict == "verified"  # untouched
        assert out.results[1].verdict == "verified"  # downgraded
        assert out.results[2].verdict == "unverifiable"  # untouched

    def test_empty_justification_leaves_verdict_unchanged(self, conductor):
        sess = Mock()
        sess.send.return_value = "   "
        conductor._register_session("anchor", sess)

        checker = Mock()
        report = self._make_report(self._fabricated())

        out = conductor._challenge_fabrications(
            report, generator_agent="anchor", fact_checker=checker,
        )

        checker.adjudicate_justification.assert_not_called()
        assert out.results[0].verdict == "fabricated"

    def test_session_send_exception_is_swallowed(self, conductor):
        sess = Mock()
        sess.send.side_effect = RuntimeError("LLM failure")
        conductor._register_session("anchor", sess)

        checker = Mock()
        report = self._make_report(self._fabricated())

        out = conductor._challenge_fabrications(
            report, generator_agent="anchor", fact_checker=checker,
        )

        assert out.results[0].verdict == "fabricated"
        checker.adjudicate_justification.assert_not_called()

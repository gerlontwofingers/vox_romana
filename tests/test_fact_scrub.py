"""Tests for the last-resort fact-check rewrite (core.fact_scrub).

When a conductor exhausts its fact-check retries, the content is rewritten
to drop the unresolved claims instead of being published with a warning.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

from vox_romana.core.config import ContentConfig
from vox_romana.core.editing.models import FactCheckResult
from vox_romana.core.fact_scrub import SCRUB_SYSTEM_PROMPT, scrub_unresolved_claims


def _claim(text="The bridge opened in 1932.", why="Reference says 1936."):
    return FactCheckResult(
        claim=text, verdict="fabricated", explanation=why, source="reference_material",
    )


def _client(reply):
    client = Mock()
    client.chat = Mock(return_value=reply)
    return client


class TestScrubUnresolvedClaims:
    def test_returns_rewritten_passages_in_order(self):
        client = _client(json.dumps(["First, cleaned.", "Second, cleaned."]))
        result = scrub_unresolved_claims(client, ["First.", "Second."], [_claim()])
        assert result == ["First, cleaned.", "Second, cleaned."]

    def test_accepts_markdown_fenced_json(self):
        client = _client('```json\n["Cleaned."]\n```')
        assert scrub_unresolved_claims(client, ["Original."], [_claim()]) == ["Cleaned."]

    def test_prompt_lists_claims_reasons_and_passages(self):
        client = _client(json.dumps(["x"]))
        scrub_unresolved_claims(client, ["Original text."], [_claim()])
        messages = client.chat.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": SCRUB_SYSTEM_PROMPT}
        user = messages[1]["content"]
        assert "The bridge opened in 1932." in user
        assert "Reference says 1936." in user
        assert "Original text." in user

    def test_prompt_includes_prior_context_and_perspective_when_given(self):
        client = _client(json.dumps(["x"]))
        scrub_unresolved_claims(
            client, ["Original."], [_claim()],
            prior_context="Earlier we discussed tolls.",
            perspective="a 1930s newspaper reporter",
        )
        user = client.chat.call_args.kwargs["messages"][1]["content"]
        assert "Earlier we discussed tolls." in user
        assert "a 1930s newspaper reporter" in user

    def test_prior_context_is_trimmed_to_the_tail(self):
        client = _client(json.dumps(["x"]))
        scrub_unresolved_claims(
            client, ["Original."], [_claim()],
            prior_context="OLDEST " + ("y" * 5000) + " NEWEST",
        )
        user = client.chat.call_args.kwargs["messages"][1]["content"]
        assert "NEWEST" in user
        assert "OLDEST" not in user

    def test_system_prompt_requires_continuity_and_no_new_facts(self):
        assert "Do NOT replace removed claims with new facts" in SCRUB_SYSTEM_PROMPT
        assert "flow" in SCRUB_SYSTEM_PROMPT
        assert "dangling" in SCRUB_SYSTEM_PROMPT

    def test_no_client_returns_none(self):
        assert scrub_unresolved_claims(None, ["x"], [_claim()]) is None

    def test_no_claims_returns_none_without_calling_llm(self):
        client = _client("[]")
        assert scrub_unresolved_claims(client, ["x"], []) is None
        client.chat.assert_not_called()

    def test_all_empty_passages_returns_none_without_calling_llm(self):
        client = _client("[]")
        assert scrub_unresolved_claims(client, ["", "  "], [_claim()]) is None
        client.chat.assert_not_called()

    def test_wrong_length_returns_none(self):
        client = _client(json.dumps(["only one"]))
        assert scrub_unresolved_claims(client, ["a", "b"], [_claim()]) is None

    def test_non_list_returns_none(self):
        client = _client(json.dumps({"text": "nope"}))
        assert scrub_unresolved_claims(client, ["a"], [_claim()]) is None

    def test_non_string_items_return_none(self):
        client = _client(json.dumps([1]))
        assert scrub_unresolved_claims(client, ["a"], [_claim()]) is None

    def test_unparseable_reply_returns_none(self):
        client = _client("sorry, I can't do that")
        assert scrub_unresolved_claims(client, ["a"], [_claim()]) is None

    def test_llm_exception_returns_none(self):
        client = Mock()
        client.chat = Mock(side_effect=RuntimeError("down"))
        assert scrub_unresolved_claims(client, ["a"], [_claim()]) is None

    def test_emptying_a_non_empty_passage_is_rejected(self):
        client = _client(json.dumps(["", "kept"]))
        assert scrub_unresolved_claims(client, ["a", "b"], [_claim()]) is None

    def test_empty_passages_stay_empty(self):
        client = _client(json.dumps(["Cleaned.", ""]))
        assert scrub_unresolved_claims(client, ["Original.", ""], [_claim()]) == [
            "Cleaned.", "",
        ]


class TestDefaultFactCheckRetries:
    def test_default_is_ten(self):
        assert ContentConfig().max_retries == 10


class TestConductorScrubMethod:
    def _conductor(self, client):
        from vox_romana.core.conductor import BaseConductor, ConductorResult

        class _Stub(BaseConductor):
            def _build_agents(self):
                pass

            def _run_turn_loop(self):
                return []

            def _package_result(self, raw):
                return ConductorResult(config=self.config)

        return _Stub(ContentConfig(content_type="podcast"), client=client)

    def test_success_returns_rewrite_and_records_scrub_warning(self):
        conductor = self._conductor(_client(json.dumps(["Cleaned."])))
        result = conductor._scrub_unresolved_claims("seg_1", ["Original."], [_claim()])
        assert result == ["Cleaned."]
        assert any(
            "seg_1" in w and "FACT_CHECK_SCRUBBED" in w and "1932" in w
            for w in conductor._length_warnings
        )

    def test_failure_returns_none_and_records_nothing(self):
        conductor = self._conductor(_client("not json"))
        assert conductor._scrub_unresolved_claims("seg_1", ["Original."], [_claim()]) is None
        assert conductor._length_warnings == []

    def test_uses_subclass_prior_context_when_available(self):
        client = _client(json.dumps(["Cleaned."]))
        conductor = self._conductor(client)
        conductor._build_prior_context = lambda: "Approved earlier segment."
        conductor._scrub_unresolved_claims("seg_2", ["Original."], [_claim()])
        user = client.chat.call_args.kwargs["messages"][1]["content"]
        assert "Approved earlier segment." in user

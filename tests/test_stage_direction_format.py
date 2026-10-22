"""Tests for B-7: canonical stage-direction format validation.

Covers:
    1. core.stage_directions.find_parenthetical_stage_directions() — the
       shared detection helper also used by the TTS renderer's stripping.
    2. BaseConductor._capture_issue_flag() — non-blocking length-warning
       emitted when an event's "text" violates the canonical
       [square-brackets]-only convention for inline scene actions.
"""

from __future__ import annotations

from vox_romana.core.conductor import BaseConductor, ConductorResult
from vox_romana.core.config import ContentConfig
from vox_romana.core.stage_directions import find_parenthetical_stage_directions


# ─────────────────────────────────────────────────────────────────────────────
# 1. find_parenthetical_stage_directions()
# ─────────────────────────────────────────────────────────────────────────────

class TestFindParentheticalStageDirections:
    def test_detects_simple_stage_direction(self):
        result = find_parenthetical_stage_directions("(laughs) Well, that's funny.")
        assert result == ["laughs"]

    def test_detects_multiple(self):
        text = "(sighs) I hear you. (leans back) Let's move on."
        assert find_parenthetical_stage_directions(text) == ["sighs", "leans back"]

    def test_ignores_acronym_expansions(self):
        text = "Universal Basic Income (UBI) proposals"
        assert find_parenthetical_stage_directions(text) == []

    def test_ignores_numeric_asides(self):
        text = "Revenue fell (down 12%) this quarter."
        assert find_parenthetical_stage_directions(text) == []

    def test_ignores_long_prose_parentheticals(self):
        text = (
            "This matters (and I say this because it directly affects "
            "how we should think about the policy going forward) a lot."
        )
        assert find_parenthetical_stage_directions(text) == []

    def test_empty_text_returns_empty_list(self):
        assert find_parenthetical_stage_directions("") == []
        assert find_parenthetical_stage_directions(None) == []  # type: ignore[arg-type]

    def test_bracket_only_text_has_no_violations(self):
        text = "[Dr. Waffleford holds the pizza aloft.] That's dinner."
        assert find_parenthetical_stage_directions(text) == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. BaseConductor._capture_issue_flag() stage-direction check
# ─────────────────────────────────────────────────────────────────────────────

class _StubConfig(ContentConfig):
    content_type: str = "podcast"


class _StubConductor(BaseConductor):
    def _build_agents(self):
        pass

    def _run_turn_loop(self):
        return []

    def _package_result(self, raw_events):
        return ConductorResult(config=self.config, events=raw_events)


def _make_conductor() -> _StubConductor:
    return _StubConductor(_StubConfig(content_type="podcast"))


class TestCaptureIssueFlagStageDirectionCheck:
    def test_parenthetical_scene_action_logs_warning(self):
        cond = _make_conductor()
        payload = {"text": "(leans forward) I disagree completely."}
        cond._capture_issue_flag(payload, from_agent="Alice")
        assert any("stage-direction format" in w for w in cond._length_warnings)
        assert any("Alice" in w for w in cond._length_warnings)

    def test_bracket_scene_action_no_warning(self):
        cond = _make_conductor()
        payload = {"text": "[Alice leans forward.] I disagree completely."}
        cond._capture_issue_flag(payload, from_agent="Alice")
        assert cond._length_warnings == []

    def test_clean_dialogue_no_warning(self):
        cond = _make_conductor()
        payload = {"text": "I disagree completely."}
        cond._capture_issue_flag(payload, from_agent="Alice")
        assert cond._length_warnings == []

    def test_acronym_parenthetical_no_warning(self):
        cond = _make_conductor()
        payload = {"text": "Universal Basic Income (UBI) is the topic today."}
        cond._capture_issue_flag(payload, from_agent="Alice")
        assert cond._length_warnings == []

    def test_payload_without_text_field_no_crash(self):
        cond = _make_conductor()
        payload = {"foo": "bar"}
        cond._capture_issue_flag(payload, from_agent="Alice")
        assert cond._length_warnings == []

    def test_issue_flag_still_captured_alongside_check(self):
        """The new validator must not interfere with existing issue_flag
        capture behaviour."""
        cond = _make_conductor()
        payload = {
            "text": "(sighs) Fine, you win this round.",
            "issue_flag": "reference material seems thin here",
        }
        result = cond._capture_issue_flag(payload, from_agent="Alice")
        assert "issue_flag" not in result
        # Both the stage-direction warning and the issue report should fire.
        assert any("stage-direction format" in w for w in cond._length_warnings)
        assert len(cond._production_log) >= 1

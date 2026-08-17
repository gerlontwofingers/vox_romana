"""Tests for typed inter-agent message routing.

Run with:
    cd /Users/bagheera/repos/vox_romana
    python -m pytest tests/test_messages.py -q
"""

from unittest.mock import Mock

import pytest

from vox_romana.core.config import ContentConfig
from vox_romana.core.conductor import BaseConductor, ConductorResult
from vox_romana.core.editing.models import EditorVerdict
from vox_romana.core.messages import AgentMessage, MessageType


class _StubConfig(ContentConfig):
    content_type: str = "debate"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _StubConductor(BaseConductor):
    def _build_agents(self) -> None:
        pass

    def _run_turn_loop(self) -> list[dict]:
        return []

    def _package_result(self, raw_events: list[dict]) -> ConductorResult:
        return ConductorResult(config=self.config, events=raw_events)


@pytest.fixture
def conductor():
    return _StubConductor(_StubConfig(content_type="debate"), client=Mock())


class TestAgentMessage:
    def test_round_trip_preserves_type(self):
        message = AgentMessage(
            type=MessageType.DIRECTIVE,
            from_agent="editor",
            to_agent="director",
            payload={"issue": "repetition"},
        )

        restored = AgentMessage.from_dict(message.to_dict())

        assert restored.type == MessageType.DIRECTIVE
        assert restored.payload == {"issue": "repetition"}
        assert restored.message_id == message.message_id

    def test_format_for_context_includes_headers(self):
        message = AgentMessage(
            type=MessageType.REJECTION,
            from_agent="editor",
            to_agent="conductor",
            payload={"summary": "Too repetitive."},
        )

        rendered = message.format_for_context()

        assert "MESSAGE TYPE: rejection" in rendered
        assert "FROM: editor" in rendered
        assert "TO: conductor" in rendered
        assert '"summary": "Too repetitive."' in rendered


class TestAgentRouter:
    def test_route_message_injects_context(self, conductor):
        session = Mock()
        conductor._register_session("director", session)

        conductor._route_message(
            AgentMessage(
                type=MessageType.DIRECTIVE,
                from_agent="editor",
                to_agent="director",
                payload={"summary": "Tighten this section."},
            )
        )

        session.inject_context.assert_called_once()
        rendered = session.inject_context.call_args.args[0]
        assert "MESSAGE TYPE: directive" in rendered
        assert '"summary": "Tighten this section."' in rendered
        assert conductor._production_log[-1]["channel"] == "direct"
        assert conductor._production_log[-1]["recipients"] == ["director"]

    def test_broadcast_injects_all_registered_sessions(self, conductor):
        director = Mock()
        narrator = Mock()
        conductor._register_session("director", director)
        conductor._register_session("narrator", narrator)

        conductor._broadcast(
            AgentMessage(
                type=MessageType.BROADCAST,
                from_agent="director",
                payload={"outline_version": 2},
            )
        )

        director.inject_context.assert_called_once()
        narrator.inject_context.assert_called_once()
        assert conductor._production_log[-1]["channel"] == "broadcast"
        assert conductor._production_log[-1]["recipients"] == ["director", "narrator"]

    def test_challenge_loop_returns_justification_message(self, conductor):
        narrator = Mock()
        narrator.send.return_value = "The phrasing reflects Guatemala's perspective."
        conductor._register_session("narrator", narrator)

        reply = conductor._challenge_loop(
            from_agent="fact_checker",
            to_agent="narrator",
            payload={"claim": "Advertencias de seguridad en la frontera norte"},
        )

        narrator.send.assert_called_once()
        assert reply.type == MessageType.JUSTIFICATION
        assert reply.from_agent == "narrator"
        assert reply.to_agent == "fact_checker"
        assert reply.reply_to
        assert reply.payload == {
            "response": "The phrasing reflects Guatemala's perspective."
        }
        assert conductor._production_log[-2]["channel"] == "challenge"
        assert conductor._production_log[-1]["channel"] == "justification"

    def test_routing_challenge_without_loop_raises(self, conductor):
        with pytest.raises(ValueError, match="_challenge_loop"):
            conductor._route_message(
                AgentMessage(
                    type=MessageType.CHALLENGE,
                    from_agent="fact_checker",
                    to_agent="narrator",
                    payload={"claim": "example"},
                )
            )


class TestBaseConductorRegistration:
    def test_register_session_rejects_conflicting_duplicate_name(self, conductor):
        conductor._register_session("director", Mock())

        with pytest.raises(ValueError, match="already registered"):
            conductor._register_session("director", Mock())

    def test_create_session_can_register_named_session(self, conductor):
        session = conductor._create_session(
            "You are a narrator.",
            session_name="narrator",
        )

        assert conductor._session_registry["narrator"] is session

    def test_broadcast_director_note_reaches_registered_sessions(self, conductor):
        narrator = Mock()
        editor = Mock()
        conductor._register_session("narrator", narrator)
        conductor._register_session("editor", editor)

        conductor._broadcast_director_note(
            "Trim the ending.",
            summary="Director updated outline guidance.",
        )

        narrator.inject_context.assert_called_once()
        editor.inject_context.assert_called_once()
        rendered = narrator.inject_context.call_args.args[0]
        assert "MESSAGE TYPE: broadcast" in rendered
        assert '"director_note": "Trim the ending."' in rendered

    def test_route_editor_rejection_sends_directive_to_target_agents(self, conductor):
        narrator = Mock()
        conductor._register_session("narrator", narrator)
        verdict = EditorVerdict(
            status="fail",
            content_id="seg1",
            outline_version=2,
            summary="Too repetitive.",
            issues=["repetition"],
            recommendations=["cut repeated setup"],
        )

        conductor._route_editor_rejection(
            verdict,
            chunk_id="seg1",
            target_agents=["narrator"],
        )

        narrator.inject_context.assert_called_once()
        rendered = narrator.inject_context.call_args.args[0]
        assert "MESSAGE TYPE: directive" in rendered
        assert '"chunk_id": "seg1"' in rendered
        assert '"summary": "Too repetitive."' in rendered

    def test_issue_reports_queue_and_consume(self, conductor):
        conductor._report_issue_to_director(
            from_agent="participant:Alice",
            summary="Cannot support the current beat with verified facts.",
            proposed_remediation="Move to the next subtopic.",
        )

        rendered = conductor._consume_pending_issue_reports()

        assert "MESSAGE TYPE: issue_report" in rendered
        assert "FROM: participant:Alice" in rendered
        assert '"proposed_remediation": "Move to the next subtopic."' in rendered
        assert conductor._production_log[-1]["event_type"] == "issue_report"
        assert conductor._consume_pending_issue_reports() == ""

    def test_run_broadcasts_initial_outline_and_attaches_log(self):
        class _BroadcastingConductor(_StubConductor):
            def __init__(self):
                config = _StubConfig(content_type="debate")
                config.title = "Table Read"
                config.topic = "Should X happen?"
                config.tone = "measured"
                config.target_turns = 4
                super().__init__(
                    config,
                    client=Mock(),
                )
                self.session = Mock()

            def _build_agents(self) -> None:
                self._register_session("participant:Alice", self.session)

        conductor = _BroadcastingConductor()

        result = conductor.run()

        conductor.session.inject_context.assert_called_once()
        rendered = conductor.session.inject_context.call_args.args[0]
        assert "MESSAGE TYPE: broadcast" in rendered
        assert "INITIAL OUTLINE CONTEXT" in rendered
        assert "Table Read" in rendered
        assert result.production_log
        assert result.production_log[0]["message"]["payload"]["phase"] == "table_read"
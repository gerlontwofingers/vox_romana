"""Tests for the RetryLoop — chunk generation with editor gating.

Tests:
    1. Pass on first attempt (no retries needed)
    2. Fail then pass on retry
    3. All retries exhausted — warning returned
    4. Feedback string propagated to generate_fn
    5. Best result selection
    6. Max retries configurable

Run with:
    cd /Users/bagheera/repos/vox_romana
    python -m pytest tests/test_retry.py -v
"""

import json
from unittest.mock import Mock

import pytest

from vox_romana.core.editing.editor import EDITOR_SYSTEM_PROMPT, ChunkEditor
from vox_romana.core.editing.models import OutlineSection
from vox_romana.core.editing.outline import DirectorOutline
from vox_romana.core.editing.retry import RetryLoop
from vox_romana.core.session import AgentSession


@pytest.fixture
def mock_client():
    return Mock()


@pytest.fixture
def sample_outline():
    return DirectorOutline(
        initial_sections=[
            OutlineSection(
                id="seg1",
                title="Opening Segment",
                expected_duration="5 min",
                key_beats=["hook", "thesis"],
                speaker_assignments=["Host"],
            ),
        ],
        outline_id="test-outline",
    )


def _make_editor(mock_client, responses):
    """Helper: create a ChunkEditor whose session returns canned responses."""
    mock_client.chat.side_effect = list(responses)
    session = AgentSession(
        client=mock_client,
        system_prompt=EDITOR_SYSTEM_PROMPT,
        recent_window=20,
    )
    return ChunkEditor(session)


def _pass_json(summary="Good."):
    return json.dumps({
        "status": "pass",
        "summary": summary,
        "issues": [],
        "recommendations": [],
    })


def _fail_json(issues=None, recs=None, summary="Needs work."):
    return json.dumps({
        "status": "fail",
        "summary": summary,
        "issues": issues or ["bland"],
        "recommendations": recs or ["make it better"],
    })


class TestPassOnFirstAttempt:
    def test_single_pass(self, mock_client, sample_outline):
        editor = _make_editor(mock_client, [_pass_json()])
        loop = RetryLoop(editor, max_retries=3)

        gen_fn = Mock(return_value=("Chunk text here", [{"speaker": "Host", "text": "Hello"}]))

        result = loop.execute(gen_fn, "seg1", sample_outline)

        assert result.chunk_id == "seg1"
        assert result.attempt_number == 1
        assert result.editor_verdict.status == "pass"
        assert result.warning is None
        gen_fn.assert_called_once_with("")  # Empty feedback on first call

    def test_content_preserved(self, mock_client, sample_outline):
        events = [{"speaker": "Host", "text": "Welcome everyone!"}]
        editor = _make_editor(mock_client, [_pass_json()])
        loop = RetryLoop(editor, max_retries=3)

        gen_fn = Mock(return_value=("Welcome everyone!", events))
        result = loop.execute(gen_fn, "seg1", sample_outline)

        assert result.content == events


class TestRetryThenPass:
    def test_fail_then_pass(self, mock_client, sample_outline):
        editor = _make_editor(mock_client, [
            _fail_json(issues=["repetition"]),
            _pass_json("Fixed."),
        ])
        loop = RetryLoop(editor, max_retries=3)

        call_count = 0
        def gen_fn(feedback):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("Bad chunk", [{"text": "bad"}])
            return ("Good chunk", [{"text": "good"}])

        result = loop.execute(gen_fn, "seg1", sample_outline)

        assert result.attempt_number == 2
        assert result.editor_verdict.status == "pass"
        assert result.warning is None

    def test_feedback_contains_issues(self, mock_client, sample_outline):
        editor = _make_editor(mock_client, [
            _fail_json(issues=["bland", "repetitive"], recs=["add metaphor", "vary phrasing"]),
            _pass_json(),
        ])
        loop = RetryLoop(editor, max_retries=3)

        received_feedback = []
        call_count = 0
        def gen_fn(feedback):
            nonlocal call_count
            call_count += 1
            received_feedback.append(feedback)
            return (f"Chunk v{call_count}", [{"text": f"v{call_count}"}])

        loop.execute(gen_fn, "seg1", sample_outline)

        assert received_feedback[0] == ""  # First call: no feedback
        assert "bland" in received_feedback[1]
        assert "repetitive" in received_feedback[1]
        assert "add metaphor" in received_feedback[1]


class TestRetriesExhausted:
    def test_warning_on_exhaustion(self, mock_client, sample_outline):
        editor = _make_editor(mock_client, [
            _fail_json(issues=["flat"]),
            _fail_json(issues=["still flat"]),
        ])
        loop = RetryLoop(editor, max_retries=2)

        gen_fn = Mock(return_value=("Mediocre chunk", [{"text": "meh"}]))
        result = loop.execute(gen_fn, "seg1", sample_outline)

        assert result.warning is not None
        assert "Max retries (2) exhausted" in result.warning
        assert gen_fn.call_count == 2

    def test_best_attempt_returned(self, mock_client, sample_outline):
        """When all attempts fail, the last attempt is returned."""
        editor = _make_editor(mock_client, [
            _fail_json(issues=["bad"]),
            _fail_json(issues=["less bad"]),
            _fail_json(issues=["still not great"]),
        ])
        loop = RetryLoop(editor, max_retries=3)

        call_count = 0
        def gen_fn(feedback):
            nonlocal call_count
            call_count += 1
            return (f"Attempt {call_count}", [{"text": f"attempt_{call_count}"}])

        result = loop.execute(gen_fn, "seg1", sample_outline)

        assert result.warning is not None
        assert result.content is not None

    def test_unresolved_issues_in_warning(self, mock_client, sample_outline):
        editor = _make_editor(mock_client, [
            _fail_json(issues=["clarity", "pacing"]),
        ])
        loop = RetryLoop(editor, max_retries=1)

        gen_fn = Mock(return_value=("Chunk", [{}]))
        result = loop.execute(gen_fn, "seg1", sample_outline)

        assert "clarity" in result.warning
        assert "pacing" in result.warning


class TestMaxRetriesConfig:
    def test_default_max_retries(self, mock_client):
        session = AgentSession(client=mock_client, system_prompt="test")
        editor = ChunkEditor(session)
        loop = RetryLoop(editor)
        assert loop.max_retries == 3

    def test_custom_max_retries(self, mock_client):
        session = AgentSession(client=mock_client, system_prompt="test")
        editor = ChunkEditor(session)
        loop = RetryLoop(editor, max_retries=5)
        assert loop.max_retries == 5

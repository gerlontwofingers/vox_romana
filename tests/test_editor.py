"""Tests for the ContentEditor — non-interactive quality gating agent.

Tests:
    1. Review parses clean JSON verdict (pass)
    2. Review parses clean JSON verdict (fail with issues)
    3. Handles markdown-wrapped JSON
    4. Handles garbage response (falls back to pass)
    5. Issues and recommendations preserved
    6. Session accumulates context across reviews

Run with:
    cd /Users/bagheera/repos/vox_romana
    python -m pytest tests/test_editor.py -v
"""

import json
from unittest.mock import Mock

import pytest

from vox_romana.core.editing.editor import EDITOR_SYSTEM_PROMPT, ContentEditor
from vox_romana.core.editing.models import OutlineSection, OutlineSnapshot
from vox_romana.core.session import AgentSession


@pytest.fixture
def mock_client():
    return Mock()


@pytest.fixture
def editor_session(mock_client):
    return AgentSession(
        client=mock_client,
        system_prompt=EDITOR_SYSTEM_PROMPT,
        recent_window=10,
    )


@pytest.fixture
def editor(editor_session):
    return ContentEditor(editor_session)


@pytest.fixture
def sample_snapshot():
    return OutlineSnapshot(
        outline_id="test",
        version=1,
        timestamp="2026-04-18T00:00:00Z",
        summary="Test outline",
        sections=[
            OutlineSection(
                id="intro",
                title="Opening",
                expected_duration="5 min",
                key_beats=["hook"],
                speaker_assignments=["Alex"],
            ),
        ],
    )


class TestPassVerdict:
    def test_clean_json_pass(self, editor, mock_client, sample_snapshot):
        mock_client.chat.return_value = json.dumps({
            "status": "pass",
            "summary": "Chunk meets all quality standards.",
            "issues": [],
            "recommendations": [],
        })
        verdict = editor.review("Alex: Great opening!", sample_snapshot, content_id="intro")
        assert verdict.status == "pass"
        assert verdict.content_id == "intro"
        assert verdict.outline_version == 1
        assert "quality standards" in verdict.summary
        assert verdict.issues == []


class TestFailVerdict:
    def test_clean_json_fail(self, editor, mock_client, sample_snapshot):
        mock_client.chat.return_value = json.dumps({
            "status": "fail",
            "summary": "Speaker identity is unclear in lines 3-5.",
            "issues": ["speaker_clarity", "repetition"],
            "recommendations": [
                "Add speaker tags to lines 3-5",
                "Remove repeated thesis from line 4",
            ],
        })
        verdict = editor.review(
            "Someone said something...",
            sample_snapshot,
            content_id="main",
        )
        assert verdict.status == "fail"
        assert "speaker_clarity" in verdict.issues
        assert len(verdict.recommendations) == 2


class TestJsonParsing:
    def test_markdown_wrapped(self, editor, mock_client, sample_snapshot):
        mock_client.chat.return_value = (
            "```json\n"
            '{"status": "pass", "summary": "Good.", "issues": [], "recommendations": []}\n'
            "```"
        )
        verdict = editor.review("chunk text", sample_snapshot, content_id="c1")
        assert verdict.status == "pass"

    def test_preamble_text(self, editor, mock_client, sample_snapshot):
        mock_client.chat.return_value = (
            'Here is my review:\n\n'
            '{"status": "fail", "summary": "Bad.", "issues": ["bland"], "recommendations": ["spice it up"]}'
        )
        verdict = editor.review("chunk text", sample_snapshot, content_id="c2")
        assert verdict.status == "fail"
        assert verdict.issues == ["bland"]

    def test_garbage_fallback(self, editor, mock_client, sample_snapshot):
        mock_client.chat.return_value = "I am confused and cannot review this"
        verdict = editor.review("chunk text", sample_snapshot, content_id="c3")
        # Garbage → conservative pass with warning
        assert verdict.status == "pass"
        assert "editor_parse_failure" in verdict.issues

    def test_default_status_is_pass(self, editor, mock_client, sample_snapshot):
        """JSON without status field should default to pass."""
        mock_client.chat.return_value = json.dumps({
            "summary": "Looks fine.",
            "issues": [],
            "recommendations": [],
        })
        verdict = editor.review("chunk text", sample_snapshot, content_id="c4")
        assert verdict.status == "pass"


class TestSessionAccumulation:
    def test_session_grows_across_reviews(self, editor, mock_client, sample_snapshot):
        mock_client.chat.return_value = json.dumps({
            "status": "pass",
            "summary": "Good.",
            "issues": [],
            "recommendations": [],
        })
        editor.review("Chunk A", sample_snapshot, content_id="c1")
        editor.review("Chunk B", sample_snapshot, content_id="c2")
        # Session should have accumulated messages (2 per review: user + assistant)
        assert editor.session.message_count >= 4

    def test_review_prompt_includes_outline(self, editor, mock_client, sample_snapshot):
        mock_client.chat.return_value = json.dumps({
            "status": "pass", "summary": "OK", "issues": [], "recommendations": [],
        })
        editor.review("Test chunk", sample_snapshot, content_id="c1")
        # Check that the prompt sent to LLM includes outline info
        call_args = mock_client.chat.call_args[0][0]
        user_messages = [m["content"] for m in call_args if m["role"] == "user"]
        assert any("OUTLINE CONTEXT" in msg for msg in user_messages)

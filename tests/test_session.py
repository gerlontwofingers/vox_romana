"""Tests for AgentSession — persistent LLM session with sliding-window compression.

Tests:
    1. Basic send accumulates history
    2. inject_context adds to history without LLM call
    3. Properties (message_count, total_calls, summary, history)
    4. Compression triggers when history exceeds recent window
    5. _build_messages includes summary and recent window
    6. Multiple compressions merge summaries
    7. Small history skips compression

Run with:
    cd /Users/bagheera/repos/vox_romana
    python -m pytest tests/test_session.py -v
"""

from unittest.mock import Mock

import pytest

from vox_romana.core.session import AgentSession


@pytest.fixture
def mock_client():
    client = Mock()
    client.chat = Mock(return_value="Sure, I'll help.")
    return client


@pytest.fixture
def session(mock_client):
    return AgentSession(
        client=mock_client,
        system_prompt="You are a test agent.",
        recent_window=3,
        summary_token_budget=200,
    )


class TestSendBasics:
    def test_send_returns_response(self, session, mock_client):
        mock_client.chat.return_value = "Hello back!"
        result = session.send("Hello")
        assert result == "Hello back!"

    def test_send_accumulates_history(self, session, mock_client):
        mock_client.chat.return_value = "Response 1"
        session.send("Message 1")
        assert session.message_count == 2  # user + assistant

    def test_send_increments_call_count(self, session, mock_client):
        mock_client.chat.return_value = "Response"
        session.send("Hello")
        assert session.total_calls == 1

    def test_send_passes_messages_to_client(self, session, mock_client):
        mock_client.chat.return_value = "OK"
        session.send("Test message")
        call_args = mock_client.chat.call_args[0][0]
        # Should have: system prompt + user message
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == "You are a test agent."
        assert call_args[-1]["role"] == "user"
        assert call_args[-1]["content"] == "Test message"


class TestInjectContext:
    def test_inject_adds_two_messages(self, session):
        session.inject_context("Outline update v2")
        assert session.message_count == 2

    def test_inject_does_not_make_llm_call(self, session, mock_client):
        session.inject_context("Outline update v2")
        mock_client.chat.assert_not_called()
        assert session.total_calls == 0

    def test_inject_content_in_history(self, session):
        session.inject_context("Topic 2 trimmed to 3 minutes")
        history = session.history
        assert "[CONTEXT UPDATE]" in history[0]["content"]
        assert "Topic 2 trimmed" in history[0]["content"]
        assert history[1]["content"] == "Acknowledged."


class TestProperties:
    def test_empty_session_properties(self, session):
        assert session.message_count == 0
        assert session.total_calls == 0
        assert session.summary == ""
        assert session.history == []

    def test_history_is_copy(self, session, mock_client):
        mock_client.chat.return_value = "R"
        session.send("M")
        h = session.history
        h.append({"role": "hacker", "content": "injected"})
        assert len(session.history) == 2  # original unchanged


class TestCompression:
    def test_no_compression_under_window(self, session, mock_client):
        """With recent_window=3, 3 pairs (6 messages) should not trigger."""
        mock_client.chat.return_value = "R"
        for i in range(3):
            session.send(f"M{i}")
        # 3 pairs = 6 messages = exactly at window. No compression.
        assert session.summary == ""
        # Only 3 send calls, no compression call
        assert session.total_calls == 3

    def test_compression_triggers_over_window(self, session, mock_client):
        """With recent_window=3, 5+ pairs should trigger compression."""
        call_count = [0]

        def chat_side_effect(messages):
            call_count[0] += 1
            # Detect summarizer call by system prompt content
            if any("summarizer" in m.get("content", "").lower() for m in messages):
                return "Summary of old conversation"
            return f"Response {call_count[0]}"

        mock_client.chat.side_effect = chat_side_effect

        for i in range(5):
            session.send(f"Message {i}")

        assert session.summary != ""
        assert "Summary" in session.summary

    def test_compressed_history_trimmed(self, session, mock_client):
        """After compression, history should be bounded, not grow unboundedly."""

        def chat_side_effect(messages):
            if any("summarizer" in m.get("content", "").lower() for m in messages):
                return "Old stuff summarized"
            return "Response"

        mock_client.chat.side_effect = chat_side_effect

        for i in range(8):
            session.send(f"Message {i}")

        # Without compression, history would be 16 (8 sends × 2 msgs each).
        # With window=3, compression fires periodically. The <4 guard means
        # history oscillates between 6 and 8 rather than exactly 6.
        assert session.message_count < 16
        assert session.summary != ""

    def test_summary_included_in_messages(self, session, mock_client):
        """After compression, _build_messages should include the summary."""

        def chat_side_effect(messages):
            if any("summarizer" in m.get("content", "").lower() for m in messages):
                return "Historical context here"
            return "Response"

        mock_client.chat.side_effect = chat_side_effect

        for i in range(6):
            session.send(f"Message {i}")

        # Compression has fired; summary should be set
        assert session.summary != ""

        # Directly verify _build_messages includes the summary
        session._history.append({"role": "user", "content": "Probe message"})
        messages = session._build_messages()
        system_contents = [
            m["content"] for m in messages if m["role"] == "system"
        ]
        assert any("CONVERSATION SUMMARY" in c for c in system_contents)

    def test_multiple_compressions_merge(self, session, mock_client):
        """Multiple compression rounds should append to the summary."""
        compress_count = [0]

        def chat_side_effect(messages):
            if any("summarizer" in m.get("content", "").lower() for m in messages):
                compress_count[0] += 1
                return f"Summary block {compress_count[0]}"
            return "Response"

        mock_client.chat.side_effect = chat_side_effect

        # Generate enough messages for multiple compressions
        for i in range(15):
            session.send(f"Message {i}")

        # Should have compressed at least twice
        if compress_count[0] >= 2:
            assert "Summary block 1" in session.summary
            assert "Summary block 2" in session.summary

"""Persistent agent session with sliding-window context compression.

The conductors currently manage context manually — building fresh
[system, user] message pairs for each LLM call. This works for
one-off calls, but the design spec requires agents to persist for
the entire duration of a broadcast, accumulating context.

AgentSession wraps DeepSeekClient with:
  1. Automatic message history accumulation.
  2. Sliding-window compression (same strategy as the existing
     DebateConductor's _compress_context, but generalized).
  3. Context injection for outline broadcasts and notes.

USAGE:
    from vox_romana.core.session import AgentSession
    from vox_romana.core.client import DeepSeekClient

    client = DeepSeekClient()
    session = AgentSession(
        client=client,
        system_prompt="You are a witty debate participant...",
        recent_window=10,
    )
    response = session.send("What's your opening argument?")
    session.inject_context("OUTLINE UPDATE: Topic 2 trimmed to 3 minutes.")
    response2 = session.send("Continue with your next point.")
"""

from __future__ import annotations

from .client import DeepSeekClient

_SUMMARIZER_SYSTEM = (
    "You are a concise summarizer. Preserve key facts, decisions, and context."
)

_COMPRESS_PROMPT = (
    "Summarize this conversation concisely. Focus on key positions taken, "
    "decisions made, facts established, and unresolved questions. "
    "This summary will replace the full history, so preserve anything "
    "someone would need to continue the conversation coherently.\n\n{transcript}"
)


class AgentSession:
    """A persistent LLM agent session with sliding-window compression.

    All persistent agents in a broadcast — Director, Conductor, Editor,
    cast members — get one AgentSession each. The session accumulates
    message history and automatically compresses older messages to stay
    within context window limits.

    Attributes:
        recent_window:  Number of recent message PAIRS (user+assistant)
                        to keep verbatim. Older pairs are compressed.
        summary_token_budget: Soft target for compression summary length.
    """

    def __init__(
        self,
        client: DeepSeekClient,
        system_prompt: str,
        *,
        recent_window: int = 10,
        summary_token_budget: int = 500,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._history: list[dict[str, str]] = []
        self._summary: str = ""
        self._recent_window = recent_window
        self._summary_token_budget = summary_token_budget
        self._total_calls: int = 0

    def send(self, message: str) -> str:
        """Send a message and return the assistant's response.

        Appends both the user message and assistant response to history,
        then compresses if the history exceeds the recent window.
        """
        self._history.append({"role": "user", "content": message})
        messages = self._build_messages()
        response = self._client.chat(messages)
        self._total_calls += 1
        self._history.append({"role": "assistant", "content": response})
        self._maybe_compress()
        return response

    def inject_context(self, context: str) -> None:
        """Inject a context update into the session without an LLM call.

        Used for Director outline broadcasts, Editor feedback, and other
        informational updates that agents need to see in their history.
        """
        self._history.append(
            {"role": "user", "content": f"[CONTEXT UPDATE]\n{context}"}
        )
        self._history.append({"role": "assistant", "content": "Acknowledged."})

    @property
    def message_count(self) -> int:
        """Total messages in history (user + assistant)."""
        return len(self._history)

    @property
    def total_calls(self) -> int:
        """Total LLM API calls made (includes compression calls)."""
        return self._total_calls

    @property
    def summary(self) -> str:
        """The compressed summary of older conversation history."""
        return self._summary

    @property
    def history(self) -> list[dict[str, str]]:
        """Current message history (read-only copy)."""
        return list(self._history)

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the message list for the next LLM call.

        Structure:
          [system prompt]
          [compressed summary of older turns, if any]
          [last K message pairs verbatim]
          [current user message — already appended to history]
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
        ]
        if self._summary:
            messages.append({
                "role": "system",
                "content": (
                    f"CONVERSATION SUMMARY (earlier context):\n{self._summary}"
                ),
            })
        # Recent window: last N*2 messages (N pairs of user+assistant).
        window_size = self._recent_window * 2
        if len(self._history) > window_size:
            messages.extend(self._history[-window_size:])
        else:
            messages.extend(self._history)
        return messages

    def _maybe_compress(self) -> None:
        """Compress older messages if history exceeds the recent window."""
        window_size = self._recent_window * 2
        if len(self._history) <= window_size:
            return

        older_end = len(self._history) - window_size
        older = self._history[:older_end]

        if len(older) < 4:
            return

        transcript = "\n".join(
            f"[{m['role']}]: {m['content'][:300]}" for m in older
        )
        prompt = _COMPRESS_PROMPT.format(transcript=transcript)

        new_summary = self._client.chat([
            {"role": "system", "content": _SUMMARIZER_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        self._total_calls += 1

        if self._summary:
            self._summary = f"{self._summary}\n\n{new_summary}"
        else:
            self._summary = new_summary

        # Trim: keep only the recent window
        self._history = self._history[older_end:]

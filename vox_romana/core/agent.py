"""DeepSeekAgent — a simple stateful conversational agent.

This is the original agent class from the early prototype. It wraps a
DeepSeekClient with an append-only message history and a single send()
method. No compression, no context injection, no sliding window.

For the production pipeline, prefer AgentSession (core/session.py), which
adds sliding-window compression and context injection for long-running
broadcasts. DeepSeekAgent is still useful for short-lived, low-turn
interactions where compression overhead isn't justified — things like
the Architect's single-shot config extraction or the Router's one-call
classification.

USAGE:
    from vox_romana.core.agent import DeepSeekAgent

    agent = DeepSeekAgent(system_prompt="You are a witty economist.")
    r1 = agent.send("What's the strongest argument for UBI?")
    r2 = agent.send("Counter that argument.")  # history preserved
    print(r2.response)
"""

from .client import DeepSeekClient
from .models import AgentResponse


class DeepSeekAgent:
    """A simple stateful conversational agent backed by DeepSeek.

    Maintains a full message history (system + all user/assistant pairs).
    Every call to send() appends the user message, calls the LLM with the
    full history, and appends the response. History grows without bound —
    for long sessions, use AgentSession instead.

    Attributes:
        _client:   The DeepSeek API client.
        _messages: The full conversation history (system + turns).
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Initialize the agent with a system prompt.

        Args:
            system_prompt: The system message that sets the agent's personality
                           and instructions. Sent as the first message in every
                           LLM call.
            api_key:       DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
            model:         Model name. Defaults to deepseek-chat.
        """
        self._client = DeepSeekClient(api_key=api_key, model=model)
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

    def send(self, prompt: str, tag: str | None = None) -> AgentResponse:
        """Send a prompt, update conversation history, and return an AgentResponse.

        The full history (system + all prior turns + this new message) is sent
        to the LLM. The response is appended to history so subsequent calls
        see the full conversation.

        Args:
            prompt: The user message to send.
            tag:    Optional label attached to the AgentResponse for downstream
                    routing (e.g., "architect", "profile_builder").

        Returns:
            AgentResponse containing the prompt, response text, and tag.
        """
        self._messages.append({"role": "user", "content": prompt})
        response_text = self._client.chat(self._messages)
        self._messages.append({"role": "assistant", "content": response_text})
        return AgentResponse(prompt=prompt, response=response_text, tag=tag)

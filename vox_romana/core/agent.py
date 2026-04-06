from .client import DeepSeekClient
from .models import AgentResponse


class DeepSeekAgent:
    """A conversational agent backed by DeepSeek. Maintains message history."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        api_key: str | None = None,
        model: str | None = None,
    ):
        self._client = DeepSeekClient(api_key=api_key, model=model)
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

    def send(self, prompt: str, tag: str | None = None) -> AgentResponse:
        """Send a prompt, update conversation history, and return an AgentResponse."""
        self._messages.append({"role": "user", "content": prompt})
        response_text = self._client.chat(self._messages)
        self._messages.append({"role": "assistant", "content": response_text})
        return AgentResponse(prompt=prompt, response=response_text, tag=tag)

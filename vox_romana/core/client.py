"""DeepSeekClient — the low-level API wrapper for DeepSeek.

This is the bottom of the stack. Every LLM call in vox_romana flows through
this client, whether it's a one-shot Router classification or a 50-turn
debate with compression. The client wraps the OpenAI SDK pointed at
DeepSeek's API endpoint.

The client is deliberately thin — it handles authentication and the
HTTP call, nothing else. Session management, compression, history
tracking, and retry logic all live in higher layers (AgentSession,
DeepSeekAgent, or directly in the conductors).

USAGE:
    from vox_romana.core.client import DeepSeekClient

    client = DeepSeekClient()  # reads DEEPSEEK_API_KEY from .env
    response = client.chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain UBI in one sentence."},
    ])
    print(response)  # "Universal basic income is..."

    # Pass kwargs through to the OpenAI API (temperature, max_tokens, etc.):
    response = client.chat(messages, temperature=0.9, max_tokens=500)

ENVIRONMENT:
    Requires DEEPSEEK_API_KEY in .env or as an environment variable.
"""

import os

from openai import OpenAI


class DeepSeekClient:
    """Thin wrapper around the DeepSeek chat API.

    Uses the OpenAI SDK under the hood, pointed at DeepSeek's endpoint.
    One client instance can be shared across the entire pipeline — it's
    stateless (no conversation history, no session tracking).

    Attributes:
        api_key: The DeepSeek API key.
        model:   The model name (default: deepseek-chat).
    """

    BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize the client.

        Args:
            api_key: DeepSeek API key. If not provided, reads from the
                     DEEPSEEK_API_KEY environment variable.
            model:   Model name to use for completions. Defaults to
                     deepseek-chat.

        Raises:
            ValueError: If no API key is found.
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key or set DEEPSEEK_API_KEY env var."
            )
        self.model = model or self.DEFAULT_MODEL
        self._openai = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> str:
        """Send a chat completion request and return the response text.

        This is the only method most callers need. It sends the messages
        to DeepSeek, waits for a completion, and returns the assistant's
        response as a plain string.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Roles: 'system', 'user', 'assistant'.
            **kwargs: Forwarded to the OpenAI API (temperature, max_tokens,
                      top_p, stop, etc.).

        Returns:
            The assistant's response text.
        """
        response = self._openai.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

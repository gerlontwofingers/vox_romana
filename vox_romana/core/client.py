import os

from openai import OpenAI


class DeepSeekClient:
    """Reusable base client for DeepSeek API calls."""

    BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(self, api_key: str | None = None, model: str | None = None):
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
        """Send a chat completion request and return the response text."""
        response = self._openai.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

"""OllamaClient — DeepSeekClient subclass pointed at a local Ollama instance.

Ollama exposes an OpenAI-compatible REST API at http://localhost:11434/v1,
so every existing vox_romana call-site that accepts a DeepSeekClient works
unchanged when given an OllamaClient instead.

API key is hardcoded to "ollama" (required by the OpenAI SDK constructor but
not validated by Ollama's server).

USAGE::

    from vox_romana.local.ollama import OllamaClient

    # Default: reads model / base_url from vox_romana.yaml [university] section.
    client = OllamaClient.from_config()

    # Explicit:
    client = OllamaClient(model="deepseek-r1:14b")
    client = OllamaClient(model="deepseek-r1:7b", base_url="http://192.168.1.5:11434/v1")

MODEL SIZE GUIDE (RAM required):
    deepseek-r1:1.5b  —  CPU-only fallback, ~2 GB
    deepseek-r1:7b    —  Apple M-series 16 GB  /  NVIDIA 8 GB VRAM
    deepseek-r1:14b   —  Apple M-series 32 GB  /  NVIDIA 12 GB VRAM
    deepseek-r1:32b   —  Apple M-series 64 GB  /  NVIDIA 24 GB VRAM

SETUP::

    brew install ollama
    ollama pull deepseek-r1:7b
    ollama serve          # starts the local API server
"""

from __future__ import annotations

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from typing import Any

from openai import OpenAI

from vox_romana.core.client import DeepSeekClient

logger = logging.getLogger(__name__)


class OllamaClient(DeepSeekClient):
    """DeepSeekClient subclass that targets a local Ollama server.

    The Ollama REST API is OpenAI-compatible, so every vox_romana call-site
    that accepts a ``DeepSeekClient`` works unchanged with this subclass.

    The class also adds a few Ollama-specific niceties used by the lens
    rack (Chunk 2):

    * :meth:`chat` translates DeepSeek's
      ``response_format={"type":"json_object"}`` kwarg into Ollama's
      ``extra_body={"format":"json"}`` so JSON-mode prompts work
      transparently.
    * :meth:`is_alive` pings ``GET /api/tags`` for a fast health check.
    * :meth:`available_models` returns the list of locally installed
      tags so the lens rack can report what's pullable.

    Attributes:
        model:    Ollama model name (e.g. "deepseek-r1:7b").
        base_url: Ollama API base URL (default: http://localhost:11434/v1).
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    DEFAULT_MODEL = "deepseek-r1:7b"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialise OllamaClient.

        Args:
            model:    Ollama model tag. Defaults to ``deepseek-r1:7b``.
            base_url: Ollama API base URL. Defaults to
                      ``http://localhost:11434/v1``.
        """
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.api_key = "ollama"  # required by OpenAI SDK; not validated by Ollama
        self._openai = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    # ── DeepSeekClient.chat override ────────────────────────────────

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send a chat completion via Ollama's OpenAI-compatible endpoint.

        Translates ``response_format={"type": "json_object"}`` (the cloud
        DeepSeek convention used throughout vox_romana) into Ollama's
        ``extra_body={"format": "json"}`` so JSON-mode prompts work
        without any caller-side changes.
        """
        rf = kwargs.get("response_format")
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            kwargs.pop("response_format", None)
            extra = dict(kwargs.get("extra_body") or {})
            extra.setdefault("format", "json")
            kwargs["extra_body"] = extra

        response = self._openai.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    # ── Health / discovery ───────────────────────────────────────────

    @property
    def _tags_url(self) -> str:
        # /v1 is the OpenAI-compat shim; /api/tags lives at the bare host.
        host = self.base_url.rstrip("/")
        if host.endswith("/v1"):
            host = host[: -len("/v1")]
        return f"{host}/api/tags"

    def is_alive(self, timeout: float = 1.0) -> bool:
        """Return True if the Ollama server responds on /api/tags."""
        try:
            with urllib.request.urlopen(self._tags_url, timeout=timeout) as r:
                return 200 <= r.status < 300
        except (urllib.error.URLError, OSError):
            return False

    def available_models(self, timeout: float = 2.0) -> list[str]:
        """Return the model tags currently installed in Ollama.

        Returns an empty list if the server is unreachable.  Errors are
        logged at DEBUG so callers (e.g. the lens probe) can degrade
        gracefully when Ollama isn't running.
        """
        import json
        try:
            with urllib.request.urlopen(self._tags_url, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.debug("Ollama /api/tags failed: %s", exc)
            return []
        return [m.get("name", "") for m in payload.get("models") or [] if m.get("name")]

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "OllamaClient":
        """Build an OllamaClient from a config dict or vox_romana.yaml.

        Resolution order:
            1. ``cfg`` kwarg (a dict, e.g. the ``university`` YAML block).
            2. ``vox_romana.yaml`` ``[university]`` section.
            3. Defaults (``deepseek-r1:7b`` @ ``http://localhost:11434/v1``).

        Args:
            cfg: Optional config dict with keys ``model`` and/or ``base_url``.

        Returns:
            A configured OllamaClient.
        """
        if cfg is None:
            # Lazy import to avoid circular dependency at module load time.
            try:
                from vox_romana.core.paths import _load_config  # type: ignore[attr-defined]
                cfg = _load_config().get("university", {})
            except Exception:
                cfg = {}

        return cls(
            model=cfg.get("model") or cls.DEFAULT_MODEL,
            base_url=cfg.get("base_url") or cls.DEFAULT_BASE_URL,
        )

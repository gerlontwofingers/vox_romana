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

from openai import OpenAI

from vox_romana.core.client import DeepSeekClient


class OllamaClient(DeepSeekClient):
    """DeepSeekClient subclass that targets a local Ollama server.

    The Ollama REST API is OpenAI-compatible, so every vox_romana call-site
    that accepts a ``DeepSeekClient`` works unchanged with this subclass.

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
        self.api_key = "ollama"  # required by OpenAI SDK; not validated by Ollama
        self._openai = OpenAI(
            api_key=self.api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
        )

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

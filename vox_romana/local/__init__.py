"""vox_romana.local — local LLM backend via Ollama.

Provides OllamaClient, a subclass of DeepSeekClient that re-targets the
OpenAI-compatible chat API at a locally running Ollama instance instead of
the DeepSeek cloud endpoint.

Usage::

    from vox_romana.local import OllamaClient
    client = OllamaClient()          # uses vox_romana.yaml [university] section
    client = OllamaClient(model="deepseek-r1:14b")
"""

from vox_romana.local.ollama import OllamaClient

__all__ = ["OllamaClient"]

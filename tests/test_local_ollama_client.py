"""Tests for vox_romana.local.ollama.OllamaClient.

Two tiers of tests live here:

1.  *Unit tests* (always-on) — exercise pure plumbing that does not need
    a running Ollama server: JSON-mode kwarg translation, ``_tags_url``
    derivation, ``available_models``/``is_alive`` behaviour when the
    server is unreachable.

2.  *Live tests* (gated by ``OLLAMA_AVAILABLE=1``) — exercise the real
    server.  Skipped in CI by default; opt in locally with::

        OLLAMA_AVAILABLE=1 pytest tests/test_local_ollama_client.py
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from vox_romana.local.ollama import OllamaClient


OLLAMA_AVAILABLE = os.environ.get("OLLAMA_AVAILABLE") == "1"


# ── Unit tests ───────────────────────────────────────────────────────


def test_default_base_url_and_model() -> None:
    client = OllamaClient()
    assert client.model == "deepseek-r1:7b"
    assert client.base_url == "http://localhost:11434/v1"


def test_tags_url_strips_v1_suffix() -> None:
    client = OllamaClient(base_url="http://localhost:11434/v1")
    assert client._tags_url == "http://localhost:11434/api/tags"


def test_tags_url_handles_bare_host() -> None:
    client = OllamaClient(base_url="http://example.com:9999")
    assert client._tags_url == "http://example.com:9999/api/tags"


def test_chat_translates_json_response_format() -> None:
    """response_format={'type':'json_object'} → extra_body={'format':'json'}."""
    client = OllamaClient()
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="{}"))]
    with patch.object(client._openai.chat.completions, "create", return_value=fake_resp) as create:
        client.chat(
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )
    kwargs = create.call_args.kwargs
    assert "response_format" not in kwargs
    assert kwargs["extra_body"] == {"format": "json"}


def test_chat_passes_through_unrelated_kwargs() -> None:
    client = OllamaClient()
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
    with patch.object(client._openai.chat.completions, "create", return_value=fake_resp) as create:
        client.chat([{"role": "user", "content": "x"}], temperature=0.3)
    assert create.call_args.kwargs["temperature"] == 0.3
    assert "extra_body" not in create.call_args.kwargs


def test_is_alive_false_when_server_unreachable() -> None:
    # Use a port nothing's listening on.
    client = OllamaClient(base_url="http://127.0.0.1:1/v1")
    assert client.is_alive(timeout=0.5) is False


def test_available_models_empty_when_unreachable() -> None:
    client = OllamaClient(base_url="http://127.0.0.1:1/v1")
    assert client.available_models(timeout=0.5) == []


# ── Live tests (need a running server) ───────────────────────────────


@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="set OLLAMA_AVAILABLE=1 to run")
def test_live_is_alive() -> None:
    assert OllamaClient().is_alive() is True


@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="set OLLAMA_AVAILABLE=1 to run")
def test_live_available_models_returns_list() -> None:
    models = OllamaClient().available_models()
    assert isinstance(models, list)
